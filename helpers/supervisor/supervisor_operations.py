from typing import Dict, Any
import sys
import os
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from helpers.supervisor.supervisor_classes import SupervisorState
from helpers.supervisor.supervisor_schemas import TaskClassifier, TaskQueue, MultiStepDecision, DetailedTaskDecomposition
from helpers.model_config import fetch_ollama_model
from helpers.supervisor.long_term_memory import ltm
from helpers.model_config import fetch_openai_model


def store_memory_node(state: SupervisorState) -> Dict[str, Any]:
    """Store important information in long-term memory."""
    try:
        messages = state.get("messages", [])
        ai_response_stored = False

        for msg in reversed(messages):
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", getattr(msg, "type", ""))
                content = getattr(msg, "content", str(msg))

            if role in ["assistant", "ai"] and content and "error" not in content.lower() and not ai_response_stored:
                # Filter out explicit denials of memory AND denials of real-time access
                if "i don't have any personal memories" in content.lower() or \
                        "i don't have real-time access" in content.lower() or \
                        "i cannot provide real-time" in content.lower() or \
                        "i'm a large language model" in content.lower():  # Added this one too
                    print(f"Skipping storing of non-useful AI memory (denial/limitation): {content[:50]}...")
                    continue


                ltm.store_memory(
                    content=content,
                    metadata={
                        "type": "assistant_response",
                        "task_type": state.get("task_classification", "unknown")
                    }
                )
                print(f"Storing AI memory: {content[:50]}...")
                ai_response_stored = True

            elif role in ["human", "user"] and content:
                if "my name is" in content.lower() or \
                        "i am a" in content.lower() or \
                        "i'm a" in content.lower() or \
                        "i work on" in content.lower():
                    ltm.store_memory(
                        content=content,
                        metadata={
                            "type": "user_introduction",
                            "task_type": state.get("task_classification", "general_chat")
                        }
                    )
                    print(f"Storing user introduction memory: {content[:50]}...")
                    break

            if ai_response_stored and (
                    role in ["human", "user"] and ("my name is" in content.lower() or "i am a" in content.lower())):
                break

        return {}

    except Exception as e:
        print(f"Error in store_memory_node: {e}")
        return {}


def retrieve_long_term_memory_node(state: SupervisorState) -> Dict[str, Any]:
    """Retrieve relevant long-term memories based on the current user input."""
    try:
        messages = state.get("messages", [])
        if not messages:
            return {"retrieved_memories": []}

        user_message = None
        for msg in reversed(messages):
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", getattr(msg, "type", ""))
                content = getattr(msg, "content", str(msg))

            if role in ["human", "user"]:
                user_message = content
                break

        if not user_message:
            return {"retrieved_memories": []}

        memories = ltm.retrieve_memories(user_message, k=3)

        filtered_memories = []
        for mem in memories:
            if "i don't have any personal memories" not in mem.get("content", "").lower() \
                or "i don't have any information about" not in mem.get("content", ""):
                filtered_memories.append(mem)
            else:
                print(f"Filtering out negative memory: {mem.get('content', '')[:50]}...")

        # print(f"Retrieved memories: {filtered_memories}")
        return {"retrieved_memories": filtered_memories}

    except Exception as e:
        print(f"Error in retrieve_long_term_memory_node: {e}")
        return {"retrieved_memories": []}


def classify_task_node(state: SupervisorState) -> Dict[str, Any]:
    last_msg = state["messages"][-1].content or ""
    msg_lc = last_msg.lower()

    # This override is kept because it's a very specific, single-tool task
    if "summarize" in msg_lc and ".pdf" in msg_lc:
        logger.info("Override: detected file summarization → summarize")
        single_task = {
            "description": last_msg,
            "type": "summarize",
            "status": "pending"
        }
        return {
            "task_classification": "summarize",
            "current_task": {"type": "summarize"},
            "task_queue": TaskQueue(tasks=[single_task])
        }

    system_prompt_content = """Classify the user input into one of these categories:
    - 'math': Mathematical calculations, equations, derivatives, integrals, statistics, algebra, calculus, trigonometry, or any symbolic math.
    - 'search': Web searches, current information, news, weather, facts, or anything requiring up-to-date external information.
    - 'code': Programming, code execution, debugging, algorithm implementation.
    - 'summarize': STRICTLY ONLY use this if the user explicitly asks to summarize a DOCUMENT or FILE, or provides a file path/name. Do NOT use for general questions about past results or information.
    - 'retrieve_info': Use this if the user is asking to recall or retrieve information that was previously discussed or stored in memory, or asking about a past result.
    - 'general_chat': General conversation, greetings, non-tool tasks, or general knowledge questions that don't fit other categories.

    For math: Include ANY mathematical operation beyond simple arithmetic (2+2).
    Examples: derivatives, integrals, solving equations, graphing, limits, etc.

    Choose the most appropriate category based on the user's request.
    If the user is asking about a previous result or information, classify as 'retrieve_info'.
    Absolutely do NOT classify as 'summarize' unless a file or document is explicitly mentioned or clearly implied.
    """

    llm = fetch_openai_model("gpt-4o-mini")
    classifier = llm.with_structured_output(TaskClassifier)
    result = classifier.invoke([
        {"role": "system", "content": system_prompt_content},
        {"role": "user",   "content": last_msg}
    ])
    logger.info(f"LLM classified → {result.task_type} (reason: {result.reasoning})")

    # ALWAYS route to decompose_task for all classifications except the specific .pdf summarize override.
    # decompose_task_node will then decide if it's truly multi-step or a single task.
    return {
        "task_classification": result.task_type,
        "current_task": {"type": result.task_type} # current_task is set, but queue is not yet
    }


def decompose_task_node(state: SupervisorState) -> Dict[str, Any]:
    """
    Decides if a task is multi-step and, if so, decomposes it into subtasks.
    Otherwise, creates a single task.
    """
    try:
        last_message = state["messages"][-1].content
        initial_task_type = state.get("task_classification", "general_chat")
        logger.info(f"Decompose Task Node: Processing message: '{last_message}' with initial type: '{initial_task_type}'")

        llm = fetch_openai_model("gpt-4o-mini")

        # Step 1: Decide if it's a multi-step task
        multi_step_decider = llm.with_structured_output(MultiStepDecision)
        decision_prompt = f"""Analyze the following user request to determine if it requires multiple distinct steps or tool invocations to be fully completed.

Consider it multi-step if:
- It contains explicit sequential indicators like 'then', 'and then', 'after that', 'first', 'next', 'finally'.
- It implies a sequence of operations, especially for math problems (e.g., 'add X, then divide by Y').
- It combines requests for different types of information or actions (e.g., 'summarize this document and then find related news').
- It's a complex programming task that might benefit from breaking down (e.g., 'write a function to do X and then test it with Y').
- It asks for multiple distinct pieces of information or actions that cannot be fulfilled by a single tool call or response.

Consider it single-step if:
- It's a direct question answerable by one tool or a single general chat response.
- It's a simple math problem that can be solved in one go.
- It's a straightforward search query.
- It's a single coding task.

User Request: "{last_message}"

Provide your decision and reasoning. If you identify initial subtasks, list them.
"""
        logger.info(f"Decomposition Decision Prompt: {decision_prompt}") # Debugging
        multi_step_result = multi_step_decider.invoke([
            {"role": "system", "content": decision_prompt},
            {"role": "user", "content": last_message}
        ])
        logger.info(f"Multi-step decision RAW: {multi_step_result}") # NEW: Log raw result
        logger.info(f"Multi-step decision: {multi_step_result.is_multi_step} (Reason: {multi_step_result.reasoning})")

        tasks_list = []

        if multi_step_result.is_multi_step:
            logger.info("Multi-step task detected. Proceeding to detailed decomposition.")
            # If multi-step, always go for detailed decomposition
            decomposer = llm.with_structured_output(DetailedTaskDecomposition)
            decomposition_prompt = f"""Break down the following user request into smaller, actionable subtasks. Each subtask should be specific and executable by one of the available tools (math, search, code, summarize, retrieve_info, general_chat).

Order the subtasks logically for sequential execution.
Ensure the output is a list of tasks, each with a 'description' and 'type'.
If a subtask requires information from a previous one, make that clear in its description.

Available tool types: 'math', 'search', 'code', 'summarize', 'retrieve_info', 'general_chat'.

Example for 'what is 2+2. add 3. divide by 7. multiply by 4':
[
    {{"description": "Calculate 2+2", "type": "math"}},
    {{"description": "Add 3 to the previous result", "type": "math"}},
    {{"description": "Divide the previous result by 7", "type": "math"}},
    {{"description": "Multiply the previous result by 4", "type": "math"}}
]

Example for 'Find the capital of France, then tell me its population':
[
    {{"description": "Find the capital of France", "type": "search"}},
    {{"description": "Find the population of the capital of France (from previous step)", "type": "search"}}
]

Example for 'Write a Python function to reverse a string and then provide an example of its usage':
[
    {{"description": "Write a Python function to reverse a string", "type": "code"}},
    {{"description": "Provide an example of how to use the string reversal function from the previous step", "type": "code"}}
]

Example for 'Show me how to reverse a linked list using python classes. Then, write a function that performs the fibonacci sequence. Finally, look up what the weather will be like in Manhattan tomorrow.':
[
    {{"description": "Show me how to reverse a linked list using python classes.", "type": "code"}},
    {{"description": "Write a function that performs the fibonacci sequence.", "type": "code"}},
    {{"description": "Look up what the weather will be like in Manhattan tomorrow.", "type": "search"}}
]

Decompose the following user request: "{last_message}"
"""
            logger.info(f"Detailed Decomposition Prompt: {decomposition_prompt}") # Debugging
            decomposition_result = decomposer.invoke([
                {"role": "system", "content": decomposition_prompt},
                {"role": "user", "content": last_message}
            ])
            logger.info(f"Detailed Decomposition RAW: {decomposition_result}") # NEW: Log raw result
            tasks_list = [
                {"description": t.description, "type": t.type, "status": "pending"}
                for t in decomposition_result.tasks
            ]
            logger.info(f"Decomposed tasks list generated: {tasks_list}") # NEW: Log final tasks_list
        else:
            logger.info("Single-step task detected. Creating single task.")
            # If not multi-step, create a single task based on the initial classification
            single_task = {
                "description": last_message,
                "type": initial_task_type, # Use the type from classify_task_node
                "status": "pending"
            }
            tasks_list.append(single_task)
            logger.info(f"Single task created: {single_task}")

        return {"task_queue": TaskQueue(tasks=tasks_list)}

    except Exception as e:
        logger.error(f"Error in decompose_task_node: {e}")
        # Fallback: wrap the last message as one general_chat task
        fallback = {
            "description": state["messages"][-1].content,
            "type": state.get("task_classification", "general_chat"),
            "status": "pending"
        }
        logger.error(f"Decomposition failed, falling back to single task: {fallback}") # NEW: Log fallback
        return {"task_queue": TaskQueue(tasks=[fallback])}


def process_queue_node(state: SupervisorState) -> Dict[str, Any]:
    """Pop the next task from the queue and set it as current task."""
    task_queue = state.get("task_queue")
    logger.info(f"Process Queue Node: Received task_queue: {task_queue.tasks if task_queue else 'None'}") # NEW LOGGING

    if not task_queue or task_queue.current_index >= len(task_queue.tasks):
        logger.info("Process Queue Node: No tasks in queue or all completed. Routing to finalize.") # NEW LOGGING
        return {"current_task": None, "queue_status": "No tasks in queue or all completed"}

    current_task = task_queue.tasks[task_queue.current_index]
    logger.info(f"Process Queue Node: Setting current_task: {current_task}") # NEW LOGGING
    return {
        "current_task": current_task,
        "queue_status": f"Processing task {task_queue.current_index + 1} of {len(task_queue.tasks)}"
    }


def task_router(state: SupervisorState) -> str:
    """Route current task to appropriate subgraph."""
    current_task = state.get("current_task")
    if not current_task:
        return "finalize"

    task_type = current_task.get("type", "general_chat")
    routing_map = {
        "math": "math_subgraph",
        "search": "search_subgraph",
        "code": "coding_subgraph",
        "summarize": "summarize_subgraph",
        "general_chat": "general_chat",
        "retrieve_info": "general_chat" # Route retrieve_info to general_chat
    }

    return routing_map.get(task_type, "general_chat")


def general_chat_node(state: SupervisorState) -> Dict[str, Any]:
    """Handle general conversation and non-tool tasks."""
    try:
        llm = fetch_openai_model("gpt-4o-mini")
        conversation = state["messages"]

        memories = state.get("retrieved_memories", [])
        # CHANGE HERE: Convert 'memory' role to 'system'
        memory_messages = [{"role": "system", "content": f"PREVIOUS CONTEXT/MEMORY: {mem['content']}"} for mem in
                           memories]

        # Build full context: memories + conversation
        full_context = memory_messages + conversation

        response = llm.invoke([
                                  {
                                      "role": "system",
                                      "content": """You are a helpful assistant.

                You have access to a long-term memory. If there are messages prefixed with 'PREVIOUS CONTEXT/MEMORY:', these are important facts or past conversations that might be relevant to the user's current query. Integrate this information naturally into your response if it helps answer the user's question.

                Provide a direct, helpful response. If you cannot adequately answer the question,
                be honest about your limitations."""
                                  },
                              ] + full_context)

        return {"messages": [{"role": "assistant", "content": response.content}]}

    except Exception as e:
        return {"messages": [
            {"role": "assistant", "content": f"I apologize, but I encountered an error: {e}"}]}


def advance_queue_node(state: SupervisorState) -> Dict[str, Any]:
    """Mark current task as completed and advance queue."""
    task_queue = state.get("task_queue")
    current_task = state.get("current_task")

    if task_queue and current_task:
        # Find the current task in the queue and mark it as completed
        for task in task_queue.tasks:
            if task.get("description") == current_task.get("description") and \
               task.get("type") == current_task.get("type") and \
               task.get("status") == "pending": # Only mark pending tasks
                task["status"] = "completed"
                break
        task_queue.completed_tasks.append(current_task) # Add to completed list

        task_queue.current_index += 1

    return {
        "task_queue": task_queue,
        "current_task": None
    }


def should_continue_queue(state: SupervisorState) -> str:
    """Check if there are more tasks in the queue."""
    task_queue = state.get("task_queue")
    if task_queue and task_queue.current_index < len(task_queue.tasks):
        return "process_queue"
    else:
        return "finalize"


def finalize_response_node(state: SupervisorState) -> Dict[str, Any]:
    """Finalize the response by synthesizing results from completed tasks."""
    task_queue = state.get("task_queue")
    completed_tasks = task_queue.completed_tasks if task_queue else []

    if not completed_tasks:
        logger.info("Finalize Response Node: No completed tasks to summarize.")
        return {
            "messages": [{
                "role": "assistant",
                "content": "I have completed processing your request, but no specific tasks were executed or results generated."
            }]
        }

    # Prepare a summary of completed tasks and their results for the LLM
    summary_parts = []
    for i, task in enumerate(completed_tasks):
        description = task.get("description", "No description provided.")
        task_type = task.get("type", "unknown")
        result = task.get("result", "No result captured.") # This is where we expect the result to be!

        # Format the result nicely, especially for code execution
        if task_type == "code" and isinstance(result, str):
            summary_parts.append(f"Task {i+1} ({task_type}): {description}\nExecution Result:\n```python\n{result}\n```\n")
        elif task_type == "search" and isinstance(result, str):
            summary_parts.append(f"Task {i+1} ({task_type}): {description}\nSearch Result: {result}\n")
        elif task_type == "summarize" and isinstance(result, str):
            summary_parts.append(f"Task {i+1} ({task_type}): {description}\nSummary: {result}\n")
        else:
            summary_parts.append(f"Task {i+1} ({task_type}): {description}\nResult: {str(result)}\n")

    full_summary_for_llm = "\n".join(summary_parts)
    logger.info(f"Finalize Response Node: Raw summary for LLM:\n{full_summary_for_llm}")

    # Use LLM to synthesize a coherent final response
    llm = fetch_openai_model("gpt-4o-mini") # Use your chosen LLM

    synthesis_prompt = f"""You are an AI assistant that has just completed a series of tasks for the user.
Your goal is to provide a concise, coherent, and helpful summary of what was done and the key results.

Here is a breakdown of the completed tasks and their individual results:
---
{full_summary_for_llm}
---

Based on the above, synthesize a single, natural language response to the user.
- Start by acknowledging that you've completed their request.
- Clearly state the main outcomes for each part of their original request.
- Present code results clearly, perhaps in code blocks.
- Present search results concisely.
- Do not include internal task numbers or statuses.
- Focus on the *information* the user asked for.
- Be polite and helpful.
"""

    try:
        # Provide the original user query as context for the synthesis LLM
        original_user_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, dict) and msg.get("role") == "human":
                original_user_query = msg.get("content", "")
                break
            elif not isinstance(msg, dict) and getattr(msg, "type", None) == "human": # For LangChain message objects
                original_user_query = getattr(msg, "content", "")
                break

        synthesized_response = llm.invoke([
            {"role": "system", "content": synthesis_prompt},
            {"role": "user", "content": original_user_query}
        ])
        logger.info(f"Finalize Response Node: Synthesized response: {synthesized_response.content}")
        return {
            "messages": [{
                "role": "assistant",
                "content": synthesized_response.content
            }]
        }
    except Exception as e:
        logger.error(f"Error during response synthesis: {e}")
        return {
            "messages": [{
                "role": "assistant",
                "content": f"I have completed the tasks, but encountered an error while synthesizing the final response: {e}. Please check the individual task outputs if available."
            }]
        }