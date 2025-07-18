import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

from helpers.supervisor.supervisor_classes import SupervisorState
from helpers.supervisor.supervisor_operations import (
    classify_task_node,
    decompose_task_node, # RE-ADDED
    process_queue_node,
    task_router,
    general_chat_node,
    advance_queue_node,
    should_continue_queue,
    finalize_response_node,
    store_memory_node,
    retrieve_long_term_memory_node,
)
from helpers.supervisor.supervisor_subgraph_wrappers import (
    math_subgraph_wrapper,
    search_subgraph_wrapper,
    summarize_subgraph_wrapper,
    coder_subgraph_wrapper)


def build_supervisor_graph():
    """Build the main supervisor graph."""
    memory = MemorySaver()
    graph = StateGraph(SupervisorState)

    # Add nodes
    graph.add_node("retrieve_memory", retrieve_long_term_memory_node)
    graph.add_node("classify_task", classify_task_node)
    graph.add_node("decompose_task", decompose_task_node) # RE-ADDED
    graph.add_node("process_queue", process_queue_node)
    graph.add_node("general_chat", general_chat_node)
    graph.add_node("advance_queue", advance_queue_node)
    graph.add_node("store_memory", store_memory_node)
    graph.add_node("finalize", finalize_response_node)

    # Add subgraph wrapper nodes
    graph.add_node("math_subgraph", math_subgraph_wrapper)
    graph.add_node("search_subgraph", search_subgraph_wrapper)
    graph.add_node("summarize_subgraph", summarize_subgraph_wrapper)
    graph.add_node("coding_subgraph", coder_subgraph_wrapper)

    # Add edges - MEMORY FIRST
    graph.add_edge(START, "retrieve_memory")
    graph.add_edge("retrieve_memory", "classify_task")

    # Conditional routing from classify_task to decompose_task or process_queue
    graph.add_conditional_edges(
        "classify_task",
        lambda state: "process_queue" if state.get("task_classification") == "summarize" else "decompose_task",
        {
            "decompose_task": "decompose_task",
            "process_queue": "process_queue"
        }
    )

    # Edge from decompose_task to process_queue
    graph.add_edge("decompose_task", "process_queue")


    # Conditional routing from process_queue
    graph.add_conditional_edges(
        "process_queue",
        task_router,
        {
            "math_subgraph": "math_subgraph",
            "search_subgraph": "search_subgraph",
            "summarize_subgraph": "summarize_subgraph",
            "general_chat": "general_chat",
            "coding_subgraph": "coding_subgraph",
            "finalize": "finalize"
        }
    )

    graph.add_edge("general_chat", "store_memory")
    graph.add_edge("math_subgraph", "store_memory")
    graph.add_edge("search_subgraph", "store_memory")
    graph.add_edge("summarize_subgraph", "store_memory")
    graph.add_edge("coding_subgraph", "store_memory")

    graph.add_edge("store_memory", "advance_queue")

    graph.add_conditional_edges(
        "advance_queue",
        should_continue_queue,
        {
            "process_queue": "process_queue",
            "finalize": "finalize"
        }
    )

    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=memory)
if __name__ == "__main__":
    load_dotenv()

    # Build the supervisor graph
    supervisor_graph = build_supervisor_graph()
    supervisor_graph.get_graph().draw_mermaid_png(output_file_path='supervisor_graph_with_memory.png')

    # Test configuration
    config = {"configurable": {"thread_id": "test_memory_session"}}

    print("🧠 Testing LangGraph Supervisor with ChromaDB Long-Term Memory")
    print("=" * 60)


    def get_current_message_count(graph_instance, current_config):
        try:
            state_values = graph_instance.get_state(current_config).values
            return len(state_values.get("messages", []))
        except Exception:
            return 0


    def print_new_messages(result, previous_count=0):
        messages = result.get("messages", [])
        # Only print messages that were added in this turn
        new_messages = messages[previous_count:]

        # Filter out intermediate tool calls if they are not the final output
        # This is a heuristic, you might need to adjust based on your exact tool output
        final_output_messages = []
        for msg in reversed(new_messages):
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", getattr(msg, "type", "unknown"))
                content = getattr(msg, "content", str(msg))

            # Only include AI, tool, or system messages that are part of the final response
            if role in ["ai", "assistant", "tool", "system"]:
                final_output_messages.insert(0, (role, content))  # Insert at beginning to maintain order
            elif role == "human":  # Stop if we hit a human message, as that's the start of the turn
                break

        for role, content in final_output_messages:
            print(f"{role}: {content}")

    # Test 1: Initial math question
    print("\n🔢 TEST 1: Initial Question")

    # Get message count *before* the invoke for this turn
    message_count_before_invoke = get_current_message_count(supervisor_graph, config)

    test_input_1 = {
        "messages": [{
            "role": "human",
            "content": "What is the weather like in Portland, Oregon today?"
        }]
    }

    try:
        result_1 = supervisor_graph.invoke(test_input_1, config)
        print("=== RESULT 1 ===")
        print_new_messages(result_1, message_count_before_invoke)

    except Exception as e:
        print(f"❌ Error in Test 1: {e}")
        import traceback

        traceback.print_exc()

