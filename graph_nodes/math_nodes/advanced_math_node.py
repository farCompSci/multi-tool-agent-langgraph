import os
from dotenv import load_dotenv
import sys
from loguru import logger
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from helpers.calculating.advanced_calculation_operations import ask_wolfram
from helpers.model_config import fetch_ollama_model
from graph_nodes.math_nodes.math_states import MathState


def advanced_math_llm(state: MathState):
    """
    Processes advanced math queries using LLM with Wolfram Alpha integration
    """
    llm = fetch_ollama_model(model_name="llama3.2")
    try:
        llm_with_tools = llm.bind_tools([ask_wolfram])
        logger.info('Advanced math tools bound to llama3.2 model')

        messages = state["messages"]

        # Add system message for better tool usage if not present
        # Ensure this system message is always at the beginning for context
        system_message_content = """You are a mathematical assistant with access to Wolfram Alpha. 
        For any mathematical computation, derivative, integral, or complex calculation, 
        use the ask_wolfram tool by calling it with the mathematical expression as the query parameter.

        Example: To find the derivative of x^2 + 3x + 1, call ask_wolfram with query="derivative of x^2 + 3x + 1"

        After a tool has been executed and its result is provided, generate a clear, concise, human-readable answer based on the tool's output.
        """

        # Ensure system message is always the first message
        if not messages or (isinstance(messages[0], dict) and messages[0].get("role") != "system") or \
                (hasattr(messages[0], "type") and messages[0].type != "system"):
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=system_message_content)] + messages
        else:  # If system message exists, update its content
            if isinstance(messages[0], dict):
                messages[0]["content"] = system_message_content
            else:
                messages[0].content = system_message_content

        logger.info(f"Sending to LLM: {len(messages)} messages")
        response = llm_with_tools.invoke(messages)
        logger.info(f"LLM response type: {type(response)}")

        return {'messages': [response]}

    except Exception as e:
        logger.error(f'Advanced math error: {e}')
        from langchain_core.messages import AIMessage
        error_msg = AIMessage(content=f"I apologize, but I encountered an error: {e}")
        return {"messages": [error_msg]}

# Build the advanced math subgraph
advanced_math_subgraph = StateGraph(MathState)
advanced_math_subgraph.add_node("llm", advanced_math_llm)
advanced_math_subgraph.add_node("tools", ToolNode([ask_wolfram]))

advanced_math_subgraph.add_edge(START, "llm")
advanced_math_subgraph.add_conditional_edges("llm", tools_condition)
advanced_math_subgraph.add_edge("tools", "llm")

advanced_math_subgraph = advanced_math_subgraph.compile()

if __name__ == "__main__":
    load_dotenv()
    state_example = {
        'messages': [
            {
                'role': 'user',
                'content': 'Find the integral of e^(7*sin(5)).'
            }
        ]
    }
    result = advanced_math_subgraph.invoke(state_example)
    for res in result['messages']:
        if hasattr(res, 'pretty_print'):
            res.pretty_print()
        else:
            print(res)