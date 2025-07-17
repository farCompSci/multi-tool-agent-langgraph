import os
from dotenv import load_dotenv
import sys
from loguru import logger
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from helpers.calculations.advanced_calculation_operations import ask_wolfram
from helpers.model_config import fetch_ollama_model
from math_states import State

# load_dotenv()

def advanced_math_llm(state: State):
    """
    Processes advanced math queries using LLM with Wolfram Alpha integration
    :param state: State - The current state containing messages
    :returns: dict - Updated state with LLM response
    """

    llm = fetch_ollama_model(model_name="llama3.2")
    try:
        llm_with_tools = llm.bind_tools([ask_wolfram])
        logger.info('Advanced math tools bound to llama3.2 model')
        response = llm_with_tools.invoke(state["messages"])
        return {'messages': [response]}

    except Exception as e:
        logger.error(f'Failed to bind advanced math tools to llama3.2 model. Details:\n{e}')
        return {"messages": [{"role": "assistant", "content": f"Error: {e}"}]}


# Build the advanced math subgraph
advanced_math_subgraph = StateGraph(State)
advanced_math_subgraph.add_node("llm", advanced_math_llm)  # Remove the parentheses here
advanced_math_subgraph.add_node("tools", ToolNode([ask_wolfram]))

advanced_math_subgraph.add_edge(START, "llm")
advanced_math_subgraph.add_conditional_edges("llm", tools_condition)
advanced_math_subgraph.add_edge("tools", "llm")

advanced_math_subgraph = advanced_math_subgraph.compile()
# advanced_math_subgraph.get_graph().draw_mermaid_png(output_file_path="output_advanced_math.png")

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
        res.pretty_print()