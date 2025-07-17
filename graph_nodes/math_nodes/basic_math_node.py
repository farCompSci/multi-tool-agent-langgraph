from langgraph.graph import START, StateGraph
from loguru import logger
import sys
import os
from langgraph.prebuilt import ToolNode, tools_condition

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from helpers.model_config import fetch_ollama_model
from helpers.calculations.basic_calculator_operations import add, subtract, multiply, divide
from math_states import State

def basic_math_llm(state: State):
    llm = fetch_ollama_model(model_name="llama3.2")
    try:
        llm_with_tools = llm.bind_tools([add, subtract, divide, multiply])
        logger.info('Basic math tools bound to llama3.2 model')

        # Pass the full message history, not just the last message
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    except Exception as e:
        logger.error(f'There was an error binding the llm to the basic math tools. Details: {e}')
        return {"messages": [{"role": "assistant", "content": f"Error: {e}"}]}


# Build the basic math subgraph
basic_math_subgraph = StateGraph(State)
basic_math_subgraph.add_node("llm", basic_math_llm)
basic_math_subgraph.add_node("tools", ToolNode([add, subtract, divide, multiply]))

basic_math_subgraph.add_edge(START, "llm")
basic_math_subgraph.add_conditional_edges("llm", tools_condition)
basic_math_subgraph.add_edge("tools", "llm")

basic_math_subgraph = basic_math_subgraph.compile()
# basic_math_subgraph.get_graph().draw_mermaid_png(output_file_path="output_basic_math.png")

if __name__ == "__main__":
    state_example = {
        'messages': [
            {
                'role': 'user',
                'content': 'Multiply 2 by 10. Divide the result by 5. '
            }
        ]
    }

    result = basic_math_subgraph.invoke(state_example)
    print(result)
    for message in result['messages']:
        message.pretty_print()