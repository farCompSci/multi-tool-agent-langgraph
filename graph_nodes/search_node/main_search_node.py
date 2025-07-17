import os
import sys
from dotenv import load_dotenv
from loguru import logger
from typing import Optional, Dict, Any
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import SystemMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from helpers.model_config import fetch_ollama_model
from helpers.searching.search_operations import search_tool
from graph_nodes.search_node.search_states import SearchState


def search_node(state) -> Dict[str, Any]:
    try:
        llm = fetch_ollama_model("llama3.2")
        llm_with_search_tools = llm.bind_tools([search_tool])

        # Use the FULL message history from state, not just the last message
        messages = state['messages']

        # Add system message if it's not already there
        if not messages or messages[0] != SystemMessage:
            system_message = {
                'role': 'system',
                'content': (
                    "You are a helpful assistant. "
                    "Use the searching tool to find information when needed. "
                    "Once you have the answer from the tool, provide it to the user without calling the tool again."
                )
            }
            messages = [system_message] + messages

        result = llm_with_search_tools.invoke(messages)
        # logger.info(f"LLM output: {result}") # for debugging

        return {"messages": [result]}

    except Exception as e:
        logger.error(f"Error in searching node: {e}")
        return {"messages": [{"role": "assistant", "content": f"Search error: {e}"}]}


search_graph = StateGraph(SearchState)
search_graph.add_node("tools", ToolNode([search_tool]))
search_graph.add_node("search_llm", search_node)
search_graph.add_edge(START,"search_llm")
search_graph.add_conditional_edges("search_llm", tools_condition)
search_graph.add_edge("tools", "search_llm")
search_graph.add_edge("search_llm", END)
search_graph = search_graph.compile()

if __name__ == "__main__":
    load_dotenv()
    state = {
        'messages': [{
            "role": 'user',
            "content": 'What is the weather like in portland,oregon today?'
        }]
    }
    response = search_graph.invoke(state)
    for i in response['messages']:
        i.pretty_print()

    search_graph.get_graph().draw_mermaid_png(output_file_path="searching.png")
