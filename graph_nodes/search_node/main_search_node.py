import os
import sys
from dotenv import load_dotenv
from loguru import logger
from typing import Optional, Dict, Any
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage  # Import all message types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from helpers.model_config import fetch_ollama_model
from helpers.searching.search_operations import search_tool
from graph_nodes.search_node.search_states import SearchState


def search_llm_node(state: SearchState) -> Dict[str, Any]:
    try:
        llm = fetch_ollama_model("llama3.2")
        bound_tools = [search_tool]
        llm_with_tools = llm.bind_tools(bound_tools)

        messages = state['messages']

        system_prompt_content = """You are a helpful assistant that can search the web for current information, news, weather, movie plots, and release dates.

        **Instructions for Tool Usage:**
        1.  **For general web searches (e.g., weather, news, facts, popular games/songs):** Use the `search_tool`.
            *   Example: User: "What's the weather in London?" -> Call `search_tool(query="weather in London")`
            *   Example: User: "Who won the World Series last year?" -> Call `search_tool(query="who won World Series last year")`
            *   Example: User: "What is the most popular rock song in the USA this month?" -> Call `search_tool(query="most popular rock song USA this month")`

        # If you have search_movie and search_release_date tools, add their instructions here:
        # 2.  **For movie plots:** Use the `search_movie` tool.
        #     *   Example: User: "Tell me the plot of Inception." -> Call `search_movie(query="plot of Inception")`
        # 3.  **For movie release dates:** Use the `search_release_date` tool.
        #     *   Example: User: "When was Avatar released?" -> Call `search_release_date(query="Avatar release date")`

        **Important:**
        *   Always extract the most relevant keywords and phrases from the user's query for the `query` parameter of the tool.
        *   After a tool has been executed and its result is provided (as a `ToolMessage`), generate a clear, concise, human-readable answer based on the tool's output. Do NOT just output the raw tool result.
        *   If the user asks for real-time information, always use a tool. Do not state that you cannot access real-time information.
        *   If the user asks a question that requires a search, and you have already performed a search for that specific query in the current turn (i.e., you see a `ToolMessage` with the result), then synthesize the answer from the `ToolMessage` content.
        """

        llm_input_messages = [SystemMessage(content=system_prompt_content)] + messages

        result = llm_with_tools.invoke(llm_input_messages)
        logger.info(f"LLM output in search_llm_node: {result}")  # for debugging

        return {"messages": [result]}

    except Exception as e:
        logger.error(f"Error in search_llm_node: {e}")
        return {"messages": [AIMessage(content=f"Search error: {e}. Please try again.")]}


search_graph = StateGraph(SearchState)
search_graph.add_node("tools", ToolNode([search_tool]))  # Add all tools here
search_graph.add_node("search_llm", search_llm_node)  # Renamed for clarity

search_graph.add_edge(START, "search_llm")
search_graph.add_conditional_edges(
    "search_llm",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END
    }
)
search_graph.add_edge("tools",
                      "search_llm")  # After tool execution, return to LLM to process output and generate final answer

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
    search_graph.get_graph().draw_mermaid_png(output_file_path='search.png')
    for i in response['messages']:
        i.pretty_print()

    # search_graph.get_graph().draw_mermaid_png(output_file_path="searching.png")
