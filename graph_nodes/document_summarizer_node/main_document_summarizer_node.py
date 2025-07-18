import os
import sys
from loguru import logger
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from helpers.model_config import fetch_ollama_model
from helpers.summarizing.summarization_operations import read_file_content
from graph_nodes.document_summarizer_node.summarization_states import SummarizationState


def summarization_llm_node(state) -> Dict[str, Any]:
    """
    LLM node that can read files and summarize them.
    """
    try:
        llm = fetch_ollama_model("llama3.2")
        llm_with_tools = llm.bind_tools([read_file_content])

        messages = state['messages']

        if not messages or messages[0] != SystemMessage:
            system_message = {
                'role': 'system',
                'content': (
                    "You are a document summarizer. "
                    "When given a file path, use the read_file_content to read the file. "
                    "After reading the file, provide a brief and very concise summary of its contents."
                )
            }
            messages = [system_message] + messages

        logger.info("Invoking LLM for file reading and summarizing")
        result = llm_with_tools.invoke(messages)

        return {"messages": [result]}

    except Exception as e:
        logger.error(f"Error in summarizing LLM node: {e}")
        return {"messages": [{"role": "assistant", "content": f"Error: {e}"}]}


# Build the graph
summarization_graph = StateGraph(SummarizationState)
summarization_graph.add_node("llm", summarization_llm_node)
summarization_graph.add_node("tools", ToolNode([read_file_content]))

summarization_graph.add_edge(START, "llm")
summarization_graph.add_conditional_edges("llm", tools_condition)
summarization_graph.add_edge("tools", "llm")
summarization_graph.add_edge("llm", END)

summarization_graph = summarization_graph.compile()

if __name__ == "__main__":
    # Create a test file first
    test_file_path = "test_document.txt"
    with open(test_file_path, 'w') as f:
        f.write("""
        This is a sample document for testing the summarizing functionality.
        It contains multiple paragraphs and discusses various topics.

        The first topic is about artificial intelligence and its applications in modern technology.
        AI has revolutionized many industries including healthcare, finance, and transportation.

        The second topic covers machine learning algorithms and their implementation.
        These algorithms can learn from data and make predictions or decisions.

        Finally, the document discusses the future of AI and its potential impact on society.
        While there are many benefits, there are also challenges that need to be addressed.
        """)

    # User provides a file path to summarize
    state = {
        'messages': [{
            "role": 'user',
            "content": f'Please read and summarize the file: {test_file_path}'
        }]
    }

    try:
        response = summarization_graph.invoke(state)
        print("=== SUMMARIZATION RESULT ===")
        for i in response['messages']:
            i.pretty_print()
        print("=" * 30)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

    summarization_graph.get_graph().draw_mermaid_png(output_file_path="summarizing.png")