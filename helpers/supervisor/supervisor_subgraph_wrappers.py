import sys
import os
from typing import Dict, Any
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers.supervisor.supervisor_classes import SupervisorState
from graph_nodes.math_nodes.main_math_node import main_graph as math_subgraph
from graph_nodes.coder_node.main_coder_node import coder_graph as coder_subgraph
from graph_nodes.search_node.main_search_node import search_graph as search_subgraph
from graph_nodes.document_summarizer_node.main_document_summarizer_node import summarization_graph as summarize_subgraph


def _extract_final_message_content(state_output: Dict[str, Any]) -> str:
    """Helper to extract the content of the last relevant message from a subgraph's output state."""
    if "messages" in state_output and state_output["messages"]:
        for msg in reversed(state_output["messages"]):
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
            else:  # Assuming it's a LangChain message object
                role = getattr(msg, "type", getattr(msg, "role", None))
                content = getattr(msg, "content", str(msg))

            if role in ["tool", "system"] and content:
                return content
            elif role == "ai" and content:
                return content
    return "No specific output captured from subgraph."


def math_subgraph_wrapper(state: SupervisorState) -> Dict[str, Any]:
    """
    Wrapper for the math subgraph.
    Captures the output of the math task and stores it in current_task['result'].
    """
    current_task = state["current_task"]
    logger.info(f"Math Subgraph Wrapper: Processing task: {current_task.get('description')}")

    try:
        math_output_state = math_subgraph.invoke(state)
        final_math_result = _extract_final_message_content(math_output_state)

        current_task["result"] = final_math_result
        logger.info(f"Math Subgraph Wrapper: Captured result for task: {current_task.get('description')}")

        return {"current_task": current_task}
    except Exception as e:
        logger.error(f"Error in math_subgraph_wrapper for task '{current_task.get('description')}': {e}")
        current_task["result"] = f"Error executing math task: {e}"
        current_task["status"] = "failed"
        return {"current_task": current_task}


def search_subgraph_wrapper(state: SupervisorState) -> Dict[str, Any]:
    """
    Wrapper for the search subgraph.
    Captures the output of the search task and stores it in current_task['result'].
    """
    current_task = state["current_task"]
    logger.info(f"Search Subgraph Wrapper: Processing task: {current_task.get('description')}")

    try:
        search_output_state = search_subgraph.invoke(state)
        final_search_result = _extract_final_message_content(search_output_state)

        current_task["result"] = final_search_result
        logger.info(f"Search Subgraph Wrapper: Captured result for task: {current_task.get('description')}")

        return {"current_task": current_task}
    except Exception as e:
        logger.error(f"Error in search_subgraph_wrapper for task '{current_task.get('description')}': {e}")
        current_task["result"] = f"Error executing search task: {e}"
        current_task["status"] = "failed"
        return {"current_task": current_task}


def coder_subgraph_wrapper(state: SupervisorState) -> Dict[str, Any]:
    """
    Wrapper for the coding subgraph.
    Captures the output of the coding task and stores it in current_task['result'].
    """
    current_task = state["current_task"]
    logger.info(f"Coder Subgraph Wrapper: Processing task: {current_task.get('description')}")

    try:
        coder_output_state = coder_subgraph.invoke(state)
        final_coder_result = _extract_final_message_content(coder_output_state)

        current_task["result"] = final_coder_result
        logger.info(f"Coder Subgraph Wrapper: Captured result for task: {current_task.get('description')}")

        return {"current_task": current_task}
    except Exception as e:
        logger.error(f"Error in coder_subgraph_wrapper for task '{current_task.get('description')}': {e}")
        current_task["result"] = f"Error executing coding task: {e}"
        current_task["status"] = "failed"
        return {"current_task": current_task}


def summarize_subgraph_wrapper(state: SupervisorState) -> Dict[str, Any]:
    """
    Wrapper for the summarize subgraph.
    Captures the output of the summarization task and stores it in current_task['result'].
    """
    current_task = state["current_task"]
    logger.info(f"Summarize Subgraph Wrapper: Processing task: {current_task.get('description')}")

    try:
        summarize_output_state = summarize_subgraph.invoke(state)
        final_summarize_result = _extract_final_message_content(summarize_output_state)

        current_task["result"] = final_summarize_result
        logger.info(f"Summarize Subgraph Wrapper: Captured result for task: {current_task.get('description')}")

        return {"current_task": current_task}
    except Exception as e:
        logger.error(f"Error in summarize_subgraph_wrapper for task '{current_task.get('description')}': {e}")
        current_task["result"] = f"Error executing summarization task: {e}"
        current_task["status"] = "failed"
        return {"current_task": current_task}
