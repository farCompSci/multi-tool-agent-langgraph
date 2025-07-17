from langchain_core.tools import tool
from typing import Optional
from tavily import TavilyClient
from loguru import logger
import os

def _search_tool_inner(query: str, api_key: str) -> Optional[str]:
    """
    Performs web searching using Tavily API
    :param query: str - The searching query from the user
    :return: str or None - The searching result answer, or None if searching fails
    """


    try:
        client = TavilyClient(api_key=api_key)
        logger.info(f'Successfully initialized Tavily client.')

        result = client.search(query, include_answer=True)

        if result and "answer" in result and result["answer"]:
            logger.info("Successfully retrieved searching results")
            return result["answer"]
        else:
            logger.warning("No answer found in searching results")
            return "No answer found for your searching query."

    except Exception as e:
        logger.error(f"Failed to perform searching. Details: {e}")
        return None


@tool
def search_tool(query:str):
    """
    Performs web searching using Tavily API, putting together the api key and query
    :param query:
    :return:
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        logger.error("TAVILY_API_KEY environment variable not found")
        return None

    resulting_message = _search_tool_inner(query, api_key)

    return resulting_message