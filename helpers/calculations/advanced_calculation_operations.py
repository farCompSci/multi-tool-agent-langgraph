import os
import requests
import xml.etree.ElementTree as ET
from loguru import logger
from typing import Optional
from langchain_core.tools import tool

@tool
def ask_wolfram(query: str) -> Optional[str]:
    """
    Public function calling Wolfram Alpha to perform complex math operations.
    :param query: str - the query from the user
    :return: str or None - the output of the complex mathematical operation, or None if not found
    """
    app_id = os.getenv("APP_ID")
    if not app_id:
        logger.error("APP_ID environment variable not found")
        return None

    try:
        result = _ask_wolfram_inner(query, app_id)
        if result:
            logger.info("Successfully retrieved results from advanced math agent.")
        else:
            logger.error("Could not retrieve results from advanced agent")
        return result
    except Exception as e:
        logger.error(f"Failed to retrieve results from advanced math agent. Details:\n{e}")
        return None


def _ask_wolfram_inner(query: str, app_id: str) -> str:
    """
    Makes a call to Wolfram Alpha and retrieves output
    :param query: str - The mathematical or computational query to send to Wolfram Alpha
    :param app_id: str - The Wolfram Alpha API application ID for authentication
    :returns: str - The computed result from Wolfram Alpha, or an error message if the request fails
    """
    url = "http://api.wolframalpha.com/v2/query"
    params = {
        "appid": app_id,
        "input": query,
        "format": "plaintext",
        "output": "XML"
    }

    logger.info(f"Making request to Wolfram Alpha API for query: {query}")
    response = requests.get(url, params=params)

    if response.status_code != 200:
        error_msg = f"HTTP error: {response.status_code}"
        logger.error(error_msg)
        return error_msg

    root = ET.fromstring(response.content)
    success = root.attrib.get("success", "false")

    if success != "true":
        error_msg = "Sorry, Wolfram Alpha could not compute the answer."
        logger.error(error_msg)
        return error_msg

    logger.info("Successfully received response from Wolfram Alpha")

    # Find the 'Result' pod
    for pod in root.findall(".//pod"):
        title = pod.attrib.get("title", "").lower()
        if "result" in title or "derivative" in title or "integral" in title:
            for subpod in pod.findall("subpod"):
                plaintext = subpod.find("plaintext")
                if plaintext is not None and plaintext.text:
                    logger.info(f"Found result in pod '{title}': {plaintext.text}")
                    return plaintext.text

    # Fallback: return all pod titles and plaintexts
    logger.info("No specific result pod found, collecting all available results")
    results = []
    for pod in root.findall(".//pod"):
        title = pod.attrib.get("title", "")
        for subpod in pod.findall("subpod"):
            plaintext = subpod.find("plaintext")
            if plaintext is not None and plaintext.text:
                results.append(f"{title}: {plaintext.text}")

    if results:
        logger.info(f"Returning {len(results)} fallback results")
        return "\n".join(results)
    else:
        error_msg = "No answer found."
        logger.error(error_msg)
        return error_msg

if __name__ == "__main__":
    result = ask_wolfram('What is the derivative of 2x^2')
    print(result)