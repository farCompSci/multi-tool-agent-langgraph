from langchain_core.tools import tool
from loguru import logger


@tool
def read_file_tool(file_path: str) -> str:
    """
    Reads a text file and returns its content for summarizing.
    :param file_path: Path to the file to read
    :return: The file's content as a string
    """
    try:
        logger.info(f"Reading file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Successfully read file with {len(content)} characters")
        return content
    except Exception as e:
        error_msg = f"Error reading file: {e}"
        logger.error(error_msg)
        return error_msg
