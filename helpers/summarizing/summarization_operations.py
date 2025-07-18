import os
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from loguru import logger


@tool
def read_file_content(file_path: str) -> str:
    """
    Reads content from a file (.txt or .pdf) and returns its raw text content.

    :param file_path: The local path to the .txt or .pdf file.
    :return: A string containing the file's full text content.
    """
    logger.info(f"Attempting to read file content from: {file_path}")

    try:
        _, extension = os.path.splitext(file_path)

        if extension.lower() == ".pdf":
            logger.info("PDF file detected. Using PyPDFLoader to extract text.")
            loader = PyPDFLoader(file_path)
            # load_and_split() gets all pages as a list of Document objects
            pages = loader.load_and_split()
            content = "\n".join(page.page_content for page in pages)

        elif extension.lower() == ".txt":
            logger.info("Text file detected. Reading raw content.")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            unsupported_msg = f"Unsupported file type: '{extension}'. This tool only supports .txt and .pdf files."
            logger.warning(unsupported_msg)
            return unsupported_msg

        logger.info(f"Successfully extracted {len(content)} characters from {file_path}")
        return content

    except FileNotFoundError:
        error_msg = f"Error: The file was not found at the specified path: {file_path}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred while reading the file: {e}"
        logger.error(error_msg)
        return error_msg