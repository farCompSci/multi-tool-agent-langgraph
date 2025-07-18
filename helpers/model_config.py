from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from loguru import logger
import os


def fetch_ollama_model(model_name: str = "llama3.2") -> ChatOllama | None:
    """
    Retrieve llm from Ollama.
    :param model_name: the model name from ollama model registry
    :returns an invokable llm

    The default choice here is llama3.2 because it supports tools and is a light model you can run easily locally.
    """
    try:
        llm = ChatOllama(model=model_name)
        logger.info(f"Successfully retrieved {model_name} from Ollama.") if model_name != "llama3.2" else None
        return llm
    except Exception as e:
        logger.error(f"There was an error in retrieving {model_name} from Ollama. Details:\n{e}")
        return None


def fetch_openai_model(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    """Fetches an OpenAI Chat model."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return ChatOpenAI(model=model_name, temperature=temperature)