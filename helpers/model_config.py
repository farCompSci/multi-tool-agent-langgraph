from langchain_ollama import ChatOllama
from loguru import logger


def fetch_ollama_model(model_name: str = "llama3.2") -> ChatOllama | None:
    """
    Retrieve llm from Ollama.
    :param model_name: the model name from ollama model registry
    :returns an invokable llm

    The default choice here is llama3.2 because it supports tools and is a light model you can run easily locally.
    """
    try:
        llm = ChatOllama(model=model_name)
        logger.info(f"Successfully retrieved {model_name} from Ollama.")
        return llm
    except Exception as e:
        logger.error(f"There was an error in retrieving {model_name} from Ollama. Details:\n{e}")
        return None
