import sys
import os
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from graph_nodes.coder_node.coder_states import CoderState
from helpers.model_config import fetch_ollama_model
from graph_nodes.coder_node.coder_schemas import CodeOutputStructure


def code_generation(state: CoderState) -> dict:
    """
    Generates Python code based on user request.
    """
    llm = fetch_ollama_model("qwen2.5-coder:3b")
    messages = state['messages']
    if not messages or not isinstance(messages[0], dict) or messages[0].get('role') != 'system':
        system_message = {
            'role': 'system',
            'content': (
                "You are an expert Python programmer. Your task is to:\n"
                "1. Understand the user's coding request\n"
                "2. Generate clean, well-commented Python code\n"
                "3. Avoid file system operations, network requests, system commands, and dangerous imports.\n"
                "4. Provide output in the following format: output_code:str , example_usage:str, and explanation in comments"
                "Focus on algorithms, data structures, and pure computation. Keep your response as succinct as possible."
            )
        }
        messages = [system_message] + messages

    logger.info("Generating code based on user request")
    llm = llm.with_structured_output(CodeOutputStructure)
    result = llm.invoke(messages)

    return {
        "generated_code": result.model_dump()['output_code'],
        "messages": [{
            "role": "assistant",
            "content": f"Here is the generated code for your query:\n```python\n{result.model_dump()['output_code']}\n```"
        }]
    }
