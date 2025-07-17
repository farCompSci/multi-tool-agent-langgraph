from loguru import logger
import sys
import os
from pydantic import BaseModel, Field
from typing import Literal, Optional
import subprocess
import tempfile
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.messages import SystemMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from helpers.model_config import fetch_ollama_model


class CodeSafetyClassifier(BaseModel):
    message_type: Literal['APPROVE', 'REJECT'] = Field(
        description='Determine whether to APPROVE or REJECT the code provided. If the code provided is not malicious then APPROVE. Otherwise, classify as REJECT'
    )
    reasoning: str = Field(
        description='Brief explanation of why the code was approved or rejected'
    )


class CodeOutputStructure(BaseModel):
    output_code: str = Field(
        description='The executable code output from the code generating llm. This should only include python code')
    code_explanation: Optional[str] = Field(
        description='A brief description of generated code if necessary and provided.')
    example_usage: Optional[str] = Field(description='Example usage of the code with a few variables.')


class CoderState(MessagesState):
    generated_code: Optional[str] = None
    llm_security_review: Optional[str] = None
    static_analysis_review: Optional[str] = None
    execution_result: Optional[str] = None


def build_few_shot_prompt(code: str) -> str:
    few_shot_examples = [
        {
            "input": "def add(a, b):\n    return a + b",
            "output": "APPROVE: This is a simple addition function with no security risks."
        },
        {
            "input": "def find_substring(main_string: str, sub_string: str) -> int:\n   # Use the find method to locate the start index of the substring\n  return main_string.find(sub_string)",
            "output": "APPROVE: This is a simple addition function with no security risks."

        },
        {
            "input": "def multiply(a, b):\n    return a * b",
            "output": "APPROVE: This is a simple multiplication function with no security risks."
        },
        {
            "input": "def power(a, b):\n    return a ** b",
            "output": "APPROVE: This is a simple power function with no security risks."
        },
        {
            "input": "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)",
            "output": "APPROVE: This is a pure computation with no security risks."
        },
        {
            "input": "import os\nos.system('rm -rf /')",
            "output": "REJECT: This code executes a dangerous system command."
        },
        {
            "input": "import socket\ns = socket.socket()\ns.connect(('example.com', 80))",
            "output": "REJECT: This code opens a network connection, which is a security risk."
        },
        {
            "input": "with open" + "('/etc/passwd') as f:\n    data = f.read()",
            "output": "REJECT: This code reads a sensitive system file."
        }
    ]
    prompt = (
        "You are a security expert reviewing Python code for execution safety. "
        "Analyze ONLY the code provided for these security risks:\n"
        "- File system access (reading/writing sensitive files)\n"
        "- Network operations (HTTP requests, socket connections)\n"
        "- System commands (os.system, subprocess)\n"
        "- Import of dangerous modules\n"
        "- Infinite loops or resource exhaustion\n"
        "- Code obfuscation or suspicious patterns\n\n"
        "APPROVE pure algorithms, computations, and data structures that don't use dangerous operations. "
        "REJECT only if you see actual dangerous operations in the code itself.\n\n"
        "Here are some examples:\n"
    )
    for ex in few_shot_examples:
        prompt += f"Input:\n{ex['input']}\nOutput:\n{ex['output']}\n\n"
    prompt += f"Now look at this code and decide whether to APPROVE or REJECT, and explain your decision:\n{code}\nOutput:"
    return prompt


def security_review_code(state: CoderState):
    try:
        code = state.get('generated_code')
        if not code:
            return {
                'llm_security_review': 'REJECT',
                'messages': [{'role': 'assistant', 'content': 'REJECT: No code to review'}]
            }

        security_llm = fetch_ollama_model("llama3.2")
        security_llm_structured = security_llm.with_structured_output(CodeSafetyClassifier)
        prompt = build_few_shot_prompt(code)
        security_prompt = [
            {'role': 'system', 'content': prompt}
        ]
        result = security_llm_structured.invoke(security_prompt)
        decision = result.message_type
        reasoning = result.reasoning

        return {
            'llm_security_review': decision,
            'messages': [{'role': 'assistant', 'content': f"{decision}: {reasoning}"}]
        }
    except Exception as e:
        logger.error(f"Security review failed: {e}")
        return {
            'llm_security_review': 'REJECT',
            'messages': [{'role': 'assistant', 'content': "REJECT: Security review system error"}]
        }


def static_analysis_bandit(state: CoderState):
    """
    Runs Bandit static analysis on the code and returns the result.
    """
    try:
        code = state.get('generated_code')
        if not code:
            return {
                'static_analysis_review': 'REJECT',
                'messages': [{'role': 'assistant', 'content': 'REJECT: No code to analyze'}]
            }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        result = subprocess.run(
            ["bandit", "-r", temp_file, "-f", "json"],
            capture_output=True,
            text=True,
            timeout=20
        )
        os.unlink(temp_file)

        if result.returncode == 0:
            return {
                'static_analysis_review': 'APPROVE',
                'messages': [{'role': 'assistant', 'content': result.stdout.strip()}]
            }
        else:
            return {
                'static_analysis_review': 'REJECT',
                'messages': [{'role': 'assistant', 'content': result.stderr}]
            }

    except Exception as e:
        logger.error(f"Static analysis failed: {e}")
        return {
            'static_analysis_review': 'REJECT',
            'messages': [{'role': 'assistant', 'content': f"Static analysis error: {e}"}]
        }


def security_aggregation_node(state: CoderState):
    """
    Waits for both security checks and determines if code should be executed.
    """
    llm_review = state.get('llm_security_review')
    static_review = state.get('static_analysis_review')

    if llm_review == 'APPROVE' and static_review == 'APPROVE':
        return {
            'messages': [{'role': 'assistant', 'content': 'Security checks passed. Proceeding to execution.'}]
        }
    else:
        return {
            'messages': [
                {'role': 'assistant', 'content': f'Security checks failed. LLM: {llm_review}, Static: {static_review}'}]
        }


def execute_python_code_with_review(state: CoderState):
    """
    Executes Python code only after security review approval.
    """
    try:
        llm_review = state.get('llm_security_review')
        static_review = state.get('static_analysis_review')

        if llm_review == 'APPROVE' and static_review == 'APPROVE':
            logger.info("Code approved by security review, executing...")
            code = state.get('generated_code')

            if not code:
                return {
                    'execution_result': 'Error: No code to execute',
                    'messages': [{'role': 'assistant', 'content': 'Error: No code to execute'}]
                }

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode == 0:
                    output = result.stdout.strip()
                    execution_result = f"Execution Result:\n{output}" if output else "Code executed successfully (no output)"
                else:
                    error = result.stderr.strip()
                    execution_result = f"Execution failed:\n{error}"

            except subprocess.TimeoutExpired:
                execution_result = "Error: Code execution timed out (30 seconds limit)"
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

        else:
            logger.warning("Code rejected by security review")
            execution_result = "Code execution blocked because it failed the security tests!"

        return {
            'execution_result': execution_result,
            'messages': [{'role': 'assistant', 'content': execution_result}]
        }

    except Exception as e:
        logger.error(f"Error in secure code execution: {e}")
        error_msg = f"Error: {e}"
        return {
            'execution_result': error_msg,
            'messages': [{'role': 'assistant', 'content': error_msg}]
        }


def code_generation_node(state: CoderState) -> dict:
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


# Build the graph
coder_graph_builder = StateGraph(CoderState)
coder_graph_builder.add_node("code_generator", code_generation_node)
coder_graph_builder.add_node("llm_security_checker", security_review_code)
coder_graph_builder.add_node("bandits_static_security_checker", static_analysis_bandit)
coder_graph_builder.add_node("security_aggregation", security_aggregation_node)
coder_graph_builder.add_node("code_executor", execute_python_code_with_review)

# Set up edges for parallel execution
coder_graph_builder.add_edge(START, "code_generator")
coder_graph_builder.add_edge("code_generator", "llm_security_checker")
coder_graph_builder.add_edge("code_generator", "bandits_static_security_checker")
coder_graph_builder.add_edge(["llm_security_checker", "bandits_static_security_checker"], "security_aggregation")
coder_graph_builder.add_edge("security_aggregation", "code_executor")
coder_graph_builder.add_edge("code_executor", END)

coder_graph = coder_graph_builder.compile()
coder_graph.get_graph().draw_mermaid_png(output_file_path="code_generator.png")

if __name__ == "__main__":
    state: CoderState = {
        'messages': [
            {'role': 'user',
             'content': 'Generate a function for finding a substring in a string, using python!'}],
        }

    output = coder_graph.invoke(state)
    for message in output['messages']:
        message.pretty_print()