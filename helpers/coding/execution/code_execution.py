from loguru import logger
import subprocess
import tempfile
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from graph_nodes.coder_node.coder_states import CoderState

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
