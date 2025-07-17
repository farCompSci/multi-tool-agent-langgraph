from loguru import logger
import sys
import os
import subprocess
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from helpers.model_config import fetch_ollama_model
from graph_nodes.coder_node.coder_states import CoderState
from graph_nodes.coder_node.coder_schemas import CodeSafetyClassifier
from helpers.prompting.coder.prompt_templates import build_few_shot_prompt


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