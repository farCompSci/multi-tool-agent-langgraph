import sys
import os
from langgraph.graph import START, END, StateGraph

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from helpers.coding.security.code_security_evaluations import security_review_code,security_aggregation_node,static_analysis_bandit
from helpers.coding.execution.code_execution import execute_python_code_with_review
from helpers.coding.generation.code_generation import code_generation
from graph_nodes.coder_node.coder_states import CoderState

# Build the graph
coder_graph_builder = StateGraph(CoderState)
coder_graph_builder.add_node("code_generator", code_generation)
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
             'content': 'Generate a function for finding out if a string is a palindrome, using python!'}],
        }

    output = coder_graph.invoke(state)
    for message in output['messages']:
        message.pretty_print()