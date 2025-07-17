from pydantic import BaseModel, Field
from typing import Literal
from langgraph.graph import StateGraph, START, END
import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from helpers.model_config import fetch_ollama_model
from basic_math_node import basic_math_subgraph
from advanced_math_node import advanced_math_subgraph
from math_states import State


class MathComplexityClassifier(BaseModel):
    message_type: Literal['basic_math', 'advanced_math'] = Field(
        ...,
        description='Classify if the message requires basic math capabilities or complex math capabilities.'
    )


def classify_math_complexity(state: State):
    last_message = state["messages"][-1]
    llm = fetch_ollama_model("llama3.2")
    classifier_llm = llm.with_structured_output(MathComplexityClassifier)
    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user input as either: 
                       'basic_math': if it involves simple mathematical requests. Those include addition, division, multiplication, subtraction, or raising a number to a certain power.
                       'advanced_math': if it involves complex mathematical requests. Those include things like derivatives, integrals, differential equations, and statistical questions. 
                       """
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ])
    return {"message_type": result.message_type}


def math_tool_router(state: State):
    message_type = state.get("message_type", "advanced_math")
    if message_type == "basic_math":
        return {"next": "basic_math"}
    return {"next": "advanced_math"}


# Build the graph
main_graph = StateGraph(State)
main_graph.add_node("math_difficulty_classifier", classify_math_complexity)
main_graph.add_node("math_tool_router", math_tool_router)
main_graph.add_node("basic_math", basic_math_subgraph)  # Subgraph as a node!
main_graph.add_node("advanced_math", advanced_math_subgraph)

main_graph.add_edge(START, "math_difficulty_classifier")
main_graph.add_edge("math_difficulty_classifier", "math_tool_router")
main_graph.add_conditional_edges(
    "math_tool_router",
    lambda state: state.get("next"),
    {"basic_math": "basic_math", "advanced_math": "advanced_math"}
)
main_graph.add_edge("advanced_math", END)
main_graph.add_edge("basic_math", END)

main_graph = main_graph.compile()

# main_graph.get_graph().draw_mermaid_png(output_file_path="output_main_math1.png")

if __name__ == "__main__":
    load_dotenv()

    state_example = {
        'messages': [
            {
                'role': 'user',
                'content': 'Multiply 2 by 10. Divide the result by 5. Then subtract 3 from the result. Finally add 1 to the result.'
            }
        ]
    }
    responses = main_graph.invoke(state_example)
    for response in responses['messages']:
        response.pretty_print()

    main_graph.get_graph().draw_mermaid_png(output_file_path='./graph_nodes/math_nodes/combined_math_nodes.png')
