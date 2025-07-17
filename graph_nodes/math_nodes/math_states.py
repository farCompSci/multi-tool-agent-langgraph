from typing import TypedDict, Annotated
from langgraph.graph import add_messages


class MathState(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    next: str | None
