from langgraph.graph import MessagesState
from typing import Optional


class CoderState(MessagesState):
    generated_code: Optional[str] = None
    llm_security_review: Optional[str] = None
    static_analysis_review: Optional[str] = None
    execution_result: Optional[str] = None