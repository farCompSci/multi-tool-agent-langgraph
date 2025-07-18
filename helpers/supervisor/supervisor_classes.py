from typing import TypedDict, Any, List, Optional, Dict, Annotated
import sys
import os
from langgraph.graph import add_messages

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from helpers.supervisor.supervisor_schemas import TaskQueue


class SupervisorState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], add_messages]  # This will automatically append messages
    task_queue: Optional[TaskQueue]
    current_task: Optional[Dict[str, Any]]
    task_classification: Optional[str]
    classification_reasoning: Optional[str]
    decomposition_reasoning: Optional[str]
    queue_status: Optional[str]
    thread_id: Optional[str]
    retrieved_memories: Optional[List[Dict[str, str]]]
