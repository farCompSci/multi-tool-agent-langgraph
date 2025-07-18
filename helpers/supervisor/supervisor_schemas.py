from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, Optional


class TaskClassifier(BaseModel):
    task_type: Literal["math", "search", "code", "summarize", "general_chat"] = Field(
        description="The type of task the user is requesting"
    )
    reasoning: str = Field(description="Brief explanation of why this classification was chosen")


class TaskQueue(BaseModel):
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    current_index: int = Field(default=0)
    completed_tasks: List[Dict[str, Any]] = Field(default_factory=list)

class Task(BaseModel):
    description: str = Field(
        ...,
        description="A clear and concise description of the subtask."
    )
    type: Literal["math", "search", "code", "summarize", "retrieve_info", "general_chat"] = Field(
        ...,
        description="The type of tool or action required for this subtask."
    )
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        "pending",
        description="The current status of the subtask."
    )

class MultiStepDecision(BaseModel):
    is_multi_step: bool = Field(
        ...,
        description="True if the user's request requires multiple distinct steps or tool invocations to complete, False otherwise."
    )
    reasoning: str = Field(
        ...,
        description="The reasoning behind the multi-step decision."
    )
    initial_subtasks: Optional[List[Task]] = Field(
        None,
        description="If is_multi_step is True, a list of initial subtasks to start the decomposition. Otherwise, None."
    )

class DetailedTaskDecomposition(BaseModel):
    tasks: List[Task] = Field(
        ...,
        description="A list of decomposed subtasks, ordered for sequential execution."
    )
