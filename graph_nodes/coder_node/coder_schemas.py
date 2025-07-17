from pydantic import BaseModel, Field
from typing import Literal, Optional


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