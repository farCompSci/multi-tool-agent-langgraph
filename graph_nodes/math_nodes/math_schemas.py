from pydantic import BaseModel,Field
from typing import Literal
class MathComplexityClassifier(BaseModel):
    message_type: Literal['basic_math', 'advanced_math'] = Field(
        ...,
        description='Classify if the message requires basic math capabilities or complex math capabilities.'
    )