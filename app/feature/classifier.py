from typing import Literal
from pydantic import BaseModel


class ClassificationOutput(BaseModel):
    category: Literal[
        "billing",
        "technical",
        "account",
        "general",
    ]
    summary: str
