from typing import Literal
from pydantic import BaseModel


class GoldenTestCase(BaseModel):
    id: str
    input: str
    expected_category: Literal[
        "billing",
        "technical",
        "account",
        "general",
    ]
    expected_summary: str
    expected_difficulty: Literal[
        "easy",
        "medium",
        "hard",
        "ambiguous",
        "noisy",
        "multilingual",
        "adversarial",
    ]
    notes: str = ""
