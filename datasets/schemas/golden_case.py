from typing import Literal
from pydantic import BaseModel, Field


class GoldenTestCase(BaseModel):
    """A human-verified customer support test case."""
    id: str = Field(..., description="Stable test case identifier, e.g. 'TC-001'")
    input: str = Field(..., description="Raw email text from the customer")
    expected_category: Literal[
        "billing",
        "technical",
        "account",
        "general",
    ]
    expected_summary: str = Field(..., description="Human-curated ideal summary")
    expected_difficulty: Literal[
        "easy",
        "medium",
        "hard",
        "ambiguous",
        "noisy",
        "multilingual",
        "adversarial",
    ]
    notes: str = Field(default="", description="Editorial rationale explaining why this case matters")

class GoldenDataset(BaseModel):
    """A versioned collection of golden test cases."""
    version: str = Field(..., description="Dataset version, e.g. 'v1.0'")
    description: str = Field(default="", description="Dataset curation notes or focus area")
    cases: list[GoldenTestCase] = Field(..., min_length=1)