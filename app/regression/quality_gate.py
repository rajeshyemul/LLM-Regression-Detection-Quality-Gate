from typing import Literal
from pydantic import BaseModel, Field


class QualityGateResult(BaseModel):
    status: Literal[
        "PASS",
        "WARNING",
        "CRITICAL",
    ]
    reasons: list[str] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    metric_deltas: dict[str, float] = Field(default_factory=dict)
