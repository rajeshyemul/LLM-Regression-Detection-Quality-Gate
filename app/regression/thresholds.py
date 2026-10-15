from pydantic import BaseModel, Field


class QualityThresholds(BaseModel):
    """Configurable quality policy thresholds for regression detection.

    All delta thresholds are maximum acceptable drops (negative numbers).
    Latency thresholds are maximum acceptable percentage increases.
    """

    # Functional: Category Accuracy drop (ratio scale: -0.03 = -3%)
    category_accuracy_warning: float = Field(
        default=-0.03,
        description="Max acceptable drop in category accuracy before WARNING.",
    )
    category_accuracy_critical: float = Field(
        default=-0.05,
        description="Max acceptable drop in category accuracy before CRITICAL.",
    )

    # Functional: Absolute case regression count (PASS -> FAIL flips)
    max_regressions_warning: int = Field(
        default=0,
        description="Max flipped cases (PASS -> FAIL) allowed before WARNING.",
    )
    max_regressions_critical: int = Field(
        default=2,
        description="Max flipped cases (PASS -> FAIL) allowed before CRITICAL.",
    )

    # Semantic: Summary Judge Score drop (1-5 rubric point scale)
    summary_score_warning: float = Field(
        default=-0.20,
        description="Max acceptable drop in average summary score before WARNING.",
    )
    summary_score_critical: float = Field(
        default=-0.40,
        description="Max acceptable drop in average summary score before CRITICAL.",
    )

    # Operational: P95 Latency inflation percentage (20.0 = +20%)
    p95_latency_warning_percent: float = Field(
        default=20.0,
        description="Max acceptable percentage increase in P95 latency before WARNING.",
    )
    p95_latency_critical_percent: float = Field(
        default=50.0,
        description="Max acceptable percentage increase in P95 latency before CRITICAL.",
    )
