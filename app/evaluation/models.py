from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field

from app.feature.classifier import ClassificationOutput


class EvalCaseResult(BaseModel):
    test_case_id: str
    expected: ClassificationOutput
    actual: ClassificationOutput | None = None
    category_match: bool
    summary_score: float | None = None
    summary_reason: str | None = None
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    status: Literal[
        "PASS",
        "FAIL",
        "ERROR",
    ]
    error: str | None = None


class RunManifest(BaseModel):
    run_id: str
    prompt_version: str
    dataset_version: str
    model_provider: str
    model_name: str
    judge_provider: str | None = None
    judge_model: str | None = None
    execution_mode: Literal[
        "real",
        "mock",
        "replay",
    ]
    baseline_id: str | None = None
    started_at: datetime
    completed_at: datetime | None = None


class RunResult(BaseModel):
    manifest: RunManifest
    total_cases: int
    passed_cases: int
    failed_cases: int
    category_accuracy: float
    average_summary_score: float | None = None
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    average_input_tokens: float
    average_output_tokens: float
    cases: list[EvalCaseResult] = Field(default_factory=list)


class Baseline(BaseModel):
    baseline_id: str
    run_id: str
    prompt_version: str
    dataset_version: str
    model_provider: str
    model_name: str
    judge_provider: str | None = None
    judge_model: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    case_results: list[EvalCaseResult] = Field(default_factory=list)
    created_at: datetime
