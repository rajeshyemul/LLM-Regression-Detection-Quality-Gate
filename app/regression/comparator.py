from typing import Any, Literal
from pydantic import BaseModel, Field

from app.evaluation.models import Baseline, RunResult


ChangeType = Literal[
    "PROMPT_CHANGE",
    "MODEL_CHANGE",
    "PROMPT_AND_MODEL_CHANGE",
    "NO_CHANGE",
]


class CaseDiff(BaseModel):
    """Evidence for a specific test case that changed status between baseline and run."""
    test_case_id: str
    diff_type: Literal["REGRESSION", "IMPROVEMENT", "UNCHANGED_PASS", "UNCHANGED_FAIL"]
    baseline_status: str
    current_status: str
    expected_category: str
    baseline_category: str | None = None
    current_category: str | None = None
    baseline_summary_score: float | None = None
    current_summary_score: float | None = None
    reason: str = ""


class ComparisonResult(BaseModel):
    """Complete diff analysis between an evaluation run and the baseline."""
    baseline_id: str
    run_id: str
    change_type: ChangeType
    attribution_message: str

    # Metric Deltas (current - baseline)
    category_accuracy_delta: float
    average_summary_score_delta: float | None = None
    p95_latency_ms_delta: float
    average_latency_ms_delta: float
    average_input_tokens_delta: float
    average_output_tokens_delta: float

    # Case-level Diff Lists
    regressions: list[CaseDiff] = Field(default_factory=list)
    improvements: list[CaseDiff] = Field(default_factory=list)
    unchanged_passes: list[str] = Field(default_factory=list)
    unchanged_fails: list[str] = Field(default_factory=list)

    # Metric Snapshots
    baseline_metrics: dict[str, Any]
    current_metrics: dict[str, Any]


def determine_change_type(
    run_prompt: str,
    baseline_prompt: str,
    run_model: str,
    baseline_model: str,
) -> tuple[ChangeType, str]:
    """Determine attribution based on which variables changed between run and baseline."""
    prompt_changed = run_prompt != baseline_prompt
    model_changed = run_model != baseline_model

    if prompt_changed and not model_changed:
        return (
            "PROMPT_CHANGE",
            f"Observed changes attributed to prompt modification ({baseline_prompt} → {run_prompt}).",
        )
    if model_changed and not prompt_changed:
        return (
            "MODEL_CHANGE",
            f"Observed changes attributed to model migration ({baseline_model} → {run_model}).",
        )
    if prompt_changed and model_changed:
        return (
            "PROMPT_AND_MODEL_CHANGE",
            "⚠️ Attribution unavailable: Both prompt and model changed simultaneously.",
        )
    return (
        "NO_CHANGE",
        "Prompt and model are identical to baseline. Variance is attributable to non-determinism or sampling.",
    )


def compare_runs(current_run: RunResult, baseline: Baseline) -> ComparisonResult:
    """Compare a current evaluation run against an approved baseline.

    Args:
        current_run: The newly executed RunResult.
        baseline: The approved reference Baseline.

    Returns:
        Comprehensive ComparisonResult with deltas, attribution, and flipped cases.
    """
    # 1. Determine Change Attribution
    change_type, attribution_msg = determine_change_type(
        run_prompt=current_run.manifest.prompt_version,
        baseline_prompt=baseline.prompt_version,
        run_model=current_run.manifest.model_name,
        baseline_model=baseline.model_name,
    )

    # 2. Compute Metric Deltas (Current - Baseline)
    base_acc = baseline.metrics.get("category_accuracy", 0.0)
    acc_delta = round(current_run.category_accuracy - base_acc, 4)

    base_summary = baseline.metrics.get("average_summary_score")
    summary_delta: float | None = None
    if current_run.average_summary_score is not None and base_summary is not None:
        summary_delta = round(current_run.average_summary_score - base_summary, 2)

    base_p95 = baseline.metrics.get("p95_latency_ms", 0.0)
    p95_delta = round(current_run.p95_latency_ms - base_p95, 2)

    base_avg_lat = baseline.metrics.get("average_latency_ms", 0.0)
    avg_lat_delta = round(current_run.average_latency_ms - base_avg_lat, 2)

    base_in_tok = baseline.metrics.get("average_input_tokens", 0.0)
    in_tok_delta = round(current_run.average_input_tokens - base_in_tok, 1)

    base_out_tok = baseline.metrics.get("average_output_tokens", 0.0)
    out_tok_delta = round(current_run.average_output_tokens - base_out_tok, 1)

    # 3. Case-Level Diffing
    baseline_cases_map = {c.test_case_id: c for c in baseline.case_results}

    regressions: list[CaseDiff] = []
    improvements: list[CaseDiff] = []
    unchanged_passes: list[str] = []
    unchanged_fails: list[str] = []

    for curr_case in current_run.cases:
        case_id = curr_case.test_case_id
        base_case = baseline_cases_map.get(case_id)

        curr_passed = curr_case.status == "PASS"
        base_passed = base_case is not None and base_case.status == "PASS"

        actual_cat = curr_case.actual.category if curr_case.actual else None
        base_cat = (
            base_case.actual.category if base_case and base_case.actual else None
        )

        if base_passed and not curr_passed:
            # REGRESSION: Previously worked, now broken
            diff = CaseDiff(
                test_case_id=case_id,
                diff_type="REGRESSION",
                baseline_status=base_case.status if base_case else "UNKNOWN",
                current_status=curr_case.status,
                expected_category=curr_case.expected.category,
                baseline_category=base_cat,
                current_category=actual_cat,
                baseline_summary_score=base_case.summary_score if base_case else None,
                current_summary_score=curr_case.summary_score,
                reason=f"Flipped from PASS to {curr_case.status} (actual: '{actual_cat}')",
            )
            regressions.append(diff)

        elif not base_passed and curr_passed:
            # IMPROVEMENT: Previously failed, now fixed
            diff = CaseDiff(
                test_case_id=case_id,
                diff_type="IMPROVEMENT",
                baseline_status=base_case.status if base_case else "UNKNOWN",
                current_status=curr_case.status,
                expected_category=curr_case.expected.category,
                baseline_category=base_cat,
                current_category=actual_cat,
                baseline_summary_score=base_case.summary_score if base_case else None,
                current_summary_score=curr_case.summary_score,
                reason=f"Flipped from {base_case.status if base_case else 'UNKNOWN'} to PASS",
            )
            improvements.append(diff)

        elif curr_passed and base_passed:
            unchanged_passes.append(case_id)
        else:
            unchanged_fails.append(case_id)

    current_metrics = {
        "total_cases": current_run.total_cases,
        "passed_cases": current_run.passed_cases,
        "failed_cases": current_run.failed_cases,
        "category_accuracy": current_run.category_accuracy,
        "average_summary_score": current_run.average_summary_score,
        "average_latency_ms": current_run.average_latency_ms,
        "p95_latency_ms": current_run.p95_latency_ms,
        "average_input_tokens": current_run.average_input_tokens,
        "average_output_tokens": current_run.average_output_tokens,
    }

    return ComparisonResult(
        baseline_id=baseline.baseline_id,
        run_id=current_run.manifest.run_id,
        change_type=change_type,
        attribution_message=attribution_msg,
        category_accuracy_delta=acc_delta,
        average_summary_score_delta=summary_delta,
        p95_latency_ms_delta=p95_delta,
        average_latency_ms_delta=avg_lat_delta,
        average_input_tokens_delta=in_tok_delta,
        average_output_tokens_delta=out_tok_delta,
        regressions=regressions,
        improvements=improvements,
        unchanged_passes=unchanged_passes,
        unchanged_fails=unchanged_fails,
        baseline_metrics=baseline.metrics,
        current_metrics=current_metrics,
    )