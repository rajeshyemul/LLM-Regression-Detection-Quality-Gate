from typing import Literal
from pydantic import BaseModel, Field

from app.regression.comparator import ComparisonResult
from app.regression.thresholds import QualityThresholds


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


def evaluate_quality_gate(
    comparison: ComparisonResult | None,
    thresholds: QualityThresholds | None = None,
) -> QualityGateResult:
    """Evaluate whether an evaluation run satisfies the quality policy.

    Args:
        comparison: ComparisonResult against baseline, or None if no baseline exists.
        thresholds: Policy thresholds (defaults to standard QualityThresholds).

    Returns:
        Structured QualityGateResult with status PASS, WARNING, or CRITICAL.
    """
    thresholds = thresholds or QualityThresholds()

    # Policy for runs without an approved baseline
    if comparison is None:
        return QualityGateResult(
            status="WARNING",
            reasons=[
                "No approved baseline exists. Evaluation completed, but regression comparison could not be performed."
            ],
            regressions=[],
            improvements=[],
            metric_deltas={},
        )

    reasons: list[str] = []
    is_critical = False
    is_warning = False

    metric_deltas: dict[str, float] = {
        "category_accuracy_delta": comparison.category_accuracy_delta,
        "p95_latency_ms_delta": comparison.p95_latency_ms_delta,
        "average_latency_ms_delta": comparison.average_latency_ms_delta,
        "average_input_tokens_delta": comparison.average_input_tokens_delta,
        "average_output_tokens_delta": comparison.average_output_tokens_delta,
    }
    if comparison.average_summary_score_delta is not None:
        metric_deltas["average_summary_score_delta"] = comparison.average_summary_score_delta

    # 1. Functional Check: Category Accuracy Drop
    acc_delta = comparison.category_accuracy_delta
    if acc_delta < thresholds.category_accuracy_critical:
        is_critical = True
        reasons.append(
            f"CRITICAL: Category accuracy dropped by {acc_delta:.2%} "
            f"(critical limit: {thresholds.category_accuracy_critical:.2%})."
        )
    elif acc_delta < thresholds.category_accuracy_warning:
        is_warning = True
        reasons.append(
            f"WARNING: Category accuracy dropped by {acc_delta:.2%} "
            f"(warning limit: {thresholds.category_accuracy_warning:.2%})."
        )

    # 2. Case Regressions Check: Number of PASS -> FAIL flips
    num_regressions = len(comparison.regressions)
    if num_regressions > thresholds.max_regressions_critical:
        is_critical = True
        reasons.append(
            f"CRITICAL: {num_regressions} test case(s) regressed from PASS to FAIL "
            f"(critical limit: {thresholds.max_regressions_critical})."
        )
    elif num_regressions > thresholds.max_regressions_warning:
        is_warning = True
        reasons.append(
            f"WARNING: {num_regressions} test case(s) regressed from PASS to FAIL "
            f"(warning limit: {thresholds.max_regressions_warning})."
        )

    # 3. Semantic Check: Summary Score Drop
    if comparison.average_summary_score_delta is not None:
        score_delta = comparison.average_summary_score_delta
        if score_delta < thresholds.summary_score_critical:
            is_critical = True
            reasons.append(
                f"CRITICAL: Summary judge score dropped by {score_delta:.2f} pts "
                f"(critical limit: {thresholds.summary_score_critical:.2f})."
            )
        elif score_delta < thresholds.summary_score_warning:
            is_warning = True
            reasons.append(
                f"WARNING: Summary judge score dropped by {score_delta:.2f} pts "
                f"(warning limit: {thresholds.summary_score_warning:.2f})."
            )

    # 4. Operational Check: P95 Latency Inflation
    base_p95 = comparison.baseline_metrics.get("p95_latency_ms", 0.0)
    if base_p95 > 0.0:
        latency_pct_change = (comparison.p95_latency_ms_delta / base_p95) * 100.0
        if latency_pct_change > thresholds.p95_latency_critical_percent:
            is_critical = True
            reasons.append(
                f"CRITICAL: P95 latency inflated by {latency_pct_change:.1f}% "
                f"(+{comparison.p95_latency_ms_delta:.1f} ms, critical limit: +{thresholds.p95_latency_critical_percent:.0f}%)."
            )
        elif latency_pct_change > thresholds.p95_latency_warning_percent:
            is_warning = True
            reasons.append(
                f"WARNING: P95 latency inflated by {latency_pct_change:.1f}% "
                f"(+{comparison.p95_latency_ms_delta:.1f} ms, warning limit: +{thresholds.p95_latency_warning_percent:.0f}%)."
            )

    # Determine overall status
    if is_critical:
        status = "CRITICAL"
    elif is_warning:
        status = "WARNING"
    else:
        status = "PASS"
        reasons.append("All quality gate criteria satisfied. No regressions detected.")

    return QualityGateResult(
        status=status,
        reasons=reasons,
        regressions=[r.test_case_id for r in comparison.regressions],
        improvements=[i.test_case_id for i in comparison.improvements],
        metric_deltas=metric_deltas,
    )
