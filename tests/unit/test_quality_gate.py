import pytest

from app.regression.comparator import CaseDiff, ComparisonResult
from app.regression.quality_gate import QualityGateResult, evaluate_quality_gate
from app.regression.thresholds import QualityThresholds


def make_comparison(
    accuracy_delta: float = 0.0,
    regressions: list[CaseDiff] | None = None,
    improvements: list[CaseDiff] | None = None,
    summary_delta: float | None = 0.0,
    p95_delta: float = 0.0,
    base_p95: float = 100.0,
) -> ComparisonResult:
    """Helper to create ComparisonResult fixtures with controlled deltas."""
    return ComparisonResult(
        baseline_id="BASE-001",
        run_id="RUN-002",
        change_type="NO_CHANGE",
        attribution_message="No change",
        category_accuracy_delta=accuracy_delta,
        average_summary_score_delta=summary_delta,
        p95_latency_ms_delta=p95_delta,
        average_latency_ms_delta=p95_delta,
        average_input_tokens_delta=0.0,
        average_output_tokens_delta=0.0,
        regressions=regressions or [],
        improvements=improvements or [],
        unchanged_passes=["TC-001"],
        unchanged_fails=[],
        baseline_metrics={"p95_latency_ms": base_p95, "category_accuracy": 0.9},
        current_metrics={"p95_latency_ms": base_p95 + p95_delta, "category_accuracy": 0.9 + accuracy_delta},
    )


def test_quality_gate_no_baseline_returns_warning():
    result = evaluate_quality_gate(comparison=None)
    assert isinstance(result, QualityGateResult)
    assert result.status == "WARNING"
    assert any("No approved baseline exists" in r for r in result.reasons)


def test_quality_gate_clean_run_returns_pass():
    comp = make_comparison()
    result = evaluate_quality_gate(comp)
    assert result.status == "PASS"
    assert len(result.regressions) == 0
    assert any("All quality gate criteria satisfied" in r for r in result.reasons)


def test_quality_gate_accuracy_warning():
    # Warning threshold is -0.03 (-3%), Critical is -0.05 (-5%)
    comp = make_comparison(accuracy_delta=-0.04)
    result = evaluate_quality_gate(comp)
    assert result.status == "WARNING"
    assert any("WARNING: Category accuracy dropped" in r for r in result.reasons)


def test_quality_gate_accuracy_critical():
    comp = make_comparison(accuracy_delta=-0.06)
    result = evaluate_quality_gate(comp)
    assert result.status == "CRITICAL"
    assert any("CRITICAL: Category accuracy dropped" in r for r in result.reasons)


def test_quality_gate_case_regression_warning():
    # 1 regression -> exceeds max_regressions_warning (0), but <= max_regressions_critical (2)
    diff = CaseDiff(
        test_case_id="TC-005",
        diff_type="REGRESSION",
        baseline_status="PASS",
        current_status="FAIL",
        expected_category="billing",
    )
    comp = make_comparison(regressions=[diff])
    result = evaluate_quality_gate(comp)
    assert result.status == "WARNING"
    assert result.regressions == ["TC-005"]
    assert any("WARNING: 1 test case(s) regressed" in r for r in result.reasons)


def test_quality_gate_case_regression_critical():
    # 3 regressions -> exceeds max_regressions_critical (2)
    diffs = [
        CaseDiff(
            test_case_id=f"TC-00{i}",
            diff_type="REGRESSION",
            baseline_status="PASS",
            current_status="FAIL",
            expected_category="billing",
        )
        for i in range(1, 4)
    ]
    comp = make_comparison(regressions=diffs)
    result = evaluate_quality_gate(comp)
    assert result.status == "CRITICAL"
    assert len(result.regressions) == 3
    assert any("CRITICAL: 3 test case(s) regressed" in r for r in result.reasons)


def test_quality_gate_summary_score_warning_and_critical():
    # Warning is -0.20, Critical is -0.40
    comp_warn = make_comparison(summary_delta=-0.25)
    res_warn = evaluate_quality_gate(comp_warn)
    assert res_warn.status == "WARNING"

    comp_crit = make_comparison(summary_delta=-0.45)
    res_crit = evaluate_quality_gate(comp_crit)
    assert res_crit.status == "CRITICAL"


def test_quality_gate_latency_inflation_warning_and_critical():
    # Base = 100ms. Warning is +20%, Critical is +50%
    comp_warn = make_comparison(p95_delta=25.0, base_p95=100.0)  # +25%
    res_warn = evaluate_quality_gate(comp_warn)
    assert res_warn.status == "WARNING"

    comp_crit = make_comparison(p95_delta=55.0, base_p95=100.0)  # +55%
    res_crit = evaluate_quality_gate(comp_crit)
    assert res_crit.status == "CRITICAL"


def test_quality_gate_multiple_violations_escalate_to_critical():
    # Warning on latency (+25%), but Critical on accuracy (-6%)
    comp = make_comparison(accuracy_delta=-0.06, p95_delta=25.0, base_p95=100.0)
    result = evaluate_quality_gate(comp)
    assert result.status == "CRITICAL"
    assert len(result.reasons) == 2


def test_quality_gate_custom_thresholds():
    strict = QualityThresholds(
        category_accuracy_warning=-0.01,
        category_accuracy_critical=-0.02,
    )
    comp = make_comparison(accuracy_delta=-0.015)
    res = evaluate_quality_gate(comp, thresholds=strict)
    assert res.status == "WARNING"
