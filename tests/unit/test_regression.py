from datetime import datetime, timezone
import pytest

from app.evaluation.models import (
    Baseline,
    EvalCaseResult,
    RunManifest,
    RunResult,
)
from app.feature.classifier import ClassificationOutput
from app.regression.comparator import (
    CaseDiff,
    ComparisonResult,
    compare_runs,
    determine_change_type,
)


# ── Attribution Tests ────────────────────────────────────────────────────────

def test_attribution_prompt_change():
    change_type, msg = determine_change_type(
        run_prompt="v2.0",
        baseline_prompt="v1.0",
        run_model="qwen2.5:3b",
        baseline_model="qwen2.5:3b",
    )
    assert change_type == "PROMPT_CHANGE"
    assert "v1.0 → v2.0" in msg


def test_attribution_model_change():
    change_type, msg = determine_change_type(
        run_prompt="v1.0",
        baseline_prompt="v1.0",
        run_model="qwen2.5:7b",
        baseline_model="qwen2.5:3b",
    )
    assert change_type == "MODEL_CHANGE"
    assert "qwen2.5:3b → qwen2.5:7b" in msg


def test_attribution_both_changed():
    change_type, msg = determine_change_type(
        run_prompt="v2.0",
        baseline_prompt="v1.0",
        run_model="qwen2.5:7b",
        baseline_model="qwen2.5:3b",
    )
    assert change_type == "PROMPT_AND_MODEL_CHANGE"
    assert "Attribution unavailable" in msg


def test_attribution_no_change():
    change_type, msg = determine_change_type(
        run_prompt="v1.0",
        baseline_prompt="v1.0",
        run_model="qwen2.5:3b",
        baseline_model="qwen2.5:3b",
    )
    assert change_type == "NO_CHANGE"
    assert "identical" in msg


# ── Diff & Regression Tests ──────────────────────────────────────────────────

def build_fixture_data():
    """Helper to build baseline and run fixtures with known flipped cases."""
    # Baseline: TC-001 passed, TC-002 failed, TC-003 passed, TC-004 failed
    base_cases = [
        EvalCaseResult(
            test_case_id="TC-001",
            expected=ClassificationOutput(category="billing", summary="Charge error"),
            actual=ClassificationOutput(category="billing", summary="Charge error"),
            category_match=True,
            summary_score=5.0,
            latency_ms=100.0,
            status="PASS",
        ),
        EvalCaseResult(
            test_case_id="TC-002",
            expected=ClassificationOutput(category="technical", summary="App crash"),
            actual=ClassificationOutput(category="billing", summary="App crash"),
            category_match=False,
            summary_score=4.0,
            latency_ms=120.0,
            status="FAIL",
        ),
        EvalCaseResult(
            test_case_id="TC-003",
            expected=ClassificationOutput(category="account", summary="Password reset"),
            actual=ClassificationOutput(category="account", summary="Password reset"),
            category_match=True,
            summary_score=5.0,
            latency_ms=90.0,
            status="PASS",
        ),
        EvalCaseResult(
            test_case_id="TC-004",
            expected=ClassificationOutput(category="general", summary="Feature request"),
            actual=ClassificationOutput(category="technical", summary="Feature request"),
            category_match=False,
            summary_score=3.0,
            latency_ms=110.0,
            status="FAIL",
        ),
    ]

    baseline = Baseline(
        baseline_id="BASE-001",
        run_id="RUN-BASE",
        prompt_version="v1.0",
        dataset_version="v1.0",
        model_provider="mock",
        model_name="qwen2.5:3b",
        metrics={
            "total_cases": 4,
            "passed_cases": 2,
            "failed_cases": 2,
            "category_accuracy": 0.50,
            "average_summary_score": 4.25,
            "average_latency_ms": 105.0,
            "p95_latency_ms": 120.0,
            "average_input_tokens": 15.0,
            "average_output_tokens": 10.0,
        },
        case_results=base_cases,
        created_at=datetime.now(timezone.utc),
    )

    # Current Run:
    # - TC-001 FLIPPED TO FAIL (Regression!)
    # - TC-002 FLIPPED TO PASS (Improvement!)
    # - TC-003 UNCHANGED PASS
    # - TC-004 UNCHANGED FAIL
    curr_cases = [
        EvalCaseResult(
            test_case_id="TC-001",
            expected=ClassificationOutput(category="billing", summary="Charge error"),
            actual=ClassificationOutput(category="general", summary="Charge error"),
            category_match=False,
            summary_score=3.0,
            latency_ms=150.0,
            status="FAIL",
        ),
        EvalCaseResult(
            test_case_id="TC-002",
            expected=ClassificationOutput(category="technical", summary="App crash"),
            actual=ClassificationOutput(category="technical", summary="App crash"),
            category_match=True,
            summary_score=5.0,
            latency_ms=110.0,
            status="PASS",
        ),
        EvalCaseResult(
            test_case_id="TC-003",
            expected=ClassificationOutput(category="account", summary="Password reset"),
            actual=ClassificationOutput(category="account", summary="Password reset"),
            category_match=True,
            summary_score=5.0,
            latency_ms=95.0,
            status="PASS",
        ),
        EvalCaseResult(
            test_case_id="TC-004",
            expected=ClassificationOutput(category="general", summary="Feature request"),
            actual=ClassificationOutput(category="technical", summary="Feature request"),
            category_match=False,
            summary_score=3.0,
            latency_ms=105.0,
            status="FAIL",
        ),
    ]

    current_run = RunResult(
        manifest=RunManifest(
            run_id="RUN-CURR",
            prompt_version="v2.0",
            dataset_version="v1.0",
            model_provider="mock",
            model_name="qwen2.5:3b",
            execution_mode="mock",
            started_at=datetime.now(timezone.utc),
        ),
        total_cases=4,
        passed_cases=2,
        failed_cases=2,
        category_accuracy=0.50,
        average_summary_score=4.0,
        average_latency_ms=115.0,
        p50_latency_ms=107.5,
        p95_latency_ms=150.0,
        average_input_tokens=18.0,
        average_output_tokens=12.0,
        cases=curr_cases,
    )

    return baseline, current_run


def test_compare_runs_identifies_regressions_and_improvements():
    baseline, current_run = build_fixture_data()
    diff = compare_runs(current_run, baseline)

    assert isinstance(diff, ComparisonResult)
    assert diff.change_type == "PROMPT_CHANGE"

    # 1. Regressions
    assert len(diff.regressions) == 1
    regression = diff.regressions[0]
    assert regression.test_case_id == "TC-001"
    assert regression.diff_type == "REGRESSION"
    assert regression.baseline_status == "PASS"
    assert regression.current_status == "FAIL"
    assert regression.baseline_category == "billing"
    assert regression.current_category == "general"
    assert regression.expected_category == "billing"

    # 2. Improvements
    assert len(diff.improvements) == 1
    improvement = diff.improvements[0]
    assert improvement.test_case_id == "TC-002"
    assert improvement.diff_type == "IMPROVEMENT"
    assert improvement.baseline_status == "FAIL"
    assert improvement.current_status == "PASS"

    # 3. Unchanged
    assert diff.unchanged_passes == ["TC-003"]
    assert diff.unchanged_fails == ["TC-004"]


def test_compare_runs_calculates_metric_deltas():
    baseline, current_run = build_fixture_data()
    diff = compare_runs(current_run, baseline)

    # Accuracy was 0.50 in both
    assert diff.category_accuracy_delta == 0.0

    # Summary score was 4.25 in baseline, 4.0 in current -> -0.25
    assert diff.average_summary_score_delta == -0.25

    # P95 latency was 120.0 in baseline, 150.0 in current -> +30.0
    assert diff.p95_latency_ms_delta == 30.0

    # Avg latency was 105.0 in baseline, 115.0 in current -> +10.0
    assert diff.average_latency_ms_delta == 10.0

    # Token deltas
    assert diff.average_input_tokens_delta == 3.0   # 18.0 - 15.0
    assert diff.average_output_tokens_delta == 2.0  # 12.0 - 10.0