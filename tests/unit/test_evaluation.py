import pytest

from app.evaluation.evaluator import evaluate_case
from app.evaluation.judge import (
    JudgeError,
    JudgeScore,
    evaluate_summary_relevance,
)
from app.evaluation.metrics import (
    calculate_percentile,
    compute_run_metrics,
)
from app.evaluation.models import EvalCaseResult
from app.evaluation.runner import run_evaluation
from app.feature.classifier import ClassificationOutput
from app.feature.prompt_loader import load_prompt
from app.llm.mock import MockProvider
from datasets.schemas.golden_case import GoldenDataset, GoldenTestCase


# ── Metrics Tests ───────────────────────────────────────────────────────────

def test_calculate_percentile_edge_cases():
    assert calculate_percentile([], 50.0) == 0.0
    assert calculate_percentile([42.0], 95.0) == 42.0
    assert calculate_percentile([10.0, 20.0], 50.0) == 15.0
    assert calculate_percentile([10.0, 20.0, 30.0, 40.0, 50.0], 90.0) == 46.0


def test_compute_run_metrics_empty():
    metrics = compute_run_metrics([])
    assert metrics["total_cases"] == 0
    assert metrics["category_accuracy"] == 0.0
    assert metrics["average_summary_score"] is None


def test_compute_run_metrics_aggregation():
    cases = [
        EvalCaseResult(
            test_case_id="TC-001",
            expected=ClassificationOutput(category="billing", summary="s1"),
            actual=ClassificationOutput(category="billing", summary="s1"),
            category_match=True,
            summary_score=5.0,
            latency_ms=100.0,
            input_tokens=20,
            output_tokens=10,
            status="PASS",
        ),
        EvalCaseResult(
            test_case_id="TC-002",
            expected=ClassificationOutput(category="technical", summary="s2"),
            actual=ClassificationOutput(category="general", summary="s2"),
            category_match=False,
            summary_score=3.0,
            latency_ms=300.0,
            input_tokens=40,
            output_tokens=20,
            status="FAIL",
        ),
        EvalCaseResult(
            test_case_id="TC-003",
            expected=ClassificationOutput(category="account", summary="s3"),
            actual=None,
            category_match=False,
            summary_score=None,
            latency_ms=50.0,
            input_tokens=10,
            output_tokens=0,
            status="ERROR",
            error="Model timeout",
        ),
    ]

    metrics = compute_run_metrics(cases)
    assert metrics["total_cases"] == 3
    assert metrics["passed_cases"] == 1
    assert metrics["failed_cases"] == 2
    assert metrics["category_accuracy"] == round(1 / 3, 4)
    assert metrics["average_summary_score"] == 4.0  # (5.0 + 3.0) / 2
    assert metrics["average_latency_ms"] == 150.0   # (100 + 300 + 50) / 3
    assert metrics["p50_latency_ms"] == 100.0


# ── Judge Tests ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_judge_scoring_success():
    provider = MockProvider(
        response='{"score": 4, "reason": "Accurate summary but omits invoice number."}'
    )
    result = await evaluate_summary_relevance(
        expected_summary="Duplicate charge on invoice #1089.",
        actual_summary="Customer was charged twice.",
        provider=provider,
        judge_model="mock-judge",
    )
    assert isinstance(result, JudgeScore)
    assert result.score == 4
    assert "omits invoice number" in result.reason


@pytest.mark.asyncio
async def test_judge_scoring_invalid_json_raises():
    provider = MockProvider(response="I give this a 4 out of 5.")
    with pytest.raises(JudgeError):
        await evaluate_summary_relevance(
            expected_summary="Expected summary.",
            actual_summary="Actual summary.",
            provider=provider,
            judge_model="mock-judge",
        )


# ── Case Evaluator Tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluator_case_pass():
    prompt = load_prompt("prompts/v1.yaml")
    provider = MockProvider(
        response='{"category":"billing","summary":"Customer requested a refund."}'
    )
    test_case = GoldenTestCase(
        id="TC-001",
        input="Please refund my order.",
        expected_category="billing",
        expected_summary="Customer requested a refund.",
        expected_difficulty="easy",
    )

    result = await evaluate_case(test_case, provider, prompt)
    assert result.status == "PASS"
    assert result.category_match is True
    assert result.error is None


@pytest.mark.asyncio
async def test_evaluator_case_fail():
    prompt = load_prompt("prompts/v1.yaml")
    provider = MockProvider(
        response='{"category":"general","summary":"Customer requested a refund."}'
    )
    test_case = GoldenTestCase(
        id="TC-001",
        input="Please refund my order.",
        expected_category="billing",
        expected_summary="Customer requested a refund.",
        expected_difficulty="easy",
    )

    result = await evaluate_case(test_case, provider, prompt)
    assert result.status == "FAIL"
    assert result.category_match is False


@pytest.mark.asyncio
async def test_evaluator_case_error():
    prompt = load_prompt("prompts/v1.yaml")
    provider = MockProvider(response="Not valid JSON at all")
    test_case = GoldenTestCase(
        id="TC-001",
        input="Please refund my order.",
        expected_category="billing",
        expected_summary="Customer requested a refund.",
        expected_difficulty="easy",
    )

    result = await evaluate_case(test_case, provider, prompt)
    assert result.status == "ERROR"
    assert result.actual is None
    assert result.error is not None


# ── Runner Tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_runner_executes_dataset():
    prompt = load_prompt("prompts/v1.yaml")
    provider = MockProvider(
        response='{"category":"billing","summary":"Refund requested."}'
    )
    dataset = GoldenDataset(
        version="v1.0-test",
        description="Mini test dataset",
        cases=[
            GoldenTestCase(
                id="TC-001",
                input="Charge issue",
                expected_category="billing",
                expected_summary="Refund requested.",
                expected_difficulty="easy",
            ),
            GoldenTestCase(
                id="TC-002",
                input="Login failed",
                expected_category="account",
                expected_summary="Login failed.",
                expected_difficulty="easy",
            ),
        ],
    )

    result = await run_evaluation(
        dataset=dataset,
        prompt_config=prompt,
        provider=provider,
        execution_mode="mock",
        concurrency=2,
    )

    assert result.total_cases == 2
    assert result.passed_cases == 1
    assert result.failed_cases == 1
    assert result.category_accuracy == 0.5
    assert result.manifest.execution_mode == "mock"
    assert result.manifest.completed_at is not None