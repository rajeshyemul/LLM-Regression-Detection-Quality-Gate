from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from app.feature.classifier import ClassificationOutput
from datasets.schemas.golden_case import GoldenTestCase
from app.llm.base import LLMRequest, Message, LLMProvider
from app.llm.mock import MockProvider
from app.llm.replay import ReplayProvider
from app.llm.ollama import OllamaProvider
from app.evaluation.models import (
    EvalCaseResult,
    RunManifest,
    RunResult,
    Baseline,
)
from app.regression.quality_gate import QualityGateResult


def test_classification_output_valid():
    output = ClassificationOutput(
        category="billing",
        summary="Customer was double charged.",
    )
    assert output.category == "billing"
    assert output.summary == "Customer was double charged."


def test_classification_output_invalid_category():
    with pytest.raises(ValidationError):
        ClassificationOutput(
            category="sales",  # Not in Literal["billing", "technical", "account", "general"]
            summary="Invalid category test",
        )


def test_golden_test_case_contract():
    case = GoldenTestCase(
        id="TC-001",
        input="I was charged twice for my subscription.",
        expected_category="billing",
        expected_summary="Customer reports duplicate charge on subscription.",
        expected_difficulty="easy",
        notes="Standard billing test",
    )
    assert case.id == "TC-001"
    assert case.expected_category == "billing"


def test_golden_test_case_invalid_difficulty():
    with pytest.raises(ValidationError):
        GoldenTestCase(
            id="TC-002",
            input="Test email",
            expected_category="technical",
            expected_summary="Summary",
            expected_difficulty="impossible",  # Invalid difficulty
        )


@pytest.mark.asyncio
async def test_mock_provider_conforms_to_llm_provider():
    provider = MockProvider()
    assert isinstance(provider, LLMProvider)

    request = LLMRequest(
        messages=[
            Message(role="system", content="You are a classifier."),
            Message(role="user", content="I need a refund."),
        ],
        model="mock-model",
        temperature=0.0,
    )
    response = await provider.generate(request)

    assert response.provider == "mock"
    assert response.model == "mock-model"
    assert response.latency_ms == 1.0
    assert response.input_tokens == 10
    assert response.output_tokens == 10

    # Verify content parses as ClassificationOutput
    output = ClassificationOutput.model_validate_json(response.content)
    assert output.category == "billing"


@pytest.mark.asyncio
async def test_replay_provider_raises_when_fixture_missing():
    provider = ReplayProvider()
    assert isinstance(provider, LLMProvider)

    request = LLMRequest(
        messages=[Message(role="user", content="Hi")],
        model="replay-model",
    )
    with pytest.raises(FileNotFoundError):
        await provider.generate(request)


def test_ollama_provider_conforms_to_llm_provider():
    provider = OllamaProvider(host="http://localhost:11434", timeout=30.0)
    assert isinstance(provider, LLMProvider)
    assert provider.host == "http://localhost:11434"
    assert provider.timeout == 30.0


def test_eval_case_result_and_run_result():
    expected = ClassificationOutput(category="billing", summary="Duplicate charge")
    actual = ClassificationOutput(category="billing", summary="Duplicate charge reported")

    case_res = EvalCaseResult(
        test_case_id="TC-001",
        expected=expected,
        actual=actual,
        category_match=True,
        summary_score=5.0,
        summary_reason="Accurate and concise",
        latency_ms=120.5,
        input_tokens=15,
        output_tokens=12,
        status="PASS",
    )

    manifest = RunManifest(
        run_id="RUN-001",
        prompt_version="v1.0",
        dataset_version="v1.0",
        model_provider="mock",
        model_name="mock-model",
        execution_mode="mock",
        started_at=datetime.now(timezone.utc),
    )

    run_result = RunResult(
        manifest=manifest,
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        category_accuracy=1.0,
        average_summary_score=5.0,
        average_latency_ms=120.5,
        p50_latency_ms=120.5,
        p95_latency_ms=120.5,
        average_input_tokens=15.0,
        average_output_tokens=12.0,
        cases=[case_res],
    )

    assert run_result.total_cases == 1
    assert run_result.category_accuracy == 1.0


def test_baseline_and_quality_gate_result():
    expected = ClassificationOutput(category="account", summary="Password reset")
    case_res = EvalCaseResult(
        test_case_id="TC-002",
        expected=expected,
        actual=expected,
        category_match=True,
        summary_score=5.0,
        latency_ms=90.0,
        status="PASS",
    )

    baseline = Baseline(
        baseline_id="BL-001",
        run_id="RUN-001",
        prompt_version="v1.0",
        dataset_version="v1.0",
        model_provider="mock",
        model_name="mock-model",
        metrics={"category_accuracy": 1.0, "average_summary_score": 5.0},
        case_results=[case_res],
        created_at=datetime.now(timezone.utc),
    )

    gate = QualityGateResult(
        status="PASS",
        reasons=["All thresholds met."],
        regressions=[],
        improvements=["TC-002 passed."],
        metric_deltas={"category_accuracy_delta": 0.0},
    )

    assert baseline.baseline_id == "BL-001"
    assert gate.status == "PASS"
