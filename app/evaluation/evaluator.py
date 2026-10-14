from app.feature.classifier import (
    ClassificationError,
    ClassificationOutput,
    classify_email,
)
from app.feature.prompt_loader import PromptConfig
from app.evaluation.judge import JudgeError, evaluate_summary_relevance
from app.evaluation.models import EvalCaseResult
from app.llm.base import LLMProvider, LLMResponse
from datasets.schemas.golden_case import GoldenTestCase


async def evaluate_case(
    test_case: GoldenTestCase,
    provider: LLMProvider,
    prompt_config: PromptConfig,
    judge_provider: LLMProvider | None = None,
    judge_model: str | None = None,
) -> EvalCaseResult:
    """Evaluate a single golden test case end-to-end.

    Runs the classifier feature, checks functional correctness,
    optionally invokes the LLM-as-judge for semantic scoring,
    and packages all evidence into an EvalCaseResult.

    Args:
        test_case: The golden test case containing input and expected output.
        provider: LLMProvider for the feature under test (classifier).
        prompt_config: Versioned prompt configuration.
        judge_provider: Optional separate LLMProvider for the judge (can be same as provider).
        judge_model: Model name for the judge. Required if judge_provider is given.

    Returns:
        Complete EvalCaseResult evidence object.
    """
    expected = ClassificationOutput(
        category=test_case.expected_category,
        summary=test_case.expected_summary,
    )

    # 1. Run the Feature Under Test (Classifier)
    actual: ClassificationOutput | None = None
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None

    try:
        # We need to capture the raw LLMResponse for operational metrics.
        # classify_email returns ClassificationOutput, so we replicate the
        # request construction to also capture the raw response.
        from app.llm.base import LLMRequest, Message
        import time

        messages: list[Message] = [
            Message(role="system", content=prompt_config.system_prompt)
        ]
        for example in prompt_config.few_shot_examples:
            messages.append(Message(role="user", content=example.input))
            messages.append(Message(role="assistant", content=example.output))
        messages.append(
            Message(
                role="user",
                content=f"Customer Email:\n\"\"\"\n{test_case.input.strip()}\n\"\"\"",
            )
        )

        request = LLMRequest(
            messages=messages,
            model=prompt_config.model.name,
            temperature=prompt_config.model.temperature,
            response_schema=ClassificationOutput.model_json_schema(),
        )

        start = time.perf_counter()
        response: LLMResponse = await provider.generate(request)
        latency_ms = (time.perf_counter() - start) * 1000.0

        input_tokens = response.input_tokens
        output_tokens = response.output_tokens

        actual = ClassificationOutput.model_validate_json(response.content)

    except ClassificationError as exc:
        error = str(exc)
    except Exception as exc:
        error = f"Unexpected error during classification: {exc}"

    # 2. Functional Dimension: Category Match
    category_match = actual is not None and actual.category == expected.category

    # 3. Semantic Dimension: Summary Judge (optional)
    summary_score: float | None = None
    summary_reason: str | None = None

    if actual is not None and judge_provider is not None and judge_model is not None:
        try:
            judge_result = await evaluate_summary_relevance(
                expected_summary=expected.summary,
                actual_summary=actual.summary,
                provider=judge_provider,
                judge_model=judge_model,
                customer_email=test_case.input,
            )
            summary_score = float(judge_result.score)
            summary_reason = judge_result.reason
        except (JudgeError, Exception) as exc:
            summary_reason = f"Judge evaluation failed: {exc}"

    # 4. Determine Status
    if error is not None:
        status = "ERROR"
    elif not category_match:
        status = "FAIL"
    else:
        status = "PASS"

    return EvalCaseResult(
        test_case_id=test_case.id,
        expected=expected,
        actual=actual,
        category_match=category_match,
        summary_score=summary_score,
        summary_reason=summary_reason,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        status=status,
        error=error,
    )