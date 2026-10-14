import asyncio
import uuid
from datetime import datetime, timezone
from typing import Literal

from app.evaluation.evaluator import evaluate_case
from app.evaluation.metrics import compute_run_metrics
from app.evaluation.models import EvalCaseResult, RunManifest, RunResult
from app.feature.prompt_loader import PromptConfig
from app.llm.base import LLMProvider
from datasets.schemas.golden_case import GoldenDataset


async def run_evaluation(
    dataset: GoldenDataset,
    prompt_config: PromptConfig,
    provider: LLMProvider,
    execution_mode: Literal["real", "mock", "replay"],
    baseline_id: str | None = None,
    judge_provider: LLMProvider | None = None,
    judge_model: str | None = None,
    concurrency: int = 5,
) -> RunResult:
    """Execute a full evaluation run across all golden test cases.

    Args:
        dataset: The loaded GoldenDataset to evaluate against.
        prompt_config: Versioned PromptConfig to evaluate.
        provider: LLMProvider instance for the feature under test.
        execution_mode: One of 'real', 'mock', or 'replay'.
        baseline_id: Optional ID of the current baseline to compare against.
        judge_provider: Optional LLMProvider for semantic judging.
        judge_model: Model name to use for judging.
        concurrency: Maximum number of test cases to run in parallel.

    Returns:
        Complete RunResult with manifest, per-case evidence, and aggregate metrics.
    """
    run_id = f"RUN-{uuid.uuid4().hex[:8].upper()}"
    started_at = datetime.now(timezone.utc)

    manifest = RunManifest(
        run_id=run_id,
        prompt_version=prompt_config.version,
        dataset_version=dataset.version,
        model_provider=execution_mode,
        model_name=prompt_config.model.name,
        judge_provider=execution_mode if judge_provider is not None else None,
        judge_model=judge_model,
        execution_mode=execution_mode,
        baseline_id=baseline_id,
        started_at=started_at,
    )

    # Semaphore limits concurrency so we don't flood Ollama with 30 simultaneous requests
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_evaluate(test_case):
        async with semaphore:
            return await evaluate_case(
                test_case=test_case,
                provider=provider,
                prompt_config=prompt_config,
                judge_provider=judge_provider,
                judge_model=judge_model,
            )

    cases: list[EvalCaseResult] = await asyncio.gather(
        *[bounded_evaluate(tc) for tc in dataset.cases]
    )

    completed_at = datetime.now(timezone.utc)
    manifest = manifest.model_copy(update={"completed_at": completed_at})

    metrics = compute_run_metrics(cases)

    return RunResult(
        manifest=manifest,
        cases=list(cases),
        **metrics,
    )