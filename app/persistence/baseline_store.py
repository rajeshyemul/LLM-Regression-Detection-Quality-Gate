import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from pydantic import ValidationError

from app.evaluation.models import Baseline, RunResult


def promote_run_to_baseline(
    run: RunResult,
    baseline_id: str | None = None,
    baselines_dir: str | Path = "baselines",
) -> Baseline:
    """Convert an approved RunResult into a Baseline and persist as current_baseline.json.

    Args:
        run: The approved RunResult to promote.
        baseline_id: Optional custom baseline ID (defaults to BASE-<hex>).
        baselines_dir: Directory where baseline JSON files are stored.

    Returns:
        The newly created and saved Baseline instance.
    """
    directory = Path(baselines_dir)
    directory.mkdir(parents=True, exist_ok=True)

    b_id = baseline_id or f"BASE-{uuid.uuid4().hex[:8].upper()}"

    metrics = {
        "total_cases": run.total_cases,
        "passed_cases": run.passed_cases,
        "failed_cases": run.failed_cases,
        "category_accuracy": run.category_accuracy,
        "average_summary_score": run.average_summary_score,
        "average_latency_ms": run.average_latency_ms,
        "p50_latency_ms": run.p50_latency_ms,
        "p95_latency_ms": run.p95_latency_ms,
        "average_input_tokens": run.average_input_tokens,
        "average_output_tokens": run.average_output_tokens,
    }

    baseline = Baseline(
        baseline_id=b_id,
        run_id=run.manifest.run_id,
        prompt_version=run.manifest.prompt_version,
        dataset_version=run.manifest.dataset_version,
        model_provider=run.manifest.model_provider,
        model_name=run.manifest.model_name,
        judge_provider=run.manifest.judge_provider,
        judge_model=run.manifest.judge_model,
        metrics=metrics,
        case_results=list(run.cases),
        created_at=datetime.now(timezone.utc),
    )

    current_baseline_path = directory / "current_baseline.json"
    with open(current_baseline_path, "w", encoding="utf-8") as f:
        f.write(baseline.model_dump_json(indent=2))

    return baseline


def load_baseline(baseline_path: str | Path = "baselines/current_baseline.json") -> Baseline:
    """Load and validate the current baseline from disk.

    Args:
        baseline_path: Path to the baseline JSON file.

    Returns:
        Validated Baseline instance.

    Raises:
        FileNotFoundError: If the baseline file does not exist.
        ValueError: If the file is corrupted or schema is invalid.
    """
    path = Path(baseline_path)
    if not path.is_file():
        raise FileNotFoundError(f"Baseline file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        return Baseline.model_validate_json(content)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Corrupted or invalid Baseline in {path}: {exc}"
        ) from exc


def has_baseline(baseline_path: str | Path = "baselines/current_baseline.json") -> bool:
    """Check whether a baseline file exists on disk."""
    return Path(baseline_path).is_file()
