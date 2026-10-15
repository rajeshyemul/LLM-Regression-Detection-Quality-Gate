import json
from pathlib import Path
from pydantic import ValidationError

from app.evaluation.models import RunResult


def save_run(run: RunResult, runs_dir: str | Path = "results/runs") -> Path:
    """Serialize and save a complete RunResult to disk.

    Args:
        run: The RunResult instance to persist.
        runs_dir: Directory where run JSON files are stored.

    Returns:
        Path to the saved JSON file (results/runs/<run_id>.json).
    """
    directory = Path(runs_dir)
    directory.mkdir(parents=True, exist_ok=True)

    file_path = directory / f"{run.manifest.run_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(run.model_dump_json(indent=2))

    return file_path


def load_run(run_id: str, runs_dir: str | Path = "results/runs") -> RunResult:
    """Load and validate a RunResult from disk by its run ID.

    Args:
        run_id: The ID of the run (e.g. 'RUN-20BBBC84' or filename 'RUN-20BBBC84.json').
        runs_dir: Directory where run JSON files are stored.

    Returns:
        Validated RunResult instance.

    Raises:
        FileNotFoundError: If the run file does not exist.
        ValueError: If the file is not valid JSON or does not conform to RunResult.
    """
    directory = Path(runs_dir)
    clean_id = run_id[:-5] if run_id.endswith(".json") else run_id
    file_path = directory / f"{clean_id}.json"

    if not file_path.is_file():
        raise FileNotFoundError(f"Run file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        return RunResult.model_validate_json(content)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Corrupted or invalid RunResult in {file_path}: {exc}"
        ) from exc


def list_runs(runs_dir: str | Path = "results/runs") -> list[str]:
    """List all available run IDs stored in runs_dir.

    Args:
        runs_dir: Directory where run JSON files are stored.

    Returns:
        Sorted list of run IDs (without the .json suffix).
    """
    directory = Path(runs_dir)
    if not directory.exists():
        return []

    return sorted(
        p.stem for p in directory.glob("*.json") if p.is_file()
    )