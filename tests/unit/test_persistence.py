from datetime import datetime, timezone
from pathlib import Path
import pytest

from app.evaluation.models import (
    EvalCaseResult,
    RunManifest,
    RunResult,
)
from app.feature.classifier import ClassificationOutput
from app.persistence.baseline_store import (
    has_baseline,
    load_baseline,
    promote_run_to_baseline,
)
from app.persistence.run_store import (
    list_runs,
    load_run,
    save_run,
)


def create_sample_run(run_id: str = "RUN-TEST001", accuracy: float = 0.9) -> RunResult:
    manifest = RunManifest(
        run_id=run_id,
        prompt_version="v1.0",
        dataset_version="v1.0",
        model_provider="mock",
        model_name="qwen2.5:3b",
        execution_mode="mock",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    cases = [
        EvalCaseResult(
            test_case_id="TC-001",
            expected=ClassificationOutput(category="billing", summary="Charge issue"),
            actual=ClassificationOutput(category="billing", summary="Charge issue"),
            category_match=True,
            summary_score=5.0,
            latency_ms=120.0,
            input_tokens=15,
            output_tokens=10,
            status="PASS",
        )
    ]
    return RunResult(
        manifest=manifest,
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        category_accuracy=accuracy,
        average_summary_score=5.0,
        average_latency_ms=120.0,
        p50_latency_ms=120.0,
        p95_latency_ms=120.0,
        average_input_tokens=15.0,
        average_output_tokens=10.0,
        cases=cases,
    )


# ── Run Store Tests ─────────────────────────────────────────────────────────

def test_save_and_load_run_roundtrip(tmp_path: Path):
    run = create_sample_run("RUN-A1")
    file_path = save_run(run, runs_dir=tmp_path)
    assert file_path.is_file()
    assert file_path.name == "RUN-A1.json"

    loaded = load_run("RUN-A1", runs_dir=tmp_path)
    assert loaded.manifest.run_id == "RUN-A1"
    assert loaded.category_accuracy == 0.9
    assert len(loaded.cases) == 1
    assert loaded.cases[0].test_case_id == "TC-001"
    assert loaded.cases[0].expected.category == "billing"


def test_load_run_supports_json_extension(tmp_path: Path):
    run = create_sample_run("RUN-EXT")
    save_run(run, runs_dir=tmp_path)

    loaded = load_run("RUN-EXT.json", runs_dir=tmp_path)
    assert loaded.manifest.run_id == "RUN-EXT"


def test_load_run_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Run file not found"):
        load_run("RUN-NONEXISTENT", runs_dir=tmp_path)


def test_load_run_corrupted_json_raises(tmp_path: Path):
    bad_file = tmp_path / "RUN-CORRUPT.json"
    bad_file.write_text("{ unclosed json", encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupted or invalid RunResult"):
        load_run("RUN-CORRUPT", runs_dir=tmp_path)


def test_list_runs(tmp_path: Path):
    assert list_runs(tmp_path) == []

    save_run(create_sample_run("RUN-002"), runs_dir=tmp_path)
    save_run(create_sample_run("RUN-001"), runs_dir=tmp_path)

    assert list_runs(tmp_path) == ["RUN-001", "RUN-002"]


# ── Baseline Store Tests ────────────────────────────────────────────────────

def test_promote_run_to_baseline(tmp_path: Path):
    run = create_sample_run("RUN-PROMOTED", accuracy=0.95)
    baseline = promote_run_to_baseline(
        run=run,
        baseline_id="BASE-TEST-1",
        baselines_dir=tmp_path,
    )

    assert baseline.baseline_id == "BASE-TEST-1"
    assert baseline.run_id == "RUN-PROMOTED"
    assert baseline.metrics["category_accuracy"] == 0.95
    assert len(baseline.case_results) == 1

    # Verify saved file
    baseline_file = tmp_path / "current_baseline.json"
    assert baseline_file.is_file()

    loaded = load_baseline(baseline_file)
    assert loaded.baseline_id == "BASE-TEST-1"
    assert loaded.run_id == "RUN-PROMOTED"


def test_has_baseline(tmp_path: Path):
    baseline_file = tmp_path / "current_baseline.json"
    assert not has_baseline(baseline_file)

    run = create_sample_run("RUN-CHECK")
    promote_run_to_baseline(run, baselines_dir=tmp_path)

    assert has_baseline(baseline_file)


def test_load_baseline_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Baseline file not found"):
        load_baseline(tmp_path / "missing_baseline.json")


def test_load_baseline_corrupted_raises(tmp_path: Path):
    bad_file = tmp_path / "corrupted_baseline.json"
    bad_file.write_text("{ bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupted or invalid Baseline"):
        load_baseline(bad_file)


def test_promotion_overwrites_previous_baseline(tmp_path: Path):
    run1 = create_sample_run("RUN-V1", accuracy=0.85)
    run2 = create_sample_run("RUN-V2", accuracy=0.92)

    promote_run_to_baseline(run1, baseline_id="BASE-1", baselines_dir=tmp_path)
    b1 = load_baseline(tmp_path / "current_baseline.json")
    assert b1.baseline_id == "BASE-1"
    assert b1.run_id == "RUN-V1"

    # Second promotion replaces the current baseline
    promote_run_to_baseline(run2, baseline_id="BASE-2", baselines_dir=tmp_path)
    b2 = load_baseline(tmp_path / "current_baseline.json")
    assert b2.baseline_id == "BASE-2"
    assert b2.run_id == "RUN-V2"
    assert b2.metrics["category_accuracy"] == 0.92