from datetime import datetime, timezone
from pathlib import Path

from app.evaluation.models import (
    EvalCaseResult,
    RunManifest,
    RunResult,
)
from app.feature.classifier import ClassificationOutput
from app.regression.comparator import CaseDiff, ComparisonResult
from app.regression.quality_gate import QualityGateResult
from app.reporting.html_report import generate_html_report


def make_test_run(run_id: str = "RUN-RPT-001") -> RunResult:
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
            summary_score=4.5,
            latency_ms=120.0,
            input_tokens=15,
            output_tokens=10,
            status="PASS",
        ),
        EvalCaseResult(
            test_case_id="TC-002",
            expected=ClassificationOutput(category="technical", summary="Login failure"),
            actual=ClassificationOutput(category="general", summary="Login failure"),
            category_match=False,
            summary_score=3.0,
            latency_ms=210.0,
            input_tokens=20,
            output_tokens=15,
            status="FAIL",
        ),
    ]
    return RunResult(
        manifest=manifest,
        total_cases=2,
        passed_cases=1,
        failed_cases=1,
        category_accuracy=0.5,
        average_summary_score=3.75,
        average_latency_ms=165.0,
        p50_latency_ms=165.0,
        p95_latency_ms=210.0,
        average_input_tokens=17.5,
        average_output_tokens=12.5,
        cases=cases,
    )


def test_generate_html_report_standalone(tmp_path: Path):
    run = make_test_run("RUN-STANDALONE")
    report_path = generate_html_report(run, reports_dir=tmp_path)

    assert report_path.is_file()
    assert report_path.name == "RUN-STANDALONE.html"

    html = report_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "RUN-STANDALONE" in html
    assert "50.0%" in html
    assert "TC-001" in html
    assert "TC-002" in html
    assert "No baseline reference provided" in html


def test_generate_html_report_with_comparison_and_gate(tmp_path: Path):
    run = make_test_run("RUN-FULL")
    regression_diff = CaseDiff(
        test_case_id="TC-002",
        diff_type="REGRESSION",
        baseline_status="PASS",
        current_status="FAIL",
        expected_category="technical",
        baseline_category="technical",
        current_category="general",
        reason="Flipped from PASS to FAIL",
    )
    comparison = ComparisonResult(
        baseline_id="BASE-ORIGINAL",
        run_id="RUN-FULL",
        change_type="PROMPT_CHANGE",
        attribution_message="Attributed to prompt v2 update",
        category_accuracy_delta=-0.05,
        average_summary_score_delta=-0.3,
        p95_latency_ms_delta=40.0,
        average_latency_ms_delta=20.0,
        average_input_tokens_delta=5.0,
        average_output_tokens_delta=2.0,
        regressions=[regression_diff],
        improvements=[],
        unchanged_passes=["TC-001"],
        unchanged_fails=[],
        baseline_metrics={"category_accuracy": 0.55, "p95_latency_ms": 170.0},
        current_metrics={"category_accuracy": 0.50, "p95_latency_ms": 210.0},
    )
    gate = QualityGateResult(
        status="WARNING",
        reasons=["WARNING: Category accuracy dropped by -5.00%"],
        regressions=["TC-002"],
        improvements=[],
    )

    report_path = generate_html_report(
        run=run,
        comparison=comparison,
        gate=gate,
        reports_dir=tmp_path,
    )

    html = report_path.read_text(encoding="utf-8")
    assert "BASE-ORIGINAL" in html
    assert "PROMPT_CHANGE" in html
    assert "Attributed to prompt v2 update" in html
    assert "badge-WARNING" in html
    assert "Regressions (Passed in Baseline, Failed in Current Run)" in html
    assert "TC-002" in html


def test_generate_html_report_custom_path(tmp_path: Path):
    run = make_test_run("RUN-CUSTOM")
    custom_file = tmp_path / "my_custom_folder" / "custom_report.html"

    result_path = generate_html_report(run, output_path=custom_file)
    assert result_path == custom_file
    assert custom_file.is_file()


def test_generate_html_report_escapes_html(tmp_path: Path):
    run = make_test_run("RUN-XSS")
    # Simulate a potentially dangerous string in test case expected category
    run.cases[0].expected = ClassificationOutput(
        category="<script>alert('pwned')</script>",  # type: ignore[arg-type]
        summary="Charge issue",
    )

    report_path = generate_html_report(run, reports_dir=tmp_path)
    html = report_path.read_text(encoding="utf-8")

    # Jinja2 autoescape must escape raw HTML tags
    assert "<script>" not in html
    assert "&lt;script&gt;" in html