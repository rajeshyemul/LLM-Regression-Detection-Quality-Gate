from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.evaluation.models import RunResult
from app.regression.comparator import ComparisonResult
from app.regression.quality_gate import QualityGateResult


def get_template_environment() -> Environment:
    """Initialize Jinja2 template environment for reporting."""
    template_dir = Path(__file__).parent / "templates"
    return Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml", "jinja2"]),
    )


def generate_html_report(
    run: RunResult,
    comparison: ComparisonResult | None = None,
    gate: QualityGateResult | None = None,
    output_path: str | Path | None = None,
    reports_dir: str | Path = "results/reports",
) -> Path:
    """Generate a self-contained HTML evaluation report.

    Args:
        run: The completed evaluation RunResult.
        comparison: Optional ComparisonResult vs baseline.
        gate: Optional QualityGateResult with pass/fail decision.
        output_path: Optional explicit file path to write the report.
        reports_dir: Directory where reports are saved if output_path is None.

    Returns:
        Path to the generated HTML file.
    """
    env = get_template_environment()
    template = env.get_template("report.html.jinja2")

    rendered_html = template.render(
        run=run,
        comparison=comparison,
        gate=gate,
    )

    if output_path is not None:
        target_path = Path(output_path)
    else:
        target_dir = Path(reports_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{run.manifest.run_id}.html"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    return target_path