#!/usr/bin/env python3
"""CLI entrypoint for the LLM Quality & Regression Evaluation Pipeline."""

import argparse
import asyncio
import sys

from rich.console import Console
from rich.table import Table
from rich import box

from app.evaluation.runner import run_evaluation
from app.feature.prompt_loader import load_prompt
from app.llm.factory import get_provider
from app.config.settings import settings
from datasets.loader import load_golden_dataset
from app.persistence.run_store import save_run
from app.persistence.baseline_store import has_baseline, load_baseline
from app.regression.comparator import compare_runs
from app.regression.quality_gate import evaluate_quality_gate
from app.reporting.html_report import generate_html_report

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLM quality evaluation against the golden dataset."
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "replay", "real"],
        default=settings.execution_mode,
        help="Execution mode: mock (offline), replay (fixtures), real (Ollama).",
    )
    parser.add_argument(
        "--prompt",
        default="prompts/v1.yaml",
        help="Path to the prompt YAML file to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        default="datasets/golden/v1.json",
        help="Path to the golden dataset JSON file.",
    )
    parser.add_argument(
        "--baseline",
        default="baselines/current_baseline.json",
        help="Path to the current approved baseline JSON file.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent evaluation requests.",
    )
    parser.add_argument(
        "--reports-dir",
        default="results/reports",
        help="Directory where HTML reports are saved.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    console.rule("[bold blue]LLM Quality Evaluation[/bold blue]")
    console.print(f"  Mode:     [bold]{args.mode}[/bold]")
    console.print(f"  Prompt:   [bold]{args.prompt}[/bold]")
    console.print(f"  Dataset:  [bold]{args.dataset}[/bold]")
    console.print()

    # Load configuration
    prompt_config = load_prompt(args.prompt)
    dataset = load_golden_dataset(args.dataset)

    console.print(
        f"  Loaded prompt [cyan]{prompt_config.version}[/cyan] "
        f"targeting model [cyan]{prompt_config.model.name}[/cyan]"
    )
    console.print(
        f"  Loaded dataset [cyan]{dataset.version}[/cyan] "
        f"with [cyan]{len(dataset.cases)}[/cyan] cases"
    )
    console.print()

    # Build providers
    provider = get_provider(
        args.mode,
        host=settings.ollama_host if args.mode == "real" else None,
        fixture_dir=settings.replay_fixture_dir if args.mode == "replay" else None,
    )

    with console.status("[bold green]Running evaluation...[/bold green]", spinner="dots"):
        result = await run_evaluation(
            dataset=dataset,
            prompt_config=prompt_config,
            provider=provider,
            execution_mode=args.mode,
            concurrency=args.concurrency,
        )
        
    saved_path = save_run(result)
    console.print(f"  Run evidence saved to: [dim]{saved_path}[/dim]")
    console.print()

    # ── Summary Scorecard ──────────────────────────────────────────
    console.rule("[bold]Evaluation Summary[/bold]")

    accuracy_color = (
        "green" if result.category_accuracy >= 0.9
        else "yellow" if result.category_accuracy >= 0.8
        else "red"
    )

    console.print(f"  Run ID:             [dim]{result.manifest.run_id}[/dim]")
    console.print(f"  Total Cases:        {result.total_cases}")
    console.print(f"  Passed:             [green]{result.passed_cases}[/green]")
    console.print(f"  Failed:             [red]{result.failed_cases}[/red]")
    console.print(
        f"  Category Accuracy:  [{accuracy_color}]{result.category_accuracy:.1%}[/{accuracy_color}]"
    )
    if result.average_summary_score is not None:
        console.print(f"  Avg Summary Score:  {result.average_summary_score:.2f} / 5.0")
    console.print(f"  Avg Latency:        {result.average_latency_ms:.1f} ms")
    console.print(f"  P95 Latency:        {result.p95_latency_ms:.1f} ms")
    console.print()

    # ── Per-Case Results Table ──────────────────────────────────────
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim",
    )
    table.add_column("ID", style="dim", width=8)
    table.add_column("Difficulty", width=12)
    table.add_column("Expected", width=12)
    table.add_column("Actual", width=12)
    table.add_column("Match", width=6, justify="center")
    table.add_column("Score", width=6, justify="center")
    table.add_column("Status", width=8, justify="center")

    for case in sorted(result.cases, key=lambda c: c.test_case_id):
        actual_cat = case.actual.category if case.actual else "—"
        match_icon = "✓" if case.category_match else "✗"
        match_style = "green" if case.category_match else "red"
        score_str = f"{case.summary_score:.1f}" if case.summary_score else "—"

        status_style = {
            "PASS": "green",
            "FAIL": "red",
            "ERROR": "bold red",
        }.get(case.status, "white")

        table.add_row(
            case.test_case_id,
            case.expected.category if hasattr(case, 'expected') else "—",
            case.expected.category,
            actual_cat,
            f"[{match_style}]{match_icon}[/{match_style}]",
            score_str,
            f"[{status_style}]{case.status}[/{status_style}]",
        )

    console.print(table)
    console.print()

    # ── Regression & Quality Gate Evaluation ───────────────────────
    console.rule("[bold]Quality Gate & Regression Analysis[/bold]")

    comparison = None
    if has_baseline(args.baseline):
        baseline = load_baseline(args.baseline)
        comparison = compare_runs(result, baseline)

        console.print(f"  Compared against Baseline: [cyan]{baseline.baseline_id}[/cyan] (Run: [dim]{baseline.run_id}[/dim])")
        console.print(f"  Change Attribution:         [bold]{comparison.change_type}[/bold]")
        console.print(f"  Details:                    {comparison.attribution_message}")
        console.print()

        # Deltas
        acc_d_str = f"{comparison.category_accuracy_delta:+.2%}"
        acc_d_color = "green" if comparison.category_accuracy_delta >= 0 else "red"
        console.print(f"  Accuracy Delta:             [{acc_d_color}]{acc_d_str}[/{acc_d_color}]")

        if comparison.average_summary_score_delta is not None:
            score_d_str = f"{comparison.average_summary_score_delta:+.2f} pts"
            score_d_color = "green" if comparison.average_summary_score_delta >= 0 else "red"
            console.print(f"  Summary Score Delta:        [{score_d_color}]{score_d_str}[/{score_d_color}]")

        lat_d_str = f"{comparison.p95_latency_ms_delta:+.1f} ms"
        lat_d_color = "green" if comparison.p95_latency_ms_delta <= 0 else "yellow"
        console.print(f"  P95 Latency Delta:          [{lat_d_color}]{lat_d_str}[/{lat_d_color}]")
        console.print(f"  Regressed Cases (to FAIL):  [red]{len(comparison.regressions)}[/red]")
        console.print(f"  Improved Cases (to PASS):   [green]{len(comparison.improvements)}[/green]")
        console.print()
    else:
        console.print(f"  [dim yellow]No baseline found at '{args.baseline}'. Running in standalone mode.[/dim yellow]")
        console.print()

    # Evaluate Quality Gate
    gate_result = evaluate_quality_gate(comparison)

    gate_styles = {
        "PASS": "bold green",
        "WARNING": "bold yellow",
        "CRITICAL": "bold red",
    }
    status_label = gate_result.status
    style = gate_styles.get(status_label, "bold white")

    console.print(f"  Decision: [{style}]{status_label}[/{style}]")
    for reason in gate_result.reasons:
        console.print(f"    • {reason}")
    console.print()

    # ── Generate HTML Report ───────────────────────────────────────
    report_path = generate_html_report(
        run=result,
        comparison=comparison,
        gate=gate_result,
        reports_dir=args.reports_dir,
    )
    console.print(f"  HTML Report: [bold cyan]file://{report_path.resolve()}[/bold cyan]")
    console.print()

    # ── Exit Code ──────────────────────────────────────────────────
    exit_codes = {
        "PASS": 0,
        "WARNING": 1,
        "CRITICAL": 2,
    }
    sys.exit(exit_codes.get(gate_result.status, 2))


if __name__ == "__main__":
    asyncio.run(main())