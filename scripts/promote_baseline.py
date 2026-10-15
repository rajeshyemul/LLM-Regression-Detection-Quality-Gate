#!/usr/bin/env python3
"""CLI tool to promote an approved evaluation run as the official baseline."""

import argparse
import sys
from rich.console import Console

from app.persistence.baseline_store import promote_run_to_baseline
from app.persistence.run_store import load_run

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Promote an approved RunResult to the official baseline."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="ID of the run to promote (e.g. 'RUN-20BBBC84').",
    )
    parser.add_argument(
        "--baseline-id",
        default=None,
        help="Optional custom baseline ID (e.g. 'BASE-V1-RELEASE').",
    )
    parser.add_argument(
        "--runs-dir",
        default="results/runs",
        help="Directory where runs are stored.",
    )
    parser.add_argument(
        "--baselines-dir",
        default="baselines",
        help="Directory where baselines are stored.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    console.rule("[bold cyan]Promote Baseline[/bold cyan]")
    console.print(f"  Target Run ID: [bold]{args.run_id}[/bold]")
    console.print()

    # 1. Load and validate run
    try:
        run = load_run(args.run_id, runs_dir=args.runs_dir)
    except FileNotFoundError:
        console.print(
            f"[bold red]Error: Run '{args.run_id}' not found in '{args.runs_dir}'.[/bold red]"
        )
        sys.exit(1)
    except ValueError as exc:
        console.print(
            f"[bold red]Error: Run '{args.run_id}' is corrupted: {exc}[/bold red]"
        )
        sys.exit(1)

    # 2. Promote to baseline
    baseline = promote_run_to_baseline(
        run=run,
        baseline_id=args.baseline_id,
        baselines_dir=args.baselines_dir,
    )

    console.print(f"  [bold green]✓ Successfully promoted run to baseline![/bold green]")
    console.print(f"    Baseline ID:       [cyan]{baseline.baseline_id}[/cyan]")
    console.print(f"    Source Run ID:     {baseline.run_id}")
    console.print(f"    Prompt Version:    {baseline.prompt_version}")
    console.print(f"    Dataset Version:   {baseline.dataset_version}")
    console.print(f"    Model:             {baseline.model_name} ({baseline.model_provider})")
    console.print(f"    Category Accuracy: {baseline.metrics.get('category_accuracy', 0.0):.1%}")
    console.print(f"    Total Cases:       {len(baseline.case_results)}")
    console.print(f"    Saved to:          [dim]{args.baselines_dir}/current_baseline.json[/dim]")
    console.print()


if __name__ == "__main__":
    main()