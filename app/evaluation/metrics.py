import math
from typing import Any

from app.evaluation.models import EvalCaseResult


def calculate_percentile(values: list[float], percentile: float) -> float:
    """Calculate the p-th percentile of a list of numeric values using linear interpolation.

    Handles small sample sizes gracefully without crashing on N=1, N=2, or empty lists.

    Args:
        values: List of floats (e.g. latency in ms).
        percentile: Float between 0.0 and 100.0 (e.g. 50.0 for median, 95.0 for P95).

    Returns:
        The calculated percentile value, or 0.0 if values is empty.
    """
    if not values:
        return 0.0

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    if n == 1:
        return float(sorted_vals[0])

    # Rank index in 0-indexed terms
    rank = (percentile / 100.0) * (n - 1)
    lower_idx = int(math.floor(rank))
    upper_idx = int(math.ceil(rank))

    if lower_idx == upper_idx:
        return float(sorted_vals[lower_idx])

    # Linear interpolation between adjacent values
    fraction = rank - lower_idx
    return float(sorted_vals[lower_idx] + fraction * (sorted_vals[upper_idx] - sorted_vals[lower_idx]))


def compute_run_metrics(cases: list[EvalCaseResult]) -> dict[str, Any]:
    """Compute all functional, semantic, and operational metrics across test cases.

    Args:
        cases: The list of completed EvalCaseResult evidence objects.

    Returns:
        Dictionary mapping directly to RunResult aggregate fields.
    """
    total = len(cases)
    if total == 0:
        return {
            "total_cases": 0,
            "passed_cases": 0,
            "failed_cases": 0,
            "category_accuracy": 0.0,
            "average_summary_score": None,
            "average_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "average_input_tokens": 0.0,
            "average_output_tokens": 0.0,
        }

    # 1. Functional Dimension (Category Correctness)
    passed_cases = sum(1 for c in cases if c.status == "PASS")
    failed_cases = sum(1 for c in cases if c.status in ("FAIL", "ERROR"))
    category_matches = sum(1 for c in cases if c.category_match)
    category_accuracy = category_matches / total

    # 2. Semantic Dimension (Summary Relevance via LLM-as-Judge)
    summary_scores = [c.summary_score for c in cases if c.summary_score is not None]
    average_summary_score = (
        sum(summary_scores) / len(summary_scores) if summary_scores else None
    )

    # 3. Operational Dimension (Latency & Token Counts)
    latencies = [c.latency_ms for c in cases]
    average_latency_ms = sum(latencies) / total
    p50_latency_ms = calculate_percentile(latencies, 50.0)
    p95_latency_ms = calculate_percentile(latencies, 95.0)

    input_tokens_list = [c.input_tokens for c in cases if c.input_tokens is not None]
    average_input_tokens = (
        sum(input_tokens_list) / len(input_tokens_list) if input_tokens_list else 0.0
    )

    output_tokens_list = [c.output_tokens for c in cases if c.output_tokens is not None]
    average_output_tokens = (
        sum(output_tokens_list) / len(output_tokens_list) if output_tokens_list else 0.0
    )

    return {
        "total_cases": total,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "category_accuracy": round(category_accuracy, 4),
        "average_summary_score": round(average_summary_score, 2) if average_summary_score is not None else None,
        "average_latency_ms": round(average_latency_ms, 2),
        "p50_latency_ms": round(p50_latency_ms, 2),
        "p95_latency_ms": round(p95_latency_ms, 2),
        "average_input_tokens": round(average_input_tokens, 1),
        "average_output_tokens": round(average_output_tokens, 1),
    }