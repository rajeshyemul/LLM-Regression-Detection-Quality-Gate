import json
from pathlib import Path
from typing import Any

from datasets.schemas.golden_case import GoldenDataset, GoldenTestCase


def load_golden_dataset(dataset_path: str | Path = "datasets/golden/v1.json") -> GoldenDataset:
    """Load and validate a GoldenDataset from a JSON file.

    Args:
        dataset_path: Path to the golden dataset JSON file.

    Returns:
        Validated GoldenDataset instance.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If JSON is invalid or duplicate test case IDs exist.
        pydantic.ValidationError: If the structure does not match GoldenDataset schema.
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Golden dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON syntax in {path}: {exc}") from exc

    dataset = GoldenDataset.model_validate(data)

    # Integrity Check: Ensure all test case IDs are unique
    seen_ids: set[str] = set()
    for case in dataset.cases:
        if case.id in seen_ids:
            raise ValueError(
                f"Data integrity violation in {path}: Duplicate test case ID '{case.id}' detected."
            )
        seen_ids.add(case.id)

    return dataset


def get_dataset_stats(dataset: GoldenDataset) -> dict[str, Any]:
    """Calculate statistical distribution of categories and difficulties.

    Args:
        dataset: The loaded GoldenDataset instance.

    Returns:
        Dictionary containing counts and percentage breakdowns.
    """
    categories: dict[str, int] = {}
    difficulties: dict[str, int] = {}

    for case in dataset.cases:
        categories[case.expected_category] = categories.get(case.expected_category, 0) + 1
        difficulties[case.expected_difficulty] = difficulties.get(case.expected_difficulty, 0) + 1

    return {
        "version": dataset.version,
        "total_cases": len(dataset.cases),
        "categories": categories,
        "difficulties": difficulties,
    }


def filter_cases(
    dataset: GoldenDataset,
    category: str | None = None,
    difficulty: str | None = None,
) -> list[GoldenTestCase]:
    """Filter test cases by category or difficulty level.

    Args:
        dataset: The loaded GoldenDataset instance.
        category: Optional category filter (e.g. 'billing').
        difficulty: Optional difficulty filter (e.g. 'hard').

    Returns:
        List of matching GoldenTestCase instances.
    """
    results = dataset.cases

    if category is not None:
        results = [c for c in results if c.expected_category == category]

    if difficulty is not None:
        results = [c for c in results if c.expected_difficulty == difficulty]

    return results