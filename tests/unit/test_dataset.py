import json
from pathlib import Path
import pytest

from datasets.loader import filter_cases, get_dataset_stats, load_golden_dataset
from datasets.schemas.golden_case import GoldenDataset


def test_load_golden_dataset_v1():
    dataset = load_golden_dataset("datasets/golden/v1.json")
    assert isinstance(dataset, GoldenDataset)
    assert dataset.version == "v1.0"
    assert len(dataset.cases) == 30

    # Ensure every ID follows TC-xxx format
    for case in dataset.cases:
        assert case.id.startswith("TC-")
        assert len(case.input) > 0
        assert len(case.expected_summary) > 0


def test_load_golden_dataset_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="Golden dataset file not found"):
        load_golden_dataset("datasets/golden/non_existent.json")


def test_load_golden_dataset_invalid_json_raises(tmp_path: Path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ this is not valid json }", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON syntax"):
        load_golden_dataset(bad_json)


def test_load_golden_dataset_duplicate_id_raises(tmp_path: Path):
    dup_dataset = {
        "version": "v1.0-dup",
        "description": "Dataset with duplicate IDs",
        "cases": [
            {
                "id": "TC-001",
                "input": "First charge problem",
                "expected_category": "billing",
                "expected_summary": "Charge problem summary",
                "expected_difficulty": "easy",
            },
            {
                "id": "TC-001",  # Duplicate ID
                "input": "Second charge problem",
                "expected_category": "billing",
                "expected_summary": "Second summary",
                "expected_difficulty": "easy",
            },
        ],
    }
    dup_file = tmp_path / "duplicate.json"
    dup_file.write_text(json.dumps(dup_dataset), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate test case ID 'TC-001' detected"):
        load_golden_dataset(dup_file)


def test_get_dataset_stats():
    dataset = load_golden_dataset("datasets/golden/v1.json")
    stats = get_dataset_stats(dataset)

    assert stats["version"] == "v1.0"
    assert stats["total_cases"] == 30
    assert stats["categories"]["billing"] == 8
    assert stats["categories"]["technical"] == 8
    assert stats["categories"]["account"] == 7
    assert stats["categories"]["general"] == 7
    assert sum(stats["categories"].values()) == 30


def test_filter_cases_by_category_and_difficulty():
    dataset = load_golden_dataset("datasets/golden/v1.json")

    billing_cases = filter_cases(dataset, category="billing")
    assert len(billing_cases) == 8
    assert all(c.expected_category == "billing" for c in billing_cases)

    hard_cases = filter_cases(dataset, difficulty="hard")
    assert len(hard_cases) == 3
    assert all(c.expected_difficulty == "hard" for c in hard_cases)

    hard_technical = filter_cases(dataset, category="technical", difficulty="hard")
    assert len(hard_technical) == 2
    assert all(
        c.expected_category == "technical" and c.expected_difficulty == "hard"
        for c in hard_technical
    )