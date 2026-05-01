"""Phase 3 tests: sandbagging rate computation and results export."""

import csv
import json
import os
import tempfile

import pytest

from analysis.metrics import compute_metrics
from pipeline.run import _extract_records, _save_results
from pipeline.config import PipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_records(overrides: list[dict]) -> list[dict]:
    """Build a minimal records list with sensible defaults."""
    base = {
        "item_id": "test_0",
        "category": "math",
        "condition": "neutral",
        "prompt": "Q?",
        "response": "A",
        "expected": "A",
        "score": 1.0,
    }
    return [{**base, **o} for o in overrides]


def _full_records(
    neutral_scores: list[float],
    subtle_scores: list[float],
    explicit_scores: list[float],
    category: str = "math",
) -> list[dict]:
    records = []
    for i, (n, s, e) in enumerate(zip(neutral_scores, subtle_scores, explicit_scores)):
        for cond, sc in [("neutral", n), ("subtle", s), ("explicit", e)]:
            records.append({
                "item_id": f"{category}_{i}_{cond}",
                "category": category,
                "condition": cond,
                "prompt": "Q?",
                "response": "A",
                "expected": "A",
                "score": sc,
            })
    return records


# ---------------------------------------------------------------------------
# Sandbagging rate formula
# ---------------------------------------------------------------------------

def test_sandbagging_rate_basic():
    records = _full_records(
        neutral_scores=[1.0, 1.0],   # neutral accuracy = 1.0
        subtle_scores=[1.0, 0.0],    # subtle accuracy  = 0.5  → rate = 0.5
        explicit_scores=[0.0, 0.0],  # explicit accuracy= 0.0  → rate = 1.0
    )
    metrics = compute_metrics(records)
    assert metrics["sandbagging_rate"]["per_category"]["math"]["subtle"] == pytest.approx(0.5)
    assert metrics["sandbagging_rate"]["per_category"]["math"]["explicit"] == pytest.approx(1.0)


def test_sandbagging_rate_zero_when_no_drop():
    records = _full_records(
        neutral_scores=[1.0, 1.0],
        subtle_scores=[1.0, 1.0],
        explicit_scores=[1.0, 1.0],
    )
    metrics = compute_metrics(records)
    assert metrics["sandbagging_rate"]["overall"]["subtle"] == 0.0
    assert metrics["sandbagging_rate"]["overall"]["explicit"] == 0.0


def test_sandbagging_rate_negative_when_eval_aware_better():
    # Edge case: model performs better under eval framing
    records = _full_records(
        neutral_scores=[0.0, 0.0],
        subtle_scores=[1.0, 1.0],
        explicit_scores=[1.0, 1.0],
    )
    metrics = compute_metrics(records)
    assert metrics["sandbagging_rate"]["overall"]["subtle"] < 0


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------

def test_accuracy_per_category_and_condition():
    records = _full_records(
        neutral_scores=[1.0, 1.0, 0.0, 0.0],  # 0.5
        subtle_scores=[1.0, 0.0, 0.0, 0.0],   # 0.25
        explicit_scores=[0.0, 0.0, 0.0, 0.0], # 0.0
    )
    metrics = compute_metrics(records)
    assert metrics["accuracy"]["per_category"]["math"]["neutral"] == pytest.approx(0.5)
    assert metrics["accuracy"]["per_category"]["math"]["subtle"] == pytest.approx(0.25)
    assert metrics["accuracy"]["per_category"]["math"]["explicit"] == pytest.approx(0.0)


def test_accuracy_overall():
    records = (
        _full_records([1.0], [0.0], [0.0], category="math")
        + _full_records([1.0], [1.0], [0.0], category="factual_recall")
    )
    metrics = compute_metrics(records)
    assert metrics["accuracy"]["overall"]["neutral"] == pytest.approx(1.0)
    assert metrics["accuracy"]["overall"]["subtle"] == pytest.approx(0.5)
    assert metrics["accuracy"]["overall"]["explicit"] == pytest.approx(0.0)


def test_metrics_output_has_required_keys():
    records = _full_records([1.0], [0.5], [0.0])
    metrics = compute_metrics(records)
    assert "accuracy" in metrics
    assert "sandbagging_rate" in metrics
    assert "per_category" in metrics["accuracy"]
    assert "overall" in metrics["accuracy"]
    assert "per_category" in metrics["sandbagging_rate"]
    assert "overall" in metrics["sandbagging_rate"]


# ---------------------------------------------------------------------------
# Results export — JSON and CSV schema
# ---------------------------------------------------------------------------

def test_save_results_creates_files(tmp_path):
    records = _full_records([1.0, 0.0], [1.0, 0.0], [0.0, 0.0])
    config = PipelineConfig(
        model="mockllm/model", dataset_path="data/dataset.json",
        scorer="exact_match", adapter="mmlu", output_dir="results", seed=42,
    )
    run_dir = str(tmp_path / "run")
    metrics_path, csv_path, _ = _save_results(records, config, run_dir)

    assert os.path.exists(csv_path)
    assert os.path.exists(metrics_path)


def test_raw_responses_csv_schema(tmp_path):
    records = _full_records([1.0], [0.0], [0.0])
    config = PipelineConfig(
        model="mockllm/model", dataset_path="data/dataset.json",
        scorer="exact_match", adapter="mmlu", output_dir="results", seed=42,
    )
    run_dir = str(tmp_path / "run")
    _, csv_path, _ = _save_results(records, config, run_dir)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(records)
    required_fields = {"item_id", "category", "condition", "response", "expected", "score"}
    assert required_fields.issubset(set(rows[0].keys()))


def test_aggregated_metrics_json_schema(tmp_path):
    records = _full_records([1.0], [0.5], [0.0])
    config = PipelineConfig(
        model="mockllm/model", dataset_path="data/dataset.json",
        scorer="exact_match", adapter="mmlu", output_dir="results", seed=42,
    )
    run_dir = str(tmp_path / "run")
    metrics_path, _, _ = _save_results(records, config, run_dir)

    with open(metrics_path) as f:
        metrics = json.load(f)

    assert "accuracy" in metrics
    assert "sandbagging_rate" in metrics
    assert "per_category" in metrics["sandbagging_rate"]
    assert "overall" in metrics["sandbagging_rate"]
