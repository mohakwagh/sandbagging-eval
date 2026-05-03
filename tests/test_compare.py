"""Tests for cross-model comparison report generation."""

import json
import os

from analysis.compare import generate_comparison_report, load_runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_metrics(neutral=0.8, subtle=0.7, explicit=0.6, categories=None):
    cats = categories or ["math", "factual_recall", "logical_reasoning"]
    return {
        "accuracy": {
            "per_category": {cat: {"neutral": neutral, "subtle": subtle, "explicit": explicit} for cat in cats},
            "overall": {"neutral": neutral, "subtle": subtle, "explicit": explicit},
        },
        "sandbagging_rate": {
            "per_category": {cat: {"subtle": round(neutral - subtle, 4), "explicit": round(neutral - explicit, 4)} for cat in cats},
            "overall": {"subtle": round(neutral - subtle, 4), "explicit": round(neutral - explicit, 4)},
        },
    }


def _write_run(tmp_path, model_slug, timestamp, adapter, metrics, scorer="exact_match"):
    run_dir = tmp_path / f"{model_slug}_{timestamp}"
    run_dir.mkdir()
    (run_dir / "aggregated_metrics.json").write_text(json.dumps(metrics))
    (run_dir / "run_config.json").write_text(json.dumps({
        "model": f"openai/{model_slug}",
        "adapter": adapter,
        "scorer": scorer,
        "seed": 42,
        "dataset_path": "data/dataset.json",
    }))
    return run_dir


# ---------------------------------------------------------------------------
# load_runs: adapter filtering
# ---------------------------------------------------------------------------

def test_load_runs_filters_by_adapter(tmp_path):
    _write_run(tmp_path, "gpt-a", "2026-05-01T10-00-00", "mmlu", _sample_metrics())
    _write_run(tmp_path, "gpt-b", "2026-05-01T11-00-00", "wmdp", _sample_metrics())

    runs = load_runs(str(tmp_path), "mmlu")
    assert len(runs) == 1
    assert "gpt-a" in runs[0]["model"]


def test_load_runs_excludes_non_matching_adapter(tmp_path):
    _write_run(tmp_path, "gpt-a", "2026-05-01T10-00-00", "wmdp", _sample_metrics())
    runs = load_runs(str(tmp_path), "mmlu")
    assert len(runs) == 0


# ---------------------------------------------------------------------------
# load_runs: latest run per model
# ---------------------------------------------------------------------------

def test_load_runs_returns_latest_run_per_model(tmp_path):
    _write_run(tmp_path, "gpt-a", "2026-05-01T10-00-00", "mmlu", _sample_metrics(neutral=0.5))
    _write_run(tmp_path, "gpt-a", "2026-05-01T12-00-00", "mmlu", _sample_metrics(neutral=0.9))

    runs = load_runs(str(tmp_path), "mmlu")
    assert len(runs) == 1
    assert runs[0]["metrics"]["accuracy"]["overall"]["neutral"] == 0.9


def test_load_runs_handles_multiple_models(tmp_path):
    _write_run(tmp_path, "gpt-a", "2026-05-01T10-00-00", "mmlu", _sample_metrics())
    _write_run(tmp_path, "gpt-b", "2026-05-01T11-00-00", "mmlu", _sample_metrics())
    _write_run(tmp_path, "gpt-c", "2026-05-01T12-00-00", "mmlu", _sample_metrics())

    runs = load_runs(str(tmp_path), "mmlu")
    assert len(runs) == 3


# ---------------------------------------------------------------------------
# load_runs: backward compat (no run_config.json)
# ---------------------------------------------------------------------------

def test_load_runs_includes_dirs_without_run_config(tmp_path, capsys):
    run_dir = tmp_path / "openai_gpt-a_2026-05-01T10-00-00"
    run_dir.mkdir()
    (run_dir / "aggregated_metrics.json").write_text(json.dumps(_sample_metrics()))

    runs = load_runs(str(tmp_path), "mmlu")
    assert len(runs) == 1
    captured = capsys.readouterr()
    assert "warn" in captured.out


# ---------------------------------------------------------------------------
# generate_comparison_report: HTML output
# ---------------------------------------------------------------------------

def test_comparison_report_generates_html(tmp_path):
    runs = [
        {"model": "openai/gpt-a", "metrics": _sample_metrics(neutral=0.8, subtle=0.7, explicit=0.6)},
        {"model": "openai/gpt-b", "metrics": _sample_metrics(neutral=0.9, subtle=0.85, explicit=0.8)},
    ]
    output_path = str(tmp_path / "comparison.html")
    generate_comparison_report(runs, output_path)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_comparison_report_is_valid_html(tmp_path):
    runs = [{"model": "openai/gpt-a", "metrics": _sample_metrics()}]
    output_path = str(tmp_path / "comparison.html")
    generate_comparison_report(runs, output_path)
    content = open(output_path).read()
    assert "<html" in content.lower()
    assert "</html>" in content.lower()


def test_comparison_report_contains_model_names(tmp_path):
    runs = [
        {"model": "openai/gpt-a", "metrics": _sample_metrics()},
        {"model": "openai/gpt-b", "metrics": _sample_metrics()},
    ]
    output_path = str(tmp_path / "comparison.html")
    generate_comparison_report(runs, output_path)
    content = open(output_path).read()
    assert "gpt-a" in content
    assert "gpt-b" in content


# ---------------------------------------------------------------------------
# generate_comparison_report: per-category conditional rendering
# ---------------------------------------------------------------------------

def test_per_category_shown_for_categorized_dataset(tmp_path):
    runs = [{"model": "openai/gpt-a", "metrics": _sample_metrics(categories=["math", "factual_recall"])}]
    output_path = str(tmp_path / "comparison.html")
    generate_comparison_report(runs, output_path)
    content = open(output_path).read()
    assert "math" in content
    assert "Per-Category" in content


def test_per_category_omitted_for_uncategorized_dataset(tmp_path):
    runs = [{"model": "openai/gpt-a", "metrics": _sample_metrics(categories=["uncategorized"])}]
    output_path = str(tmp_path / "comparison.html")
    generate_comparison_report(runs, output_path)
    content = open(output_path).read()
    assert "Per-Category" not in content


# ---------------------------------------------------------------------------
# load_runs: scorer filtering
# ---------------------------------------------------------------------------

def test_load_runs_filters_by_scorer(tmp_path):
    _write_run(tmp_path, "gpt-a", "2026-05-01T10-00-00", "mmlu", _sample_metrics(), scorer="exact_match")
    _write_run(tmp_path, "gpt-b", "2026-05-01T11-00-00", "mmlu", _sample_metrics(), scorer="llm_judge")

    runs = load_runs(str(tmp_path), "mmlu", scorer="exact_match")
    assert len(runs) == 1
    assert "gpt-a" in runs[0]["model"]


def test_load_runs_excludes_mismatched_scorer(tmp_path):
    _write_run(tmp_path, "gpt-a", "2026-05-01T10-00-00", "mmlu", _sample_metrics(), scorer="llm_judge")

    runs = load_runs(str(tmp_path), "mmlu", scorer="exact_match")
    assert len(runs) == 0


def test_load_runs_raises_on_mixed_scorers(tmp_path):
    _write_run(tmp_path, "gpt-a", "2026-05-01T10-00-00", "mmlu", _sample_metrics(), scorer="exact_match")
    _write_run(tmp_path, "gpt-b", "2026-05-01T11-00-00", "mmlu", _sample_metrics(), scorer="llm_judge")

    import pytest
    with pytest.raises(SystemExit):
        load_runs(str(tmp_path), "mmlu")


def test_load_runs_no_error_when_single_scorer(tmp_path):
    _write_run(tmp_path, "gpt-a", "2026-05-01T10-00-00", "mmlu", _sample_metrics(), scorer="exact_match")
    _write_run(tmp_path, "gpt-b", "2026-05-01T11-00-00", "mmlu", _sample_metrics(), scorer="exact_match")

    runs = load_runs(str(tmp_path), "mmlu")
    assert len(runs) == 2
