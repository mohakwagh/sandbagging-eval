"""Phase 4 tests: Plotly report generation."""

import json
import os

from analysis.visualization import generate_report, CONDITIONS, SANDBAGGING_CONDITIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_metrics():
    categories = ["factual_recall", "logical_reasoning", "math"]
    return {
        "accuracy": {
            "per_category": {
                cat: {"neutral": 0.8, "subtle": 0.65, "explicit": 0.5}
                for cat in categories
            },
            "overall": {"neutral": 0.8, "subtle": 0.65, "explicit": 0.5},
        },
        "sandbagging_rate": {
            "per_category": {
                cat: {"subtle": 0.15, "explicit": 0.3}
                for cat in categories
            },
            "overall": {"subtle": 0.15, "explicit": 0.3},
        },
    }


# ---------------------------------------------------------------------------
# 1. Report generates without errors
# ---------------------------------------------------------------------------

def test_report_generates_without_errors(tmp_path):
    output_path = str(tmp_path / "report.html")
    generate_report(_sample_metrics(), output_path)


def test_report_generates_from_json_metrics(tmp_path):
    metrics_path = str(tmp_path / "aggregated_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(_sample_metrics(), f)

    with open(metrics_path) as f:
        metrics = json.load(f)

    output_path = str(tmp_path / "report.html")
    generate_report(metrics, output_path)


# ---------------------------------------------------------------------------
# 2. Output file exists at expected path
# ---------------------------------------------------------------------------

def test_report_file_exists_at_output_path(tmp_path):
    output_path = str(tmp_path / "report.html")
    generate_report(_sample_metrics(), output_path)
    assert os.path.exists(output_path)


# ---------------------------------------------------------------------------
# 3. HTML file is valid and non-empty
# ---------------------------------------------------------------------------

def test_report_html_is_non_empty(tmp_path):
    output_path = str(tmp_path / "report.html")
    generate_report(_sample_metrics(), output_path)
    assert os.path.getsize(output_path) > 0


def test_report_is_valid_html(tmp_path):
    output_path = str(tmp_path / "report.html")
    generate_report(_sample_metrics(), output_path)
    with open(output_path) as f:
        content = f.read()
    assert content.strip().startswith("<")
    assert "<html" in content.lower()
    assert "</html>" in content.lower()


# ---------------------------------------------------------------------------
# 4. Charts contain expected data keys
# ---------------------------------------------------------------------------

def test_report_contains_all_conditions(tmp_path):
    output_path = str(tmp_path / "report.html")
    generate_report(_sample_metrics(), output_path)
    with open(output_path) as f:
        content = f.read()
    for condition in CONDITIONS:
        assert condition in content


def test_report_contains_all_categories(tmp_path):
    output_path = str(tmp_path / "report.html")
    generate_report(_sample_metrics(), output_path)
    with open(output_path) as f:
        content = f.read()
    for category in ["math", "factual_recall", "logical_reasoning"]:
        assert category in content


def test_report_contains_sandbagging_rate_values(tmp_path):
    output_path = str(tmp_path / "report.html")
    generate_report(_sample_metrics(), output_path)
    with open(output_path) as f:
        content = f.read()
    for condition in SANDBAGGING_CONDITIONS:
        assert condition in content
    assert "Sandbagging Rate" in content
