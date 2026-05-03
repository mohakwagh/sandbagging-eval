"""
Cross-model comparison report for a single dataset.

Run this after collecting results from multiple models to generate a side-by-side
comparison of accuracy and sandbagging rates:

    python -m analysis.compare --results_dir results/ --dataset mmlu --open

Only runs that used the specified adapter (dataset) are included. For each model,
the most recent run is used. Runs without a run_config.json (produced by older
pipeline versions) are included with a warning.

Output: results/comparison_{dataset}.html
"""

import argparse
import json
import os
import re
import webbrowser

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from analysis.visualization import CONDITION_COLORS, SANDBAGGING_CONDITIONS

# One distinct color per model (cycles if > 8 models)
_MODEL_COLORS = [
    "#4C9BE8", "#E85454", "#2ECC71", "#9B59B6",
    "#F39C12", "#1ABC9C", "#E74C3C", "#3498DB",
]

_TIMESTAMP_RE = re.compile(r"_(\d{4}-\d{2}-\d{2}T[\d-]+)$")


def _parse_run_dir(run_dir: str) -> tuple[str, str]:
    """Return (model_slug, timestamp_str) parsed from a run directory name."""
    match = _TIMESTAMP_RE.search(run_dir)
    if not match:
        return run_dir, ""
    timestamp = match.group(1)
    model_slug = run_dir[: match.start()]
    return model_slug, timestamp


def load_runs(results_dir: str, dataset: str, scorer: str = None) -> list[dict]:
    """
    Scan results_dir for run subdirectories and return one entry per model
    (the latest run by timestamp) where the adapter matches dataset.

    If scorer is specified, only runs with a matching scorer are included.
    Runs without run_config.json are included with a warning (backward compat).

    Raises SystemExit if scorer is not specified and mixed scorers are detected
    across the collected runs — comparing runs with different scorers produces
    misleading results and must be prevented.

    Returns a list of dicts: {"model": str, "metrics": dict}
    """
    candidates: dict[str, dict] = {}  # model_slug → {timestamp, metrics, model, scorer}

    for entry in os.listdir(results_dir):
        run_dir = os.path.join(results_dir, entry)
        metrics_path = os.path.join(run_dir, "aggregated_metrics.json")
        if not os.path.isdir(run_dir) or not os.path.exists(metrics_path):
            continue

        model_slug, timestamp = _parse_run_dir(entry)
        if not timestamp:
            continue

        config_path = os.path.join(run_dir, "run_config.json")
        run_scorer = None
        if os.path.exists(config_path):
            with open(config_path) as f:
                run_config = json.load(f)
            if run_config.get("adapter") != dataset:
                continue
            if scorer and run_config.get("scorer") != scorer:
                continue
            model_name = run_config["model"]
            run_scorer = run_config.get("scorer")
        else:
            # Backward compat: no run_config.json — assume it matches
            print(f"  [warn] {entry}: no run_config.json, assuming adapter='{dataset}'")
            model_name = model_slug.replace("openai_", "openai/", 1)

        with open(metrics_path) as f:
            metrics = json.load(f)

        # Keep only the latest run per model
        if model_slug not in candidates or timestamp > candidates[model_slug]["timestamp"]:
            candidates[model_slug] = {
                "timestamp": timestamp,
                "model": model_name,
                "metrics": metrics,
                "scorer": run_scorer,
            }

    # Detect mixed scorers and abort — comparing runs with different scorers
    # produces misleading charts since scoring methods are not equivalent
    if not scorer:
        scorers_found = {v["scorer"] for v in candidates.values() if v["scorer"] is not None}
        if len(scorers_found) > 1:
            scorers_list = ", ".join(sorted(scorers_found))
            print(f"\n[error] Mixed scorers found for dataset '{dataset}': {scorers_list}")
            print("        Re-run with --scorer to restrict to one scorer:")
            for s in sorted(scorers_found):
                print(f"          python -m analysis.compare --results_dir {results_dir} "
                      f"--dataset {dataset} --scorer {s} --open")
            raise SystemExit(1)

    runs = [
        {"model": v["model"], "metrics": v["metrics"], "scorer": v["scorer"]}
        for v in sorted(candidates.values(), key=lambda x: x["model"])
    ]
    return runs


def generate_comparison_report(runs: list[dict], output_path: str) -> None:
    """
    Generate a standalone HTML comparison report across multiple models.

    Always renders:
      1. Overall accuracy by model (neutral / subtle / explicit)
      2. Overall sandbagging rate by model (subtle / explicit)

    Conditionally renders (when categories are not ["uncategorized"]):
      3. Per-category sandbagging rate — subtle condition
      4. Per-category sandbagging rate — explicit condition
    """
    model_labels = [r["model"].split("/")[-1] for r in runs]
    model_colors = {
        label: _MODEL_COLORS[i % len(_MODEL_COLORS)]
        for i, label in enumerate(model_labels)
    }

    # Detect whether per-category breakdown is meaningful
    all_categories = set()
    for r in runs:
        all_categories.update(r["metrics"]["accuracy"]["per_category"].keys())
    show_per_category = all_categories != {"uncategorized"} and len(all_categories) > 0
    categories = sorted(all_categories) if show_per_category else []

    n_rows = 4 if show_per_category else 2
    subplot_titles = [
        "Overall Accuracy by Model",
        "Overall Sandbagging Rate by Model",
    ]
    if show_per_category:
        subplot_titles += [
            "Per-Category Sandbagging Rate (Subtle)",
            "Per-Category Sandbagging Rate (Explicit)",
        ]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=subplot_titles,
        specs=[[{"type": "bar"}]] * n_rows,
        vertical_spacing=0.08,
    )

    # Chart 1: overall accuracy per condition per model
    for condition in ["neutral", "subtle", "explicit"]:
        fig.add_trace(
            go.Bar(
                name=condition,
                x=model_labels,
                y=[r["metrics"]["accuracy"]["overall"][condition] for r in runs],
                marker_color=CONDITION_COLORS[condition],
                legendgroup=condition,
            ),
            row=1, col=1,
        )

    # Chart 2: overall sandbagging rate per condition per model
    for condition in SANDBAGGING_CONDITIONS:
        fig.add_trace(
            go.Bar(
                name=f"{condition} (delta)",
                x=model_labels,
                y=[r["metrics"]["sandbagging_rate"]["overall"][condition] for r in runs],
                marker_color=CONDITION_COLORS[condition],
                legendgroup=f"{condition}_delta",
            ),
            row=2, col=1,
        )

    if show_per_category:
        # Chart 3: per-category sandbagging rate, subtle condition, bars by model
        for label, run in zip(model_labels, runs):
            fig.add_trace(
                go.Bar(
                    name=label,
                    x=categories,
                    y=[
                        run["metrics"]["sandbagging_rate"]["per_category"].get(cat, {}).get("subtle", 0.0)
                        for cat in categories
                    ],
                    marker_color=model_colors[label],
                    legendgroup=label,
                    showlegend=True,
                ),
                row=3, col=1,
            )

        # Chart 4: per-category sandbagging rate, explicit condition, bars by model
        for label, run in zip(model_labels, runs):
            fig.add_trace(
                go.Bar(
                    name=label,
                    x=categories,
                    y=[
                        run["metrics"]["sandbagging_rate"]["per_category"].get(cat, {}).get("explicit", 0.0)
                        for cat in categories
                    ],
                    marker_color=model_colors[label],
                    legendgroup=label,
                    showlegend=False,
                ),
                row=4, col=1,
            )

    fig.update_layout(
        title_text="LLM Sandbagging Detection — Cross-Model Comparison",
        barmode="group",
        height=500 * n_rows,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Sandbagging Rate", row=2, col=1)
    if show_per_category:
        fig.update_yaxes(title_text="Sandbagging Rate (Subtle)", row=3, col=1)
        fig.update_yaxes(title_text="Sandbagging Rate (Explicit)", row=4, col=1)

    pio.write_html(fig, output_path, full_html=True, include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate cross-model sandbagging comparison report",
        epilog="Example: python -m analysis.compare --results_dir results/ --dataset mmlu --open",
    )
    parser.add_argument("--results_dir", default="results", help="Directory containing run subdirectories")
    parser.add_argument("--dataset", required=True, help="Adapter name to filter runs (e.g. mmlu)")
    parser.add_argument("--scorer", default=None, help="Scorer to filter runs (e.g. exact_match). Required when multiple scorers exist for the dataset.")
    parser.add_argument("--open", action="store_true", help="Open the report in browser after generating")
    args = parser.parse_args()

    print(f"Scanning {args.results_dir}/ for '{args.dataset}' runs...")
    runs = load_runs(args.results_dir, args.dataset, scorer=args.scorer)

    if not runs:
        print(f"No runs found for dataset '{args.dataset}'. Run the pipeline first.")
        return

    print(f"Found {len(runs)} model(s): {[r['model'] for r in runs]}")

    # Include scorer in filename so runs with different scorers don't overwrite each other.
    # Use the explicitly requested scorer if given; otherwise infer from runs.
    effective_scorer = args.scorer
    if not effective_scorer:
        inferred = {r.get("scorer") for r in runs if r.get("scorer")}
        if len(inferred) == 1:
            effective_scorer = inferred.pop()

    if effective_scorer:
        filename = f"comparison_{args.dataset}_{effective_scorer}.html"
    else:
        filename = f"comparison_{args.dataset}.html"

    output_path = os.path.join(args.results_dir, filename)
    generate_comparison_report(runs, output_path)
    print(f"Comparison report saved to {output_path}")

    if args.open:
        webbrowser.open(f"file://{os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
