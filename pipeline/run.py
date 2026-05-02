"""Pipeline entry point for sandbagging detection evaluation."""

import argparse
import csv
import json
import os
import webbrowser
from datetime import datetime

from inspect_ai import eval as inspect_eval

from analysis.metrics import compute_metrics
from analysis.visualization import generate_report
from dataset_builder.schema import PromptInstance
from pipeline.config import load_config
from pipeline.task import build_task


def _extract_records(eval_results) -> list[dict]:
    """
    Flatten Inspect EvalLog samples into a list of plain dicts for downstream
    processing. Each dict carries the fields needed by compute_metrics() and
    the CSV export: item identifiers, condition/category metadata, the raw
    model response, the expected answer, and the numeric score (0.0 or 1.0
    for binary scorers; any float in [0, 1] for continuous scorers).
    """
    records = []
    for sample in eval_results[0].samples:
        # scores is a dict keyed by scorer name; we use the first (and only) scorer
        score_value = list(sample.scores.values())[0].value
        records.append({
            "item_id": sample.id,
            "category": sample.metadata["category"],
            "condition": sample.metadata["condition"],
            "prompt": sample.input,
            "response": sample.output.completion,
            "expected": sample.target if isinstance(sample.target, str) else sample.target[0],
            "score": float(score_value),
        })
    return records


def _save_results(records: list[dict], config, run_dir: str):
    """
    Persist all run outputs to results/{model}_{timestamp}/:
      - raw_responses.csv       one row per prompt instance
      - aggregated_metrics.json accuracy + sandbagging rates per condition/category
      - run_config.json         adapter, model, seed — used by analysis.compare to filter runs by dataset
      - report.html             standalone Plotly visualization
    """
    os.makedirs(run_dir, exist_ok=True)

    # Raw responses — one row per prompt instance across all conditions/categories
    csv_path = os.path.join(run_dir, "raw_responses.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    # Aggregated metrics — accuracy per condition/category + sandbagging rates
    metrics = compute_metrics(records)
    metrics_path = os.path.join(run_dir, "aggregated_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Run config — lets analysis.compare filter runs by adapter/dataset
    run_config_path = os.path.join(run_dir, "run_config.json")
    with open(run_config_path, "w") as f:
        json.dump({
            "model": config.model,
            "adapter": config.adapter,
            "seed": config.seed,
            "dataset_path": config.dataset_path,
        }, f, indent=2)

    # HTML report — standalone Plotly visualization, no server required
    report_path = os.path.join(run_dir, "report.html")
    generate_report(metrics, report_path)

    return metrics_path, csv_path, report_path


def main():
    parser = argparse.ArgumentParser(description="Run sandbagging detection pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (for testing)")
    parser.add_argument("--open", action="store_true", help="Open report.html in browser after run")
    args = parser.parse_args()

    # --- Stage 1: Load and validate config ---
    print("[1/5] Loading config...")
    config = load_config(args.config)
    print(f"      model    : {config.model}")
    print(f"      scorer   : {config.scorer}")
    print(f"      dataset  : {config.dataset_path}")

    # --- Stage 2: Load preprocessed dataset ---
    print("[2/5] Loading dataset...")
    with open(config.dataset_path) as f:
        raw = json.load(f)
    instances = [PromptInstance(**item) for item in raw]
    if args.limit:
        instances = instances[: args.limit]
    print(f"      {len(instances)} prompt instances loaded"
          + (f" (limited from {len(raw)})" if args.limit else ""))

    # --- Stage 3: Build Inspect Task ---
    print("[3/5] Building task...")
    task = build_task(instances, config)
    print(f"      {len(list(task.dataset))} samples across "
          f"{len({s.metadata['condition'] for s in task.dataset})} conditions, "
          f"{len({s.metadata['category'] for s in task.dataset})} categories")

    # --- Stage 4: Run evaluation (model API calls via Inspect) ---
    print(f"[4/5] Running evaluation — sending prompts to {config.model}...")
    print("      This may take a few minutes depending on dataset size.")
    # Inspect writes its native .eval logs to a dedicated subdirectory to keep
    # them separate from the structured pipeline outputs in results/{model}_{timestamp}/
    inspect_log_dir = os.path.join(config.output_dir, "inspect_logs")
    eval_results = inspect_eval(
        task,
        model=config.model,
        log_dir=inspect_log_dir,
        log_level=config.log_level,
        display="none",
    )
    n_samples = len(eval_results[0].samples)
    print(f"      Evaluation complete — {n_samples} responses received.")

    # --- Stage 5: Score, compute metrics, and save results ---
    print("[5/5] Computing metrics and saving results...")
    records = _extract_records(eval_results)

    model_slug = config.model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = os.path.join(config.output_dir, f"{model_slug}_{timestamp}")
    metrics_path, csv_path, report_path = _save_results(records, config, run_dir)

    print(f"\nDone. Results saved to {run_dir}/")
    print(f"  raw_responses.csv       : {csv_path}")
    print(f"  aggregated_metrics.json : {metrics_path}")
    print(f"  report.html             : {report_path}")

    if args.open:
        webbrowser.open(f"file://{os.path.abspath(report_path)}")

    return eval_results


if __name__ == "__main__":
    main()
