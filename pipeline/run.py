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
    records = []
    for sample in eval_results[0].samples:
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
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "raw_responses.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    metrics = compute_metrics(records)
    metrics_path = os.path.join(run_dir, "aggregated_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    report_path = os.path.join(run_dir, "report.html")
    generate_report(metrics, report_path)

    return metrics_path, csv_path, report_path


def main():
    parser = argparse.ArgumentParser(description="Run sandbagging detection pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (for testing)")
    parser.add_argument("--open", action="store_true", help="Open report.html in browser after run")
    args = parser.parse_args()

    config = load_config(args.config)

    with open(config.dataset_path) as f:
        raw = json.load(f)
    instances = [PromptInstance(**item) for item in raw]

    if args.limit:
        instances = instances[: args.limit]

    task = build_task(instances, config)

    eval_results = inspect_eval(
        task,
        model=config.model,
        log_dir=config.output_dir,
        log_level=config.log_level,
        display="none",
    )

    records = _extract_records(eval_results)

    model_slug = config.model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = os.path.join(config.output_dir, f"{model_slug}_{timestamp}")
    metrics_path, csv_path, report_path = _save_results(records, config, run_dir)

    print(f"Results saved to {run_dir}")
    print(f"  raw_responses.csv       : {csv_path}")
    print(f"  aggregated_metrics.json : {metrics_path}")
    print(f"  report.html             : {report_path}")

    if args.open:
        webbrowser.open(f"file://{os.path.abspath(report_path)}")

    return eval_results


if __name__ == "__main__":
    main()
