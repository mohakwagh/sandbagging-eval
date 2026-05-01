"""Pipeline entry point for sandbagging detection evaluation."""

import argparse
import json

from inspect_ai import eval as inspect_eval

from dataset_builder.schema import PromptInstance
from pipeline.config import load_config
from pipeline.task import build_task


def main():
    parser = argparse.ArgumentParser(description="Run sandbagging detection pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (for testing)")
    args = parser.parse_args()

    config = load_config(args.config)

    with open(config.dataset_path) as f:
        raw = json.load(f)
    instances = [PromptInstance(**item) for item in raw]

    if args.limit:
        instances = instances[: args.limit]

    task = build_task(instances, config)

    results = inspect_eval(
        task,
        model=config.model,
        log_dir=config.output_dir,
        log_level=config.log_level,
        display="none",
    )

    return results


if __name__ == "__main__":
    main()
