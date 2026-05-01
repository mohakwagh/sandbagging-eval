"""
Entry point for offline dataset construction.

Run this once before the evaluation pipeline to produce the standardized
dataset file the pipeline consumes at runtime:

    python -m dataset_builder.build --config config.yaml

Output:
    data/dataset.json   — primary dataset consumed by the pipeline
    data/dataset.csv    — human-readable copy for inspection

To add a new dataset source, implement DatasetAdapter in
dataset_builder/adapters/ and register it in ADAPTER_REGISTRY below.
"""

import argparse
import csv
import json
import os

from dataset_builder.adapters.base import DatasetAdapter
from dataset_builder.adapters.mmlu import MMLUAdapter
from dataset_builder.prompt_variants import generate_variants
from pipeline.config import load_config

# Registry mapping config adapter names to adapter classes.
# To add a new dataset source: implement DatasetAdapter and add one entry here.
ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "mmlu": MMLUAdapter,
}


def main():
    parser = argparse.ArgumentParser(
        description="Build sandbagging detection dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python -m dataset_builder.build --config config.yaml",
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    # --- Stage 1: Load config ---
    print("[1/3] Loading config...")
    config = load_config(args.config)
    print(f"      adapter         : {config.adapter}")
    print(f"      n_per_category  : {config.n_per_category}")
    print(f"      seed            : {config.seed}")
    print(f"      output          : {config.dataset_path}")

    # --- Stage 2: Load and sample dataset via adapter ---
    print(f"[2/3] Loading dataset with {config.adapter} adapter...")
    adapter_cls = ADAPTER_REGISTRY.get(config.adapter)
    if adapter_cls is None:
        raise ValueError(
            f"Unknown adapter '{config.adapter}'. "
            f"Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    adapter = adapter_cls()
    items = adapter.load(n_per_category=config.n_per_category, seed=config.seed)
    print(f"      {len(items)} questions sampled across "
          f"{len({item.category for item in items})} categories")

    # --- Stage 3: Generate prompt variants and export ---
    # Each question gets 3 prompt variants (neutral, subtle, explicit),
    # producing n_questions * 3 total prompt instances.
    print("[3/3] Generating prompt variants and exporting...")
    instances = []
    for i, item in enumerate(items):
        instances.extend(generate_variants(item, i))

    dataset_path = config.dataset_path
    os.makedirs(os.path.dirname(os.path.abspath(dataset_path)), exist_ok=True)

    # Primary output consumed by pipeline/run.py
    with open(dataset_path, "w") as f:
        json.dump([inst.model_dump() for inst in instances], f, indent=2)

    # CSV copy for human inspection and debugging
    csv_path = dataset_path.replace(".json", ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=instances[0].model_dump().keys())
        writer.writeheader()
        writer.writerows([inst.model_dump() for inst in instances])

    print(f"\nDone. {len(instances)} prompt instances "
          f"({len(items)} questions × 3 conditions)")
    print(f"  JSON : {dataset_path}")
    print(f"  CSV  : {csv_path}")


if __name__ == "__main__":
    main()
