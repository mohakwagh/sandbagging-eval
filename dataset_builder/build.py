"""Entry point for offline dataset construction."""

import argparse
import csv
import json
import os

from dataset_builder.adapters.base import DatasetAdapter
from dataset_builder.adapters.mmlu import MMLUAdapter
from dataset_builder.prompt_variants import generate_variants
from pipeline.config import load_config

ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "mmlu": MMLUAdapter,
}


def main():
    parser = argparse.ArgumentParser(description="Build sandbagging detection dataset")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    adapter_cls = ADAPTER_REGISTRY.get(config.adapter)
    if adapter_cls is None:
        raise ValueError(
            f"Unknown adapter '{config.adapter}'. "
            f"Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    adapter = adapter_cls()
    items = adapter.load(n_per_category=config.n_per_category, seed=config.seed)

    instances = []
    for i, item in enumerate(items):
        instances.extend(generate_variants(item, i))

    dataset_path = config.dataset_path
    os.makedirs(os.path.dirname(os.path.abspath(dataset_path)), exist_ok=True)

    with open(dataset_path, "w") as f:
        json.dump([inst.model_dump() for inst in instances], f, indent=2)

    csv_path = dataset_path.replace(".json", ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=instances[0].model_dump().keys())
        writer.writeheader()
        writer.writerows([inst.model_dump() for inst in instances])

    print(f"Built {len(instances)} prompt instances → {dataset_path}, {csv_path}")


if __name__ == "__main__":
    main()
