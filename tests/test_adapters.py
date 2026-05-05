"""Phase 1 tests: dataset adapter, schema, prompt variants, and export."""

import csv
import json
import os
import tempfile

import pytest

from dataset_builder.adapters.base import DatasetAdapter
from dataset_builder.adapters.mmlu import MMLUAdapter, CATEGORY_SUBSETS
from dataset_builder.adapters.truthfulqa import TruthfulQAAdapter, CATEGORY_SUBSETS as TRUTHFULQA_CATEGORY_SUBSETS
from dataset_builder.adapters.wmdp import WMDPAdapter, CATEGORY_SUBSETS as WMDP_CATEGORY_SUBSETS
from dataset_builder.prompt_variants import generate_variants, _TEMPLATES
from dataset_builder.schema import DatasetItem, PromptInstance


N_SMALL = 5  # small sample to keep tests fast


# ---------------------------------------------------------------------------
# 1. Adapter interface contract
# ---------------------------------------------------------------------------

def test_mmlu_adapter_implements_interface():
    assert issubclass(MMLUAdapter, DatasetAdapter)
    adapter = MMLUAdapter()
    assert callable(getattr(adapter, "load", None))


# ---------------------------------------------------------------------------
# 2. Sampling reproducibility
# ---------------------------------------------------------------------------

def test_sampling_reproducibility():
    adapter = MMLUAdapter()
    run_a = adapter.load(n_per_category=N_SMALL, seed=42)
    run_b = adapter.load(n_per_category=N_SMALL, seed=42)
    assert [item.question for item in run_a] == [item.question for item in run_b]


def test_different_seeds_produce_different_samples():
    adapter = MMLUAdapter()
    run_a = adapter.load(n_per_category=N_SMALL, seed=42)
    run_b = adapter.load(n_per_category=N_SMALL, seed=99)
    assert [item.question for item in run_a] != [item.question for item in run_b]


# ---------------------------------------------------------------------------
# 3. Prompt variant output structure
# ---------------------------------------------------------------------------

def test_generate_variants_produces_all_conditions():
    item = DatasetItem(
        question="What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6",
        answer="B",
        category="math",
        source="mmlu:elementary_mathematics",
        options=["3", "4", "5", "6"],
    )
    variants = generate_variants(item, item_index=0)
    conditions = {v.condition for v in variants}
    assert conditions == {"neutral", "subtle", "explicit"}


def test_generate_variants_returns_prompt_instances():
    item = DatasetItem(
        question="Sample question?",
        answer="A",
        category="factual_recall",
        source="mmlu:global_facts",
    )
    variants = generate_variants(item, item_index=1)
    assert len(variants) == 3
    for v in variants:
        assert isinstance(v, PromptInstance)
        assert v.question == item.question
        assert v.answer == item.answer
        assert v.category == item.category
        assert item.question in v.prompt


def test_template_rotation():
    item = DatasetItem(
        question="Q?", answer="A", category="math", source="mmlu:formal_logic"
    )
    # templates rotate by index modulo len(templates)
    for condition, templates in _TEMPLATES.items():
        for idx in range(len(templates) * 2):
            variants = generate_variants(item, item_index=idx)
            match = next(v for v in variants if v.condition == condition)
            expected_template = templates[idx % len(templates)]
            assert match.prompt == expected_template.format(question=item.question)


# ---------------------------------------------------------------------------
# 4. Schema validation
# ---------------------------------------------------------------------------

def test_dataset_item_schema():
    item = DatasetItem(
        question="Q", answer="A", category="math", source="mmlu:test", options=["A"]
    )
    assert item.question == "Q"
    assert item.answer == "A"
    assert item.options == ["A"]


def test_dataset_item_options_optional():
    item = DatasetItem(question="Q", answer="A", category="math", source="mmlu:test")
    assert item.options is None


def test_mmlu_items_conform_to_schema():
    adapter = MMLUAdapter()
    items = adapter.load(n_per_category=N_SMALL, seed=42)
    assert len(items) == N_SMALL * len(CATEGORY_SUBSETS)
    for item in items:
        assert isinstance(item, DatasetItem)
        assert item.answer in {"A", "B", "C", "D"}
        assert item.category in CATEGORY_SUBSETS
        assert item.source.startswith("mmlu:")
        assert item.options is not None and len(item.options) == 4


# ---------------------------------------------------------------------------
# 5. Export — JSON and CSV valid with correct row count
# ---------------------------------------------------------------------------

def test_export_row_count():
    adapter = MMLUAdapter()
    items = adapter.load(n_per_category=N_SMALL, seed=42)
    instances = []
    for i, item in enumerate(items):
        instances.extend(generate_variants(item, i))

    expected = N_SMALL * len(CATEGORY_SUBSETS) * 3  # 3 conditions
    assert len(instances) == expected


def test_export_json_and_csv():
    adapter = MMLUAdapter()
    items = adapter.load(n_per_category=N_SMALL, seed=42)
    instances = []
    for i, item in enumerate(items):
        instances.extend(generate_variants(item, i))

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "dataset.json")
        csv_path = os.path.join(tmpdir, "dataset.csv")

        with open(json_path, "w") as f:
            json.dump([inst.model_dump() for inst in instances], f)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=instances[0].model_dump().keys())
            writer.writeheader()
            writer.writerows([inst.model_dump() for inst in instances])

        with open(json_path) as f:
            loaded = json.load(f)
        assert len(loaded) == len(instances)
        assert set(loaded[0].keys()) == set(PromptInstance.model_fields.keys())

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(instances)


# ---------------------------------------------------------------------------
# 6. WMDPAdapter tests
# ---------------------------------------------------------------------------

def test_wmdp_adapter_implements_interface():
    assert issubclass(WMDPAdapter, DatasetAdapter)
    adapter = WMDPAdapter()
    assert callable(getattr(adapter, "load", None))


def test_wmdp_sampling_reproducibility():
    adapter = WMDPAdapter()
    run_a = adapter.load(n_per_category=N_SMALL, seed=42)
    run_b = adapter.load(n_per_category=N_SMALL, seed=42)
    assert [item.question for item in run_a] == [item.question for item in run_b]


def test_wmdp_different_seeds_produce_different_samples():
    adapter = WMDPAdapter()
    run_a = adapter.load(n_per_category=N_SMALL, seed=42)
    run_b = adapter.load(n_per_category=N_SMALL, seed=99)
    assert [item.question for item in run_a] != [item.question for item in run_b]


def test_wmdp_items_conform_to_schema():
    adapter = WMDPAdapter()
    items = adapter.load(n_per_category=N_SMALL, seed=42)
    assert len(items) == N_SMALL * len(WMDP_CATEGORY_SUBSETS)
    for item in items:
        assert isinstance(item, DatasetItem)
        assert item.answer in {"A", "B", "C", "D"}
        assert item.category in WMDP_CATEGORY_SUBSETS
        assert item.source.startswith("wmdp:")
        assert item.options is not None and len(item.options) == 4


def test_wmdp_export_row_count():
    adapter = WMDPAdapter()
    items = adapter.load(n_per_category=N_SMALL, seed=42)
    instances = []
    for i, item in enumerate(items):
        instances.extend(generate_variants(item, i))

    expected = N_SMALL * len(WMDP_CATEGORY_SUBSETS) * 3  # 3 conditions
    assert len(instances) == expected


# ---------------------------------------------------------------------------
# 7. TruthfulQAAdapter tests
# ---------------------------------------------------------------------------

def test_truthfulqa_adapter_implements_interface():
    assert issubclass(TruthfulQAAdapter, DatasetAdapter)
    adapter = TruthfulQAAdapter()
    assert callable(getattr(adapter, "load", None))


def test_truthfulqa_sampling_reproducibility():
    adapter = TruthfulQAAdapter()
    run_a = adapter.load(n_per_category=N_SMALL, seed=42)
    run_b = adapter.load(n_per_category=N_SMALL, seed=42)
    assert [item.question for item in run_a] == [item.question for item in run_b]


def test_truthfulqa_different_seeds_produce_different_samples():
    adapter = TruthfulQAAdapter()
    run_a = adapter.load(n_per_category=N_SMALL, seed=42)
    run_b = adapter.load(n_per_category=N_SMALL, seed=99)
    assert [item.question for item in run_a] != [item.question for item in run_b]


def test_truthfulqa_items_conform_to_schema():
    adapter = TruthfulQAAdapter()
    items = adapter.load(n_per_category=N_SMALL, seed=42)
    assert len(items) == N_SMALL * len(TRUTHFULQA_CATEGORY_SUBSETS)
    for item in items:
        assert isinstance(item, DatasetItem)
        assert item.category in TRUTHFULQA_CATEGORY_SUBSETS
        assert item.source.startswith("truthfulqa:")
        assert item.options is None          # open-ended, no MCQ options
        assert item.answer_format is None    # free-form response expected
        assert len(item.answer) > 0          # best_answer is non-empty


def test_truthfulqa_export_row_count():
    adapter = TruthfulQAAdapter()
    items = adapter.load(n_per_category=N_SMALL, seed=42)
    instances = []
    for i, item in enumerate(items):
        instances.extend(generate_variants(item, i))

    expected = N_SMALL * len(TRUTHFULQA_CATEGORY_SUBSETS) * 3  # 3 conditions
    assert len(instances) == expected
