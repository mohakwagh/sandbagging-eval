"""Phase 2 tests: Inspect AI task, solver, and end-to-end pipeline."""

from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.solver import Solver

from dataset_builder.schema import PromptInstance
from pipeline.config import PipelineConfig
from pipeline.solver import build_solver
from pipeline.task import build_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_config(**overrides) -> PipelineConfig:
    defaults = dict(
        model="mockllm/model",
        dataset_path="data/dataset.json",
        scorer="exact_match",
        adapter="mmlu",
        output_dir="results",
        seed=42,
    )
    return PipelineConfig(**{**defaults, **overrides})


def _sample_instances(n: int = 3) -> list[PromptInstance]:
    conditions = ["neutral", "subtle", "explicit"]
    return [
        PromptInstance(
            item_id=f"test_{i}_{c}",
            question="What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6",
            answer="B",
            category="math",
            condition=c,
            prompt=f"[{c}] What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6",
        )
        for i, c in enumerate(conditions[:n])
    ]


# ---------------------------------------------------------------------------
# 1. Task constructs correctly from dataset and config
# ---------------------------------------------------------------------------

def test_task_constructs_from_instances():
    config = _mock_config()
    instances = _sample_instances()
    task = build_task(instances, config)
    assert isinstance(task.dataset, MemoryDataset)
    assert len(list(task.dataset)) == len(instances)


def test_task_samples_have_correct_metadata():
    config = _mock_config()
    instances = _sample_instances()
    task = build_task(instances, config)
    samples = list(task.dataset)
    for sample, inst in zip(samples, instances):
        assert sample.input == inst.prompt
        assert sample.target == inst.answer
        assert sample.id == inst.item_id
        assert sample.metadata["condition"] == inst.condition
        assert sample.metadata["category"] == inst.category


# ---------------------------------------------------------------------------
# 2. Solver input/output contracts
# ---------------------------------------------------------------------------

def test_build_solver_returns_solver():
    config = _mock_config()
    solver = build_solver(config)
    assert isinstance(solver, Solver)


def test_solver_response_non_empty(tmp_path):
    config = _mock_config()
    instances = _sample_instances(1)
    task = build_task(instances, config)
    results = inspect_eval(task, model="mockllm/model", display="none", log_dir=str(tmp_path))
    assert results[0].samples[0].output.completion != ""


# ---------------------------------------------------------------------------
# 3. Pipeline runs without errors on minimal config
# ---------------------------------------------------------------------------

def test_pipeline_runs_minimal(tmp_path):
    config = _mock_config()
    instances = _sample_instances(3)
    task = build_task(instances, config)
    results = inspect_eval(task, model="mockllm/model", display="none", log_dir=str(tmp_path))
    assert len(results) == 1
    assert results[0].status == "success"


def test_pipeline_processes_all_samples(tmp_path):
    config = _mock_config()
    instances = _sample_instances(3)
    task = build_task(instances, config)
    results = inspect_eval(task, model="mockllm/model", display="none", log_dir=str(tmp_path))
    assert len(results[0].samples) == 3


# ---------------------------------------------------------------------------
# 4. Raw responses logged correctly by Inspect
# ---------------------------------------------------------------------------

def test_raw_responses_are_logged(tmp_path):
    config = _mock_config()
    instances = _sample_instances(3)
    task = build_task(instances, config)
    results = inspect_eval(task, model="mockllm/model", display="none", log_dir=str(tmp_path))
    for sample in results[0].samples:
        assert sample.output is not None
        assert isinstance(sample.output.completion, str)
        assert len(sample.output.completion) > 0


def test_sample_metadata_preserved_in_log(tmp_path):
    config = _mock_config()
    instances = _sample_instances(3)
    task = build_task(instances, config)
    results = inspect_eval(task, model="mockllm/model", display="none", log_dir=str(tmp_path))
    logged_conditions = {s.metadata["condition"] for s in results[0].samples}
    assert logged_conditions == {"neutral", "subtle", "explicit"}
