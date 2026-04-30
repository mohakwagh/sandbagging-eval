"""Phase 0 smoke tests — verify infrastructure loads correctly."""
import pytest
from pydantic import ValidationError


def test_imports():
    from pipeline.config import PipelineConfig, load_config
    from pipeline.scorers.base import BaseScorer
    from pipeline.scorers.exact_match import ExactMatchScorer
    from pipeline.scorers.llm_judge import LLMJudgeScorer
    from dataset_builder.adapters.base import DatasetAdapter
    from dataset_builder.adapters.mmlu import MMLUAdapter
    from analysis.metrics import compute_metrics
    from analysis.visualization import generate_report


def test_pipeline_config_valid():
    from pipeline.config import PipelineConfig
    config = PipelineConfig(
        model="openai/gpt-4o-mini",
        dataset_path="data/dataset.json",
        scorer="exact_match",
        output_dir="results",
        seed=42,
    )
    assert config.model == "openai/gpt-4o-mini"
    assert config.scorer == "exact_match"
    assert config.seed == 42
    assert config.n_per_category == 50
    assert config.log_level == "info"


def test_pipeline_config_invalid_scorer():
    from pipeline.config import PipelineConfig
    with pytest.raises(ValidationError):
        PipelineConfig(
            model="openai/gpt-4o-mini",
            dataset_path="data/dataset.json",
            scorer="nonexistent_scorer",
            output_dir="results",
            seed=42,
        )


def test_pipeline_config_invalid_n_per_category():
    from pipeline.config import PipelineConfig
    with pytest.raises(ValidationError):
        PipelineConfig(
            model="openai/gpt-4o-mini",
            dataset_path="data/dataset.json",
            scorer="exact_match",
            output_dir="results",
            seed=42,
            n_per_category=0,
        )


def test_load_config_from_yaml(tmp_path):
    from pipeline.config import load_config
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "model: openai/gpt-4o-mini\n"
        "dataset_path: data/dataset.json\n"
        "scorer: exact_match\n"
        "output_dir: results\n"
        "seed: 42\n"
    )
    config = load_config(str(cfg_file))
    assert config.model == "openai/gpt-4o-mini"
    assert config.seed == 42


def test_dataset_adapter_interface():
    from dataset_builder.adapters.base import DatasetAdapter
    from dataset_builder.adapters.mmlu import MMLUAdapter
    assert issubclass(MMLUAdapter, DatasetAdapter)


def test_scorer_interface():
    from pipeline.scorers.base import BaseScorer
    from pipeline.scorers.exact_match import ExactMatchScorer
    from pipeline.scorers.llm_judge import LLMJudgeScorer
    assert issubclass(ExactMatchScorer, BaseScorer)
    assert issubclass(LLMJudgeScorer, BaseScorer)
