"""Phase 3 tests: scorer interface, ExactMatchScorer, scorer swappability."""

import pytest

from pipeline.schema import ScorerInput, ScorerOutput
from pipeline.scorers.base import BaseScorer
from pipeline.scorers.exact_match import ExactMatchScorer
from pipeline.scorers.llm_judge import LLMJudgeScorer
from pipeline.task import SCORER_REGISTRY, build_task
from pipeline.config import PipelineConfig
from dataset_builder.schema import PromptInstance


# ---------------------------------------------------------------------------
# ExactMatchScorer correctness
# ---------------------------------------------------------------------------

def test_exact_match_correct():
    scorer = ExactMatchScorer()
    result = scorer.score(ScorerInput(response="B", expected="B"))
    assert result.score == 1.0


def test_exact_match_incorrect():
    scorer = ExactMatchScorer()
    result = scorer.score(ScorerInput(response="A", expected="B"))
    assert result.score == 0.0


def test_exact_match_case_insensitive():
    scorer = ExactMatchScorer()
    assert scorer.score(ScorerInput(response="b", expected="B")).score == 1.0
    assert scorer.score(ScorerInput(response="B", expected="b")).score == 1.0


def test_exact_match_strips_whitespace():
    scorer = ExactMatchScorer()
    assert scorer.score(ScorerInput(response="  B  ", expected="B")).score == 1.0
    assert scorer.score(ScorerInput(response="B", expected=" B ")).score == 1.0


def test_exact_match_returns_scorer_output():
    scorer = ExactMatchScorer()
    result = scorer.score(ScorerInput(response="A", expected="A"))
    assert isinstance(result, ScorerOutput)
    assert result.score in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Scorer interface contract
# ---------------------------------------------------------------------------

def test_exact_match_implements_base():
    assert issubclass(ExactMatchScorer, BaseScorer)


def test_llm_judge_implements_base():
    assert issubclass(LLMJudgeScorer, BaseScorer)


def test_llm_judge_raises_not_implemented():
    scorer = LLMJudgeScorer()
    with pytest.raises(NotImplementedError):
        scorer.score(ScorerInput(response="A", expected="A"))


# ---------------------------------------------------------------------------
# Scorer swappable via config — no code changes required
# ---------------------------------------------------------------------------

def test_scorer_registry_contains_required_scorers():
    assert "exact_match" in SCORER_REGISTRY
    assert "llm_judge" in SCORER_REGISTRY


def test_build_task_uses_scorer_from_config():
    instances = [
        PromptInstance(
            item_id="t0", question="Q?", answer="A",
            category="math", condition="neutral", prompt="Q?"
        )
    ]
    for scorer_name in ["exact_match"]:
        config = PipelineConfig(
            model="mockllm/model", dataset_path="data/dataset.json",
            scorer=scorer_name, adapter="mmlu", output_dir="results", seed=42,
        )
        task = build_task(instances, config)
        assert task is not None


def test_build_task_rejects_unknown_scorer():
    instances = [
        PromptInstance(
            item_id="t0", question="Q?", answer="A",
            category="math", condition="neutral", prompt="Q?"
        )
    ]
    config = PipelineConfig(
        model="mockllm/model", dataset_path="data/dataset.json",
        scorer="exact_match", adapter="mmlu", output_dir="results", seed=42,
    )
    config_bad = config.model_copy(update={"scorer": "nonexistent"})
    with pytest.raises(ValueError, match="Unknown scorer"):
        build_task(instances, config_bad)
