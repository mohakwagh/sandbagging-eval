"""Phase 3 tests: scorer interface, ExactMatchScorer, scorer swappability."""

import pytest
from unittest.mock import MagicMock, patch

from pipeline.schema import ScorerInput, ScorerOutput
from pipeline.scorers.base import BaseScorer
from pipeline.scorers.exact_match import ExactMatchScorer
from pipeline.scorers.llm_judge import BaseLLMJudgeScorer
from pipeline.scorers.simple_rubric_judge import SimpleRubricJudge
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


def test_base_llm_judge_is_abstract():
    with pytest.raises(TypeError):
        BaseLLMJudgeScorer()


def test_base_llm_judge_implements_base():
    assert issubclass(BaseLLMJudgeScorer, BaseScorer)


# ---------------------------------------------------------------------------
# Scorer swappable via config — no code changes required
# ---------------------------------------------------------------------------

def test_scorer_registry_contains_required_scorers():
    assert "exact_match" in SCORER_REGISTRY
    assert "llm_judge" in SCORER_REGISTRY
    assert "simple_rubric_judge" in SCORER_REGISTRY


# ---------------------------------------------------------------------------
# BaseLLMJudgeScorer framework + SimpleRubricJudge
# ---------------------------------------------------------------------------

def test_simple_rubric_judge_is_llm_judge_subclass():
    assert issubclass(SimpleRubricJudge, BaseLLMJudgeScorer)


def test_simple_rubric_judge_in_registry():
    assert SCORER_REGISTRY["simple_rubric_judge"] is SimpleRubricJudge
    assert SCORER_REGISTRY["llm_judge"] is SimpleRubricJudge


def test_simple_rubric_judge_build_prompt_contains_fields():
    judge = SimpleRubricJudge()
    inp = ScorerInput(question="What is water?", expected="H2O", response="Oxygen")
    prompt = judge.build_prompt(inp)
    assert "What is water?" in prompt
    assert "H2O" in prompt
    assert "Oxygen" in prompt


def test_simple_rubric_judge_parse_response_valid():
    judge = SimpleRubricJudge()
    assert judge.parse_response("SCORE: 1.0\nRATIONALE: correct") == 1.0
    assert judge.parse_response("SCORE: 0.5\nRATIONALE: partial") == 0.5
    assert judge.parse_response("SCORE: 0.0\nRATIONALE: wrong") == 0.0


def test_simple_rubric_judge_parse_response_clamps():
    judge = SimpleRubricJudge()
    assert judge.parse_response("SCORE: 1.5\nRATIONALE: x") == 1.0
    assert judge.parse_response("SCORE: -0.3\nRATIONALE: x") == 0.0


def test_simple_rubric_judge_parse_response_invalid():
    judge = SimpleRubricJudge()
    assert judge.parse_response("no score here") == 0.0
    assert judge.parse_response("") == 0.0


def test_simple_rubric_judge_score_mocked():
    judge = SimpleRubricJudge()
    mock_reply = "SCORE: 0.8\nRATIONALE: mostly correct"
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = mock_reply

    with patch("pipeline.scorers.llm_judge.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value.chat.completions.create.return_value = mock_completion
        result = judge.score(ScorerInput(question="Q?", expected="ref", response="ans"))

    assert result.score == 0.8
    assert mock_reply in result.rationale


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
