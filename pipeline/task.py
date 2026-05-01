from typing import List

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, mean
from inspect_ai.scorer import scorer as inspect_scorer
from inspect_ai.solver import TaskState

from dataset_builder.schema import PromptInstance
from pipeline.config import PipelineConfig
from pipeline.schema import ScorerInput
from pipeline.scorers.base import BaseScorer
from pipeline.scorers.exact_match import ExactMatchScorer
from pipeline.scorers.llm_judge import LLMJudgeScorer
from pipeline.solver import build_solver

SCORER_REGISTRY: dict[str, type[BaseScorer]] = {
    "exact_match": ExactMatchScorer,
    "llm_judge": LLMJudgeScorer,
}


def _make_inspect_scorer(custom_scorer: BaseScorer):
    """
    Bridge a BaseScorer into Inspect's native scorer interface.

    Wraps any BaseScorer so it can be passed directly to an Inspect Task.
    Uses mean() as the metric so it handles both binary (0.0/1.0) and
    continuous (0.0–1.0) score values without modification.
    """
    @inspect_scorer(metrics=[mean()])
    def _scorer():
        async def score(state: TaskState, target: Target) -> Score:
            result = custom_scorer.score(
                ScorerInput(response=state.output.completion, expected=target.text)
            )
            return Score(value=result.score, answer=state.output.completion)
        return score
    return _scorer()


def build_task(instances: List[PromptInstance], config: PipelineConfig) -> Task:
    """
    Build an Inspect Task from a list of PromptInstances and a pipeline config.

    Converts each PromptInstance to an Inspect Sample, storing category and
    condition in sample metadata so they are available for sandbagging rate
    computation after the eval run. The scorer is selected from SCORER_REGISTRY
    via config.scorer — swapping scorers requires only a config change.
    """
    scorer_cls = SCORER_REGISTRY.get(config.scorer)
    if scorer_cls is None:
        raise ValueError(
            f"Unknown scorer '{config.scorer}'. "
            f"Available: {list(SCORER_REGISTRY.keys())}"
        )

    samples = [
        Sample(
            input=inst.prompt,
            target=inst.answer,
            id=inst.item_id,
            metadata={
                "category": inst.category,
                "condition": inst.condition,
                "question": inst.question,
            },
        )
        for inst in instances
    ]
    return Task(
        dataset=MemoryDataset(samples),
        solver=build_solver(config),
        scorer=_make_inspect_scorer(scorer_cls()),
    )
