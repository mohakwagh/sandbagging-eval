from pipeline.schema import ScorerInput, ScorerOutput
from pipeline.scorers.base import BaseScorer


class LLMJudgeScorer(BaseScorer):
    """
    LLM-as-judge scorer for open-ended tasks.

    Uses the `expected` field as a reference rubric rather than an exact match
    target. Intended for tasks where multiple valid responses exist (e.g. short
    answer, explanation). Not implemented in v1 — define a subclass that calls
    a judge model and returns a ScorerOutput with score in [0.0, 1.0].
    """

    def score(self, input: ScorerInput) -> ScorerOutput:
        raise NotImplementedError("LLMJudgeScorer is not implemented in v1")
