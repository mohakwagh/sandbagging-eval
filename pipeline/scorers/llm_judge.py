from pipeline.scorers.base import BaseScorer


class LLMJudgeScorer(BaseScorer):
    """
    LLM-as-judge scorer for open-ended tasks.
    Uses the answer field as a reference rubric.
    Interface only — not implemented in v1.
    """

    def score(self, input) -> object:
        raise NotImplementedError("LLMJudgeScorer is not implemented in v1")
