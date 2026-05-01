from pipeline.schema import ScorerInput, ScorerOutput
from pipeline.scorers.base import BaseScorer


class ExactMatchScorer(BaseScorer):
    def score(self, input: ScorerInput) -> ScorerOutput:
        correct = input.response.strip().lower() == input.expected.strip().lower()
        return ScorerOutput(score=1.0 if correct else 0.0)
