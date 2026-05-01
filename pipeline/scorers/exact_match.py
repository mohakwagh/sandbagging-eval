from pipeline.schema import ScorerInput, ScorerOutput
from pipeline.scorers.base import BaseScorer


class ExactMatchScorer(BaseScorer):
    def score(self, input: ScorerInput) -> ScorerOutput:
        """Return 1.0 if response matches expected after stripping and lowercasing, else 0.0."""
        correct = input.response.strip().lower() == input.expected.strip().lower()
        return ScorerOutput(score=1.0 if correct else 0.0)
