from pipeline.scorers.base import BaseScorer


class ExactMatchScorer(BaseScorer):
    def score(self, input) -> object:
        raise NotImplementedError("Phase 3: ExactMatchScorer not yet implemented")
