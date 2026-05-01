from abc import ABC, abstractmethod
from pipeline.schema import ScorerInput, ScorerOutput


class BaseScorer(ABC):
    @abstractmethod
    def score(self, input: ScorerInput) -> ScorerOutput:
        """Score a single model response against the expected answer."""
        pass
