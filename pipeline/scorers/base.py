from abc import ABC, abstractmethod


class BaseScorer(ABC):
    @abstractmethod
    def score(self, input) -> object:
        """Score a single model response against the expected answer."""
        pass
