from abc import ABC, abstractmethod
from typing import List


class DatasetAdapter(ABC):
    @abstractmethod
    def load(self, n_per_category: int, seed: int) -> List:
        """Load and sample dataset items, returning a list of DatasetItems."""
        pass
