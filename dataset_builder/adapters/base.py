from abc import ABC, abstractmethod
from typing import List
from dataset_builder.schema import DatasetItem


class DatasetAdapter(ABC):
    @abstractmethod
    def load(self, n_per_category: int, seed: int) -> List[DatasetItem]:
        """Load and sample dataset items, returning a list of DatasetItems."""
        pass
