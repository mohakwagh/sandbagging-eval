from typing import List
from dataset_builder.adapters.base import DatasetAdapter


class MMLUAdapter(DatasetAdapter):
    def load(self, n_per_category: int, seed: int) -> List:
        raise NotImplementedError("Phase 1: MMLUAdapter not yet implemented")
