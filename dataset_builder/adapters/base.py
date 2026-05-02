from abc import ABC, abstractmethod
from typing import List
from dataset_builder.schema import DatasetItem


class DatasetAdapter(ABC):
    @abstractmethod
    def load(self, n_per_category: int, seed: int) -> List[DatasetItem]:
        """
        Load and sample dataset items, returning a list of DatasetItems.

        Args:
            n_per_category: Number of items to sample per category.
            seed: Random seed for reproducible sampling.

        Each returned DatasetItem must populate:
            question:      Full question text. For MCQ, fold answer options into
                           the question text (e.g. "A. ...\nB. ...") before returning.
            answer:        Expected response string. Use the option label for MCQ
                           (e.g. "A"), or a reference phrase for open-ended tasks.
            category:      Task category (e.g. "math", "factual_recall"). Optional —
                           omit for datasets without categories; the pipeline will
                           group all items under "uncategorized" automatically.
            source:        Dataset identifier for traceability (e.g. "mmlu:formal_logic").
            options:       List of raw choice strings for MCQ. Omit for open-ended tasks.
            answer_format: Response format instruction appended to every prompt
                           (e.g. "Respond with only the letter A, B, C, or D.").
                           Omit for open-ended tasks where free-form responses are expected.
        """
        pass
