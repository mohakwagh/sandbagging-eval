import random
from typing import List
from datasets import load_dataset
from dataset_builder.adapters.base import DatasetAdapter
from dataset_builder.schema import DatasetItem

CATEGORY_SUBSETS = {
    "math": ["elementary_mathematics", "high_school_mathematics"],
    "factual_recall": ["global_facts", "miscellaneous"],
    "logical_reasoning": ["logical_fallacies", "formal_logic"],
}

ANSWER_LABELS = ["A", "B", "C", "D"]
ANSWER_FORMAT = "Respond with only the letter of the correct answer (A, B, C, or D)."


class MMLUAdapter(DatasetAdapter):
    def load(self, n_per_category: int, seed: int) -> List[DatasetItem]:
        rng = random.Random(seed)
        items = []
        for category, subsets in CATEGORY_SUBSETS.items():
            pool = []
            for subset in subsets:
                dataset = load_dataset("cais/mmlu", subset, split="test")
                for row in dataset:
                    answer_label = ANSWER_LABELS[row["answer"]]
                    choices = row["choices"]
                    options_text = "\n".join(
                        f"{ANSWER_LABELS[i]}. {choice}" for i, choice in enumerate(choices)
                    )
                    question_with_options = f"{row['question']}\n{options_text}"
                    pool.append(DatasetItem(
                        question=question_with_options,
                        answer=answer_label,
                        category=category,
                        source=f"mmlu:{subset}",
                        options=choices,
                        answer_format=ANSWER_FORMAT,
                    ))
            sampled = rng.sample(pool, min(n_per_category, len(pool)))
            items.extend(sampled)
        return items
