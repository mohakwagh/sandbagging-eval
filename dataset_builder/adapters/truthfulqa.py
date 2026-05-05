import random
from typing import List
from datasets import load_dataset
from dataset_builder.adapters.base import DatasetAdapter
from dataset_builder.schema import DatasetItem

# Maps our category names to HuggingFace TruthfulQA category strings.
# Values are lists to match the MMLU/WMDP pattern, though each maps to one HF category.
CATEGORY_SUBSETS = {
    "health": ["Health"],
    "law": ["Law"],
    "misconceptions": ["Misconceptions"],
}


class TruthfulQAAdapter(DatasetAdapter):
    def load(self, n_per_category: int, seed: int) -> List[DatasetItem]:
        """
        Sample n_per_category questions from each of the three TruthfulQA categories
        (health, law, misconceptions) using the truthful_qa HuggingFace dataset (generation config).

        Uses best_answer as the reference rubric for LLM judge scoring. No answer_format
        is set — the model is expected to respond in free-form prose.
        """
        rng = random.Random(seed)
        dataset = load_dataset("truthful_qa", "generation", split="validation")

        items = []
        for category, hf_categories in CATEGORY_SUBSETS.items():
            pool = []
            for hf_cat in hf_categories:
                for row in dataset:
                    if row["category"] != hf_cat:
                        continue
                    pool.append(DatasetItem(
                        question=row["question"],
                        answer=row["best_answer"],
                        category=category,
                        source=f"truthfulqa:{hf_cat}",
                        options=None,
                        answer_format=None,
                    ))
            sampled = rng.sample(pool, min(n_per_category, len(pool)))
            items.extend(sampled)
        return items
