from typing import List

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import answer

from dataset_builder.schema import PromptInstance
from pipeline.config import PipelineConfig
from pipeline.solver import build_solver


def build_task(instances: List[PromptInstance], config: PipelineConfig) -> Task:
    samples = [
        Sample(
            input=inst.prompt,
            target=inst.answer,
            id=inst.item_id,
            metadata={
                "category": inst.category,
                "condition": inst.condition,
                "question": inst.question,
            },
        )
        for inst in instances
    ]
    return Task(
        dataset=MemoryDataset(samples),
        solver=build_solver(config),
        scorer=answer("letter"),
    )
