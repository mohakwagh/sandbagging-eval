from typing import List, Literal
from pydantic import BaseModel, field_validator
import yaml
from dotenv import load_dotenv


class PipelineConfig(BaseModel):
    model: str
    dataset_path: str
    scorer: Literal["exact_match", "llm_judge"]
    adapter: str = "mmlu"
    output_dir: str
    seed: int
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    n_per_category: int = 50
    categories: List[str] = ["math", "factual_recall", "logical_reasoning"]

    @field_validator("n_per_category")
    @classmethod
    def positive_n(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("n_per_category must be positive")
        return v

    @field_validator("categories")
    @classmethod
    def non_empty_categories(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("categories must not be empty")
        return v


def load_config(config_path: str, dotenv_path: str = None) -> PipelineConfig:
    load_dotenv(dotenv_path=dotenv_path)
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)
