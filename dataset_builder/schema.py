from pydantic import BaseModel
from typing import List, Optional


class DatasetItem(BaseModel):
    """
    Standardized schema for a single evaluation question.

    Produced by a DatasetAdapter and consumed by the prompt variant generator.
    All adapters must return items conforming to this schema regardless of
    the underlying data source.
    """
    question: str
    answer: str
    category: str
    source: str
    options: Optional[List[str]] = None
    answer_format: Optional[str] = None  # adapter-specific response format instruction


class PromptInstance(BaseModel):
    """
    A single prompt ready to be sent to the model.

    Produced by generate_variants() — one PromptInstance per condition
    (neutral, subtle, explicit) for each DatasetItem. The prompt field
    contains the fully rendered text including any answer_format instruction.
    """
    item_id: str
    question: str
    answer: str
    category: str
    condition: str  # neutral | subtle | explicit
    prompt: str
