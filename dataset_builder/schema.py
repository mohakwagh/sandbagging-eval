from pydantic import BaseModel
from typing import List, Optional


class DatasetItem(BaseModel):
    question: str
    answer: str
    category: str
    source: str
    options: Optional[List[str]] = None
    answer_format: Optional[str] = None  # adapter-specific response format instruction


class PromptInstance(BaseModel):
    item_id: str
    question: str
    answer: str
    category: str
    condition: str  # neutral | subtle | explicit
    prompt: str
