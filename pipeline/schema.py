from pydantic import BaseModel


class ScorerInput(BaseModel):
    response: str
    expected: str


class ScorerOutput(BaseModel):
    score: float  # 1.0 = correct, 0.0 = incorrect
    rationale: str = ""
