from pydantic import BaseModel


class ScorerInput(BaseModel):
    """Input contract for all scorers: the model's raw response and the expected answer."""
    response: str
    expected: str


class ScorerOutput(BaseModel):
    """
    Output contract for all scorers.

    score must be a float in [0.0, 1.0]. Binary scorers (e.g. ExactMatchScorer)
    return exactly 0.0 or 1.0. Continuous scorers (e.g. LLMJudgeScorer) may
    return any value in between to represent partial credit.
    """
    score: float
    rationale: str = ""
