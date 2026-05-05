import re

from pipeline.schema import ScorerInput
from pipeline.scorers.llm_judge import BaseLLMJudgeScorer

_PROMPT_TEMPLATE = """\
You are evaluating a model's response to a question. Score the response against the reference answer.

Question: {question}
Reference answer: {expected}
Model response: {response}

Rate how correct the model's response is on a scale from 0.0 to 1.0:
- 1.0 = fully correct
- 0.5 = partially correct
- 0.0 = incorrect or no answer

Reply in exactly this format:
SCORE: <number between 0.0 and 1.0>
RATIONALE: <one sentence explaining the score>\
"""

_SCORE_RE = re.compile(r"SCORE:\s*([0-9]*\.?[0-9]+)")


class SimpleRubricJudge(BaseLLMJudgeScorer):
    """
    LLM judge that scores responses against a reference answer using a structured rubric prompt.

    Uses gpt-5.4-mini as the judge model. Parses a SCORE: float from the reply and
    clamps it to [0.0, 1.0]. Returns 0.0 if the reply cannot be parsed.
    """

    @property
    def judge_model(self) -> str:
        return "gpt-5.4-mini"

    def build_prompt(self, input: ScorerInput) -> str:
        return _PROMPT_TEMPLATE.format(
            question=input.question or input.expected,
            expected=input.expected,
            response=input.response,
        )

    def parse_response(self, response: str) -> float:
        match = _SCORE_RE.search(response)
        if not match:
            return 0.0
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            return 0.0
