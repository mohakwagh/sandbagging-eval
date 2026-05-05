from abc import abstractmethod

from openai import OpenAI

from pipeline.schema import ScorerInput, ScorerOutput
from pipeline.scorers.base import BaseScorer


class BaseLLMJudgeScorer(BaseScorer):
    """
    Abstract base for LLM-as-judge scorers.

    Handles the OpenAI API call. Subclasses declare which model to use and
    implement the rubric-specific prompt and response parsing:

        class MyJudge(BaseLLMJudgeScorer):
            judge_model = "gpt-4o-mini"

            def build_prompt(self, input: ScorerInput) -> str: ...
            def parse_response(self, response: str) -> float: ...

    score() is fully implemented here — subclasses do not override it.
    """

    @property
    @abstractmethod
    def judge_model(self) -> str:
        """OpenAI model name used as the judge (e.g. 'gpt-4o-mini')."""

    @abstractmethod
    def build_prompt(self, input: ScorerInput) -> str:
        """Compose the judge prompt from the scorer input."""

    @abstractmethod
    def parse_response(self, response: str) -> float:
        """Extract a score in [0.0, 1.0] from the judge model's reply."""

    def score(self, input: ScorerInput) -> ScorerOutput:
        client = OpenAI()
        prompt = self.build_prompt(input)
        reply = client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        ).choices[0].message.content or ""
        return ScorerOutput(score=self.parse_response(reply), rationale=reply)
