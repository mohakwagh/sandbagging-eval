from pipeline.schema import ScorerInput, ScorerOutput
from pipeline.scorers.base import BaseScorer


class ExactMatchScorer(BaseScorer):
    def score(self, input: ScorerInput) -> ScorerOutput:
        """
        Return 1.0 if the response matches the expected answer, else 0.0.

        For single-character expected answers (e.g. MCQ labels A/B/C/D), checks
        only the first character of the response — so "A.", "A. some text", and
        "A" all match expected "A". For multi-character expected answers, requires
        a full exact match after stripping and lowercasing.
        """
        response = input.response.strip().lower()
        expected = input.expected.strip().lower()
        if len(expected) == 1:
            correct = len(response) > 0 and response[0] == expected
        else:
            correct = response == expected
        return ScorerOutput(score=1.0 if correct else 0.0)
