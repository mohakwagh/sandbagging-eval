from inspect_ai.solver import generate, Solver
from pipeline.config import PipelineConfig


def build_solver(config: PipelineConfig) -> Solver:
    """
    Return an Inspect Solver for the configured model.

    The solver is intentionally thin — it delegates all model API calls,
    retries, and rate limiting to Inspect's generate() built-in. Dataset-
    specific interpretation (e.g. extracting a letter answer) is handled
    entirely at the scoring layer, keeping the solver dataset-agnostic.
    """
    return generate()
