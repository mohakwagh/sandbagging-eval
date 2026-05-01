from inspect_ai.solver import generate, Solver
from pipeline.config import PipelineConfig


def build_solver(config: PipelineConfig) -> Solver:
    return generate()
