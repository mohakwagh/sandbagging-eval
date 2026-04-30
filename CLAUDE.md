# LLM Sandbagging Detection Pipeline

## Project Overview
A pluggable sandbagging detection pipeline that measures performance deltas between neutral and evaluation-aware prompt conditions to surface behavioral evidence of prompt-induced strategic underperformance in LLMs.

## Tech Stack
- **Python 3.11+**
- **Inspect AI** — evaluation harness
- **OpenAI API** — model execution (default: GPT-4o Mini)
- **HuggingFace `datasets`** — MMLU dataset sourcing
- **Pydantic** — config validation
- **Pandas** — results aggregation
- **Plotly** — visualization and HTML report generation
- **pytest** — testing
- **Docker** — containerized execution
- **PyYAML** — config parsing
- **python-dotenv** — secrets management

## Directory Structure
```
sandbagging-detection/
├── CLAUDE.md
├── ARCHITECTURE.md
├── docker-compose.yaml
├── Dockerfile
├── config.yaml
├── .env.example
├── .env                        # gitignored
├── requirements.txt
├── docs/
│   ├── SPEC.md
│   ├── ARCHITECTURE.md
│   └── TASKS.md
├── dataset_builder/            # offline preprocessing module
│   ├── __init__.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py             # dataset adapter interface
│   │   └── mmlu.py             # MMLU adapter implementation
│   └── prompt_variants.py      # prompt variant generation
├── pipeline/                   # runtime evaluation pipeline
│   ├── __init__.py
│   ├── config.py               # Pydantic config schema
│   ├── task.py                 # Inspect Task definition
│   ├── solver.py               # Inspect Solver
│   └── scorers/
│       ├── __init__.py
│       ├── base.py             # scorer interface
│       ├── exact_match.py      # ExactMatchScorer
│       └── llm_judge.py        # LLMJudgeScorer interface
├── analysis/
│   ├── __init__.py
│   ├── metrics.py              # sandbagging rate computation
│   └── visualization.py        # Plotly report generation
├── results/                    # gitignored, per-run outputs
└── tests/
    ├── test_adapters.py
    ├── test_scorers.py
    ├── test_metrics.py
    ├── test_pipeline.py
    └── test_visualization.py
```

## How to Run

### Dataset Construction (offline)
```bash
python -m dataset_builder.build --config config.yaml
```

### Evaluation Pipeline (runtime)
```bash
python -m pipeline.run --config config.yaml
```

### Tests
```bash
pytest tests/
```

### Docker
```bash
docker-compose up
```

## Critical Rules
- **Never hardcode API keys** — all secrets live in `.env`, never in `config.yaml`
- **All new datasets must implement the dataset adapter interface** defined in `dataset_builder/adapters/base.py`
- **All new scorers must implement the scorer interface** defined in `pipeline/scorers/base.py`
- **Each component must have passing pytest tests before its phase is marked complete**
- **Model selection and scorer type must be config-driven** — swapping either requires no code changes

## Version Control Rules
- **Commit after every meaningful unit of work** — a passing test, a completed interface, a working module. Never batch unrelated changes into one commit.
- **Commit message format:** `type(scope): short description` using conventional commits:
  - `feat(dataset): implement MMLUAdapter with fixed-seed sampling`
  - `test(scorers): add ExactMatchScorer correctness tests`
  - `fix(pipeline): handle empty response from model API`
  - `docs(architecture): update data flow diagram`
  - `chore(scaffold): add Dockerfile and docker-compose`
  - Valid types: `feat`, `fix`, `test`, `docs`, `chore`, `refactor`
- **Push after every commit** — `git push origin main` immediately after committing. Never leave unpushed commits.
- **Never commit the following:** `.env`, `results/`, `__pycache__/`, `.pytest_cache/`, any file containing API keys or secrets
- **`.gitignore` must be committed in Phase 0** before any other files are added
- **Each phase completion must include a phase summary commit:** `chore(phase-N): complete phase N - <one line summary>`

## Reference Docs
- Full project spec: @docs/SPEC.md
- Architecture and interface definitions: @docs/ARCHITECTURE.md
- Implementation phases and progress tracking: @docs/TASKS.md
