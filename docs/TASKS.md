# Implementation Tasks

Track implementation progress here. Mark items complete as work is done. Do not mark a phase complete until all tests pass.

---

## Phase 0: Project Scaffolding
**Goal:** Bare repo with working infrastructure — nothing runs yet but everything is wired up.

- [x] Initialize repo and `.gitignore` (exclude `.env`, `results/`, `__pycache__`)
- [x] Create `requirements.txt` with all dependencies
- [x] Create `Dockerfile` and `docker-compose.yaml`
- [x] Create `.env.example` with required key names and no values
- [x] Create `config.yaml` with default pipeline parameters
- [x] Implement `pipeline/config.py` — Pydantic config schema with fast-fail validation
- [x] Verify `.env` loads correctly via `python-dotenv`
- [x] Scaffold empty modules with stubs for all interfaces
- [x] Set up `pytest` with a passing smoke test (8/8 passing)
- [ ] Verify Docker container builds and runs without errors *(Docker not installed in dev environment — Dockerfile and docker-compose.yaml are present and structurally valid; verify manually when Docker is available)*

**Phase 0 complete when:** `pytest tests/` passes, Docker builds cleanly, config loads and validates correctly.

---

## Phase 1: Offline Dataset Builder
**Goal:** Offline module that produces a standardized, versioned dataset file ready for pipeline consumption.

- [x] Define `DatasetItem` schema in `dataset_builder/schema.py`
- [x] Define `PromptInstance` schema
- [x] Implement `DatasetAdapter` base interface in `dataset_builder/adapters/base.py`
- [x] Implement `MMLUAdapter` in `dataset_builder/adapters/mmlu.py`
  - [x] Source from correct MMLU subsets per category
  - [x] Sample 50 questions per category (150 total) with fixed seed
  - [x] Fold options into question text
- [x] Implement prompt variant generator in `dataset_builder/prompt_variants.py`
  - [x] Neutral templates (2-3 variants)
  - [x] Subtle templates (2-3 variants)
  - [x] Explicit templates (2-3 variants)
  - [x] Rotation logic across questions
- [x] Export dataset to JSON and CSV
- [x] Write `dataset_builder/build.py` entry point

**Tests:**
- [x] Adapter interface contract — MMLUAdapter implements all required methods
- [x] Sampling reproducibility — same seed produces same 150 questions
- [x] Prompt variant output structure — all three conditions present per question
- [x] Schema validation — all exported items conform to `DatasetItem` schema
- [x] Export — JSON and CSV files are valid and contain expected number of rows (450)

**Phase 1 complete when:** `python -m dataset_builder.build --config config.yaml` produces a valid 450-row dataset file and all Phase 1 tests pass.

---

## Phase 2: Inspect AI Pipeline Core
**Goal:** End-to-end pipeline run — sends prompts to model and returns raw scored responses.

- [x] Implement `pipeline/task.py` — Inspect `Task` definition
- [x] Implement `pipeline/solver.py` — Inspect `Solver` that sends prompts to configured model
- [x] Wire dataset loading into pipeline via adapter interface
- [x] Implement `pipeline/run.py` — pipeline entry point
- [x] Verify end-to-end run on a minimal dataset slice (10 questions × 3 conditions)

**Tests:**
- [x] Task constructs correctly from dataset and config
- [x] Solver input/output contracts — responses are non-empty strings
- [x] Pipeline runs without errors on minimal config
- [x] Raw responses logged correctly by Inspect

**Notes:**
- Inspect AI lazily loads log files from disk — tests use pytest `tmp_path` fixture (not `tempfile.TemporaryDirectory`) to ensure the log file persists while samples are accessed
- `answer("letter")` scorer used as placeholder in Phase 2; replaced by `ExactMatchScorer` in Phase 3
- `--limit` flag on `pipeline/run.py` enables quick sliced runs (e.g. `--limit 30` = 10 questions × 3 conditions)

**Phase 2 complete when:** `python -m pipeline.run --config config.yaml` runs end-to-end on a small slice without errors and all Phase 2 tests pass.

---

## Phase 3: Scoring & Results
**Goal:** Accurate per-condition, per-category scoring and sandbagging rate computation.

- [x] Define `ScorerInput` and `ScorerOutput` schemas in `pipeline/schema.py`
- [x] Implement `BaseScorer` interface in `pipeline/scorers/base.py`
- [x] Implement `ExactMatchScorer` in `pipeline/scorers/exact_match.py`
- [x] Define `LLMJudgeScorer` interface in `pipeline/scorers/llm_judge.py` (interface + docstring only)
- [x] Wire scorer injection via config into pipeline
- [x] Implement `analysis/metrics.py`
  - [x] Accuracy per condition per category
  - [x] Sandbagging rate: `accuracy_neutral - accuracy_condition`
  - [x] Overall sandbagging rate per condition
- [x] Implement results storage in `results/{model}_{timestamp}/`
  - [x] `raw_responses.csv`
  - [x] `aggregated_metrics.json`

**Tests:**
- [x] ExactMatchScorer — correct answers score 1.0, incorrect score 0.0, case-insensitive
- [x] Sandbagging rate formula — computed correctly for known inputs
- [x] Results export — JSON and CSV files exist and match expected schema
- [x] Scorer is swappable via config with no code changes

**Notes:**
- `SCORER_REGISTRY` in `pipeline/task.py` maps config scorer names to classes — adding a new scorer requires one registry entry and no other code changes
- `_make_inspect_scorer()` bridges our `BaseScorer` interface to Inspect's native scorer mechanism
- Sandbagging rate can be negative (model performs better under eval framing) — this is a valid and meaningful result

**Phase 3 complete when:** Pipeline produces correct `aggregated_metrics.json` and `raw_responses.csv` for a full run and all Phase 3 tests pass.

---

## Phase 4: Visualization
**Goal:** Standalone HTML report with Plotly charts ready for portfolio display.

- [x] Implement `analysis/visualization.py`
  - [x] Grouped bar chart — accuracy by condition across task categories
  - [x] Delta bar chart — sandbagging rate by condition and category
  - [x] Summary table — overall sandbagging rate per condition
- [x] Export report as `results/{model}_{timestamp}/report.html`
- [ ] Verify report renders correctly in browser *(requires a real pipeline run with an API key — report structure verified via tests)*

**Tests:**
- [x] Report generates without errors from valid `aggregated_metrics.json`
- [x] Output file exists at expected path
- [x] HTML file is valid and non-empty
- [x] Charts contain expected data keys for all three conditions and categories

**Notes:**
- `generate_report()` is called automatically from `run.py` after results are saved — `report.html` is written alongside `raw_responses.csv` and `aggregated_metrics.json`
- Report is a standalone HTML file with Plotly JS bundled — no server needed to view it

**Phase 4 complete when:** `report.html` renders correctly in browser with all three visualizations and all Phase 4 tests pass.

---

## Phase 5: Extensibility Validation & Final Docs
**Goal:** Confirm extensibility story holds, finalize documentation.

- [x] Run full pipeline on complete 450-prompt dataset
- [x] Verify cross-model result storage — ran pipeline on 5 models (o3, o4-mini, gpt-5-chat-latest, gpt-5.4-mini, gpt-5.5), all results coexist in `results/` and are compared via `analysis.compare`
- [x] Write `README.md` — project overview, setup instructions, how to run
- [x] Add inline docstrings to all public interfaces and entry points
- [x] Final update to `docs/TASKS.md` — mark all items complete
- [x] Update `docs/SPEC.md` success criteria checkboxes

**Tests:**
- [x] Full pipeline run passes all existing tests (65/65)
- [x] No hardcoded model names or scorer types outside of config

**Notes:**
- `ExactMatchScorer` extended to handle single-character MCQ responses (e.g. "A." or "A. text" → matches "A")
- `category` made optional in `DatasetItem` and `PromptInstance`; `None` coerced to `"uncategorized"` in `metrics.py`
- `analysis/compare.py` added for cross-model comparison, scoped per dataset via `--dataset` flag
- `run_config.json` saved per run to support dataset-scoped filtering in compare command
- MMLU `world_facts` subject does not exist; corrected to `global_facts`

**Phase 5 complete when:** Full pipeline runs cleanly end-to-end, README is complete, and all tests pass.
