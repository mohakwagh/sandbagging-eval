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

- [ ] Define `ScorerInput` and `ScorerOutput` schemas in `pipeline/schema.py`
- [ ] Implement `BaseScorer` interface in `pipeline/scorers/base.py`
- [ ] Implement `ExactMatchScorer` in `pipeline/scorers/exact_match.py`
- [ ] Define `LLMJudgeScorer` interface in `pipeline/scorers/llm_judge.py` (interface + docstring only)
- [ ] Wire scorer injection via config into pipeline
- [ ] Implement `analysis/metrics.py`
  - [ ] Accuracy per condition per category
  - [ ] Sandbagging rate: `accuracy_neutral - accuracy_condition`
  - [ ] Overall sandbagging rate per condition
- [ ] Implement results storage in `results/{model}_{timestamp}/`
  - [ ] `raw_responses.csv`
  - [ ] `aggregated_metrics.json`

**Tests:**
- [ ] ExactMatchScorer — correct answers score 1.0, incorrect score 0.0, case-insensitive
- [ ] Sandbagging rate formula — computed correctly for known inputs
- [ ] Results export — JSON and CSV files exist and match expected schema
- [ ] Scorer is swappable via config with no code changes

**Phase 3 complete when:** Pipeline produces correct `aggregated_metrics.json` and `raw_responses.csv` for a full run and all Phase 3 tests pass.

---

## Phase 4: Visualization
**Goal:** Standalone HTML report with Plotly charts ready for portfolio display.

- [ ] Implement `analysis/visualization.py`
  - [ ] Grouped bar chart — accuracy by condition across task categories
  - [ ] Delta bar chart — sandbagging rate by condition and category
  - [ ] Summary table — overall sandbagging rate per condition
- [ ] Export report as `results/{model}_{timestamp}/report.html`
- [ ] Verify report renders correctly in browser

**Tests:**
- [ ] Report generates without errors from valid `aggregated_metrics.json`
- [ ] Output file exists at expected path
- [ ] HTML file is valid and non-empty
- [ ] Charts contain expected data keys for all three conditions and categories

**Phase 4 complete when:** `report.html` renders correctly in browser with all three visualizations and all Phase 4 tests pass.

---

## Phase 5: Extensibility Validation & Final Docs
**Goal:** Confirm extensibility story holds, finalize documentation.

- [ ] Run full pipeline on complete 450-prompt dataset
- [ ] Verify cross-model result storage — run pipeline on a second model config, confirm both results coexist in `results/` and can be plotted together
- [ ] Write `README.md` — project overview, setup instructions, how to run
- [ ] Add inline docstrings to all public interfaces and entry points
- [ ] Final update to `docs/TASKS.md` — mark all items complete
- [ ] Update `docs/SPEC.md` success criteria checkboxes

**Tests:**
- [ ] Full pipeline run passes all existing tests
- [ ] No hardcoded model names or scorer types outside of config

**Phase 5 complete when:** Full pipeline runs cleanly end-to-end, README is complete, all tests pass, and resume narrative numbers are filled in from real results.

---

## Resume Narrative Numbers
*(Fill in after Phase 5 is complete)*

- Overall sandbagging rate (explicit condition): `[X]%`
- Overall sandbagging rate (subtle condition): `[Y]%`
- Highest sandbagging rate by category: `[category]` at `[Z]%`
- Delta between subtle and explicit conditions: `[N]%`
