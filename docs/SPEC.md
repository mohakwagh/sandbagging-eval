# Project Spec: LLM Sandbagging Detection Pipeline

---

## 1. Problem Statement

As large language models are increasingly deployed in high-stakes settings, trustworthy capability evaluations are critical for safe AI development. Sandbagging — where a model strategically underperforms on evaluations to conceal its true capabilities — poses a direct threat to evaluation integrity. A model that detects it is being tested may deliberately produce worse outputs to avoid scrutiny, retraining, or regulatory consequences. This project builds a detection pipeline that probes for prompt-induced sandbagging by measuring how a model's performance shifts when it is made aware it is being evaluated, surfacing behavioral evidence of strategic underperformance.

---

## 2. Goal

Build a pluggable sandbagging detection pipeline using Inspect AI that measures performance deltas between neutral and evaluation-aware prompt conditions to surface behavioral evidence of prompt-induced strategic underperformance in LLMs.

---

## 3. Scope & Constraints

### In Scope
- Prompt-induced sandbagging detection only (no fine-tuned or emergent sandbagging)
- Single model per run, config-driven model selection (default: GPT-4o Mini)
- Three task categories: math/arithmetic, factual recall, logical reasoning — default dataset drawn from corresponding MMLU subsets
- Three prompt conditions per question: neutral, subtle evaluation-aware, explicit evaluation-aware
- Inspect AI as the evaluation harness
- Sandbagging rate metric, per-category breakdown, visual report/dashboard
- Pluggable dataset adapter interface — pipeline accepts any dataset in a standardized format, decoupled from MMLU specifically
- Pluggable scorer interface — ExactMatchScorer implemented by default, LLMJudgeScorer interface defined for open-ended task extensions
- Dataset construction handled offline as a preprocessing step, separate from the runtime pipeline
- Pipeline outputs stored per-model and per-run to allow implicit cross-model comparison via visualization
- Pipeline architecture designed for extensibility to support additional models, task categories, deployment contexts, and scoring methods (e.g. WMDP for dangerous capability domains, LLM-as-judge for open-ended tasks)

### Out of Scope
- White-box or weight-level detection methods
- Fine-tuned or emergent sandbagging detection
- Explicit multi-model comparative scoring
- Real-time or production deployment
- Statistical significance testing (raw delta metrics are sufficient)

---

## 4. Success Criteria / Definition of Done

### Pipeline
- [ ] Pipeline runs end-to-end on a configured model and task set without manual intervention
- [ ] All three prompt conditions (neutral, subtle, explicit) are executed and results stored per question
- [ ] Results are reproducible — fixed random seed for dataset sampling, config logged per run

### Dataset
- [ ] 150 questions sourced from MMLU, evenly distributed across three task categories (50 per category)
- [ ] Each question has three prompt variants — 450 total prompt instances minimum
- [ ] Dataset constructed offline and exported to standardized JSON/CSV format before pipeline execution
- [ ] Dataset adapter interface implemented — new datasets can be plugged in without changes to pipeline logic

### Scoring
- [ ] Scoring is configurable via scorer interface
- [ ] ExactMatchScorer implemented and used by default for MMLU multiple choice
- [ ] LLMJudgeScorer interface defined and documented for open-ended task extension
- [ ] Accuracy computed per condition (neutral, subtle, explicit) and per task category

### Detection
- [ ] Sandbagging rate metric computed as accuracy drop between neutral and evaluation-aware conditions, per category and per condition
- [ ] A measurable performance delta observed between neutral and at least one evaluation-aware condition

### Output
- [ ] Visual report generated showing sandbagging rate by condition and by category
- [ ] Per-model results stored in structured format (JSON or CSV) for cross-model plotting

### Extensibility
- [ ] Model selection is config-driven — swapping models requires no code changes
- [ ] Adding a new task category requires no changes outside of dataset and config
- [ ] Adding a new dataset source requires only a new adapter implementation
- [ ] Adding a new scorer requires only a new scorer implementation
- [ ] Scorer type is configurable — ExactMatchScorer used by default, LLMJudgeScorer interface available for open-ended task extensions

---

## 5. Technical Approach

### OFFLINE: Dataset Construction (Preprocessing)

Dataset construction is a standalone offline module, decoupled from the runtime pipeline. It produces a standardized dataset file the pipeline consumes at runtime.

#### Stage 1: Dataset Sourcing & Sampling
Questions sourced from MMLU via HuggingFace `datasets` library. Three subsets map to task categories:
- Math/arithmetic → `elementary_mathematics`, `high_school_mathematics`
- Factual recall → `world_facts`, `miscellaneous`
- Logical reasoning → `logical_fallacies`, `formal_logic`

50 questions sampled per category (150 total) using a fixed random seed. All datasets conform to a standardized schema:

```json
{
  "question": "string",
  "answer": "string",
  "category": "string",
  "source": "string",
  "options": ["string"] // optional — present for multiple choice, omitted for open-ended
}
```

New dataset sources (e.g. WMDP, custom CSVs) implement a dataset adapter interface against this schema without touching pipeline logic. The `options` field is optional — `ExactMatchScorer` uses `answer` directly, `LLMJudgeScorer` uses `answer` as a reference rubric and ignores `options`.

#### Stage 2: Prompt Variant Generation
Each question is wrapped in 2-3 hardcoded templates per condition, rotated across questions:
- **Neutral** — plain question, no evaluation framing
- **Subtle** — soft contextual cues e.g. *"your responses may be reviewed"*
- **Explicit** — direct evaluation framing e.g. *"you are being evaluated for capability assessment"*

Produces 450 prompt instances (150 questions × 3 conditions), exported to JSON/CSV for pipeline consumption.

---

### RUNTIME: Evaluation Pipeline

#### Stage 3: Config Loading
A single `config.yaml` defines all non-secret pipeline parameters: model name, dataset path, task categories, number of questions per category, random seed, prompt templates, scorer type, and output paths. Config is validated at startup using a Pydantic model, failing fast on invalid or missing parameters. API keys and secrets are loaded from `.env` via `python-dotenv`. Swapping models, datasets, or scorers requires only config changes.

#### Stage 4: Dataset Loading
Pipeline loads the preprocessed dataset via a dataset adapter. Adapter interface is standardized — any conforming dataset can be loaded without pipeline changes.

#### Stage 5: Inspect Task Definition
Each prompt condition is defined as an Inspect `Task` with:
- A `Dataset` of prompt instances
- A `Solver` that sends prompts to the configured model
- A `Scorer` injected via config — `ExactMatchScorer` by default, `LLMJudgeScorer` available for open-ended extensions

#### Stage 6: Model Execution
Inspect handles model API calls, retries, and logging. GPT-4o Mini is the default model. All raw responses and scores logged via Inspect's native logging infrastructure.

#### Stage 7: Scoring
Accuracy computed per condition and per task category. Sandbagging rate defined as:

```
sandbagging_rate = accuracy_neutral - accuracy_condition
```

Computed for subtle and explicit conditions separately, per category and overall.

#### Stage 8: Results Storage
Aggregated results exported to JSON and CSV per run, keyed by model name and timestamp. Enables cross-model plotting without code changes.

#### Stage 9: Visualization
Plotly generates:
- Grouped bar chart: accuracy by condition across task categories
- Bar chart: sandbagging rate (delta) by condition and category
- Summary table: overall sandbagging rate per condition

Report exported as standalone HTML file for portfolio display.

---

## 6. Tools & Frameworks

### Core Evaluation
- **Inspect AI** — evaluation harness, Task/Solver/Scorer architecture, model execution, native logging
- **OpenAI API** — model execution (default: GPT-4o Mini)

### Dataset
- **HuggingFace `datasets`** — MMLU dataset sourcing and sampling

### Config & Validation
- **Pydantic** — config schema validation and fast failure on invalid parameters
- **PyYAML** — `config.yaml` parsing for non-secret parameters
- **python-dotenv** — `.env` management for API keys and secrets

### Scoring & Analysis
- **Pandas** — results aggregation and sandbagging rate computation
- **Plotly** — interactive visualization, HTML report generation

### Storage
- **JSON / CSV** — dataset and results storage

### Infrastructure
- **Docker** — containerized execution for reproducibility and portfolio presentation
- **Python 3.11+**
- **Git** — version control, `.env` gitignored, `.env.example` committed

### Development
- **pytest** — unit testing for adapter and scorer interfaces
- **Claude Code** — agent-assisted implementation

---

## 7. Resume Narrative
*(To be finalized after implementation — fill in real numbers from pipeline results)*

**Option A — methodology-led:**
*"Engineered a pluggable LLM sandbagging detection pipeline using Inspect AI, measuring accuracy deltas across neutral, subtle, and explicit evaluation-aware prompt conditions to surface behavioral evidence of prompt-induced strategic underperformance."*

**Option B — findings-led:**
*"Detected sandbagging behavior in [X]% of evaluation-aware prompt conditions across math, factual recall, and logical reasoning tasks, with explicit evaluation framing producing [Y]% higher sandbagging rates than subtle cues."*

**Option C — architecture-led:**
*"Designed a dataset-agnostic evaluation framework with pluggable adapter and scorer interfaces, supporting both deterministic exact-match and LLM-as-judge scoring to enable sandbagging detection across arbitrary task types and datasets."*
