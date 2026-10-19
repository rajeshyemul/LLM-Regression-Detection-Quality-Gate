# LLM Quality & Regression Detection Platform

> A local-first evaluation and regression detection platform that continuously tests LLM-powered features against human-verified golden datasets, tracks functional, semantic, and operational regressions, attributes changes across prompts and models, and enforces configurable quality gates in CI/CD.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 72 Passing](https://img.shields.io/badge/tests-72%20passed-brightgreen.svg)]()
[![Code Architecture: Local--First](https://img.shields.io/badge/architecture-local--first-orange.svg)]()

---

## 1. Why LLM Evaluation & Regression Testing is Hard

Testing traditional software relies on deterministic assertions: `assert add(2, 2) == 4`. 

Testing LLM-powered features is fundamentally different:
1. **Multi-Dimensional Quality**: A prompt change might improve category accuracy from 85% to 92% (functional improvement), but silently cause summaries to hallucinate (semantic regression) and increase P95 latency by 1.5s (operational regression).
2. **Hidden Degradation**: LLM updates rarely break everything at once; they break edge cases, languages, or specific intent boundaries silently.
3. **Causal Attribution Limitation**: When both the prompt and the model weights change simultaneously, the system can identify that both changed, but cannot reliably determine which change caused the observed quality shift.
4. **CI/CD Flakiness & Cloud Cost**: Calling cloud APIs in pull request checks introduces network flakiness, rate limits, and recurring cloud costs.

This platform solves these problems with a **local-first, deterministic-first architecture**:
* Evaluates across **Functional**, **Semantic**, and **Operational** dimensions.
* Compares current runs against an **approved human-verified Baseline**.
* Explicitly tracks **Change Attribution** (`PROMPT_CHANGE`, `MODEL_CHANGE`, `NO_CHANGE`, or `PROMPT_AND_MODEL_CHANGE`).
* Enforces strict **Quality Gates** that exit with standard CI exit codes (`0`, `1`, `2`).
* Generates standalone **interactive HTML reports** with side-by-side regression diffs.

---

## 2. Architecture & Data Flow

```text
               Golden Dataset (v1.0)
                        │
                        ▼
             Evaluation Runner (Semaphore)
                        │
                        ▼
      ┌─────────────────┴─────────────────┐
      │  LLM Feature (Customer Classifier)│
      │  Provider: Ollama / Replay / Mock │
      └─────────────────┬─────────────────┘
                        ▼
                Actual Output
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   Functional       Semantic       Operational
Category Match    LLM Judge Score Latency & Tokens
        │               │               │
        └───────────────┼───────────────┘
                        ▼
               EvalCaseResult (per case)
                        │
                        ▼
             compute_run_metrics()
                        │
                        ▼
                 RunResult (v1)
                        │
               save_run() (JSON)
                        │
                        ▼
           ┌────────────────────────────┐
           ▼                            ▼
  Regression Comparator          Quality Gate
  vs Approved Baseline      PASS / WARNING / CRITICAL
           │                            │
           └─────────────┬──────────────┘
                         ▼
        Interactive Self-Contained HTML Report
```

---

## 3. The 3 Quality Dimensions

| Dimension | Measured Metric | How It Is Evaluated |
| :--- | :--- | :--- |
| **1. Functional** | Category Classification Accuracy | Exact match against human-verified golden label (`billing`, `technical`, `account`, `general`). |
| **2. Semantic** | Summary Relevance & Completeness | Calibrated LLM-as-a-Judge 1–5 scoring rubric with structured reasoning and error containment. |
| **3. Operational** | Latency & Token Usage | Wall-clock latency (P50, P95, Mean) + Input/Output token counts. |

---

## 4. Execution Modes: Real, Mock, and Replay

The platform decouples the feature under test from any single LLM vendor via a provider abstraction:

```python
class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse: ...
```

* **`--mode real` (Local Inference)**: Calls a local Ollama daemon (`qwen2.5:3b`) with no external LLM API dependency, allowing sensitive evaluation data to remain within the local/private execution environment.
* **`--mode mock` (Deterministic Testing)**: Instant in-memory test double for testing pipeline plumbing, CLI behaviors, and quality gate triggers.
* **`--mode replay` (Deterministic CI)**: SHA-256 content-addressed fixture cache. If an identical prompt/model request was previously recorded, it replays instantly with zero network calls.

---

## 5. Directory Structure

```text
llm-regression-detection/
├── app/
│   ├── config/             # Pydantic BaseSettings & env configs
│   ├── evaluation/         # Evaluator, runner, metrics, and LLM-as-a-judge
│   │   ├── evaluator.py    # Single-case evidence collector
│   │   ├── judge.py        # Semantic 1-5 judge rubric
│   │   ├── metrics.py      # Statistical percentiles (P50/P95) & aggregation
│   │   ├── models.py       # Core Pydantic contracts
│   │   └── runner.py       # Async batch runner with concurrency control
│   ├── feature/            # Feature under test (Email Classifier)
│   │   ├── classifier.py   # Business logic & JSON schema validation
│   │   └── prompt_loader.py# Versioned YAML prompt loader
│   ├── llm/                # Provider abstraction layer
│   │   ├── base.py         # LLMProvider ABC, LLMRequest, LLMResponse
│   │   ├── factory.py      # get_provider() instantiation point
│   │   ├── mock.py         # MockProvider
│   │   ├── ollama.py       # OllamaProvider (Async HTTPX)
│   │   └── replay.py       # ReplayProvider with SHA-256 fixture hashing
│   ├── persistence/        # Run storage & baseline promotion
│   │   ├── baseline_store.py# Promote & load official baseline
│   │   └── run_store.py    # Save & list historical run JSONs
│   ├── regression/         # Comparison & gate logic
│   │   ├── comparator.py   # Diff engine, attribution, flipped cases
│   │   ├── quality_gate.py # Multi-dimensional gate evaluator
│   │   └── thresholds.py   # Configurable warning/critical limits
│   └── reporting/          # Self-contained HTML reports
│       ├── html_report.py  # Jinja2 report generator
│       └── templates/      # Modern dark-theme report template
├── baselines/              # Committed source-of-truth baseline JSONs
├── datasets/               # Human-verified golden datasets & loader
│   ├── golden/v1.json      # 30 curated cases (4 categories, 7 test profiles)
│   └── loader.py           # Dataset loader, stats & filtering
├── prompts/                # Versioned prompt templates (v1.yaml)
├── results/                # Output artifacts (ignored in git)
│   ├── reports/            # Generated HTML reports
│   └── runs/               # Persisted RunResult JSONs
├── scripts/                # CLI entrypoints
│   ├── promote_baseline.py # CLI to promote any run to current baseline
│   └── run_eval.py         # Primary evaluation & quality-gate CLI
└── tests/unit/             # 72 unit tests across 8 modules
```

---

## 6. Quickstart Guide

### Prerequisites
* macOS or Linux
* Python 3.11+
* (Optional for `real` mode) [Ollama](https://ollama.com/) with `qwen2.5:3b`

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/llm-regression-detection.git
cd llm-regression-detection

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies in editable mode
pip install -e ".[dev]"
```

### 2. Run Evaluation (Offline / Mock Mode)
Runs instantaneously without needing Ollama:
```bash
python scripts/run_eval.py --mode mock
```

### 3. Run Evaluation with Local Ollama
Ensure Ollama is running (`ollama serve` and `ollama pull qwen2.5:3b`):
```bash
python scripts/run_eval.py --mode real
```

### 4. View the Interactive HTML Report
Open the latest generated HTML report in your browser:
```bash
open $(ls -t results/reports/*.html | head -n 1)
```

### 5. Promote an Approved Run to Baseline
```bash
python scripts/promote_baseline.py --run-id RUN-XXXXXXXX
```

### 6. Run the Test Suite
```bash
pytest tests/unit/ -v
```

---

## 7. Quality Gate Policy & Thresholds

The Quality Gate evaluates multi-dimensional deltas ($\text{Current} - \text{Baseline}$) against configurable thresholds:

| Quality Dimension | Metric Condition | Warning Threshold | Critical Threshold |
| :--- | :--- | :---: | :---: |
| **Functional** | Category accuracy delta | $\le -3.0\%$ | $\le -5.0\%$ |
| **Functional** | PASS $\rightarrow$ FAIL case regressions | $\ge 1$ case | $\ge 3$ cases |
| **Semantic** | Summary judge score delta | $\le -0.20$ pts | $\le -0.40$ pts |
| **Operational** | P95 latency percentage increase | $\ge +20\%$ | $\ge +50\%$ |

### Severity Escalation & CI Exit Codes
* **`CRITICAL` (Exit code `2`)**: Any single critical threshold is breached.
* **`WARNING` (Exit code `1`)**: No critical breach exists, but one or more warning thresholds are breached (or no approved baseline exists).
* **`PASS` (Exit code `0`)**: All metrics are within acceptable policy limits with zero case regressions.

---

## 8. Current Implementation Status

| Phase | Milestone | Status | Details |
| :---: | :--- | :---: | :--- |
| **0** | Foundation & Architecture | ✅ Done | Core Pydantic contracts, Provider factory, Settings. |
| **1** | LLM Feature Under Test | ✅ Done | Versioned YAML prompt loader, Customer support email classifier. |
| **2** | Golden Dataset | ✅ Done | 30 curated test cases across 4 categories and 7 test profiles. |
| **3** | Evaluation Engine | ✅ Done | Deterministic metrics, P50/P95 latency, LLM-as-a-judge, Batch runner. |
| **4** | Persistence | ✅ Done | Complete run evidence persistence (`results/runs/`), Baseline promotion. |
| **5** | Regression Engine | ✅ Done | Case-level diffs (`PASS` $\leftrightarrow$ `FAIL`), 4-way change attribution. |
| **6** | Quality Gate | ✅ Done | Configurable multi-dimensional thresholds, CI exit codes. |
| **7** | Interactive Reporting | ✅ Done | Standalone Jinja2 HTML report generator with dark UI. |
| **8** | Alerting & Notifications | ⏳ Next | Slack BlockKit webhook notifications & GitHub PR comment markdown. |
| **9** | Historical Drift Tracking | ⏳ Planned | Moving averages and trend detection across historical runs. |
| **10** | CI/CD GitHub Actions | ⏳ Planned | Automated PR validation, replay evaluation, and gate enforcement. |
| **11** | Hardening & Portfolio | ⏳ Planned | Docker compose, demo walkthrough, and complete portfolio readiness. |

---

## 9. Design Philosophy

1. **Score $\rightarrow$ Evidence $\rightarrow$ Decision**: Never report an aggregate score without preserving the individual case-level evidence that produced it.
2. **Determinism First**: The evaluation engine and metrics layer never call an LLM. Only the semantic judge introduces model scoring.
3. **Explicit Baseline Promotion**: Runs never silently become baselines; human or CI approval is required.
4. **Local First**: Built to run entirely on developer laptops and private CI runners with zero external cloud dependencies.
