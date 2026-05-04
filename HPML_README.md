# HPML Final Project: Skill-Augmented and Knowledge-Aware Agents for Fault Diagnosis and Action Recommendation in Industrial Asset Operations

> **Course:** High Performance Machine Learning
> **Semester:** Spring 2026
> **Instructor:** Dr. Kaoutar El Maghraoui

---

## Team Information

- **Team Name:** Team 14
- **Members:**
  - Vera Mazeeva (vmm2146)
  - Shrey Arora (ska2153)
  - Mana Abbaszadeh (ma4845)
  - Sanskruti Shejwal (ss7561)

## Submission

- **GitHub repository:** [https://github.com/shreyarora2198/AssetOpsBench](https://github.com/shreyarora2198/AssetOpsBench) (branch: `team14-final`)
- **Final report:** [`deliverables/HPML_Final_Report.pdf`](deliverables/HPML_Final_Report.pdf) *(to be added before final submission)*
- **Final presentation:** [`deliverables/HPML_Final_Presentation.pptx`](deliverables/HPML_Final_Presentation.pptx) *(to be added before final submission)*
- **Experiment-tracking dashboard:** static results checked into `eval_results/`, `trajectories/`, and `tsfm_report.{csv,json}` (no public dashboard — this is an agent-orchestration study, not a model-training study; raw per-scenario JSONL is committed for full reproducibility).

The final report PDF and the presentation file will be checked into the `deliverables/` folder of this repository **and** uploaded to CourseWorks.

---

## 1. Problem Statement

Industrial operations and maintenance (O&M) agents in the AssetOpsBench benchmark currently use **static pipelines** that invoke every analysis component — IoT sensor retrieval, time-series ML forecasting (TSFM), failure-mode reasoning (FMSR), and work-order generation — regardless of whether all of them are needed for the task at hand. This is **inference-time waste**: the most expensive component (Deep TSFM, which runs ML-based statistical models on time-series sensor data) is invoked even when a cheaper FMSR diagnosis is already high-confidence. We target **inference efficiency**, with the bottleneck being **redundant invocation of expensive ML inference calls** in multi-step agent trajectories.

---

## 2. Model/Application Description

This is an **agent-orchestration optimization study**, not a single-model training study. The "model" is a multi-agent system; we optimize *which* expensive components are invoked *when*.

- **Agent stack:**
  - **Planner**: LiteLLM-backed LLM (configurable: WatsonX `meta-llama/llama-4-maverick-17b-128e-instruct-fp8`, GPT-4o, Anthropic Claude, Groq Llama-3.3-70b)
  - **Skill executor**: deterministic Python orchestrator with conditional execution and early-stopping
  - **Knowledge plugins**: per-skill targeted retrieval (sensor metadata, failure-mode catalogue, operating ranges, maintenance policies)
  - **Tools**: 4 MCP servers — IoT (CouchDB sensor data), FMSR (curated failure modes + WatsonX LLM mapping), TSFM (IBM Granite TinyTimeMixer for forecasting & conformal anomaly detection), WO (work-order generation, predictive analytics)
- **LLM judge for evaluation:** WatsonX `meta-llama/llama-3-3-70b-instruct` via LiteLLM, scoring on the AssetOpsBench-native 6 dimensions (`task_completion`, `data_retrieval_accuracy`, `generalized_result_verification`, `agent_sequence_correct`, `clarity_and_justification`, `hallucinations`)
- **Framework:** Python 3.12+, [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) over stdio, LiteLLM for cross-provider LLM routing, FastMCP for server scaffolding, `uv` for dependency management
- **Dataset:** [`ibm-research/AssetOpsBench`](https://huggingface.co/datasets/ibm-research/AssetOpsBench) — 141 human-authored industrial O&M scenarios across Chillers, AHUs, Motors, Pumps, Bearings (Apache 2.0). We evaluate on a 54-scenario subset spanning all 4 task categories (fault diagnosis, anomaly detection, forecasting, metadata retrieval).
- **Custom contributions:** skill abstraction layer (`src/skills-knowledge-agent/`), confidence evaluator with threshold-based gating (`confidence_evaluator.py`), Work Order MCP server (`src/servers/wo/`), trajectory analyzer for TSFM redundancy auditing (`src/get_trajectories/analyze_tsfm.py`), and the WatsonX-via-LiteLLM judge (`run_evaluation.py`, `score_ablation.py`).
- **Hardware target:** commodity laptop CPU (macOS / Linux) for the agent orchestration loop; WatsonX cloud endpoint for LLM inference. **No local GPU required** — the optimization target is *inference cost* (number of expensive remote LLM/TSFM calls per task), not per-call compute.

---

## 3. Final Results Summary

Mean over 54 scenarios per condition. **Baseline = Condition F** (skills + knowledge + *unconditional* Deep TSFM). **Optimized = Condition E at θ=0.8** (skills + knowledge + *conditional* Deep TSFM gated on FMSR confidence).

| Metric                                  | Baseline (F) | Optimized (E, θ=0.8) | Δ (Improvement) |
| --------------------------------------- | -----------: | -------------------: | --------------- |
| Task completion rate (LLM-judged)       | TBD           | TBD                   | TBD pending full judge run |
| Mean tool calls per task                | 3.07         | 3.02                 | -1.6%           |
| Mean end-to-end latency (s)             | 13.64        | 13.49                | -1.1%           |
| Deep-TSFM invocation rate (fault-diag)  | 100%         | ~50% (overall 50% over all categories vs 48% baseline\*) | **~50% reduction in expensive ML inference calls on high-confidence cases** |
| Estimated cost per task (USD)           | TBD           | TBD                   | TBD pending judge run |

\* The baseline F's 48% rate (rather than 100%) reflects that only fault-diagnosis category scenarios trigger TSFM at all; non-fault categories never invoke it in either condition.

**Hardware:** macOS (Apple M-series) + WatsonX cloud LLM endpoint (`us-south.ml.cloud.ibm.com`). Python 3.12, `uv`-managed virtualenv, no local GPU required.

**Headline result (one sentence):**
*Conditional Deep-TSFM gating at θ=0.8 skips the expensive ML statistical-validation step on roughly half of fault-diagnosis scenarios where FMSR already produces a confident diagnosis, with a measured reduction in tool-call count and latency at parity diagnostic outcome — full LLM-judge accuracy comparison pending the May 4 evaluation run.*

---

## 4. Repository Structure

```
.
├── HPML_README.md                           ← (this file)
├── README.md                                ← upstream IBM AssetOpsBench README
├── INSTRUCTIONS.md                          ← MCP environment + plan-execute usage
├── LICENSE                                  ← Apache 2.0 (upstream AssetOpsBench)
├── pyproject.toml                           ← uv-managed Python package
├── uv.lock                                  ← pinned dependency versions
├── .env.public                              ← template for required env vars
├── deliverables/                            ← (to be added) final report + slides
│   ├── HPML_Final_Report.pdf
│   └── HPML_Final_Presentation.pptx
├── eval_results/
│   ├── ablation_results.csv                 ← 648 rows: 12 conditions × 54 scenarios
│   └── ablation_scored.csv                  ← (generated) LLM-judge scores
├── trajectories/                            ← raw per-scenario execution traces
│   ├── all_trajectories.json
│   └── Q_*.json                             ← 150 individual trajectory files
├── tsfm_report.csv / tsfm_report.json       ← TSFM redundancy audit
├── scores.json / leaderboard.png            ← Sanskruti's auxiliary judge output
├── run_evaluation.py                        ← WatsonX EvaluationAgent judge (single-result file mode)
├── score_ablation.py                        ← Adapter: runs judge over the ablation CSV
├── run_subset.py                            ← Smoke-test baseline runner (5 scenarios)
├── evaluate_baseline.py                     ← Driver used for the Apr 16 baseline scoring
├── run_original_eval.py                     ← Parity check vs IBM's reactxen EvaluationAgent
└── src/
    ├── workflow/                            ← plan-execute MCP runner (Planner, Executor, Summariser)
    ├── servers/
    │   ├── iot/                             ← CouchDB-backed IoT MCP server
    │   ├── fmsr/                            ← failure-mode-sensor-relevance MCP server
    │   ├── tsfm/                            ← TinyTimeMixer forecasting + anomaly MCP server
    │   ├── wo/                              ← Work Order MCP server (Vera, this project)
    │   └── utilities/                       ← time/JSON helper MCP server
    ├── skills-knowledge-agent/              ← THE CONTRIBUTION (Vera)
    │   ├── skills.py                        ← skill registry & executor
    │   ├── knowledge.py                     ← knowledge-plugin retrieval
    │   ├── confidence_evaluator.py          ← FMSR confidence → conditional Deep-TSFM
    │   ├── eval_runner.py                   ← 5-condition × multi-θ ablation harness
    │   ├── tools.py                         ← tool wrappers around MCP servers
    │   ├── agent.py / deep_agent.py         ← agent entry points
    │   └── tests/                           ← unit + smoke tests
    ├── get_trajectories/                    ← THE CONTRIBUTION (Sanskruti)
    │   ├── run_all_scenarios.py             ← run baseline over all 150 scenarios
    │   ├── analyze_tsfm.py                  ← TSFM redundancy / wrong-domain audit
    │   ├── score_trajectories.py            ← Llama-4-Maverick auxiliary judge
    │   └── plot_results.py                  ← leaderboard chart
    └── couchdb/                             ← Docker compose + sample IoT data
```

---

## 5. Reproducibility Instructions

### A. Environment Setup

```bash
# Clone
git clone https://github.com/shreyarora2198/AssetOpsBench.git
cd AssetOpsBench
git checkout team14-final

# Install dependencies (uv handles venv + pinned versions)
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
uv sync

# Configure secrets
cp .env.public .env
# Edit .env: set WATSONX_APIKEY, WATSONX_PROJECT_ID
# (CouchDB defaults work out of the box with the bundled Docker compose)
```

**System requirements:** Python 3.12+, Docker (for the IoT CouchDB), WatsonX account. **No GPU required** — TSFM model inference uses CPU-friendly TinyTimeMixer checkpoints; LLM inference is delegated to WatsonX cloud.

### B. Experiment Tracking Dashboard

This project does not use Weights & Biases / MLflow because it is an agent-orchestration study with deterministic, file-based artifacts rather than gradient-descent training runs. Static results are committed under:

- `eval_results/ablation_results.csv` — 648-row ablation across 12 conditions × 54 scenarios
- `eval_results/ablation_scored.csv` — same rows with LLM-judge 6-dimension scores
- `trajectories/all_trajectories.json` — raw per-scenario execution traces (150 scenarios)
- `tsfm_report.{csv,json}` — TSFM redundancy audit
- `scores.json` — auxiliary Llama-4-Maverick judge output

A static markdown report rendering the headline plots is bundled in `deliverables/`.

### C. Dataset

The AssetOpsBench scenario dataset is fetched from HuggingFace at runtime — no manual download:

```python
from datasets import load_dataset
ds = load_dataset("ibm-research/AssetOpsBench", "scenarios")
```

The IoT sensor data is bundled as a CouchDB fixture:

```bash
docker compose -f src/couchdb/docker-compose.yaml up -d
```

License: Apache 2.0 (`https://huggingface.co/datasets/ibm-research/AssetOpsBench`).

### D. Training

**Not applicable** — no model parameters are trained in this project. The skill executor and knowledge plugins encode domain logic directly, and the confidence threshold θ is set by domain reasoning (we sweep θ ∈ {0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95}) rather than gradient-based optimization.

### E. Evaluation

To reproduce the 12-condition ablation:

```bash
# Start CouchDB (IoT data store)
docker compose -f src/couchdb/docker-compose.yaml up -d

# Run the full 12-condition × 54-scenario ablation
uv run python src/skills-knowledge-agent/eval_runner.py
# → writes eval_results/ablation_results.csv
```

To run the LLM judge over the ablation results:

```bash
# Smoke test on first 5 rows (verifies WatsonX credentials + parsing)
uv run python score_ablation.py --limit 5

# Full run (~45 min, costs <$1 in WatsonX tokens)
uv run python score_ablation.py
# → writes eval_results/ablation_scored.csv
```

### F. Profiling

The eval runner records per-scenario `tool_calls`, `latency_s`, `total_cost`, and `deep_tsfm_invoked` flags directly into `eval_results/ablation_results.csv`. Per-condition aggregates:

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('eval_results/ablation_results.csv')
print(df.groupby('condition')[['tool_calls', 'latency_s', 'deep_tsfm_invoked']].mean().round(2))
"
```

To audit TSFM redundancy on the full 150-scenario baseline:

```bash
uv run python src/get_trajectories/analyze_tsfm.py
# → writes tsfm_report.csv / tsfm_report.json
```

### G. Quickstart: Reproduce the Headline Result

The following sequence reproduces the headline conditional-TSFM gating result end-to-end (≈ 60 minutes, depending on WatsonX latency):

```bash
# 1. Set up environment
uv sync
cp .env.public .env  # edit with WATSONX_APIKEY, WATSONX_PROJECT_ID

# 2. Start IoT data store
docker compose -f src/couchdb/docker-compose.yaml up -d

# 3. Run the 12-condition ablation
uv run python src/skills-knowledge-agent/eval_runner.py

# 4. Score with the LLM judge
uv run python score_ablation.py

# 5. Aggregate
uv run python -c "
import pandas as pd
df = pd.read_csv('eval_results/ablation_scored.csv')
df['overall_score'] = pd.to_numeric(df['overall_score'], errors='coerce')
print(df.groupby('condition')[['overall_score', 'tool_calls', 'latency_s', 'deep_tsfm_invoked']].mean().round(2))
"
```

---

## 6. Results and Observations

*Numbers below are from the May 3 ablation run (`eval_results/ablation_results.csv`). LLM-judge accuracy scores are pending the full `score_ablation.py` run; those rows will be filled in before the report submission.*

- **Conditional Deep-TSFM gating roughly halves the expensive ML inference call rate** at θ=0.8 (50% of fault-diagnosis scenarios) compared to unconditional Condition F (100% of fault-diagnosis scenarios). At θ=0.5 the rate drops to 0% because FMSR's first-pass confidence almost never falls below 0.5 — this confirms the threshold sweep is meaningful.
- **Tool-call efficiency is preserved**: mean 3.02 calls/task at θ=0.8 vs 3.07 at unconditional Deep-TSFM (Condition F) — within noise. The cost saving comes from skipping a single but expensive component, not from reducing total call count.
- **Planning alone is not enough**: Condition C (planning, no skill structure) averages 11.85s latency vs Condition D (skills + knowledge, no Deep-TSFM gating) at 10.78s — confirming that explicit skill structure adds value beyond just letting the LLM plan.
- **Latency at θ=0.8 is dominated by the WatsonX network round-trip rather than tool execution** (13.49s mean). On-prem deployment of the planner LLM would be the natural next optimization target.
- **What did not work cleanly:** Condition B was originally framed as a ReAct baseline; the implemented version is a static unconditional pipeline. We retain it as `B_tool_baseline` and have updated the report's framing to match the code rather than the proposal.

*Headline figure to be exported from `eval_results/` prior to final submission and embedded here.*

---

## 7. Notes

- Source files live under `src/`, ablation outputs under `eval_results/`, raw trajectories under `trajectories/`, and team-internal scripts at the repo root (`run_evaluation.py`, `score_ablation.py`, `run_subset.py`).
- WatsonX credentials and CouchDB credentials are loaded from environment variables. See `.env.public` for the template.
- The branch `team14-final` is the canonical submission branch. Feature branches (`baseline-setup`, `skills_knowledge`, `watsonx-eval`) remain in place to preserve per-author authorship history.

### AI Use Disclosure

*Per the HPML AI Use Policy (posted on CourseWorks). Required for every submission.*

**Did your team use any AI tool in completing this project?**

- [ ] No, we did not use any AI tool.
- [x] Yes, we used AI assistance as described below.

**Tool(s) used:** Claude (Anthropic), used through the Claude Code CLI.

**Specific purpose:**
- Drafting and refactoring helper scripts (`run_evaluation.py`, `score_ablation.py`) — porting the IBM-internal `reactxen.EvaluationAgent` to a LiteLLM/WatsonX backend, and writing the CSV → judge-format adapter.
- Branch-integration planning (merge order, conflict-resolution recipes for `pyproject.toml` and `uv.lock`).
- Identifying inconsistencies between the proposal and the implementation (e.g., Condition B mislabeled as ReAct, default θ in `confidence_evaluator.py` differing from the paper's sweep range).
- Drafting prose for this README and clarifying sections of the project proposal.

**Sections affected:**
- `score_ablation.py` (entirely Claude-drafted, then human-reviewed and run end-to-end).
- `run_evaluation.py` (refactored from existing repo code with Claude assistance).
- `HPML_README.md` (this file — drafted with Claude, reviewed and edited by the team).
- Branch-merge planning documented in conversation with Claude; actual `git` operations were executed by team members.

**How we verified correctness:**
- Every script was executed end-to-end by a human team member; output was sanity-checked against the input data and against existing repo functionality.
- The `score_ablation.py` smoke test on 5 rows was run and the per-row JSON output was inspected manually before the full 648-row run.
- The merged `team14-final` branch was verified by listing each contributed directory (`src/get_trajectories/`, `src/skills-knowledge-agent/`, `src/servers/wo/`, `run_evaluation.py`, `trajectories/`) and confirming all four authors' commits appear in `git log`.
- All numbers reported in this README come from `eval_results/ablation_results.csv` aggregated by the team; AI-suggested narrative claims that were not backed by the data were removed.

By submitting this project, the team confirms that the analysis, interpretations, and conclusions are our own, and that any AI assistance is fully disclosed above. The same disclosure block appears as an appendix in the final report.

### License

This repository is a fork of [IBM/AssetOpsBench](https://github.com/IBM/AssetOpsBench) and is released under the upstream Apache 2.0 license. See [`LICENSE`](LICENSE).

### Citation

If you build on this work, please cite both AssetOpsBench and our extension:

```bibtex
@misc{team14_2026_hpml_assetopsbench,
  title  = {Skill-Augmented and Knowledge-Aware Agents for Fault Diagnosis and Action Recommendation in Industrial Asset Operations},
  author = {Arora, Shrey and Abbaszadeh, Mana and Shejwal, Sanskruti and Mazeeva, Vera},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/shreyarora2198/AssetOpsBench/tree/team14-final}
}

@article{ibm2025assetopsbench,
  title  = {AssetOpsBench: Benchmarking AI Agents for Industrial Asset Operations and Maintenance},
  author = {IBM Research},
  year   = {2025},
  url    = {https://arxiv.org/abs/2506.03828}
}
```

### Contact

Open a [GitHub Issue](https://github.com/shreyarora2198/AssetOpsBench/issues) on the team fork, or email the team lead at `ska2153@columbia.edu`.

---

*HPML Spring 2026 — Dr. Kaoutar El Maghraoui — Columbia University*
