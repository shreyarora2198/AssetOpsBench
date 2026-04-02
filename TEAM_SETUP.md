# AssetOpsBench — Team Setup & Baseline Run Guide

> **HPML Spring 2026 — Team #14**
> Skill-Augmented and Knowledge-Aware Agents for Fault Diagnosis

This guide explains how to set up the environment and run the baseline agent locally.
The baseline is the existing `plan-execute` orchestrator from the AssetOpsBench repo —
this corresponds to the **"baseline agent"** in our evaluation ablation table.

---

## What the Baseline Agent Does

```
Your question
    │
    ├─ LLM (gpt-4o-mini) plans which tools to call
    │
    ├─ Calls MCP servers (FMSRAgent, Utilities, etc.)
    │       └─ Tools return structured data
    │
    └─ LLM summarizes tool results into a final answer
```

No conditional execution, no skill plugins, no knowledge plugins — that is what we are building on top of this.

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.12+ | Already installed on most machines |
| `uv` | any | `brew install uv` (Mac) or `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker | any | Only needed for IoT/TSFM scenarios — skip for now |
| OpenAI API key | — | See below |

---

## Step 1 — Clone the repo

```bash
git clone <your-team-repo-url>
cd AssetOpsBench
```

---

## Step 2 — Install dependencies

```bash
uv sync
```

This creates a `.venv/` directory and installs all packages including the MCP servers.

---

## Step 3 — Get an OpenAI API key

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign in → click **Create new secret key**
3. Copy the key (starts with `sk-proj-...`)

> **Cost:** Running all 5 baseline scenarios costs less than $0.05 total using `gpt-4o-mini`.

> **Security:** Never paste your API key in Slack, chat, or commit it to git. Only put it in `.env` locally.

---

## Step 4 — Configure environment

Copy the template and fill in your key:

```bash
cp .env.public .env
```

Open `.env` and set these values:

```bash
OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE
LITELLM_API_KEY=sk-proj-YOUR_KEY_HERE   # same key
LITELLM_BASE_URL=https://api.openai.com/v1
FMSR_MODEL_ID=openai/gpt-4o-mini
```

Leave all other variables (`COUCHDB_*`, `WATSONX_*`) as-is for now.

> **Note:** Both `OPENAI_API_KEY` and `LITELLM_API_KEY` should be set to the same key.
> `OPENAI_API_KEY` is used by `run_subset.py`; `LITELLM_API_KEY` is used by the FMSR
> MCP server subprocess internally.

---

## Step 5 — Run the baseline

```bash
uv run python run_subset.py
```

This runs 5 scenarios (FMSR + Utilities domains) that work without Docker:

| Scenario | Domain | What it tests |
|----------|--------|---------------|
| `util_1` | Utilities | Current date/time — zero dependencies |
| `fmsr_1` | FMSR | Chiller failure modes — curated data lookup |
| `fmsr_2` | FMSR | AHU failure modes — curated data lookup |
| `fmsr_3` | FMSR | Sensor → failure mode mapping (LLM reasoning) |
| `fmsr_4` | FMSR + Util | Multi-agent: both servers in one plan |

Results are saved to `results/subset_results.json`.

---

## What a Good Output Looks Like

```
============================================================
[fmsr_1] (FMSR) What are the failure modes for a chiller?
============================================================

PLAN:
  Step 1 [FMSRAgent]: Retrieve the failure modes for the chiller asset.

ANSWER:
The failure modes for the chiller include:
1. Compressor Overheating: Failed due to normal wear and overheating.
2. Heat Exchangers: Fans with degraded motor or worn bearing.
3. Evaporator water side fouling.
...
```

The plan should show **real tool names** (`get_failure_modes`, `current_date_time`) — not vague steps like "gather historical data". If you see vague steps, the MCP servers are not connecting properly (re-run `uv sync`).

---

## Architecture Overview

```
src/
├── workflow/          # Baseline plan-execute orchestrator (the baseline agent)
│   ├── runner.py      # PlanExecuteRunner — top-level entry point
│   ├── planner.py     # LLM decomposes question into steps
│   └── executor.py    # Routes steps to MCP servers via stdio
│
├── servers/           # MCP tool servers (the "skills" in current form)
│   ├── fmsr/          # Failure Mode Sensor Reasoning tools
│   ├── iot/           # IoT sensor data tools (needs Docker)
│   ├── tsfm/          # Time-series forecasting tools (needs model checkpoints)
│   └── utilities/     # Date/time utilities
│
└── llm/               # LLM backend abstraction
    ├── base.py         # LLMBackend abstract class — implement this to swap models
    └── litellm.py      # LiteLLM-based backend (WatsonX / LiteLLM proxy)

run_subset.py          # Our baseline runner script (FMSR + Utilities only)
results/               # Output directory — baseline results saved here
```

---

## Running the Full IoT + TSFM Scenarios (needs Docker)

The fault diagnosis scenarios from our proposal (*"Why is Chiller 6 behaving abnormally?"*)
require CouchDB for sensor data. If Docker is available:

```bash
# Start CouchDB
docker compose -f src/couchdb/docker-compose.yaml up -d

# Verify it is running
curl http://localhost:5984/
```

Then update `run_subset.py` to add `IoTAgent` and `TSFMAgent` to `server_paths` and add
the relevant scenarios.

---

## Swapping the LLM (for ablation experiments)

The `OpenAIBackend` class in `run_subset.py` implements the `LLMBackend` interface.
To benchmark a different model, create a new subclass:

```python
from llm import LLMBackend

class MyModelBackend(LLMBackend):
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        # call your model here
        return "..."
```

Pass it to the runner:

```python
runner = PlanExecuteRunner(llm=MyModelBackend(), server_paths={...})
```

---

## What We Are Building on Top of This (Proposal §4)

The baseline runs all tools unconditionally. Our contribution adds:

1. **Confidence Evaluator** — reads FMSR output confidence, decides whether to invoke TSFM
2. **Skill Plugins** — encapsulate multi-step workflows (root cause analysis, anomaly detection, work order generation)
3. **Knowledge Plugins** — targeted retrieval of sensor metadata, failure mode library, maintenance policy
4. **Conditional Executor** — skips expensive TSFM agent when FMSR confidence > threshold

These go on top of the existing `src/workflow/` infrastructure without modifying the core MCP servers.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'servers'` | Run `uv sync` again (reinstalls package) |
| `OPENAI_API_KEY not set` | Check `.env` has the key with no quotes or spaces |
| Plan shows vague steps (no tool names) | MCP servers failed to start; run `uv sync --reinstall-package assetopsbench-mcp` |
| Rate limit error | Add a longer `time.sleep()` between scenarios, or switch to a higher-tier API key |
| `uv: command not found` | `brew install uv` (Mac) or see [docs.astral.sh/uv](https://docs.astral.sh/uv/) |

---

## File to Commit — Do NOT Commit

| Commit | Do NOT commit |
|--------|---------------|
| `run_subset.py` | `.env` (contains your API key) |
| `TEAM_SETUP.md` | `results/` (large output files) |
| `pyproject.toml` | `.venv/` (auto-generated) |
| `src/` | Any file with `sk-proj-` in it |

`.env` is already in `.gitignore` — but double-check before pushing.
