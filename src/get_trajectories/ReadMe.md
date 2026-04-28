# AssetOpsBench Trajectory Analysis

Scripts for running the AssetOpsBench baseline agent, analyzing TSFM misuse, scoring trajectories, and reproducing the leaderboard bar chart.

All scripts live in `src/get_trajectories/`.

---

## Setup

```bash
uv sync
uv add datasets huggingface_hub matplotlib
```

Copy the env template and fill in your credentials:

```bash
cp .env.public .env
```

Required keys in `.env`:

```
WATSONX_APIKEY=your_key
WATSONX_PROJECT_ID=your_project_id
HF_APIKEY=your_huggingface_token
COUCHDB_URL=http://localhost:5984
COUCHDB_USERNAME=admin
COUCHDB_PASSWORD=password
```

Start CouchDB (required for IoT and Work Order scenarios):

```bash
docker compose -f src/couchdb/docker-compose.yaml up -d
```

---

## Scripts

### 1. `run_all_scenarios.py` — Run the baseline agent

Loads all scenarios from HuggingFace, runs the plan-execute agent on each one, and saves trajectories.

```bash
uv run python src/get_trajectories/run_all_scenarios.py
```

**Output:**
- `trajectories/Q_<id>.json` — one file per scenario
- `trajectories/all_trajectories.json` — all scenarios combined

Each trajectory contains the question, final answer, and every tool call the agent made (agent, tool name, args, response, success/fail).

**To run a different model:**

```bash
MODEL_ID=watsonx/ibm/granite-3-3-8b-instruct uv run python src/get_trajectories/run_all_scenarios.py
```

---

### 2. `analyze_tsfm.py` — Detect unnecessary TSFM calls

TSFM inference tools (`run_tsfm_forecasting`, `run_tsad`, `run_integrated_tsad`, `run_tsfm_finetuning`) should only be called for TSFM inference scenarios (IDs 216–223). Calling them anywhere else is a violation.

| Scenario type | IDs | Should use | TSFM call = |
|---|---|---|---|
| IoT | 1–48 | `sites`, `assets`, `sensors`, `history` | ❌ Unnecessary |
| FMSA (fault diagnosis) | 101–120 | `get_failure_modes`, `get_failure_mode_sensor_mapping` | ❌ Unnecessary |
| TSFM knowledge | 201–215 | `get_ai_tasks`, `get_tsfm_models` | ❌ Unnecessary |
| TSFM inference | 216–223 | `run_tsfm_forecasting`, `run_tsad`, etc. | ✅ Expected |
| Work Order | 400+ | WO tools | ❌ Unnecessary |
| Multiagent | 500+ | IoT + FMSR, then optionally TSFM | ❌ If skipping IoT |

```bash
uv run python src/get_trajectories/analyze_tsfm.py
```

**Output:**
- Printed summary — total violations, breakdown by label (FAULT-DIAG vs WRONG-DOMAIN), which tools were misused and how many times
- `tsfm_report.json` — full list of violations with scenario ID, type, question text, and tools called

**Labels:**
- `FAULT-DIAG` — FMSA scenario that called TSFM (the core problem this project targets)
- `WRONG-DOMAIN` — any other non-TSFM scenario that called TSFM

---

### 3. `score_trajectories.py` — Score with LLM judge

Scores each trajectory using the LLM judge model. For each scenario, the judge is given the question (`Q`), the agent's trajectory (`T`), and the characteristic form / ground truth guide (`C`) from the HuggingFace dataset, and returns scores on 6 dimensions.

**Scoring function:** `f(Q, T, C) → (y₁…y₆, pass/fail)`

| Dimension | What it measures |
|---|---|
| `task_completion` | Did the agent complete the task? |
| `tool_selection` | Were the right tools used? |
| `answer_correctness` | Is the final answer correct? |
| `reasoning_quality` | Is the reasoning logical? |
| `data_handling` | Were data values handled accurately? |
| `efficiency` | Was the trajectory concise? |

The judge is called 5 times per scenario and scores are averaged for consistency. A scenario **passes** if the average score ≥ 6/10.

```bash
uv run python src/get_trajectories/score_trajectories.py
```

**Output:**
- `scores.json` - saved after every scenario so progress is not lost if interrupted

---

### 4. `plot_results.py` — Reproduce the leaderboard bar chart

Reads `scores.json` and plots Pass₁ rate (y-axis) per scenario type (IoT, FMSA, TSFM, Workorder) as a bar chart, matching the style of the README leaderboard graph.

```bash
uv run python src/get_trajectories/plot_results.py
```

**Output:**
- `leaderboard.png`

---

## Run everything in order

```bash
uv run python src/get_trajectories/run_all_scenarios.py
uv run python src/get_trajectories/analyze_tsfm.py
uv run python src/get_trajectories/score_trajectories.py
uv run python src/get_trajectories/plot_results.py
```

---

## Output files

| File | Created by | Contents |
|---|---|---|
| `trajectories/all_trajectories.json` | `run_all_scenarios.py` | All agent trajectories |
| `trajectories/Q_<id>.json` | `run_all_scenarios.py` | Per-scenario trajectory |
| `tsfm_report.json` | `analyze_tsfm.py` | TSFM violation report |
| `tsfm_report.csv` | `analyze_tsfm.py` | Same report as CSV |
| `scores.json` | `score_trajectories.py` | LLM judge scores per scenario |
| `leaderboard.png` | `plot_results.py` | Pass rate bar chart |

---

## Notes

- Scoring takes several hours for all scenarios. It saves after every scenario so you can stop and resume safely.
- CouchDB must be running for IoT (1–48) and Work Order (400+) scenarios. FMSA (101–120) and TSFM knowledge (201–215) scenarios work without it.
- The HuggingFace dataset requires your account to have accepted the dataset terms at https://huggingface.co/datasets/ibm-research/AssetOpsBench before all 152 scenarios load.
