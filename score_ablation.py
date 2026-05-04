"""score_ablation.py — Run the WatsonX EvaluationAgent over ablation_results.csv.

Reads eval_results/ablation_results.csv produced by src/skills-knowledge-agent/eval_runner.py,
fetches characteristic_form per scenario_id from the AssetOpsBench HF dataset, and
calls run_evaluation.EvaluationAgent on each row to attach 6-dimension scores.

Output: eval_results/ablation_scored.csv (resumable — re-running skips rows already scored).

Usage:
    uv run python score_ablation.py --limit 5         # smoke test, first 5 rows
    uv run python score_ablation.py                   # full 649 rows
    uv run python score_ablation.py --condition E_full_system --theta 0.8   # subset
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# Reuse the existing judge implementation
from run_evaluation import EvaluationAgent, compute_score

DIM_KEYS = [
    "task_completion",
    "data_retrieval_accuracy",
    "generalized_result_verification",
    "agent_sequence_correct",
    "clarity_and_justification",
    "hallucinations",
]


def extract_agent_answer(result_json_str: str) -> str:
    """Pull the final answer text out of the eval_runner result_json blob."""
    if not result_json_str or pd.isna(result_json_str):
        return ""
    try:
        obj = json.loads(result_json_str)
    except (json.JSONDecodeError, TypeError):
        return str(result_json_str)[:2000]
    if isinstance(obj, dict):
        for key in ("answer", "diagnosis", "work_order", "forecast", "metadata"):
            if key in obj and obj[key]:
                v = obj[key]
                return v if isinstance(v, str) else json.dumps(v)[:2000]
        return json.dumps(obj)[:2000]
    return str(obj)[:2000]


def extract_agent_plan(plan_str: str, trace_str: str) -> str:
    """Build a short plan/trace summary for the judge's 'agent_think' field."""
    parts = []
    if plan_str and not pd.isna(plan_str):
        try:
            plan = json.loads(plan_str)
            parts.append("Plan: " + " -> ".join(str(s) for s in plan))
        except Exception:
            parts.append(f"Plan: {plan_str}")
    if trace_str and not pd.isna(trace_str):
        try:
            trace = json.loads(trace_str)
            steps = trace.get("skill_steps") or []
            if steps:
                parts.append("Steps:")
                for s in steps[:8]:
                    parts.append(f"  - {s}")
        except Exception:
            pass
    return "\n".join(parts) or "(no plan recorded)"


def load_characteristic_forms() -> dict[int, str]:
    """Fetch ground-truth characteristic_form per scenario id from HF."""
    print("Loading AssetOpsBench scenarios from HuggingFace...")
    ds = load_dataset("ibm-research/AssetOpsBench", "scenarios")
    rows = ds["train"].to_list()
    forms = {}
    for r in rows:
        sid = r.get("id")
        cf = r.get("characteristic_form") or r.get("characteristic_answer") or ""
        if sid is not None:
            forms[int(sid)] = cf
    print(f"  Loaded {len(forms)} characteristic forms")
    return forms


def already_scored(out_path: Path) -> set[tuple]:
    """Return set of (condition, theta, task_id) keys already in the output file."""
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["condition"], row["theta"], row["task_id"]))
    return done


def parse_task_id_to_int(tid: str) -> int | None:
    """Convert task_id like 'TSFM_117' or '117' to int 117 for HF lookup."""
    if not tid:
        return None
    tid = str(tid).strip()
    if tid.isdigit():
        return int(tid)
    parts = tid.replace("Q_", "").replace("TSFM_", "").split("_")
    for p in parts:
        if p.isdigit():
            return int(p)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="eval_results/ablation_results.csv")
    parser.add_argument("--output", default="eval_results/ablation_scored.csv")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only score the first N rows (after filters). For smoke tests.")
    parser.add_argument("--condition", default=None,
                        help="Filter to a single condition, e.g. E_full_system")
    parser.add_argument("--theta", default=None,
                        help="Filter to a single theta value, e.g. 0.8")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds to sleep between calls (rate limiting)")
    args = parser.parse_args()

    for var in ("WATSONX_APIKEY", "WATSONX_PROJECT_ID"):
        if not os.environ.get(var):
            print(f"ERROR: {var} not set in .env")
            sys.exit(1)

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found. Did you copy the CSV into eval_results/?")
        sys.exit(1)

    df = pd.read_csv(in_path)
    if args.condition:
        df = df[df["condition"] == args.condition]
    if args.theta is not None:
        df = df[df["theta"].astype(str) == str(args.theta)]
    if args.limit:
        df = df.head(args.limit)

    print(f"Rows to score: {len(df)}")
    if len(df) == 0:
        print("Nothing to do.")
        return

    forms = load_characteristic_forms()
    done = already_scored(out_path)
    print(f"Already scored (resume): {len(done)} rows")

    # Output: input columns + new score columns
    score_cols = [f"score_{k}" for k in DIM_KEYS] + ["overall_score", "judge_status"]
    fieldnames = list(df.columns) + score_cols

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    fout = open(out_path, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    judge = EvaluationAgent(max_retries=3)
    n_done = n_skipped = n_error = 0
    t_start = time.time()

    try:
        for i, row in df.iterrows():
            key = (row["condition"], str(row.get("theta", "")), row["task_id"])
            if key in done:
                n_skipped += 1
                continue

            sid = parse_task_id_to_int(str(row.get("scenario_id") or row.get("task_id")))
            char_form = forms.get(sid, "")
            agent_answer = extract_agent_answer(row.get("result_json", ""))
            agent_plan = extract_agent_plan(row.get("plan", ""), row.get("trace_json", ""))

            print(f"  [{n_done + n_skipped + n_error + 1}/{len(df)}] "
                  f"{row['condition']} θ={row.get('theta', '-')} {row['task_id']}...",
                  end=" ", flush=True)

            scores = judge.evaluate_response(
                question=str(row.get("task", "")),
                agent_think=agent_plan,
                agent_response=agent_answer,
                characteristic_answer=char_form,
            )

            out_row = {col: row[col] for col in df.columns}
            if scores.get("status") == "Error":
                out_row["judge_status"] = "error"
                out_row["overall_score"] = ""
                for k in DIM_KEYS:
                    out_row[f"score_{k}"] = ""
                n_error += 1
                print("error")
            else:
                out_row["judge_status"] = "ok"
                out_row["overall_score"] = compute_score(scores)
                for k in DIM_KEYS:
                    v = scores.get(k)
                    out_row[f"score_{k}"] = "" if v is None else str(v)
                n_done += 1
                print(f"{out_row['overall_score']:.1f}%")

            writer.writerow(out_row)
            fout.flush()
            time.sleep(args.sleep)
    finally:
        fout.close()

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  scored OK : {n_done}")
    print(f"  skipped   : {n_skipped} (already in {out_path})")
    print(f"  errors    : {n_error}")
    print(f"\nOutput → {out_path}")


if __name__ == "__main__":
    main()
