"""
run_original_eval.py — Score results using the exact EvaluationAgent from
src/tmp/evaluation_agent/agent.py with reactxen's watsonx_llm.

Mirrors analyze.py from the paper: NUM_ITER=5 majority vote per scenario.
Judge model: meta-llama/llama-3-3-70b-instruct (model_id=12, matches paper).

Requirements:
    .venv312/bin/python run_original_eval.py --results results/evaluated_baseline.json

(Must use Python 3.12 venv because ibm_watsonx_ai requires Python < 3.14)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from functools import partial

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from reactxen.utils.model_inference import watsonx_llm
from reactxen.agents.evaluation_agent.agent import EvaluationAgent

NUM_ITER = 5   # majority vote — same as analyze.py
TEMP = 0.3     # same as analyze.py
MODEL_ID = 12  # meta-llama/llama-3-3-70b-instruct

SCORE_DIMS = [
    "task_completion",
    "data_retrieval_accuracy",
    "generalized_result_verification",
    "agent_sequence_correct",
    "clarity_and_justification",
    "hallucinations",
]


def majority_vote(results: list[dict]) -> dict:
    """Majority vote across NUM_ITER runs, same logic as analyze.py."""
    counts = {k: 0 for k in SCORE_DIMS}
    for r in results:
        for k in SCORE_DIMS:
            if r.get(k):
                counts[k] += 1
    voted = {}
    for k in SCORE_DIMS:
        voted[k] = counts[k] > (NUM_ITER / 2.0)
    return voted


def compute_score(scores: dict) -> float:
    dims = [d for d in SCORE_DIMS if d != "hallucinations"]
    passed = sum(1 for d in dims if scores.get(d) is True)
    penalty = 1 if scores.get("hallucinations") is True else 0
    return round(max(0, passed - penalty) / len(dims) * 100, 1)


def print_summary(evaluated: list[dict]):
    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY — reactxen EvaluationAgent (model_id={MODEL_ID})")
    print(f"Judge: meta-llama/llama-3-3-70b-instruct | {NUM_ITER}x majority vote")
    print("=" * 70)
    print(f"{'ID':<5} {'Score':>6}  {'TC':>4} {'DR':>4} {'GV':>4} {'AS':>4} {'CJ':>4} {'HL':>4}  Question")
    print("-" * 70)

    all_scores = []
    for r in evaluated:
        sc = r.get("scores", {})
        overall = r.get("overall_score", 0.0)
        all_scores.append(overall)

        def fmt(key, invert=False):
            v = sc.get(key)
            if v is None:
                return " — "
            good = not v if invert else v
            return " ✓ " if good else " ✗ "

        print(
            f"{r['id']:<5} {overall:>5.1f}%  "
            f"{fmt('task_completion')}"
            f"{fmt('data_retrieval_accuracy')}"
            f"{fmt('generalized_result_verification')}"
            f"{fmt('agent_sequence_correct')}"
            f"{fmt('clarity_and_justification')}"
            f"{fmt('hallucinations', invert=True)}  "
            f"{r['question'][:35]}..."
        )

    print("-" * 70)
    avg = sum(all_scores) / len(all_scores) if all_scores else 0
    passed = len([s for s in all_scores if s >= 60])
    print(f"\nAverage: {avg:.1f}%  ({passed}/{len(all_scores)} scenarios ≥60%)")
    print("TC=Task Completion  DR=Data Retrieval  GV=Result Verification")
    print("AS=Agent Sequence   CJ=Clarity         HL=No Hallucinations (✓=good)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to JSON results file to evaluate")
    args = parser.parse_args()

    for var in ("WATSONX_APIKEY", "WATSONX_PROJECT_ID"):
        val = os.environ.get(var, "")
        if not val or val.startswith("PASTE"):
            print(f"ERROR: {var} not set in .env")
            sys.exit(1)

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results from {results_path}")
    print(f"Judge: meta-llama/llama-3-3-70b-instruct (model_id={MODEL_ID}), {NUM_ITER}x majority vote\n")

    selected_llm = partial(watsonx_llm, max_tokens=512, temperature=TEMP)
    eval_agent = EvaluationAgent(model_id=MODEL_ID, llm=selected_llm)

    evaluated = []
    for i, r in enumerate(results):
        print(f"  [{i+1}/{len(results)}] ID {r['id']}...", end=" ", flush=True)

        if r.get("status") == "error":
            r["scores"] = {}
            r["overall_score"] = 0.0
            print("skipped (agent error)")
            evaluated.append(r)
            continue

        iter_results = []
        for v in range(NUM_ITER):
            try:
                res = eval_agent.evaluate_response(
                    question=r["question"],
                    agent_think=r.get("agent_plan", ""),
                    agent_response=r.get("agent_answer", ""),
                    characteristic_answer=r["characteristic_answer"],
                )
                iter_results.append(res)
            except Exception as e:
                print(f"\n    iter {v+1} error: {e}")

        if not iter_results:
            r["scores"] = {"error": "all iterations failed"}
            r["overall_score"] = 0.0
        else:
            r["scores"] = majority_vote(iter_results)
            r["overall_score"] = compute_score(r["scores"])
            r["scores"]["_iter_results"] = iter_results

        evaluated.append(r)
        print(f"{r['overall_score']:.1f}%")

        if i < len(results) - 1:
            time.sleep(2)

    with open(results_path, "w") as f:
        json.dump(evaluated, f, indent=2)

    print_summary(evaluated)
    print(f"\nScores written back to {results_path}")


if __name__ == "__main__":
    main()
