"""
run_evaluation.py — Score pre-run agent results using the AssetOpsBench
EvaluationAgent (same prompt, same 6-dimension JSON), backed by WatsonX
via litellm instead of the IBM-internal reactxen package.

Usage:
    uv run python run_evaluation.py --results results/evaluated_baseline.json

    # Or point at any other results file:
    uv run python run_evaluation.py --results results/skills_knowledge_results.json

The results file must be a JSON array where each element has at minimum:
    {
        "id": <int>,
        "question": <str>,
        "characteristic_answer": <str>,
        "agent_plan": <str>,          # agent's thinking / plan steps
        "agent_answer": <str>,        # agent's final response
        "status": "success" | "error"
    }

Scores are written back into the same file and a summary is printed.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import litellm

# Reuse the exact prompt from the repo
sys.path.insert(0, str(Path(__file__).parent / "src"))
from tmp.evaluation_agent.result_evaluation_prompt import system_prompt_template


# ── WatsonX LLM call via litellm ──────────────────────────────────────────────

def _watsonx_call(prompt: str) -> str:
    """Call WatsonX llama-3-3-70b-instruct via litellm, matching the paper's judge."""
    api_key = os.environ["WATSONX_APIKEY"]
    project_id = os.environ["WATSONX_PROJECT_ID"]
    api_base = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    model = os.environ.get("EVAL_JUDGE_MODEL", "watsonx/meta-llama/llama-3-3-70b-instruct")

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
        api_key=api_key,
        project_id=project_id,
        api_base=api_base,
        stop=["\n(END OF RESPONSE)"],
    )
    return response.choices[0].message.content


# ── EvaluationAgent — mirrors src/tmp/evaluation_agent/agent.py ───────────────

class EvaluationAgent:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def _extract_json(self, response: str) -> dict:
        try:
            match = re.search(r"\{.*\}", response.strip(), re.DOTALL)
            if match:
                return json.loads(match.group(0).strip())
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            return {"status": "Error", "reasoning": str(e)}

    def _extract_json_manual(self, response: str) -> dict:
        cleaned = response.strip().replace("\n", " ").replace("\\n", " ").replace("\\", "")
        # Try the 6-dimension keys used in result_evaluation_prompt
        keys = [
            "task_completion", "data_retrieval_accuracy",
            "generalized_result_verification", "agent_sequence_correct",
            "clarity_and_justification", "hallucinations",
        ]
        result = {}
        for key in keys:
            m = re.search(rf'"{key}":\s*(true|false)', cleaned, re.IGNORECASE)
            if m:
                result[key] = m.group(1).lower() == "true"
        if result:
            sugg = re.search(r'"suggestions":\s*"([^"]+)"', cleaned)
            result["suggestions"] = sugg.group(1) if sugg else ""
            return result
        return {"status": "Error", "reasoning": "Could not parse 6-dimension JSON"}

    def evaluate_response(self, question: str, agent_think: str,
                          agent_response: str, characteristic_answer: str) -> dict:
        prompt = system_prompt_template.format(
            question=question,
            agent_think=agent_think,
            agent_response=agent_response,
            characteristic_answer=characteristic_answer,
        )

        for attempt in range(self.max_retries):
            try:
                raw = _watsonx_call(prompt)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {"status": "Error", "reasoning": str(e)}
                time.sleep(5)
                continue

            parsed = self._extract_json(raw)
            if parsed.get("status") != "Error" and "task_completion" in parsed:
                return parsed

            parsed = self._extract_json_manual(raw)
            if "task_completion" in parsed:
                return parsed

            # Refine prompt on failure
            refinement = (
                "Your previous response had JSON formatting errors. "
                "Respond with ONLY a valid JSON object containing exactly these keys: "
                "task_completion, data_retrieval_accuracy, generalized_result_verification, "
                "agent_sequence_correct, clarity_and_justification, hallucinations, suggestions.\n"
                f"Previous response: {raw}"
            )
            prompt = f"{prompt}\n\n{refinement}"

        return {"status": "Error", "reasoning": f"Failed after {self.max_retries} attempts"}


# ── Scoring helper ─────────────────────────────────────────────────────────────

def compute_score(scores: dict) -> float:
    dims = ["task_completion", "data_retrieval_accuracy",
            "generalized_result_verification", "agent_sequence_correct",
            "clarity_and_justification"]
    passed = sum(1 for d in dims if scores.get(d) is True)
    penalty = 1 if scores.get("hallucinations") is True else 0
    return round(max(0, passed - penalty) / len(dims) * 100, 1)


# ── Summary printing ───────────────────────────────────────────────────────────

def print_summary(evaluated: list[dict]):
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY — WatsonX llama-3-3-70b-instruct judge")
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to JSON results file to evaluate")
    args = parser.parse_args()

    # Check WatsonX credentials
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
    print(f"Judge: {os.environ.get('EVAL_JUDGE_MODEL', 'watsonx/meta-llama/llama-3-3-70b-instruct')}\n")

    agent = EvaluationAgent(max_retries=3)
    evaluated = []

    for i, r in enumerate(results):
        print(f"  [{i+1}/{len(results)}] ID {r['id']}...", end=" ", flush=True)

        if r.get("status") == "error":
            r["scores"] = {}
            r["overall_score"] = 0.0
            print("skipped (agent error)")
            evaluated.append(r)
            continue

        scores = agent.evaluate_response(
            question=r["question"],
            agent_think=r.get("agent_plan", ""),
            agent_response=r.get("agent_answer", ""),
            characteristic_answer=r["characteristic_answer"],
        )
        r["scores"] = scores
        r["overall_score"] = compute_score(scores)
        evaluated.append(r)
        print(f"{r['overall_score']:.1f}%")

        if i < len(results) - 1:
            time.sleep(2)

    # Write scores back into the same file
    with open(results_path, "w") as f:
        json.dump(evaluated, f, indent=2)

    print_summary(evaluated)
    print(f"\nScores written back to {results_path}")


if __name__ == "__main__":
    main()
