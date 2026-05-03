"""
evaluate_baseline.py — Run real AssetOpsBench FMSA scenarios through the
baseline plan-execute agent, then score each answer using GPT-4o as judge.

Uses the same 6-dimension evaluation criteria as the repo's EvaluationAgent
(src/tmp/evaluation_agent/) but calls OpenAI instead of WatsonX.

No Docker / CouchDB required — runs FMSA (Failure Mode Sensor Analysis)
scenarios only.

Usage:
    uv run python evaluate_baseline.py

Results saved to:
    results/evaluated_baseline.json   — full results with scores
    results/evaluated_baseline_summary.txt  — human-readable summary table
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import litellm
from datasets import load_dataset
from llm import LLMBackend
from workflow import PlanExecuteRunner


# ── FMSA scenario IDs that work without Docker (no IoT/CouchDB needed) ────────

FMSA_NO_DOCKER = [101, 102, 103, 106, 107, 108, 110, 111, 112, 113,
                  115, 116, 117, 118, 119, 120]


# ── Evaluation prompt (from src/tmp/evaluation_agent/result_evaluation_prompt.py)

EVAL_PROMPT = """You are a critical reviewer tasked with evaluating the effectiveness and accuracy of an AI agent's response to a given task. Your goal is to determine whether the agent has successfully accomplished the task correctly based on the expected or characteristic behavior.

Evaluation Criteria:
1. **Task Completion:** Verify if the agent executed all necessary actions and the response aligns with expected behavior.
2. **Data Retrieval & Accuracy:** Ensure the correct asset, sensor, and time period were used and data is accurate.
3. **Generalized Result Verification:** Verify the result matches the expected format and values from the characteristic answer.
4. **Agent Sequence & Order:** Ensure agents were called in the correct order matching expected behavior.
5. **Clarity and Justification:** Ensure the response is clear and justified with no contradictions.
6. **Hallucination Check:** Identify if the agent claims success without performing necessary actions.

Question: {question}
Characteristic Answer (Expected Behavior): {characteristic_answer}
Agent's Thinking: {agent_think}
Agent's Final Response: {agent_response}

Output Format:
Respond with ONLY a valid JSON object, no markdown, no extra text:
{{
    "task_completion": true/false,
    "data_retrieval_accuracy": true/false,
    "generalized_result_verification": true/false,
    "agent_sequence_correct": true/false,
    "clarity_and_justification": true/false,
    "hallucinations": true/false,
    "suggestions": "optional improvements"
}}
(END OF RESPONSE)"""


# ── OpenAI backend ─────────────────────────────────────────────────────────────

class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY in your .env file")
        self._model = model
        self._api_key = api_key

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        response = litellm.completion(
            model=f"openai/{self._model}",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2048,
            api_key=self._api_key,
        )
        return response.choices[0].message.content


# ── Judge: score one result ────────────────────────────────────────────────────

def score_result(question: str, characteristic_answer: str,
                 agent_plan: str, agent_answer: str) -> dict:
    """Judge the agent's answer using WatsonX (matches AssetOpsBench paper evaluator).
    Falls back to OpenAI if WatsonX credentials are not set."""
    prompt = EVAL_PROMPT.format(
        question=question,
        characteristic_answer=characteristic_answer,
        agent_think=agent_plan,
        agent_response=agent_answer,
    )

    # Choose judge: WatsonX if credentials present, else OpenAI
    watsonx_key = os.environ.get("WATSONX_APIKEY")
    judge_model = os.environ.get("EVAL_JUDGE_MODEL", "watsonx/meta-llama/llama-3-3-70b-instruct")

    if watsonx_key and not watsonx_key.startswith("PASTE"):
        kwargs = {
            "model": judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 512,
            "api_key": watsonx_key,
            "project_id": os.environ.get("WATSONX_PROJECT_ID"),
        }
        if url := os.environ.get("WATSONX_URL"):
            kwargs["api_base"] = url
        print("  [judge: WatsonX]", end=" ", flush=True)
    else:
        openai_key = os.environ.get("OPENAI_API_KEY")
        kwargs = {
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 512,
            "api_key": openai_key,
        }
        print("  [judge: OpenAI gpt-4o]", end=" ", flush=True)

    for attempt in range(3):
        try:
            response = litellm.completion(**kwargs)
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(raw[start:end])
        except Exception as e:
            if attempt == 2:
                return {"error": str(e)}
            time.sleep(5)
    return {"error": "Failed to parse judge response"}


def compute_score(scores: dict) -> float:
    """Convert 6-dimension boolean scores to a 0-100 score.
    hallucinations=true is bad (penalizes), all others true is good."""
    dims = ["task_completion", "data_retrieval_accuracy",
            "generalized_result_verification", "agent_sequence_correct",
            "clarity_and_justification"]
    passed = sum(1 for d in dims if scores.get(d) is True)
    hallucination_penalty = 1 if scores.get("hallucinations") is True else 0
    raw = passed - hallucination_penalty
    return round(max(0, raw) / len(dims) * 100, 1)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def load_fmsa_scenarios(ids: list[int]) -> list[dict]:
    print("Loading scenarios from HuggingFace (ibm-research/AssetOpsBench)...")
    ds = load_dataset("ibm-research/AssetOpsBench", "scenarios")
    df = ds["train"].to_pandas()
    filtered = df[df["id"].isin(ids)]
    return filtered.to_dict(orient="records")


async def run_agent(scenarios: list[dict]) -> list[dict]:
    llm = OpenAIBackend(model="gpt-4o-mini")
    runner = PlanExecuteRunner(
        llm=llm,
        server_paths={
            "FMSRAgent": "fmsr-mcp-server",
            "Utilities": "utilities-mcp-server",
        },
    )

    results = []
    for i, s in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}] ID {s['id']}: {s['text'][:70]}...")
        try:
            result = await runner.run(s["text"])
            plan_text = " → ".join(
                f"{st.agent}:{st.tool}" for st in result.plan.steps
            )
            results.append({
                "id": s["id"],
                "question": s["text"],
                "characteristic_answer": s["characteristic_form"],
                "agent_plan": plan_text,
                "agent_answer": result.answer,
                "plan_steps": [
                    {"step": st.step_number, "agent": st.agent,
                     "task": st.task, "tool": st.tool}
                    for st in result.plan.steps
                ],
                "status": "success",
            })
            print(f"  Plan: {plan_text}")
            print(f"  Answer: {result.answer[:120]}...")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "id": s["id"],
                "question": s["text"],
                "characteristic_answer": s["characteristic_form"],
                "agent_plan": "",
                "agent_answer": "",
                "plan_steps": [],
                "status": "error",
                "error": str(e),
            })

        if i < len(scenarios) - 1:
            time.sleep(2)  # small buffer, OpenAI has high rate limits

    return results


def evaluate_results(results: list[dict]) -> list[dict]:
    print("\n\nScoring results with judge...")
    evaluated = []
    for i, r in enumerate(results):
        print(f"  Scoring {i+1}/{len(results)} — ID {r['id']}...")
        if r["status"] == "error":
            r["scores"] = {}
            r["overall_score"] = 0.0
            evaluated.append(r)
            continue

        scores = score_result(
            question=r["question"],
            characteristic_answer=r["characteristic_answer"],
            agent_plan=r["agent_plan"],
            agent_answer=r["agent_answer"],
        )
        r["scores"] = scores
        r["overall_score"] = compute_score(scores)
        evaluated.append(r)
        time.sleep(3)

    return evaluated


def print_summary(evaluated: list[dict]):
    print("\n" + "="*70)
    print("EVALUATION SUMMARY — Baseline Agent (gpt-4o-mini) on FMSA Scenarios")
    print("="*70)

    header = f"{'ID':<5} {'Score':>6}  {'TC':>4} {'DR':>4} {'GV':>4} {'AS':>4} {'CJ':>4} {'HL':>4}  Question"
    print(header)
    print("-"*70)

    scores = []
    for r in evaluated:
        sc = r.get("scores", {})
        overall = r.get("overall_score", 0.0)
        scores.append(overall)

        def fmt(key, invert=False):
            v = sc.get(key)
            if v is None:
                return " — "
            good = not v if invert else v
            return " ✓ " if good else " ✗ "

        row = (
            f"{r['id']:<5} {overall:>5.1f}%  "
            f"{fmt('task_completion')}"
            f"{fmt('data_retrieval_accuracy')}"
            f"{fmt('generalized_result_verification')}"
            f"{fmt('agent_sequence_correct')}"
            f"{fmt('clarity_and_justification')}"
            f"{fmt('hallucinations', invert=True)}  "
            f"{r['question'][:35]}..."
        )
        print(row)

    print("-"*70)
    avg = sum(scores) / len(scores) if scores else 0
    print(f"\nAverage score: {avg:.1f}%  ({len([s for s in scores if s >= 60])}/{len(scores)} scenarios passed ≥60%)")
    print("\nDimensions: TC=Task Completion  DR=Data Retrieval  GV=Result Verification")
    print("            AS=Agent Sequence  CJ=Clarity  HL=No Hallucinations (✓=good)")


def save_summary(evaluated: list[dict], path: Path):
    lines = ["EVALUATION SUMMARY — Baseline Agent (gpt-4o-mini)\n",
             f"{'ID':<5} {'Score':>6}  Question\n", "-"*60 + "\n"]
    scores = []
    for r in evaluated:
        overall = r.get("overall_score", 0.0)
        scores.append(overall)
        lines.append(f"{r['id']:<5} {overall:>5.1f}%  {r['question'][:50]}\n")
    avg = sum(scores) / len(scores) if scores else 0
    lines.append(f"\nAverage: {avg:.1f}%\n")
    path.write_text("".join(lines))


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY in your .env file")
        sys.exit(1)

    print("AssetOpsBench — Baseline Evaluation (FMSA scenarios, no Docker)")
    print(f"Running {len(FMSA_NO_DOCKER)} scenarios...\n")

    scenarios = load_fmsa_scenarios(FMSA_NO_DOCKER)
    print(f"Loaded {len(scenarios)} scenarios from HuggingFace\n")

    # Run agent
    results = asyncio.run(run_agent(scenarios))

    # Score with judge
    evaluated = evaluate_results(results)

    # Save
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / "evaluated_baseline.json"
    with open(json_path, "w") as f:
        json.dump(evaluated, f, indent=2)

    txt_path = out_dir / "evaluated_baseline_summary.txt"
    save_summary(evaluated, txt_path)

    print_summary(evaluated)
    print(f"\nFull results: {json_path}")
    print(f"Summary:      {txt_path}")


if __name__ == "__main__":
    main()
