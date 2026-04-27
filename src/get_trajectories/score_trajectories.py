import json, os, sys, statistics, re
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
import litellm

load_dotenv()

JUDGE_MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
N_CALLS = 5

PROMPT = """You are evaluating an industrial AI agent.

Question: {question}
Answer: {answer}
Trajectory: {trajectory}
Ground Truth Guide: {characteristic_form}

Score 0-10 on each dimension. You MUST respond with a JSON object and nothing else.
Do not explain. Do not add text before or after. Only output the JSON.

Example of valid response:
{{"task_completion": 7, "tool_selection": 8, "answer_correctness": 7,
  "reasoning_quality": 6, "data_handling": 7, "efficiency": 8, "pass": 1}}

Your response:"""

def extract_json(text):
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try finding a JSON object anywhere in the text
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None

def summarize(traj):
    lines = []
    for step in traj[:10]:
        if "tool" in step:
            lines.append(
                f"{step['tool']}({step['tool_args']}) -> {str(step['response'])[:150]}"
            )
    return "\n".join(lines) or "(no tool calls)"

def judge(q, answer, traj, cf):
    prompt = PROMPT.format(
        question=q,
        answer=answer,
        trajectory=summarize(traj),
        characteristic_form=cf
    )
    resp = litellm.completion(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
        api_key=os.environ["WATSONX_APIKEY"],
        project_id=os.environ["WATSONX_PROJECT_ID"],
    )
    raw = resp.choices[0].message.content or ""
    result = extract_json(raw)
    if result is None:
        raise ValueError(f"Could not parse JSON from: {repr(raw[:200])}")
    return result

def main():
    traj_file = Path(__file__).parent.parent.parent / "trajectories" / "all_trajectories.json"
    scores_file = Path(__file__).parent.parent.parent / "scores.json"

    if not traj_file.exists():
        print("No trajectories found. Run run_all_scenarios.py first.")
        sys.exit(1)

    ds = load_dataset("ibm-research/AssetOpsBench", "scenarios")
    char_forms = {r["id"]: r["characteristic_form"] for r in ds["train"].to_list()}

    data = json.loads(traj_file.read_text())
    results = []

    for i, rec in enumerate(data):
        sid = rec["id"]
        print(f"[{i+1}/{len(data)}] Scoring Q{sid}...")
        cf = char_forms.get(sid, "")
        scores = []
        for _ in range(N_CALLS):
            try:
                scores.append(judge(
                    rec["text"],
                    rec.get("answer", ""),
                    rec.get("trajectory", []),
                    cf
                ))
            except Exception as e:
                print(f"  judge error: {e}")

        if not scores:
            results.append({"id": sid, "type": rec["type"], "error": "all judge calls failed"})
            continue

        dims = [
            "task_completion", "tool_selection", "answer_correctness",
            "reasoning_quality", "data_handling", "efficiency"
        ]
        avg = {d: statistics.mean(s[d] for s in scores if d in s) for d in dims}
        avg["overall"] = statistics.mean(avg.values())
        avg["pass_rate"] = statistics.mean(s.get("pass", 0) for s in scores)
        results.append({"id": sid, "type": rec["type"], **avg})

        scores_file.write_text(json.dumps(results, indent=2))

    print(f"\nDone. Scores saved to {scores_file}")

if __name__ == "__main__":
    main()