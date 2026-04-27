import asyncio, json, os, sys
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
from huggingface_hub import login
hf_token = os.getenv("HF_APIKEY")
if hf_token:
    login(token=hf_token)

# Add src to path so agent/llm modules are found
sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_ID = os.getenv("MODEL_ID",
           "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8")

OUT_DIR = Path(__file__).parent.parent.parent / "trajectories"

CONCURRENCY = 1


async def run_one(runner, scenario):
    sid = scenario["id"]
    text = scenario["text"]
    print(f"[{sid}] {text}")
    try:
        result = await runner.run(text)
        traj = [
            {
                "step": r.step_number,
                "task": r.task,
                "server": r.agent,
                "tool": r.tool,
                "tool_args": r.tool_args,
                "response": r.response,
                "error": r.error,
                "success": r.success
            }
            for r in result.history
        ]
        return {
            "id": sid,
            "type": scenario["type"],
            "text": text,
            "answer": result.answer,
            "trajectory": traj,
            "error": None
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "id": sid,
            "type": scenario["type"],
            "text": text,
            "answer": "",
            "trajectory": [],
            "error": str(e)
        }

async def main():
    ds = load_dataset("ibm-research/AssetOpsBench", "scenarios")
    df = ds["train"].to_pandas()
    scenarios = df.to_dict(orient="records")
    print(f"Loaded {len(scenarios)} scenarios. Model: {MODEL_ID}")

    from llm.litellm import LiteLLMBackend
    from workflow.runner import PlanExecuteRunner
    runner = PlanExecuteRunner(llm=LiteLLMBackend(MODEL_ID))

    OUT_DIR.mkdir(exist_ok=True)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded(s):
        async with sem:
            r = await run_one(runner, s)
            (OUT_DIR / f"Q_{s['id']}.json").write_text(
                json.dumps(r, indent=2)
            )
            return r

    results = await asyncio.gather(*[bounded(s) for s in scenarios])
    (OUT_DIR / "all_trajectories.json").write_text(
        json.dumps(results, indent=2)
    )
    print(f"\nDone. {len(results)} trajectories saved to {OUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())