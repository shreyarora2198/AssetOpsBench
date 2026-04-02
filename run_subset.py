"""
run_subset.py — Run a small subset of AssetOpsBench scenarios against
the plan-execute orchestrator using OpenAI as the LLM backend.

Skips IoT and TSFM scenarios (require Docker/CouchDB and model checkpoints).
Runs FMSR + Utilities scenarios only.

Usage:
    uv run python run_subset.py

Results are saved to results/subset_results.json
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Make src packages importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import litellm
from llm import LLMBackend
from workflow import PlanExecuteRunner


# ── Custom LLM backend — calls OpenAI directly via litellm ────────────────────

class OpenAIBackend(LLMBackend):
    """Calls OpenAI directly without needing a LiteLLM proxy server."""

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


# ── Scenarios to test (no Docker/CouchDB needed) ──────────────────────────────

SCENARIOS = [
    {
        "id": "util_1",
        "domain": "Utilities",
        "text": "What is the current date and time?",
    },
    {
        "id": "fmsr_1",
        "domain": "FMSR",
        "text": "What are the failure modes for a chiller?",
    },
    {
        "id": "fmsr_2",
        "domain": "FMSR",
        "text": "What are the failure modes for an AHU (air handling unit)?",
    },
    {
        "id": "fmsr_3",
        "domain": "FMSR",
        "text": (
            "For a chiller, which sensors can help monitor or detect "
            "'Compressor Overheating'? List the relevant sensors and explain why."
        ),
    },
    {
        "id": "fmsr_4",
        "domain": "FMSR",
        "text": (
            "What are the failure modes for a chiller, and what is the current time?"
        ),
    },
]


# ── Runner ─────────────────────────────────────────────────────────────────────

async def run_all(scenarios: list[dict]) -> list[dict]:
    llm = OpenAIBackend(model="gpt-4o-mini")

    # Register only the servers we can run (no IoTAgent, no TSFMAgent)
    runner = PlanExecuteRunner(
        llm=llm,
        server_paths={
            "FMSRAgent": "fmsr-mcp-server",
            "Utilities": "utilities-mcp-server",
        },
    )

    results = []
    for s in scenarios:
        print(f"\n{'='*60}")
        print(f"[{s['id']}] ({s['domain']}) {s['text']}")
        print("="*60)
        try:
            result = await runner.run(s["text"])
            print(f"\nPLAN:")
            for step in result.plan.steps:
                deps = f" (depends on {step.dependencies})" if step.dependencies else ""
                print(f"  Step {step.step_number} [{step.agent}]: {step.task}{deps}")
            print(f"\nANSWER:\n{result.answer}")
            entry = {
                "id": s["id"],
                "domain": s["domain"],
                "question": s["text"],
                "answer": result.answer,
                "plan": [
                    {
                        "step": st.step_number,
                        "agent": st.agent,
                        "task": st.task,
                        "tool": st.tool,
                    }
                    for st in result.plan.steps
                ],
                "status": "success",
            }
        except Exception as e:
            print(f"\nERROR: {e}")
            entry = {
                "id": s["id"],
                "domain": s["domain"],
                "question": s["text"],
                "answer": None,
                "plan": [],
                "status": "error",
                "error": str(e),
            }
        results.append(entry)
        if len(results) < len(scenarios):
            time.sleep(20)  # avoid Groq free-tier TPM rate limit

    return results


def main():
    print("AssetOpsBench — Subset Run (FMSR + Utilities, no Docker needed)")
    print(f"Running {len(SCENARIOS)} scenarios...\n")

    results = asyncio.run(run_all(SCENARIOS))

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "subset_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    print(f"\n{'='*60}")
    print(f"SUMMARY: {success}/{len(results)} scenarios succeeded")
    print(f"Results saved to: {out_file}")

    for r in results:
        status = "OK" if r["status"] == "success" else "FAIL"
        print(f"  [{status}] {r['id']} ({r['domain']}): {r['question'][:60]}...")


if __name__ == "__main__":
    main()
