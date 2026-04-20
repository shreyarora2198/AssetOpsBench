"""eval_runner.py

Five-condition ablation:

  A. Raw LLM — no tools
  B. Tool baseline — unconditional tool chain (stand-in for ReAct; see note below)
  C. +Planning only — planner + skills, all executed, no knowledge / no skips
  D. +Skills + knowledge — full SkillAgent, conditional execution, **no** deep TSFM gating
  E. Full system — D plus conditional deep TSFM (RCA_CONFIDENCE_THETA)

Condition B is a static IoT→TSFM-lite→FMSR→WO pipeline without skill/registry
abstraction.  Swap in a true observe–act ReAct loop when ready.

For E, set RCA_CONFIDENCE_THETA to 0.7, 0.8, 0.9 (main.tex Sec.~\\ref{sec:thresh}).

Usage:
    python eval_runner.py
    # results → eval_results/ablation_results.csv
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

import dotenv

dotenv.load_dotenv()

from scenario_loader import BUILTIN_TASK_BANK

TASK_BANK = BUILTIN_TASK_BANK


def run_condition_a(task: str) -> dict:
    t0 = time.time()
    answer = ""
    system = (
        "You are an industrial asset operations agent. "
        "Answer the following question about equipment maintenance."
    )
    try:
        if os.getenv("LLM_PROVIDER") == "anthropic":
            from anthropic import Anthropic

            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": f"Task: {task}"}],
            )
            answer = resp.content[0].text.strip()
        elif os.getenv("LLM_PROVIDER") == "groq":
            from groq import Groq

            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=256,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Task: {task}"},
                ],
            )
            answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"error: {e}"

    return {
        "task": task,
        "plan": ["direct_llm"],
        "result": {"answer": answer},
        "metrics": {
            "plan": ["direct_llm"],
            "tool_calls": 1,
            "skills_skipped": [],
            "skipped_conditional": [],
            "skipped_early_stop": [],
            "total_cost": 0.0,
            "latency_s": round(time.time() - t0, 3),
            "diagnosis_confidence": None,
            "deep_tsfm_invoked": False,
        },
    }


def run_condition_b(task: str) -> dict:
    """Unconditional tool pipeline — no skill registry (ReAct stand-in)."""
    from agent import _extract_asset
    from tools import (
        detect_anomaly,
        forecast_sensor,
        generate_work_order,
        get_asset_metadata,
        get_sensor_data,
        map_failure,
    )

    asset_id = _extract_asset(task)
    t0 = time.time()
    context: dict = {}
    calls = 0
    t = task.lower()

    if any(w in t for w in ("what sensor", "metadata", "available")):
        context["metadata"] = get_asset_metadata(asset_id)
        calls = 1
    elif any(w in t for w in ("forecast", "predict", "next week", "future")):
        data = get_sensor_data(asset_id)
        calls += 1
        forecast = forecast_sensor(
            asset_id,
            "condenser_flow_GPM",
            horizon_days=7,
            sensor_data=data,
        )
        calls += 1
        context["sensor_data"] = data
        context["forecast"] = forecast
    else:
        data = get_sensor_data(asset_id)
        calls += 1
        anomaly = detect_anomaly(data)
        calls += 1
        failure = map_failure(anomaly, None, asset_id)
        calls += 1
        context["work_order"] = generate_work_order(asset_id, failure, "high")
        calls += 1
        context.update({"sensor_data": data, "anomaly_analysis": anomaly, "failure": failure})

    return {
        "task": task,
        "plan": ["B_static_tools"],
        "result": context,
        "metrics": {
            "plan": ["B_static_tools"],
            "tool_calls": calls,
            "skills_skipped": [],
            "skipped_conditional": [],
            "skipped_early_stop": [],
            "total_cost": round(calls * 0.25, 3),
            "latency_s": round(time.time() - t0, 3),
            "diagnosis_confidence": None,
            "deep_tsfm_invoked": False,
        },
    }


def run_condition_c(task: str) -> dict:
    """Planner + skills; run every skill in plan (no should_skip, no knowledge).

    Matches proposal Table~5: condition C disables knowledge injection and deep
    TSFM gating (E adds both).
    """
    from agent import SkillAgent, _extract_asset
    from skills import SKILL_REGISTRY

    # No confidence-based deep TSFM inside RCA (that is introduced in cond.~E).
    with _env_override("ENABLE_CONDITIONAL_DEEP_TSFM", "0"), _env_override(
        "KNOWLEDGE_INJECTION", "0"
    ):
        agent = SkillAgent()
        asset_id = _extract_asset(task)
        plan = agent.plan(task)
        context = {}
        cost = 0.0
        calls = 0
        t0 = time.time()

        for skill_name in plan:
            skill = SKILL_REGISTRY.get(skill_name)
            if not skill:
                continue
            try:
                result = skill["fn"](asset_id, context=context, task="")
                context.update(result.get("output", {}))
                cost += skill["cost"]
                calls += 1
            except Exception:
                pass

        return {
            "task": task,
            "plan": plan,
            "result": context,
            "metrics": {
                "plan": plan,
                "tool_calls": calls,
                "skills_skipped": [],
                "skipped_conditional": [],
                "skipped_early_stop": [],
                "total_cost": round(cost, 3),
                "latency_s": round(time.time() - t0, 3),
                "diagnosis_confidence": context.get("diagnosis_confidence"),
                "deep_tsfm_invoked": bool(context.get("deep_tsfm_invoked", False)),
            },
        }


def run_condition_d(task: str) -> dict:
    """Full SkillAgent with knowledge; disable deep TSFM gating (proposal cond.~D)."""
    with _env_override("ENABLE_CONDITIONAL_DEEP_TSFM", "0"):
        from agent import SkillAgent

        return SkillAgent().run(task)


def run_condition_e(task: str) -> dict:
    """Full system including conditional deep TSFM (proposal cond.~E)."""
    from agent import SkillAgent

    with _env_override("ENABLE_CONDITIONAL_DEEP_TSFM", "1"):
        return SkillAgent().run(task)


@contextmanager
def _env_override(key: str, value: str):
    prev = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


THETA_VALUES = ("0.7", "0.8", "0.9")


def evaluate_all(
    output_dir: str = "eval_results",
    *,
    task_bank: list[tuple[str, str, str]] | None = None,
    trajectory_log_path: str | None = None,
) -> None:
    """Run ablations. Pass ``task_bank=load_hf_scenario_tasks(limit=...)`` for HF scenarios.

    If ``trajectory_log_path`` is set, append one JSON object per task × condition
    (summarized context; see ``trajectory_log``).
    """
    tasks = task_bank if task_bank is not None else TASK_BANK
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    static_conditions = [
        ("A_raw_llm", run_condition_a),
        ("B_tool_baseline", run_condition_b),
        ("C_planning_only", run_condition_c),
        ("D_skills_knowledge_no_deep_tsfm", run_condition_d),
    ]

    for cond_name, run_fn in static_conditions:
        print(f"\n{'─' * 60}\nCondition: {cond_name}\n{'─' * 60}")
        for task_id, task, category in tasks:
            print(f"  {task_id} [{category}] {task[:55]}...")
            _append_row(
                rows,
                cond_name,
                "",
                task_id,
                category,
                task,
                run_fn,
                trajectory_log_path=trajectory_log_path,
            )

    for theta in THETA_VALUES:
        cond_name = f"E_full_theta_{theta.replace('.', '_')}"
        print(f"\n{'─' * 60}\nCondition: {cond_name} (RCA_CONFIDENCE_THETA={theta})\n{'─' * 60}")
        with _env_override("RCA_CONFIDENCE_THETA", theta):
            for task_id, task, category in tasks:
                print(f"  {task_id} [{category}] {task[:55]}...")
                _append_row(
                    rows,
                    cond_name,
                    theta,
                    task_id,
                    category,
                    task,
                    run_condition_e,
                    trajectory_log_path=trajectory_log_path,
                )

    csv_path = Path(output_dir) / "ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Results -> {csv_path}")
    _print_summary(rows)


def _append_row(
    rows: list,
    cond_name: str,
    theta: str,
    task_id: str,
    category: str,
    task: str,
    run_fn,
    *,
    trajectory_log_path: str | None = None,
) -> None:
    try:
        out = run_fn(task)
        m = out.get("metrics", {})
        if trajectory_log_path:
            from trajectory_log import append_trajectory_line, build_eval_trajectory

            append_trajectory_line(
                build_eval_trajectory(
                    condition=cond_name,
                    theta=theta,
                    task_id=task_id,
                    category=category,
                    task=task,
                    run_output=out,
                ),
                path=trajectory_log_path,
            )
        rows.append(
            {
                "condition": cond_name,
                "theta": theta,
                "task_id": task_id,
                "category": category,
                "task": task[:80],
                "plan": json.dumps(m.get("plan", [])),
                "tool_calls": m.get("tool_calls", -1),
                "skipped_conditional": len(m.get("skipped_conditional", [])),
                "skipped_early_stop": len(m.get("skipped_early_stop", [])),
                "skills_skipped": len(m.get("skills_skipped", [])),
                "total_cost": m.get("total_cost", -1),
                "latency_s": m.get("latency_s", -1),
                "diagnosis_confidence": m.get("diagnosis_confidence", ""),
                "deep_tsfm_invoked": m.get("deep_tsfm_invoked", ""),
                "task_completion": "",
                "error": "",
            }
        )
        print(
            f"    ok  calls={m.get('tool_calls')} "
            f"deep_tsfm={m.get('deep_tsfm_invoked')} "
            f"cost={m.get('total_cost')} lat={m.get('latency_s')}s"
        )
    except Exception as e:
        logger.error("    fail %s: %s", task_id, e)
        rows.append(
            {
                "condition": cond_name,
                "theta": theta,
                "task_id": task_id,
                "category": category,
                "task": task[:80],
                "plan": "",
                "tool_calls": -1,
                "skipped_conditional": -1,
                "skipped_early_stop": -1,
                "skills_skipped": -1,
                "total_cost": -1,
                "latency_s": -1,
                "diagnosis_confidence": "",
                "deep_tsfm_invoked": "",
                "task_completion": "",
                "error": str(e),
            }
        )


FIELDS = [
    "condition",
    "theta",
    "task_id",
    "category",
    "task",
    "plan",
    "tool_calls",
    "skipped_conditional",
    "skipped_early_stop",
    "skills_skipped",
    "total_cost",
    "latency_s",
    "diagnosis_confidence",
    "deep_tsfm_invoked",
    "task_completion",
    "error",
]


def _print_summary(rows: list) -> None:
    from collections import defaultdict

    by_cond = defaultdict(list)
    for r in rows:
        if isinstance(r["tool_calls"], int) and r["tool_calls"] >= 0:
            by_cond[r["condition"]].append(r)

    print(f"\n{'Condition':<38} {'n':>4} {'avg_calls':>10} {'avg_cost':>10} {'avg_lat(s)':>12}")
    for cond, task_rows in sorted(by_cond.items()):
        n = len(task_rows)
        avg = lambda k: sum(float(r[k]) for r in task_rows) / n
        print(f"{cond:<38} {n:>4} {avg('tool_calls'):>10.1f} {avg('total_cost'):>10.3f} {avg('latency_s'):>12.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation conditions A–E on a task bank.")
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Directory for ablation_results.csv",
    )
    parser.add_argument(
        "--hf-limit",
        type=int,
        default=None,
        metavar="N",
        help="If set, load up to N scenarios from Hugging Face (falls back to builtin on error).",
    )
    parser.add_argument(
        "--trajectory-log",
        default=None,
        metavar="PATH",
        help="Append JSONL trajectory records (one per task × condition) to this file.",
    )
    args = parser.parse_args()
    task_bank = None
    if args.hf_limit is not None:
        from scenario_loader import load_hf_scenario_tasks

        task_bank = load_hf_scenario_tasks(limit=args.hf_limit)
    evaluate_all(
        args.output_dir,
        task_bank=task_bank,
        trajectory_log_path=args.trajectory_log,
    )
