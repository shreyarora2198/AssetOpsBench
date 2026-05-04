#!/usr/bin/env python3
"""Compute the six AssetOps EvaluationAgent metrics per ablation row.

Loads rubrics from ``AssetOpsBench/.../tsfm_utterance.json``, reads
``ablation_results.csv`` (``result_json`` / ``trace_json``), and calls the same
``evaluation_agent`` used by the TSFM scenario server — **via uv** in
``AssetOpsBench/aobench/scenario-server`` (needs ReActXen).

Example::

    cd SkillsAgent
    python scripts/grade_assetops_metrics.py \\
        --ablation-csv eval_results/colab_20260503_2349/ablation_results.csv \\
        --aobench-root ../AssetOpsBench/aobench \\
        --out-csv eval_results/colab_20260503_2349/assetops_metrics.csv \\
        --pivot-csv eval_results/colab_20260503_2349/assetops_metrics_by_condition.csv

First sync the grader stack once::

    cd ../AssetOpsBench/aobench && uv sync

Environment (optional)::

    export ASSETOPS_RUBRIC_JSON=/path/to/tsfm_utterance.json

The ReActXen ``EvaluationAgent`` (model id 16) talks to **IBM watsonx / WML** and
needs the same credentials as SkillsAgent: set ``WATSONX_API_KEY`` and
``WATSONX_PROJECT_ID`` in ``SkillsAgent/.env`` (this script loads that file before
``uv run``) or export them in the shell.

If ``--aobench-root`` is omitted, the script builds the payload and exits 0 after
printing how many rows matched (use ``--print-payload`` to dump JSON).

**Pivot:** ``*_rate`` columns are the fraction of ``True`` values. For
``hallucinations``, ``True`` means the grader detected hallucinations — a
*lower* ``hallucinations_rate`` is better.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bootstrap_env_for_grader() -> None:
    """Load ``.env`` files and align IBM env names so the uv subprocess inherits keys."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = _skills_root()
    load_dotenv(root / ".env")
    aobench_env = root.parent / "AssetOpsBench" / "aobench" / ".env"
    if aobench_env.is_file():
        load_dotenv(aobench_env)
    key = (os.environ.get("WATSONX_API_KEY") or "").strip()
    if key and not (os.environ.get("WATSONX_APIKEY") or "").strip():
        os.environ["WATSONX_APIKEY"] = key


def _default_rubric_path() -> Path:
    return (
        _skills_root().parent
        / "AssetOpsBench"
        / "src"
        / "tmp"
        / "assetopsbench"
        / "scenarios"
        / "single_agent"
        / "tsfm_utterance.json"
    )


def _load_rubric(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    for row in data:
        if str(row.get("type", "")).upper() != "TSFM":
            continue
        sid = str(row.get("id", "")).strip()
        if not sid:
            continue
        out[sid] = {
            "text": str(row.get("text", "")),
            "characteristic_form": str(row.get("characteristic_form", "")),
        }
    return out


def _result_to_string(result_obj: object) -> str:
    if isinstance(result_obj, dict):
        ans = result_obj.get("answer")
        if isinstance(ans, str) and ans.strip():
            return ans
        return json.dumps(result_obj, ensure_ascii=False)
    if result_obj in (None, ""):
        return ""
    return json.dumps(result_obj, ensure_ascii=False)


def _trace_to_string(trace_obj: object) -> str:
    if isinstance(trace_obj, str):
        return trace_obj
    return json.dumps(trace_obj, ensure_ascii=False, default=str)


def _build_payload(
    ablation_csv: Path,
    rubric: dict[str, dict[str, str]],
) -> list[dict]:
    csv.field_size_limit(min(2**31 - 1, 10_000_000))
    rows_out: list[dict] = []
    with open(ablation_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tid = str(row.get("task_id") or row.get("scenario_id") or "").strip()
            if not tid or tid not in rubric:
                continue
            err = (row.get("error") or "").strip()
            if err:
                continue
            rj = row.get("result_json") or ""
            tj = row.get("trace_json") or ""
            try:
                result_obj = json.loads(rj) if rj else {}
            except json.JSONDecodeError:
                continue
            try:
                trace_obj = json.loads(tj) if tj else {}
            except json.JSONDecodeError:
                trace_obj = {}
            spec = rubric[tid]
            rows_out.append(
                {
                    "task_id": tid,
                    "condition": row.get("condition", ""),
                    "theta": row.get("theta", "") or "",
                    "query": spec["text"],
                    "characteristic_form": spec["characteristic_form"],
                    "result": _result_to_string(result_obj),
                    "trace": _trace_to_string(trace_obj),
                    "model_id": 16,
                }
            )
    return rows_out


def _run_worker(aobench_root: Path, worker: Path, payload_path: Path) -> list[dict]:
    proc = subprocess.run(
        [
            "uv",
            "run",
            "--directory",
            str(aobench_root / "scenario-server"),
            "python",
            str(worker),
            str(payload_path),
        ],
        cwd=str(aobench_root),
        capture_output=True,
        text=True,
        timeout=7200,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"grader worker failed: {msg}")
    return json.loads(proc.stdout)


def _coerce_bool_metric(v: object) -> float | None:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if v in (0, 1):
        return float(v)
    if v is None or v == "":
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return 1.0
    if s in ("false", "0", "no"):
        return 0.0
    return None


def _aggregate_by_condition(graded: list[dict]) -> list[dict]:
    """Mean of boolean-ish metrics per ``condition`` (and ``theta`` when present)."""
    from collections import defaultdict

    metric_cols = [
        "overall_correct",
        "task_completion",
        "data_retrieval_accuracy",
        "generalized_result_verification",
        "agent_sequence_correct",
        "clarity_and_justification",
        "hallucinations",
    ]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in graded:
        if r.get("error"):
            continue
        key = (str(r.get("condition", "")), str(r.get("theta", "")))
        groups[key].append(r)
    out: list[dict] = []
    for (cond, theta), rs in sorted(groups.items()):
        n = len(rs)
        if n == 0:
            continue
        row: dict = {"condition": cond, "theta": theta, "n": n}
        for col in metric_cols:
            nums: list[float] = []
            for r in rs:
                if col not in r:
                    continue
                c = _coerce_bool_metric(r.get(col))
                if c is not None:
                    nums.append(c)
            if nums:
                row[f"{col}_rate"] = round(sum(nums) / len(nums), 4)
        out.append(row)
    return out


def _write_metrics_csv(path: Path, graded: list[dict]) -> None:
    if not graded:
        path.write_text("", encoding="utf-8")
        return
    # union of keys across rows for header stability
    keys: list[str] = []
    seen: set[str] = set()
    priority = [
        "task_id",
        "condition",
        "theta",
        "overall_correct",
        "evaluator_error",
        "task_completion",
        "data_retrieval_accuracy",
        "generalized_result_verification",
        "agent_sequence_correct",
        "clarity_and_justification",
        "hallucinations",
        "suggestions",
        "error",
    ]
    for k in priority:
        if any(k in r for r in graded) and k not in seen:
            keys.append(k)
            seen.add(k)
    for r in graded:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in graded:
            w.writerow({k: r.get(k, "") for k in keys})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ablation-csv",
        type=Path,
        default=None,
        help="Path to ablation_results.csv from eval_runner (not needed with --only-pivot)",
    )
    ap.add_argument(
        "--rubric-json",
        type=Path,
        default=None,
        help="Path to tsfm_utterance.json (default: sibling AssetOpsBench path)",
    )
    ap.add_argument(
        "--aobench-root",
        type=Path,
        default=None,
        help="Path to AssetOpsBench/aobench (enables uv grader subprocess)",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV (default: next to ablation csv: assetops_metrics.csv)",
    )
    ap.add_argument(
        "--print-payload",
        action="store_true",
        help="Print JSON payload to stdout and exit (no grading)",
    )
    ap.add_argument(
        "--pivot-csv",
        type=Path,
        default=None,
        help="After grading, write per-condition mean rates to this CSV",
    )
    ap.add_argument(
        "--only-pivot",
        type=Path,
        default=None,
        help="Read existing per-row metrics CSV and only write --pivot-csv aggregates",
    )
    args = ap.parse_args()

    if args.only_pivot is not None:
        if not args.pivot_csv:
            print("--only-pivot requires --pivot-csv", file=sys.stderr)
            return 1
        src = args.only_pivot.expanduser().resolve()
        if not src.is_file():
            print(f"Not found: {src}", file=sys.stderr)
            return 1
        with open(src, newline="", encoding="utf-8") as fh:
            graded = list(csv.DictReader(fh))
        pivot = args.pivot_csv.expanduser().resolve()
        _write_metrics_csv(pivot, _aggregate_by_condition(graded))
        print(f"Wrote aggregates to {pivot}")
        return 0

    if args.ablation_csv is None:
        print("Either --ablation-csv or --only-pivot is required.", file=sys.stderr)
        return 1

    rubric_path = args.rubric_json
    if rubric_path is None:
        env_p = os.environ.get("ASSETOPS_RUBRIC_JSON")
        rubric_path = Path(env_p).resolve() if env_p else _default_rubric_path()
    rubric_path = rubric_path.resolve()
    if not rubric_path.is_file():
        print(f"Rubric not found: {rubric_path}", file=sys.stderr)
        return 1

    rubric = _load_rubric(rubric_path)
    ablation = args.ablation_csv.expanduser().resolve()
    if not ablation.is_file():
        print(f"Ablation CSV not found: {ablation}", file=sys.stderr)
        return 1

    payload = _build_payload(ablation, rubric)
    if not payload:
        print("No rows matched rubric task_ids (TSFM ids 201–223).", file=sys.stderr)
        return 1

    if args.print_payload:
        json.dump(payload, sys.stdout, indent=2)
        print()
        return 0

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = ablation.parent / "assetops_metrics.csv"
    else:
        out_csv = out_csv.expanduser().resolve()

    if args.aobench_root is None:
        print(
            f"Built payload with {len(payload)} rows. Pass --aobench-root /path/to/AssetOpsBench/aobench "
            f"to run the grader, or use --print-payload to inspect JSON.",
            file=sys.stderr,
        )
        return 0

    _bootstrap_env_for_grader()
    if not (os.environ.get("WATSONX_API_KEY") or os.environ.get("WATSONX_APIKEY") or "").strip():
        print(
            "Warning: WATSONX_API_KEY is empty after loading .env — the grader will fail with "
            "WMLClientError until you set WATSONX_API_KEY and WATSONX_PROJECT_ID in SkillsAgent/.env "
            "or export them.",
            file=sys.stderr,
        )

    aobench = args.aobench_root.expanduser().resolve()
    worker = _skills_root() / "scripts" / "assetops_grader_worker.py"
    if not worker.is_file():
        print(f"Worker missing: {worker}", file=sys.stderr)
        return 1

    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        json.dump(payload, tmp)
        tmp_path = Path(tmp.name)

    try:
        graded = _run_worker(aobench, worker, tmp_path)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    _write_metrics_csv(out_csv, graded)
    print(f"Wrote {len(graded)} rows to {out_csv}")
    if args.pivot_csv is not None:
        pivot = args.pivot_csv.expanduser().resolve()
        agg = _aggregate_by_condition(graded)
        _write_metrics_csv(pivot, agg)
        print(f"Wrote {len(agg)} condition aggregates to {pivot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
