import json, sys
from pathlib import Path
import csv

TSFM_TOOLS = {
    "run_tsfm_forecasting",
    "run_tsfm_finetuning",
    "run_tsad",
    "run_integrated_tsad",
}

# Only these scenario IDs should ever call TSFM inference tools
TSFM_INFERENCE_OK = set(range(216, 224))

# Fault diagnosis — the most important ones to catch
FAULT_DIAGNOSIS_IDS = set(range(101, 121))

def extract_tools(rec):
    tools = []
    for step in rec.get("trajectory", []):
        if "tool" in step and step["tool"]:
            tools.append(step["tool"])
        for tc in step.get("tool_calls", []):
            tools.append(tc.get("name", ""))
    return tools

def main():
    traj_file = Path(__file__).parent.parent.parent / "trajectories" / "all_trajectories.json"

    if not traj_file.exists():
        print(f"No trajectories file found at {traj_file}")
        print("Run run_all_scenarios.py first.")
        sys.exit(1)

    data = json.loads(traj_file.read_text())
    violations = []

    for rec in data:
        sid = rec["id"]
        tools = extract_tools(rec)
        tsfm_called = [t for t in tools if t in TSFM_TOOLS]

        if tsfm_called and sid not in TSFM_INFERENCE_OK:
            label = "FAULT-DIAG" if sid in FAULT_DIAGNOSIS_IDS else "WRONG-DOMAIN"
            violations.append({
                "id": sid,
                "type": rec["type"],
                "label": label,
                "text": rec["text"],
                "tsfm_tools_called": tsfm_called,
            })

    print(f"\n{'='*60}")
    print(f"Total scenarios analyzed : {len(data)}")
    print(f"Unnecessary TSFM calls   : {len(violations)}")
    print(f"Violation rate           : {len(violations)/len(data)*100:.1f}%")
    print(f"{'='*60}")

    if violations:
        fault_diag = [v for v in violations if v["label"] == "FAULT-DIAG"]
        other = [v for v in violations if v["label"] == "WRONG-DOMAIN"]
        print(f"\nFault-diagnosis scenarios calling TSFM: {len(fault_diag)}")
        print(f"Other wrong-domain TSFM calls         : {len(other)}")
        print()
        for v in violations:
            print(f"  [{v['label']}] Q{v['id']} ({v['type']})")
            print(f"    Text  : {v['text']}")
            print(f"    Called: {v['tsfm_tools_called']}")
            print()
    else:
        print("\nNo unnecessary TSFM calls found.")

    out_json = Path(__file__).parent.parent.parent / "tsfm_report.json"
    out_json.write_text(json.dumps(violations, indent=2))
    print(f"Report saved to {out_json}")

    # --- CSV export ---
    out_csv = Path(__file__).parent.parent.parent / "tsfm_report.csv"

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "type", "label", "text", "tsfm_tools_called"]
        )
        writer.writeheader()
        for v in violations:
            writer.writerow({
                "id": v["id"],
                "type": v["type"],
                "label": v["label"],
                "text": v["text"],
                "tsfm_tools_called": ", ".join(v["tsfm_tools_called"]),
            })

    print(f"CSV report saved to {out_csv}")

if __name__ == "__main__":
    main()