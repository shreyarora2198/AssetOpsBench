import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def main():
    scores_file = Path(__file__).parent.parent.parent / "scores.json"

    if not scores_file.exists():
        print("No scores.json found. Run score_trajectories.py first.")
        return

    data = json.loads(scores_file.read_text())

    types = ["IoT", "FMSA", "TSFM", "Workorder"]
    colors = ["#60a5fa", "#34d399", "#f59e0b", "#f87171"]

    by_type = defaultdict(list)
    for r in data:
        if "pass_rate" in r:
            by_type[r["type"]].append(r["pass_rate"])

    means = [
        np.mean(by_type[t]) * 100 if by_type[t] else 0
        for t in types
    ]
    counts = [len(by_type[t]) for t in types]
    overall = np.mean([r["pass_rate"] for r in data if "pass_rate" in r]) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(types, means, color=colors, width=0.5, edgecolor="white")

    for bar, m, n in zip(bars, means, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{m:.1f}%\n(n={n})",
            ha="center", fontsize=9
        )

    ax.axhline(overall, color="black", ls="--", lw=1.5,
               label=f"Overall: {overall:.1f}%")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("AssetOpsBench — Llama-4-Maverick — MetaAgent")
    ax.legend()
    plt.tight_layout()

    out = Path(__file__).parent.parent.parent / "leaderboard.png"
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()