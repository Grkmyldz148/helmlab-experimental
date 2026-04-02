"""V3 Grid Search — M1 perturbation × α scan with ColorBench validation.

Scans M1[0,2] and α to find the combination that maximizes ColorBench WINs.
Each point runs full ColorBench (helmct) — ~30s per eval.

Grid: M1[0,2] δ ∈ {-0.010, -0.006, -0.004, -0.002, 0, +0.002, +0.004}
      α ∈ {0.015, 0.020, 0.025}
      = 21 points × ~30s = ~10 min

Usage:
    python helmgen-next/grid_search_v3.py
"""

import numpy as np
import json
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).parent.parent
COLORBENCH = PROJ / "colorbench"
CHECKPOINTS = Path(__file__).parent / "checkpoints"
REPORTS = Path(__file__).parent / "reports"

# Production M1 baseline
prod = json.loads((PROJ / "src/helmlab/data/gen_params.json").read_text())
M1_BASE = np.array(prod["M1"])
M2_BASE = np.array(prod["M2"])


def make_checkpoint(d02, alpha):
    """Create checkpoint with M1[0,2] perturbation and given alpha."""
    M1 = M1_BASE.copy()
    M1[0, 2] += d02

    # Keep enrichment and PW from production (they give 36 WINs)
    params = prod.copy()
    params["M1"] = M1.tolist()
    params["depcubic_alpha"] = alpha

    name = f"grid_d{d02:+.3f}_a{alpha:.3f}"
    path = CHECKPOINTS / f"{name}.json"
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return path, name


def run_colorbench(checkpoint_path):
    """Run ColorBench and extract h2h score."""
    cmd = [
        sys.executable, "run.py", "oklab", "helmct",
        "--json", str(checkpoint_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(COLORBENCH), timeout=600)
    output = result.stdout

    # Extract h2h score
    for line in output.split("\n"):
        if "Head-to-Head:" in line:
            # Format: "OKLab vs HelmCT(...): 6-36 (tie 19)"
            parts = line.split(":")[-1].strip()
            # parts = "6-36 (tie 19)"
            score_part = parts.split("(")[0].strip()
            oklab_wins, our_wins = map(int, score_part.split("-"))
            tie_part = parts.split("tie")[-1].strip().rstrip(")")
            ties = int(tie_part)
            return our_wins, oklab_wins, ties, output

    return 0, 0, 0, output


def main():
    print("=" * 70)
    print("V3 Grid Search — M1[0,2] × α with Full ColorBench")
    print("=" * 70)

    # Grid definition
    d02_values = [-0.010, -0.006, -0.004, -0.002, 0.000, +0.002, +0.004]
    alpha_values = [0.015, 0.020, 0.025]

    results = []
    best_wins = 0
    best_config = None
    t0 = time.time()

    total = len(d02_values) * len(alpha_values)
    i = 0

    for d02 in d02_values:
        for alpha in alpha_values:
            i += 1
            path, name = make_checkpoint(d02, alpha)
            elapsed = time.time() - t0

            print(f"\n[{i}/{total}] {name} [{elapsed:.0f}s]")
            try:
                wins, losses, ties, output = run_colorbench(path)
                print(f"  → {wins} WIN, {losses} LOSS, {ties} TIE", flush=True)

                results.append({
                    "name": name,
                    "d02": d02,
                    "alpha": alpha,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                })

                if wins > best_wins:
                    best_wins = wins
                    best_config = (d02, alpha, name)
                    print(f"  ★ NEW BEST: {wins} WINs!")

                    # Save the full output for best
                    (REPORTS / f"grid_best_{wins}wins.txt").write_text(output)

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "name": name, "d02": d02, "alpha": alpha,
                    "wins": 0, "losses": 0, "ties": 0, "error": str(e)
                })

    # Summary
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"GRID SEARCH COMPLETE — {elapsed:.0f}s total")
    print("=" * 70)

    # Sort by wins
    results.sort(key=lambda x: -x["wins"])

    print(f"\n{'Config':30s} {'WIN':>4} {'LOSS':>5} {'TIE':>4}")
    print("-" * 50)
    for r in results:
        marker = " ★" if r["wins"] == best_wins else ""
        print(f"  d02={r['d02']:+.3f} α={r['alpha']:.3f}    {r['wins']:4d}  {r['losses']:5d}  {r['ties']:4d}{marker}")

    if best_config:
        print(f"\nBEST: d02={best_config[0]:+.3f}, α={best_config[1]:.3f} → {best_wins} WINs")
        print(f"Checkpoint: {CHECKPOINTS / best_config[2]}.json")

    # Save results
    with open(REPORTS / "17_grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(REPORTS / "17_grid_search_results.md", "w") as f:
        f.write(f"# Grid Search Results\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Grid**: M1[0,2] δ × α, {total} points\n")
        f.write(f"**Best**: {best_wins} WINs at d02={best_config[0]:+.3f}, α={best_config[1]:.3f}\n\n")
        f.write(f"| d02 | α | WIN | LOSS | TIE |\n")
        f.write(f"|-----|---|-----|------|-----|\n")
        for r in results:
            f.write(f"| {r['d02']:+.3f} | {r['alpha']:.3f} | {r['wins']} | {r['losses']} | {r['ties']} |\n")


if __name__ == "__main__":
    main()
