"""Overnight CMA-ES: Rational transfer M2 optimization.

Optimizes M2 ab-rows (6 DOF) + rational a,c (2 DOF) = 8 DOF.
Uses fast_eval (18 critical metrics, ~9s/eval on CPU).

Target: maximize WINs while keeping all current WINs as hard constraints.

Usage:
    python helmgen-next/v2/optimize_rational.py        # CPU, ~5 hours
    nohup python helmgen-next/v2/optimize_rational.py > helmgen-next/v2/opt_log.txt 2>&1 &
"""

import numpy as np
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fast_eval import fast_score, _get_oklab_baseline

PROJ = Path(__file__).parent.parent.parent
CHECKPOINTS = Path(__file__).parent.parent / "checkpoints"
REPORTS = Path(__file__).parent.parent / "reports"

# Production M1 (fixed — don't touch)
prod = json.loads((PROJ / "src/helmlab/data/gen_params.json").read_text())
M1 = prod["M1"]

# OKLab M2 as starting point for ab-rows
OKLAB_M2 = [
    [0.2104542553,  0.793617785, -0.0040720468],
    [1.9779984951, -2.428592205,  0.4505937099],
    [0.0259040371, 0.7827717662, -0.808675766]
]


def build_checkpoint(params):
    """Build checkpoint from 8 parameters: [a, c, ab1, ab2, ab3, ab4, ab5, ab6]."""
    a_rat = params[0]
    c_rat = params[1]
    b_rat = max(1.0 + c_rat - a_rat, 0.01)  # normalization f(1)≈1

    # M2: L-row fixed (OKLab), ab-rows from params
    M2 = [
        OKLAB_M2[0],  # L-row fixed
        [params[2], params[3], params[4]],
        [params[5], params[6], params[7]],
    ]

    return {
        "M1": M1,
        "M2": M2,
        "transfer": "rational",
        "rational_a": float(a_rat),
        "rational_b": float(b_rat),
        "rational_c": float(c_rat),
    }


def objective(params):
    """Minimize: penalized score. Lower = better."""
    try:
        ckpt = build_checkpoint(params)

        # Check M2 invertibility
        M2_arr = np.array(ckpt["M2"])
        if np.linalg.cond(M2_arr) > 50:
            return 1000.0

        # Check ab-row min singular value (prevent collapse)
        ab = M2_arr[1:3, :]
        sv = np.linalg.svd(ab, compute_uv=False)
        if sv.min() < 0.1:
            return 1000.0

        # Check rational params valid
        if ckpt["rational_a"] < 1.5 or ckpt["rational_a"] > 8.0:
            return 1000.0
        if ckpt["rational_c"] < 1.0 or ckpt["rational_c"] > 10.0:
            return 1000.0
        if ckpt["rational_b"] < 0.01:
            return 1000.0

        with open("/tmp/opt_rational.json", "w") as f:
            json.dump(ckpt, f)

        w, l, t, details = fast_score("/tmp/opt_rational.json")

        # Hard constraints: must keep these WINs
        if details.get('cusps_srgb', {}).get('result') != 'WIN':
            return 500.0
        if details.get('cusps_p3', {}).get('result') != 'WIN':
            return 500.0
        if details.get('mono', {}).get('result') != 'WIN':
            return 500.0

        # Soft objective: maximize wins, minimize losses
        score = -w * 10 + l * 15 + t * 1
        return score

    except Exception as e:
        return 1000.0


def main():
    print("=" * 60)
    print("Rational Transfer M2 Optimization (overnight)")
    print(f"DOF: 8 (rational a,c + M2 ab-rows 6)")
    print(f"Eval: fast_eval (~9s/eval)")
    print("=" * 60)

    # Warm up OKLab cache
    print("\nWarming up OKLab baseline cache...")
    t0 = time.time()
    _get_oklab_baseline()
    print(f"  Cache ready in {time.time()-t0:.1f}s")

    # Starting point: a=4.0, c=5.0, OKLab M2 ab-rows
    x0 = np.array([
        4.0, 5.0,  # rational a, c
        OKLAB_M2[1][0], OKLAB_M2[1][1], OKLAB_M2[1][2],  # ab-row 1
        OKLAB_M2[2][0], OKLAB_M2[2][1], OKLAB_M2[2][2],  # ab-row 2
    ])

    print(f"\nBaseline (OKLab M2 + rational a=4, c=5):")
    baseline = objective(x0)
    print(f"  Score: {baseline}")

    # CMA-ES
    try:
        import cma
    except ImportError:
        print("pip install cma")
        sys.exit(1)

    es = cma.CMAEvolutionStrategy(x0, 0.3, {
        'maxiter': 150,
        'popsize': 16,
        'seed': 42,
        'verbose': -1,
        'bounds': [
            [1.5, 1.0, -5, -5, -2, -2, -2, -2],
            [8.0, 10.0, 5, 2, 2, 2, 2, 2]
        ],
    })

    best_score = baseline
    best_params = x0.copy()
    gen = 0
    t_start = time.time()
    log_lines = []

    while not es.stop():
        solutions = es.ask()
        values = [objective(x) for x in solutions]
        es.tell(solutions, values)
        gen += 1

        idx = np.argmin(values)
        if values[idx] < best_score:
            best_score = values[idx]
            best_params = solutions[idx].copy()
            elapsed = time.time() - t_start

            ckpt = build_checkpoint(best_params)
            with open("/tmp/opt_rational.json", "w") as f:
                json.dump(ckpt, f)
            w, l, t, _ = fast_score("/tmp/opt_rational.json")

            msg = f"Gen {gen:4d} [{elapsed/60:.1f}m] NEW BEST: {w}W-{l}L-{t}T (score={best_score:.1f}) a={best_params[0]:.2f} c={best_params[1]:.2f}"
            print(msg, flush=True)
            log_lines.append(msg)

            # Save checkpoint
            ckpt_path = CHECKPOINTS / "v2_rational_opt_best.json"
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, indent=2)

        if gen % 25 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / gen * (150 - gen)
            print(f"Gen {gen:4d} [{elapsed/60:.1f}m] ETA: {eta/60:.0f}m  best={best_score:.1f}", flush=True)

    # Final
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/3600:.1f} hours ({gen} generations)")
    print(f"Best score: {best_score}")

    ckpt = build_checkpoint(best_params)
    with open("/tmp/opt_rational.json", "w") as f:
        json.dump(ckpt, f)
    w, l, t, details = fast_score("/tmp/opt_rational.json", verbose=True)
    print(f"\nFinal: {w}-{l}-{t}")

    # Save
    final_path = CHECKPOINTS / f"v2_rational_final_{w}w_{l}l.json"
    with open(final_path, "w") as f:
        json.dump(ckpt, f, indent=2)
    print(f"Saved: {final_path}")

    # Save log
    with open(REPORTS / "30_rational_opt_log.md", "w") as f:
        f.write(f"# Rational Transfer Optimization Log\n\n")
        f.write(f"Duration: {elapsed/3600:.1f} hours\n")
        f.write(f"Generations: {gen}\n")
        f.write(f"Final score: {w}-{l}-{t}\n\n")
        for line in log_lines:
            f.write(f"- {line}\n")


if __name__ == "__main__":
    main()
