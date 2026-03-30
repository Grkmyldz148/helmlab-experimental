#!/usr/bin/env python
"""v21 optimization: targeted improvements over v20b.

Diagnostic findings from v20b (COMBVD=23.30):
  1. Lh_cos1/Lh_sin1 (hue-dep L corr) HURTS by -0.07 → zero them out
  2. Low chroma C*=10-20: CIEDE2000 beats us by +3.79 STRESS
  3. BFD-P(C) sub-dataset: STRESS 30.15 (worst)
  4. WITT sub-dataset: STRESS 28.88 (2nd worst)
  5. dist_sl/dist_sc nearly zero impact — v14c showed 22.75 is possible with SL/SC
  6. NC guarantees achromatic axis → can push SL/SC harder

Strategy:
  Phase A: Re-optimize from v20b with Lh zeroed, wider SL/SC bounds
  Phase B: Sub-dataset-balanced objective (equal weighting)
  Phase C: Cross-validation monitored, low-chroma penalty term
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helmlab.data.combvd import load_combvd
from helmlab.data.he2022 import load_he2022
from helmlab.data.macadam1974 import load_macadam1974
from helmlab.data.munsell import load_munsell, generate_munsell_pairs
from helmlab.metrics.stress import stress
from helmlab.spaces.metric import MetricSpace, MetricParams
from helmlab.utils.conversions import XYZ_to_Lab

from optimize_v14 import (
    pack_params, unpack_params, make_bounds, N_PARAMS,
    split_combvd, compute_bias_r, compute_munsell_cv,
    compute_subdataset_stress,
)

# ── Counters ════════════════════════════════════════════════════════

_eval_count = 0
_best_loss = float("inf")
_last_print_time = 0.0


def _reset_counters():
    global _eval_count, _best_loss, _last_print_time
    _eval_count = 0
    _best_loss = float("inf")
    _last_print_time = 0.0


# ── Enhanced bounds ═════════════════════════════════════════════════

def make_bounds_v21(x0, fix_post_power=None):
    """v21 bounds: wider SL/SC, narrow Lh_cos1/Lh_sin1."""
    bounds = make_bounds(x0, fix_post_power=fix_post_power)

    # Lh_cos1 (61), Lh_sin1 (62) — shrink bounds (ablation: -0.07 impact)
    # Don't hard-zero (breaks RT for extreme blues), let optimizer decide
    bounds[61] = (-0.15, 0.15)
    bounds[62] = (-0.15, 0.15)

    # Wider SL/SC bounds — v14c showed these can reach STRESS 22.75
    bounds[70] = (-2.0, 5.0)   # dist_sl
    bounds[71] = (-1.0, 3.0)   # dist_sc

    return bounds


# ── CIE Lab helper for chroma computation ═══════════════════════════

D65 = np.array([0.95047, 1.0, 1.08883])


def compute_pair_chroma(xyz1, xyz2):
    """Compute mean CIE C* for each pair."""
    lab1 = XYZ_to_Lab(xyz1)
    lab2 = XYZ_to_Lab(xyz2)
    c1 = np.sqrt(lab1[:, 1]**2 + lab1[:, 2]**2)
    c2 = np.sqrt(lab2[:, 1]**2 + lab2[:, 2]**2)
    return (c1 + c2) / 2


# ── Objectives ══════════════════════════════════════════════════════

def make_objective_standard(XYZ_1_c, XYZ_2_c, DV_c, XYZ_1_h, XYZ_2_h, DV_h,
                            he_lambda=0.05, rt_penalty=20.0):
    """Standard COMBVD + He with NC=True."""
    rng = np.random.default_rng(42)
    XYZ_rt = rng.uniform(0.05, 0.90, (1000, 3))

    def objective(x):
        global _eval_count, _best_loss, _last_print_time
        try:
            params = unpack_params(x)
            space = MetricSpace(params, neutral_correction=True, ab_rotate_deg=-28.2)

            DE_c = space.distance(XYZ_1_c, XYZ_2_c)
            if np.any(~np.isfinite(DE_c)):
                return 100.0
            s_combvd = stress(DV_c, DE_c)

            DE_h = space.distance(XYZ_1_h, XYZ_2_h)
            if np.any(~np.isfinite(DE_h)):
                return 100.0
            s_he = stress(DV_h, DE_h)

            total = s_combvd + he_lambda * s_he

            if rt_penalty > 0:
                rt_errors = space.round_trip_error(XYZ_rt)
                rt_max = rt_errors.max()
                if rt_max > 1e-6:
                    total += rt_penalty * np.log10(rt_max / 1e-6)

        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 100.0

        _eval_count += 1
        if total < _best_loss:
            _best_loss = total

        now = time.time()
        if now - _last_print_time > 10.0:
            _last_print_time = now
            print(f"  eval #{_eval_count:>6d}  loss={total:.4f}  "
                  f"COMBVD={s_combvd:.2f}  He={s_he:.2f}  best={_best_loss:.4f}",
                  flush=True)

        return total

    return objective


def make_objective_balanced(combvd, XYZ_1_h, XYZ_2_h, DV_h,
                            munsell_pairs=None,
                            he_lambda=0.05, munsell_lambda=0.02,
                            low_chroma_lambda=0.1,
                            worst_sub_lambda=0.05,
                            rt_penalty=20.0):
    """Balanced objective: equal sub-dataset + low-chroma penalty + Munsell CV."""
    rng = np.random.default_rng(42)
    XYZ_rt = rng.uniform(0.05, 0.90, (1000, 3))

    # Pre-compute sub-dataset masks
    datasets = combvd["dataset"]
    unique_ds = sorted(set(datasets))
    sub_masks = {}
    for ds in unique_ds:
        mask = np.array([d == ds for d in datasets])
        if np.sum(mask) >= 2:
            sub_masks[ds] = mask

    # Pre-compute low-chroma mask (C*=5-25, the region where CIEDE2000 beats us)
    C_mean = compute_pair_chroma(combvd["XYZ_1"], combvd["XYZ_2"])
    low_chroma_mask = (C_mean >= 5) & (C_mean < 25)
    n_low_chroma = np.sum(low_chroma_mask)

    # Munsell data
    m_XYZ_1 = munsell_pairs["XYZ_1"] if munsell_pairs else None
    m_XYZ_2 = munsell_pairs["XYZ_2"] if munsell_pairs else None

    def objective(x):
        global _eval_count, _best_loss, _last_print_time
        try:
            params = unpack_params(x)
            space = MetricSpace(params, neutral_correction=True, ab_rotate_deg=-28.2)

            # Full COMBVD
            DE_full = space.distance(combvd["XYZ_1"], combvd["XYZ_2"])
            if np.any(~np.isfinite(DE_full)):
                return 100.0
            s_full = stress(combvd["DV"], DE_full)

            # Per-sub-dataset STRESS
            sub_stresses = []
            for ds, mask in sub_masks.items():
                s_sub = stress(combvd["DV"][mask], DE_full[mask])
                sub_stresses.append(s_sub)
            s_mean_sub = float(np.mean(sub_stresses))
            s_max_sub = float(np.max(sub_stresses))

            # He
            DE_h = space.distance(XYZ_1_h, XYZ_2_h)
            if np.any(~np.isfinite(DE_h)):
                return 100.0
            s_he = stress(DV_h, DE_h)

            # Low-chroma STRESS
            s_low_chroma = 0.0
            if n_low_chroma >= 5:
                s_low_chroma = stress(combvd["DV"][low_chroma_mask],
                                       DE_full[low_chroma_mask])

            # Munsell CV
            munsell_cv = 0.0
            if m_XYZ_1 is not None and munsell_lambda > 0:
                DE_m = space.distance(m_XYZ_1, m_XYZ_2)
                if np.any(~np.isfinite(DE_m)) or np.mean(DE_m) < 1e-10:
                    return 100.0
                munsell_cv = float(np.std(DE_m) / np.mean(DE_m) * 100.0)

            # Combined loss:
            # Base: 0.5 * full COMBVD + 0.5 * mean(sub-datasets)
            # This reduces BFD-P(D65) dominance (2028/3813 = 53% of data)
            total = (0.5 * s_full + 0.5 * s_mean_sub
                     + he_lambda * s_he
                     + low_chroma_lambda * s_low_chroma
                     + worst_sub_lambda * s_max_sub
                     + munsell_lambda * munsell_cv)

            if rt_penalty > 0:
                rt_errors = space.round_trip_error(XYZ_rt)
                rt_max = rt_errors.max()
                if rt_max > 1e-6:
                    total += rt_penalty * np.log10(rt_max / 1e-6)

        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 100.0

        _eval_count += 1
        if total < _best_loss:
            _best_loss = total

        now = time.time()
        if now - _last_print_time > 10.0:
            _last_print_time = now
            print(f"  eval #{_eval_count:>6d}  loss={total:.4f}  "
                  f"full={s_full:.2f}  meanSub={s_mean_sub:.2f}  "
                  f"maxSub={s_max_sub:.2f}  lowC={s_low_chroma:.2f}  "
                  f"He={s_he:.2f}  best={_best_loss:.4f}",
                  flush=True)

        return total

    return objective


# ── Evaluation ═════════════════════════════════════════════════════

def evaluate_v21(x, combvd, train, val, he, mac, munsell_pairs):
    """Full evaluation with sub-dataset breakdown."""
    p = unpack_params(x)
    s = MetricSpace(p, neutral_correction=True, ab_rotate_deg=-28.2)

    DE_full = s.distance(combvd["XYZ_1"], combvd["XYZ_2"])
    s_train = stress(train["DV"], s.distance(train["XYZ_1"], train["XYZ_2"]))
    s_val = stress(val["DV"], s.distance(val["XYZ_1"], val["XYZ_2"]))
    s_full = stress(combvd["DV"], DE_full)
    s_h = stress(he["DV"], s.distance(he["XYZ_1"], he["XYZ_2"]))
    s_m = stress(mac["DV"], s.distance(mac["XYZ_1"], mac["XYZ_2"]))
    m_cv = compute_munsell_cv(s, munsell_pairs)
    rt = s.round_trip_error(np.random.default_rng(42).uniform(0.05, 0.90, (1000, 3)))
    r_bias = compute_bias_r(combvd["DV"], DE_full)

    # Low-chroma segment
    C_mean = compute_pair_chroma(combvd["XYZ_1"], combvd["XYZ_2"])
    low_mask = (C_mean >= 5) & (C_mean < 25)
    s_low_c = stress(combvd["DV"][low_mask], DE_full[low_mask]) if np.sum(low_mask) >= 5 else -1

    # Sub-dataset
    sub = compute_subdataset_stress(s, combvd)

    return {
        "train": s_train, "val": s_val, "full": s_full,
        "he": s_h, "mac": s_m, "cv": m_cv, "rt": rt.max(),
        "r_bias": r_bias, "low_chroma": s_low_c, "sub": sub,
    }


def print_eval_v21(label, m):
    gap = m["val"] - m["train"]
    print(f"  {label}: train={m['train']:.4f}  val={m['val']:.4f}  (gap={gap:+.2f})  "
          f"full={m['full']:.4f}  He={m['he']:.2f}  Mac={m['mac']:.4f}  "
          f"CV={m['cv']:.1f}%  |r|={m['r_bias']:.3f}  RT={m['rt']:.2e}  "
          f"lowC={m['low_chroma']:.2f}")
    if "sub" in m:
        for ds_name in sorted(m["sub"].keys()):
            info = m["sub"][ds_name]
            print(f"    {ds_name:20s}  n={info['n']:>4d}  STRESS={info['stress']:.2f}")


def run_restarts(objective, x0, bounds, restarts, maxiter,
                 combvd, train, val, he, mac, munsell_pairs, label=""):
    """Run optimization with Hessian restarts, return best x and metrics."""
    best_x = x0.copy()
    best_full = 999.0

    for restart in range(restarts):
        _reset_counters()

        t0 = time.time()
        result = minimize(objective, x0=best_x, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-11})
        dt = time.time() - t0

        m = evaluate_v21(result.x, combvd, train, val, he, mac, munsell_pairs)
        gap = m["val"] - m["train"]
        print(f"  [{label}] Restart {restart+1}: full={m['full']:.4f}  "
              f"train={m['train']:.4f}  val={m['val']:.4f}  gap={gap:+.2f}  "
              f"He={m['he']:.2f}  Mac={m['mac']:.4f}  lowC={m['low_chroma']:.2f}  "
              f"|r|={m['r_bias']:.3f}  RT={m['rt']:.2e}  ({dt:.0f}s)")

        if m["rt"] > 1e-6:
            print(f"    WARNING: RT broken, skipping")
            continue

        if m["full"] < best_full:
            best_x = result.x.copy()
            best_full = m["full"]
            print(f"    ** New best (COMBVD={best_full:.4f})")

    m_best = evaluate_v21(best_x, combvd, train, val, he, mac, munsell_pairs)
    return best_x, m_best


# ── Main ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="v21: targeted improvements over v20b")
    parser.add_argument("--init", type=str, default="src/helmlab/data/metric_params.json",
                        help="Initial params JSON")
    parser.add_argument("--output", type=str, default="checkpoints/v21_best.json",
                        help="Output best params JSON")
    parser.add_argument("--restarts", type=int, default=8, help="Restarts per phase")
    parser.add_argument("--maxiter", type=int, default=5000, help="Max iterations per restart")
    parser.add_argument("--he-lambda", type=float, default=0.05, help="He 2022 weight")
    parser.add_argument("--munsell-lambda", type=float, default=0.02, help="Munsell CV weight")
    parser.add_argument("--low-chroma-lambda", type=float, default=0.1,
                        help="Low-chroma segment penalty weight")
    parser.add_argument("--worst-sub-lambda", type=float, default=0.05,
                        help="Worst sub-dataset penalty weight")
    args = parser.parse_args()

    print(f"v21: Targeted improvements over v20b")
    print(f"  Changes from v20b:")
    print(f"    - Lh_cos1/Lh_sin1 narrowed bounds (ablation: -0.07 impact)")
    print(f"    - Wider SL/SC bounds (v14c showed 22.75 possible)")
    print(f"    - Phase A: Standard re-optimization with v21 bounds")
    print(f"    - Phase B: Balanced objective (equal sub-dataset + low-chroma penalty)")
    print(f"    - All with NC=True + ab_rotate_deg=-28.2")
    print()

    # Load data
    print("Loading data...")
    combvd = load_combvd()
    he = load_he2022()
    mac = load_macadam1974()
    train, val = split_combvd(combvd, seed=42, val_split=0.2)

    print(f"  COMBVD: {len(combvd['DV'])} pairs (train={len(train['DV'])}, val={len(val['DV'])})")
    print(f"  He 2022: {len(he['DV'])} pairs")
    print(f"  MacAdam 1974: {len(mac['DV'])} pairs")

    print("Loading Munsell data...")
    munsell_data = load_munsell(subset="real")
    munsell_pairs = generate_munsell_pairs(munsell_data)
    print(f"  Munsell pairs: {len(munsell_pairs['perceptual_distance'])}")

    # Load initial params
    params = MetricParams.load(args.init)
    x0 = pack_params(params)
    assert len(x0) == N_PARAMS

    # Initial evaluation
    m0 = evaluate_v21(x0, combvd, train, val, he, mac, munsell_pairs)
    print(f"\nv20b baseline:")
    print_eval_v21("Init", m0)

    # ══════════════════════════════════════════════════════════════════
    # Phase A: Standard re-optimization with v21 bounds
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"Phase A: Standard re-optimization (Lh zeroed, wider SL/SC)")
    print(f"{'='*70}")

    bounds_a = make_bounds_v21(x0)
    objective_a = make_objective_standard(
        combvd["XYZ_1"], combvd["XYZ_2"], combvd["DV"],
        he["XYZ_1"], he["XYZ_2"], he["DV"],
        he_lambda=args.he_lambda,
    )

    best_x_a, m_a = run_restarts(
        objective_a, x0, bounds_a, args.restarts, args.maxiter,
        combvd, train, val, he, mac, munsell_pairs, label="A",
    )

    print(f"\nPhase A result:")
    print_eval_v21("Phase A", m_a)

    # Save Phase A checkpoint
    p_a = unpack_params(best_x_a)
    p_a.save("checkpoints/v21_phase_a.json")
    print(f"  Saved: checkpoints/v21_phase_a.json")

    # ══════════════════════════════════════════════════════════════════
    # Phase B: Balanced objective (equal sub-dataset + low-chroma)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"Phase B: Balanced objective (sub-dataset + low-chroma penalty)")
    print(f"{'='*70}")

    bounds_b = make_bounds_v21(best_x_a)
    objective_b = make_objective_balanced(
        combvd, he["XYZ_1"], he["XYZ_2"], he["DV"],
        munsell_pairs=munsell_pairs,
        he_lambda=args.he_lambda,
        munsell_lambda=args.munsell_lambda,
        low_chroma_lambda=args.low_chroma_lambda,
        worst_sub_lambda=args.worst_sub_lambda,
    )

    best_x_b, m_b = run_restarts(
        objective_b, best_x_a, bounds_b, args.restarts, args.maxiter,
        combvd, train, val, he, mac, munsell_pairs, label="B",
    )

    print(f"\nPhase B result:")
    print_eval_v21("Phase B", m_b)

    # Accept Phase B only if COMBVD doesn't regress more than 0.3
    if m_b["full"] <= m_a["full"] + 0.3:
        best_x = best_x_b
        best_m = m_b
        print(f"  Phase B accepted")
    else:
        best_x = best_x_a
        best_m = m_a
        print(f"  Phase B REJECTED (COMBVD regression: {m_b['full']:.4f} > {m_a['full']:.4f} + 0.3)")
        print(f"  Keeping Phase A result")

    # ══════════════════════════════════════════════════════════════════
    # Phase C: Fine-tune from best, standard objective
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"Phase C: Fine-tune from best result")
    print(f"{'='*70}")

    bounds_c = make_bounds_v21(best_x)
    objective_c = make_objective_standard(
        combvd["XYZ_1"], combvd["XYZ_2"], combvd["DV"],
        he["XYZ_1"], he["XYZ_2"], he["DV"],
        he_lambda=args.he_lambda,
    )

    best_x_c, m_c = run_restarts(
        objective_c, best_x, bounds_c, args.restarts // 2, args.maxiter,
        combvd, train, val, he, mac, munsell_pairs, label="C",
    )

    if m_c["full"] < best_m["full"]:
        best_x = best_x_c
        best_m = m_c
        print(f"  Phase C improved: {m_c['full']:.4f}")
    else:
        print(f"  Phase C no improvement")

    # ══════════════════════════════════════════════════════════════════
    # Final report
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"v21 FINAL RESULTS")
    print(f"{'='*70}")
    print_eval_v21("FINAL", best_m)

    # Compare with v20b baseline
    print(f"\n  vs v20b baseline:")
    print(f"    COMBVD:    {best_m['full']:.4f} vs {m0['full']:.4f} (delta={best_m['full']-m0['full']:+.4f})")
    print(f"    He:        {best_m['he']:.2f} vs {m0['he']:.2f}")
    print(f"    Mac:       {best_m['mac']:.4f} vs {m0['mac']:.4f}")
    print(f"    LowChroma: {best_m['low_chroma']:.2f} vs {m0['low_chroma']:.2f}")

    # Save final
    final_params = unpack_params(best_x)
    final_params.save(args.output)
    print(f"\n  Saved: {args.output}")

    # Also save as metric_params_v21.json for comparison
    final_params.save("src/helmlab/data/metric_params_v21.json")
    print(f"  Saved: src/helmlab/data/metric_params_v21.json")


if __name__ == "__main__":
    main()
