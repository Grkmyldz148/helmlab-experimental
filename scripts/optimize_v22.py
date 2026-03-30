#!/usr/bin/env python
"""v22: Multi-dataset optimization — COMBVD + He + MacAdam + Munsell.

Target: Beat CAM16-UCS on MacAdam (18.71) while keeping COMBVD ≤ 23.5.

Key changes from v21:
  - MacAdam STRESS in objective (lambda_mac=0.3)
  - Corrected Munsell Y-scale (Y=1)
  - Balanced sub-dataset weighting
  - Euclidean dE tracking (space quality)
"""

import argparse, json, time, sys
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

from optimize_v14 import (
    pack_params, unpack_params, make_bounds, N_PARAMS,
    split_combvd, compute_bias_r, compute_munsell_cv,
    compute_subdataset_stress,
)

_eval_count = 0
_best_loss = float("inf")
_last_print_time = 0.0

def _reset():
    global _eval_count, _best_loss, _last_print_time
    _eval_count = 0
    _best_loss = float("inf")
    _last_print_time = 0.0


def make_bounds_v22(x0):
    bounds = make_bounds(x0)
    bounds[61] = (-0.15, 0.15)  # Lh_cos1
    bounds[62] = (-0.15, 0.15)  # Lh_sin1
    bounds[70] = (-2.0, 5.0)    # dist_sl
    bounds[71] = (-1.0, 3.0)    # dist_sc
    return bounds


def make_objective(combvd, he, mac, munsell_pairs,
                   he_lam=0.05, mac_lam=0.3, munsell_lam=0.02,
                   worst_sub_lam=0.05, x0_ref=None):
    """Multi-dataset objective: COMBVD (balanced) + He + MacAdam + Munsell."""
    # RT test points: filter to non-negative LMS (clamped points have expected RT error)
    rng = np.random.default_rng(42)
    _xyz = rng.uniform(0.05, 0.90, (2000, 3))
    if x0_ref is not None and isinstance(x0_ref, MetricParams):
        _p0 = x0_ref
    elif x0_ref is not None:
        _p0 = unpack_params(x0_ref)
    else:
        _p0 = MetricParams.load("src/helmlab/data/metric_params.json")
    _lms = _xyz @ _p0.M1.T
    XYZ_rt = _xyz[(_lms >= 0).all(axis=1)][:1000]

    # Sub-dataset masks
    ds_arr = np.array(combvd["dataset"])
    unique_ds = sorted(set(combvd["dataset"]))
    sub_masks = {ds: ds_arr == ds for ds in unique_ds}

    m_X1, m_X2 = munsell_pairs["XYZ_1"], munsell_pairs["XYZ_2"]

    def objective(x):
        global _eval_count, _best_loss, _last_print_time
        try:
            params = unpack_params(x)
            space = MetricSpace(params, neutral_correction=True, ab_rotate_deg=-28.2)

            # COMBVD full + sub-dataset balanced
            DE_c = space.distance(combvd["XYZ_1"], combvd["XYZ_2"])
            if np.any(~np.isfinite(DE_c)):
                return 100.0
            s_full = stress(combvd["DV"], DE_c)

            sub_s = [stress(combvd["DV"][m], DE_c[m]) for m in sub_masks.values()]
            s_mean_sub = float(np.mean(sub_s))
            s_max_sub = float(np.max(sub_s))

            # He2022
            DE_h = space.distance(he["XYZ_1"], he["XYZ_2"])
            if np.any(~np.isfinite(DE_h)):
                return 100.0
            s_he = stress(he["DV"], DE_h)

            # MacAdam — key addition for v22
            DE_m = space.distance(mac["XYZ_1"], mac["XYZ_2"])
            if np.any(~np.isfinite(DE_m)):
                return 100.0
            s_mac = stress(mac["DV"], DE_m)

            # Munsell CV
            DE_mu = space.distance(m_X1, m_X2)
            if np.any(~np.isfinite(DE_mu)) or np.mean(DE_mu) < 1e-10:
                return 100.0
            munsell_cv = float(np.std(DE_mu) / np.mean(DE_mu) * 100.0)

            # Combined: balanced COMBVD + He + MacAdam + Munsell + worst-sub
            total = (0.5 * s_full + 0.5 * s_mean_sub
                     + he_lam * s_he
                     + mac_lam * s_mac
                     + munsell_lam * munsell_cv
                     + worst_sub_lam * s_max_sub)

            # Round-trip penalty
            rt = space.round_trip_error(XYZ_rt).max()
            if rt > 1e-6:
                total += 20.0 * np.log10(rt / 1e-6)

        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 100.0

        _eval_count += 1
        if total < _best_loss:
            _best_loss = total

        now = time.time()
        if now - _last_print_time > 10.0:
            _last_print_time = now
            print(f"  #{_eval_count:>6d}  loss={total:.4f}  "
                  f"COMBVD={s_full:.2f}  Mac={s_mac:.2f}  He={s_he:.2f}  "
                  f"MunsCV={munsell_cv:.1f}%  best={_best_loss:.4f}", flush=True)

        return total

    return objective


def evaluate(x, combvd, train, val, he, mac, munsell_pairs):
    p = unpack_params(x)
    s = MetricSpace(p, neutral_correction=True, ab_rotate_deg=-28.2)

    DE_c = s.distance(combvd["XYZ_1"], combvd["XYZ_2"])
    DE_t = s.distance(train["XYZ_1"], train["XYZ_2"])
    DE_v = s.distance(val["XYZ_1"], val["XYZ_2"])

    s_full = stress(combvd["DV"], DE_c)
    s_train = stress(train["DV"], DE_t)
    s_val = stress(val["DV"], DE_v)
    s_he = stress(he["DV"], s.distance(he["XYZ_1"], he["XYZ_2"]))
    s_mac = stress(mac["DV"], s.distance(mac["XYZ_1"], mac["XYZ_2"]))
    m_cv = compute_munsell_cv(s, munsell_pairs)

    # Euclidean dE (space quality check)
    Lab1 = s.from_XYZ(combvd["XYZ_1"])
    Lab2 = s.from_XYZ(combvd["XYZ_2"])
    DE_euclid = np.sqrt(np.sum((Lab1 - Lab2)**2, axis=-1))
    s_euclid = stress(combvd["DV"], DE_euclid)

    Lab1m = s.from_XYZ(mac["XYZ_1"])
    Lab2m = s.from_XYZ(mac["XYZ_2"])
    DE_mac_euclid = np.sqrt(np.sum((Lab1m - Lab2m)**2, axis=-1))
    s_mac_euclid = stress(mac["DV"], DE_mac_euclid)

    # RT on valid (non-negative LMS) points only
    _rng = np.random.default_rng(42)
    _xyz_rt = _rng.uniform(0.05, 0.90, (2000, 3))
    _lms_rt = _xyz_rt @ p.M1.T
    _xyz_rt = _xyz_rt[(_lms_rt >= 0).all(axis=1)][:1000]
    rt = s.round_trip_error(_xyz_rt)

    sub = compute_subdataset_stress(s, combvd)

    return {
        "full": s_full, "train": s_train, "val": s_val,
        "he": s_he, "mac": s_mac, "munsell_cv": m_cv,
        "euclid": s_euclid, "mac_euclid": s_mac_euclid,
        "rt": rt.max(), "sub": sub,
    }


def print_eval(label, m):
    gap = m["val"] - m["train"]
    print(f"  {label}:")
    print(f"    COMBVD={m['full']:.2f}  train={m['train']:.2f}  val={m['val']:.2f}  gap={gap:+.2f}")
    print(f"    He={m['he']:.2f}  Mac={m['mac']:.2f}  Munsell_CV={m['munsell_cv']:.1f}%")
    print(f"    Euclid(COMBVD)={m['euclid']:.2f}  Euclid(Mac)={m['mac_euclid']:.2f}")
    print(f"    RT={m['rt']:.2e}")
    if "sub" in m:
        for ds in sorted(m["sub"]):
            print(f"    {ds:20s}  STRESS={m['sub'][ds]['stress']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default="src/helmlab/data/metric_params.json")
    parser.add_argument("--output", default="checkpoints/v22_best.json")
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--mac-lambda", type=float, default=0.3)
    parser.add_argument("--he-lambda", type=float, default=0.05)
    parser.add_argument("--munsell-lambda", type=float, default=0.02)
    args = parser.parse_args()

    print("v22: Multi-dataset optimization (COMBVD + He + MacAdam + Munsell)")
    print(f"  mac_lambda={args.mac_lambda}  he_lambda={args.he_lambda}")

    # Load data
    print("Loading data...")
    combvd = load_combvd()
    he = load_he2022()
    mac = load_macadam1974()
    munsell = load_munsell("real")
    m_pairs = generate_munsell_pairs(munsell)
    train, val = split_combvd(combvd, seed=42, val_split=0.2)

    print(f"  COMBVD={len(combvd['DV'])} He={len(he['DV'])} Mac={len(mac['DV'])} "
          f"Munsell={len(m_pairs['XYZ_1'])} pairs")

    # Initial params
    params = MetricParams.load(args.init)
    x0 = pack_params(params)
    m0 = evaluate(x0, combvd, train, val, he, mac, m_pairs)
    print_eval("v20b baseline", m0)

    # Objective
    bounds = make_bounds_v22(x0)
    obj = make_objective(combvd, he, mac, m_pairs,
                         he_lam=args.he_lambda, mac_lam=args.mac_lambda,
                         munsell_lam=args.munsell_lambda, x0_ref=params)

    best_x = x0.copy()
    best_full = m0["full"]

    for restart in range(args.restarts):
        _reset()
        print(f"\n{'='*60}")
        print(f"Restart {restart+1}/{args.restarts}")
        print(f"{'='*60}")

        t0 = time.time()
        result = minimize(obj, x0=best_x, method="L-BFGS-B", bounds=bounds,
                         options={"maxiter": args.maxiter, "ftol": 1e-13, "gtol": 1e-11})
        dt = time.time() - t0

        m = evaluate(result.x, combvd, train, val, he, mac, m_pairs)
        print_eval(f"Restart {restart+1} ({dt:.0f}s)", m)

        if m["rt"] > 1e-6:
            print("    WARNING: RT broken, skipping")
            continue

        # Accept if combined quality improves (COMBVD + MacAdam weighted)
        combined = m["full"] + 0.3 * m["mac"]
        best_combined = best_full + 0.3 * m0["mac"] if best_full == m0["full"] else best_full + 0.3 * evaluate(best_x, combvd, train, val, he, mac, m_pairs)["mac"]
        improved = combined < best_combined and m["full"] <= m0["full"] + 1.0

        if improved:
            best_x = result.x.copy()
            best_full = m["full"]
            print(f"    ** New best (COMBVD={m['full']:.2f}, Mac={m['mac']:.2f})")

            # Save checkpoint
            p_best = unpack_params(best_x)
            p_best.save(args.output)

    # Final
    m_final = evaluate(best_x, combvd, train, val, he, mac, m_pairs)
    print(f"\n{'='*60}")
    print(f"v22 FINAL")
    print(f"{'='*60}")
    print_eval("FINAL", m_final)

    print(f"\n  vs v20b:")
    print(f"    COMBVD: {m_final['full']:.2f} vs {m0['full']:.2f} ({m_final['full']-m0['full']:+.2f})")
    print(f"    MacAdam: {m_final['mac']:.2f} vs {m0['mac']:.2f} ({m_final['mac']-m0['mac']:+.2f})")
    print(f"    He: {m_final['he']:.2f} vs {m0['he']:.2f} ({m_final['he']-m0['he']:+.2f})")
    print(f"    Munsell: {m_final['munsell_cv']:.1f}% vs {m0['munsell_cv']:.1f}%")
    print(f"    Euclid: {m_final['euclid']:.2f} vs {m0['euclid']:.2f}")

    # Save
    final_p = unpack_params(best_x)
    final_p.save(args.output)
    print(f"\n  Saved: {args.output}")


if __name__ == "__main__":
    main()
