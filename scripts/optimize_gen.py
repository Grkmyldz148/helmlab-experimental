#!/usr/bin/env python
"""v17-gen: Achromatic + Hue linearity joint optimization.

Combines the achromatic constraint from v16 with a new hue linearity
constraint that keeps sRGB primary/secondary hue angles near their
expected positions (R=0°, Y=60°, G=120°, C=180°, B=240°, M=300°).

Sweeps hue_lambda with fixed ach_lambda to find the best trade-off
between STRESS, achromatic quality, and hue linearity.
"""

import argparse
import time

import numpy as np
from scipy.optimize import minimize

from helmlab.data.combvd import load_combvd
from helmlab.data.he2022 import load_he2022
from helmlab.data.macadam1974 import load_macadam1974
from helmlab.data.munsell import load_munsell, generate_munsell_pairs
from helmlab.metrics.stress import stress
from helmlab.spaces.analytical import AnalyticalParams, AnalyticalSpace
from helmlab.utils.srgb_convert import sRGB_to_XYZ

N_PARAMS = 72

# ── Import pack/unpack/bounds from optimize_v14 ──────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_v14 import (
    pack_params, unpack_params, make_bounds,
    split_combvd, compute_munsell_cv, compute_bias_r,
)

# ── Gray test stimuli ────────────────────────────────────────────
_GRAY_XYZ = np.array([
    sRGB_to_XYZ(np.array([v, v, v]))
    for v in np.linspace(0.05, 0.95, 19)
])

# ── sRGB primary/secondary XYZ values and target hues ────────────
_PRIMARY_SRGB = np.array([
    [1, 0, 0],  # Red
    [1, 1, 0],  # Yellow
    [0, 1, 0],  # Green
    [0, 1, 1],  # Cyan
    [0, 0, 1],  # Blue
    [1, 0, 1],  # Magenta
], dtype=np.float64)

_PRIMARY_XYZ = np.array([sRGB_to_XYZ(c) for c in _PRIMARY_SRGB])

_TARGET_HUE_RAD = np.array([
    0,                  # Red     = 0°
    np.pi / 3,          # Yellow  = 60°
    2 * np.pi / 3,      # Green   = 120°
    np.pi,              # Cyan    = 180°
    4 * np.pi / 3,      # Blue    = 240°
    5 * np.pi / 3,      # Magenta = 300°
])

_PRIMARY_NAMES = ["Red", "Yellow", "Green", "Cyan", "Blue", "Magenta"]


# ── Achromatic stats ─────────────────────────────────────────────

def compute_achromatic_stats(space):
    """Compute achromatic chroma stats."""
    lab = space.from_XYZ(_GRAY_XYZ)
    C_sq = lab[:, 1]**2 + lab[:, 2]**2
    C = np.sqrt(C_sq)
    return {
        "rms": float(np.sqrt(np.mean(C_sq))),
        "max": float(np.max(C)),
        "mean": float(np.mean(C)),
    }


def compute_neutral_ramp_cv(space, N=21):
    srgb_ramp = np.linspace(0.0, 1.0, N)
    labs = []
    for v in srgb_ramp:
        if v < 1e-10:
            xyz = np.array([[0.0, 0.0, 0.0]])
        else:
            xyz = sRGB_to_XYZ(np.array([v, v, v])).reshape(1, 3)
        labs.append(space.from_XYZ(xyz)[0])
    labs = np.array(labs)
    dists = []
    for i in range(1, N):
        de = space.distance(labs[i-1:i], labs[i:i+1])
        dists.append(de[0])
    dists = np.array(dists)
    if np.mean(dists) < 1e-12:
        return 200.0
    return float(np.std(dists) / np.mean(dists) * 100.0)


# ── Hue linearity stats ─────────────────────────────────────────

def compute_hue_penalty(space):
    """Mean squared angular error (radians²) for 6 sRGB primaries."""
    lab = space.from_XYZ(_PRIMARY_XYZ)
    H = np.arctan2(lab[:, 2], lab[:, 1])
    diff = H - _TARGET_HUE_RAD
    angular_err = np.arctan2(np.sin(diff), np.cos(diff))  # wrap to [-π, π]
    return float(np.mean(angular_err ** 2))


def compute_hue_stats(space):
    """Detailed hue stats for each primary/secondary."""
    lab = space.from_XYZ(_PRIMARY_XYZ)
    H_deg = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360
    C = np.sqrt(lab[:, 1]**2 + lab[:, 2]**2)
    targets = [0, 60, 120, 180, 240, 300]
    errors = []
    for h, t in zip(H_deg, targets):
        diff = h - t
        diff = (diff + 180) % 360 - 180  # wrap to [-180, 180]
        errors.append(abs(diff))
    errors = np.array(errors)
    return {
        "rms": float(np.sqrt(np.mean(errors**2))),
        "max": float(np.max(errors)),
        "per_color": {
            name: f"H={h:.1f}° C={c:.3f} (err {e:.1f}°)"
            for name, h, c, e in zip(_PRIMARY_NAMES, H_deg, C, errors)
        },
    }


# ── Objective ────────────────────────────────────────────────────

_eval_count = 0
_best_loss = float("inf")
_last_print_time = 0.0


def _reset_counters():
    global _eval_count, _best_loss, _last_print_time
    _eval_count = 0
    _best_loss = float("inf")
    _last_print_time = 0.0


def make_objective(XYZ_1_c, XYZ_2_c, DV_c, XYZ_1_h, XYZ_2_h, DV_h,
                   he_lambda=0.05, ach_lambda=500.0, hue_lambda=100.0,
                   rt_penalty=20.0):
    """COMBVD + He + achromatic + hue linearity regularizers."""
    rng = np.random.default_rng(42)
    XYZ_rt = rng.uniform(0.05, 0.90, (1000, 3))

    def objective(x):
        global _eval_count, _best_loss, _last_print_time
        try:
            params = unpack_params(x)
            space = AnalyticalSpace(params)

            DE_c = space.distance(XYZ_1_c, XYZ_2_c)
            if np.any(~np.isfinite(DE_c)):
                return 100.0
            s_combvd = stress(DV_c, DE_c)

            DE_h = space.distance(XYZ_1_h, XYZ_2_h)
            if np.any(~np.isfinite(DE_h)):
                return 100.0
            s_he = stress(DV_h, DE_h)

            # Achromatic chroma penalty: mean(C²) for grays
            lab_gray = space.from_XYZ(_GRAY_XYZ)
            ach_C_sq = float(np.mean(lab_gray[:, 1]**2 + lab_gray[:, 2]**2))

            # Hue linearity penalty: mean squared angular error
            hue_pen = compute_hue_penalty(space)

            total = (s_combvd
                     + he_lambda * s_he
                     + ach_lambda * ach_C_sq
                     + hue_lambda * hue_pen)

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
                  f"COMBVD={s_combvd:.2f}  He={s_he:.2f}  "
                  f"achC²={ach_C_sq:.4f}  huePen={hue_pen:.4f}  "
                  f"best={_best_loss:.4f}",
                  flush=True)

        return total

    return objective


# ── Evaluation ───────────────────────────────────────────────────

def evaluate(x, combvd, train, val, he, mac, munsell_pairs):
    p = unpack_params(x)
    s = AnalyticalSpace(p)

    DE_full = s.distance(combvd["XYZ_1"], combvd["XYZ_2"])
    s_train = stress(train["DV"], s.distance(train["XYZ_1"], train["XYZ_2"]))
    s_val = stress(val["DV"], s.distance(val["XYZ_1"], val["XYZ_2"]))
    s_full = stress(combvd["DV"], DE_full)
    s_h = stress(he["DV"], s.distance(he["XYZ_1"], he["XYZ_2"]))
    s_m = stress(mac["DV"], s.distance(mac["XYZ_1"], mac["XYZ_2"]))
    m_cv = compute_munsell_cv(s, munsell_pairs)
    rt = s.round_trip_error(np.random.default_rng(42).uniform(0.05, 0.90, (1000, 3)))
    r_bias = compute_bias_r(combvd["DV"], DE_full)
    ach = compute_achromatic_stats(s)
    ramp_cv = compute_neutral_ramp_cv(s)
    hue = compute_hue_stats(s)

    return {
        "train": s_train, "val": s_val, "full": s_full,
        "he": s_h, "mac": s_m, "cv": m_cv, "rt": rt.max(),
        "r_bias": r_bias,
        "ach_rms": ach["rms"], "ach_max": ach["max"],
        "ramp_cv": ramp_cv,
        "hue_rms": hue["rms"], "hue_max": hue["max"],
        "hue_per_color": hue["per_color"],
    }


def print_eval(label, m):
    gap = m["val"] - m["train"]
    print(f"  {label}: COMBVD={m['full']:.4f}  train={m['train']:.4f}  val={m['val']:.4f}  "
          f"(gap={gap:+.2f})  He={m['he']:.2f}  Mac={m['mac']:.4f}")
    print(f"         CV={m['cv']:.1f}%  |r|={m['r_bias']:.3f}  RT={m['rt']:.2e}  "
          f"achC={m['ach_rms']:.4f}  ramp_CV={m['ramp_cv']:.1f}%")
    print(f"         hue_rms={m['hue_rms']:.1f}°  hue_max={m['hue_max']:.1f}°")
    for name, info in m["hue_per_color"].items():
        print(f"           {name:8s}: {info}")


def run_restarts(objective, x0, bounds, restarts, maxiter,
                 combvd, train, val, he, mac, munsell_pairs):
    best_x = x0.copy()
    best_full = 999.0

    for restart in range(restarts):
        _reset_counters()
        t0 = time.time()
        result = minimize(objective, x0=best_x, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-11})
        dt = time.time() - t0

        try:
            m = evaluate(result.x, combvd, train, val, he, mac, munsell_pairs)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as e:
            print(f"\n  Restart {restart+1}/{restarts} ({dt:.0f}s): FAILED ({e}), skipping")
            continue

        print(f"\n  Restart {restart+1}/{restarts} ({dt:.0f}s):")
        print_eval(f"  R{restart+1}", m)

        if m["rt"] > 1e-6:
            print(f"  WARNING: RT broken, skipping")
            continue

        if m["full"] < best_full:
            best_x = result.x.copy()
            best_full = m["full"]
            print(f"  ** New best (COMBVD={best_full:.4f})")

    return best_x


def main():
    parser = argparse.ArgumentParser(
        description="v17-gen: Achromatic + Hue linearity joint optimization")
    parser.add_argument("--init", type=str, required=True, help="Initial params JSON")
    parser.add_argument("--output", type=str, required=True, help="Output best params JSON")
    parser.add_argument("--restarts", type=int, default=5, help="Restarts per lambda")
    parser.add_argument("--maxiter", type=int, default=5000, help="Max iters per restart")
    parser.add_argument("--he-lambda", type=float, default=0.05)
    parser.add_argument("--q", type=float, default=1.1, help="Fixed dist_post_power")
    parser.add_argument("--ach-lambda", type=float, default=500.0,
                        help="Achromatic lambda (fixed, default=500)")
    parser.add_argument("--hue-lambdas", type=str, default="0,10,50,100,200,500",
                        help="Comma-separated hue lambda values to sweep")
    args = parser.parse_args()

    hue_lambdas = [float(v) for v in args.hue_lambdas.split(",")]

    print(f"v17-gen: Achromatic + Hue linearity optimization")
    print(f"  72 params")
    print(f"  ach_lambda = {args.ach_lambda} (fixed)")
    print(f"  Sweeping hue_lambda: {hue_lambdas}")
    print(f"  dist_post_power = {args.q} (fixed)")
    print(f"  {args.restarts} restarts, {args.maxiter} iters each")
    print(f"Loading data...")

    combvd = load_combvd()
    he = load_he2022()
    mac = load_macadam1974()
    train, val = split_combvd(combvd, seed=42, val_split=0.2)
    munsell_data = load_munsell(subset="real")
    munsell_pairs = generate_munsell_pairs(munsell_data)

    print(f"  COMBVD: {len(combvd['DV'])} pairs")
    print(f"  He: {len(he['DV'])} pairs")
    print(f"  Munsell: {len(munsell_pairs['perceptual_distance'])} pairs")

    # Load initial params
    params = AnalyticalParams.load(args.init)
    x0 = pack_params(params)
    x0[69] = args.q  # fix dist_post_power

    # Baseline evaluation
    m0 = evaluate(x0, combvd, train, val, he, mac, munsell_pairs)
    print(f"\nBaseline (v16-gen):")
    print_eval("v16", m0)

    # ══════════════════════════════════════════════════════════════════
    # Pareto sweep over hue_lambda
    # ══════════════════════════════════════════════════════════════════
    results = []

    for hue_lambda in hue_lambdas:
        print(f"\n{'='*70}")
        print(f"hue_lambda = {hue_lambda}  (ach_lambda = {args.ach_lambda})")
        print(f"{'='*70}")

        obj = make_objective(
            combvd["XYZ_1"], combvd["XYZ_2"], combvd["DV"],
            he["XYZ_1"], he["XYZ_2"], he["DV"],
            he_lambda=args.he_lambda,
            ach_lambda=args.ach_lambda,
            hue_lambda=hue_lambda,
        )

        bounds = make_bounds(x0, fix_post_power=args.q)
        best_x = run_restarts(obj, x0, bounds, args.restarts, args.maxiter,
                              combvd, train, val, he, mac, munsell_pairs)

        m = evaluate(best_x, combvd, train, val, he, mac, munsell_pairs)
        results.append({
            "hue_lambda": hue_lambda,
            "x": best_x,
            "m": m,
        })

    # ══════════════════════════════════════════════════════════════════
    # Summary table
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"PARETO SWEEP SUMMARY  (ach_lambda={args.ach_lambda})")
    print(f"{'='*100}")
    print(f"  {'λ_hue':>6s}  {'COMBVD':>8s}  {'Val':>8s}  {'He':>6s}  {'Mac':>8s}  "
          f"{'CV%':>7s}  {'achC':>7s}  {'hueRMS':>7s}  {'hueMax':>7s}  {'|r|':>5s}")
    print(f"  {'-'*92}")

    for r in results:
        m = r["m"]
        print(f"  {r['hue_lambda']:6.0f}  {m['full']:8.4f}  {m['val']:8.4f}  "
              f"{m['he']:6.2f}  {m['mac']:8.4f}  {m['cv']:6.1f}%  "
              f"{m['ach_rms']:7.4f}  {m['hue_rms']:6.1f}°  {m['hue_max']:6.1f}°  "
              f"{m['r_bias']:5.3f}")

    # Print per-color hue details for each result
    print(f"\n  Per-color hue details:")
    for r in results:
        m = r["m"]
        print(f"  λ_hue={r['hue_lambda']:.0f}:")
        for name, info in m["hue_per_color"].items():
            print(f"    {name:8s}: {info}")

    # Select best trade-off: lowest hue_rms with achC < 0.05 and COMBVD < 28
    good = [r for r in results
            if r["m"]["ach_rms"] < 0.05 and r["m"]["rt"] <= 1e-6]
    if good:
        # Among those with good achromatic, pick lowest hue_rms with COMBVD < 28
        good_hue = [r for r in good if r["m"]["full"] < 28.0]
        if good_hue:
            best = min(good_hue, key=lambda r: r["m"]["hue_rms"])
            print(f"\n  Best trade-off (achC<0.05, COMBVD<28): "
                  f"λ_hue={best['hue_lambda']}, "
                  f"COMBVD={best['m']['full']:.4f}, "
                  f"achC={best['m']['ach_rms']:.4f}, "
                  f"hue_rms={best['m']['hue_rms']:.1f}°")
        else:
            best = min(good, key=lambda r: r["m"]["full"])
            print(f"\n  No result with COMBVD<28. Best achC<0.05: "
                  f"λ_hue={best['hue_lambda']}, "
                  f"COMBVD={best['m']['full']:.4f}, "
                  f"achC={best['m']['ach_rms']:.4f}, "
                  f"hue_rms={best['m']['hue_rms']:.1f}°")
    else:
        # Fallback: pick lowest COMBVD overall
        best = min(results, key=lambda r: r["m"]["full"])
        print(f"\n  No result with achC<0.05. Saving lowest COMBVD: "
              f"λ_hue={best['hue_lambda']}, "
              f"COMBVD={best['m']['full']:.4f}")

    final_params = unpack_params(best["x"])
    final_params.save(args.output)
    print(f"  Saved to {args.output}")

    # Save all checkpoints
    print(f"\n  All checkpoints saved:")
    for r in results:
        path = f"checkpoints/v17_hue{r['hue_lambda']:.0f}.json"
        unpack_params(r["x"]).save(path)
        m = r["m"]
        print(f"    {path}: COMBVD={m['full']:.4f}, achC={m['ach_rms']:.4f}, "
              f"hue_rms={m['hue_rms']:.1f}°")


if __name__ == "__main__":
    main()
