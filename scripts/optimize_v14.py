#!/usr/bin/env python
"""v14 optimization: 70 params — post-compress power grid search + Munsell CV + sub-dataset weighting.

Strategy (revised after bias analysis):
  dist_linear was ineffective (v12 basin too deep, optimizer can't compensate).
  Post-compress power DE^q is much more effective: q=1.1 → |r|=0.24, q=1.3 → |r|≈0.

  Phase B: Grid search over fixed dist_post_power values
           At each q, re-optimize all other params (COMBVD + He).
           Pick best Pareto point (COMBVD ≤ baseline + tolerance, lowest |r|).

  Phase A: Munsell CV fine-tuning from Phase B best.
  Phase C: Equal sub-dataset weighting (optional).

Param layout: 68 v13 params + [68]=dist_linear + [69]=dist_post_power + [70]=dist_sl + [71]=dist_sc
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

N_PARAMS = 72


def split_combvd(combvd, seed=42, val_split=0.2):
    """Split COMBVD into train/val using same split as neural model."""
    n = len(combvd["DV"])
    import torch
    g = torch.Generator().manual_seed(seed)
    n_val = int(n * val_split)
    n_train = n - n_val
    indices = torch.randperm(n, generator=g).numpy()
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    train = {
        "XYZ_1": combvd["XYZ_1"][train_idx],
        "XYZ_2": combvd["XYZ_2"][train_idx],
        "DV": combvd["DV"][train_idx],
    }
    val = {
        "XYZ_1": combvd["XYZ_1"][val_idx],
        "XYZ_2": combvd["XYZ_2"][val_idx],
        "DV": combvd["DV"][val_idx],
    }
    return train, val


def pack_params(p: AnalyticalParams) -> np.ndarray:
    return np.concatenate([
        p.M1.ravel(), p.gamma, p.M2.ravel(),                   # 0-20
        [p.hk_weight], [p.hk_power], [p.hk_hue_mod],          # 21-23
        [p.L_corr_p1], [p.L_corr_p2], [p.L_corr_p3],          # 24-26
        [p.cs_cos1], [p.cs_sin1], [p.cs_cos2], [p.cs_sin2],    # 27-30
        [p.cs_cos3], [p.cs_sin3],                               # 31-32
        [p.lc1], [p.lc2],                                       # 33-34
        [p.hk_sin1], [p.hk_cos2], [p.hk_sin2],                 # 35-37
        [p.hue_cos1], [p.hue_sin1], [p.hue_cos2], [p.hue_sin2],# 38-41
        [p.hue_cos3], [p.hue_sin3],                             # 42-43
        [p.hlc_cos1], [p.hlc_sin1], [p.hlc_cos2], [p.hlc_sin2],# 44-47
        [p.hl_cos1], [p.hl_sin1], [p.hl_cos2], [p.hl_sin2],    # 48-51
        [p.cp_cos1], [p.cp_sin1], [p.cp_cos2], [p.cp_sin2],    # 52-55
        [p.lp_dark],                                             # 56
        [p.dist_power], [p.dist_wC],                             # 57-58
        [p.hue_cos4], [p.hue_sin4],                              # 59-60
        [p.Lh_cos1], [p.Lh_sin1],                                # 61-62
        [p.cs_cos4], [p.cs_sin4],                                 # 63-64
        [p.dist_compress],                                        # 65 (v12)
        [p.lp_dark_hcos], [p.lp_dark_hsin],                      # 66-67 (v13)
        [p.dist_linear],                                          # 68 (v14)
        [p.dist_post_power],                                      # 69 (v14b)
        [p.dist_sl],                                              # 70 (v14c NEW)
        [p.dist_sc],                                              # 71 (v14c NEW)
    ])


def unpack_params(x: np.ndarray) -> AnalyticalParams:
    return AnalyticalParams(
        M1=x[0:9].reshape(3, 3), gamma=x[9:12], M2=x[12:21].reshape(3, 3),
        hk_weight=float(x[21]), hk_power=float(x[22]), hk_hue_mod=float(x[23]),
        L_corr_p1=float(x[24]), L_corr_p2=float(x[25]), L_corr_p3=float(x[26]),
        cs_cos1=float(x[27]), cs_sin1=float(x[28]),
        cs_cos2=float(x[29]), cs_sin2=float(x[30]),
        cs_cos3=float(x[31]), cs_sin3=float(x[32]),
        lc1=float(x[33]), lc2=float(x[34]),
        hk_sin1=float(x[35]), hk_cos2=float(x[36]), hk_sin2=float(x[37]),
        hue_cos1=float(x[38]), hue_sin1=float(x[39]),
        hue_cos2=float(x[40]), hue_sin2=float(x[41]),
        hue_cos3=float(x[42]), hue_sin3=float(x[43]),
        hlc_cos1=float(x[44]), hlc_sin1=float(x[45]),
        hlc_cos2=float(x[46]), hlc_sin2=float(x[47]),
        hl_cos1=float(x[48]), hl_sin1=float(x[49]),
        hl_cos2=float(x[50]), hl_sin2=float(x[51]),
        cp_cos1=float(x[52]), cp_sin1=float(x[53]),
        cp_cos2=float(x[54]), cp_sin2=float(x[55]),
        lp_dark=float(x[56]),
        dist_power=float(x[57]), dist_wC=float(x[58]),
        hue_cos4=float(x[59]), hue_sin4=float(x[60]),
        Lh_cos1=float(x[61]), Lh_sin1=float(x[62]),
        cs_cos4=float(x[63]), cs_sin4=float(x[64]),
        dist_compress=float(x[65]),
        lp_dark_hcos=float(x[66]),
        lp_dark_hsin=float(x[67]),
        dist_linear=float(x[68]),
        dist_post_power=float(x[69]),
        dist_sl=float(x[70]),
        dist_sc=float(x[71]),
    )


def make_bounds(x0, fix_post_power=None):
    """Create parameter bounds. Optionally fix dist_post_power to a specific value."""
    bounds = []
    for i in range(N_PARAMS):
        if i < 9 or (12 <= i < 21):
            center = x0[i]
            half = max(abs(center) * 1.0, 0.5)
            bounds.append((max(center - half, -3.0), min(center + half, 3.0)))
        elif 9 <= i < 12:
            bounds.append((0.15, 0.95))
        elif i == 21:  # hk_weight
            bounds.append((0.0, 3.0))
        elif i == 22:  # hk_power
            bounds.append((0.1, 2.0))
        elif i == 23:  # hk_hue_mod
            bounds.append((-1.0, 1.0))
        elif 24 <= i <= 26:  # L_corr
            if i == 25:  # L_corr_p2
                bounds.append((-1.2, 1.2))
            else:
                bounds.append((-0.8, 0.8))
        elif 27 <= i <= 32:
            bounds.append((-1.0, 1.0))
        elif i in (33, 34):
            bounds.append((-1.0, 1.0))
        elif 35 <= i <= 37:  # hk harmonics
            bounds.append((-1.5, 1.5))
        elif 38 <= i <= 43:
            bounds.append((-0.5, 0.5))
        elif 44 <= i <= 47:
            bounds.append((-1.0, 1.0))
        elif 48 <= i <= 51:
            bounds.append((-0.3, 0.3))
        elif 52 <= i <= 55:
            bounds.append((-0.5, 0.5))
        elif i == 56:  # lp_dark
            bounds.append((-0.5, 1.0))
        elif i == 57:  # dist_power
            bounds.append((0.5, 1.5))
        elif i == 58:  # dist_wC
            bounds.append((0.3, 3.0))
        elif 59 <= i <= 60:
            bounds.append((-0.3, 0.3))
        elif 61 <= i <= 62:
            bounds.append((-0.5, 0.5))
        elif 63 <= i <= 64:
            bounds.append((-0.5, 0.5))
        elif i == 65:  # dist_compress (v12) — must be >= 0
            bounds.append((0.0, 10.0))
        elif 66 <= i <= 67:  # lp_dark_hcos, lp_dark_hsin (v13)
            bounds.append((-0.5, 0.5))
        elif i == 68:  # dist_linear (v14) — fix at 0
            bounds.append((0.0, 0.0))
        elif i == 69:  # dist_post_power (v14b)
            if fix_post_power is not None:
                bounds.append((fix_post_power, fix_post_power))
            else:
                bounds.append((0.8, 1.5))
        elif i == 70:  # dist_sl (v14c) — L-dep weight
            bounds.append((-5.0, 10.0))
        elif i == 71:  # dist_sc (v14c) — C-dep weight
            bounds.append((-2.0, 5.0))
    return bounds


def compute_bias_r(DV, DE):
    """Compute |r| = |corr(residual, DV)| where residual = DV - F*DE."""
    DE = np.asarray(DE, dtype=np.float64)
    DV = np.asarray(DV, dtype=np.float64)
    sum_DV_DE = np.sum(DV * DE)
    sum_DE2 = np.sum(DE ** 2)
    if sum_DE2 < 1e-20:
        return 1.0
    F = sum_DV_DE / sum_DE2
    residual = DV - F * DE
    r = np.corrcoef(residual, DV)[0, 1]
    if not np.isfinite(r):
        return 1.0
    return abs(r)


def compute_munsell_cv(space, munsell_pairs):
    """Compute coefficient of variation for Munsell neighbor distances."""
    DE = space.distance(munsell_pairs["XYZ_1"], munsell_pairs["XYZ_2"])
    if np.any(~np.isfinite(DE)) or np.mean(DE) < 1e-10:
        return 200.0
    return float(np.std(DE) / np.mean(DE) * 100.0)


def compute_subdataset_stress(space, combvd):
    """Compute STRESS per sub-dataset in COMBVD."""
    datasets = combvd["dataset"]
    unique_ds = sorted(set(datasets))
    results = {}
    for ds in unique_ds:
        mask = np.array([d == ds for d in datasets])
        n = np.sum(mask)
        if n < 2:
            continue
        DE = space.distance(combvd["XYZ_1"][mask], combvd["XYZ_2"][mask])
        DV = combvd["DV"][mask]
        s = stress(DV, DE)
        results[ds] = {"stress": s, "n": int(n)}
    return results


# ── Objective factories ───────────────────────────────────────────

_eval_count = 0
_best_loss = float("inf")
_last_print_time = 0.0


def _reset_counters():
    global _eval_count, _best_loss, _last_print_time
    _eval_count = 0
    _best_loss = float("inf")
    _last_print_time = 0.0


def make_objective_combvd_he(XYZ_1_c, XYZ_2_c, DV_c, XYZ_1_h, XYZ_2_h, DV_h,
                             he_lambda=0.05, rt_penalty=20.0):
    """Standard COMBVD + He objective (no bias penalty)."""
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


def make_objective_with_cv(XYZ_1_c, XYZ_2_c, DV_c, XYZ_1_h, XYZ_2_h, DV_h,
                           munsell_pairs, he_lambda=0.05, munsell_lambda=0.02,
                           rt_penalty=20.0):
    """COMBVD + He + Munsell CV objective."""
    rng = np.random.default_rng(42)
    XYZ_rt = rng.uniform(0.05, 0.90, (1000, 3))
    m_XYZ_1 = munsell_pairs["XYZ_1"]
    m_XYZ_2 = munsell_pairs["XYZ_2"]

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

            DE_m = space.distance(m_XYZ_1, m_XYZ_2)
            if np.any(~np.isfinite(DE_m)) or np.mean(DE_m) < 1e-10:
                return 100.0
            munsell_cv = float(np.std(DE_m) / np.mean(DE_m) * 100.0)

            total = s_combvd + he_lambda * s_he + munsell_lambda * munsell_cv

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
                  f"CV={munsell_cv:.1f}%  best={_best_loss:.4f}",
                  flush=True)

        return total

    return objective


def make_objective_equal_sub(combvd, XYZ_1_h, XYZ_2_h, DV_h,
                             munsell_pairs, he_lambda=0.05, munsell_lambda=0.02,
                             rt_penalty=20.0):
    """Equal sub-dataset weighting + He + Munsell CV."""
    rng = np.random.default_rng(42)
    XYZ_rt = rng.uniform(0.05, 0.90, (1000, 3))
    m_XYZ_1 = munsell_pairs["XYZ_1"]
    m_XYZ_2 = munsell_pairs["XYZ_2"]

    datasets = combvd["dataset"]
    unique_ds = sorted(set(datasets))
    sub_masks = {}
    for ds in unique_ds:
        mask = np.array([d == ds for d in datasets])
        if np.sum(mask) >= 2:
            sub_masks[ds] = mask

    def objective(x):
        global _eval_count, _best_loss, _last_print_time
        try:
            params = unpack_params(x)
            space = AnalyticalSpace(params)

            sub_stresses = []
            for ds, mask in sub_masks.items():
                DE_sub = space.distance(combvd["XYZ_1"][mask], combvd["XYZ_2"][mask])
                if np.any(~np.isfinite(DE_sub)):
                    return 100.0
                sub_stresses.append(stress(combvd["DV"][mask], DE_sub))

            s_equal = float(np.mean(sub_stresses))

            DE_h = space.distance(XYZ_1_h, XYZ_2_h)
            if np.any(~np.isfinite(DE_h)):
                return 100.0
            s_he = stress(DV_h, DE_h)

            DE_m = space.distance(m_XYZ_1, m_XYZ_2)
            if np.any(~np.isfinite(DE_m)) or np.mean(DE_m) < 1e-10:
                return 100.0
            munsell_cv = float(np.std(DE_m) / np.mean(DE_m) * 100.0)

            total = s_equal + he_lambda * s_he + munsell_lambda * munsell_cv

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
                  f"EqSTRESS={s_equal:.2f}  He={s_he:.2f}  "
                  f"CV={munsell_cv:.1f}%  best={_best_loss:.4f}",
                  flush=True)

        return total

    return objective


# ── Evaluation and reporting ──────────────────────────────────────

def evaluate(x, combvd, train, val, he, mac, munsell_pairs):
    """Full evaluation: returns dict of all metrics."""
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

    return {
        "train": s_train, "val": s_val, "full": s_full,
        "he": s_h, "mac": s_m, "cv": m_cv, "rt": rt.max(),
        "r_bias": r_bias,
    }


def print_eval(label, m):
    """Print evaluation metrics."""
    gap = m["val"] - m["train"]
    print(f"  {label}: train={m['train']:.4f}  val={m['val']:.4f}  (gap={gap:+.2f})  "
          f"full={m['full']:.4f}  He={m['he']:.2f}  Mac={m['mac']:.4f}  "
          f"CV={m['cv']:.1f}%  |r|={m['r_bias']:.3f}  RT={m['rt']:.2e}")


def print_subdataset(x, combvd):
    """Print per-sub-dataset STRESS."""
    p = unpack_params(x)
    s = AnalyticalSpace(p)
    sub = compute_subdataset_stress(s, combvd)
    print(f"  Per-sub-dataset STRESS:")
    for ds_name in sorted(sub.keys()):
        info = sub[ds_name]
        print(f"    {ds_name:20s}  n={info['n']:>4d}  STRESS={info['stress']:.2f}")


def run_restarts(objective, x0, bounds, restarts, maxiter,
                 combvd, train, val, he, mac, munsell_pairs):
    """Run optimization with Hessian restarts, return best x and metrics."""
    best_x = x0.copy()
    best_full = 999.0

    for restart in range(restarts):
        _reset_counters()

        t0 = time.time()
        result = minimize(objective, x0=best_x, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-11})
        dt = time.time() - t0

        m = evaluate(result.x, combvd, train, val, he, mac, munsell_pairs)
        gap = m["val"] - m["train"]
        print(f"  Restart {restart+1}: train={m['train']:.4f}  val={m['val']:.4f}  "
              f"(gap={gap:+.2f})  full={m['full']:.4f}  He={m['he']:.2f}  "
              f"Mac={m['mac']:.4f}  CV={m['cv']:.1f}%  |r|={m['r_bias']:.3f}  "
              f"RT={m['rt']:.2e}  ({dt:.0f}s)")

        if m["rt"] > 1e-6:
            print(f"  WARNING: RT broken, skipping")
            continue

        if m["full"] < best_full:
            best_x = result.x.copy()
            best_full = m["full"]
            print(f"  ** New best (COMBVD={best_full:.4f})")

    m_best = evaluate(best_x, combvd, train, val, he, mac, munsell_pairs)
    return best_x, m_best


def main():
    parser = argparse.ArgumentParser(description="v14: dist_linear grid search + Munsell CV + sub-dataset weighting")
    parser.add_argument("--init", type=str, required=True, help="Initial params JSON")
    parser.add_argument("--output", type=str, required=True, help="Output best params JSON")
    parser.add_argument("--restarts-b", type=int, default=5, help="Restarts per alpha in Phase B grid")
    parser.add_argument("--restarts-a", type=int, default=5, help="Restarts for Phase A")
    parser.add_argument("--restarts-c", type=int, default=5, help="Restarts for Phase C")
    parser.add_argument("--maxiter-b", type=int, default=5000, help="Max iterations per restart, Phase B")
    parser.add_argument("--maxiter-a", type=int, default=3000, help="Max iterations per restart, Phase A")
    parser.add_argument("--maxiter-c", type=int, default=3000, help="Max iterations per restart, Phase C")
    parser.add_argument("--he-lambda", type=float, default=0.05, help="He 2022 regularizer weight")
    parser.add_argument("--munsell-lambda", type=float, default=0.02, help="Munsell CV weight (Phase A)")
    parser.add_argument("--combvd-tolerance", type=float, default=0.15,
                        help="Max COMBVD regression allowed for bias gain (default: 0.15)")
    parser.add_argument("--q-values", type=str, default="1.0,1.05,1.1,1.15,1.2",
                        help="Comma-separated dist_post_power values to grid search")
    parser.add_argument("--skip-phase-a", action="store_true", help="Skip Phase A (Munsell)")
    parser.add_argument("--skip-phase-c", action="store_true", help="Skip Phase C (sub-dataset)")
    args = parser.parse_args()

    q_values = [float(v) for v in args.q_values.split(",")]

    print(f"v14: {N_PARAMS} params — post-compress power grid + Munsell CV + sub-dataset weighting")
    print(f"  Phase B: Grid search dist_post_power over {q_values}")
    print(f"           COMBVD + {args.he_lambda}*He ({args.restarts_b} restarts/q)")
    print(f"           COMBVD tolerance: +{args.combvd_tolerance}")
    print(f"  Phase A: + {args.munsell_lambda}*Munsell_CV" + (" [SKIPPED]" if args.skip_phase_a else ""))
    print(f"  Phase C: Equal sub-dataset weighting" + (" [SKIPPED]" if args.skip_phase_c else ""))
    print(f"  {N_PARAMS} parameters (v13 + dist_linear + dist_post_power + dist_sl + dist_sc)")
    print("Loading data...")

    combvd = load_combvd()
    he = load_he2022()
    mac = load_macadam1974()

    train, val = split_combvd(combvd, seed=42, val_split=0.2)
    print(f"  COMBVD total: {len(combvd['DV'])} pairs")
    print(f"  COMBVD train: {len(train['DV'])} pairs")
    print(f"  COMBVD val:   {len(val['DV'])} pairs")
    print(f"  He 2022: {len(he['DV'])} pairs")

    print("Loading Munsell data...")
    munsell_data = load_munsell(subset="real")
    munsell_pairs = generate_munsell_pairs(munsell_data)
    print(f"  Munsell pairs: {len(munsell_pairs['perceptual_distance'])}")

    # Load initial params
    params = AnalyticalParams.load(args.init)
    if params.dist_nl != 0.0 and params.dist_compress == 0.0:
        params.dist_compress = abs(params.dist_nl) * 1.5
        params.dist_nl = 0.0
        print(f"  Migrated dist_nl -> dist_compress = {params.dist_compress:.4f}")

    x0 = pack_params(params)
    assert len(x0) == N_PARAMS, f"Expected {N_PARAMS} params, got {len(x0)}"

    # Initial evaluation
    m0 = evaluate(x0, combvd, train, val, he, mac, munsell_pairs)
    print(f"\nInitial state:")
    print_eval("Init", m0)
    print_subdataset(x0, combvd)
    baseline_full = m0["full"]

    # ══════════════════════════════════════════════════════════════════
    # Phase B: Grid search over fixed dist_linear values
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"Phase B: dist_linear grid search")
    print(f"{'='*70}")

    objective_b = make_objective_combvd_he(
        combvd["XYZ_1"], combvd["XYZ_2"], combvd["DV"],
        he["XYZ_1"], he["XYZ_2"], he["DV"],
        he_lambda=args.he_lambda,
    )

    grid_results = []
    for q in q_values:
        print(f"\n{'─'*70}")
        print(f"  dist_post_power = {q:.3f} (fixed)")
        print(f"{'─'*70}")

        # Set dist_post_power and fix it via bounds
        x_init = x0.copy()
        x_init[69] = q
        bounds = make_bounds(x0, fix_post_power=q)

        best_x, m = run_restarts(
            objective_b, x_init, bounds, args.restarts_b, args.maxiter_b,
            combvd, train, val, he, mac, munsell_pairs,
        )

        grid_results.append({
            "q": q, "x": best_x, "m": m,
        })
        print(f"\n  q={q:.3f} summary:")
        print_eval(f"q={q:.3f}", m)

    # Print grid summary
    print(f"\n{'='*70}")
    print(f"Phase B grid summary:")
    print(f"{'='*70}")
    print(f"  {'q':>6s}  {'COMBVD':>8s}  {'Val':>8s}  {'|r|':>6s}  {'He':>6s}  {'Mac':>8s}  {'CV':>7s}  {'RT':>10s}")
    for r in grid_results:
        m = r["m"]
        print(f"  {r['q']:6.3f}  {m['full']:8.4f}  {m['val']:8.4f}  {m['r_bias']:6.3f}  "
              f"{m['he']:6.2f}  {m['mac']:8.4f}  {m['cv']:6.1f}%  {m['rt']:.2e}")

    # Select best: lowest |r| within COMBVD tolerance
    eligible = [r for r in grid_results
                if r["m"]["full"] <= baseline_full + args.combvd_tolerance
                and r["m"]["rt"] <= 1e-6]
    if not eligible:
        print(f"\n  WARNING: No q within tolerance. Using q=1.0 (baseline).")
        eligible = [r for r in grid_results if r["q"] == 1.0]
        if not eligible:
            eligible = [grid_results[0]]

    best_grid = min(eligible, key=lambda r: r["m"]["r_bias"])
    best_x = best_grid["x"]
    best_q = best_grid["q"]
    m_b = best_grid["m"]

    print(f"\n  Selected: q={best_q:.3f}  COMBVD={m_b['full']:.4f}  |r|={m_b['r_bias']:.3f}")
    print_subdataset(best_x, combvd)

    # Save Phase B checkpoint
    p_b = unpack_params(best_x)
    p_b.save("checkpoints/v14_phase_b.json")
    print(f"  Saved: checkpoints/v14_phase_b.json")

    # ══════════════════════════════════════════════════════════════════
    # Phase A: Munsell CV fine-tuning
    # ══════════════════════════════════════════════════════════════════
    if not args.skip_phase_a:
        print(f"\n{'='*70}")
        print(f"Phase A: Munsell CV fine-tuning (from q={best_q:.3f})")
        print(f"{'='*70}")

        objective_a = make_objective_with_cv(
            combvd["XYZ_1"], combvd["XYZ_2"], combvd["DV"],
            he["XYZ_1"], he["XYZ_2"], he["DV"],
            munsell_pairs,
            he_lambda=args.he_lambda, munsell_lambda=args.munsell_lambda,
        )
        bounds_a = make_bounds(x0, fix_post_power=best_q)

        best_x, m_a = run_restarts(
            objective_a, best_x, bounds_a, args.restarts_a, args.maxiter_a,
            combvd, train, val, he, mac, munsell_pairs,
        )

        print(f"\n  Phase A best:")
        print_eval("Result", m_a)
        print_subdataset(best_x, combvd)

        p_a = unpack_params(best_x)
        p_a.save("checkpoints/v14_phase_a.json")
        print(f"  Saved: checkpoints/v14_phase_a.json")

    # ══════════════════════════════════════════════════════════════════
    # Phase C: Equal sub-dataset weighting (optional)
    # ══════════════════════════════════════════════════════════════════
    if not args.skip_phase_c:
        print(f"\n{'='*70}")
        print(f"Phase C: Equal sub-dataset weighting")
        print(f"{'='*70}")

        objective_c = make_objective_equal_sub(
            combvd, he["XYZ_1"], he["XYZ_2"], he["DV"],
            munsell_pairs,
            he_lambda=args.he_lambda, munsell_lambda=args.munsell_lambda,
        )
        bounds_c = make_bounds(x0, fix_post_power=best_q)

        m_before_c = evaluate(best_x, combvd, train, val, he, mac, munsell_pairs)
        best_x_c, m_c = run_restarts(
            objective_c, best_x, bounds_c, args.restarts_c, args.maxiter_c,
            combvd, train, val, he, mac, munsell_pairs,
        )

        if m_c["full"] <= m_before_c["full"] + 0.1:
            best_x = best_x_c
            print(f"\n  Phase C accepted (COMBVD {m_c['full']:.4f} vs pre-C {m_before_c['full']:.4f})")
        else:
            print(f"\n  Phase C REJECTED — COMBVD regression: {m_c['full']:.4f} > {m_before_c['full']:.4f} + 0.1")
            print(f"  Keeping Phase A result")

    # ══════════════════════════════════════════════════════════════════
    # Final report
    # ══════════════════════════════════════════════════════════════════
    final_params = unpack_params(best_x)
    m_final = evaluate(best_x, combvd, train, val, he, mac, munsell_pairs)

    bounds_final = make_bounds(x0)
    at_bounds = []
    for j, (lo, hi) in enumerate(bounds_final):
        if abs(best_x[j] - lo) < 1e-4 or abs(best_x[j] - hi) < 1e-4:
            at_bounds.append(j)

    print(f"\n{'='*70}")
    print(f"v14 FINAL RESULTS")
    print(f"{'='*70}")
    print_eval("Final", m_final)
    print(f"{'─'*70}")
    print(f"Distance params:")
    print(f"  dist_compress: {final_params.dist_compress:.6f}")
    print(f"  dist_linear:   {final_params.dist_linear:.6f}")
    print(f"  dist_post_power: {final_params.dist_post_power:.6f}  (v14b, grid-selected)")
    print(f"  dist_power:    {final_params.dist_power:.4f}")
    print(f"  dist_wC:       {final_params.dist_wC:.4f}")
    print(f"  dist_sl:       {final_params.dist_sl:.6f}  (v14c, L-dep weight)")
    print(f"  dist_sc:       {final_params.dist_sc:.6f}  (v14c, C-dep weight)")
    print(f"Dark L params:")
    print(f"  lp_dark:       {final_params.lp_dark:.6f}")
    print(f"  lp_dark_hcos:  {final_params.lp_dark_hcos:.6f}")
    print(f"  lp_dark_hsin:  {final_params.lp_dark_hsin:.6f}")
    print(f"{'─'*70}")
    print_subdataset(best_x, combvd)
    print(f"{'─'*70}")
    print(f"vs v12 baseline:")
    print(f"  COMBVD: {m_final['full']:.4f}  (v12: {baseline_full:.4f})")
    print(f"  |r|:    {m_final['r_bias']:.3f}  (v12: {m0['r_bias']:.3f})")
    print(f"  CV:     {m_final['cv']:.1f}%  (v12: {m0['cv']:.1f}%)")
    print(f"  He:     {m_final['he']:.2f}  (v12: {m0['he']:.2f})")
    print(f"  Mac:    {m_final['mac']:.4f}  (v12: {m0['mac']:.4f})")
    if at_bounds:
        print(f"{'─'*70}")
        print(f"WARNING: {len(at_bounds)} params at bounds: {at_bounds}")
    print(f"{'='*70}")

    final_params.save(args.output)
    print(f"\nSaved to {args.output}")

    final_params.save("checkpoints/v14_best.json")
    print(f"Saved to checkpoints/v14_best.json")


if __name__ == "__main__":
    main()
