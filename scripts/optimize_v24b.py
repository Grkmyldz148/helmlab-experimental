#!/usr/bin/env python
"""v24b: Targeted optimization from Pareto sweep results.

Strategy:
1. Start from mac++osa++ (best MacAdam+OSA), try to recover COMBVD
2. Start from baseline, use massive Mac+OSA weights but with COMBVD floor
3. CMA-ES from multiple starting points with balanced objective

Key insight from sweep:
- Low mac/osa weights: optimizer doesn't move (COMBVD gradient dominates)
- High weights: Mac=18.78 achievable, OSA=18.1% achievable, but COMBVD rises to ~24.4

Need: Either find a basin where all metrics are good, or accept a trade-off.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.optimize import minimize

from helmlab.data.combvd import load_combvd
from helmlab.data.he2022 import load_he2022
from helmlab.data.macadam1974 import load_macadam1974
from helmlab.data.munsell import load_munsell, generate_munsell_pairs
from helmlab.data.osa_ucs import load_osa_ucs, generate_osa_pairs
from helmlab.metrics.stress import stress
from helmlab.spaces.metric import MetricSpace, MetricParams

N_PARAMS = 72


def pack_params(p: MetricParams) -> np.ndarray:
    return np.concatenate([
        p.M1.ravel(), p.gamma, p.M2.ravel(),
        [p.hk_weight], [p.hk_power], [p.hk_hue_mod],
        [p.L_corr_p1], [p.L_corr_p2], [p.L_corr_p3],
        [p.cs_cos1], [p.cs_sin1], [p.cs_cos2], [p.cs_sin2],
        [p.cs_cos3], [p.cs_sin3],
        [p.lc1], [p.lc2],
        [p.hk_sin1], [p.hk_cos2], [p.hk_sin2],
        [p.hue_cos1], [p.hue_sin1], [p.hue_cos2], [p.hue_sin2],
        [p.hue_cos3], [p.hue_sin3],
        [p.hlc_cos1], [p.hlc_sin1], [p.hlc_cos2], [p.hlc_sin2],
        [p.hl_cos1], [p.hl_sin1], [p.hl_cos2], [p.hl_sin2],
        [p.cp_cos1], [p.cp_sin1], [p.cp_cos2], [p.cp_sin2],
        [p.lp_dark],
        [p.dist_power], [p.dist_wC],
        [p.hue_cos4], [p.hue_sin4],
        [p.Lh_cos1], [p.Lh_sin1],
        [p.cs_cos4], [p.cs_sin4],
        [p.dist_compress],
        [p.lp_dark_hcos], [p.lp_dark_hsin],
        [p.dist_linear],
        [p.dist_post_power],
        [p.dist_sl],
        [p.dist_sc],
    ])


def unpack_params(x: np.ndarray) -> MetricParams:
    return MetricParams(
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


def make_bounds(x0, wider=False):
    w = 1.5 if wider else 1.0
    bounds = []
    for i in range(N_PARAMS):
        if i < 9 or (12 <= i < 21):
            center = x0[i]
            half = max(abs(center) * w, 0.5 * w)
            bounds.append((max(center - half, -3.0), min(center + half, 3.0)))
        elif 9 <= i < 12:
            bounds.append((0.15, 0.95))
        elif i == 21:
            bounds.append((0.0, 3.0))
        elif i == 22:
            bounds.append((0.1, 2.0))
        elif i == 23:
            bounds.append((-1.0, 1.0))
        elif 24 <= i <= 26:
            bounds.append((-1.2 if i == 25 else -0.8, 1.2 if i == 25 else 0.8))
        elif 27 <= i <= 32:
            bounds.append((-1.0, 1.0))
        elif i in (33, 34):
            bounds.append((-1.0, 1.0))
        elif 35 <= i <= 37:
            bounds.append((-1.5, 1.5))
        elif 38 <= i <= 43:
            bounds.append((-0.5, 0.5))
        elif 44 <= i <= 47:
            bounds.append((-1.0, 1.0))
        elif 48 <= i <= 51:
            bounds.append((-0.3, 0.3))
        elif 52 <= i <= 55:
            bounds.append((-0.5, 0.5))
        elif i == 56:
            bounds.append((-0.5, 1.0))
        elif i == 57:
            bounds.append((0.5, 1.5))
        elif i == 58:
            bounds.append((0.3, 3.0))
        elif 59 <= i <= 60:
            bounds.append((-0.3, 0.3))
        elif 61 <= i <= 62:
            bounds.append((-0.5, 0.5))
        elif 63 <= i <= 64:
            bounds.append((-0.5, 0.5))
        elif i == 65:
            bounds.append((0.0, 10.0))
        elif 66 <= i <= 67:
            bounds.append((-0.5, 0.5))
        elif i == 68:
            bounds.append((0.0, 0.0))
        elif i == 69:
            bounds.append((0.8, 1.5))
        elif i == 70:
            bounds.append((-5.0, 10.0))
        elif i == 71:
            bounds.append((-2.0, 5.0))
    return bounds


def full_eval(x, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt):
    try:
        p = unpack_params(x)
        s = MetricSpace(p, neutral_correction=True)
        DE_c = s.distance(combvd["XYZ_1"], combvd["XYZ_2"])
        s_combvd = stress(combvd["DV"], DE_c)
        DE_h = s.distance(he["XYZ_1"], he["XYZ_2"])
        s_he = stress(he["DV"], DE_h)
        DE_m = s.distance(mac["XYZ_1"], mac["XYZ_2"])
        s_mac = stress(mac["DV"], DE_m)
        DE_mu = s.distance(munsell_pairs["XYZ_1"], munsell_pairs["XYZ_2"])
        munsell_cv = float(np.std(DE_mu) / np.mean(DE_mu) * 100.0)
        DE_osa = s.distance(osa_pairs["XYZ_1"], osa_pairs["XYZ_2"])
        osa_cv = float(np.std(DE_osa) / np.mean(DE_osa) * 100.0)
        rt_errors = s.round_trip_error(XYZ_rt)
        rt = float(rt_errors.max())
        return {"combvd": s_combvd, "he": s_he, "mac": s_mac,
                "munsell_cv": munsell_cv, "osa_cv": osa_cv, "rt": rt}
    except Exception as e:
        return {"combvd": 100.0, "he": 100.0, "mac": 100.0,
                "munsell_cv": 200.0, "osa_cv": 200.0, "rt": 1.0}


def print_eval(label, m):
    flags = ""
    flags += "C" if m["combvd"] <= 23.30 else "."
    flags += "H" if m["he"] <= 30.33 else "."
    flags += "M" if m["mac"] <= 18.71 else "."
    flags += "U" if m["munsell_cv"] <= 40.0 else "."
    flags += "O" if m["osa_cv"] <= 16.6 else "."
    print(f"  {label}: COMBVD={m['combvd']:.2f}  He={m['he']:.2f}  Mac={m['mac']:.2f}  "
          f"MunCV={m['munsell_cv']:.1f}%  OsaCV={m['osa_cv']:.1f}%  RT={m['rt']:.2e}  [{flags}]")


def score_v2(m):
    """Score with asymmetric penalties — harder penalty for being over target."""
    targets = {"combvd": 23.30, "he": 30.33, "mac": 18.71, "munsell_cv": 40.0, "osa_cv": 16.6}
    # Weights: how important each metric is
    weights = {"combvd": 2.0, "he": 1.0, "mac": 2.0, "munsell_cv": 0.5, "osa_cv": 2.0}
    penalty = 0.0
    for key, target in targets.items():
        val = m[key]
        if val > target:
            excess = (val - target) / target
            penalty += weights[key] * excess ** 2  # quadratic penalty
    return penalty


_eval_count = 0
_best_loss = float("inf")
_last_print_time = 0.0


def reset_counters():
    global _eval_count, _best_loss, _last_print_time
    _eval_count = 0
    _best_loss = float("inf")
    _last_print_time = 0.0


def make_constrained_objective(combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt,
                                 combvd_cap=23.50, he_cap=31.0):
    """Objective that HEAVILY penalizes COMBVD/He regression while pushing Mac/OSA.

    Strategy: Use barrier function for COMBVD and He, optimize Mac+OSA directly.
    """
    c_XYZ1, c_XYZ2, c_DV = combvd["XYZ_1"], combvd["XYZ_2"], combvd["DV"]
    h_XYZ1, h_XYZ2, h_DV = he["XYZ_1"], he["XYZ_2"], he["DV"]
    m_XYZ1, m_XYZ2, m_DV = mac["XYZ_1"], mac["XYZ_2"], mac["DV"]
    mu_XYZ1, mu_XYZ2 = munsell_pairs["XYZ_1"], munsell_pairs["XYZ_2"]
    o_XYZ1, o_XYZ2 = osa_pairs["XYZ_1"], osa_pairs["XYZ_2"]

    def objective(x):
        global _eval_count, _best_loss, _last_print_time
        try:
            params = unpack_params(x)
            space = MetricSpace(params, neutral_correction=True)

            DE_c = space.distance(c_XYZ1, c_XYZ2)
            if np.any(~np.isfinite(DE_c)):
                return 100.0
            s_combvd = stress(c_DV, DE_c)

            DE_h = space.distance(h_XYZ1, h_XYZ2)
            if np.any(~np.isfinite(DE_h)):
                return 100.0
            s_he = stress(h_DV, DE_h)

            DE_m = space.distance(m_XYZ1, m_XYZ2)
            if np.any(~np.isfinite(DE_m)):
                return 100.0
            s_mac = stress(m_DV, DE_m)

            DE_mu = space.distance(mu_XYZ1, mu_XYZ2)
            if np.any(~np.isfinite(DE_mu)) or np.mean(DE_mu) < 1e-10:
                return 100.0
            munsell_cv = float(np.std(DE_mu) / np.mean(DE_mu) * 100.0)

            DE_osa = space.distance(o_XYZ1, o_XYZ2)
            if np.any(~np.isfinite(DE_osa)) or np.mean(DE_osa) < 1e-10:
                return 100.0
            osa_cv = float(np.std(DE_osa) / np.mean(DE_osa) * 100.0)

            # Primary: minimize Mac + OSA
            total = s_mac + 3.0 * osa_cv

            # Barrier: COMBVD and He must stay under cap
            if s_combvd > combvd_cap:
                total += 50.0 * (s_combvd - combvd_cap) ** 2
            else:
                total += 0.5 * s_combvd  # light pull toward lower COMBVD

            if s_he > he_cap:
                total += 20.0 * (s_he - he_cap) ** 2
            else:
                total += 0.1 * s_he

            total += 0.01 * munsell_cv

            # RT
            rt_errors = space.round_trip_error(XYZ_rt)
            rt_max = rt_errors.max()
            if rt_max > 1e-6:
                total += 20.0 * np.log10(rt_max / 1e-6)

        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 200.0

        _eval_count += 1
        if total < _best_loss:
            _best_loss = total

        now = time.time()
        if now - _last_print_time > 15.0:
            _last_print_time = now
            print(f"  #{_eval_count:>6d}  loss={total:.3f}  "
                  f"C={s_combvd:.2f} H={s_he:.2f} M={s_mac:.2f} "
                  f"MuCV={munsell_cv:.1f}% OCV={osa_cv:.1f}%  best={_best_loss:.3f}",
                  flush=True)

        return total

    return objective


def make_balanced_objective(combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt,
                            alpha=0.5):
    """Balanced objective: weighted sum where alpha controls COMBVD-vs-Mac/OSA.

    alpha=0: pure Mac+OSA, alpha=1: pure COMBVD+He
    """
    c_XYZ1, c_XYZ2, c_DV = combvd["XYZ_1"], combvd["XYZ_2"], combvd["DV"]
    h_XYZ1, h_XYZ2, h_DV = he["XYZ_1"], he["XYZ_2"], he["DV"]
    m_XYZ1, m_XYZ2, m_DV = mac["XYZ_1"], mac["XYZ_2"], mac["DV"]
    mu_XYZ1, mu_XYZ2 = munsell_pairs["XYZ_1"], munsell_pairs["XYZ_2"]
    o_XYZ1, o_XYZ2 = osa_pairs["XYZ_1"], osa_pairs["XYZ_2"]

    def objective(x):
        global _eval_count, _best_loss, _last_print_time
        try:
            params = unpack_params(x)
            space = MetricSpace(params, neutral_correction=True)

            DE_c = space.distance(c_XYZ1, c_XYZ2)
            if np.any(~np.isfinite(DE_c)):
                return 100.0
            s_combvd = stress(c_DV, DE_c)

            DE_h = space.distance(h_XYZ1, h_XYZ2)
            if np.any(~np.isfinite(DE_h)):
                return 100.0
            s_he = stress(h_DV, DE_h)

            DE_m = space.distance(m_XYZ1, m_XYZ2)
            if np.any(~np.isfinite(DE_m)):
                return 100.0
            s_mac = stress(m_DV, DE_m)

            DE_mu = space.distance(mu_XYZ1, mu_XYZ2)
            if np.any(~np.isfinite(DE_mu)) or np.mean(DE_mu) < 1e-10:
                return 100.0
            munsell_cv = float(np.std(DE_mu) / np.mean(DE_mu) * 100.0)

            DE_osa = space.distance(o_XYZ1, o_XYZ2)
            if np.any(~np.isfinite(DE_osa)) or np.mean(DE_osa) < 1e-10:
                return 100.0
            osa_cv = float(np.std(DE_osa) / np.mean(DE_osa) * 100.0)

            # Normalize to similar scale (STRESS ~20-30, CV ~15-35)
            # COMBVD + He: weight alpha
            combvd_term = s_combvd + 0.05 * s_he
            # Mac + OSA: weight (1-alpha)
            mac_osa_term = s_mac + 2.0 * osa_cv  # OSA CV needs bigger weight since gap is larger

            total = alpha * combvd_term + (1.0 - alpha) * mac_osa_term
            total += 0.01 * munsell_cv

            rt_errors = space.round_trip_error(XYZ_rt)
            rt_max = rt_errors.max()
            if rt_max > 1e-6:
                total += 20.0 * np.log10(rt_max / 1e-6)

        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 200.0

        _eval_count += 1
        if total < _best_loss:
            _best_loss = total

        now = time.time()
        if now - _last_print_time > 15.0:
            _last_print_time = now
            print(f"  #{_eval_count:>6d}  loss={total:.3f}  "
                  f"C={s_combvd:.2f} H={s_he:.2f} M={s_mac:.2f} "
                  f"MuCV={munsell_cv:.1f}% OCV={osa_cv:.1f}%  best={_best_loss:.3f}",
                  flush=True)

        return total

    return objective


def main():
    print("v24b: Targeted multi-objective optimization")
    print("Loading data...")

    combvd = load_combvd()
    he = load_he2022()
    mac = load_macadam1974()
    munsell_data = load_munsell(subset="real")
    munsell_pairs = generate_munsell_pairs(munsell_data)
    osa_data = load_osa_ucs()
    osa_pairs = generate_osa_pairs(osa_data)

    # RT test points
    p0 = MetricParams.load("checkpoints/v23_he_fix.json")
    rng = np.random.default_rng(42)
    _xyz = rng.uniform(0.05, 0.90, (2000, 3))
    _lms = _xyz @ p0.M1.T
    XYZ_rt = _xyz[(_lms >= 0).all(axis=1)][:1000]

    print(f"  Data: COMBVD={len(combvd['DV'])}, He={len(he['DV'])}, Mac={len(mac['DV'])}, "
          f"Munsell={len(munsell_pairs['XYZ_1'])}, OSA={len(osa_pairs['XYZ_1'])}")

    # Starting points
    x_baseline = pack_params(p0)
    m_baseline = full_eval(x_baseline, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt)
    print(f"\nBaseline:")
    print_eval("v23_he_fix", m_baseline)

    # Try to load mac++osa++ from sweep
    x_mac_osa = None
    if os.path.exists("checkpoints/v24_mac++osa++.json"):
        p_mo = MetricParams.load("checkpoints/v24_mac++osa++.json")
        x_mac_osa = pack_params(p_mo)
        m_mo = full_eval(x_mac_osa, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt)
        print_eval("mac++osa++", m_mo)

    best_x = x_baseline.copy()
    best_score = score_v2(m_baseline)

    # ═══════════════════════════════════════════════════════════
    # Strategy 1: Constrained optimization from baseline
    #   Push Mac+OSA hard, but with COMBVD/He ceiling
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"Strategy 1: Constrained (COMBVD≤23.5, He≤31.0)")
    print(f"{'='*70}")

    for cap_c, cap_h, label in [
        (23.50, 31.0, "tight"),
        (23.80, 31.5, "medium"),
        (24.00, 32.0, "loose"),
    ]:
        reset_counters()
        print(f"\n  {label}: caps COMBVD≤{cap_c}, He≤{cap_h}")

        obj = make_constrained_objective(
            combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt,
            combvd_cap=cap_c, he_cap=cap_h,
        )
        bounds = make_bounds(x_baseline, wider=True)

        # Multiple restarts
        for restart in range(3):
            x_init = best_x.copy() if restart == 0 else x_baseline.copy()
            if restart > 0:
                rng_r = np.random.default_rng(42 + restart)
                noise = rng_r.normal(0, 0.005, len(x_init))
                x_init = np.clip(x_init + noise,
                                 [b[0] for b in bounds],
                                 [b[1] for b in bounds])

            t0 = time.time()
            result = minimize(obj, x0=x_init, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-11})
            dt = time.time() - t0

            m = full_eval(result.x, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt)
            print_eval(f"{label} r{restart+1}", m)
            print(f"    Time: {dt:.0f}s")

            if m["rt"] > 1e-6:
                continue

            s = score_v2(m)
            if s < best_score:
                best_x = result.x.copy()
                best_score = s
                print(f"    ** New best (score={s:.4f})")
                unpack_params(best_x).save("checkpoints/v24_best.json")
                with open("checkpoints/v24_best_metrics.json", "w") as f:
                    json.dump(m, f, indent=2)

    # ═══════════════════════════════════════════════════════════
    # Strategy 2: Balanced alpha sweep from mac++osa++ point
    # ═══════════════════════════════════════════════════════════
    if x_mac_osa is not None:
        print(f"\n{'='*70}")
        print(f"Strategy 2: Recovery from mac++osa++ with balanced objective")
        print(f"{'='*70}")

        for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
            reset_counters()
            print(f"\n  alpha={alpha:.1f}")

            obj = make_balanced_objective(
                combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt,
                alpha=alpha,
            )
            bounds = make_bounds(x_mac_osa)

            t0 = time.time()
            result = minimize(obj, x0=x_mac_osa, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-11})
            dt = time.time() - t0

            m = full_eval(result.x, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt)
            print_eval(f"alpha={alpha:.1f}", m)
            print(f"    Time: {dt:.0f}s")

            if m["rt"] > 1e-6:
                continue

            s = score_v2(m)
            if s < best_score:
                best_x = result.x.copy()
                best_score = s
                print(f"    ** New best (score={s:.4f})")
                unpack_params(best_x).save("checkpoints/v24_best.json")
                with open("checkpoints/v24_best_metrics.json", "w") as f:
                    json.dump(m, f, indent=2)

    # ═══════════════════════════════════════════════════════════
    # Strategy 3: Balanced alpha sweep from baseline
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"Strategy 3: Balanced sweep from baseline")
    print(f"{'='*70}")

    for alpha in [0.2, 0.3, 0.4, 0.5]:
        reset_counters()
        print(f"\n  alpha={alpha:.1f}")

        obj = make_balanced_objective(
            combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt,
            alpha=alpha,
        )
        bounds = make_bounds(x_baseline, wider=True)

        t0 = time.time()
        result = minimize(obj, x0=x_baseline, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 8000, "ftol": 1e-13, "gtol": 1e-11})
        dt = time.time() - t0

        m = full_eval(result.x, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt)
        print_eval(f"alpha={alpha:.1f}", m)
        print(f"    Time: {dt:.0f}s")

        if m["rt"] > 1e-6:
            continue

        s = score_v2(m)
        if s < best_score:
            best_x = result.x.copy()
            best_score = s
            print(f"    ** New best (score={s:.4f})")
            unpack_params(best_x).save("checkpoints/v24_best.json")
            with open("checkpoints/v24_best_metrics.json", "w") as f:
                json.dump(m, f, indent=2)

    # ═══════════════════════════════════════════════════════════
    # Strategy 4: Interpolate between baseline and mac++osa++ starting points
    # ═══════════════════════════════════════════════════════════
    if x_mac_osa is not None:
        print(f"\n{'='*70}")
        print(f"Strategy 4: Interpolated starting points")
        print(f"{'='*70}")

        for t in [0.3, 0.5, 0.7]:
            reset_counters()
            x_interp = (1 - t) * x_baseline + t * x_mac_osa
            print(f"\n  t={t:.1f} (interp baseline↔mac++osa++)")

            # Use constrained objective
            obj = make_constrained_objective(
                combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt,
                combvd_cap=23.50, he_cap=31.0,
            )
            bounds = make_bounds(x_interp, wider=True)

            t0 = time.time()
            result = minimize(obj, x0=x_interp, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-11})
            dt = time.time() - t0

            m = full_eval(result.x, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt)
            print_eval(f"t={t:.1f}", m)
            print(f"    Time: {dt:.0f}s")

            if m["rt"] > 1e-6:
                continue

            s = score_v2(m)
            if s < best_score:
                best_x = result.x.copy()
                best_score = s
                print(f"    ** New best (score={s:.4f})")
                unpack_params(best_x).save("checkpoints/v24_best.json")
                with open("checkpoints/v24_best_metrics.json", "w") as f:
                    json.dump(m, f, indent=2)

    # ═══════════════════════════════════════════════════════════
    # Strategy 5: CMA-ES from best so far
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"Strategy 5: CMA-ES from best result")
    print(f"{'='*70}")

    try:
        import cma

        obj_cma = make_constrained_objective(
            combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt,
            combvd_cap=23.50, he_cap=31.0,
        )
        bounds_cma = make_bounds(best_x, wider=True)
        lb = [b[0] for b in bounds_cma]
        ub = [b[1] for b in bounds_cma]

        # Fix dist_linear at 0
        lb[68] = 0.0
        ub[68] = 0.0

        opts = cma.CMAOptions()
        opts.set("bounds", [lb, ub])
        opts.set("maxfevals", 30000)
        opts.set("timeout", 600)  # 10 min max
        opts.set("tolx", 1e-12)
        opts.set("verbose", -1)

        sigma0 = 0.01
        es = cma.CMAEvolutionStrategy(best_x, sigma0, opts)

        gen_count = 0
        while not es.stop():
            solutions = es.ask()
            fitnesses = [obj_cma(s) for s in solutions]
            es.tell(solutions, fitnesses)
            gen_count += 1

            if gen_count % 10 == 0:
                m_cma = full_eval(es.best.x, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt)
                print(f"  CMA-ES gen {gen_count}: ", end="")
                print_eval("best", m_cma)

                if m_cma["rt"] <= 1e-6:
                    s_cma = score_v2(m_cma)
                    if s_cma < best_score:
                        best_x = es.best.x.copy()
                        best_score = s_cma
                        print(f"    ** New best CMA (score={s_cma:.4f})")
                        unpack_params(best_x).save("checkpoints/v24_best.json")
                        with open("checkpoints/v24_best_metrics.json", "w") as f:
                            json.dump(m_cma, f, indent=2)

        m_cma_final = full_eval(es.best.x, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt)
        print_eval("CMA final", m_cma_final)
        if m_cma_final["rt"] <= 1e-6:
            s_cma = score_v2(m_cma_final)
            if s_cma < best_score:
                best_x = es.best.x.copy()
                best_score = s_cma
                unpack_params(best_x).save("checkpoints/v24_best.json")
                with open("checkpoints/v24_best_metrics.json", "w") as f:
                    json.dump(m_cma_final, f, indent=2)

    except ImportError:
        print("  CMA-ES not available (pip install cma)")

    # ═══════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════
    m_final = full_eval(best_x, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt)
    print(f"\n{'='*70}")
    print(f"v24b FINAL RESULTS")
    print(f"{'='*70}")
    print_eval("BEST", m_final)
    print(f"  Score: {score_v2(m_final):.4f}")
    print(f"\n  vs targets:")
    targets = {"combvd": 23.30, "he": 30.33, "mac": 18.71, "munsell_cv": 40.0, "osa_cv": 16.6}
    for key, target in targets.items():
        val = m_final[key]
        delta = val - target
        status = "PASS" if val <= target else f"FAIL ({delta:+.2f})"
        print(f"    {key:>12s}: {val:8.2f}  target={target:.2f}  {status}")

    print(f"\n  Saved to: checkpoints/v24_best.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
