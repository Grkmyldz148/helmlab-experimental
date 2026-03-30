#!/usr/bin/env python
"""v24 optimization: Multi-objective — COMBVD + He + MacAdam + OSA-UCS + Munsell.

Goal: Make MetricSpace #1 on ALL metrics simultaneously.

Current v23_he_fix metrics (with NC=True):
  - COMBVD: 23.24 (#1)
  - He: 30.12 (#1)
  - MacAdam: 20.65 (#2, CAM16-UCS=18.71)
  - Munsell CV: 35.4% (#1)
  - OSA-UCS CV: 21.5% (#4-5, IPT=16.6%)

Targets:
  - COMBVD ≤ 23.30
  - He ≤ 30.33
  - MacAdam ≤ 18.71
  - Munsell CV ≤ 40%
  - OSA-UCS CV ≤ 16.6%
  - RT < 1e-6
"""

import sys
import os
import json
import time
import argparse

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
        p.M1.ravel(), p.gamma, p.M2.ravel(),                     # 0-20
        [p.hk_weight], [p.hk_power], [p.hk_hue_mod],            # 21-23
        [p.L_corr_p1], [p.L_corr_p2], [p.L_corr_p3],            # 24-26
        [p.cs_cos1], [p.cs_sin1], [p.cs_cos2], [p.cs_sin2],      # 27-30
        [p.cs_cos3], [p.cs_sin3],                                 # 31-32
        [p.lc1], [p.lc2],                                         # 33-34
        [p.hk_sin1], [p.hk_cos2], [p.hk_sin2],                   # 35-37
        [p.hue_cos1], [p.hue_sin1], [p.hue_cos2], [p.hue_sin2],  # 38-41
        [p.hue_cos3], [p.hue_sin3],                               # 42-43
        [p.hlc_cos1], [p.hlc_sin1], [p.hlc_cos2], [p.hlc_sin2],  # 44-47
        [p.hl_cos1], [p.hl_sin1], [p.hl_cos2], [p.hl_sin2],      # 48-51
        [p.cp_cos1], [p.cp_sin1], [p.cp_cos2], [p.cp_sin2],      # 52-55
        [p.lp_dark],                                               # 56
        [p.dist_power], [p.dist_wC],                               # 57-58
        [p.hue_cos4], [p.hue_sin4],                                # 59-60
        [p.Lh_cos1], [p.Lh_sin1],                                  # 61-62
        [p.cs_cos4], [p.cs_sin4],                                   # 63-64
        [p.dist_compress],                                          # 65
        [p.lp_dark_hcos], [p.lp_dark_hsin],                        # 66-67
        [p.dist_linear],                                            # 68
        [p.dist_post_power],                                        # 69
        [p.dist_sl],                                                # 70
        [p.dist_sc],                                                # 71
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
    """Create parameter bounds."""
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
    """Full evaluation with NC=True."""
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
        munsell_cv = float(np.std(DE_mu) / np.mean(DE_mu) * 100.0) if np.mean(DE_mu) > 1e-10 else 200.0

        DE_osa = s.distance(osa_pairs["XYZ_1"], osa_pairs["XYZ_2"])
        osa_cv = float(np.std(DE_osa) / np.mean(DE_osa) * 100.0) if np.mean(DE_osa) > 1e-10 else 200.0

        rt_errors = s.round_trip_error(XYZ_rt)
        rt = float(rt_errors.max())

        return {
            "combvd": s_combvd, "he": s_he, "mac": s_mac,
            "munsell_cv": munsell_cv, "osa_cv": osa_cv, "rt": rt,
        }
    except Exception as e:
        return {
            "combvd": 100.0, "he": 100.0, "mac": 100.0,
            "munsell_cv": 200.0, "osa_cv": 200.0, "rt": 1.0,
            "error": str(e),
        }


def print_eval(label, m):
    beat_combvd = "Y" if m["combvd"] <= 23.30 else "N"
    beat_he = "Y" if m["he"] <= 30.33 else "N"
    beat_mac = "Y" if m["mac"] <= 18.71 else "N"
    beat_mun = "Y" if m["munsell_cv"] <= 40.0 else "N"
    beat_osa = "Y" if m["osa_cv"] <= 16.6 else "N"
    flags = f"[{beat_combvd}{beat_he}{beat_mac}{beat_mun}{beat_osa}]"
    print(f"  {label}: COMBVD={m['combvd']:.2f}  He={m['he']:.2f}  Mac={m['mac']:.2f}  "
          f"MunCV={m['munsell_cv']:.1f}%  OsaCV={m['osa_cv']:.1f}%  RT={m['rt']:.2e}  {flags}")


def score_metrics(m):
    """Score: sum of relative excess over targets. 0 = all targets met."""
    targets = {"combvd": 23.30, "he": 30.33, "mac": 18.71, "munsell_cv": 40.0, "osa_cv": 16.6}
    penalty = 0.0
    for key, target in targets.items():
        val = m[key]
        if val > target:
            penalty += (val - target) / target
    return penalty


# ── Objective factory ─────────────────────────────────────────────────

_eval_count = 0
_best_loss = float("inf")
_last_print_time = 0.0
_best_metrics = None


def reset_counters():
    global _eval_count, _best_loss, _last_print_time, _best_metrics
    _eval_count = 0
    _best_loss = float("inf")
    _last_print_time = 0.0
    _best_metrics = None


def make_multi_objective(combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt,
                          mac_lambda=0.5, osa_lambda=0.05, he_lambda=0.05,
                          munsell_lambda=0.02, rt_penalty=20.0,
                          max_sub_lambda=0.0):
    """Multi-objective with NC=True for every evaluation."""
    c_XYZ1, c_XYZ2, c_DV = combvd["XYZ_1"], combvd["XYZ_2"], combvd["DV"]
    h_XYZ1, h_XYZ2, h_DV = he["XYZ_1"], he["XYZ_2"], he["DV"]
    m_XYZ1, m_XYZ2, m_DV = mac["XYZ_1"], mac["XYZ_2"], mac["DV"]
    mu_XYZ1, mu_XYZ2 = munsell_pairs["XYZ_1"], munsell_pairs["XYZ_2"]
    o_XYZ1, o_XYZ2 = osa_pairs["XYZ_1"], osa_pairs["XYZ_2"]

    # Sub-dataset masks
    sub_masks = {}
    if max_sub_lambda > 0 and "dataset" in combvd:
        datasets = combvd["dataset"]
        for ds in sorted(set(datasets)):
            mask = np.array([d == ds for d in datasets])
            if np.sum(mask) >= 2:
                sub_masks[ds] = mask

    def objective(x):
        global _eval_count, _best_loss, _last_print_time, _best_metrics
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

            total = s_combvd
            total += he_lambda * s_he
            total += mac_lambda * s_mac
            total += munsell_lambda * munsell_cv
            total += osa_lambda * osa_cv

            if sub_masks and max_sub_lambda > 0:
                sub_stresses = []
                for ds, mask in sub_masks.items():
                    sub_DE = space.distance(combvd["XYZ_1"][mask], combvd["XYZ_2"][mask])
                    sub_stresses.append(stress(combvd["DV"][mask], sub_DE))
                total += max_sub_lambda * max(sub_stresses)

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
            _best_metrics = {
                "combvd": s_combvd, "he": s_he, "mac": s_mac,
                "munsell_cv": munsell_cv, "osa_cv": osa_cv,
            }

        now = time.time()
        if now - _last_print_time > 15.0:
            _last_print_time = now
            print(f"  eval #{_eval_count:>6d}  loss={total:.4f}  "
                  f"C={s_combvd:.2f} H={s_he:.2f} M={s_mac:.2f} "
                  f"MuCV={munsell_cv:.1f}% OCV={osa_cv:.1f}%  best={_best_loss:.4f}",
                  flush=True)

        return total

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=str, default="checkpoints/v23_he_fix.json")
    parser.add_argument("--phase", type=str, default="sweep",
                        choices=["diagnose", "sweep", "optimize", "refine", "cmaes", "all"])
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--restarts", type=int, default=3)
    args = parser.parse_args()

    print(f"v24: Multi-objective optimization — beat ALL competitors")
    print(f"  Init: {args.init}")
    print(f"  Phase: {args.phase}")
    print(f"Loading data...")

    combvd = load_combvd()
    he = load_he2022()
    mac = load_macadam1974()
    munsell_data = load_munsell(subset="real")
    munsell_pairs = generate_munsell_pairs(munsell_data)
    osa_data = load_osa_ucs()
    osa_pairs = generate_osa_pairs(osa_data)

    print(f"  COMBVD: {len(combvd['DV'])} pairs")
    print(f"  He: {len(he['DV'])} pairs")
    print(f"  MacAdam: {len(mac['DV'])} pairs")
    print(f"  Munsell: {len(munsell_pairs['XYZ_1'])} pairs")
    print(f"  OSA-UCS: {len(osa_pairs['XYZ_1'])} pairs")

    params = MetricParams.load(args.init)
    x0 = pack_params(params)
    assert len(x0) == N_PARAMS

    # RT test points
    rng = np.random.default_rng(42)
    _xyz = rng.uniform(0.05, 0.90, (2000, 3))
    _lms = _xyz @ params.M1.T
    XYZ_rt = _xyz[(_lms >= 0).all(axis=1)][:1000]

    m0 = full_eval(x0, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt)
    print(f"\nInitial state:")
    print_eval("v23_he_fix", m0)

    # ═══════════════════════════════════════════════════════════
    # DIAGNOSE
    # ═══════════════════════════════════════════════════════════
    if args.phase in ("diagnose", "all"):
        print(f"\n{'='*70}")
        print(f"DIAGNOSTIC ANALYSIS")
        print(f"{'='*70}")

        space = MetricSpace(params, neutral_correction=True)

        # MacAdam per-hue analysis
        DE_mac = space.distance(mac["XYZ_1"], mac["XYZ_2"])
        DV_mac = mac["DV"]
        Lab_mac1 = space.from_XYZ(mac["XYZ_1"])
        Lab_mac2 = space.from_XYZ(mac["XYZ_2"])
        h_mac_avg = np.rad2deg(np.arctan2(
            (Lab_mac1[:, 2] + Lab_mac2[:, 2]) / 2,
            (Lab_mac1[:, 1] + Lab_mac2[:, 1]) / 2
        )) % 360

        print(f"\n  MacAdam per-hue STRESS:")
        for lo, hi, name in [(0, 60, "Red-Yel"), (60, 120, "Yel-Grn"),
                              (120, 180, "Grn-Cyn"), (180, 240, "Cyn-Blu"),
                              (240, 300, "Blu-Mag"), (300, 360, "Mag-Red")]:
            mask = (h_mac_avg >= lo) & (h_mac_avg < hi)
            n = np.sum(mask)
            if n >= 2:
                print(f"    {name:8s} [{lo:3d},{hi:3d}) n={n:3d} STRESS={stress(DV_mac[mask], DE_mac[mask]):.2f}")

        # OSA per-type + per lightness
        DE_osa = space.distance(osa_pairs["XYZ_1"], osa_pairs["XYZ_2"])
        print(f"\n  OSA-UCS per-type CV:")
        for ptype in ["L", "j", "g"]:
            mask = np.array([t == ptype for t in osa_pairs["pair_type"]])
            de_t = DE_osa[mask]
            print(f"    {ptype}: n={np.sum(mask):3d}  CV={100*np.std(de_t)/np.mean(de_t):.1f}%  "
                  f"mean={np.mean(de_t):.4f}  min={np.min(de_t):.4f}  max={np.max(de_t):.4f}")

    # ═══════════════════════════════════════════════════════════
    # SWEEP — Pareto grid search
    # ═══════════════════════════════════════════════════════════
    if args.phase in ("sweep", "all"):
        print(f"\n{'='*70}")
        print(f"PARETO SWEEP")
        print(f"{'='*70}")

        bounds = make_bounds(x0)
        results = []

        # Key sweep: vary mac_lambda and osa_lambda
        configs = [
            # (mac_λ, osa_λ, he_λ, mun_λ, label)
            (0.3, 0.03, 0.05, 0.02, "base"),
            (0.5, 0.05, 0.05, 0.02, "mid"),
            (0.5, 0.10, 0.05, 0.02, "osa+"),
            (0.8, 0.05, 0.05, 0.02, "mac+"),
            (0.8, 0.10, 0.05, 0.02, "mac+osa+"),
            (1.0, 0.10, 0.03, 0.01, "mac++osa+"),
            (1.0, 0.20, 0.03, 0.01, "mac++osa++"),
            (0.5, 0.20, 0.05, 0.02, "osa++"),
            (1.5, 0.15, 0.03, 0.01, "mac+++osa+"),
            (0.8, 0.30, 0.03, 0.01, "mac+osa+++"),
        ]

        for ml, ol, hl, mul, label in configs:
            reset_counters()
            print(f"\n  Config: {label} (mac={ml}, osa={ol}, he={hl}, mun={mul})")

            obj = make_multi_objective(
                combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt,
                mac_lambda=ml, osa_lambda=ol,
                he_lambda=hl, munsell_lambda=mul,
            )

            t0 = time.time()
            result = minimize(obj, x0=x0, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": args.maxiter, "ftol": 1e-13, "gtol": 1e-11})
            dt = time.time() - t0

            m = full_eval(result.x, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt)
            print_eval(label, m)
            print(f"    Time: {dt:.0f}s, nfev: {result.nfev}")

            results.append({
                "label": label, "ml": ml, "ol": ol, "x": result.x.copy(),
                "metrics": m, "dt": dt,
            })

            # Save if it's best so far
            s = score_metrics(m)
            if m["rt"] <= 1e-6:
                p_save = unpack_params(result.x)
                p_save.save(f"checkpoints/v24_{label}.json")

        # Summary
        print(f"\n{'='*70}")
        print(f"SWEEP SUMMARY")
        print(f"{'='*70}")
        print(f"{'Label':>15s} {'COMBVD':>8s} {'He':>6s} {'Mac':>8s} {'MunCV':>7s} {'OsaCV':>7s} {'RT':>10s} {'Score':>7s}")
        for r in results:
            m = r["metrics"]
            s = score_metrics(m)
            print(f"{r['label']:>15s} {m['combvd']:8.2f} {m['he']:6.2f} {m['mac']:8.2f} "
                  f"{m['munsell_cv']:6.1f}% {m['osa_cv']:6.1f}% {m['rt']:.2e} {s:.4f}")

        # Find best
        valid = [r for r in results if r["metrics"]["rt"] <= 1e-6]
        if valid:
            best = min(valid, key=lambda r: score_metrics(r["metrics"]))
            print(f"\n  Best: {best['label']}")
            print_eval("Best", best["metrics"])
            p_best = unpack_params(best["x"])
            p_best.save("checkpoints/v24_sweep_best.json")
            with open("checkpoints/v24_sweep_best_metrics.json", "w") as f:
                json.dump(best["metrics"], f, indent=2)
        else:
            print("  No valid results (all RT broken)")
            best = results[0]

        x_sweep_best = best["x"]

    # ═══════════════════════════════════════════════════════════
    # OPTIMIZE — Multi-restart from best sweep
    # ═══════════════════════════════════════════════════════════
    if args.phase in ("optimize", "all"):
        print(f"\n{'='*70}")
        print(f"MULTI-RESTART OPTIMIZATION")
        print(f"{'='*70}")

        if args.phase == "all":
            x_start = x_sweep_best
        elif os.path.exists("checkpoints/v24_sweep_best.json"):
            x_start = pack_params(MetricParams.load("checkpoints/v24_sweep_best.json"))
        else:
            x_start = x0

        bounds = make_bounds(x_start)
        best_x = x_start.copy()
        best_score = score_metrics(full_eval(x_start, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt))

        configs_opt = [
            (0.8, 0.10, 0.05, 0.02, "balanced"),
            (1.0, 0.15, 0.03, 0.01, "mac+osa"),
            (1.2, 0.20, 0.03, 0.01, "aggressive"),
            (0.6, 0.25, 0.05, 0.02, "osa-heavy"),
            (1.5, 0.10, 0.03, 0.01, "mac-heavy"),
        ]

        for ml, ol, hl, mul, label in configs_opt:
            for restart in range(args.restarts):
                reset_counters()
                print(f"\n  {label} r{restart+1}/{args.restarts}")

                obj = make_multi_objective(
                    combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt,
                    mac_lambda=ml, osa_lambda=ol,
                    he_lambda=hl, munsell_lambda=mul,
                )

                x_init = best_x.copy()
                if restart > 0:
                    rng_r = np.random.default_rng(42 + restart)
                    noise = rng_r.normal(0, 0.002, len(x_init))
                    x_init = np.clip(x_init + noise,
                                     [b[0] for b in bounds],
                                     [b[1] for b in bounds])

                t0 = time.time()
                result = minimize(obj, x0=x_init, method="L-BFGS-B", bounds=bounds,
                                  options={"maxiter": args.maxiter, "ftol": 1e-13, "gtol": 1e-11})
                dt = time.time() - t0

                m = full_eval(result.x, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt)
                print_eval(f"{label} r{restart+1}", m)
                print(f"    Time: {dt:.0f}s")

                if m["rt"] > 1e-6:
                    print(f"    RT broken, skip")
                    continue

                s = score_metrics(m)
                if s < best_score:
                    best_x = result.x.copy()
                    best_score = s
                    print(f"    ** New best (score={s:.4f})")
                    p_best = unpack_params(best_x)
                    p_best.save("checkpoints/v24_best.json")
                    with open("checkpoints/v24_best_metrics.json", "w") as f:
                        json.dump(m, f, indent=2)

        m_final = full_eval(best_x, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt)
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION RESULT")
        print(f"{'='*70}")
        print_eval("Final", m_final)

    # ═══════════════════════════════════════════════════════════
    # REFINE — Wider bounds from best
    # ═══════════════════════════════════════════════════════════
    if args.phase in ("refine", "all"):
        print(f"\n{'='*70}")
        print(f"WIDER-BOUND REFINEMENT")
        print(f"{'='*70}")

        if os.path.exists("checkpoints/v24_best.json"):
            x_ref = pack_params(MetricParams.load("checkpoints/v24_best.json"))
        elif args.phase == "all":
            x_ref = best_x
        else:
            x_ref = x0

        bounds_w = make_bounds(x_ref, wider=True)
        best_x_r = x_ref.copy()
        best_score_r = score_metrics(full_eval(x_ref, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt))

        configs_ref = [
            (1.0, 0.15, 0.03, 0.01, "wide-balanced"),
            (1.5, 0.20, 0.03, 0.01, "wide-aggressive"),
            (0.8, 0.30, 0.03, 0.01, "wide-osa"),
        ]

        for ml, ol, hl, mul, label in configs_ref:
            for restart in range(args.restarts):
                reset_counters()
                print(f"\n  {label} r{restart+1}")

                obj = make_multi_objective(
                    combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt,
                    mac_lambda=ml, osa_lambda=ol,
                    he_lambda=hl, munsell_lambda=mul,
                )

                x_init = best_x_r.copy()
                if restart > 0:
                    rng_r = np.random.default_rng(200 + restart)
                    noise = rng_r.normal(0, 0.003, len(x_init))
                    x_init = np.clip(x_init + noise,
                                     [b[0] for b in bounds_w],
                                     [b[1] for b in bounds_w])

                t0 = time.time()
                result = minimize(obj, x0=x_init, method="L-BFGS-B", bounds=bounds_w,
                                  options={"maxiter": args.maxiter, "ftol": 1e-13, "gtol": 1e-11})
                dt = time.time() - t0

                m = full_eval(result.x, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt)
                print_eval(f"{label} r{restart+1}", m)
                print(f"    Time: {dt:.0f}s")

                if m["rt"] > 1e-6:
                    continue

                s = score_metrics(m)
                if s < best_score_r:
                    best_x_r = result.x.copy()
                    best_score_r = s
                    print(f"    ** New best refine (score={s:.4f})")
                    p_best = unpack_params(best_x_r)
                    p_best.save("checkpoints/v24_refine_best.json")
                    with open("checkpoints/v24_refine_metrics.json", "w") as f:
                        json.dump(m, f, indent=2)

        m_ref = full_eval(best_x_r, combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt)
        print(f"\n{'='*70}")
        print(f"REFINEMENT RESULT")
        print(f"{'='*70}")
        print_eval("Refined", m_ref)

    print(f"\nDONE")


if __name__ == "__main__":
    main()
