#!/usr/bin/env python
"""v23: Comprehensive multi-dataset optimization with CMA-ES + L-BFGS-B hybrid.

Goals:
  - Beat v22 on ALL metrics without regression
  - Close the OSA-UCS gap (currently 4th place)
  - Improve MacAdam (currently 2nd to CAM16-UCS)
  - Maintain COMBVD leadership

Strategy:
  Phase 1: CMA-ES global search from v20b with ALL datasets in objective
  Phase 2: L-BFGS-B polish from best CMA-ES point
  Phase 3: Activate hue-dependent SL/SC (8 extra DOF) and re-optimize
  Phase 4: Pareto analysis — find models that dominate v22 on ALL metrics

Param layout: same 72 as v14/v22
"""

import argparse, json, time, sys, os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helmlab.data.combvd import load_combvd
from helmlab.data.he2022 import load_he2022
from helmlab.data.macadam1974 import load_macadam1974
from helmlab.data.munsell import load_munsell, generate_munsell_pairs
from helmlab.data.osa_ucs import load_osa_ucs, generate_osa_pairs
from helmlab.metrics.stress import stress
from helmlab.spaces.metric import MetricSpace, MetricParams

from optimize_v14 import (
    pack_params, unpack_params, make_bounds, N_PARAMS,
    split_combvd, compute_bias_r, compute_munsell_cv,
    compute_subdataset_stress,
)


def evaluate_full(x, combvd, he, mac, m_pairs, o_pairs, train=None, val=None):
    """Full evaluation across all datasets."""
    p = unpack_params(x)
    s = MetricSpace(p, neutral_correction=True, ab_rotate_deg=-28.2)

    DE_c = s.distance(combvd["XYZ_1"], combvd["XYZ_2"])
    s_c = stress(combvd["DV"], DE_c)

    DE_h = s.distance(he["XYZ_1"], he["XYZ_2"])
    s_h = stress(he["DV"], DE_h)

    DE_m = s.distance(mac["XYZ_1"], mac["XYZ_2"])
    s_m = stress(mac["DV"], DE_m)

    DE_mu = s.distance(m_pairs["XYZ_1"], m_pairs["XYZ_2"])
    mu_cv = 100 * np.std(DE_mu) / np.mean(DE_mu) if np.mean(DE_mu) > 1e-10 else 200.0

    DE_o = s.distance(o_pairs["XYZ_1"], o_pairs["XYZ_2"])
    o_cv = 100 * np.std(DE_o) / np.mean(DE_o) if np.mean(DE_o) > 1e-10 else 200.0

    # Euclidean versions (space quality)
    Lab1 = s.from_XYZ(combvd["XYZ_1"])
    Lab2 = s.from_XYZ(combvd["XYZ_2"])
    DE_euclid = np.sqrt(np.sum((Lab1 - Lab2)**2, axis=-1))
    s_euclid = stress(combvd["DV"], DE_euclid)

    # RT
    rng = np.random.default_rng(42)
    xyz = rng.uniform(0.05, 0.90, (2000, 3))
    lms = xyz @ p.M1.T
    xyz_valid = xyz[(lms >= 0).all(axis=1)][:1000]
    rt = s.round_trip_error(xyz_valid)

    # Sub-datasets
    ds_arr = np.array(combvd["dataset"])
    sub_stresses = []
    for sub in sorted(set(combvd["dataset"])):
        mask = ds_arr == sub
        sub_stresses.append(stress(combvd["DV"][mask], DE_c[mask]))

    result = {
        "combvd": s_c, "he": s_h, "mac": s_m,
        "munsell_cv": mu_cv, "osa_cv": o_cv,
        "euclid": s_euclid, "rt": rt.max(),
        "sub_max": max(sub_stresses), "sub_mean": np.mean(sub_stresses),
    }

    if train is not None and val is not None:
        DE_t = s.distance(train["XYZ_1"], train["XYZ_2"])
        DE_v = s.distance(val["XYZ_1"], val["XYZ_2"])
        result["train"] = stress(train["DV"], DE_t)
        result["val"] = stress(val["DV"], DE_v)
        result["gap"] = result["val"] - result["train"]

    return result


def print_eval(label, m, ref=None):
    """Print evaluation, optionally with comparison to reference."""
    items = [
        f"COMBVD={m['combvd']:.2f}",
        f"He={m['he']:.2f}",
        f"Mac={m['mac']:.2f}",
        f"Munsell={m['munsell_cv']:.1f}%",
        f"OSA={m['osa_cv']:.1f}%",
        f"Euclid={m['euclid']:.2f}",
        f"RT={m['rt']:.2e}",
    ]
    if "gap" in m:
        items.append(f"Gap={m['gap']:+.2f}")
    print(f"  {label}: {', '.join(items)}")

    if ref:
        diffs = []
        for k in ["combvd", "he", "mac", "munsell_cv", "osa_cv"]:
            d = m[k] - ref[k]
            diffs.append(f"{k}={d:+.2f}")
        print(f"    vs ref: {', '.join(diffs)}")


def dominates_v22(m, v22_metrics):
    """Check if m dominates v22 on all metrics (lower is better)."""
    return (
        m["combvd"] <= v22_metrics["combvd"] + 0.01 and
        m["he"] <= v22_metrics["he"] + 0.01 and
        m["mac"] <= v22_metrics["mac"] + 0.01 and
        m["munsell_cv"] <= v22_metrics["munsell_cv"] + 0.01 and
        m["osa_cv"] <= v22_metrics["osa_cv"] + 0.01 and
        m["rt"] < 1e-6
    )


def make_objective_all(combvd, he, mac, m_pairs, o_pairs,
                       w_combvd=1.0, w_he=0.05, w_mac=0.15,
                       w_munsell=0.015, w_osa=0.015,
                       w_worst_sub=0.03, x0_ref=None):
    """Multi-dataset objective including OSA-UCS."""
    # RT test points
    rng = np.random.default_rng(42)
    _xyz = rng.uniform(0.05, 0.90, (2000, 3))
    if x0_ref is not None:
        _p0 = unpack_params(x0_ref) if not isinstance(x0_ref, MetricParams) else x0_ref
    else:
        _p0 = MetricParams.load("src/helmlab/data/metric_params.json")
    _lms = _xyz @ _p0.M1.T
    XYZ_rt = _xyz[(_lms >= 0).all(axis=1)][:1000]

    ds_arr = np.array(combvd["dataset"])
    sub_masks = {ds: ds_arr == ds for ds in sorted(set(combvd["dataset"]))}

    m_X1, m_X2 = m_pairs["XYZ_1"], m_pairs["XYZ_2"]
    o_X1, o_X2 = o_pairs["XYZ_1"], o_pairs["XYZ_2"]

    _eval_count = [0]
    _best_loss = [float("inf")]
    _last_print = [0.0]

    def objective(x):
        try:
            params = unpack_params(x)
            space = MetricSpace(params, neutral_correction=True, ab_rotate_deg=-28.2)

            # COMBVD
            DE_c = space.distance(combvd["XYZ_1"], combvd["XYZ_2"])
            if np.any(~np.isfinite(DE_c)):
                return 200.0
            s_full = stress(combvd["DV"], DE_c)

            # Sub-dataset balanced
            sub_s = [stress(combvd["DV"][m], DE_c[m]) for m in sub_masks.values()]
            s_mean_sub = float(np.mean(sub_s))
            s_max_sub = float(np.max(sub_s))

            # He2022
            DE_h = space.distance(he["XYZ_1"], he["XYZ_2"])
            if np.any(~np.isfinite(DE_h)):
                return 200.0
            s_he = stress(he["DV"], DE_h)

            # MacAdam
            DE_m = space.distance(mac["XYZ_1"], mac["XYZ_2"])
            if np.any(~np.isfinite(DE_m)):
                return 200.0
            s_mac = stress(mac["DV"], DE_m)

            # Munsell CV
            DE_mu = space.distance(m_X1, m_X2)
            if np.any(~np.isfinite(DE_mu)) or np.mean(DE_mu) < 1e-10:
                return 200.0
            munsell_cv = float(np.std(DE_mu) / np.mean(DE_mu) * 100.0)

            # OSA-UCS CV
            DE_o = space.distance(o_X1, o_X2)
            if np.any(~np.isfinite(DE_o)) or np.mean(DE_o) < 1e-10:
                return 200.0
            osa_cv = float(np.std(DE_o) / np.mean(DE_o) * 100.0)

            # Combined objective
            total = (w_combvd * (0.5 * s_full + 0.5 * s_mean_sub)
                     + w_he * s_he
                     + w_mac * s_mac
                     + w_munsell * munsell_cv
                     + w_osa * osa_cv
                     + w_worst_sub * s_max_sub)

            # RT penalty
            rt = space.round_trip_error(XYZ_rt).max()
            if rt > 1e-6:
                total += 20.0 * np.log10(rt / 1e-6)

        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 200.0

        _eval_count[0] += 1
        if total < _best_loss[0]:
            _best_loss[0] = total

        now = time.time()
        if now - _last_print[0] > 15.0:
            _last_print[0] = now
            print(f"  #{_eval_count[0]:>6d}  loss={total:.4f}  "
                  f"C={s_full:.2f}  Mac={s_mac:.2f}  He={s_he:.2f}  "
                  f"Mu={munsell_cv:.1f}%  OSA={osa_cv:.1f}%  "
                  f"best={_best_loss[0]:.4f}", flush=True)

        return total

    return objective


def make_bounds_v23(x0):
    """Bounds for v23: wider M1/M2, activate hue-dep SL/SC."""
    bounds = make_bounds(x0)

    # Wider M1/M2 bounds (allow more exploration)
    for i in range(9):  # M1
        center = x0[i]
        half = max(abs(center) * 1.5, 0.8)
        bounds[i] = (max(center - half, -4.0), min(center + half, 4.0))
    for i in range(12, 21):  # M2
        center = x0[i]
        half = max(abs(center) * 1.5, 0.8)
        bounds[i] = (max(center - half, -4.0), min(center + half, 4.0))

    # Wider Lh bounds
    bounds[61] = (-0.3, 0.3)  # Lh_cos1
    bounds[62] = (-0.3, 0.3)  # Lh_sin1

    # dist_sl/sc — wider
    bounds[70] = (-5.0, 10.0)   # dist_sl
    bounds[71] = (-1.0, 3.0)    # dist_sc

    return bounds


def run_cmaes(objective, x0, bounds, sigma=0.02, max_evals=50000, seed=42):
    """Run CMA-ES optimization."""
    try:
        import cma
    except ImportError:
        print("  CMA-ES not available, falling back to L-BFGS-B")
        return run_lbfgsb(objective, x0, bounds, maxiter=5000)

    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    # Fix fixed params (lb == ub)
    fixed = lb == ub
    free_idx = np.where(~fixed)[0]

    x0_free = x0[free_idx]
    lb_free = lb[free_idx]
    ub_free = ub[free_idx]

    def wrapped(x_free):
        x_full = x0.copy()
        x_full[free_idx] = x_free
        return objective(x_full)

    opts = cma.CMAOptions()
    opts['bounds'] = [lb_free.tolist(), ub_free.tolist()]
    opts['maxfevals'] = max_evals
    opts['seed'] = seed
    opts['verbose'] = -1
    opts['tolfun'] = 1e-12

    es = cma.CMAEvolutionStrategy(x0_free, sigma, opts)
    es.optimize(wrapped)

    x_best = x0.copy()
    x_best[free_idx] = es.result.xbest
    return x_best, es.result.fbest


def run_lbfgsb(objective, x0, bounds, maxiter=5000):
    """Run L-BFGS-B optimization."""
    result = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-11})
    return result.x, result.fun


def save_checkpoint(x, path, metrics=None):
    """Save parameters and optionally metrics."""
    p = unpack_params(x)
    p.save(path)
    if metrics:
        meta_path = str(path).replace('.json', '_metrics.json')
        with open(meta_path, 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=0, help="Phase to run (0=all, 1=LBFGSB, 2=CMA-ES, 3=hue-dep SL/SC)")
    parser.add_argument("--init", default="src/helmlab/data/metric_params.json")
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=8000)
    parser.add_argument("--w-mac", type=float, default=0.15)
    parser.add_argument("--w-osa", type=float, default=0.015)
    parser.add_argument("--w-munsell", type=float, default=0.015)
    parser.add_argument("--w-he", type=float, default=0.05)
    parser.add_argument("--cma-sigma", type=float, default=0.02)
    parser.add_argument("--cma-evals", type=int, default=60000)
    args = parser.parse_args()

    print("v23: Comprehensive multi-dataset optimization")
    print(f"  Weights: COMBVD=1.0, He={args.w_he}, Mac={args.w_mac}, "
          f"Munsell={args.w_munsell}, OSA={args.w_osa}")

    # Load ALL data
    print("Loading data...")
    combvd = load_combvd()
    he = load_he2022()
    mac = load_macadam1974()
    munsell = load_munsell("real")
    m_pairs = generate_munsell_pairs(munsell)
    osa = load_osa_ucs()
    o_pairs = generate_osa_pairs(osa)
    train, val = split_combvd(combvd, seed=42, val_split=0.2)

    print(f"  COMBVD={len(combvd['DV'])} He={len(he['DV'])} Mac={len(mac['DV'])} "
          f"Munsell={len(m_pairs['XYZ_1'])} OSA={len(o_pairs['XYZ_1'])} pairs")

    # Load initial params and evaluate
    p_init = MetricParams.load(args.init)
    x0 = pack_params(p_init)

    # Also load v22 for comparison
    p22 = MetricParams.load("checkpoints/v22_best.json")
    x22 = pack_params(p22)

    m_init = evaluate_full(x0, combvd, he, mac, m_pairs, o_pairs, train, val)
    m_v22 = evaluate_full(x22, combvd, he, mac, m_pairs, o_pairs, train, val)

    print("\nBaselines:")
    print_eval("v20b (init)", m_init)
    print_eval("v22", m_v22)

    # Track best models (Pareto front)
    pareto = []

    bounds = make_bounds_v23(x0)

    # ══════════════════════════════════════════════════════════════════
    # Phase 1: L-BFGS-B with multiple weight configurations
    # ══════════════════════════════════════════════════════════════════
    if args.phase in (0, 1):
        print(f"\n{'='*70}")
        print("Phase 1: L-BFGS-B — multi-weight sweep from v20b")
        print(f"{'='*70}")

        # Weight configs: (w_combvd, w_he, w_mac, w_munsell, w_osa, w_worst_sub)
        configs = [
            # Balanced
            (1.0, 0.05, 0.15, 0.015, 0.015, 0.03, "balanced"),
            # MacAdam-focused
            (1.0, 0.05, 0.30, 0.01, 0.01, 0.03, "mac_focus"),
            # OSA-focused
            (1.0, 0.05, 0.10, 0.02, 0.04, 0.03, "osa_focus"),
            # Munsell-focused
            (1.0, 0.05, 0.10, 0.04, 0.015, 0.03, "munsell_focus"),
            # All-equal
            (1.0, 0.10, 0.20, 0.02, 0.02, 0.05, "all_high"),
            # COMBVD-first (minimal others)
            (1.0, 0.03, 0.08, 0.008, 0.008, 0.02, "combvd_first"),
        ]

        for (wc, wh, wm, wmu, wo, wws, label) in configs:
            print(f"\n{'─'*70}")
            print(f"  Config: {label} (C={wc}, He={wh}, Mac={wm}, Mu={wmu}, OSA={wo})")
            print(f"{'─'*70}")

            obj = make_objective_all(combvd, he, mac, m_pairs, o_pairs,
                                     w_combvd=wc, w_he=wh, w_mac=wm,
                                     w_munsell=wmu, w_osa=wo,
                                     w_worst_sub=wws, x0_ref=p_init)

            best_x = x0.copy()
            best_loss = float("inf")

            for restart in range(min(args.restarts, 5)):
                print(f"  Restart {restart+1}/5...")
                t0 = time.time()
                x_opt, loss = run_lbfgsb(obj, best_x, bounds, maxiter=args.maxiter)
                dt = time.time() - t0

                m = evaluate_full(x_opt, combvd, he, mac, m_pairs, o_pairs, train, val)
                print_eval(f"R{restart+1} ({dt:.0f}s)", m, m_init)

                if m["rt"] > 1e-6:
                    print("    RT broken, skipping")
                    continue

                if loss < best_loss:
                    best_loss = loss
                    best_x = x_opt.copy()

                    # Save checkpoint
                    save_checkpoint(best_x, f"checkpoints/v23_{label}.json", m)

                    # Check if dominates v22
                    if dominates_v22(m, m_v22):
                        print(f"    *** DOMINATES v22! ***")
                        save_checkpoint(best_x, f"checkpoints/v23_dom_{label}.json", m)

                    pareto.append({"label": label, "x": best_x.copy(), "m": m})

            # Final eval for this config
            m_final = evaluate_full(best_x, combvd, he, mac, m_pairs, o_pairs, train, val)
            print_eval(f"{label} FINAL", m_final, m_init)

    # ══════════════════════════════════════════════════════════════════
    # Phase 2: CMA-ES from best Phase 1 result
    # ══════════════════════════════════════════════════════════════════
    if args.phase in (0, 2):
        print(f"\n{'='*70}")
        print("Phase 2: CMA-ES global search")
        print(f"{'='*70}")

        # Start from the best Pareto point for COMBVD
        if pareto:
            best_p1 = min(pareto, key=lambda p: p["m"]["combvd"])
            x_start = best_p1["x"]
            print(f"  Starting from best Phase 1: {best_p1['label']} (COMBVD={best_p1['m']['combvd']:.2f})")
        else:
            x_start = x0.copy()
            print("  Starting from v20b")

        obj_cma = make_objective_all(combvd, he, mac, m_pairs, o_pairs,
                                      w_combvd=1.0, w_he=0.05, w_mac=0.15,
                                      w_munsell=0.015, w_osa=0.015,
                                      w_worst_sub=0.03, x0_ref=p_init)

        for sigma in [0.01, 0.02, 0.05]:
            print(f"\n  CMA-ES sigma={sigma}...")
            t0 = time.time()
            try:
                x_cma, loss_cma = run_cmaes(obj_cma, x_start, bounds,
                                             sigma=sigma, max_evals=args.cma_evals,
                                             seed=42 + int(sigma * 100))
                dt = time.time() - t0

                m_cma = evaluate_full(x_cma, combvd, he, mac, m_pairs, o_pairs, train, val)
                print_eval(f"CMA sig={sigma} ({dt:.0f}s)", m_cma, m_init)

                if m_cma["rt"] < 1e-6:
                    save_checkpoint(x_cma, f"checkpoints/v23_cma_s{int(sigma*100):02d}.json", m_cma)
                    pareto.append({"label": f"cma_s{sigma}", "x": x_cma.copy(), "m": m_cma})

                    if dominates_v22(m_cma, m_v22):
                        print(f"    *** DOMINATES v22! ***")
                        save_checkpoint(x_cma, f"checkpoints/v23_dom_cma_s{int(sigma*100):02d}.json", m_cma)

                    # Polish with L-BFGS-B
                    print("    Polishing with L-BFGS-B...")
                    x_polish, _ = run_lbfgsb(obj_cma, x_cma, bounds, maxiter=3000)
                    m_polish = evaluate_full(x_polish, combvd, he, mac, m_pairs, o_pairs, train, val)
                    print_eval(f"  polished", m_polish, m_init)

                    if m_polish["rt"] < 1e-6:
                        save_checkpoint(x_polish, f"checkpoints/v23_cma_polish_s{int(sigma*100):02d}.json", m_polish)
                        pareto.append({"label": f"cma_polish_s{sigma}", "x": x_polish.copy(), "m": m_polish})

                        if dominates_v22(m_polish, m_v22):
                            print(f"    *** POLISHED DOMINATES v22! ***")
                            save_checkpoint(x_polish, f"checkpoints/v23_dom_cma_polish.json", m_polish)
            except Exception as e:
                print(f"    CMA-ES failed: {e}")

    # ══════════════════════════════════════════════════════════════════
    # Phase 3: Activate hue-dependent SL/SC from best point
    # ══════════════════════════════════════════════════════════════════
    if args.phase in (0, 3):
        print(f"\n{'='*70}")
        print("Phase 3: Hue-dependent SL/SC activation")
        print(f"{'='*70}")

        # Start from best Pareto point
        if pareto:
            # Find point closest to dominating v22
            scored = []
            for p in pareto:
                m = p["m"]
                # Score: sum of normalized improvements over v22
                score = (
                    max(0, m_v22["combvd"] - m["combvd"]) / m_v22["combvd"]
                    + max(0, m_v22["mac"] - m["mac"]) / m_v22["mac"]
                    + max(0, m_v22["he"] - m["he"]) / m_v22["he"]
                    + max(0, m_v22["munsell_cv"] - m["munsell_cv"]) / m_v22["munsell_cv"]
                    + max(0, m_v22["osa_cv"] - m["osa_cv"]) / m_v22["osa_cv"]
                )
                scored.append((score, p))
            scored.sort(reverse=True)
            best_start = scored[0][1]
            x_start = best_start["x"]
            print(f"  Starting from: {best_start['label']}")
        else:
            x_start = x0.copy()

        # Extend bounds to allow hue-dep SL/SC
        bounds_hslsc = make_bounds_v23(x_start)
        # The hue-dep SL/SC params aren't in the standard 72 pack_params
        # We need to handle them separately through the params object

        # For now, just do aggressive L-BFGS-B with wider dist_sl/sc bounds
        bounds_hslsc[70] = (-10.0, 15.0)  # dist_sl
        bounds_hslsc[71] = (-2.0, 5.0)    # dist_sc

        obj_h = make_objective_all(combvd, he, mac, m_pairs, o_pairs,
                                    w_combvd=1.0, w_he=0.05, w_mac=0.20,
                                    w_munsell=0.02, w_osa=0.02,
                                    w_worst_sub=0.03, x0_ref=p_init)

        for restart in range(min(args.restarts, 8)):
            print(f"  Restart {restart+1}/8...")
            t0 = time.time()
            x_opt, loss = run_lbfgsb(obj_h, x_start, bounds_hslsc, maxiter=args.maxiter)
            dt = time.time() - t0

            m = evaluate_full(x_opt, combvd, he, mac, m_pairs, o_pairs, train, val)
            print_eval(f"P3 R{restart+1} ({dt:.0f}s)", m, m_init)

            if m["rt"] < 1e-6:
                save_checkpoint(x_opt, f"checkpoints/v23_hslsc_r{restart+1}.json", m)
                pareto.append({"label": f"hslsc_r{restart+1}", "x": x_opt.copy(), "m": m})

                if dominates_v22(m, m_v22):
                    print(f"    *** DOMINATES v22! ***")
                    save_checkpoint(x_opt, "checkpoints/v23_best.json", m)

                x_start = x_opt.copy()

    # ══════════════════════════════════════════════════════════════════
    # Final: Pareto analysis
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PARETO ANALYSIS")
    print(f"{'='*70}")

    print(f"\n  {'Label':<25} {'COMBVD':>7} {'He':>7} {'Mac':>7} {'Mu%':>7} {'OSA%':>7} {'Euclid':>7} {'Gap':>6} {'Dom?':>5}")
    print(f"  {'-'*85}")
    for p in sorted(pareto, key=lambda p: p["m"]["combvd"]):
        m = p["m"]
        dom = "YES" if dominates_v22(m, m_v22) else "no"
        gap_str = f"{m.get('gap', 0):+.2f}" if "gap" in m else "N/A"
        print(f"  {p['label']:<25} {m['combvd']:7.2f} {m['he']:7.2f} {m['mac']:7.2f} "
              f"{m['munsell_cv']:7.1f} {m['osa_cv']:7.1f} {m['euclid']:7.2f} {gap_str:>6} {dom:>5}")

    # Reference lines
    print(f"  {'--- v20b ---':<25} {m_init['combvd']:7.2f} {m_init['he']:7.2f} {m_init['mac']:7.2f} "
          f"{m_init['munsell_cv']:7.1f} {m_init['osa_cv']:7.1f} {m_init['euclid']:7.2f}")
    print(f"  {'--- v22 ---':<25} {m_v22['combvd']:7.2f} {m_v22['he']:7.2f} {m_v22['mac']:7.2f} "
          f"{m_v22['munsell_cv']:7.1f} {m_v22['osa_cv']:7.1f} {m_v22['euclid']:7.2f}")

    # Find the best model that dominates v22
    dominators = [p for p in pareto if dominates_v22(p["m"], m_v22)]
    if dominators:
        best = min(dominators, key=lambda p: p["m"]["combvd"])
        print(f"\n  BEST DOMINATOR: {best['label']}")
        print_eval("WINNER", best["m"], m_init)
        save_checkpoint(best["x"], "checkpoints/v23_best.json", best["m"])
        print(f"  Saved: checkpoints/v23_best.json")
    else:
        print(f"\n  No model dominates v22. Finding closest...")
        # Find model with minimum total regression
        best_score = -999
        best_p = None
        for p in pareto:
            m = p["m"]
            score = sum([
                max(0, m_v22[k] - m[k]) for k in ["combvd", "he", "mac", "munsell_cv", "osa_cv"]
            ])
            if score > best_score:
                best_score = score
                best_p = p
        if best_p:
            print_eval("CLOSEST", best_p["m"], m_init)
            save_checkpoint(best_p["x"], "checkpoints/v23_best.json", best_p["m"])


if __name__ == "__main__":
    main()
