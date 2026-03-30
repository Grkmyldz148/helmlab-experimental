#!/usr/bin/env python
"""v24c: Fine-grained Pareto exploration + CMA-ES.

Key findings from v24b:
  - Alpha=0.7 from mac++osa++: COMBVD=24.43, Mac=18.77, OsaCV=18.0%
  - t=0.3 interpolation: COMBVD=23.53, Mac=19.84, OsaCV=20.1%
  - t=0.7 interpolation: COMBVD=23.87, Mac=19.27, OsaCV=19.2%

Strategy: Combine interpolation + balanced alpha for fine Pareto search.
Then CMA-ES from best feasible points.
"""

import sys, os, json, time
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

def pack_params(p):
    return np.concatenate([
        p.M1.ravel(), p.gamma, p.M2.ravel(),
        [p.hk_weight, p.hk_power, p.hk_hue_mod],
        [p.L_corr_p1, p.L_corr_p2, p.L_corr_p3],
        [p.cs_cos1, p.cs_sin1, p.cs_cos2, p.cs_sin2, p.cs_cos3, p.cs_sin3],
        [p.lc1, p.lc2],
        [p.hk_sin1, p.hk_cos2, p.hk_sin2],
        [p.hue_cos1, p.hue_sin1, p.hue_cos2, p.hue_sin2, p.hue_cos3, p.hue_sin3],
        [p.hlc_cos1, p.hlc_sin1, p.hlc_cos2, p.hlc_sin2],
        [p.hl_cos1, p.hl_sin1, p.hl_cos2, p.hl_sin2],
        [p.cp_cos1, p.cp_sin1, p.cp_cos2, p.cp_sin2],
        [p.lp_dark, p.dist_power, p.dist_wC],
        [p.hue_cos4, p.hue_sin4, p.Lh_cos1, p.Lh_sin1, p.cs_cos4, p.cs_sin4],
        [p.dist_compress, p.lp_dark_hcos, p.lp_dark_hsin],
        [p.dist_linear, p.dist_post_power, p.dist_sl, p.dist_sc],
    ])

def unpack_params(x):
    return MetricParams(
        M1=x[0:9].reshape(3,3), gamma=x[9:12], M2=x[12:21].reshape(3,3),
        hk_weight=float(x[21]), hk_power=float(x[22]), hk_hue_mod=float(x[23]),
        L_corr_p1=float(x[24]), L_corr_p2=float(x[25]), L_corr_p3=float(x[26]),
        cs_cos1=float(x[27]), cs_sin1=float(x[28]), cs_cos2=float(x[29]), cs_sin2=float(x[30]),
        cs_cos3=float(x[31]), cs_sin3=float(x[32]), lc1=float(x[33]), lc2=float(x[34]),
        hk_sin1=float(x[35]), hk_cos2=float(x[36]), hk_sin2=float(x[37]),
        hue_cos1=float(x[38]), hue_sin1=float(x[39]), hue_cos2=float(x[40]), hue_sin2=float(x[41]),
        hue_cos3=float(x[42]), hue_sin3=float(x[43]),
        hlc_cos1=float(x[44]), hlc_sin1=float(x[45]), hlc_cos2=float(x[46]), hlc_sin2=float(x[47]),
        hl_cos1=float(x[48]), hl_sin1=float(x[49]), hl_cos2=float(x[50]), hl_sin2=float(x[51]),
        cp_cos1=float(x[52]), cp_sin1=float(x[53]), cp_cos2=float(x[54]), cp_sin2=float(x[55]),
        lp_dark=float(x[56]), dist_power=float(x[57]), dist_wC=float(x[58]),
        hue_cos4=float(x[59]), hue_sin4=float(x[60]),
        Lh_cos1=float(x[61]), Lh_sin1=float(x[62]),
        cs_cos4=float(x[63]), cs_sin4=float(x[64]),
        dist_compress=float(x[65]),
        lp_dark_hcos=float(x[66]), lp_dark_hsin=float(x[67]),
        dist_linear=float(x[68]), dist_post_power=float(x[69]),
        dist_sl=float(x[70]), dist_sc=float(x[71]),
    )

def make_bounds(x0, wider=False):
    w = 1.5 if wider else 1.0
    bounds = []
    for i in range(N_PARAMS):
        if i < 9 or (12 <= i < 21):
            c = x0[i]
            half = max(abs(c)*w, 0.5*w)
            bounds.append((max(c-half,-3.0), min(c+half,3.0)))
        elif 9 <= i < 12: bounds.append((0.15, 0.95))
        elif i == 21: bounds.append((0.0, 3.0))
        elif i == 22: bounds.append((0.1, 2.0))
        elif i == 23: bounds.append((-1.0, 1.0))
        elif i == 25: bounds.append((-1.2, 1.2))
        elif 24 <= i <= 26: bounds.append((-0.8, 0.8))
        elif 27 <= i <= 32: bounds.append((-1.0, 1.0))
        elif i in (33,34): bounds.append((-1.0, 1.0))
        elif 35 <= i <= 37: bounds.append((-1.5, 1.5))
        elif 38 <= i <= 43: bounds.append((-0.5, 0.5))
        elif 44 <= i <= 47: bounds.append((-1.0, 1.0))
        elif 48 <= i <= 51: bounds.append((-0.3, 0.3))
        elif 52 <= i <= 55: bounds.append((-0.5, 0.5))
        elif i == 56: bounds.append((-0.5, 1.0))
        elif i == 57: bounds.append((0.5, 1.5))
        elif i == 58: bounds.append((0.3, 3.0))
        elif 59 <= i <= 60: bounds.append((-0.3, 0.3))
        elif 61 <= i <= 62: bounds.append((-0.5, 0.5))
        elif 63 <= i <= 64: bounds.append((-0.5, 0.5))
        elif i == 65: bounds.append((0.0, 10.0))
        elif 66 <= i <= 67: bounds.append((-0.5, 0.5))
        elif i == 68: bounds.append((0.0, 0.0))  # dist_linear fixed
        elif i == 69: bounds.append((0.8, 1.5))
        elif i == 70: bounds.append((-5.0, 10.0))
        elif i == 71: bounds.append((-2.0, 5.0))
    return bounds

def full_eval(x, data):
    combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt = data
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
        munsell_cv = 100*np.std(DE_mu)/np.mean(DE_mu) if np.mean(DE_mu)>1e-10 else 200
        DE_osa = s.distance(osa_pairs["XYZ_1"], osa_pairs["XYZ_2"])
        osa_cv = 100*np.std(DE_osa)/np.mean(DE_osa) if np.mean(DE_osa)>1e-10 else 200
        rt = float(s.round_trip_error(XYZ_rt).max())
        return dict(combvd=s_combvd, he=s_he, mac=s_mac, munsell_cv=munsell_cv, osa_cv=osa_cv, rt=rt)
    except:
        return dict(combvd=100, he=100, mac=100, munsell_cv=200, osa_cv=200, rt=1)

def print_eval(label, m):
    flags = ("C" if m["combvd"]<=23.3 else ".") + ("H" if m["he"]<=30.33 else ".") + \
            ("M" if m["mac"]<=18.71 else ".") + ("U" if m["munsell_cv"]<=40 else ".") + \
            ("O" if m["osa_cv"]<=16.6 else ".")
    print(f"  {label}: C={m['combvd']:.2f} H={m['he']:.2f} M={m['mac']:.2f} "
          f"MuCV={m['munsell_cv']:.1f}% OCV={m['osa_cv']:.1f}% RT={m['rt']:.1e} [{flags}]")

_ec = 0; _bl = float("inf"); _lp = 0.0

def make_obj(data, alpha=0.5, combvd_cap=None, osa_weight=2.0):
    """Balanced + optional cap objective."""
    combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt = data
    c1, c2, cD = combvd["XYZ_1"], combvd["XYZ_2"], combvd["DV"]
    h1, h2, hD = he["XYZ_1"], he["XYZ_2"], he["DV"]
    m1, m2, mD = mac["XYZ_1"], mac["XYZ_2"], mac["DV"]
    mu1, mu2 = munsell_pairs["XYZ_1"], munsell_pairs["XYZ_2"]
    o1, o2 = osa_pairs["XYZ_1"], osa_pairs["XYZ_2"]

    def objective(x):
        global _ec, _bl, _lp
        try:
            p = unpack_params(x)
            s = MetricSpace(p, neutral_correction=True)
            DE_c = s.distance(c1, c2)
            if np.any(~np.isfinite(DE_c)): return 200
            sc = stress(cD, DE_c)
            DE_h = s.distance(h1, h2)
            if np.any(~np.isfinite(DE_h)): return 200
            sh = stress(hD, DE_h)
            DE_m = s.distance(m1, m2)
            if np.any(~np.isfinite(DE_m)): return 200
            sm = stress(mD, DE_m)
            DE_mu = s.distance(mu1, mu2)
            if np.any(~np.isfinite(DE_mu)) or np.mean(DE_mu)<1e-10: return 200
            mcv = 100*np.std(DE_mu)/np.mean(DE_mu)
            DE_o = s.distance(o1, o2)
            if np.any(~np.isfinite(DE_o)) or np.mean(DE_o)<1e-10: return 200
            ocv = 100*np.std(DE_o)/np.mean(DE_o)

            # Balanced objective
            combvd_term = sc + 0.05*sh
            macosa_term = sm + osa_weight*ocv
            total = alpha*combvd_term + (1-alpha)*macosa_term + 0.01*mcv

            # Hard cap on COMBVD if requested
            if combvd_cap is not None and sc > combvd_cap:
                total += 100*(sc - combvd_cap)**2

            rt = s.round_trip_error(XYZ_rt).max()
            if rt > 1e-6:
                total += 20*np.log10(rt/1e-6)
        except:
            return 200

        _ec += 1
        if total < _bl: _bl = total
        now = time.time()
        if now - _lp > 20:
            _lp = now
            print(f"  #{_ec:>5d} C={sc:.2f} H={sh:.2f} M={sm:.2f} MuCV={mcv:.1f}% OCV={ocv:.1f}%  best={_bl:.3f}", flush=True)
        return total
    return objective


def main():
    print("v24c: Fine Pareto + CMA-ES")
    print("Loading...")
    combvd = load_combvd()
    he = load_he2022()
    mac = load_macadam1974()
    munsell_pairs = generate_munsell_pairs(load_munsell(subset="real"))
    osa_pairs = generate_osa_pairs(load_osa_ucs())

    p0 = MetricParams.load("checkpoints/v23_he_fix.json")
    rng = np.random.default_rng(42)
    _xyz = rng.uniform(0.05, 0.90, (2000, 3))
    _lms = _xyz @ p0.M1.T
    XYZ_rt = _xyz[(_lms >= 0).all(axis=1)][:1000]

    data = (combvd, he, mac, munsell_pairs, osa_pairs, XYZ_rt)

    x_base = pack_params(p0)
    m_base = full_eval(x_base, data)
    print_eval("baseline", m_base)

    # Load mac++osa++
    if os.path.exists("checkpoints/v24_mac++osa++.json"):
        x_mo = pack_params(MetricParams.load("checkpoints/v24_mac++osa++.json"))
    else:
        x_mo = None
        print("  WARNING: No mac++osa++ checkpoint. Run v24 sweep first.")
        return

    m_mo = full_eval(x_mo, data)
    print_eval("mac++osa++", m_mo)

    # Collect all Pareto results
    pareto = []

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Fine alpha sweep from mac++osa++ (0.65-0.85 range)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Phase 1: Fine alpha sweep from mac++osa++")
    print(f"{'='*70}")

    for alpha in [0.60, 0.65, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.85]:
        global _ec, _bl, _lp
        _ec, _bl, _lp = 0, float("inf"), 0.0
        print(f"\n  alpha={alpha:.2f}")
        obj = make_obj(data, alpha=alpha, osa_weight=2.0)
        bounds = make_bounds(x_mo, wider=True)
        result = minimize(obj, x0=x_mo, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-11})
        m = full_eval(result.x, data)
        print_eval(f"a={alpha:.2f}", m)
        if m["rt"] <= 1e-6:
            pareto.append({"alpha": alpha, "x": result.x.copy(), "m": m, "src": "mo"})
            unpack_params(result.x).save(f"checkpoints/v24c_a{alpha:.2f}.json")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Fine alpha sweep from mac++osa++ with COMBVD cap
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Phase 2: Alpha + COMBVD cap from mac++osa++")
    print(f"{'='*70}")

    for alpha, cap in [(0.5, 24.0), (0.5, 23.5), (0.4, 24.0), (0.4, 23.5),
                        (0.3, 24.0), (0.3, 23.5), (0.6, 24.0)]:
        _ec, _bl, _lp = 0, float("inf"), 0.0
        label = f"a={alpha:.1f}_cap={cap:.1f}"
        print(f"\n  {label}")
        obj = make_obj(data, alpha=alpha, combvd_cap=cap, osa_weight=2.0)
        bounds = make_bounds(x_mo, wider=True)
        result = minimize(obj, x0=x_mo, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-11})
        m = full_eval(result.x, data)
        print_eval(label, m)
        if m["rt"] <= 1e-6:
            pareto.append({"label": label, "x": result.x.copy(), "m": m, "src": "mo+cap"})

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Interpolation + alpha sweep
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Phase 3: Interpolation + alpha sweep")
    print(f"{'='*70}")

    for t_val in [0.4, 0.5, 0.6]:
        x_interp = (1-t_val)*x_base + t_val*x_mo
        for alpha in [0.5, 0.6, 0.7]:
            _ec, _bl, _lp = 0, float("inf"), 0.0
            label = f"t={t_val:.1f}_a={alpha:.1f}"
            print(f"\n  {label}")
            obj = make_obj(data, alpha=alpha, osa_weight=2.0)
            bounds = make_bounds(x_interp, wider=True)
            result = minimize(obj, x0=x_interp, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-11})
            m = full_eval(result.x, data)
            print_eval(label, m)
            if m["rt"] <= 1e-6:
                pareto.append({"label": label, "x": result.x.copy(), "m": m, "src": "interp"})

    # ═══════════════════════════════════════════════════════════
    # Phase 4: CMA-ES from best near-feasible points
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Phase 4: CMA-ES refinement")
    print(f"{'='*70}")

    try:
        import cma

        # Find best candidates with COMBVD <= 24.5
        candidates = [p for p in pareto if p["m"]["combvd"] <= 24.5 and p["m"]["rt"] <= 1e-6]
        if candidates:
            # Sort by combined metric quality
            def combo_score(p):
                m = p["m"]
                return m["combvd"] + m["mac"] + m["osa_cv"]
            candidates.sort(key=combo_score)

            for ci, cand in enumerate(candidates[:3]):
                _ec, _bl, _lp = 0, float("inf"), 0.0
                print(f"\n  CMA-ES from candidate {ci+1}: C={cand['m']['combvd']:.2f} M={cand['m']['mac']:.2f} O={cand['m']['osa_cv']:.1f}%")

                obj_cma = make_obj(data, alpha=0.65, combvd_cap=24.0, osa_weight=3.0)
                bounds_cma = make_bounds(cand["x"], wider=True)
                lb = [b[0] for b in bounds_cma]
                ub = [b[1] for b in bounds_cma]
                # Fix dist_linear — make lb slightly less than ub to avoid CMA error
                lb[68] = -1e-10
                ub[68] = 1e-10

                opts = cma.CMAOptions()
                opts.set("bounds", [lb, ub])
                opts.set("maxfevals", 20000)
                opts.set("timeout", 300)
                opts.set("verbose", -1)

                es = cma.CMAEvolutionStrategy(cand["x"].tolist(), 0.005, opts)
                gen = 0
                while not es.stop():
                    solutions = es.ask()
                    fitnesses = [obj_cma(s) for s in solutions]
                    es.tell(solutions, fitnesses)
                    gen += 1
                    if gen % 20 == 0:
                        m_cma = full_eval(np.array(es.best.x), data)
                        print_eval(f"CMA g{gen}", m_cma)

                m_cma = full_eval(np.array(es.best.x), data)
                print_eval(f"CMA final {ci+1}", m_cma)
                if m_cma["rt"] <= 1e-6:
                    pareto.append({"label": f"cma_{ci}", "x": np.array(es.best.x), "m": m_cma, "src": "cma"})

    except ImportError:
        print("  CMA-ES not available")
    except Exception as e:
        print(f"  CMA-ES error: {e}")

    # ═══════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PARETO FRONTIER SUMMARY")
    print(f"{'='*70}")

    # Sort by COMBVD
    valid = [p for p in pareto if p["m"]["rt"] <= 1e-6]
    valid.sort(key=lambda p: p["m"]["combvd"])

    print(f"{'Label':>20s} {'COMBVD':>8s} {'He':>6s} {'Mac':>8s} {'MunCV':>7s} {'OsaCV':>7s}")
    for p in valid:
        m = p["m"]
        label = p.get("label", f"a={p.get('alpha','?')}")
        flags = ("C" if m["combvd"]<=23.3 else ".") + ("H" if m["he"]<=30.33 else ".") + \
                ("M" if m["mac"]<=18.71 else ".") + ("U" if m["munsell_cv"]<=40 else ".") + \
                ("O" if m["osa_cv"]<=16.6 else ".")
        print(f"{label:>20s} {m['combvd']:8.2f} {m['he']:6.2f} {m['mac']:8.2f} "
              f"{m['munsell_cv']:6.1f}% {m['osa_cv']:6.1f}% [{flags}]")

    # Find best that meets COMBVD constraint
    feasible = [p for p in valid if p["m"]["combvd"] <= 23.50]
    if feasible:
        best_f = min(feasible, key=lambda p: p["m"]["mac"] + p["m"]["osa_cv"])
        print(f"\n  Best with COMBVD<=23.5:")
        print_eval("Best", best_f["m"])
        unpack_params(best_f["x"]).save("checkpoints/v24_best_feasible.json")
        with open("checkpoints/v24_best_feasible_metrics.json", "w") as f:
            json.dump(best_f["m"], f, indent=2)

    # Best overall
    if valid:
        targets = {"combvd": 23.3, "he": 30.33, "mac": 18.71, "munsell_cv": 40, "osa_cv": 16.6}
        def total_gap(p):
            return sum(max(0, p["m"][k]-v)/v for k,v in targets.items())
        best_all = min(valid, key=total_gap)
        print(f"\n  Best overall (min total gap):")
        print_eval("Best", best_all["m"])
        unpack_params(best_all["x"]).save("checkpoints/v24_best.json")
        with open("checkpoints/v24_best_metrics.json", "w") as f:
            json.dump(best_all["m"], f, indent=2)

    print(f"\n  Targets: COMBVD≤23.30  He≤30.33  Mac≤18.71  MunCV≤40%  OsaCV≤16.6%")
    print(f"  Assessment: ", end="")

    # Check if simultaneous targets are achievable
    mac_feasible = any(p["m"]["mac"] <= 18.71 for p in valid)
    osa_feasible = any(p["m"]["osa_cv"] <= 16.6 for p in valid)
    both = any(p["m"]["mac"] <= 18.71 and p["m"]["osa_cv"] <= 16.6 for p in valid)
    all5 = any(p["m"]["combvd"]<=23.3 and p["m"]["he"]<=30.33 and p["m"]["mac"]<=18.71
               and p["m"]["munsell_cv"]<=40 and p["m"]["osa_cv"]<=16.6 for p in valid)

    if all5:
        print("ALL 5 TARGETS ACHIEVABLE!")
    elif both:
        print(f"Mac+OSA achievable but COMBVD/He conflicts. Best COMBVD with both: "
              f"{min(p['m']['combvd'] for p in valid if p['m']['mac']<=18.71 and p['m']['osa_cv']<=16.6):.2f}")
    elif mac_feasible:
        print(f"Mac achievable alone (best with Mac<18.71: COMBVD="
              f"{min(p['m']['combvd'] for p in valid if p['m']['mac']<=18.71):.2f})")
    elif osa_feasible:
        print(f"OSA achievable alone (best with OSA<16.6: COMBVD="
              f"{min(p['m']['combvd'] for p in valid if p['m']['osa_cv']<=16.6):.2f})")
    else:
        print("Neither Mac nor OSA target reachable with current parameterization")

    print(f"{'='*70}")

if __name__ == "__main__":
    main()
