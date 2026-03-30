#!/usr/bin/env python
"""v23 from v22: Start from v22 checkpoint and try to recover COMBVD
while maintaining MacAdam/Munsell/OSA improvements.

Key insight: v22 has great MacAdam (18.83) and OSA (21.0%) but COMBVD
regressed to 23.92. Can we get COMBVD back to ~23.3 without losing these?

Strategy: Heavy COMBVD weight + penalties for regressing Mac/Munsell/OSA
"""

import sys, time, json
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

from optimize_v14 import pack_params, unpack_params, make_bounds, N_PARAMS, split_combvd


def evaluate(x, combvd, he, mac, m_pairs, o_pairs):
    p = unpack_params(x)
    s = MetricSpace(p, neutral_correction=True, ab_rotate_deg=-28.2)
    DE_c = s.distance(combvd["XYZ_1"], combvd["XYZ_2"])
    s_c = stress(combvd["DV"], DE_c)
    s_h = stress(he["DV"], s.distance(he["XYZ_1"], he["XYZ_2"]))
    s_m = stress(mac["DV"], s.distance(mac["XYZ_1"], mac["XYZ_2"]))
    DE_mu = s.distance(m_pairs["XYZ_1"], m_pairs["XYZ_2"])
    mu_cv = 100 * np.std(DE_mu) / np.mean(DE_mu) if np.mean(DE_mu) > 1e-10 else 999
    DE_o = s.distance(o_pairs["XYZ_1"], o_pairs["XYZ_2"])
    o_cv = 100 * np.std(DE_o) / np.mean(DE_o) if np.mean(DE_o) > 1e-10 else 999
    rng = np.random.default_rng(42)
    xyz = rng.uniform(0.05, 0.90, (2000, 3))
    lms = xyz @ p.M1.T
    xyz_v = xyz[(lms >= 0).all(axis=1)][:1000]
    rt = s.round_trip_error(xyz_v).max()
    return {"combvd": s_c, "he": s_h, "mac": s_m, "munsell_cv": mu_cv, "osa_cv": o_cv, "rt": rt}


def main():
    print("v23-from-v22: Recover COMBVD while maintaining Mac/Munsell/OSA gains")
    print("Loading data...")

    combvd = load_combvd()
    he = load_he2022()
    mac = load_macadam1974()
    munsell = load_munsell("real")
    m_pairs = generate_munsell_pairs(munsell)
    osa = load_osa_ucs()
    o_pairs = generate_osa_pairs(osa)

    # Load v22
    p22 = MetricParams.load("checkpoints/v22_best.json")
    x22 = pack_params(p22)
    m22 = evaluate(x22, combvd, he, mac, m_pairs, o_pairs)
    print(f"v22: COMBVD={m22['combvd']:.2f} He={m22['he']:.2f} Mac={m22['mac']:.2f} "
          f"Mu={m22['munsell_cv']:.1f}% OSA={m22['osa_cv']:.1f}%")

    # v20b reference
    p20 = MetricParams.load("src/helmlab/data/metric_params.json")
    x20 = pack_params(p20)
    m20 = evaluate(x20, combvd, he, mac, m_pairs, o_pairs)

    # RT test
    rng = np.random.default_rng(42)
    _xyz = rng.uniform(0.05, 0.90, (2000, 3))
    _lms = _xyz @ p22.M1.T
    XYZ_rt = _xyz[(_lms >= 0).all(axis=1)][:1000]

    ds_arr = np.array(combvd["dataset"])
    sub_masks = {ds: ds_arr == ds for ds in sorted(set(combvd["dataset"]))}

    m_X1, m_X2 = m_pairs["XYZ_1"], m_pairs["XYZ_2"]
    o_X1, o_X2 = o_pairs["XYZ_1"], o_pairs["XYZ_2"]

    # Targets (v22 values -- don't let them regress)
    mac_target = m22["mac"]  # 18.83
    munsell_target = m22["munsell_cv"]  # 36.4
    osa_target = m22["osa_cv"]  # 21.0
    he_target = m22["he"]  # 30.34

    _count = [0]
    _best = [float("inf")]
    _last_t = [0.0]

    def objective(x):
        try:
            params = unpack_params(x)
            space = MetricSpace(params, neutral_correction=True, ab_rotate_deg=-28.2)

            DE_c = space.distance(combvd["XYZ_1"], combvd["XYZ_2"])
            if np.any(~np.isfinite(DE_c)):
                return 200.0
            s_full = stress(combvd["DV"], DE_c)

            sub_s = [stress(combvd["DV"][m], DE_c[m]) for m in sub_masks.values()]
            s_mean_sub = float(np.mean(sub_s))

            DE_h = space.distance(he["XYZ_1"], he["XYZ_2"])
            if np.any(~np.isfinite(DE_h)):
                return 200.0
            s_he = stress(he["DV"], DE_h)

            DE_m = space.distance(mac["XYZ_1"], mac["XYZ_2"])
            if np.any(~np.isfinite(DE_m)):
                return 200.0
            s_mac = stress(mac["DV"], DE_m)

            DE_mu = space.distance(m_X1, m_X2)
            if np.any(~np.isfinite(DE_mu)) or np.mean(DE_mu) < 1e-10:
                return 200.0
            munsell_cv = float(np.std(DE_mu) / np.mean(DE_mu) * 100.0)

            DE_o = space.distance(o_X1, o_X2)
            if np.any(~np.isfinite(DE_o)) or np.mean(DE_o) < 1e-10:
                return 200.0
            osa_cv = float(np.std(DE_o) / np.mean(DE_o) * 100.0)

            # Primary: COMBVD (heavily weighted)
            total = 0.6 * s_full + 0.4 * s_mean_sub

            # Secondary: He
            total += 0.03 * s_he

            # Penalty terms: don't let Mac/Munsell/OSA regress from v22
            mac_penalty = max(0, s_mac - mac_target) ** 2
            munsell_penalty = max(0, munsell_cv - munsell_target) ** 2
            osa_penalty = max(0, osa_cv - osa_target) ** 2
            he_penalty = max(0, s_he - he_target) ** 2

            total += 0.5 * mac_penalty
            total += 0.1 * munsell_penalty
            total += 0.1 * osa_penalty
            total += 0.3 * he_penalty

            # Small bonus for improving Mac/Munsell/OSA beyond v22
            total += 0.05 * s_mac + 0.005 * munsell_cv + 0.005 * osa_cv

            # RT
            rt = space.round_trip_error(XYZ_rt).max()
            if rt > 1e-6:
                total += 20.0 * np.log10(rt / 1e-6)

        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 200.0

        _count[0] += 1
        if total < _best[0]:
            _best[0] = total

        now = time.time()
        if now - _last_t[0] > 15.0:
            _last_t[0] = now
            print(f"  #{_count[0]:>6d}  loss={total:.4f}  "
                  f"C={s_full:.2f}  Mac={s_mac:.2f}  He={s_he:.2f}  "
                  f"Mu={munsell_cv:.1f}%  OSA={osa_cv:.1f}%  "
                  f"best={_best[0]:.4f}", flush=True)

        return total

    bounds = make_bounds(x22)
    bounds[61] = (-0.3, 0.3)
    bounds[62] = (-0.3, 0.3)
    bounds[70] = (-5.0, 10.0)
    bounds[71] = (-1.0, 3.0)

    best_x = x22.copy()
    best_combvd = m22["combvd"]

    for restart in range(12):
        _count[0] = 0
        _best[0] = float("inf")
        _last_t[0] = 0.0

        print(f"\n{'='*60}")
        print(f"Restart {restart+1}/12")
        print(f"{'='*60}")

        t0 = time.time()
        result = minimize(objective, x0=best_x, method="L-BFGS-B", bounds=bounds,
                         options={"maxiter": 8000, "ftol": 1e-14, "gtol": 1e-12})
        dt = time.time() - t0

        m = evaluate(result.x, combvd, he, mac, m_pairs, o_pairs)
        print(f"  R{restart+1} ({dt:.0f}s): COMBVD={m['combvd']:.2f} He={m['he']:.2f} "
              f"Mac={m['mac']:.2f} Mu={m['munsell_cv']:.1f}% OSA={m['osa_cv']:.1f}% RT={m['rt']:.2e}")

        # Accept if COMBVD improved and Mac/Munsell/OSA/He not regressed more than small margin
        if (m["rt"] < 1e-6 and
            m["combvd"] < best_combvd + 0.1 and
            m["mac"] <= mac_target + 0.5 and
            m["munsell_cv"] <= munsell_target + 1.0 and
            m["osa_cv"] <= osa_target + 0.5):

            if m["combvd"] < best_combvd:
                best_x = result.x.copy()
                best_combvd = m["combvd"]
                print(f"  ** New best COMBVD={best_combvd:.2f}")

                p = unpack_params(best_x)
                p.save("checkpoints/v23_from_v22.json")

                # Check if this dominates v22
                if (m["combvd"] <= m22["combvd"] and
                    m["mac"] <= m22["mac"] + 0.01 and
                    m["he"] <= m22["he"] + 0.01 and
                    m["munsell_cv"] <= m22["munsell_cv"] + 0.01 and
                    m["osa_cv"] <= m22["osa_cv"] + 0.01):
                    print(f"  *** DOMINATES v22! ***")
                    p.save("checkpoints/v23_best.json")

    # Final
    m_final = evaluate(best_x, combvd, he, mac, m_pairs, o_pairs)
    print(f"\n{'='*60}")
    print(f"FINAL")
    print(f"{'='*60}")
    print(f"  COMBVD={m_final['combvd']:.2f} He={m_final['he']:.2f} "
          f"Mac={m_final['mac']:.2f} Mu={m_final['munsell_cv']:.1f}% OSA={m_final['osa_cv']:.1f}%")
    print(f"  vs v22: COMBVD={m_final['combvd']-m22['combvd']:+.2f} "
          f"Mac={m_final['mac']-m22['mac']:+.2f} Mu={m_final['munsell_cv']-m22['munsell_cv']:+.1f}")
    print(f"  vs v20: COMBVD={m_final['combvd']-m20['combvd']:+.2f} "
          f"Mac={m_final['mac']-m20['mac']:+.2f}")


if __name__ == "__main__":
    main()
