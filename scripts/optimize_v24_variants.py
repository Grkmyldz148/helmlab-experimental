#!/usr/bin/env python3
"""v24 variants: Test 4 strategies to fix yellow concavity.

A) M1 fixed, only M2 optimized (4 free params)
B) Full M1+M2 with yellow L constraint (L_yellow > 0.95)
C) Full M1+M2 with tight condition number (cond < 5)
D) All constraints combined

Usage: python scripts/optimize_v24_variants.py
"""

import sys
sys.path.insert(0, "src")

import json
import time
import numpy as np
import cma
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════

D65 = np.array([0.95047, 1.0, 1.08883])

M_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
M_XYZ_TO_SRGB = np.linalg.inv(M_SRGB_TO_XYZ)

# v14 matrices (starting point)
V14_M1 = np.array([
    [0.7583761294836658, 0.38380162590825084, -0.09608055040602373],
    [0.12671393631532843, 0.8421628149123207, 0.03434823621506485],
    [0.07639223722200054, 0.258943526275451, 0.6139139663787314],
])
V14_M2 = np.array([
    [0.10058070589596230, 1.01558970993941444, -0.11617041583537688],
    [2.36157646996164416, -2.44099737506293479, 0.07942090510129070],
    [0.04565327074453784, 0.81875488445424471, -0.86440815519878267],
])

# Yellow XYZ
YELLOW_LIN = np.array([1.0, 1.0, 0.0])
YELLOW_XYZ = M_SRGB_TO_XYZ @ YELLOW_LIN


# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════

def signed_cbrt(x):
    return np.sign(x) * np.abs(x) ** (1/3)

def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1/2.4) - 0.055)

def xyz_to_cielab(xyz):
    r = xyz / D65
    f = np.where(r > 0.008856, r ** (1/3), 7.787 * r + 16/116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])

def ortho_basis(s):
    s_n = s / np.linalg.norm(s)
    v = np.array([1.0, 0.0, 0.0]) if abs(s_n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = v - np.dot(v, s_n) * s_n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s_n, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


# ══════════════════════════════════════════════════════════════════════
# Parameterizations
# ══════════════════════════════════════════════════════════════════════

def m2_params_to_matrix(x, M1):
    """4 params → M2 (with M1 fixed). x=[a1,a2,b1,b2]"""
    s = signed_cbrt(M1 @ D65)
    e1, e2 = ortho_basis(s)
    M2 = np.zeros((3, 3))
    # L-row: keep v14's L-row (normalized)
    M2[0] = V14_M2[0]
    L_white = M2[0] @ s
    M2[0] = M2[0] / L_white
    M2[1] = x[0] * e1 + x[1] * e2
    M2[2] = x[2] * e1 + x[3] * e2
    return M2

def m2_matrix_to_params(M2, M1):
    """M2 → 4 params (a,b projections)."""
    s = signed_cbrt(M1 @ D65)
    e1, e2 = ortho_basis(s)
    return np.array([M2[1]@e1, M2[1]@e2, M2[2]@e1, M2[2]@e2])

def full_params_to_matrices(x):
    """13 params → M1 (D65-norm) + M2 (ach-constrained)."""
    M1 = np.zeros((3, 3))
    for i in range(3):
        M1[i, 0] = x[2*i]
        M1[i, 1] = x[2*i + 1]
        M1[i, 2] = (1.0 - M1[i, 0]*D65[0] - M1[i, 1]*D65[1]) / D65[2]
    s = signed_cbrt(M1 @ D65)
    e1, e2 = ortho_basis(s)
    M2 = np.zeros((3, 3))
    M2[0] = x[6:9]
    L_w = M2[0] @ s
    if abs(L_w) < 1e-10: raise ValueError
    M2[0] /= L_w
    M2[1] = x[9]*e1 + x[10]*e2
    M2[2] = x[11]*e1 + x[12]*e2
    return M1, M2

def full_matrices_to_params(M1, M2):
    """M1, M2 → 13 params."""
    x = np.zeros(13)
    for i in range(3):
        x[2*i] = M1[i, 0]
        x[2*i+1] = M1[i, 1]
    x[6:9] = M2[0]
    s = signed_cbrt(M1 @ D65)
    e1, e2 = ortho_basis(s)
    x[9], x[10] = M2[1]@e1, M2[1]@e2
    x[11], x[12] = M2[2]@e1, M2[2]@e2
    return x


# ══════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════

def generate_training_pairs(n_random=80, seed=42):
    rng = np.random.RandomState(seed)
    pairs = []
    prims = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[0,0,0]]
    for i in range(len(prims)):
        for j in range(i+1, len(prims)):
            pairs.append((prims[i], prims[j]))
    for g1 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for g2 in [g1+0.2, g1+0.4]:
            if g2 <= 1.0: pairs.append(([g1]*3, [g2]*3))
    for _ in range(n_random):
        pairs.append((rng.rand(3).tolist(), rng.rand(3).tolist()))
    xyz_pairs = []
    for c1, c2 in pairs:
        xyz_pairs.append((M_SRGB_TO_XYZ @ srgb_to_linear(np.array(c1)),
                          M_SRGB_TO_XYZ @ srgb_to_linear(np.array(c2))))
    return xyz_pairs


def ciede2000_simple(lab1, lab2):
    L1, a1, b1 = lab1; L2, a2, b2 = lab2
    dL = L2-L1; C1 = np.sqrt(a1**2+b1**2); C2 = np.sqrt(a2**2+b2**2)
    dC = C2-C1; dH2 = (a2-a1)**2+(b2-b1)**2-dC**2
    dH = np.sqrt(max(0, dH2))
    SL = 1+0.015*(L1-50)**2/np.sqrt(20+(L1-50)**2)
    SC = 1+0.045*C1; SH = 1+0.015*C1
    return np.sqrt((dL/SL)**2+(dC/SC)**2+(dH/SH)**2)


def compute_cv(M1, M2, xyz_pairs, n_steps=25):
    M1i, M2i = np.linalg.inv(M1), np.linalg.inv(M2)
    cvs = []
    for xyz1, xyz2 in xyz_pairs:
        lab1, lab2 = M2@signed_cbrt(M1@xyz1), M2@signed_cbrt(M1@xyz2)
        ts = np.linspace(0, 1, n_steps+1)
        deltas, prev = [], None
        for t in ts:
            lab = lab1 + t*(lab2-lab1)
            lms_c = M2i@lab; lms = np.sign(lms_c)*np.abs(lms_c)**3
            xyz = M1i@lms; lin = M_XYZ_TO_SRGB@xyz
            s8 = np.round(linear_to_srgb(np.clip(lin,0,1))*255)/255
            cl = xyz_to_cielab(np.maximum(M_SRGB_TO_XYZ@srgb_to_linear(s8), 1e-10))
            if prev is not None: deltas.append(ciede2000_simple(prev, cl))
            prev = cl
        if deltas:
            arr = np.array(deltas); m = np.mean(arr)
            if m > 0.001: cvs.append(np.std(arr)/m)
    return (np.mean(cvs), cvs) if cvs else (999.0, [])


def compute_monotonicity(M1, M2, hue_range=(70,100), hue_step=5,
                         L_start=0.85, L_end=1.0, L_step=0.005, cusp_L_max=0.975):
    try: M1i, M2i = np.linalg.inv(M1), np.linalg.inv(M2)
    except: return 10.0
    pen = 0.0; n = 0
    for hd in range(hue_range[0], hue_range[1]+1, hue_step):
        h = np.radians(hd); ch, sh = np.cos(h), np.sin(h)
        Ls = np.arange(L_start, L_end+L_step/2, L_step)
        mcs = []
        for L in Ls:
            lo, hi = 0.0, 0.5
            for _ in range(35):
                mid = (lo+hi)/2
                lab = np.array([L, mid*ch, mid*sh])
                lc = M2i@lab; lm = np.sign(lc)*np.abs(lc)**3
                lin = M_XYZ_TO_SRGB@(M1i@lm)
                if np.all(lin >= -0.001) and np.all(lin <= 1.001): lo = mid
                else: hi = mid
            mcs.append(lo)
        mcs = np.array(mcs)
        ci = np.argmax(mcs); cL = Ls[ci]
        if cL > cusp_L_max: pen += (cL - cusp_L_max)**2 * 100
        for i in range(1, len(mcs)):
            if mcs[i] > mcs[i-1] + 1e-5:
                pen += ((mcs[i]-mcs[i-1])/L_step)**2
        n += 1
    return pen / max(n, 1)


def compute_hue_penalty(M1, M2):
    prims = [([1,0,0],0),([1,1,0],60),([0,1,0],120),
             ([0,1,1],180),([0,0,1],240),([1,0,1],300)]
    err = 0.0
    for srgb, exp_h in prims:
        lab = M2 @ signed_cbrt(M1 @ (M_SRGB_TO_XYZ @ srgb_to_linear(np.array(srgb, dtype=float))))
        h = np.degrees(np.arctan2(lab[2], lab[1])) % 360
        dh = h - exp_h
        if dh > 180: dh -= 360
        if dh < -180: dh += 360
        err += dh*dh
    return err / len(prims)


def yellow_L(M1, M2):
    """Get yellow primary's L value."""
    return (M2 @ signed_cbrt(M1 @ YELLOW_XYZ))[0]


def analyze_yellow(M1, M2, label=""):
    """Print yellow boundary analysis."""
    try: M1i, M2i = np.linalg.inv(M1), np.linalg.inv(M2)
    except: print(f"  {label}: SINGULAR"); return
    yL = yellow_L(M1, M2)
    ylab = M2 @ signed_cbrt(M1 @ YELLOW_XYZ)
    yC = np.sqrt(ylab[1]**2 + ylab[2]**2)
    mono = compute_monotonicity(M1, M2)
    cv, _ = compute_cv(M1, M2, _pairs)
    hue = compute_hue_penalty(M1, M2)
    c1, c2 = np.linalg.cond(M1), np.linalg.cond(M2)
    print(f"  {label}: CV={cv*100:.2f}% mono={mono:.4f} hue={hue:.1f}°² "
          f"yL={yL:.3f} yC={yC:.3f} cond=({c1:.1f},{c2:.1f})")

    # Yellow boundary at key L values
    h = np.arctan2(ylab[2], ylab[1])
    ch, sh = np.cos(h), np.sin(h)
    print(f"    Yellow boundary (hue={np.degrees(h):.1f}°):")
    prev_mc = None
    for L in [0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0]:
        lo, hi = 0.0, 0.5
        for _ in range(40):
            mid = (lo+hi)/2
            lab = np.array([L, mid*ch, mid*sh])
            lc = M2i@lab; lm = np.sign(lc)*np.abs(lc)**3
            lin = M_XYZ_TO_SRGB@(M1i@lm)
            if np.all(lin >= -0.001) and np.all(lin <= 1.001): lo = mid
            else: hi = mid
        arrow = ""
        if prev_mc is not None:
            arrow = " ↑" if lo > prev_mc + 0.0005 else " ↓" if lo < prev_mc - 0.0005 else " ="
        prev_mc = lo
        print(f"      L={L:.3f} C={lo:.4f}{arrow}")


# ══════════════════════════════════════════════════════════════════════
# Run a single variant
# ══════════════════════════════════════════════════════════════════════

def run_variant(name, x0, param_to_mat, mat_to_param, sigma, gens, popsize,
                mono_lam=2.0, hue_lam=0.3, cond_max=15, yellow_L_min=0.0):
    print(f"\n{'='*60}")
    print(f"Variant {name}")
    print(f"{'='*60}")
    print(f"  params={len(x0)}, sigma={sigma}, gens={gens}, pop={popsize}")
    print(f"  mono_lam={mono_lam}, hue_lam={hue_lam}, cond_max={cond_max}, yL_min={yellow_L_min}")

    best_loss = float("inf")
    best_x = x0.copy()
    best_info = {}
    evals = [0]
    t0 = time.time()

    def objective(x):
        try:
            result = param_to_mat(x)
            if isinstance(result, tuple):
                M1, M2 = result
            else:
                M1, M2 = V14_M1, result

            c1, c2 = np.linalg.cond(M1), np.linalg.cond(M2)
            if c1 > 100 or c2 > 100: return 999.0

            cv, cvs = compute_cv(M1, M2, _pairs)
            if cv > 50: return 999.0

            top10 = np.mean(sorted(cvs, reverse=True)[:max(1, len(cvs)//10)])
            mono = compute_monotonicity(M1, M2)
            hue = compute_hue_penalty(M1, M2)

            # Conditioning
            cond_pen = 0.0
            if c1 > cond_max: cond_pen += 0.05 * (c1 - cond_max)**2
            if c2 > cond_max: cond_pen += 0.05 * (c2 - cond_max)**2

            # Yellow L constraint
            yL_pen = 0.0
            if yellow_L_min > 0:
                yL = yellow_L(M1, M2)
                if yL < yellow_L_min:
                    yL_pen = (yellow_L_min - yL)**2 * 500

            loss = cv + 0.3*top10 + cond_pen + yL_pen + mono_lam*mono + hue_lam*hue

        except Exception:
            return 999.0

        evals[0] += 1
        if loss < best_loss:
            best_info.update(cv=cv, mono=mono, hue=hue, c1=c1, c2=c2,
                             yL=yellow_L(M1, M2) if yellow_L_min > 0 else 0)
            if evals[0] % 100 < 5 or loss < best_loss * 0.95:
                el = time.time() - t0
                print(f"    #{evals[0]:>5d} [{el:5.0f}s] loss={loss:.4f} "
                      f"CV={cv*100:.1f}% mono={mono:.4f} hue={hue:.1f}°²")
        return loss

    # Run CMA-ES
    opts = cma.CMAOptions()
    opts.set("maxiter", gens)
    opts.set("popsize", popsize)
    opts.set("tolfun", 1e-9)
    opts.set("verbose", -1)
    es = cma.CMAEvolutionStrategy(x0, sigma, opts)

    while not es.stop():
        sols = es.ask()
        fits = [objective(x) for x in sols]
        # Track best
        min_idx = np.argmin(fits)
        if fits[min_idx] < best_loss:
            best_loss = fits[min_idx]
            best_x = sols[min_idx].copy()
        es.tell(sols, fits)

    # Extract final matrices
    result = param_to_mat(best_x)
    if isinstance(result, tuple):
        M1_best, M2_best = result
    else:
        M1_best, M2_best = V14_M1, result

    print(f"\n  Finished: {evals[0]} evals in {time.time()-t0:.0f}s")
    analyze_yellow(M1_best, M2_best, name)

    return M1_best, M2_best, best_loss


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

_pairs = []

def main():
    global _pairs
    _pairs = generate_training_pairs()
    print(f"Training pairs: {len(_pairs)}")

    # Baselines
    print("\n--- Baselines ---")
    analyze_yellow(V14_M1, V14_M2, "v14")

    GENS = 80
    POP = 24

    # ── Variant A: M1 fixed, only M2 ──
    x0_a = m2_matrix_to_params(V14_M2, V14_M1)
    M1_a, M2_a, loss_a = run_variant(
        "A (M2 only)", x0_a,
        lambda x: m2_params_to_matrix(x, V14_M1),
        lambda M2: m2_matrix_to_params(M2, V14_M1),
        sigma=0.05, gens=GENS, popsize=POP,
        mono_lam=2.0, hue_lam=0.3
    )

    # ── Variant B: Full M1+M2 with yellow L > 0.95 ──
    x0_b = full_matrices_to_params(V14_M1, V14_M2)
    M1_b, M2_b, loss_b = run_variant(
        "B (yL>0.95)", x0_b,
        full_params_to_matrices,
        lambda M1, M2: full_matrices_to_params(M1, M2),
        sigma=0.01, gens=GENS, popsize=POP,
        mono_lam=2.0, hue_lam=0.3, yellow_L_min=0.95
    )

    # ── Variant C: Full M1+M2 with cond < 5 ──
    x0_c = full_matrices_to_params(V14_M1, V14_M2)
    M1_c, M2_c, loss_c = run_variant(
        "C (cond<5)", x0_c,
        full_params_to_matrices,
        lambda M1, M2: full_matrices_to_params(M1, M2),
        sigma=0.01, gens=GENS, popsize=POP,
        mono_lam=2.0, hue_lam=0.3, cond_max=5
    )

    # ── Variant D: All combined ──
    x0_d = full_matrices_to_params(V14_M1, V14_M2)
    M1_d, M2_d, loss_d = run_variant(
        "D (combined)", x0_d,
        full_params_to_matrices,
        lambda M1, M2: full_matrices_to_params(M1, M2),
        sigma=0.008, gens=GENS*2, popsize=POP,
        mono_lam=2.0, hue_lam=0.3, cond_max=5, yellow_L_min=0.95
    )

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for label, M1, M2 in [("v14", V14_M1, V14_M2),
                           ("A", M1_a, M2_a),
                           ("B", M1_b, M2_b),
                           ("C", M1_c, M2_c),
                           ("D", M1_d, M2_d)]:
        analyze_yellow(M1, M2, label)

    # Save best
    best_label = "D"
    M1_save, M2_save = M1_d, M2_d
    ckpt = {
        "version": "v24-D",
        "M1": M1_save.tolist(), "M2": M2_save.tolist(),
        "M1_inv": np.linalg.inv(M1_save).tolist(),
        "M2_inv": np.linalg.inv(M2_save).tolist(),
    }
    p = Path("checkpoints/gen_v24_variants.json")
    p.parent.mkdir(exist_ok=True)
    with open(p, "w") as f: json.dump(ckpt, f, indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
