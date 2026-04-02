"""V3 M2 Optimization — Target: chroma amp <3x + achromatic <1e-14 + 360 cusps.

Pipeline: M1(prod) → depcubic(0.02) → M2*(6 DOF) → Lab
No enrichment, no PW, no Newton.

M2 construction:
- L-row: analytically from achromatic constraint M2[0] @ depcubic(M1 @ D65) = 1, a=b=0
- Ab-rows: 6 free parameters (2×3 matrix entries)

CMA-ES objective: minimize chroma_amp subject to:
- achromatic < 1e-12 (analytic, should be ~1e-15)
- cusps = 360
- Blue G/R ≥ 1.50
"""

import numpy as np
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from helmlab.utils.srgb_convert import sRGB_to_XYZ

D65 = np.array([0.95047, 1.0, 1.08883])
XYZ_TO_SRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
])

# Production M1
prod = json.loads((Path(__file__).parent.parent / "src/helmlab/data/gen_params.json").read_text())
M1 = np.array(prod["M1"])
M1_INV = np.linalg.inv(M1)
ALPHA = 0.02

OKLAB_M2 = np.array([
    [0.2104542553,  0.793617785, -0.0040720468],
    [1.9779984951, -2.428592205,  0.4505937099],
    [0.0259040371, 0.7827717662, -0.808675766]
])

PRIMARIES = [[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,0,1],[1,1,0]]


def depcubic_fwd(x, alpha):
    s = np.sqrt(alpha / 3.0)
    t = x / (2.0 * s**3)
    y = 2.0 * s * np.sinh(np.arcsinh(t) / 3.0)
    f = y**3 + alpha * y - x
    fp = 3.0 * y**2 + alpha
    fpp = 6.0 * y
    d = 2.0 * fp * fp - f * fpp
    mask = np.abs(d) > 1e-30
    y = np.where(mask, y - 2.0 * f * fp / np.where(mask, d, 1.0), y)
    return y


def depcubic_inv(y, alpha):
    return y**3 + alpha * y


def build_M2(ab_params):
    """Build M2 from 6 ab-row parameters + analytical L-row.

    ab_params: [a1,a2,a3, b1,b2,b3] — raw ab-row entries
    L-row: from achromatic constraint M2[0] @ w_c = 1, and M2[1:] @ w_c = 0
    """
    lms_w = M1 @ D65
    w_c = depcubic_fwd(lms_w, ALPHA)

    # OKLab L-row as template, normalized
    L_row = OKLAB_M2[0].copy()
    L_row = L_row / (L_row @ w_c)  # L(white) = 1

    a_row = np.array(ab_params[:3])
    b_row = np.array(ab_params[3:])

    # Orthogonalize ab-rows to achromatic axis (a=b=0 for grays)
    a_row = a_row - (a_row @ w_c) * L_row
    b_row = b_row - (b_row @ w_c) * L_row

    return np.array([L_row, a_row, b_row])


def forward(xyz, M2):
    lms = M1 @ xyz
    lms_c = depcubic_fwd(lms, ALPHA)
    return M2 @ lms_c


def inverse(lab, M2_inv):
    lms_c = M2_inv @ lab
    lms = depcubic_inv(lms_c, ALPHA)
    return M1_INV @ lms


def measure_chroma_amp(M2):
    """Max Jacobian norm at 6 primaries — proxy for chroma amplification."""
    max_amp = 0
    for rgb in PRIMARIES:
        xyz = sRGB_to_XYZ(np.array(rgb, dtype=float))
        lab = forward(xyz, M2)
        for dx in [0.001, -0.001]:
            for dim in range(3):
                rgb2 = np.array(rgb, dtype=float)
                rgb2[dim] = np.clip(rgb2[dim] + dx, 0, 1)
                xyz2 = sRGB_to_XYZ(rgb2)
                lab2 = forward(xyz2, M2)
                dlab = np.sqrt(np.sum((lab - lab2)**2))
                dxyz = np.sqrt(np.sum((xyz - xyz2)**2))
                if dxyz > 1e-10:
                    max_amp = max(max_amp, dlab / dxyz)
    return max_amp


def measure_achromatic(M2):
    max_err = 0
    for v in np.linspace(0.01, 1.0, 100):
        xyz = sRGB_to_XYZ(np.array([v, v, v]))
        lab = forward(xyz, M2)
        max_err = max(max_err, abs(lab[1]) + abs(lab[2]))
    return max_err


def count_cusps(M2, M2_inv, n_hues=360):
    valid = 0
    for hue_deg in range(n_hues):
        h = np.radians(hue_deg)
        ch, sh = np.cos(h), np.sin(h)
        best_C, best_L = 0, 0
        for L in np.arange(0.02, 0.98, 0.02):
            lo, hi = 0.0, 0.5
            for _ in range(16):
                mid = (lo + hi) / 2
                lab = np.array([L, mid * ch, mid * sh])
                xyz = inverse(lab, M2_inv)
                rgb = XYZ_TO_SRGB @ xyz
                if np.all(rgb >= -0.001) and np.all(rgb <= 1.001):
                    lo = mid
                else:
                    hi = mid
            if lo > best_C:
                best_C, best_L = lo, L
        if 0.05 < best_L < 0.99:
            valid += 1
    return valid


def measure_blue_gr(M2, M2_inv):
    blue_xyz = sRGB_to_XYZ(np.array([0.0, 0.0, 1.0]))
    white_xyz = sRGB_to_XYZ(np.array([1.0, 1.0, 1.0]))
    lab_b = forward(blue_xyz, M2)
    lab_w = forward(white_xyz, M2)
    lab_mid = (lab_b + lab_w) / 2
    xyz_mid = inverse(lab_mid, M2_inv)
    rgb_mid = np.clip(XYZ_TO_SRGB @ xyz_mid, 0, 1)
    return rgb_mid[1] / max(rgb_mid[0], 1e-10)


def measure_round_trip(M2, M2_inv, n=5000):
    np.random.seed(42)
    max_err = 0
    for _ in range(n):
        rgb = np.random.rand(3)
        xyz = sRGB_to_XYZ(rgb)
        lab = forward(xyz, M2)
        xyz2 = inverse(lab, M2_inv)
        err = np.max(np.abs(xyz - xyz2))
        max_err = max(max_err, err)
    return max_err


def objective(ab_params):
    """Feasibility-gated objective for CMA-ES."""
    M2 = build_M2(ab_params)

    try:
        M2_inv = np.linalg.inv(M2)
    except:
        return 1e10

    # Hard gate: matrix not degenerate
    if np.linalg.cond(M2) > 100:
        return 1e10

    # Hard gate: primary chromas > 0.05
    min_chroma = 999
    for rgb in PRIMARIES:
        xyz = sRGB_to_XYZ(np.array(rgb, dtype=float))
        lab = forward(xyz, M2)
        C = np.sqrt(lab[1]**2 + lab[2]**2)
        min_chroma = min(min_chroma, C)
    if min_chroma < 0.05:
        return 1e10

    # Hard gate: coarse cusps
    cusps_coarse = count_cusps(M2, M2_inv, n_hues=36)
    if cusps_coarse < 34:
        return 1e10

    # Hard gate: Blue G/R ≥ 1.40
    bgr = measure_blue_gr(M2, M2_inv)
    if bgr < 1.40:
        return 1e10

    # Soft objective: minimize chroma amp (primary goal)
    amp = measure_chroma_amp(M2)

    # Achromatic should be ~0 by construction, but check
    ach = measure_achromatic(M2)

    return amp + ach * 1e6  # amp is ~3-6, ach should be ~1e-14


def main():
    print("V3 M2 Optimization — Target: chroma amp <3x")
    print(f"M1: production (perturbed OKLab)")
    print(f"Transfer: depcubic, alpha={ALPHA}")
    print(f"DOF: 6 (M2 ab-rows, L-row from achromatic constraint)")
    print()

    # Starting point: OKLab M2 ab-rows
    x0 = np.concatenate([OKLAB_M2[1], OKLAB_M2[2]])
    print("Baseline (OKLab M2 ab-rows):")
    M2_0 = build_M2(x0)
    M2_0_inv = np.linalg.inv(M2_0)
    amp_0 = measure_chroma_amp(M2_0)
    ach_0 = measure_achromatic(M2_0)
    bgr_0 = measure_blue_gr(M2_0, M2_0_inv)
    rt_0 = measure_round_trip(M2_0, M2_0_inv)
    cusps_0 = count_cusps(M2_0, M2_0_inv, 36)
    print(f"  Chroma amp: {amp_0:.2f}x  Ach: {ach_0:.2e}  G/R: {bgr_0:.2f}  RT: {rt_0:.2e}  Cusps/36: {cusps_0}")
    print()

    # Also test production M2
    prod_ab = np.concatenate([np.array(prod["M2"])[1], np.array(prod["M2"])[2]])
    print("Production M2 ab-rows:")
    M2_p = build_M2(prod_ab)
    M2_p_inv = np.linalg.inv(M2_p)
    amp_p = measure_chroma_amp(M2_p)
    ach_p = measure_achromatic(M2_p)
    bgr_p = measure_blue_gr(M2_p, M2_p_inv)
    rt_p = measure_round_trip(M2_p, M2_p_inv)
    cusps_p = count_cusps(M2_p, M2_p_inv, 36)
    print(f"  Chroma amp: {amp_p:.2f}x  Ach: {ach_p:.2e}  G/R: {bgr_p:.2f}  RT: {rt_p:.2e}  Cusps/36: {cusps_p}")
    print()

    # CMA-ES
    try:
        import cma
    except ImportError:
        print("pip install cma")
        sys.exit(1)

    # Start from OKLab ab-rows (known good basin)
    sigma = 0.3
    es = cma.CMAEvolutionStrategy(x0, sigma, {
        'maxiter': 200,
        'popsize': 30,
        'seed': 42,
        'verbose': -1,
    })

    best_score = objective(x0)
    best_params = x0.copy()
    best_amp = amp_0
    gen = 0
    t0 = time.time()

    while not es.stop():
        solutions = es.ask()
        values = [objective(x) for x in solutions]
        es.tell(solutions, values)
        gen += 1

        best_idx = np.argmin(values)
        if values[best_idx] < best_score:
            best_score = values[best_idx]
            best_params = solutions[best_idx].copy()
            M2_best = build_M2(best_params)
            best_amp = measure_chroma_amp(M2_best)
            elapsed = time.time() - t0
            print(f"Gen {gen:4d} [{elapsed:6.1f}s] amp={best_amp:.3f}x score={best_score:.4f}", flush=True)

        if gen % 50 == 0:
            elapsed = time.time() - t0
            print(f"Gen {gen:4d} [{elapsed:6.1f}s] best_amp={best_amp:.3f}x", flush=True)

    # Final evaluation
    M2_final = build_M2(best_params)
    M2_final_inv = np.linalg.inv(M2_final)
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    amp = measure_chroma_amp(M2_final)
    ach = measure_achromatic(M2_final)
    bgr = measure_blue_gr(M2_final, M2_final_inv)
    rt = measure_round_trip(M2_final, M2_final_inv)
    cusps = count_cusps(M2_final, M2_final_inv, 360)

    print(f"  Chroma amp:  {amp:.3f}x  {'PASS' if amp < 3.0 else 'FAIL'} (target: <3x)")
    print(f"  Achromatic:  {ach:.2e}  {'PASS' if ach < 1e-10 else 'FAIL'} (target: <1e-10)")
    print(f"  Blue G/R:    {bgr:.3f}  {'PASS' if bgr >= 1.50 else 'FAIL'} (target: ≥1.50)")
    print(f"  Round-trip:  {rt:.2e}  {'PASS' if rt < 1e-14 else 'FAIL'} (target: <1e-14)")
    print(f"  Cusps:       {cusps}/360  {'PASS' if cusps == 360 else 'FAIL'}")
    print(f"\n  M2:")
    for row in M2_final:
        print(f"    [{row[0]:.12f}, {row[1]:.12f}, {row[2]:.12f}]")

    # Save checkpoint
    out = {
        "M1": M1.tolist(),
        "M2": M2_final.tolist(),
        "transfer": "depcubic",
        "depcubic_alpha": ALPHA,
        "_v3_metrics": {
            "chroma_amp": float(amp),
            "achromatic": float(ach),
            "blue_gr": float(bgr),
            "round_trip": float(rt),
            "cusps": cusps,
        }
    }
    outpath = Path(__file__).parent / "checkpoints" / "v3_m2opt.json"
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {outpath}")


if __name__ == "__main__":
    main()
