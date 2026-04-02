"""HelmGen Next — Depressed Cubic Transfer Prototype.

Phase 1: OKLab M1 + Depressed Cubic + OKLab M2 baseline.
Scan α ∈ [0.005, 0.05], evaluate via ColorBench.
"""

import numpy as np
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from helmlab.utils.srgb_convert import sRGB_to_XYZ, XYZ_to_sRGB

# OKLab M1
M1 = np.array([
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715, 0.0361456387],
    [0.0482003018, 0.2643662691, 0.633851707]
])
M1_INV = np.linalg.inv(M1)

# OKLab M2 (starting point)
OK_M2 = np.array([
    [0.2104542553, 0.793617785, -0.0040720468],
    [1.9779984951, -2.428592205, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.808675766]
])

D65 = np.array([0.95047, 1.0, 1.08883])
XYZ_TO_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252]
])


def depressed_cubic_fwd(x, alpha):
    """Forward: solve y³ + αy = x. sinh/asinh initial + Halley polish."""
    s = np.sqrt(alpha / 3)
    t = x / (2 * s**3)
    y = 2 * s * np.sinh(np.arcsinh(t) / 3)

    # One Halley step: cubic convergence, ~1e-7 → ~1e-16
    f = y**3 + alpha * y - x
    fp = 3 * y**2 + alpha
    fpp = 6 * y
    denom = 2 * fp * fp - f * fpp
    if isinstance(denom, np.ndarray):
        safe = np.abs(denom) > 1e-30
        y = np.where(safe, y - 2 * f * fp / np.where(safe, denom, 1.0), y)
    else:
        if abs(denom) > 1e-30:
            y -= 2 * f * fp / denom

    return y


def depressed_cubic_inv(y, alpha):
    """Inverse: trivially exact."""
    return y**3 + alpha * y


def normalize_M2(M2, M1, alpha):
    """Normalize M2 so that white → L=1, a=0, b=0."""
    lms_w = M1 @ D65
    lms_c_w = depressed_cubic_fwd(lms_w, alpha)

    M2n = M2.copy()
    M2n[0] = M2[0] / (M2[0] @ lms_c_w)
    M2n[1] = M2[1] - (M2[1] @ lms_c_w) * M2n[0]
    M2n[2] = M2[2] - (M2[2] @ lms_c_w) * M2n[0]
    return M2n


def forward(xyz, M2, alpha):
    """Full forward: XYZ → Lab."""
    lms = M1 @ xyz
    lms_c = depressed_cubic_fwd(lms, alpha)
    return M2 @ lms_c


def inverse(lab, M2, M2_inv, alpha):
    """Full inverse: Lab → XYZ."""
    lms_c = M2_inv @ lab
    lms = depressed_cubic_inv(lms_c, alpha)
    return M1_INV @ lms


def measure_round_trip(M2, M2_inv, alpha, n=10000):
    """Measure round-trip precision in XYZ (not sRGB, to avoid gamma bottleneck)."""
    np.random.seed(42)
    rgbs = np.random.rand(n, 3)
    max_err = 0
    for rgb in rgbs:
        xyz = sRGB_to_XYZ(rgb)
        lab = forward(xyz, M2, alpha)
        xyz2 = inverse(lab, M2, M2_inv, alpha)
        err = np.max(np.abs(xyz - xyz2))
        max_err = max(max_err, err)
    return max_err


def measure_achromatic(M2, alpha):
    """Measure achromatic axis error."""
    max_err = 0
    for v in np.linspace(0.01, 1.0, 100):
        xyz = sRGB_to_XYZ(np.array([v, v, v]))
        lab = forward(xyz, M2, alpha)
        max_err = max(max_err, abs(lab[1]) + abs(lab[2]))
    return max_err


def count_gamut_holes(M2, M2_inv, alpha, n_hues=120):
    """Count interior gamut holes."""
    total = 0
    for hue_deg in range(0, 360, 360 // n_hues):
        h = np.radians(hue_deg)
        ch, sh = np.cos(h), np.sin(h)
        for L in np.arange(0.005, 0.5, 0.003):
            for C in np.arange(0.005, 0.35, 0.003):
                lab = np.array([L, C * ch, C * sh])
                xyz = inverse(lab, M2, M2_inv, alpha)
                rgb = XYZ_TO_SRGB @ xyz
                ok = np.all(rgb >= -0.0005) and np.all(rgb <= 1.0005)
                if not ok:
                    lab2 = np.array([L, (C + 0.005) * ch, (C + 0.005) * sh])
                    xyz2 = inverse(lab2, M2, M2_inv, alpha)
                    rgb2 = XYZ_TO_SRGB @ xyz2
                    ok2 = np.all(rgb2 >= -0.0005) and np.all(rgb2 <= 1.0005)
                    if ok2:
                        total += 1
    return total


def count_cusps(M2, M2_inv, alpha, gamut_mat=None):
    """Count valid cusps (cusp_L in [0.05, 0.99])."""
    if gamut_mat is None:
        gamut_mat = XYZ_TO_SRGB
    gamut_inv = np.linalg.inv(gamut_mat) if gamut_mat is not XYZ_TO_SRGB else None

    valid = 0
    for hue_deg in range(360):
        h = np.radians(hue_deg)
        ch, sh = np.cos(h), np.sin(h)
        best_C = 0
        best_L = 0
        for L in np.arange(0.02, 0.98, 0.005):
            lo, hi = 0.0, 0.5
            for _ in range(30):
                mid = (lo + hi) / 2
                lab = np.array([L, mid * ch, mid * sh])
                xyz = inverse(lab, M2, M2_inv, alpha)
                rgb = XYZ_TO_SRGB @ xyz
                if np.all(rgb >= -0.001) and np.all(rgb <= 1.001):
                    lo = mid
                else:
                    hi = mid
            if lo > best_C:
                best_C = lo
                best_L = L
        if best_L > 0.05 and best_L < 0.99:
            valid += 1
    return valid


def scan_alpha():
    """Scan α values and report metrics."""
    print(f"{'α':>8s} | {'RT':>10s} | {'Ach':>10s} | {'Holes':>6s} | {'Cusps':>6s} | Note")
    print("-" * 65)

    results = []
    for alpha in [0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
        M2 = normalize_M2(OK_M2.copy(), M1, alpha)
        M2_inv = np.linalg.inv(M2)

        rt = measure_round_trip(M2, M2_inv, alpha, n=5000)
        ach = measure_achromatic(M2, alpha)
        holes = count_gamut_holes(M2, M2_inv, alpha, n_hues=36)
        cusps = count_cusps(M2, M2_inv, alpha)

        note = ""
        if rt < 1e-14 and holes == 0 and cusps == 360:
            note = " ★ PERFECT"
        elif rt < 1e-14 and cusps == 360:
            note = " ✓ good"

        print(f"{alpha:>8.3f} | {rt:>10.2e} | {ach:>10.2e} | {holes:>6d} | {cusps:>6d} |{note}", flush=True)

        results.append({
            "alpha": alpha,
            "round_trip": rt,
            "achromatic": ach,
            "holes": holes,
            "cusps": cusps,
            "M2": M2.tolist()
        })

    # Save best
    best = min(results, key=lambda r: (r["holes"], -r["cusps"], r["round_trip"]))
    print(f"\nBest: α={best['alpha']}, RT={best['round_trip']:.2e}, holes={best['holes']}, cusps={best['cusps']}")

    out = {
        "M1": M1.tolist(),
        "M2": best["M2"],
        "transfer": "depcubic",
        "depcubic_alpha": best["alpha"],
        "gamma": [1/3, 1/3, 1/3],
    }

    outpath = Path(__file__).parent / "checkpoints"
    outpath.mkdir(exist_ok=True)
    with open(outpath / "depcubic_best.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: helmgen-next/checkpoints/depcubic_best.json")

    return results


if __name__ == "__main__":
    print("HelmGen Next — Phase 1: Depressed Cubic α Scan")
    print(f"M1: OKLab (fixed)")
    print(f"M2: OKLab (normalized per α)")
    print(f"Transfer: y³ + αy = x (sinh/asinh form)")
    print()
    scan_alpha()
