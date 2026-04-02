"""HelmGen Next — M2 optimization with feasibility-gated structured search.

Fixes three bugs from optimize_gamut_convex.py:
1. Uses depressed cubic (not softcbrt) — matches depcubic_best.json checkpoint
2. Structured 3-DOF parameterization: R(θ) @ diag(exp(s1), exp(s2)) @ OKLab_M2_ab
3. Feasibility-gated objective with chroma floor, cusp check, hue RMS

Usage:
    python helmgen-next/optimize_m2.py
    python helmgen-next/optimize_m2.py --alpha 0.015 --maxiter 300
"""

import numpy as np
import json
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from helmlab.utils.srgb_convert import sRGB_to_XYZ, XYZ_to_sRGB

# ─── Constants ──────────────────────────────────────────────────────

D65 = np.array([0.95047, 1.0, 1.08883])
XYZ_TO_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [0.0556434, -0.2040259,  1.0572252]
])

# OKLab M1 (fixed)
M1 = np.array([
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715,  0.0361456387],
    [0.0482003018, 0.2643662691,  0.633851707]
])
M1_INV = np.linalg.inv(M1)

# OKLab M2 (baseline for structured transform)
OKLAB_M2 = np.array([
    [0.2104542553,  0.793617785, -0.0040720468],
    [1.9779984951, -2.428592205,  0.4505937099],
    [0.0259040371, 0.7827717662, -0.808675766]
])

# Primary sRGB colors for chroma checks
PRIMARIES_RGB = [
    [1, 0, 0],  # Red
    [0, 1, 0],  # Green
    [0, 0, 1],  # Blue
    [0, 1, 1],  # Cyan
    [1, 0, 1],  # Magenta
    [1, 1, 0],  # Yellow
]


# ─── Transfer function: Depressed Cubic ────────────────────────────

def depcubic_fwd(x, alpha):
    """Forward: solve y³ + αy = x via sinh/asinh + Halley polish."""
    s = np.sqrt(alpha / 3.0)
    t = x / (2.0 * s**3)
    y = 2.0 * s * np.sinh(np.arcsinh(t) / 3.0)

    # One Halley step for machine-epsilon precision
    f = y**3 + alpha * y - x
    fp = 3.0 * y**2 + alpha
    fpp = 6.0 * y
    denom = 2.0 * fp * fp - f * fpp
    mask = np.abs(denom) > 1e-30
    y = np.where(mask, y - 2.0 * f * fp / np.where(mask, denom, 1.0), y)
    return y


def depcubic_inv(y, alpha):
    """Inverse: trivially exact."""
    return y**3 + alpha * y


# ─── M2 construction ───────────────────────────────────────────────

def build_M2_structured(params, alpha):
    """Build M2 from 3 structured parameters: [θ, s1, s2].

    M2_ab = R(θ) @ diag(exp(s1), exp(s2)) @ OKLab_M2_ab
    L-row from achromatic constraint: M2[0] @ LMS_C_W = 1, a=b=0.
    """
    theta, s1, s2 = params

    # Rotation matrix in ab-plane
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[ct, -st], [st, ct]])

    # Scale matrix (exp prevents collapse: s→-∞ required for zero)
    S = np.diag([np.exp(s1), np.exp(s2)])

    # Transform OKLab ab-rows
    oklab_ab = OKLAB_M2[1:3, :]  # 2×3
    new_ab = R @ S @ oklab_ab     # 2×3

    # L-row from achromatic constraint
    lms_w = M1 @ D65
    lms_c_w = depcubic_fwd(lms_w, alpha)

    L_row = OKLAB_M2[0].copy()
    L_row = L_row / (L_row @ lms_c_w)  # normalize: L(white) = 1

    # Orthogonalize ab-rows to achromatic axis
    a_row = new_ab[0] - (new_ab[0] @ lms_c_w) * L_row
    b_row = new_ab[1] - (new_ab[1] @ lms_c_w) * L_row

    return np.array([L_row, a_row, b_row])


# ─── Measurement functions ─────────────────────────────────────────

def forward(xyz, M2, alpha):
    lms = M1 @ xyz
    lms_c = depcubic_fwd(lms, alpha)
    return M2 @ lms_c


def inverse(lab, M2_inv, alpha):
    lms_c = M2_inv @ lab
    lms = depcubic_inv(lms_c, alpha)
    return M1_INV @ lms


def measure_primary_chromas(M2, alpha):
    """Measure chroma of 6 primary sRGB colors. Returns min chroma."""
    chromas = []
    for rgb in PRIMARIES_RGB:
        xyz = sRGB_to_XYZ(np.array(rgb, dtype=float))
        lab = forward(xyz, M2, alpha)
        chromas.append(np.sqrt(lab[1]**2 + lab[2]**2))
    return chromas, min(chromas)


def measure_achromatic(M2, alpha, n=50):
    """Max |a| + |b| across gray ramp."""
    max_err = 0
    for v in np.linspace(0.01, 1.0, n):
        xyz = sRGB_to_XYZ(np.array([v, v, v]))
        lab = forward(xyz, M2, alpha)
        max_err = max(max_err, abs(lab[1]) + abs(lab[2]))
    return max_err


def count_cusps(M2, M2_inv, alpha, n_hues=360):
    """Count valid cusps (cusp_L in [0.05, 0.99])."""
    valid = 0
    for hue_deg in range(n_hues):
        h = np.radians(hue_deg * (360 / n_hues))
        ch, sh = np.cos(h), np.sin(h)
        best_C = 0
        best_L = 0
        for L in np.arange(0.02, 0.98, 0.01):
            lo, hi = 0.0, 0.5
            for _ in range(20):
                mid = (lo + hi) / 2
                lab = np.array([L, mid * ch, mid * sh])
                xyz = inverse(lab, M2_inv, alpha)
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


def count_cusps_coarse(M2, M2_inv, alpha):
    """Fast coarse cusp check — 36 hues."""
    return count_cusps(M2, M2_inv, alpha, n_hues=36)


def count_gamut_holes(M2, M2_inv, alpha, n_hues=60, L_step=0.005, C_step=0.005):
    """Count interior gamut holes."""
    total = 0
    for hue_deg in range(0, 360, 360 // n_hues):
        h = np.radians(hue_deg)
        ch, sh = np.cos(h), np.sin(h)
        for L in np.arange(0.005, 0.5, L_step):
            for C in np.arange(0.005, 0.35, C_step):
                lab = np.array([L, C * ch, C * sh])
                xyz = inverse(lab, M2_inv, alpha)
                rgb = XYZ_TO_SRGB @ xyz
                ok = np.all(rgb >= -0.0005) and np.all(rgb <= 1.0005)
                if not ok:
                    lab2 = np.array([L, (C + 0.005) * ch, (C + 0.005) * sh])
                    xyz2 = inverse(lab2, M2_inv, alpha)
                    rgb2 = XYZ_TO_SRGB @ xyz2
                    ok2 = np.all(rgb2 >= -0.0005) and np.all(rgb2 <= 1.0005)
                    if ok2:
                        total += 1
    return total


def measure_hue_rms(M2, alpha, n=50):
    """Hue RMS error: how much do hues of primary→white gradients drift?"""
    hue_errors = []
    for rgb_primary in PRIMARIES_RGB:
        primary = np.array(rgb_primary, dtype=float)
        white = np.array([1.0, 1.0, 1.0])

        lab_p = forward(sRGB_to_XYZ(primary), M2, alpha)
        h_start = np.degrees(np.arctan2(lab_p[2], lab_p[1])) % 360

        for t in np.linspace(0.1, 0.9, n):
            rgb_t = primary * (1 - t) + white * t
            xyz_t = sRGB_to_XYZ(rgb_t)
            lab_t = forward(xyz_t, M2, alpha)

            C_t = np.sqrt(lab_t[1]**2 + lab_t[2]**2)
            if C_t > 0.005:  # skip near-achromatic
                h_t = np.degrees(np.arctan2(lab_t[2], lab_t[1])) % 360
                dh = abs(h_t - h_start)
                if dh > 180:
                    dh = 360 - dh
                hue_errors.append(dh)

    if not hue_errors:
        return 180.0  # worst case
    return np.sqrt(np.mean(np.array(hue_errors)**2))


def measure_round_trip(M2, M2_inv, alpha, n=5000):
    """Round-trip precision."""
    np.random.seed(42)
    rgbs = np.random.rand(n, 3)
    max_err = 0
    for rgb in rgbs:
        xyz = sRGB_to_XYZ(rgb)
        lab = forward(xyz, M2, alpha)
        xyz2 = inverse(lab, M2_inv, alpha)
        err = np.max(np.abs(xyz - xyz2))
        max_err = max(max_err, err)
    return max_err


def measure_blue_gr(M2, alpha):
    """Blue→White midpoint G/R ratio. Target: ≥1.50 (sky blue, not purple)."""
    blue_xyz = sRGB_to_XYZ(np.array([0.0, 0.0, 1.0]))
    white_xyz = sRGB_to_XYZ(np.array([1.0, 1.0, 1.0]))

    lab_b = forward(blue_xyz, M2, alpha)
    lab_w = forward(white_xyz, M2, alpha)
    lab_mid = (lab_b + lab_w) / 2

    lms_c_mid = np.linalg.inv(M2) @ lab_mid
    lms_mid = depcubic_inv(lms_c_mid, alpha)
    xyz_mid = M1_INV @ lms_mid
    rgb_mid = XYZ_TO_SRGB @ xyz_mid

    # Clip for display
    rgb_clipped = np.clip(rgb_mid, 0, 1)
    if rgb_clipped[0] > 0.001:
        return rgb_clipped[1] / rgb_clipped[0]
    return 99.0  # very blue


# ─── Objective function ────────────────────────────────────────────

def objective(params, alpha, verbose=False):
    """Feasibility-gated multi-criterion objective."""
    M2 = build_M2_structured(params, alpha)

    # === HARD GATE 1: matrix invertibility ===
    try:
        M2_inv = np.linalg.inv(M2)
    except np.linalg.LinAlgError:
        return 1e10

    # === HARD GATE 2: chroma floor (prevents ab-row collapse) ===
    chromas, min_chroma = measure_primary_chromas(M2, alpha)
    if min_chroma < 0.05:
        if verbose:
            print(f"  REJECT: min_chroma={min_chroma:.4f}")
        return 1e10

    # === HARD GATE 3: coarse cusp check ===
    cusps_coarse = count_cusps_coarse(M2, M2_inv, alpha)
    if cusps_coarse < 30:  # out of 36
        if verbose:
            print(f"  REJECT: cusps_coarse={cusps_coarse}/36")
        return 1e10

    # === SOFT OBJECTIVE (only reached if feasible) ===
    holes = count_gamut_holes(M2, M2_inv, alpha, n_hues=36, L_step=0.008, C_step=0.008)
    cusps = count_cusps(M2, M2_inv, alpha, n_hues=360)
    ach = measure_achromatic(M2, alpha)
    hue_rms = measure_hue_rms(M2, alpha, n=20)

    score = (
        holes * 500                              # P0: zero holes
        + max(0, 360 - cusps) * 100              # P1: 360 cusps
        + ach * 1e5                               # P2: achromatic < 1e-8
        + max(0, hue_rms - 10) * 20              # P3: hue fidelity
        - min_chroma * 50                         # P4: reward higher chroma
    )

    if verbose:
        print(f"  holes={holes}, cusps={cusps}, ach={ach:.2e}, "
              f"hue_rms={hue_rms:.1f}°, min_C={min_chroma:.3f}, score={score:.1f}")

    return score


# ─── Main optimization ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HelmGen Next M2 Optimization")
    parser.add_argument("--alpha", type=float, default=0.015)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.3)
    args = parser.parse_args()

    alpha = args.alpha

    print("HelmGen Next — Phase 2: Structured M2 Optimization")
    print(f"Transfer: depressed cubic, α={alpha}")
    print(f"Parameterization: R(θ) @ diag(exp(s1), exp(s2)) @ OKLab_M2_ab")
    print(f"DOF: 3 (θ, s1, s2)")
    print()

    # Starting point: identity transform (θ=0, s1=0, s2=0 → OKLab M2 exactly)
    x0 = np.array([0.0, 0.0, 0.0])

    # Evaluate baseline
    print("Baseline (OKLab M2):")
    baseline_score = objective(x0, alpha, verbose=True)
    print(f"  → score={baseline_score:.1f}")
    print()

    # CMA-ES
    try:
        import cma
    except ImportError:
        print("pip install cma")
        sys.exit(1)

    # Bounds: θ ∈ [-60°, 60°] = [-1.05, 1.05] rad, s1/s2 ∈ [-0.7, 0.7]
    bounds = [[-1.05, -0.7, -0.7], [1.05, 0.7, 0.7]]

    es = cma.CMAEvolutionStrategy(x0, args.sigma, {
        'maxiter': args.maxiter,
        'popsize': args.popsize,
        'seed': 42,
        'verbose': -1,
        'bounds': bounds,
    })

    best_score = baseline_score
    best_params = x0.copy()
    best_M2 = build_M2_structured(x0, alpha)
    gen = 0

    t0 = time.time()

    while not es.stop():
        solutions = es.ask()
        values = [objective(x, alpha) for x in solutions]
        es.tell(solutions, values)

        gen += 1
        best_idx = np.argmin(values)
        if values[best_idx] < best_score:
            best_score = values[best_idx]
            best_params = solutions[best_idx].copy()
            best_M2 = build_M2_structured(best_params, alpha)

            M2_inv = np.linalg.inv(best_M2)
            elapsed = time.time() - t0

            print(f"Gen {gen:4d} [{elapsed:6.1f}s] NEW BEST: score={best_score:.1f} "
                  f"θ={np.degrees(best_params[0]):.1f}° "
                  f"s=({best_params[1]:.3f}, {best_params[2]:.3f})", flush=True)

            # Save checkpoint
            save_checkpoint(best_M2, alpha, best_params, best_score)

        if gen % 50 == 0:
            elapsed = time.time() - t0
            print(f"Gen {gen:4d} [{elapsed:6.1f}s] best_score={best_score:.1f}", flush=True)

    # Final detailed evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    M2_final = build_M2_structured(best_params, alpha)
    M2_inv = np.linalg.inv(M2_final)

    # Fine-grained metrics
    holes_fine = count_gamut_holes(M2_final, M2_inv, alpha, n_hues=120, L_step=0.003, C_step=0.003)
    cusps = count_cusps(M2_final, M2_inv, alpha, n_hues=360)
    ach = measure_achromatic(M2_final, alpha, n=100)
    hue_rms = measure_hue_rms(M2_final, alpha, n=50)
    rt = measure_round_trip(M2_final, M2_inv, alpha, n=10000)
    chromas, min_c = measure_primary_chromas(M2_final, alpha)
    blue_gr = measure_blue_gr(M2_final, alpha)

    print(f"  Params:     θ={np.degrees(best_params[0]):.2f}°, "
          f"s1={best_params[1]:.4f}, s2={best_params[2]:.4f}")
    print(f"  Holes:      {holes_fine}")
    print(f"  Cusps:      {cusps}/360")
    print(f"  Achromatic: {ach:.2e}")
    print(f"  Hue RMS:    {hue_rms:.1f}°")
    print(f"  Round-trip: {rt:.2e}")
    print(f"  Min chroma: {min_c:.3f}")
    print(f"  Blue G/R:   {blue_gr:.2f}")
    print(f"  Chromas:    R={chromas[0]:.3f} G={chromas[1]:.3f} B={chromas[2]:.3f} "
          f"C={chromas[3]:.3f} M={chromas[4]:.3f} Y={chromas[5]:.3f}")
    print(f"\n  M2:")
    for row in M2_final:
        print(f"    [{row[0]:.12f}, {row[1]:.12f}, {row[2]:.12f}]")

    # Goal check
    print(f"\n  GOAL CHECK:")
    print(f"    Holes = 0:     {'PASS' if holes_fine == 0 else 'FAIL'} ({holes_fine})")
    print(f"    Cusps = 360:   {'PASS' if cusps == 360 else 'FAIL'} ({cusps})")
    print(f"    Ach < 1e-8:    {'PASS' if ach < 1e-8 else 'FAIL'} ({ach:.2e})")
    print(f"    Hue RMS < 15:  {'PASS' if hue_rms < 15 else 'FAIL'} ({hue_rms:.1f}°)")
    print(f"    Chroma > 0.2:  {'PASS' if min_c > 0.2 else 'NOTE'} ({min_c:.3f})")
    print(f"    Grad CV < 38:  (run ColorBench)")
    print(f"    Blue G/R ≥1.5: {'PASS' if blue_gr >= 1.5 else 'FAIL'} ({blue_gr:.2f})")
    print(f"    RT < 1e-14:    {'PASS' if rt < 1e-14 else 'FAIL'} ({rt:.2e})")

    print(f"\n  Checkpoint: helmgen-next/checkpoints/depcubic_m2_structured.json")
    print(f"  Next: cd colorbench && python run.py oklab helmct --json ../helmgen-next/checkpoints/depcubic_m2_structured.json")


def save_checkpoint(M2, alpha, params, score):
    outdir = Path(__file__).parent / "checkpoints"
    outdir.mkdir(exist_ok=True)

    out = {
        "M1": M1.tolist(),
        "M2": M2.tolist(),
        "transfer": "depcubic",
        "depcubic_alpha": alpha,
        "gamma": [1/3, 1/3, 1/3],
        "_opt_params": {
            "theta_deg": float(np.degrees(params[0])),
            "s1": float(params[1]),
            "s2": float(params[2]),
            "score": float(score),
        }
    }

    with open(outdir / "depcubic_m2_structured.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
