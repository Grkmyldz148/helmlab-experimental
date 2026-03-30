#!/usr/bin/env python3
"""v24: Fix yellow gamut boundary concavity in GenSpace.

Starting from v14 CMA-ES matrices, adds a gamut boundary monotonicity
penalty to ensure the yellow cusp happens at a reasonable lightness
(< 0.97) and chroma decreases smoothly toward white.

Usage:
    python scripts/optimize_v24_monotonic.py [--generations N] [--sigma S]
"""

import sys
sys.path.insert(0, "src")

import argparse
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

# OKLab reference
OKLAB_M1_SRGB = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
OKLAB_M1 = OKLAB_M1_SRGB @ M_XYZ_TO_SRGB
OKLAB_M2 = np.array([
    [ 0.2104542553,  0.7936177850, -0.0040720468],
    [ 1.9779984951, -2.4285922050,  0.4505937099],
    [ 0.0259040371,  0.7827717662, -0.8086757660],
])

# Current v14 matrices (starting point)
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
    """XYZ → CIE Lab (D65)."""
    r = xyz / D65
    f = np.where(r > 0.008856, r ** (1/3), 7.787 * r + 16/116)
    L = 116 * f[1] - 16
    a = 500 * (f[0] - f[1])
    b = 200 * (f[1] - f[2])
    return np.array([L, a, b])


# ══════════════════════════════════════════════════════════════════════
# Parameterization: 16 core params → M1, M2
# ══════════════════════════════════════════════════════════════════════

def ortho_basis(s):
    """Orthonormal basis perpendicular to vector s."""
    s_norm = s / np.linalg.norm(s)
    if abs(s_norm[0]) < 0.9:
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])
    e1 = v - np.dot(v, s_norm) * s_norm
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s_norm, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def core_params_to_matrices(x):
    """13 params → M1 (D65-normalized) + M2 (achromatic-constrained).

    x[0:6]  = M1 free params (2 per row, 3rd determined by M1@D65=[1,1,1])
    x[6:9]  = M2 L-row
    x[9:11] = M2 a-row (projected onto orthonormal basis)
    x[11:13]= M2 b-row (projected onto orthonormal basis)
    """
    # M1 with D65 normalization: each row dot D65 = 1
    M1 = np.zeros((3, 3))
    for i in range(3):
        M1[i, 0] = x[2*i]
        M1[i, 1] = x[2*i + 1]
        M1[i, 2] = (1.0 - M1[i, 0] * D65[0] - M1[i, 1] * D65[1]) / D65[2]

    # Gray axis
    lms_d65 = M1 @ D65  # should be [1,1,1]
    lms_c_d65 = signed_cbrt(lms_d65)  # should be [1,1,1]
    e1, e2 = ortho_basis(lms_c_d65)

    # M2
    M2 = np.zeros((3, 3))
    M2[0] = x[6:9]
    # Normalize L-row so L(D65)=1
    L_white = M2[0] @ lms_c_d65
    if abs(L_white) < 1e-10:
        raise ValueError("L_white ≈ 0")
    M2[0] = M2[0] / L_white

    M2[1] = x[9] * e1 + x[10] * e2
    M2[2] = x[11] * e1 + x[12] * e2

    return M1, M2


def matrices_to_core_params(M1, M2):
    """Pack M1, M2 into 13 core parameters."""
    x = np.zeros(13)
    # M1: first 2 columns of each row
    for i in range(3):
        x[2*i] = M1[i, 0]
        x[2*i + 1] = M1[i, 1]

    # M2 L-row (unnormalized)
    x[6:9] = M2[0]

    # Gray axis
    lms_d65 = M1 @ D65
    lms_c_d65 = signed_cbrt(lms_d65)
    e1, e2 = ortho_basis(lms_c_d65)

    x[9] = M2[1] @ e1
    x[10] = M2[1] @ e2
    x[11] = M2[2] @ e1
    x[12] = M2[2] @ e2

    return x


# ══════════════════════════════════════════════════════════════════════
# Training pairs (gradient CV evaluation)
# ══════════════════════════════════════════════════════════════════════

def generate_training_pairs(n_random=80, seed=42):
    """Generate diverse sRGB color pairs for gradient evaluation."""
    rng = np.random.RandomState(seed)
    pairs = []

    # Primary-to-primary
    primaries = [
        [1,0,0], [0,1,0], [0,0,1],
        [1,1,0], [0,1,1], [1,0,1],
        [1,1,1], [0,0,0],
    ]
    for i in range(len(primaries)):
        for j in range(i+1, len(primaries)):
            pairs.append((primaries[i], primaries[j]))

    # Achromatic gradients
    for g1 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for g2 in [g1 + 0.2, g1 + 0.4]:
            if g2 <= 1.0:
                pairs.append(([g1]*3, [g2]*3))

    # Random pairs
    for _ in range(n_random):
        c1 = rng.rand(3)
        c2 = rng.rand(3)
        pairs.append((c1.tolist(), c2.tolist()))

    # Convert to XYZ
    xyz_pairs = []
    for c1, c2 in pairs:
        lin1 = srgb_to_linear(np.array(c1))
        lin2 = srgb_to_linear(np.array(c2))
        xyz1 = M_SRGB_TO_XYZ @ lin1
        xyz2 = M_SRGB_TO_XYZ @ lin2
        xyz_pairs.append((xyz1, xyz2))

    return xyz_pairs


def ciede2000_simple(lab1, lab2):
    """Simplified CIEDE2000 (no parametric weights)."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    dL = L2 - L1
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    dC = C2 - C1
    da = a2 - a1
    db = b2 - b1
    dH2 = da**2 + db**2 - dC**2
    dH = np.sqrt(max(0, dH2))
    SL = 1 + 0.015 * (L1 - 50)**2 / np.sqrt(20 + (L1 - 50)**2)
    SC = 1 + 0.045 * C1
    SH = 1 + 0.015 * C1
    return np.sqrt((dL/SL)**2 + (dC/SC)**2 + (dH/SH)**2)


def compute_gradient_cv(M1, M2, xyz_pairs, n_steps=25):
    """Compute gradient uniformity CV across all pairs."""
    M1_inv = np.linalg.inv(M1)
    M2_inv = np.linalg.inv(M2)

    cvs = []
    for xyz1, xyz2 in xyz_pairs:
        # Forward
        lab1 = M2 @ signed_cbrt(M1 @ xyz1)
        lab2 = M2 @ signed_cbrt(M1 @ xyz2)

        # Interpolate in Lab
        ts = np.linspace(0, 1, n_steps + 1)
        labs = np.array([lab1 + t * (lab2 - lab1) for t in ts])

        # Round-trip through 8-bit sRGB
        deltas = []
        prev_cielab = None
        for lab in labs:
            # Inverse: Lab → XYZ
            lms_c = M2_inv @ lab
            lms = np.sign(lms_c) * np.abs(lms_c) ** 3
            xyz = M1_inv @ lms

            # XYZ → sRGB 8-bit → back
            lin = M_XYZ_TO_SRGB @ xyz
            srgb = linear_to_srgb(np.clip(lin, 0, 1))
            srgb8 = np.round(srgb * 255) / 255
            lin_back = srgb_to_linear(srgb8)
            xyz_back = M_SRGB_TO_XYZ @ lin_back

            # CIE Lab
            cielab = xyz_to_cielab(np.maximum(xyz_back, 1e-10))

            if prev_cielab is not None:
                d = ciede2000_simple(prev_cielab, cielab)
                deltas.append(d)
            prev_cielab = cielab

        if deltas:
            arr = np.array(deltas)
            mean_d = np.mean(arr)
            if mean_d > 0.001:
                cv = np.std(arr) / mean_d
                cvs.append(cv)

    if not cvs:
        return 999.0, []
    return np.mean(cvs), cvs


# ══════════════════════════════════════════════════════════════════════
# Gamut boundary monotonicity penalty
# ══════════════════════════════════════════════════════════════════════

def compute_monotonicity_penalty(M1, M2, hue_range=(70, 100), hue_step=5,
                                  L_start=0.85, L_end=1.0, L_step=0.005,
                                  cusp_L_max=0.97):
    """Penalize gamut boundary concavity near white.

    For each hue in the target range:
    1. Find the cusp (max chroma) position
    2. Penalize if cusp L > cusp_L_max
    3. Penalize if chroma increases after L > L_start

    Returns: scalar penalty (0 = no violations).
    """
    try:
        M1_inv = np.linalg.inv(M1)
        M2_inv = np.linalg.inv(M2)
    except np.linalg.LinAlgError:
        return 10.0

    penalty = 0.0
    n_hues = 0

    for hue_deg in range(hue_range[0], hue_range[1] + 1, hue_step):
        hue = np.radians(hue_deg)
        cos_h, sin_h = np.cos(hue), np.sin(hue)

        Ls = np.arange(L_start, L_end + L_step/2, L_step)
        max_chromas = []

        for L in Ls:
            # Binary search for max chroma
            lo, hi = 0.0, 0.5
            for _ in range(35):
                mid = (lo + hi) / 2
                lab = np.array([L, mid * cos_h, mid * sin_h])
                lms_c = M2_inv @ lab
                lms = np.sign(lms_c) * np.abs(lms_c) ** 3
                xyz = M1_inv @ lms
                lin = M_XYZ_TO_SRGB @ xyz
                if np.all(lin >= -0.001) and np.all(lin <= 1.001):
                    lo = mid
                else:
                    hi = mid
            max_chromas.append(lo)

        max_chromas = np.array(max_chromas)

        # Find cusp
        cusp_idx = np.argmax(max_chromas)
        cusp_L = Ls[cusp_idx]

        # Penalty 1: cusp too high
        if cusp_L > cusp_L_max:
            penalty += (cusp_L - cusp_L_max) ** 2 * 100

        # Penalty 2: chroma increasing after L_start
        for i in range(1, len(max_chromas)):
            if max_chromas[i] > max_chromas[i-1] + 1e-5:
                # Positive slope = bad
                slope = (max_chromas[i] - max_chromas[i-1]) / L_step
                penalty += slope ** 2

        n_hues += 1

    return penalty / max(n_hues, 1)


def compute_hue_penalty(M1, M2):
    """Measure hue distortion for sRGB primaries."""
    primaries_srgb = [
        ([1,0,0], 0),    # red
        ([1,1,0], 60),   # yellow
        ([0,1,0], 120),  # green
        ([0,1,1], 180),  # cyan
        ([0,0,1], 240),  # blue
        ([1,0,1], 300),  # magenta
    ]

    total_err = 0.0
    for srgb, expected_hue in primaries_srgb:
        lin = srgb_to_linear(np.array(srgb, dtype=float))
        xyz = M_SRGB_TO_XYZ @ lin
        lab = M2 @ signed_cbrt(M1 @ xyz)
        h = np.degrees(np.arctan2(lab[2], lab[1])) % 360

        # Angular error
        dh = h - expected_hue
        if dh > 180: dh -= 360
        if dh < -180: dh += 360
        total_err += dh * dh

    return total_err / len(primaries_srgb)


def compute_chroma_range_penalty(M1, M2, min_mean_chroma=0.15):
    """Penalize if sRGB primaries have too-low chroma (space is too compressed)."""
    primaries = [
        [1,0,0], [0,1,0], [0,0,1],
        [1,1,0], [0,1,1], [1,0,1],
    ]
    chromas = []
    for srgb in primaries:
        lin = srgb_to_linear(np.array(srgb, dtype=float))
        xyz = M_SRGB_TO_XYZ @ lin
        lab = M2 @ signed_cbrt(M1 @ xyz)
        C = np.sqrt(lab[1]**2 + lab[2]**2)
        chromas.append(C)

    mean_C = np.mean(chromas)
    if mean_C < min_mean_chroma:
        return (min_mean_chroma - mean_C) ** 2 * 1000
    return 0.0


# ══════════════════════════════════════════════════════════════════════
# CMA-ES Optimization
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="v24: Fix yellow concavity")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--mono-lambda", type=float, default=2.0)
    parser.add_argument("--hue-lambda", type=float, default=0.5)
    parser.add_argument("--popsize", type=int, default=32)
    args = parser.parse_args()

    print("=== v24: GenSpace CMA-ES with monotonicity penalty ===")
    print(f"Starting from v14 matrices")
    print(f"sigma={args.sigma}, mono_lambda={args.mono_lambda}, hue_lambda={args.hue_lambda}")
    print(f"generations={args.generations}, popsize={args.popsize}")
    print()

    # Generate training pairs
    xyz_pairs = generate_training_pairs()
    print(f"Training pairs: {len(xyz_pairs)}")

    # Baseline v14 performance
    v14_cv, _ = compute_gradient_cv(V14_M1, V14_M2, xyz_pairs)
    v14_mono = compute_monotonicity_penalty(V14_M1, V14_M2)
    v14_hue = compute_hue_penalty(V14_M1, V14_M2)
    print(f"v14 baseline: CV={v14_cv*100:.2f}%, mono={v14_mono:.6f}, hue={v14_hue:.2f}°²")

    # OKLab baseline
    ok_cv, _ = compute_gradient_cv(OKLAB_M1, OKLAB_M2, xyz_pairs)
    ok_mono = compute_monotonicity_penalty(OKLAB_M1, OKLAB_M2)
    print(f"OKLab baseline: CV={ok_cv*100:.2f}%, mono={ok_mono:.6f}")
    print()

    # Pack v14 as starting point
    x0 = matrices_to_core_params(V14_M1, V14_M2)

    # Verify round-trip
    M1_check, M2_check = core_params_to_matrices(x0)
    assert np.allclose(M1_check, V14_M1, atol=1e-10), "M1 round-trip failed"
    cv_check, _ = compute_gradient_cv(M1_check, M2_check, xyz_pairs)
    print(f"Param round-trip check: CV={cv_check*100:.2f}% (should match v14)")
    print()

    best_loss = float("inf")
    best_x = x0.copy()
    best_cv = v14_cv
    best_mono = v14_mono
    eval_count = 0
    start_time = time.time()

    def objective(x):
        nonlocal eval_count, best_loss, best_x, best_cv, best_mono
        try:
            M1, M2 = core_params_to_matrices(x)
            if np.linalg.cond(M1) > 50 or np.linalg.cond(M2) > 50:
                return 999.0

            cv, cvs = compute_gradient_cv(M1, M2, xyz_pairs)
            if cv > 50:
                return 999.0

            sorted_cvs = sorted(cvs, reverse=True)
            n_top = max(1, len(cvs) // 10)
            top10 = np.mean(sorted_cvs[:n_top])

            mono = compute_monotonicity_penalty(M1, M2)
            hue = compute_hue_penalty(M1, M2)
            chroma_pen = compute_chroma_range_penalty(M1, M2)

            # Conditioning penalty
            cond_pen = 0.0
            c1, c2 = np.linalg.cond(M1), np.linalg.cond(M2)
            if c1 > 15: cond_pen += 0.01 * (c1 - 15)
            if c2 > 15: cond_pen += 0.01 * (c2 - 15)

            loss = (cv + 0.3 * top10 + cond_pen + chroma_pen
                    + args.mono_lambda * mono
                    + args.hue_lambda * hue)

        except Exception:
            return 999.0

        eval_count += 1
        if loss < best_loss:
            best_loss = loss
            best_x = x.copy()
            best_cv = cv
            best_mono = mono
            elapsed = time.time() - start_time
            print(f"  #{eval_count:>5d} [{elapsed:6.1f}s] loss={loss:.5f}  "
                  f"CV={cv*100:.2f}%  mono={mono:.6f}  hue={hue:.1f}°²  "
                  f"cond=({c1:.1f},{c2:.1f})")

        return loss

    # Run CMA-ES
    print("Starting CMA-ES...")
    opts = cma.CMAOptions()
    opts.set("maxiter", args.generations)
    opts.set("popsize", args.popsize)
    opts.set("tolfun", 1e-8)
    opts.set("verbose", -1)  # quiet

    es = cma.CMAEvolutionStrategy(x0, args.sigma, opts)

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)

    print()
    print("=" * 60)
    print("CMA-ES finished")
    print(f"Total evaluations: {eval_count}")
    print(f"Elapsed: {time.time() - start_time:.1f}s")
    print()

    # Extract best matrices
    M1_best, M2_best = core_params_to_matrices(best_x)
    M1_inv_best = np.linalg.inv(M1_best)
    M2_inv_best = np.linalg.inv(M2_best)

    final_cv, _ = compute_gradient_cv(M1_best, M2_best, xyz_pairs)
    final_mono = compute_monotonicity_penalty(M1_best, M2_best)
    final_hue = compute_hue_penalty(M1_best, M2_best)

    print(f"Final: CV={final_cv*100:.2f}%, mono={final_mono:.6f}, hue={final_hue:.2f}°²")
    print(f"v14:   CV={v14_cv*100:.2f}%, mono={v14_mono:.6f}, hue={v14_hue:.2f}°²")
    print(f"OKLab: CV={ok_cv*100:.2f}%, mono={ok_mono:.6f}")
    print()

    # Print matrices
    print("M1 =")
    for row in M1_best:
        print(f"  [{row[0]:>22.16f}, {row[1]:>22.16f}, {row[2]:>22.16f}],")
    print()
    print("M2 =")
    for row in M2_best:
        print(f"  [{row[0]:>22.16f}, {row[1]:>22.16f}, {row[2]:>22.16f}],")
    print()
    print("M1_inv =")
    for row in M1_inv_best:
        print(f"  [{row[0]:>22.16f}, {row[1]:>22.16f}, {row[2]:>22.16f}],")
    print()
    print("M2_inv =")
    for row in M2_inv_best:
        print(f"  [{row[0]:>22.16f}, {row[1]:>22.16f}, {row[2]:>22.16f}],")

    # Gamut boundary analysis
    print()
    print("=== Yellow gamut boundary (hue 85°) ===")
    hue = np.radians(85)
    cos_h, sin_h = np.cos(hue), np.sin(hue)
    for L in np.arange(0.90, 1.001, 0.005):
        lo, hi = 0.0, 0.5
        for _ in range(40):
            mid = (lo + hi) / 2
            lab = np.array([L, mid * cos_h, mid * sin_h])
            lms_c = M2_inv_best @ lab
            lms = np.sign(lms_c) * np.abs(lms_c) ** 3
            xyz = M1_inv_best @ lms
            lin = M_XYZ_TO_SRGB @ xyz
            if np.all(lin >= -0.001) and np.all(lin <= 1.001):
                lo = mid
            else:
                hi = mid
        print(f"  L={L:.3f}  maxC={lo:.6f}")

    # Save checkpoint
    checkpoint = {
        "version": "v24",
        "M1": M1_best.tolist(),
        "M2": M2_best.tolist(),
        "M1_inv": M1_inv_best.tolist(),
        "M2_inv": M2_inv_best.tolist(),
        "cv": final_cv,
        "mono_penalty": final_mono,
        "hue_penalty": final_hue,
    }

    ckpt_path = Path("checkpoints/gen_v24_monotonic.json")
    ckpt_path.parent.mkdir(exist_ok=True)
    with open(ckpt_path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    print(f"\nCheckpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
