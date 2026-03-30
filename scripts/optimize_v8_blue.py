#!/usr/bin/env python3
"""v8 GenSpace — CMA-ES optimization with Blue→White sky-blue constraint.

Pipeline: XYZ → M1 → cbrt → M2 → L_corr → Lab
21 DOF: M1(9) + M2(9) + L_corr(3)
Hard constraint: Blue→White midpoint G/R ≥ 1.20

Usage:
    python scripts/optimize_v8_blue.py [--seed v7b|oklab|random] [--gens 300] [--pop 64]
    python scripts/optimize_v8_blue.py --all  # Run all 3 seeds sequentially
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import torch
    HAS_TORCH = True
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
except ImportError:
    HAS_TORCH = False
    DEVICE = None

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════
SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])
XYZ_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ)

# v7b parameters (current best)
V7B_M1 = np.array([
    [6.213663274448127, -0.5041794153770129, -0.40416891025666857],
    [-1.1592256796157883, 4.350194381717271, 0.5254938968299478],
    [0.0008170122534259527, 0.7226718820884986, 2.227799849833172]
])
V7B_M2 = np.array([
    [0.4675499211910323, 0.20915320090703618, -0.08488334505679182],
    [0.4843952725673558, -0.3665958307304812, -0.17266206907852755],
    [-0.04418360083197623, 0.39383739736845824, -0.36863136176600936]
])
V7B_LCORR = np.array([-0.09792777021381058, -0.26695959819582816, 0.30350768100715936])

# OKLab parameters
OK_M1 = np.array([
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715, 0.0361456387],
    [0.0482003018, 0.2643662691, 0.6338517070]
])
OK_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660]
])

# Test colors
BLUE = np.array([0.0, 0.0, 1.0])
WHITE = np.array([1.0, 1.0, 1.0])
RED = np.array([1.0, 0.0, 0.0])
GREEN = np.array([0.0, 1.0, 0.0])
CYAN = np.array([0.0, 1.0, 1.0])
MAGENTA = np.array([1.0, 0.0, 1.0])
YELLOW = np.array([1.0, 1.0, 0.0])
PRIMARIES = [RED, YELLOW, GREEN, CYAN, BLUE, MAGENTA]
PRIMARY_NAMES = ["Red", "Yellow", "Green", "Cyan", "Blue", "Magenta"]

# ═══════════════════════════════════════════════════════════════
# sRGB utilities (numpy)
# ═══════════════════════════════════════════════════════════════
def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(c):
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1/2.4) - 0.055)

def srgb_to_xyz(rgb):
    lin = srgb_to_linear(np.asarray(rgb, dtype=np.float64))
    return lin @ SRGB_TO_XYZ.T

def xyz_to_srgb(xyz):
    lin = np.asarray(xyz, dtype=np.float64) @ XYZ_TO_SRGB.T
    return np.clip(linear_to_srgb(lin), 0, 1)

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*(int(round(np.clip(v,0,1)*255)) for v in rgb))

# ═══════════════════════════════════════════════════════════════
# GenSpace forward/inverse (numpy, for validation)
# ═══════════════════════════════════════════════════════════════
def gen_forward(xyz, M1, M2, lc):
    """XYZ → GenSpace Lab."""
    xyz = np.asarray(xyz, dtype=np.float64)
    lms = xyz @ M1.T
    lms_c = np.cbrt(np.maximum(lms, 0))
    lab = lms_c @ M2.T
    L = lab[..., 0]
    t = L * (1 - L)
    lab[..., 0] = L + lc[0]*t + lc[1]*t*(0.5-L) + lc[2]*t*t
    return lab

def gen_inverse(lab, M1, M2, lc):
    """GenSpace Lab → XYZ."""
    lab = np.asarray(lab, dtype=np.float64).copy()
    M1_inv = np.linalg.inv(M1)
    M2_inv = np.linalg.inv(M2)
    # Undo L_corr (Newton)
    L = lab[..., 0].copy()
    L0 = L.copy()
    for _ in range(20):
        t = L * (1 - L)
        dt = 1 - 2*L
        f = L + lc[0]*t + lc[1]*t*(0.5-L) + lc[2]*t*t - L0
        df = 1 + lc[0]*dt + lc[1]*(dt*(0.5-L)-t) + lc[2]*2*t*dt
        df = np.where(np.abs(df) < 1e-12, 1, df)
        L = L - f / df
    lab[..., 0] = L
    lms_c = lab @ M2_inv.T
    lms = np.maximum(lms_c, 0) ** 3
    return lms @ M1_inv.T

# ═══════════════════════════════════════════════════════════════
# Pack/Unpack 21 DOF
# ═══════════════════════════════════════════════════════════════
def pack(M1, M2, lc):
    return np.concatenate([M1.ravel(), M2.ravel(), lc])

def unpack(x):
    x = np.asarray(x, dtype=np.float64)
    M1 = x[:9].reshape(3, 3)
    M2 = x[9:18].reshape(3, 3)
    lc = x[18:21]
    return M1, M2, lc

# ═══════════════════════════════════════════════════════════════
# Gradient CV computation (batched numpy)
# ═══════════════════════════════════════════════════════════════
def compute_gradient_cv(M1, M2, lc, pairs, n_steps=32):
    """Compute mean gradient CV for a set of sRGB pairs."""
    cvs = []
    M1_inv = np.linalg.inv(M1)
    M2_inv = np.linalg.inv(M2)

    for rgb1, rgb2 in pairs:
        xyz1 = srgb_to_xyz(rgb1)
        xyz2 = srgb_to_xyz(rgb2)
        lab1 = gen_forward(xyz1, M1, M2, lc)
        lab2 = gen_forward(xyz2, M1, M2, lc)

        # Interpolate in Lab
        steps_lab = np.array([lab1 + (lab2 - lab1) * t/(n_steps-1) for t in range(n_steps)])

        # Convert back to sRGB
        steps_xyz = gen_inverse(steps_lab, M1, M2, lc)
        steps_srgb = np.clip(xyz_to_srgb(steps_xyz), 0, 1)

        # Compute step-to-step dE in sRGB (simplified Euclidean in linear sRGB)
        steps_lin = srgb_to_linear(steps_srgb)
        diffs = np.sqrt(np.sum((steps_lin[1:] - steps_lin[:-1])**2, axis=-1))

        if np.mean(diffs) > 1e-10:
            cv = np.std(diffs) / np.mean(diffs)
            cvs.append(cv)

    return np.mean(cvs) if cvs else 1.0

def compute_hue_rms(M1, M2, lc, n_steps=16):
    """Compute hue linearity RMS for 6 primaries → White."""
    deviations = []
    for prim in PRIMARIES:
        xyz_p = srgb_to_xyz(prim)
        xyz_w = srgb_to_xyz(WHITE)
        lab_p = gen_forward(xyz_p, M1, M2, lc)
        lab_w = gen_forward(xyz_w, M1, M2, lc)

        h_start = np.arctan2(lab_p[1], lab_p[0]) if lab_p[0]**2 + lab_p[1]**2 > 1e-20 else 0
        h_end = np.arctan2(lab_w[1], lab_w[0]) if lab_w[0]**2 + lab_w[1]**2 > 1e-20 else h_start

        # Use a,b channels for hue tracking
        a_start, b_start = lab_p[1], lab_p[2]
        a_end, b_end = lab_w[1], lab_w[2]

        max_dev = 0
        for i in range(1, n_steps - 1):
            t = i / (n_steps - 1)
            lab_t = lab_p + (lab_w - lab_p) * t
            xyz_t = gen_inverse(lab_t, M1, M2, lc)
            srgb_t = np.clip(xyz_to_srgb(xyz_t), 0, 1)
            lab_back = gen_forward(srgb_to_xyz(srgb_t), M1, M2, lc)

            a_t, b_t = lab_back[1], lab_back[2]
            # Expected hue angle at this t
            a_exp = a_start + (a_end - a_start) * t
            b_exp = b_start + (b_end - b_start) * t

            h_actual = np.arctan2(b_t, a_t)
            h_expected = np.arctan2(b_exp, a_exp)

            dh = h_actual - h_expected
            while dh > np.pi: dh -= 2*np.pi
            while dh < -np.pi: dh += 2*np.pi

            max_dev = max(max_dev, abs(np.degrees(dh)))

        deviations.append(max_dev)

    return np.sqrt(np.mean(np.array(deviations)**2))

def compute_blue_white_gr(M1, M2, lc):
    """Blue→White midpoint G/R ratio."""
    lab_b = gen_forward(srgb_to_xyz(BLUE), M1, M2, lc)
    lab_w = gen_forward(srgb_to_xyz(WHITE), M1, M2, lc)
    lab_mid = (lab_b + lab_w) / 2
    xyz_mid = gen_inverse(lab_mid, M1, M2, lc)
    srgb_mid = np.clip(xyz_to_srgb(xyz_mid), 0, 1)
    gr = srgb_mid[1] / max(srgb_mid[0], 1e-10)
    return gr, srgb_mid

def compute_red_white_gb(M1, M2, lc):
    """Red→White midpoint G-B difference."""
    lab_r = gen_forward(srgb_to_xyz(RED), M1, M2, lc)
    lab_w = gen_forward(srgb_to_xyz(WHITE), M1, M2, lc)
    lab_mid = (lab_r + lab_w) / 2
    xyz_mid = gen_inverse(lab_mid, M1, M2, lc)
    srgb_mid = np.clip(xyz_to_srgb(xyz_mid), 0, 1)
    return srgb_mid[1] - srgb_mid[2], srgb_mid

def compute_achromatic(M1, M2, lc, n=20):
    """Max |a|, |b| on gray ramp."""
    max_ab = 0
    for i in range(n):
        gray = (i + 1) / (n + 1)
        xyz = srgb_to_xyz(np.array([gray, gray, gray]))
        lab = gen_forward(xyz, M1, M2, lc)
        max_ab = max(max_ab, abs(lab[1]), abs(lab[2]))
    return max_ab

def compute_roundtrip(M1, M2, lc, n=200):
    """Max round-trip error on random sRGB colors."""
    rng = np.random.default_rng(42)
    max_err = 0
    for _ in range(n):
        srgb = rng.uniform(0, 1, 3)
        xyz = srgb_to_xyz(srgb)
        lab = gen_forward(xyz, M1, M2, lc)
        xyz_back = gen_inverse(lab, M1, M2, lc)
        err = np.max(np.abs(xyz - xyz_back))
        max_err = max(max_err, err)
    return max_err

# ═══════════════════════════════════════════════════════════════
# Generate random gradient pairs
# ═══════════════════════════════════════════════════════════════
def generate_pairs(n=200, seed=42):
    """Generate random sRGB color pairs for CV measurement."""
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        rgb1 = rng.uniform(0, 1, 3)
        rgb2 = rng.uniform(0, 1, 3)
        # Ensure reasonable distance
        if np.sum((rgb1 - rgb2)**2) > 0.1:
            pairs.append((rgb1, rgb2))
    return pairs[:n]

# ═══════════════════════════════════════════════════════════════
# Objective function
# ═══════════════════════════════════════════════════════════════
PAIRS_CACHE = None

def objective(x, pairs=None, verbose=False):
    """CMA-ES objective: lower is better."""
    M1, M2, lc = unpack(x)

    # Sanity checks
    try:
        det1 = np.linalg.det(M1)
        det2 = np.linalg.det(M2)
        if abs(det1) < 0.01 or abs(det2) < 0.001:
            return 1e6
        cond1 = np.linalg.cond(M1)
        cond2 = np.linalg.cond(M2)
        if cond1 > 50 or cond2 > 50:
            return 1e6
    except:
        return 1e6

    # Blue→White G/R — HARD CONSTRAINT
    try:
        blue_gr, blue_mid = compute_blue_white_gr(M1, M2, lc)
    except:
        return 1e6

    if blue_gr < 1.15:  # hard reject below 1.15
        return 1e6

    # Red→White G-B
    try:
        red_gb, red_mid = compute_red_white_gb(M1, M2, lc)
    except:
        return 1e6

    # Achromatic
    ach = compute_achromatic(M1, M2, lc)
    if ach > 0.01:  # reject if achromatic is badly broken
        return 1e6

    # Round-trip check (quick, 50 colors)
    rt = compute_roundtrip(M1, M2, lc, n=50)
    if rt > 1e-4:  # reject if inverse is broken
        return 1e6

    # Gradient CV (main metric)
    if pairs is None:
        global PAIRS_CACHE
        if PAIRS_CACHE is None:
            PAIRS_CACHE = generate_pairs(150, seed=42)
        pairs = PAIRS_CACHE

    cv = compute_gradient_cv(M1, M2, lc, pairs, n_steps=24)

    # Hue linearity
    hue_rms = compute_hue_rms(M1, M2, lc, n_steps=12)

    # Compose loss
    loss = 0.0

    # CV (main, weight=1.0)
    loss += cv

    # Hue linearity (weight=0.3)
    loss += 0.3 * (hue_rms / 30.0)  # normalize: 30° → 1.0

    # Blue G/R soft bonus (encourage > 1.25)
    if blue_gr < 1.25:
        loss += 2.0 * (1.25 - blue_gr) ** 2

    # Red G-B penalty (keep orange shift small)
    if abs(red_gb) > 0.10:
        loss += 1.0 * (abs(red_gb) - 0.10) ** 2

    # Condition number regularization
    if cond1 > 5:
        loss += 0.1 * (cond1 - 5) ** 2
    if cond2 > 15:
        loss += 0.05 * (cond2 - 15) ** 2

    # Achromatic penalty
    if ach > 1e-6:
        loss += 0.5 * np.log10(ach / 1e-6)

    if verbose:
        print(f"  CV={cv:.4f} hue={hue_rms:.1f}° B-W G/R={blue_gr:.3f} "
              f"R-W G-B={red_gb:.3f} ach={ach:.2e} cond=({cond1:.1f},{cond2:.1f}) "
              f"RT={rt:.2e} loss={loss:.4f}")

    return loss

# ═══════════════════════════════════════════════════════════════
# Seeds
# ═══════════════════════════════════════════════════════════════
def get_seed(name):
    if name == "v7b":
        return pack(V7B_M1, V7B_M2, V7B_LCORR), 0.03
    elif name == "oklab":
        # Start from OKLab matrices (scaled for our pipeline)
        return pack(OK_M1, OK_M2, np.zeros(3)), 0.1
    elif name == "random":
        # Random perturbation of v7b
        rng = np.random.default_rng(123)
        x0 = pack(V7B_M1, V7B_M2, V7B_LCORR)
        x0 += rng.normal(0, 0.05, len(x0))
        return x0, 0.08
    elif name == "random2":
        rng = np.random.default_rng(456)
        x0 = pack(V7B_M1, V7B_M2, V7B_LCORR)
        x0 += rng.normal(0, 0.08, len(x0))
        return x0, 0.1
    elif name == "random3":
        rng = np.random.default_rng(789)
        x0 = pack(V7B_M1, V7B_M2, V7B_LCORR)
        x0 += rng.normal(0, 0.03, len(x0))
        return x0, 0.05
    else:
        raise ValueError(f"Unknown seed: {name}")

# ═══════════════════════════════════════════════════════════════
# CMA-ES optimization
# ═══════════════════════════════════════════════════════════════
def run_cmaes(seed_name, n_gens=300, popsize=64):
    print(f"\n{'='*70}")
    print(f"  CMA-ES Optimization — Seed: {seed_name}")
    print(f"  DOF: 21, Population: {popsize}, Generations: {n_gens}")
    print(f"{'='*70}")

    x0, sigma0 = get_seed(seed_name)

    # Evaluate initial point
    print(f"\n  Initial point ({seed_name}):")
    loss0 = objective(x0, verbose=True)

    if not HAS_CMA:
        print("  ERROR: cma package not installed. Run: pip install cma")
        return None, None

    # CMA-ES options
    opts = cma.CMAOptions()
    opts['popsize'] = popsize
    opts['maxiter'] = n_gens
    opts['verb_disp'] = 0  # quiet
    opts['verb_log'] = 0
    opts['tolfun'] = 1e-8
    opts['tolx'] = 1e-10
    opts['seed'] = hash(seed_name) % (2**31)

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_loss = 1e9
    best_x = None
    gen = 0
    t0 = time.time()

    while not es.stop() and gen < n_gens:
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)

        # Track best
        idx = np.argmin(fitnesses)
        if fitnesses[idx] < best_loss:
            best_loss = fitnesses[idx]
            best_x = solutions[idx].copy()

        gen += 1
        if gen % 20 == 0 or gen == 1:
            elapsed = time.time() - t0
            rate = gen / elapsed if elapsed > 0 else 0
            M1, M2, lc = unpack(best_x)
            try:
                gr, _ = compute_blue_white_gr(M1, M2, lc)
                cv = compute_gradient_cv(M1, M2, lc, PAIRS_CACHE[:30], n_steps=16)
                print(f"  Gen {gen:4d} | loss={best_loss:.5f} CV={cv:.4f} "
                      f"B-W G/R={gr:.3f} | {rate:.1f} gen/s")
            except:
                print(f"  Gen {gen:4d} | loss={best_loss:.5f} | {rate:.1f} gen/s")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s ({gen} generations)")

    if best_x is not None:
        print(f"\n  Best solution:")
        objective(best_x, verbose=True)

    return best_x, best_loss

# ═══════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════
def full_report(x, label=""):
    M1, M2, lc = unpack(x)

    print(f"\n{'='*70}")
    print(f"  FULL REPORT: {label}")
    print(f"{'='*70}")

    # Blue→White
    gr, blue_mid = compute_blue_white_gr(M1, M2, lc)
    print(f"  Blue→White midpoint: {rgb_to_hex(blue_mid)} G/R={gr:.3f}")

    # Red→White
    gb, red_mid = compute_red_white_gb(M1, M2, lc)
    print(f"  Red→White  midpoint: {rgb_to_hex(red_mid)} G-B={gb:.3f}")

    # All primary midpoints
    print(f"\n  Primary → White midpoints:")
    for name, prim in zip(PRIMARY_NAMES, PRIMARIES):
        lab_p = gen_forward(srgb_to_xyz(prim), M1, M2, lc)
        lab_w = gen_forward(srgb_to_xyz(WHITE), M1, M2, lc)
        lab_mid = (lab_p + lab_w) / 2
        xyz_mid = gen_inverse(lab_mid, M1, M2, lc)
        srgb_mid = np.clip(xyz_to_srgb(xyz_mid), 0, 1)
        print(f"    {name:8s} → White: {rgb_to_hex(srgb_mid)} "
              f"R={srgb_mid[0]:.3f} G={srgb_mid[1]:.3f} B={srgb_mid[2]:.3f}")

    # Gradient CV (full evaluation)
    pairs = generate_pairs(200, seed=42)
    cv = compute_gradient_cv(M1, M2, lc, pairs, n_steps=32)
    print(f"\n  Gradient CV (200 pairs, 32 steps): {cv:.4f}")

    # Hue RMS
    hue_rms = compute_hue_rms(M1, M2, lc, n_steps=16)
    print(f"  Hue RMS: {hue_rms:.2f}°")

    # Achromatic
    ach = compute_achromatic(M1, M2, lc)
    print(f"  Achromatic max|a,b|: {ach:.2e}")

    # Condition numbers
    cond1 = np.linalg.cond(M1)
    cond2 = np.linalg.cond(M2)
    print(f"  Condition: M1={cond1:.2f} M2={cond2:.2f}")

    # Round-trip
    rt = compute_roundtrip(M1, M2, lc, n=500)
    print(f"  Round-trip max error: {rt:.2e}")

    # Matrices
    print(f"\n  M1 = {M1.tolist()}")
    print(f"  M2 = {M2.tolist()}")
    print(f"  L_corr = {lc.tolist()}")

    return {
        "blue_gr": float(gr), "blue_mid": rgb_to_hex(blue_mid),
        "red_gb": float(gb), "red_mid": rgb_to_hex(red_mid),
        "cv": float(cv), "hue_rms": float(hue_rms),
        "ach": float(ach), "cond_m1": float(cond1), "cond_m2": float(cond2),
        "rt": float(rt),
        "M1": M1.tolist(), "M2": M2.tolist(), "L_corr": lc.tolist()
    }

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="v8 GenSpace optimizer")
    parser.add_argument("--seed", type=str, default="v7b",
                       choices=["v7b", "oklab", "random", "random2", "random3"],
                       help="Starting seed")
    parser.add_argument("--all", action="store_true",
                       help="Run all seeds")
    parser.add_argument("--gens", type=int, default=300,
                       help="CMA-ES generations")
    parser.add_argument("--pop", type=int, default=64,
                       help="CMA-ES population size")
    args = parser.parse_args()

    print(f"v8 GenSpace Optimizer — Blue→White Sky-Blue Constraint")
    print(f"Device: {DEVICE}")
    print(f"CMA: {'available' if HAS_CMA else 'NOT FOUND'}")
    print(f"Torch: {'available' if HAS_TORCH else 'NOT FOUND'}")

    # Generate pairs cache
    global PAIRS_CACHE
    PAIRS_CACHE = generate_pairs(150, seed=42)

    # Baselines
    print(f"\n{'='*70}")
    print(f"  BASELINES")
    print(f"{'='*70}")

    print("\n  v7b:")
    objective(pack(V7B_M1, V7B_M2, V7B_LCORR), verbose=True)

    print("\n  OKLab:")
    objective(pack(OK_M1, OK_M2, np.zeros(3)), verbose=True)

    seeds = ["v7b", "oklab", "random", "random2", "random3"] if args.all else [args.seed]

    results = {}
    for seed_name in seeds:
        best_x, best_loss = run_cmaes(seed_name, n_gens=args.gens, popsize=args.pop)
        if best_x is not None:
            report = full_report(best_x, label=f"Seed={seed_name}")
            results[seed_name] = {"x": best_x.tolist(), "loss": best_loss, "report": report}

    # Find overall best
    if results:
        best_seed = min(results, key=lambda k: results[k]["loss"])
        best_x = np.array(results[best_seed]["x"])
        M1, M2, lc = unpack(best_x)

        print(f"\n{'='*70}")
        print(f"  OVERALL BEST: Seed={best_seed}, loss={results[best_seed]['loss']:.5f}")
        print(f"{'='*70}")

        report = results[best_seed]["report"]
        print(f"  Blue→White G/R: {report['blue_gr']:.3f} ({report['blue_mid']})")
        print(f"  Gradient CV: {report['cv']:.4f}")
        print(f"  Hue RMS: {report['hue_rms']:.2f}°")
        print(f"  Achromatic: {report['ach']:.2e}")

        # Save checkpoint
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = Path("checkpoints") / f"v8_blue_{best_seed}_{ts}.json"
        ckpt_path.parent.mkdir(exist_ok=True)

        ckpt = {
            "version": "v8-blue",
            "seed": best_seed,
            "loss": float(results[best_seed]["loss"]),
            "M1": M1.tolist(),
            "M2": M2.tolist(),
            "gamma": [1/3, 1/3, 1/3],
            "L_corr_p1": float(lc[0]),
            "L_corr_p2": float(lc[1]),
            "L_corr_p3": float(lc[2]),
            "metrics": report
        }

        with open(ckpt_path, "w") as f:
            json.dump(ckpt, f, indent=2)
        print(f"\n  Checkpoint saved: {ckpt_path}")

        # Save gen_params format
        gen_params = {
            "M1": M1.tolist(),
            "gamma": [1/3, 1/3, 1/3],
            "M2": M2.tolist(),
            "hue_cos1": 0, "hue_sin1": 0,
            "hue_cos2": 0, "hue_sin2": 0,
            "hue_cos3": 0, "hue_sin3": 0,
            "hue_cos4": 0, "hue_sin4": 0,
            "L_corr_p1": float(lc[0]),
            "L_corr_p2": float(lc[1]),
            "L_corr_p3": float(lc[2]),
            "lp_dark": 0, "lp_dark_hcos": 0, "lp_dark_hsin": 0,
            "lc1": 0, "lc2": 0,
            "chroma_power": 1.0,
            "hue_L_amp": 0, "hue_L_center": 0,
            "hue_L_width": 1, "hue_L_knee": 1
        }

        gen_params_path = Path("checkpoints") / f"v8_blue_{best_seed}_{ts}_gen_params.json"
        with open(gen_params_path, "w") as f:
            json.dump(gen_params, f, indent=2)
        print(f"  Gen params saved: {gen_params_path}")

if __name__ == "__main__":
    main()
