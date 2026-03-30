#!/usr/bin/env python
"""v5 GenSpace optimization — Ottosson-style objective with cusp penalty.

Key innovations vs optimize_gen_space.py:
1. Cusp penalty: scan 36 hues, penalize missing cusps + cliff edges
2. Gamut volume penalty: reward larger Lab gamut coverage
3. L-row free (already in param space, but now cusp-aware)
4. Multiple seeds: OKLab, v4b (best analytic), deployed gen_params.json
5. Soft hue penalty (not hard constraints)

Structure: XYZ → M1 → cbrt → M2 → Lab  (16 params, shared γ=1/3)
"""

import argparse
import colorsys
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

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

# OKLab reference (XYZ-input M1)
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

# sRGB primaries
_PRIMARY_SRGB = np.array([
    [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 1, 1], [0, 0, 1], [1, 0, 1],
], dtype=np.float64)
_TARGET_HUE_RAD = np.array([
    0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3,
])
_PRIMARY_NAMES = ["Red", "Yellow", "Green", "Cyan", "Blue", "Magenta"]

# ══════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════

def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1/2.4) - 0.055)

def signed_cbrt(x):
    return np.sign(x) * np.abs(x) ** (1/3)

def hex_to_xyz(h):
    rgb = np.array([int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]) / 255.0
    return M_SRGB_TO_XYZ @ srgb_to_linear(rgb)

def ortho_basis(s):
    sn = s / (np.linalg.norm(s) + 1e-30)
    if abs(sn[0]) < 0.9:
        v1 = np.array([1, 0, 0]) - sn[0] * sn
    else:
        v1 = np.array([0, 1, 0]) - sn[1] * sn
    v1 /= (np.linalg.norm(v1) + 1e-30)
    v2 = np.cross(sn, v1)
    v2 /= (np.linalg.norm(v2) + 1e-30)
    return v1, v2

_PRIMARY_XYZ = np.array([M_SRGB_TO_XYZ @ srgb_to_linear(c) for c in _PRIMARY_SRGB])

# ══════════════════════════════════════════════════════════════════════
# CIE Lab + CIEDE2000
# ══════════════════════════════════════════════════════════════════════

def xyz_to_cielab(xyz):
    r = xyz / D65
    f = np.where(r > 0.008856, np.cbrt(r), 7.787 * r + 16/116)
    return np.array([116*f[1]-16, 500*(f[0]-f[1]), 200*(f[1]-f[2])])

def ciede2000(lab1, lab2):
    L1, a1, b1 = lab1; L2, a2, b2 = lab2
    C1 = np.sqrt(a1**2 + b1**2); C2 = np.sqrt(a2**2 + b2**2)
    Cb = (C1+C2)/2; Cb7 = Cb**7
    G = 0.5 * (1 - np.sqrt(Cb7 / (Cb7 + 25**7)))
    ap1 = a1*(1+G); ap2 = a2*(1+G)
    Cp1 = np.sqrt(ap1**2 + b1**2); Cp2 = np.sqrt(ap2**2 + b2**2)
    hp1 = np.arctan2(b1, ap1) % (2*np.pi)
    hp2 = np.arctan2(b2, ap2) % (2*np.pi)
    dLp = L2 - L1; dCp = Cp2 - Cp1
    dhp = hp2 - hp1
    if Cp1*Cp2 == 0: dhp = 0
    elif abs(dhp) > np.pi: dhp -= np.sign(dhp) * 2*np.pi
    dHp = 2 * np.sqrt(Cp1*Cp2) * np.sin(dhp/2)
    Lp = (L1+L2)/2; Cp = (Cp1+Cp2)/2
    if Cp1*Cp2 == 0: hp = hp1 + hp2
    elif abs(hp1-hp2) <= np.pi: hp = (hp1+hp2)/2
    else: hp = (hp1+hp2)/2 + (np.pi if hp1+hp2 < 2*np.pi else -np.pi)
    T = (1 - 0.17*np.cos(hp - np.pi/6) + 0.24*np.cos(2*hp)
         + 0.32*np.cos(3*hp + np.pi/30) - 0.20*np.cos(4*hp - 63*np.pi/180))
    Lp50 = (Lp-50)**2
    SL = 1 + 0.015*Lp50/np.sqrt(20+Lp50)
    SC = 1 + 0.045*Cp
    SH = 1 + 0.015*Cp*T
    Cp7 = Cp**7; RC = 2*np.sqrt(Cp7/(Cp7+25**7))
    da = (hp*180/np.pi - 275)/25; dth = 30*np.exp(-(da*da))
    RT = -np.sin(2*dth*np.pi/180)*RC
    r1, r2, r3 = dLp/SL, dCp/SC, dHp/SH
    return np.sqrt(r1**2 + r2**2 + r3**2 + RT*r2*r3)

# ══════════════════════════════════════════════════════════════════════
# Pack / Unpack (16 params: M1(9) + M2_L(3) + M2_ab(4))
# ══════════════════════════════════════════════════════════════════════

def matrices_to_params(M1, M2):
    x = np.zeros(16)
    x[:9] = M1.flatten()
    x[9:12] = M2[0]
    s = signed_cbrt(M1 @ D65)
    v1, v2 = ortho_basis(s)
    x[12] = M2[1] @ v1; x[13] = M2[1] @ v2
    x[14] = M2[2] @ v1; x[15] = M2[2] @ v2
    return x

def params_to_matrices(x):
    M1 = x[:9].reshape(3, 3)
    s = signed_cbrt(M1 @ D65)
    v1, v2 = ortho_basis(s)
    M2 = np.zeros((3, 3))
    M2[0] = x[9:12]
    M2[1] = x[12]*v1 + x[13]*v2
    M2[2] = x[14]*v1 + x[15]*v2
    return M1, M2

# ══════════════════════════════════════════════════════════════════════
# Gamut boundary sampling (vectorized, fast)
# ══════════════════════════════════════════════════════════════════════

def _build_boundary_rgb(n_edge=32):
    """Sample sRGB boundary (6 faces of the RGB cube).

    Returns (N, 3) array of linear RGB values on cube faces.
    """
    t = np.linspace(0, 1, n_edge)
    u, v = np.meshgrid(t, t)
    u, v = u.ravel(), v.ravel()
    zeros = np.zeros_like(u)
    ones = np.ones_like(u)

    faces = [
        np.stack([zeros, u, v], axis=1),  # R=0
        np.stack([ones, u, v], axis=1),   # R=1
        np.stack([u, zeros, v], axis=1),  # G=0
        np.stack([u, ones, v], axis=1),   # G=1
        np.stack([u, v, zeros], axis=1),  # B=0
        np.stack([u, v, ones], axis=1),   # B=1
    ]
    rgb = np.concatenate(faces, axis=0)  # (6*n^2, 3)

    # Deduplicate (edges/corners shared between faces)
    rgb_quant = np.round(rgb * 10000).astype(np.int64)
    _, idx = np.unique(rgb_quant, axis=0, return_index=True)
    return rgb[idx]

# Pre-compute boundary XYZ (shared across all evaluations)
_BOUNDARY_RGB = _build_boundary_rgb(n_edge=40)
_BOUNDARY_LINEAR = srgb_to_linear(_BOUNDARY_RGB)
_BOUNDARY_XYZ = _BOUNDARY_LINEAR @ M_SRGB_TO_XYZ.T  # (N, 3)

N_HUE_BINS = 36  # every 10°
_HUE_BIN_EDGES = np.linspace(-np.pi, np.pi, N_HUE_BINS + 1)


def compute_cusp_profile(M1, M2):
    """Compute cusp chroma at each hue bin.

    Returns (N_HUE_BINS,) array of max chroma per bin.
    Fast: ~1ms for 5000 boundary points.
    """
    # Forward transform all boundary points
    LMS = np.maximum(_BOUNDARY_XYZ @ M1.T, 0.0)
    LMS_c = LMS ** (1/3)
    Lab = LMS_c @ M2.T  # (N, 3)

    a = Lab[:, 1]
    b = Lab[:, 2]
    C = np.sqrt(a**2 + b**2)
    h = np.arctan2(b, a)  # [-π, π]

    # Bin by hue
    bin_idx = np.digitize(h, _HUE_BIN_EDGES) - 1
    bin_idx = np.clip(bin_idx, 0, N_HUE_BINS - 1)

    cusp_C = np.zeros(N_HUE_BINS)
    for i in range(N_HUE_BINS):
        mask = bin_idx == i
        if mask.any():
            cusp_C[i] = C[mask].max()

    return cusp_C


def compute_cusp_penalty(cusp_C, min_C=0.03):
    """Penalize hue bins with chroma below threshold.

    Returns: (penalty_value, n_valid_cusps, min_cusp, cliff_ratio)
    """
    # Missing cusp penalty: quadratic below threshold
    deficit = np.maximum(min_C - cusp_C, 0)
    missing_pen = np.sum(deficit ** 2) * 500.0

    # Cliff penalty: max ratio between adjacent cusps
    # (large ratio = steep cliff = bad)
    cusp_shifted = np.roll(cusp_C, 1)
    both_valid = (cusp_C > 0.01) & (cusp_shifted > 0.01)
    if both_valid.any():
        ratios = cusp_C[both_valid] / cusp_shifted[both_valid]
        ratios = np.where(ratios > 1, 1/ratios, ratios)  # always < 1
        cliff = 1.0 - ratios.min()  # 0 = smooth, 1 = cliff
    else:
        cliff = 1.0

    cliff_pen = max(0, cliff - 0.5) ** 2 * 100.0

    n_valid = int(np.sum(cusp_C > min_C))

    return missing_pen + cliff_pen, n_valid, float(cusp_C.min()), cliff


# ══════════════════════════════════════════════════════════════════════
# Hue penalty
# ══════════════════════════════════════════════════════════════════════

def compute_hue_penalty(M1, M2):
    """Mean squared angular error for 6 sRGB primaries."""
    errors_sq = []
    for xyz, target_h in zip(_PRIMARY_XYZ, _TARGET_HUE_RAD):
        lab = M2 @ signed_cbrt(M1 @ xyz)
        h = np.arctan2(lab[2], lab[1])
        diff = h - target_h
        angular_err = np.arctan2(np.sin(diff), np.cos(diff))
        errors_sq.append(angular_err ** 2)
    return float(np.mean(errors_sq))


def compute_hue_stats(M1, M2):
    """Detailed hue stats for each primary/secondary."""
    H_deg, C_vals, errors = [], [], []
    for xyz, target_h in zip(_PRIMARY_XYZ, _TARGET_HUE_RAD):
        lab = M2 @ signed_cbrt(M1 @ xyz)
        h = np.degrees(np.arctan2(lab[2], lab[1])) % 360
        C = np.sqrt(lab[1]**2 + lab[2]**2)
        target_deg = np.degrees(target_h) % 360
        err = (h - target_deg + 180) % 360 - 180
        H_deg.append(h); C_vals.append(C); errors.append(abs(err))

    errors = np.array(errors)
    return {
        "rms": float(np.sqrt(np.mean(errors**2))),
        "max": float(np.max(errors)),
        "per_color": {
            name: f"H={h:.1f}° C={c:.3f} (err {e:.1f}°)"
            for name, h, c, e in zip(_PRIMARY_NAMES, H_deg, C_vals, errors)
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Gradient CV computation
# ══════════════════════════════════════════════════════════════════════

STEPS = 25

def compute_cv(M1, M2, pairs_xyz):
    """Compute gradient CVs for core pipeline."""
    try:
        M2i = np.linalg.inv(M2)
        M1i = np.linalg.inv(M1)
    except np.linalg.LinAlgError:
        return 999.0, []

    cvs = []
    for xyz1, xyz2 in pairs_xyz:
        try:
            lab1 = M2 @ signed_cbrt(M1 @ xyz1)
            lab2 = M2 @ signed_cbrt(M1 @ xyz2)
            des = []
            prev = None
            for i in range(STEPS):
                t = i / (STEPS - 1)
                lab = lab1 + (lab2 - lab1) * t
                c = M2i @ lab
                lms = np.sign(c) * np.abs(c)**3
                xyz = M1i @ lms
                rgb = linear_to_srgb(M_XYZ_TO_SRGB @ xyz)
                rgb8 = np.clip(np.round(rgb * 255), 0, 255)
                xyz_q = M_SRGB_TO_XYZ @ srgb_to_linear(rgb8 / 255.0)
                clab = xyz_to_cielab(xyz_q)
                if prev is not None:
                    des.append(ciede2000(prev, clab))
                prev = clab
            des = np.array(des)
            m = des.mean()
            cvs.append(des.std() / m if m > 1e-10 else 0)
        except Exception:
            cvs.append(10.0)

    return float(np.mean(cvs)), cvs


# ══════════════════════════════════════════════════════════════════════
# Gamut volume (Lab volume of sRGB cube)
# ══════════════════════════════════════════════════════════════════════

def compute_gamut_volume(M1, M2, n_samples=5000):
    """Approximate Lab volume of sRGB gamut using random sampling.

    Higher volume → better use of Lab space → fewer 8-bit collisions.
    """
    rng = np.random.default_rng(42)
    rgb = rng.uniform(0, 1, (n_samples, 3))
    linear = srgb_to_linear(rgb)
    xyz = linear @ M_SRGB_TO_XYZ.T
    lms = np.maximum(xyz @ M1.T, 0.0)
    lms_c = lms ** (1/3)
    lab = lms_c @ M2.T

    # Approximate volume via convex hull (or just use ranges as proxy)
    L_range = lab[:, 0].max() - lab[:, 0].min()
    a_range = lab[:, 1].max() - lab[:, 1].min()
    b_range = lab[:, 2].max() - lab[:, 2].min()
    return L_range * a_range * b_range


# ══════════════════════════════════════════════════════════════════════
# Training pairs
# ══════════════════════════════════════════════════════════════════════

def generate_training_pairs():
    pairs = []

    presets = [
        ('#ff6b00','#00d4ff'), ('#ff0000','#0000ff'), ('#000000','#ffffff'),
        ('#ff0000','#00ff00'), ('#0000ff','#ffff00'), ('#8000ff','#ff8000'),
        ('#ff0000','#ffffff'), ('#0000ff','#ffffff'), ('#00ff00','#ffffff'),
        ('#000000','#ff0000'), ('#000000','#0000ff'), ('#000000','#00ff00'),
        ('#00ff00','#ff00ff'), ('#ff00ff','#ffff00'), ('#00ffff','#ff0000'),
    ]
    pairs.extend(presets)

    primaries = ['#ff0000','#00ff00','#0000ff','#ffff00','#ff00ff','#00ffff']
    for i in range(len(primaries)):
        for j in range(i+1, len(primaries)):
            p = (primaries[i], primaries[j])
            if p not in pairs and (p[1],p[0]) not in pairs:
                pairs.append(p)

    grays = ['#000000','#333333','#666666','#999999','#cccccc','#ffffff']
    for i in range(len(grays)):
        for j in range(i+1, len(grays)):
            if abs(i-j) >= 2:
                pairs.append((grays[i], grays[j]))

    for s, v in [(1.0, 1.0), (0.7, 0.8), (0.5, 0.6)]:
        for h_start in range(0, 360, 30):
            h_end = (h_start + 60) % 360
            r1,g1,b1 = colorsys.hsv_to_rgb(h_start/360, s, v)
            r2,g2,b2 = colorsys.hsv_to_rgb(h_end/360, s, v)
            h1 = f"#{int(r1*255):02x}{int(g1*255):02x}{int(b1*255):02x}"
            h2 = f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"
            pairs.append((h1, h2))

    for s, v in [(1.0, 1.0), (0.6, 0.8)]:
        for h in range(0, 180, 20):
            r1,g1,b1 = colorsys.hsv_to_rgb(h/360, s, v)
            r2,g2,b2 = colorsys.hsv_to_rgb((h+180)/360, s, v)
            h1 = f"#{int(r1*255):02x}{int(g1*255):02x}{int(b1*255):02x}"
            h2 = f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"
            pairs.append((h1, h2))

    for h in [0, 30, 60, 120, 180, 240, 300]:
        r1,g1,b1 = colorsys.hsv_to_rgb(h/360, 0.8, 0.95)
        r2,g2,b2 = colorsys.hsv_to_rgb(h/360, 0.8, 0.3)
        h1 = f"#{int(r1*255):02x}{int(g1*255):02x}{int(b1*255):02x}"
        h2 = f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"
        pairs.append((h1, h2))

    np.random.seed(42)
    for _ in range(50):
        c1 = np.random.randint(0, 256, 3)
        c2 = np.random.randint(0, 256, 3)
        pairs.append((f"#{c1[0]:02x}{c1[1]:02x}{c1[2]:02x}",
                       f"#{c2[0]:02x}{c2[1]:02x}{c2[2]:02x}"))

    seen = set()
    unique = []
    for a, b in pairs:
        key = tuple(sorted([a.lower(), b.lower()]))
        if key not in seen:
            seen.add(key)
            unique.append((a, b))
    return unique


# ══════════════════════════════════════════════════════════════════════
# Objective function
# ══════════════════════════════════════════════════════════════════════

_eval_count = 0
_best_loss = float("inf")
_best_metrics = {}
_last_print = 0.0


def make_objective(train_xyz, cusp_lambda=5.0, hue_lambda=0.5, vol_lambda=0.0):
    """Ottosson-style objective: CV + cusp + hue + conditioning.

    Args:
        train_xyz: list of (xyz1, xyz2) pairs for CV
        cusp_lambda: weight for cusp quality penalty
        hue_lambda: weight for hue angle penalty (sRGB primaries)
        vol_lambda: weight for gamut volume (negative = reward larger)
    """
    # Pre-compute OKLab gamut volume for normalization
    ok_vol = compute_gamut_volume(OKLAB_M1, OKLAB_M2)

    def objective(x):
        global _eval_count, _best_loss, _best_metrics, _last_print
        try:
            M1, M2 = params_to_matrices(x)

            # Quick reject: ill-conditioned
            c1 = np.linalg.cond(M1)
            c2 = np.linalg.cond(M2)
            if c1 > 50 or c2 > 50:
                return 999.0

            # 1. Gradient CV
            mean_cv, cvs = compute_cv(M1, M2, train_xyz)
            if not cvs or mean_cv > 50:
                return 999.0

            sorted_cvs = sorted(cvs, reverse=True)
            n_top = max(1, len(cvs) // 10)
            top10 = np.mean(sorted_cvs[:n_top])

            # 2. Cusp quality
            cusp_C = compute_cusp_profile(M1, M2)
            cusp_pen, n_cusps, min_cusp, cliff = compute_cusp_penalty(cusp_C)

            # 3. Hue penalty
            hue_pen = compute_hue_penalty(M1, M2)

            # 4. Gamut volume (reward)
            vol = compute_gamut_volume(M1, M2)
            vol_pen = -vol_lambda * (vol / ok_vol - 1.0)  # 0 when same as OKLab

            # 5. Conditioning penalty
            cond_pen = 0.0
            if c1 > 15: cond_pen += 0.005 * (c1 - 15)
            if c2 > 15: cond_pen += 0.005 * (c2 - 15)

            # 6. Yellow chroma penalty (specific concern from v4/v4b)
            # Bin ~85° (yellow) should have decent chroma
            yellow_bin = int((np.radians(85) + np.pi) / (2*np.pi) * N_HUE_BINS) % N_HUE_BINS
            yellow_C = cusp_C[yellow_bin]
            yellow_pen = max(0, 0.10 - yellow_C) ** 2 * 200.0

            loss = (mean_cv + 0.3 * top10 +
                    cusp_lambda * cusp_pen +
                    hue_lambda * hue_pen +
                    vol_pen + cond_pen + yellow_pen)

        except Exception:
            return 999.0

        _eval_count += 1
        if loss < _best_loss:
            _best_loss = loss
            _best_metrics = {
                "cv": mean_cv, "top10": top10,
                "cusps": n_cusps, "min_cusp": min_cusp, "cliff": cliff,
                "hue_mse": hue_pen, "yellow_C": yellow_C,
                "c1": c1, "c2": c2,
            }

        now = time.time()
        if now - _last_print > 10.0:
            _last_print = now
            m = _best_metrics
            print(f"  eval #{_eval_count:>6d}  loss={_best_loss:.5f}  "
                  f"CV={m['cv']*100:.1f}%  cusps={m['cusps']}/36  "
                  f"hue={np.degrees(np.sqrt(m['hue_mse'])):.1f}°  "
                  f"yellow_C={m['yellow_C']:.3f}  cliff={m['cliff']:.2f}  "
                  f"cond={m['c1']:.1f}/{m['c2']:.1f}", flush=True)

        return loss

    return objective


# ══════════════════════════════════════════════════════════════════════
# Seeds
# ══════════════════════════════════════════════════════════════════════

def prepare_seeds():
    seeds = []

    # 1. OKLab
    seeds.append(("OKLab", matrices_to_params(OKLAB_M1, OKLAB_M2)))

    # 2. v4b (best analytic, better gamut)
    v4b_path = Path("checkpoints/analytic_v4b_genspace.json")
    if v4b_path.exists():
        d = json.load(open(v4b_path))
        M1 = np.array(d["M1"]); M2 = np.array(d["M2"])
        seeds.append(("v4b", matrices_to_params(M1, M2)))

    # 3. Deployed gen_params.json
    gen_path = Path("src/helmlab/data/gen_params.json")
    if gen_path.exists():
        d = json.load(open(gen_path))
        M1 = np.array(d["M1"]); M2 = np.array(d["M2"])
        seeds.append(("deployed", matrices_to_params(M1, M2)))

    # 4. Phase1H best (if available)
    p1h_path = Path("checkpoints/gen_1h_best.json")
    if p1h_path.exists():
        d = json.load(open(p1h_path))
        M1 = np.array(d["M1"]); M2 = np.array(d["M2"])
        seeds.append(("Phase1H", matrices_to_params(M1, M2)))

    return seeds


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════

def full_evaluate(label, M1, M2, train_xyz, val_xyz, show_pairs, show_xyz):
    """Full evaluation with all metrics."""
    train_cv, _ = compute_cv(M1, M2, train_xyz)
    val_cv, val_cvs = compute_cv(M1, M2, val_xyz)
    show_cv, show_cvs = compute_cv(M1, M2, show_xyz)
    hue = compute_hue_stats(M1, M2)
    cusp_C = compute_cusp_profile(M1, M2)
    cusp_pen, n_cusps, min_cusp, cliff = compute_cusp_penalty(cusp_C)
    vol = compute_gamut_volume(M1, M2)

    # Achromatic
    s = signed_cbrt(M1 @ D65)
    ach_a = float(M2[1] @ s)
    ach_b = float(M2[2] @ s)

    # Round-trip
    try:
        M2i = np.linalg.inv(M2); M1i = np.linalg.inv(M1)
        rng = np.random.default_rng(42)
        test = rng.uniform(0.05, 0.9, (200, 3))
        rt_max = 0.0
        for xyz in test:
            lab = M2 @ signed_cbrt(M1 @ xyz)
            c = M2i @ lab
            xyz_rt = M1i @ (np.sign(c) * np.abs(c)**3)
            rt_max = max(rt_max, np.max(np.abs(xyz - xyz_rt)))
    except Exception:
        rt_max = 999.0

    # Yellow chroma specifically
    yellow_bin = int((np.radians(85) + np.pi) / (2*np.pi) * N_HUE_BINS) % N_HUE_BINS
    yellow_C = cusp_C[yellow_bin]

    print(f"\n  {label}:")
    print(f"    CV: train={train_cv*100:.2f}% val={val_cv*100:.2f}% show={show_cv*100:.2f}%")
    print(f"    Cusps: {n_cusps}/36 valid  min_C={min_cusp:.4f}  cliff={cliff:.3f}")
    print(f"    Yellow (h~85°): C={yellow_C:.4f}")
    print(f"    Hue: RMS={hue['rms']:.1f}°  max={hue['max']:.1f}°")
    for name, s in hue["per_color"].items():
        print(f"      {name}: {s}")
    print(f"    Ach: a={ach_a:.2e}  b={ach_b:.2e}")
    print(f"    RT: {rt_max:.2e}")
    print(f"    Cond: M1={np.linalg.cond(M1):.1f}  M2={np.linalg.cond(M2):.1f}")
    print(f"    Volume: {vol:.4f}")

    if show_pairs and show_cvs:
        print(f"    Show pairs:")
        for (a, b), cv in zip(show_pairs, show_cvs):
            print(f"      {a} -> {b}: {cv*100:.1f}%")

    return {
        "train_cv": train_cv, "val_cv": val_cv, "show_cv": show_cv,
        "n_cusps": n_cusps, "min_cusp": min_cusp, "cliff": cliff,
        "yellow_C": yellow_C, "hue_rms": hue["rms"],
        "ach_a": ach_a, "ach_b": ach_b, "rt_max": rt_max,
        "cond_M1": np.linalg.cond(M1), "cond_M2": np.linalg.cond(M2),
        "volume": vol,
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="v5 GenSpace optimization — Ottosson-style with cusp penalty")
    parser.add_argument("--output", default="checkpoints/analytic_v5_ottosson.json")
    parser.add_argument("--de-maxiter", type=int, default=150,
                        help="DE iterations per run")
    parser.add_argument("--de-popsize", type=int, default=25)
    parser.add_argument("--de-runs", type=int, default=3,
                        help="DE runs per seed")
    parser.add_argument("--cusp-lambda", type=float, default=5.0,
                        help="Cusp penalty weight")
    parser.add_argument("--hue-lambda", type=float, default=0.5,
                        help="Hue angle penalty weight")
    parser.add_argument("--vol-lambda", type=float, default=0.01,
                        help="Volume reward weight")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep cusp_lambda values for Pareto front")
    args = parser.parse_args()

    print("=" * 70)
    print("v5 GenSpace — Ottosson-style CMA-ES with Cusp Penalty")
    print("=" * 70)
    print(f"  cusp_lambda={args.cusp_lambda}  hue_lambda={args.hue_lambda}  "
          f"vol_lambda={args.vol_lambda}")
    print(f"  DE: popsize={args.de_popsize}, maxiter={args.de_maxiter}, "
          f"runs={args.de_runs}/seed")

    # ── Generate training data ─────────────────────────────────────
    print("\nGenerating training pairs...")
    ALL_PAIRS = generate_training_pairs()
    np.random.seed(123)
    indices = np.random.permutation(len(ALL_PAIRS))
    split = int(len(ALL_PAIRS) * 0.8)
    TRAIN_PAIRS = [ALL_PAIRS[i] for i in indices[:split]]
    VAL_PAIRS = [ALL_PAIRS[i] for i in indices[split:]]
    TRAIN_XYZ = [(hex_to_xyz(a), hex_to_xyz(b)) for a, b in TRAIN_PAIRS]
    VAL_XYZ = [(hex_to_xyz(a), hex_to_xyz(b)) for a, b in VAL_PAIRS]

    SHOW_PAIRS = [
        ('#ff6b00','#00d4ff'), ('#ff0000','#0000ff'), ('#000000','#ffffff'),
        ('#ff0000','#00ff00'), ('#0000ff','#ffff00'), ('#8000ff','#ff8000'),
    ]
    SHOW_XYZ = [(hex_to_xyz(a), hex_to_xyz(b)) for a, b in SHOW_PAIRS]

    print(f"  Total: {len(ALL_PAIRS)}, Train: {len(TRAIN_PAIRS)}, Val: {len(VAL_PAIRS)}")

    # ── Baselines ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINES")
    print("=" * 70)

    ok_metrics = full_evaluate("OKLab", OKLAB_M1, OKLAB_M2,
                               TRAIN_XYZ, VAL_XYZ, SHOW_PAIRS, SHOW_XYZ)

    # v4b
    v4b_path = Path("checkpoints/analytic_v4b_genspace.json")
    if v4b_path.exists():
        d = json.load(open(v4b_path))
        M1 = np.array(d["M1"]); M2 = np.array(d["M2"])
        full_evaluate("v4b (analytic)", M1, M2,
                      TRAIN_XYZ, VAL_XYZ, SHOW_PAIRS, SHOW_XYZ)

    # Deployed
    gen_path = Path("src/helmlab/data/gen_params.json")
    if gen_path.exists():
        d = json.load(open(gen_path))
        M1 = np.array(d["M1"]); M2 = np.array(d["M2"])
        full_evaluate("Deployed (gen_params.json)", M1, M2,
                      TRAIN_XYZ, VAL_XYZ, SHOW_PAIRS, SHOW_XYZ)

    sys.stdout.flush()

    # ── Cusp lambda sweep (Pareto front) ───────────────────────────
    if args.sweep:
        cusp_lambdas = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    else:
        cusp_lambdas = [args.cusp_lambda]

    overall_best_loss = float("inf")
    overall_best_M1 = None
    overall_best_M2 = None
    pareto_results = []

    for cl in cusp_lambdas:
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION — cusp_lambda={cl}")
        print(f"{'='*70}")

        global _eval_count, _best_loss, _best_metrics, _last_print

        objective = make_objective(TRAIN_XYZ,
                                   cusp_lambda=cl,
                                   hue_lambda=args.hue_lambda,
                                   vol_lambda=args.vol_lambda)

        seeds = prepare_seeds()
        run_best_loss = float("inf")
        run_best_x = None

        for seed_name, x0 in seeds:
            # Compute bounds centered on seed
            bounds = []
            cap = 5.0
            for i in range(12):
                half = max(abs(x0[i]) * 0.8, 0.3)
                lo = max(x0[i] - half, -cap)
                hi = min(x0[i] + half, cap)
                if x0[i] < lo: lo = x0[i] - 0.3
                if x0[i] > hi: hi = x0[i] + 0.3
                bounds.append((lo, hi))
            for i in range(12, 16):
                bounds.append((-5.0, 5.0))

            for run in range(args.de_runs):
                _eval_count = 0
                _best_loss = float("inf")
                _best_metrics = {}
                _last_print = 0.0

                seed_val = hash(f"{seed_name}_{cl}_{run}") % 2**31
                print(f"\n  --- DE from {seed_name}, run {run+1}/{args.de_runs} ---")
                sys.stdout.flush()

                t0 = time.time()
                result = differential_evolution(
                    objective, bounds, seed=seed_val,
                    maxiter=args.de_maxiter, popsize=args.de_popsize,
                    tol=1e-8, mutation=(0.5, 1.5), recombination=0.9,
                    x0=x0, disp=False,
                )

                # Local refinement
                result2 = minimize(objective, result.x, method="L-BFGS-B",
                                   bounds=bounds,
                                   options={"maxiter": 1000, "ftol": 1e-12})

                final_loss = min(result.fun, result2.fun)
                final_x = result2.x if result2.fun <= result.fun else result.x

                dt = time.time() - t0
                M1t, M2t = params_to_matrices(final_x)
                cusp_C = compute_cusp_profile(M1t, M2t)
                _, nc, _, clf = compute_cusp_penalty(cusp_C)
                hp = compute_hue_penalty(M1t, M2t)
                cv, _ = compute_cv(M1t, M2t, TRAIN_XYZ[:30])  # quick check

                print(f"    loss={final_loss:.5f}  CV={cv*100:.1f}%  "
                      f"cusps={nc}/36  hue={np.degrees(np.sqrt(hp)):.1f}°  "
                      f"cliff={clf:.2f}  ({dt:.0f}s)")

                if final_loss < run_best_loss:
                    run_best_loss = final_loss
                    run_best_x = final_x.copy()
                    print(f"    ** New best for cusp_lambda={cl}!")

                sys.stdout.flush()

        # Final tight refinement
        if run_best_x is not None:
            print(f"\n  --- Final L-BFGS-B refinement ---")
            _eval_count = 0; _best_loss = float("inf"); _last_print = 0.0
            bounds = []
            for i in range(12):
                half = max(abs(run_best_x[i]) * 0.3, 0.1)
                bounds.append((run_best_x[i] - half, run_best_x[i] + half))
            for i in range(12, 16):
                bounds.append((-5.0, 5.0))

            result = minimize(objective, run_best_x, method="L-BFGS-B",
                              bounds=bounds,
                              options={"maxiter": 3000, "ftol": 1e-14})
            if result.fun < run_best_loss:
                run_best_x = result.x.copy()
                run_best_loss = result.fun

            M1_best, M2_best = params_to_matrices(run_best_x)
            metrics = full_evaluate(f"v5 (cusp_lambda={cl})", M1_best, M2_best,
                                    TRAIN_XYZ, VAL_XYZ, SHOW_PAIRS, SHOW_XYZ)
            metrics["cusp_lambda"] = cl
            metrics["loss"] = run_best_loss
            pareto_results.append((cl, M1_best.copy(), M2_best.copy(), metrics))

            if run_best_loss < overall_best_loss:
                overall_best_loss = run_best_loss
                overall_best_M1 = M1_best.copy()
                overall_best_M2 = M2_best.copy()

    # ── Save best ──────────────────────────────────────────────────
    if overall_best_M1 is not None:
        out = {
            "M1": overall_best_M1.tolist(),
            "gamma": [1/3, 1/3, 1/3],
            "M2": overall_best_M2.tolist(),
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved best to {args.output}")

        # Also save Pareto results
        if len(pareto_results) > 1:
            print(f"\n{'='*70}")
            print("PARETO FRONT (cusp_lambda vs CV)")
            print(f"{'='*70}")
            print(f"  {'lambda':>8s}  {'CV%':>6s}  {'cusps':>6s}  {'hue°':>6s}  "
                  f"{'yellow':>8s}  {'cliff':>6s}  {'loss':>8s}")
            for cl, _, _, m in pareto_results:
                print(f"  {cl:8.1f}  {m['val_cv']*100:6.2f}  {m['n_cusps']:6d}  "
                      f"{m['hue_rms']:6.1f}  {m['yellow_C']:8.4f}  "
                      f"{m['cliff']:6.3f}  {m['loss']:8.4f}")

        # Final comparison
        print(f"\n{'='*70}")
        print("FINAL COMPARISON")
        print(f"{'='*70}")
        full_evaluate("v5 BEST", overall_best_M1, overall_best_M2,
                      TRAIN_XYZ, VAL_XYZ, SHOW_PAIRS, SHOW_XYZ)
        full_evaluate("OKLab", OKLAB_M1, OKLAB_M2,
                      TRAIN_XYZ, VAL_XYZ, SHOW_PAIRS, SHOW_XYZ)


if __name__ == "__main__":
    main()
