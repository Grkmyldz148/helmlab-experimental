#!/usr/bin/env python
"""v5 GenSpace optimization — GPU-accelerated, Ottosson-style with cusp penalty.

Run on RTX 3080:
    python scripts/optimize_v5_gpu.py --de-maxiter 200 --de-popsize 30 --de-runs 3

Key features:
1. Cusp penalty: 36 hue bins, penalize missing cusps + cliff edges
2. Soft hue penalty (not hard constraints)
3. Batch CIEDE2000 + CV on GPU (100x faster than CPU)
4. Multiple seeds: OKLab, v4b, deployed, Phase1H
"""

import argparse
import colorsys
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import differential_evolution, minimize

# ══════════════════════════════════════════════════════════════════════
# Device
# ══════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ══════════════════════════════════════════════════════════════════════
# Constants (torch tensors on GPU)
# ══════════════════════════════════════════════════════════════════════

D65 = np.array([0.95047, 1.0, 1.08883])
D65_T = torch.tensor(D65, dtype=torch.float64, device=DEVICE)

M_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
M_XYZ_TO_SRGB = np.linalg.inv(M_SRGB_TO_XYZ)

M_S2X_T = torch.tensor(M_SRGB_TO_XYZ, dtype=torch.float64, device=DEVICE)
M_X2S_T = torch.tensor(M_XYZ_TO_SRGB, dtype=torch.float64, device=DEVICE)

# OKLab
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
    [1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1],
], dtype=np.float64)
_TARGET_HUE_RAD = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
_PRIMARY_NAMES = ["Red", "Yellow", "Green", "Cyan", "Blue", "Magenta"]


# ══════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════

def srgb_to_linear_np(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def signed_cbrt_np(x):
    return np.sign(x) * np.abs(x) ** (1/3)

def hex_to_xyz(h):
    rgb = np.array([int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]) / 255.0
    return M_SRGB_TO_XYZ @ srgb_to_linear_np(rgb)

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

_PRIMARY_XYZ = np.array([M_SRGB_TO_XYZ @ srgb_to_linear_np(c) for c in _PRIMARY_SRGB])


# ══════════════════════════════════════════════════════════════════════
# Pack / Unpack
# ══════════════════════════════════════════════════════════════════════

def matrices_to_params(M1, M2):
    x = np.zeros(16)
    x[:9] = M1.flatten()
    x[9:12] = M2[0]
    s = signed_cbrt_np(M1 @ D65)
    v1, v2 = ortho_basis(s)
    x[12] = M2[1] @ v1; x[13] = M2[1] @ v2
    x[14] = M2[2] @ v1; x[15] = M2[2] @ v2
    return x

def params_to_matrices(x):
    M1 = x[:9].reshape(3, 3)
    s = signed_cbrt_np(M1 @ D65)
    v1, v2 = ortho_basis(s)
    M2 = np.zeros((3, 3))
    M2[0] = x[9:12]
    M2[1] = x[12]*v1 + x[13]*v2
    M2[2] = x[14]*v1 + x[15]*v2
    return M1, M2


# ══════════════════════════════════════════════════════════════════════
# GPU: sRGB / CIE Lab / CIEDE2000 (batch)
# ══════════════════════════════════════════════════════════════════════

def srgb_to_linear_t(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))

def linear_to_srgb_t(c):
    c = c.clamp(0, 1)
    return torch.where(c <= 0.0031308, 12.92 * c,
                       1.055 * c.clamp(min=1e-12).pow(1/2.4) - 0.055)

def xyz_to_cielab_t(xyz):
    """(N,3) XYZ → (N,3) CIE L*a*b*."""
    r = xyz / D65_T
    f = torch.where(r > 0.008856, r.clamp(min=1e-12).pow(1/3),
                    7.787 * r + 16/116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return torch.stack([L, a, b], dim=1)


def ciede2000_batch(lab1, lab2):
    """Batch CIEDE2000: (N,3) × (N,3) → (N,)."""
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    C1 = (a1**2 + b1**2).sqrt()
    C2 = (a2**2 + b2**2).sqrt()
    Cb = (C1 + C2) / 2
    Cb7 = Cb.pow(7)
    G = 0.5 * (1 - (Cb7 / (Cb7 + 25**7)).sqrt())

    ap1 = a1 * (1 + G)
    ap2 = a2 * (1 + G)
    Cp1 = (ap1**2 + b1**2).sqrt()
    Cp2 = (ap2**2 + b2**2).sqrt()

    hp1 = torch.atan2(b1, ap1) % (2 * torch.pi)
    hp2 = torch.atan2(b2, ap2) % (2 * torch.pi)

    dLp = L2 - L1
    dCp = Cp2 - Cp1

    dhp = hp2 - hp1
    prod = Cp1 * Cp2
    dhp = torch.where(prod < 1e-10, torch.zeros_like(dhp),
                      torch.where(dhp.abs() > torch.pi,
                                  dhp - dhp.sign() * 2 * torch.pi, dhp))
    dHp = 2 * prod.sqrt() * (dhp / 2).sin()

    Lp = (L1 + L2) / 2
    Cp = (Cp1 + Cp2) / 2

    # Average hue
    hsum = hp1 + hp2
    hdiff = (hp1 - hp2).abs()
    hp = torch.where(prod < 1e-10, hsum,
                     torch.where(hdiff <= torch.pi, hsum / 2,
                                 torch.where(hsum < 2 * torch.pi,
                                             hsum / 2 + torch.pi,
                                             hsum / 2 - torch.pi)))

    T = (1 - 0.17 * (hp - torch.pi/6).cos()
         + 0.24 * (2*hp).cos()
         + 0.32 * (3*hp + torch.pi/30).cos()
         - 0.20 * (4*hp - 63*torch.pi/180).cos())

    Lp50 = (Lp - 50)**2
    SL = 1 + 0.015 * Lp50 / (20 + Lp50).sqrt()
    SC = 1 + 0.045 * Cp
    SH = 1 + 0.015 * Cp * T

    Cp7 = Cp.pow(7)
    RC = 2 * (Cp7 / (Cp7 + 25**7)).sqrt()
    da = (hp * 180 / torch.pi - 275) / 25
    dth = 30 * (-(da**2)).exp()
    RT = -(2 * dth * torch.pi / 180).sin() * RC

    r1 = dLp / SL
    r2 = dCp / SC
    r3 = dHp / SH

    return (r1**2 + r2**2 + r3**2 + RT * r2 * r3).clamp(min=0).sqrt()


# ══════════════════════════════════════════════════════════════════════
# GPU: Gradient CV (fully vectorized)
# ══════════════════════════════════════════════════════════════════════

STEPS = 25

# Pre-compute interpolation fractions
_T_FRACS = torch.linspace(0, 1, STEPS, dtype=torch.float64, device=DEVICE)  # (S,)


def compute_cv_gpu(M1_t, M2_t, M2i_t, M1i_t, pairs_xyz_t):
    """Fully vectorized CV on GPU.

    pairs_xyz_t: (N, 2, 3) — N pairs of XYZ endpoints
    Returns: (mean_cv, per_pair_cvs as tensor)
    """
    N = pairs_xyz_t.shape[0]
    S = STEPS

    # Forward: endpoints → Lab
    xyz1 = pairs_xyz_t[:, 0]  # (N, 3)
    xyz2 = pairs_xyz_t[:, 1]  # (N, 3)

    lms1 = (xyz1 @ M1_t.T).clamp(min=0).pow(1/3)
    lms2 = (xyz2 @ M1_t.T).clamp(min=0).pow(1/3)
    lab1 = lms1 @ M2_t.T  # (N, 3)
    lab2 = lms2 @ M2_t.T  # (N, 3)

    # Interpolate: (N, S, 3)
    t = _T_FRACS.view(1, S, 1)  # (1, S, 1)
    lab_interp = lab1.unsqueeze(1) * (1 - t) + lab2.unsqueeze(1) * t  # (N, S, 3)

    # Flatten for batch inverse
    lab_flat = lab_interp.reshape(N * S, 3)  # (N*S, 3)

    # Inverse: Lab → XYZ → sRGB → quantize → XYZ → CIE Lab
    lms_c = lab_flat @ M2i_t.T
    lms = lms_c.sign() * lms_c.abs().pow(3)
    xyz = lms @ M1i_t.T

    rgb_linear = xyz @ M_X2S_T.T
    rgb_srgb = linear_to_srgb_t(rgb_linear)
    rgb8 = (rgb_srgb * 255).round().clamp(0, 255) / 255.0
    rgb_lin_q = srgb_to_linear_t(rgb8)
    xyz_q = rgb_lin_q @ M_S2X_T.T

    cielab = xyz_to_cielab_t(xyz_q)  # (N*S, 3)
    cielab = cielab.reshape(N, S, 3)

    # CIEDE2000 between consecutive steps: (N, S-1)
    lab_prev = cielab[:, :-1].reshape(-1, 3)  # (N*(S-1), 3)
    lab_next = cielab[:, 1:].reshape(-1, 3)
    des = ciede2000_batch(lab_prev, lab_next).reshape(N, S - 1)

    # CV per pair
    mean_de = des.mean(dim=1)
    std_de = des.std(dim=1)
    cvs = torch.where(mean_de > 1e-10, std_de / mean_de, torch.zeros_like(mean_de))

    return cvs.mean().item(), cvs


# ══════════════════════════════════════════════════════════════════════
# GPU: Cusp profiling
# ══════════════════════════════════════════════════════════════════════

N_HUE_BINS = 36
_HUE_BIN_EDGES_T = torch.linspace(-torch.pi, torch.pi, N_HUE_BINS + 1,
                                    dtype=torch.float64, device=DEVICE)

def _build_boundary_xyz_gpu(n_edge=40):
    """Pre-compute sRGB boundary XYZ on GPU."""
    t = torch.linspace(0, 1, n_edge, dtype=torch.float64, device=DEVICE)
    u, v = torch.meshgrid(t, t, indexing='ij')
    u, v = u.reshape(-1), v.reshape(-1)
    z = torch.zeros_like(u)
    o = torch.ones_like(u)

    faces = torch.cat([
        torch.stack([z, u, v], dim=1),
        torch.stack([o, u, v], dim=1),
        torch.stack([u, z, v], dim=1),
        torch.stack([u, o, v], dim=1),
        torch.stack([u, v, z], dim=1),
        torch.stack([u, v, o], dim=1),
    ], dim=0)

    # Deduplicate
    faces = torch.unique((faces * 10000).round(), dim=0) / 10000.0

    # sRGB → linear → XYZ
    linear = srgb_to_linear_t(faces)
    return linear @ M_S2X_T.T

_BOUNDARY_XYZ_T = _build_boundary_xyz_gpu(n_edge=40)


def compute_cusp_profile_gpu(M1_t, M2_t):
    """Compute cusp chroma at each of 36 hue bins. Returns numpy array."""
    lms = (_BOUNDARY_XYZ_T @ M1_t.T).clamp(min=0).pow(1/3)
    lab = lms @ M2_t.T

    a = lab[:, 1]
    b = lab[:, 2]
    C = (a**2 + b**2).sqrt()
    h = torch.atan2(b, a)

    bin_idx = torch.bucketize(h, _HUE_BIN_EDGES_T) - 1
    bin_idx = bin_idx.clamp(0, N_HUE_BINS - 1)

    cusp_C = torch.zeros(N_HUE_BINS, dtype=torch.float64, device=DEVICE)
    for i in range(N_HUE_BINS):
        mask = bin_idx == i
        if mask.any():
            cusp_C[i] = C[mask].max()

    return cusp_C.cpu().numpy()


def compute_cusp_penalty(cusp_C, min_C=0.03):
    """Penalize hue bins with low chroma."""
    deficit = np.maximum(min_C - cusp_C, 0)
    missing_pen = np.sum(deficit ** 2) * 500.0

    cusp_shifted = np.roll(cusp_C, 1)
    both_valid = (cusp_C > 0.01) & (cusp_shifted > 0.01)
    if both_valid.any():
        ratios = cusp_C[both_valid] / cusp_shifted[both_valid]
        ratios = np.where(ratios > 1, 1/ratios, ratios)
        cliff = 1.0 - ratios.min()
    else:
        cliff = 1.0

    cliff_pen = max(0, cliff - 0.5) ** 2 * 100.0
    n_valid = int(np.sum(cusp_C > min_C))

    return missing_pen + cliff_pen, n_valid, float(cusp_C.min()), cliff


# ══════════════════════════════════════════════════════════════════════
# Hue penalty (CPU, fast enough)
# ══════════════════════════════════════════════════════════════════════

def compute_hue_penalty(M1, M2):
    errors_sq = []
    for xyz, target_h in zip(_PRIMARY_XYZ, _TARGET_HUE_RAD):
        lab = M2 @ signed_cbrt_np(M1 @ xyz)
        h = np.arctan2(lab[2], lab[1])
        diff = h - target_h
        angular_err = np.arctan2(np.sin(diff), np.cos(diff))
        errors_sq.append(angular_err ** 2)
    return float(np.mean(errors_sq))

def compute_hue_stats(M1, M2):
    H_deg, C_vals, errors = [], [], []
    for xyz, target_h in zip(_PRIMARY_XYZ, _TARGET_HUE_RAD):
        lab = M2 @ signed_cbrt_np(M1 @ xyz)
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


def pairs_to_gpu_tensor(pairs):
    """Convert hex pairs → (N, 2, 3) XYZ tensor on GPU."""
    xyz_list = []
    for a, b in pairs:
        xyz_list.append([hex_to_xyz(a), hex_to_xyz(b)])
    return torch.tensor(np.array(xyz_list), dtype=torch.float64, device=DEVICE)


# ══════════════════════════════════════════════════════════════════════
# Objective
# ══════════════════════════════════════════════════════════════════════

_eval_count = 0
_best_loss = float("inf")
_best_metrics = {}
_last_print = 0.0


def make_objective(train_t, cusp_lambda=5.0, hue_lambda=0.5):
    """GPU-accelerated objective: CV + cusp + hue + cond."""

    def objective(x):
        global _eval_count, _best_loss, _best_metrics, _last_print
        try:
            M1, M2 = params_to_matrices(x)

            c1 = np.linalg.cond(M1)
            c2 = np.linalg.cond(M2)
            if c1 > 50 or c2 > 50:
                return 999.0

            M1_t = torch.tensor(M1, dtype=torch.float64, device=DEVICE)
            M2_t = torch.tensor(M2, dtype=torch.float64, device=DEVICE)

            try:
                M2i_t = torch.linalg.inv(M2_t)
                M1i_t = torch.linalg.inv(M1_t)
            except Exception:
                return 999.0

            # 1. Gradient CV (GPU)
            mean_cv, cvs = compute_cv_gpu(M1_t, M2_t, M2i_t, M1i_t, train_t)
            if mean_cv > 50:
                return 999.0

            sorted_cvs = cvs.sort(descending=True).values
            n_top = max(1, len(cvs) // 10)
            top10 = sorted_cvs[:n_top].mean().item()

            # 2. Cusp quality (GPU)
            cusp_C = compute_cusp_profile_gpu(M1_t, M2_t)
            cusp_pen, n_cusps, min_cusp, cliff = compute_cusp_penalty(cusp_C)

            # 3. Hue penalty (CPU, fast)
            hue_pen = compute_hue_penalty(M1, M2)

            # 4. Conditioning penalty
            cond_pen = 0.0
            if c1 > 15: cond_pen += 0.005 * (c1 - 15)
            if c2 > 15: cond_pen += 0.005 * (c2 - 15)

            # 5. Yellow chroma penalty
            yellow_bin = int((np.radians(85) + np.pi) / (2*np.pi) * N_HUE_BINS) % N_HUE_BINS
            yellow_C = cusp_C[yellow_bin]
            yellow_pen = max(0, 0.10 - yellow_C) ** 2 * 200.0

            loss = (mean_cv + 0.3 * top10 +
                    cusp_lambda * cusp_pen +
                    hue_lambda * hue_pen +
                    cond_pen + yellow_pen)

        except Exception as e:
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
        if now - _last_print > 5.0:
            _last_print = now
            m = _best_metrics
            print(f"  eval #{_eval_count:>6d}  loss={_best_loss:.5f}  "
                  f"CV={m['cv']*100:.1f}%  cusps={m['cusps']}/36  "
                  f"hue={np.degrees(np.sqrt(m['hue_mse'])):.1f}°  "
                  f"yC={m['yellow_C']:.3f}  cliff={m['cliff']:.2f}  "
                  f"cond={m['c1']:.1f}/{m['c2']:.1f}", flush=True)

        return loss

    return objective


# ══════════════════════════════════════════════════════════════════════
# Seeds
# ══════════════════════════════════════════════════════════════════════

def prepare_seeds():
    seeds = [("OKLab", matrices_to_params(OKLAB_M1, OKLAB_M2))]

    for name, path in [
        ("v4b", "checkpoints/analytic_v4b_genspace.json"),
        ("deployed", "src/helmlab/data/gen_params.json"),
        ("Phase1H", "checkpoints/gen_1h_best.json"),
    ]:
        p = Path(path)
        if p.exists():
            d = json.load(open(p))
            M1 = np.array(d["M1"]); M2 = np.array(d["M2"])
            seeds.append((name, matrices_to_params(M1, M2)))

    return seeds


# ══════════════════════════════════════════════════════════════════════
# Full evaluation
# ══════════════════════════════════════════════════════════════════════

def full_evaluate(label, M1, M2, train_t, val_t, show_pairs, show_t):
    M1_t = torch.tensor(M1, dtype=torch.float64, device=DEVICE)
    M2_t = torch.tensor(M2, dtype=torch.float64, device=DEVICE)
    M2i_t = torch.linalg.inv(M2_t)
    M1i_t = torch.linalg.inv(M1_t)

    train_cv, _ = compute_cv_gpu(M1_t, M2_t, M2i_t, M1i_t, train_t)
    val_cv, _ = compute_cv_gpu(M1_t, M2_t, M2i_t, M1i_t, val_t)
    show_cv, show_cvs = compute_cv_gpu(M1_t, M2_t, M2i_t, M1i_t, show_t)

    hue = compute_hue_stats(M1, M2)
    cusp_C = compute_cusp_profile_gpu(M1_t, M2_t)
    cusp_pen, n_cusps, min_cusp, cliff = compute_cusp_penalty(cusp_C)

    # Achromatic
    s = signed_cbrt_np(M1 @ D65)
    ach_a = float(M2[1] @ s)
    ach_b = float(M2[2] @ s)

    # Yellow
    yb = int((np.radians(85) + np.pi) / (2*np.pi) * N_HUE_BINS) % N_HUE_BINS
    yellow_C = cusp_C[yb]

    # Round-trip
    try:
        M2i = np.linalg.inv(M2); M1i = np.linalg.inv(M1)
        rng = np.random.default_rng(42)
        test = rng.uniform(0.05, 0.9, (200, 3))
        rt_max = 0.0
        for xyz in test:
            lab = M2 @ signed_cbrt_np(M1 @ xyz)
            c = M2i @ lab
            xyz_rt = M1i @ (np.sign(c) * np.abs(c)**3)
            rt_max = max(rt_max, np.max(np.abs(xyz - xyz_rt)))
    except Exception:
        rt_max = 999.0

    print(f"\n  {label}:")
    print(f"    CV: train={train_cv*100:.2f}% val={val_cv*100:.2f}% show={show_cv*100:.2f}%")
    print(f"    Cusps: {n_cusps}/36  min_C={min_cusp:.4f}  cliff={cliff:.3f}")
    print(f"    Yellow (h~85°): C={yellow_C:.4f}")
    print(f"    Hue: RMS={hue['rms']:.1f}° max={hue['max']:.1f}°")
    for name, s in hue["per_color"].items():
        print(f"      {name}: {s}")
    print(f"    Ach: a={ach_a:.2e} b={ach_b:.2e}  RT={rt_max:.2e}")
    print(f"    Cond: M1={np.linalg.cond(M1):.1f} M2={np.linalg.cond(M2):.1f}")

    if show_pairs:
        scv = show_cvs.cpu().numpy()
        for (a, b), cv in zip(show_pairs, scv):
            print(f"      {a} -> {b}: {cv*100:.1f}%")

    return {
        "train_cv": train_cv, "val_cv": val_cv,
        "n_cusps": n_cusps, "min_cusp": min_cusp, "cliff": cliff,
        "yellow_C": yellow_C, "hue_rms": hue["rms"],
        "ach_a": ach_a, "ach_b": ach_b, "rt_max": rt_max,
        "cond_M1": np.linalg.cond(M1), "cond_M2": np.linalg.cond(M2),
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="v5 GPU GenSpace — Ottosson-style cusp penalty")
    parser.add_argument("--output", default="checkpoints/analytic_v5_ottosson.json")
    parser.add_argument("--de-maxiter", type=int, default=200)
    parser.add_argument("--de-popsize", type=int, default=30)
    parser.add_argument("--de-runs", type=int, default=3)
    parser.add_argument("--cusp-lambda", type=float, default=5.0)
    parser.add_argument("--hue-lambda", type=float, default=0.5)
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep cusp_lambda for Pareto front")
    args = parser.parse_args()

    print("=" * 70)
    print("v5 GenSpace GPU — Ottosson-style with Cusp Penalty")
    print("=" * 70)
    print(f"  cusp_lambda={args.cusp_lambda}  hue_lambda={args.hue_lambda}")
    print(f"  DE: popsize={args.de_popsize}, maxiter={args.de_maxiter}, "
          f"runs={args.de_runs}/seed")

    # ── Training data → GPU tensors ────────────────────────────────
    print("\nGenerating training pairs...")
    ALL_PAIRS = generate_training_pairs()
    np.random.seed(123)
    indices = np.random.permutation(len(ALL_PAIRS))
    split = int(len(ALL_PAIRS) * 0.8)
    TRAIN_PAIRS = [ALL_PAIRS[i] for i in indices[:split]]
    VAL_PAIRS = [ALL_PAIRS[i] for i in indices[split:]]
    SHOW_PAIRS = [
        ('#ff6b00','#00d4ff'), ('#ff0000','#0000ff'), ('#000000','#ffffff'),
        ('#ff0000','#00ff00'), ('#0000ff','#ffff00'), ('#8000ff','#ff8000'),
    ]

    train_t = pairs_to_gpu_tensor(TRAIN_PAIRS)
    val_t = pairs_to_gpu_tensor(VAL_PAIRS)
    show_t = pairs_to_gpu_tensor(SHOW_PAIRS)
    print(f"  Total: {len(ALL_PAIRS)}, Train: {len(TRAIN_PAIRS)}, Val: {len(VAL_PAIRS)}")
    print(f"  GPU tensors ready: train={tuple(train_t.shape)}")

    # ── Baselines ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("BASELINES")
    print(f"{'='*70}")

    full_evaluate("OKLab", OKLAB_M1, OKLAB_M2,
                  train_t, val_t, SHOW_PAIRS, show_t)

    for name, path in [
        ("v4b", "checkpoints/analytic_v4b_genspace.json"),
        ("Deployed", "src/helmlab/data/gen_params.json"),
    ]:
        p = Path(path)
        if p.exists():
            d = json.load(open(p))
            M1 = np.array(d["M1"]); M2 = np.array(d["M2"])
            full_evaluate(name, M1, M2, train_t, val_t, SHOW_PAIRS, show_t)

    sys.stdout.flush()

    # ── Optimization ───────────────────────────────────────────────
    if args.sweep:
        cusp_lambdas = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    else:
        cusp_lambdas = [args.cusp_lambda]

    overall_best_loss = float("inf")
    overall_best_M1 = None
    overall_best_M2 = None
    pareto = []

    for cl in cusp_lambdas:
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION — cusp_lambda={cl}")
        print(f"{'='*70}")

        global _eval_count, _best_loss, _best_metrics, _last_print

        objective = make_objective(train_t, cusp_lambda=cl,
                                   hue_lambda=args.hue_lambda)

        seeds = prepare_seeds()
        run_best_loss = float("inf")
        run_best_x = None

        for seed_name, x0 in seeds:
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

                # L-BFGS-B refinement
                result2 = minimize(objective, result.x, method="L-BFGS-B",
                                   bounds=bounds,
                                   options={"maxiter": 1500, "ftol": 1e-12})

                final_loss = min(result.fun, result2.fun)
                final_x = result2.x if result2.fun <= result.fun else result.x

                dt = time.time() - t0
                M1t, M2t = params_to_matrices(final_x)
                cusp_C = compute_cusp_profile_gpu(
                    torch.tensor(M1t, dtype=torch.float64, device=DEVICE),
                    torch.tensor(M2t, dtype=torch.float64, device=DEVICE))
                _, nc, _, clf = compute_cusp_penalty(cusp_C)
                hp = compute_hue_penalty(M1t, M2t)

                print(f"    loss={final_loss:.5f}  cusps={nc}/36  "
                      f"hue={np.degrees(np.sqrt(hp)):.1f}°  "
                      f"cliff={clf:.2f}  ({dt:.0f}s)")

                if final_loss < run_best_loss:
                    run_best_loss = final_loss
                    run_best_x = final_x.copy()
                    print(f"    ** New best!")

                sys.stdout.flush()

        # Final tight refinement
        if run_best_x is not None:
            print(f"\n  --- Final refinement ---")
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
            metrics = full_evaluate(f"v5 (cl={cl})", M1_best, M2_best,
                                    train_t, val_t, SHOW_PAIRS, show_t)
            metrics["cusp_lambda"] = cl
            metrics["loss"] = run_best_loss
            pareto.append((cl, M1_best.copy(), M2_best.copy(), metrics))

            # Save per-lambda checkpoint
            ckpt = {
                "M1": M1_best.tolist(),
                "gamma": [1/3, 1/3, 1/3],
                "M2": M2_best.tolist(),
                "metrics": {k: v for k, v in metrics.items()
                           if isinstance(v, (int, float, str))},
            }
            ckpt_path = f"checkpoints/v5_cl{cl:.0f}.json"
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, indent=2)
            print(f"  Saved -> {ckpt_path}")

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
        print(f"\nSaved best -> {args.output}")

        if len(pareto) > 1:
            print(f"\n{'='*70}")
            print("PARETO FRONT")
            print(f"{'='*70}")
            print(f"  {'cl':>6s}  {'CV%':>6s}  {'cusps':>6s}  {'hue°':>6s}  "
                  f"{'yC':>6s}  {'cliff':>6s}")
            for cl, _, _, m in pareto:
                print(f"  {cl:6.1f}  {m['val_cv']*100:6.2f}  {m['n_cusps']:6d}  "
                      f"{m['hue_rms']:6.1f}  {m['yellow_C']:6.4f}  "
                      f"{m['cliff']:6.3f}")

        print(f"\n{'='*70}")
        print("FINAL COMPARISON")
        print(f"{'='*70}")
        full_evaluate("v5 BEST", overall_best_M1, overall_best_M2,
                      train_t, val_t, SHOW_PAIRS, show_t)
        full_evaluate("OKLab", OKLAB_M1, OKLAB_M2,
                      train_t, val_t, SHOW_PAIRS, show_t)


if __name__ == "__main__":
    main()
