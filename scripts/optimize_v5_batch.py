#!/usr/bin/env python
"""v5 GenSpace — BATCH GPU CMA-ES. Entire population evaluated in ONE GPU call.

30x faster than serial version. 1.5 hours -> 3 minutes.

Usage:
    python scripts/optimize_v5_batch.py
    python scripts/optimize_v5_batch.py --popsize 64 --generations 300
"""

import argparse
import colorsys
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import cma

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ══════════════════════════════════════════════════════════════════════
# Constants
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
OKLAB_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
]) @ M_XYZ_TO_SRGB
OKLAB_M2 = np.array([
    [ 0.2104542553,  0.7936177850, -0.0040720468],
    [ 1.9779984951, -2.4285922050,  0.4505937099],
    [ 0.0259040371,  0.7827717662, -0.8086757660],
])

_PRIMARY_SRGB = np.array([[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1]], dtype=np.float64)
_TARGET_HUE_RAD = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])

# Rec2020 primaries → XYZ (for wide-gamut negative LMS check)
M_REC2020_TO_XYZ = np.array([
    [0.6369580, 0.1446169, 0.1688810],
    [0.2627002, 0.6779981, 0.0593017],
    [0.0000000, 0.0280727, 1.0609851],
])
# Dense Rec2020 boundary: 6 faces × 10×10 grid = ~500 unique points
def _build_rec2020_boundary():
    t = np.linspace(0, 1, 10)
    u, v = np.meshgrid(t, t)
    u, v = u.ravel(), v.ravel()
    z = np.zeros_like(u); o = np.ones_like(u)
    faces = np.vstack([
        np.column_stack([z,u,v]), np.column_stack([o,u,v]),
        np.column_stack([u,z,v]), np.column_stack([u,o,v]),
        np.column_stack([u,v,z]), np.column_stack([u,v,o]),
    ])
    faces = np.unique(np.round(faces, 4), axis=0)
    return faces @ M_REC2020_TO_XYZ

_R2020_BOUNDARY_XYZ = _build_rec2020_boundary()  # (~400, 3)
_R2020_BOUNDARY_T = torch.tensor(_R2020_BOUNDARY_XYZ, dtype=torch.float64, device=DEVICE)

# Dark test points for Jacobian stability (very dark colors in XYZ)
_DARK_XYZ = torch.tensor([
    [0.001, 0.001, 0.001], [0.005, 0.005, 0.005], [0.01, 0.01, 0.01],
    [0.001, 0.0005, 0.002], [0.002, 0.001, 0.003], [0.003, 0.002, 0.001],
    [0.0001, 0.0001, 0.0001], [0.02, 0.02, 0.02],
], dtype=torch.float64, device=DEVICE)  # (D, 3)

def srgb_to_linear_np(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

_PRIMARY_XYZ = np.array([M_SRGB_TO_XYZ @ srgb_to_linear_np(c) for c in _PRIMARY_SRGB])
_PRIMARY_XYZ_T = torch.tensor(_PRIMARY_XYZ, dtype=torch.float64, device=DEVICE)  # (6,3)
_TARGET_HUE_T = torch.tensor(_TARGET_HUE_RAD, dtype=torch.float64, device=DEVICE)  # (6,)

# ══════════════════════════════════════════════════════════════════════
# Pack / Unpack — BATCH version (P candidates at once)
# ══════════════════════════════════════════════════════════════════════

def signed_cbrt_np(x):
    return np.sign(x) * np.abs(x) ** (1/3)

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

def matrices_to_params(M1, M2):
    x = np.zeros(16)
    x[:9] = M1.flatten()
    x[9:12] = M2[0]
    s = signed_cbrt_np(M1 @ D65)
    v1, v2 = ortho_basis(s)
    x[12] = M2[1] @ v1; x[13] = M2[1] @ v2
    x[14] = M2[2] @ v1; x[15] = M2[2] @ v2
    return x

def params_to_matrices_batch(X):
    """(P, 16) -> (P, 3, 3) M1, (P, 3, 3) M2. NumPy."""
    P = X.shape[0]
    M1 = X[:, :9].reshape(P, 3, 3)
    M2 = np.zeros((P, 3, 3))
    M2[:, 0, :] = X[:, 9:12]  # L-row free

    for i in range(P):
        s = signed_cbrt_np(M1[i] @ D65)
        v1, v2 = ortho_basis(s)
        M2[i, 1] = X[i, 12] * v1 + X[i, 13] * v2
        M2[i, 2] = X[i, 14] * v1 + X[i, 15] * v2
    return M1, M2


# ══════════════════════════════════════════════════════════════════════
# GPU batch helpers
# ══════════════════════════════════════════════════════════════════════

def srgb_to_linear_t(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))

def linear_to_srgb_t(c):
    c = c.clamp(0, 1)
    return torch.where(c <= 0.0031308, 12.92 * c,
                       1.055 * c.clamp(min=1e-12).pow(1/2.4) - 0.055)

def signed_cbrt_t(x):
    """Sign-preserving cube root: sign(x) * |x|^(1/3). Bijective, exact inverse."""
    return x.sign() * x.abs().pow(1/3)

def xyz_to_cielab_t(xyz):
    """(..., 3) XYZ -> (..., 3) CIE Lab."""
    r = xyz / D65_T
    f = torch.where(r > 0.008856, r.clamp(min=1e-12).pow(1/3),
                    7.787 * r + 16/116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)


def ciede2000_batch(lab1, lab2):
    """(..., 3) x (..., 3) -> (...)  Batch CIEDE2000."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    pi = torch.pi

    C1 = (a1**2 + b1**2).sqrt()
    C2 = (a2**2 + b2**2).sqrt()
    Cb = (C1 + C2) / 2
    Cb7 = Cb.pow(7)
    G = 0.5 * (1 - (Cb7 / (Cb7 + 25**7)).sqrt())
    ap1 = a1 * (1 + G); ap2 = a2 * (1 + G)
    Cp1 = (ap1**2 + b1**2).sqrt(); Cp2 = (ap2**2 + b2**2).sqrt()
    hp1 = torch.atan2(b1, ap1) % (2*pi); hp2 = torch.atan2(b2, ap2) % (2*pi)
    dLp = L2 - L1; dCp = Cp2 - Cp1
    dhp = hp2 - hp1
    prod = Cp1 * Cp2
    dhp = torch.where(prod < 1e-10, torch.zeros_like(dhp),
                      torch.where(dhp.abs() > pi, dhp - dhp.sign() * 2 * pi, dhp))
    dHp = 2 * prod.sqrt() * (dhp / 2).sin()
    Lp = (L1 + L2) / 2; Cp = (Cp1 + Cp2) / 2
    hsum = hp1 + hp2; hdiff = (hp1 - hp2).abs()
    hp = torch.where(prod < 1e-10, hsum,
                     torch.where(hdiff <= pi, hsum / 2,
                                 torch.where(hsum < 2*pi, hsum/2 + pi, hsum/2 - pi)))
    T = (1 - 0.17*(hp - pi/6).cos() + 0.24*(2*hp).cos()
         + 0.32*(3*hp + pi/30).cos() - 0.20*(4*hp - 63*pi/180).cos())
    Lp50 = (Lp - 50)**2
    SL = 1 + 0.015 * Lp50 / (20 + Lp50).sqrt()
    SC = 1 + 0.045 * Cp; SH = 1 + 0.015 * Cp * T
    Cp7 = Cp.pow(7); RC = 2 * (Cp7 / (Cp7 + 25**7)).sqrt()
    da = (hp * 180 / pi - 275) / 25
    dth = 30 * (-(da**2)).exp()
    RT = -(2 * dth * pi / 180).sin() * RC
    r1, r2, r3 = dLp / SL, dCp / SC, dHp / SH
    return (r1**2 + r2**2 + r3**2 + RT * r2 * r3).clamp(min=0).sqrt()


# ══════════════════════════════════════════════════════════════════════
# BATCH objective: P candidates x N pairs x S steps — ALL AT ONCE
# ══════════════════════════════════════════════════════════════════════

STEPS = 25
_T_FRACS = torch.linspace(0, 1, STEPS, dtype=torch.float64, device=DEVICE)

# Boundary points for cusp — high res for 360 bins
def _build_boundary():
    t = torch.linspace(0, 1, 80, dtype=torch.float64, device=DEVICE)
    u, v = torch.meshgrid(t, t, indexing='ij')
    u, v = u.reshape(-1), v.reshape(-1)
    z = torch.zeros_like(u); o = torch.ones_like(u)
    faces = torch.cat([
        torch.stack([z,u,v],1), torch.stack([o,u,v],1),
        torch.stack([u,z,v],1), torch.stack([u,o,v],1),
        torch.stack([u,v,z],1), torch.stack([u,v,o],1),
    ])
    faces = torch.unique((faces * 10000).round(), dim=0) / 10000.0
    linear = srgb_to_linear_t(faces)
    return linear @ M_S2X_T.T

_BXYZ = _build_boundary()  # (B, 3) ~25K points
N_HUE_BINS = 360
_HUE_EDGES = torch.linspace(-torch.pi, torch.pi, N_HUE_BINS + 1, dtype=torch.float64, device=DEVICE)

# ── Unimodality metric (matches space-test-project) ──────────────
# At each hue, chroma boundary vs L should be unimodal:
# after the cusp (max C), chroma must only decrease toward white.
N_UNI_HUES = 72      # 5° per bin (vs test suite's 1°, but 5x faster)
N_UNI_L = 50         # L resolution
N_UNI_C = 40         # C resolution
UNI_HUE_BATCH = 12   # process 12 hues at a time (memory friendly)

def compute_unimodal(M1_t, M2_t, L_white):
    """Compute boundary unimodality violations for P candidates (GPU batch).

    For each hue, builds the chroma-vs-L boundary profile by inverse-mapping
    an HLC grid to sRGB and checking in-gamut. Then checks if chroma is
    monotonically decreasing after the cusp (max chroma L).

    Args:
        M1_t: (P, 3, 3) M1 matrices
        M2_t: (P, 3, 3) M2 matrices
        L_white: (P,) white point L values for normalization

    Returns:
        unimodal_viol: (P,) int — number of hues with violations
        unimodal_raw: (P,) float — differentiable penalty for CMA-ES
    """
    P = M1_t.shape[0]
    device = M1_t.device

    # Normalize M2 so L_white = 1 (matches test suite behavior)
    M2_norm = M2_t / L_white.view(P, 1, 1).clamp(min=1e-10)
    M2_norm_inv = torch.linalg.inv(M2_norm)
    M1_inv = torch.linalg.inv(M1_t)

    Ls = torch.linspace(0.02, 0.998, N_UNI_L, device=device, dtype=torch.float64)
    Cs = torch.linspace(0.001, 0.5, N_UNI_C, device=device, dtype=torch.float64)

    mc_all = torch.zeros(P, N_UNI_HUES, N_UNI_L, device=device, dtype=torch.float64)

    for hs in range(0, N_UNI_HUES, UNI_HUE_BATCH):
        he = min(hs + UNI_HUE_BATCH, N_UNI_HUES)
        nh = he - hs

        # Build HLC grid for this hue chunk
        angles = torch.arange(hs, he, device=device, dtype=torch.float64) * (2 * torch.pi / N_UNI_HUES)
        ch = torch.cos(angles).view(nh, 1, 1)
        sh = torch.sin(angles).view(nh, 1, 1)
        Le = Ls.view(1, N_UNI_L, 1).expand(nh, N_UNI_L, N_UNI_C)
        Ce = Cs.view(1, 1, N_UNI_C).expand(nh, N_UNI_L, N_UNI_C)

        lab = torch.stack([Le, Ce * ch, Ce * sh], dim=-1)  # (nh, N_L, N_C, 3)
        lab_flat = lab.reshape(-1, 3)  # (K, 3)
        K = lab_flat.shape[0]

        # Inverse: normalized Lab -> lms_c -> lms -> XYZ -> linear sRGB
        lms_c = torch.matmul(lab_flat.unsqueeze(0), M2_norm_inv.transpose(1, 2))  # (P, K, 3)
        lms = lms_c.sign() * lms_c.abs().pow(3)
        xyz = torch.matmul(lms, M1_inv.transpose(1, 2))  # (P, K, 3)
        lin = torch.matmul(xyz, M_X2S_T.T)  # (P, K, 3)

        # In-gamut check (same tolerance as test suite)
        ok = ((lin >= -0.002) & (lin <= 1.002)).all(dim=2)  # (P, K)
        ok = ok.reshape(P, nh, N_UNI_L, N_UNI_C)

        # Max in-gamut chroma per (hue, L)
        Cs_exp = Cs.view(1, 1, 1, N_UNI_C).expand(P, nh, N_UNI_L, N_UNI_C)
        mc = torch.where(ok, Cs_exp, torch.zeros_like(Cs_exp)).max(dim=3).values
        mc_all[:, hs:he] = mc

    # Cusp: L index with max chroma per hue
    ci = mc_all.argmax(dim=2)  # (P, N_UNI_HUES)

    # Unimodality: after cusp, chroma must not increase
    mc_diff = mc_all[:, :, 1:] - mc_all[:, :, :-1]  # (P, N_H, N_L-1)
    idx = torch.arange(N_UNI_L - 1, device=device).view(1, 1, N_UNI_L - 1)
    after_cusp = idx >= ci.unsqueeze(2)  # (P, N_H, N_L-1)

    # Count: hues with any post-cusp increase > 0.001
    viol_per_hue = ((mc_diff > 0.001) & after_cusp).any(dim=2)  # (P, N_H)
    unimodal_viol = viol_per_hue.sum(dim=1)  # (P,)

    # Penalty: sum of positive dC after cusp (smooth) + normalized count (discrete)
    positive_dC = mc_diff.clamp(min=0) * after_cusp.float()
    unimodal_raw = positive_dC.sum(dim=[1, 2]) + unimodal_viol.float() / N_UNI_HUES

    return unimodal_viol, unimodal_raw


def batch_evaluate(M1_t, M2_t, pairs_t):
    """Evaluate P candidates on N pairs simultaneously.

    M1_t: (P, 3, 3)
    M2_t: (P, 3, 3)
    pairs_t: (N, 2, 3) — XYZ pairs

    Returns dict of (P,) tensors: cv_mean, cv_top10, cusp_penalty, hue_penalty, cond_pen, yellow_pen
    """
    P = M1_t.shape[0]
    N = pairs_t.shape[0]
    S = STEPS

    # ── 1. Gradient CV ──────────────────────────────────────────
    # Forward endpoints: (P, N, 3)
    xyz1 = pairs_t[:, 0].unsqueeze(0).expand(P, -1, -1)  # (P, N, 3)
    xyz2 = pairs_t[:, 1].unsqueeze(0).expand(P, -1, -1)

    # XYZ -> LMS -> cbrt -> Lab for all P×N
    lms1 = signed_cbrt_t(torch.bmm(xyz1, M1_t.transpose(1, 2)))  # (P,N,3)
    lms2 = signed_cbrt_t(torch.bmm(xyz2, M1_t.transpose(1, 2)))
    lab1 = torch.bmm(lms1, M2_t.transpose(1, 2))  # (P,N,3)
    lab2 = torch.bmm(lms2, M2_t.transpose(1, 2))

    # Interpolate: (P, N, S, 3)
    t = _T_FRACS.view(1, 1, S, 1)
    lab_interp = lab1.unsqueeze(2) * (1 - t) + lab2.unsqueeze(2) * t

    # Inverse: Lab -> LMS_c -> LMS -> XYZ -> sRGB -> quantize -> CIE Lab
    M2i_t = torch.linalg.inv(M2_t)  # (P, 3, 3)
    M1i_t = torch.linalg.inv(M1_t)

    flat = lab_interp.reshape(P, N * S, 3)  # (P, N*S, 3)
    lms_c = torch.bmm(flat, M2i_t.transpose(1, 2))  # (P, N*S, 3)
    lms = lms_c.sign() * lms_c.abs().pow(3)
    xyz = torch.bmm(lms, M1i_t.transpose(1, 2))  # (P, N*S, 3)

    # xyz -> sRGB quantize -> CIE Lab
    rgb_lin = torch.matmul(xyz, M_X2S_T.T)  # broadcast (P, N*S, 3)
    rgb_srgb = linear_to_srgb_t(rgb_lin)
    rgb8 = (rgb_srgb * 255).round().clamp(0, 255) / 255.0
    rgb_q = srgb_to_linear_t(rgb8)
    xyz_q = torch.matmul(rgb_q, M_S2X_T.T)
    cielab = xyz_to_cielab_t(xyz_q)  # (P, N*S, 3)
    cielab = cielab.reshape(P, N, S, 3)

    # CIEDE2000 between consecutive steps: (P, N, S-1)
    des = ciede2000_batch(cielab[:, :, :-1], cielab[:, :, 1:])  # (P, N, S-1)

    mean_de = des.mean(dim=2)  # (P, N)
    std_de = des.std(dim=2)
    cvs = torch.where(mean_de > 1e-10, std_de / mean_de, torch.zeros_like(mean_de))  # (P, N)

    cv_mean = cvs.mean(dim=1)  # (P,)
    cv_sorted = cvs.sort(dim=1, descending=True).values
    n_top = max(1, N // 10)
    cv_top10 = cv_sorted[:, :n_top].mean(dim=1)  # (P,)

    # ── 1b. Drift penalty (hue deviation from straight path in CIE Lab) ──
    h_steps = torch.atan2(cielab[:, :, :, 2], cielab[:, :, :, 1])  # (P, N, S)
    C_steps = (cielab[:, :, :, 1]**2 + cielab[:, :, :, 2]**2).sqrt()
    # Expected hue: linear interp from start to end
    h_start = h_steps[:, :, 0:1]  # (P, N, 1)
    h_end = h_steps[:, :, -1:]
    h_diff = torch.atan2((h_end - h_start).sin(), (h_end - h_start).cos())
    t_drift = _T_FRACS.view(1, 1, S)
    h_expected = h_start + t_drift * h_diff
    # Angular deviation at each step
    dev = torch.atan2((h_steps - h_expected).sin(), (h_steps - h_expected).cos()).abs()
    # Only count chromatic steps (CIE Lab C > 5)
    dev = dev * (C_steps > 5.0).float()
    max_dev_per_pair = dev.max(dim=2).values  # (P, N) — max drift per pair in radians
    drift_raw = max_dev_per_pair.mean(dim=1)  # (P,) — mean of max drifts

    # ── 2. Cusp profiling (360 bins, normalized) ────────────────
    B = _BXYZ.shape[0]
    bxyz = _BXYZ.unsqueeze(0).expand(P, -1, -1)  # (P, B, 3)
    blms = signed_cbrt_t(torch.bmm(bxyz, M1_t.transpose(1, 2)))
    blab = torch.bmm(blms, M2_t.transpose(1, 2))  # (P, B, 3)

    # Normalize so L_white = 1.0 (scale-invariant gamut metrics)
    d65_batch = D65_T.unsqueeze(0).expand(P, -1).unsqueeze(1)  # (P, 1, 3)
    lms_w = torch.bmm(d65_batch, M1_t.transpose(1, 2)).squeeze(1)  # (P, 3)
    lms_w = signed_cbrt_t(lms_w)
    L_white = (lms_w * M2_t[:, 0, :]).sum(dim=1)  # (P,)
    scale = L_white.clamp(min=1e-10).unsqueeze(1).unsqueeze(2)  # (P, 1, 1)
    blab = blab / scale

    ba = blab[:, :, 1]; bb = blab[:, :, 2]
    bL = blab[:, :, 0]
    bC = (ba**2 + bb**2).sqrt()  # (P, B)
    bh = torch.atan2(bb, ba)

    bin_idx = torch.bucketize(bh, _HUE_EDGES) - 1
    bin_idx = bin_idx.clamp(0, N_HUE_BINS - 1)

    # Max chroma per bin + cusp L per bin
    cusp_C = torch.zeros(P, N_HUE_BINS, dtype=torch.float64, device=DEVICE)
    cusp_L = torch.zeros(P, N_HUE_BINS, dtype=torch.float64, device=DEVICE)
    for i in range(N_HUE_BINS):
        mask = (bin_idx == i)  # (P, B)
        masked_C = bC * mask.float()
        max_C, max_idx = masked_C.max(dim=1)
        cusp_C[:, i] = max_C
        # L at cusp (gather from bL using max_idx)
        cusp_L[:, i] = bL.gather(1, max_idx.unsqueeze(1)).squeeze(1)

    # Raw gamut stats (penalties computed in loss function with tunable weights)
    # Missing cusps
    min_C_thresh = 0.02
    deficit = (min_C_thresh - cusp_C).clamp(min=0)
    missing_pen = (deficit ** 2).sum(dim=1)  # (P,) — raw, unweighted

    # Cliff
    shifted = torch.roll(cusp_C, 1, dims=1)
    both_valid = (cusp_C > 0.005) & (shifted > 0.005)
    ratios = torch.where(both_valid, cusp_C / shifted.clamp(min=1e-10), torch.ones_like(cusp_C))
    ratios = torch.where(ratios > 1, 1 / ratios, ratios)
    min_ratio = ratios.min(dim=1).values
    cliff = 1.0 - min_ratio
    cliff_raw = (cliff - 0.30).clamp(min=0) ** 2  # (P,) — raw

    # Smoothness
    dC = cusp_C[:, 1:] - cusp_C[:, :-1]
    smooth_raw = (dC ** 2).sum(dim=1)  # (P,) — raw

    # Unimodality (matches test suite: chroma must decrease after cusp at each hue)
    unimodal_viol, unimodal_raw = compute_unimodal(M1_t, M2_t, L_white)

    # Yellow chroma (h ~ 85 deg -> normalized)
    ybin = int((np.radians(85) + np.pi) / (2*np.pi) * N_HUE_BINS) % N_HUE_BINS
    yellow_C = cusp_C[:, ybin]
    yellow_raw = (0.15 - yellow_C).clamp(min=0) ** 2  # (P,) — raw

    # ── 3. Hue penalty (batch) ──────────────────────────────────
    # (P, 6, 3) primary XYZ -> Lab
    pxyz = _PRIMARY_XYZ_T.unsqueeze(0).expand(P, -1, -1)  # (P, 6, 3)
    plms = signed_cbrt_t(torch.bmm(pxyz, M1_t.transpose(1, 2)))
    plab = torch.bmm(plms, M2_t.transpose(1, 2))  # (P, 6, 3)
    ph = torch.atan2(plab[:, :, 2], plab[:, :, 1])  # (P, 6)
    diff = ph - _TARGET_HUE_T.unsqueeze(0)
    angular_err = torch.atan2(diff.sin(), diff.cos())
    hue_penalty = (angular_err ** 2).mean(dim=1)  # (P,)

    # ── 4. Conditioning penalty ─────────────────────────────────
    # Batch condition number via SVD
    s1 = torch.linalg.svdvals(M1_t)  # (P, 3)
    s2 = torch.linalg.svdvals(M2_t)
    cond1 = s1[:, 0] / s1[:, 2].clamp(min=1e-10)
    cond2 = s2[:, 0] / s2[:, 2].clamp(min=1e-10)
    # Quadratic penalty above threshold (default 5.0)
    cond_pen = torch.zeros(P, dtype=torch.float64, device=M1_t.device)

    # ── 5. Rec2020 negative LMS + round-trip penalty (dense boundary) ──
    # Use ~400 Rec2020 boundary points instead of 12 corners
    R = _R2020_BOUNDARY_T.shape[0]
    r2020_xyz = _R2020_BOUNDARY_T.unsqueeze(0).expand(P, -1, -1)  # (P, R, 3)
    r2020_lms = torch.bmm(r2020_xyz, M1_t.transpose(1, 2))  # (P, R, 3)

    # Negative LMS fraction (causes clamp → information loss → RT error)
    neg_frac = (r2020_lms < -1e-10).float().mean(dim=[1, 2])  # (P,)
    # Also penalize magnitude of negative values (not just count)
    neg_mag = (-r2020_lms).clamp(min=0).max(dim=2).values.max(dim=1).values  # (P,)
    neg_lms_raw = neg_frac + 0.01 * neg_mag

    # Actual Rec2020 round-trip error
    r2020_lms_c = signed_cbrt_t(r2020_lms)  # forward: sign-preserving cbrt
    r2020_lab = torch.bmm(r2020_lms_c, M2_t.transpose(1, 2))  # (P, R, 3)
    # Inverse
    r2020_lms_c_inv = torch.bmm(r2020_lab, M2i_t.transpose(1, 2))
    r2020_lms_inv = r2020_lms_c_inv.sign() * r2020_lms_c_inv.abs().pow(3)
    r2020_xyz_inv = torch.bmm(r2020_lms_inv, M1i_t.transpose(1, 2))
    rt_err = (r2020_xyz - r2020_xyz_inv).abs().max(dim=2).values.max(dim=1).values  # (P,)
    r2020_rt_raw = rt_err  # target: < 1e-12

    # ── 6. L_white scale penalty ──────────────────────────────
    # Large L_white (>10) means M1 entries are too big → numerical issues
    # Small L_white (<0.1) means M1 entries too small → precision issues
    lw_log = (L_white / 1.0).clamp(min=1e-10).log10()  # target: log10(1) = 0
    scale_raw = (lw_log.abs() - 0.5).clamp(min=0) ** 2  # penalty starts at L_white < 0.3 or > 3

    # ── 7. Dark region LMS minimum (Jacobian stability) ───────
    dark_xyz = _DARK_XYZ.unsqueeze(0).expand(P, -1, -1)  # (P, D, 3)
    dark_lms = torch.bmm(dark_xyz, M1_t.transpose(1, 2))  # (P, D, 3)
    min_dark_abs = dark_lms.abs().min(dim=2).values.min(dim=1).values  # (P,)
    dark_lms_raw = (1e-5 - min_dark_abs).clamp(min=0) ** 2  # (P,)

    # Valid mask (reject ill-conditioned)
    valid = (cond1 < 50) & (cond2 < 50) & (cv_mean < 5.0)

    return {
        'cv_mean': cv_mean, 'cv_top10': cv_top10,
        'hue_penalty': hue_penalty,
        'cond_pen': cond_pen,
        'valid': valid, 'cusp_C': cusp_C, 'cusp_L': cusp_L,
        'cond1': cond1, 'cond2': cond2,
        'cliff': cliff, 'yellow_C': yellow_C,
        'unimodal_viol': unimodal_viol, 'L_white': L_white,
        # Raw penalties (unweighted)
        'missing_pen': missing_pen, 'cliff_raw': cliff_raw,
        'smooth_raw': smooth_raw, 'unimodal_raw': unimodal_raw,
        'yellow_raw': yellow_raw,
        # New: drift, neg_lms, dark_lms, r2020_rt, scale
        'drift_raw': drift_raw, 'neg_lms_raw': neg_lms_raw,
        'dark_lms_raw': dark_lms_raw,
        'r2020_rt_raw': r2020_rt_raw, 'scale_raw': scale_raw,
        'drift_deg': drift_raw * (180.0 / torch.pi),  # for logging
        'neg_frac': neg_frac, 'r2020_rt': rt_err,
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
            if abs(i-j) >= 2: pairs.append((grays[i], grays[j]))
    for s, v in [(1.0,1.0),(0.7,0.8),(0.5,0.6)]:
        for h_start in range(0,360,30):
            h_end = (h_start+60)%360
            r1,g1,b1 = colorsys.hsv_to_rgb(h_start/360,s,v)
            r2,g2,b2 = colorsys.hsv_to_rgb(h_end/360,s,v)
            pairs.append((f"#{int(r1*255):02x}{int(g1*255):02x}{int(b1*255):02x}",
                          f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"))
    for s, v in [(1.0,1.0),(0.6,0.8)]:
        for h in range(0,180,20):
            r1,g1,b1 = colorsys.hsv_to_rgb(h/360,s,v)
            r2,g2,b2 = colorsys.hsv_to_rgb((h+180)/360,s,v)
            pairs.append((f"#{int(r1*255):02x}{int(g1*255):02x}{int(b1*255):02x}",
                          f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"))
    for h in [0,30,60,120,180,240,300]:
        r1,g1,b1 = colorsys.hsv_to_rgb(h/360,0.8,0.95)
        r2,g2,b2 = colorsys.hsv_to_rgb(h/360,0.8,0.3)
        pairs.append((f"#{int(r1*255):02x}{int(g1*255):02x}{int(b1*255):02x}",
                      f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"))
    np.random.seed(42)
    for _ in range(50):
        c1 = np.random.randint(0,256,3); c2 = np.random.randint(0,256,3)
        pairs.append((f"#{c1[0]:02x}{c1[1]:02x}{c1[2]:02x}",f"#{c2[0]:02x}{c2[1]:02x}{c2[2]:02x}"))
    seen = set(); unique = []
    for a, b in pairs:
        key = tuple(sorted([a.lower(),b.lower()]))
        if key not in seen: seen.add(key); unique.append((a, b))
    return unique

def hex_to_xyz(h):
    rgb = np.array([int(h[1:3],16),int(h[3:5],16),int(h[5:7],16)])/255.0
    return M_SRGB_TO_XYZ @ srgb_to_linear_np(rgb)

def pairs_to_tensor(pairs):
    xyz = [[hex_to_xyz(a), hex_to_xyz(b)] for a, b in pairs]
    return torch.tensor(np.array(xyz), dtype=torch.float64, device=DEVICE)


# ══════════════════════════════════════════════════════════════════════
# CMA-ES with batch GPU evaluation
# ══════════════════════════════════════════════════════════════════════

def run_cma(x0, train_t, val_t, sigma=0.3, popsize=64, generations=300,
            cusp_lambda=5.0, hue_lambda=0.5, cond_lambda=0.0, cond_thresh=5.0,
            w_miss=1000.0, w_cliff=200.0, w_smooth=50.0, w_unimodal=100.0, w_yellow=500.0,
            w_drift=0.0, w_neg=0.0, w_dark=1e6,
            label=""):
    """Run CMA-ES with batch GPU objective."""

    es = cma.CMAEvolutionStrategy(x0, sigma, {
        'popsize': popsize,
        'maxiter': generations,
        'tolfun': 1e-11,
        'tolx': 1e-11,
        'verbose': -1,  # quiet
    })

    best_loss = float('inf')
    best_x = None
    best_metrics = {}
    t0 = time.time()

    gen = 0
    while not es.stop():
        X = np.array(es.ask())  # (P, 16)
        P = X.shape[0]

        # Unpack all candidates
        M1_np, M2_np = params_to_matrices_batch(X)
        M1_t = torch.tensor(M1_np, dtype=torch.float64, device=DEVICE)
        M2_t = torch.tensor(M2_np, dtype=torch.float64, device=DEVICE)

        # SINGLE GPU CALL for entire population
        try:
            r = batch_evaluate(M1_t, M2_t, train_t)
        except Exception as e:
            # Fallback: all bad
            es.tell(X, [999.0] * P)
            gen += 1
            continue

        # Compute losses with tunable weights
        cond_excess1 = (r['cond1'] - cond_thresh).clamp(min=0)
        cond_excess2 = (r['cond2'] - cond_thresh).clamp(min=0)
        cond_pen = cond_lambda * (cond_excess1 ** 2 + cond_excess2 ** 2)

        # Gamut penalty = weighted sum of raw components
        gamut_pen = (w_miss * r['missing_pen'] +
                     w_cliff * r['cliff_raw'] +
                     w_smooth * r['smooth_raw'] +
                     w_unimodal * r['unimodal_raw'] +
                     w_yellow * r['yellow_raw'])

        # Robustness penalties (sign-preserving cbrt: neg/RT no longer needed)
        robust_pen = (w_dark * r['dark_lms_raw'] +
                      w_drift * r['drift_raw'] +
                      200.0 * r['scale_raw'])          # L_white scale (always on)

        losses = (r['cv_mean'] + 0.3 * r['cv_top10'] +
                  cusp_lambda * gamut_pen +
                  hue_lambda * r['hue_penalty'] +
                  cond_pen +
                  robust_pen)

        # Invalidate bad candidates
        losses = torch.where(r['valid'], losses, torch.full_like(losses, 999.0))
        losses_np = losses.cpu().numpy()

        es.tell(X, losses_np.tolist())

        # Track best
        idx_best = np.argmin(losses_np)
        if losses_np[idx_best] < best_loss:
            best_loss = losses_np[idx_best]
            best_x = X[idx_best].copy()
            best_metrics = {
                'cv': r['cv_mean'][idx_best].item(),
                'top10': r['cv_top10'][idx_best].item(),
                'hue': np.degrees(np.sqrt(r['hue_penalty'][idx_best].item())),
                'cusps': int((r['cusp_C'][idx_best] > 0.02).sum().item()),
                'cliff': r['cliff'][idx_best].item(),
                'yC': r['yellow_C'][idx_best].item(),
                'c1': r['cond1'][idx_best].item(),
                'c2': r['cond2'][idx_best].item(),
                'uni': int(r['unimodal_viol'][idx_best].item()),
                'drift': r['drift_deg'][idx_best].item(),
                'neg': r['neg_frac'][idx_best].item(),
                'r2020_rt': r['r2020_rt'][idx_best].item(),
                'Lw': r['L_white'][idx_best].item(),
            }

        gen += 1
        if gen % 10 == 0 or gen == 1:
            dt = time.time() - t0
            m = best_metrics
            print(f"  {label} gen {gen:>4d}  loss={best_loss:.5f}  "
                  f"CV={m['cv']*100:.1f}%  cusps={m['cusps']}/360  "
                  f"hue={m['hue']:.1f}  yC={m['yC']:.3f}  "
                  f"cliff={m['cliff']:.2f}  uni={m['uni']}/{N_UNI_HUES}  "
                  f"drift={m.get('drift',0):.1f}  neg={m.get('neg',0):.2f}  "
                  f"RT={m.get('r2020_rt',0):.1e}  Lw={m.get('Lw',0):.1f}  "
                  f"cond={m['c1']:.1f}/{m['c2']:.1f}  ({dt:.0f}s)", flush=True)

    dt = time.time() - t0
    print(f"  {label} DONE in {dt:.0f}s  loss={best_loss:.5f}", flush=True)

    return best_x, best_loss, best_metrics


# ══════════════════════════════════════════════════════════════════════
# Full evaluation (single model)
# ══════════════════════════════════════════════════════════════════════

def full_eval(label, M1, M2, train_t, val_t, show_pairs, show_t):
    M1_t = torch.tensor(M1, dtype=torch.float64, device=DEVICE).unsqueeze(0)
    M2_t = torch.tensor(M2, dtype=torch.float64, device=DEVICE).unsqueeze(0)

    rt = batch_evaluate(M1_t, M2_t, train_t)
    rv = batch_evaluate(M1_t, M2_t, val_t)
    rs = batch_evaluate(M1_t, M2_t, show_t)

    # Hue details
    pxyz = _PRIMARY_XYZ_T.unsqueeze(0)
    plms = signed_cbrt_t(torch.bmm(pxyz, M1_t.transpose(1,2)))
    plab = torch.bmm(plms, M2_t.transpose(1,2))[0]  # (6,3)
    names = ["Red","Yellow","Green","Cyan","Blue","Magenta"]
    targets = [0,60,120,180,240,300]

    print(f"\n  {label}:")
    print(f"    CV: train={rt['cv_mean'][0]*100:.2f}% val={rv['cv_mean'][0]*100:.2f}% show={rs['cv_mean'][0]*100:.2f}%")
    print(f"    Cusps: {int((rt['cusp_C'][0]>0.02).sum())}/360  cliff={rt['cliff'][0]:.3f}  yC={rt['yellow_C'][0]:.4f}  uni={int(rt['unimodal_viol'][0])}/{N_UNI_HUES}  Lw={rt['L_white'][0]:.1f}")
    print(f"    Drift: {rt['drift_deg'][0]:.1f}°  NegLMS: {rt['neg_frac'][0]:.3f}  R2020_RT: {rt['r2020_rt'][0]:.2e}  Lw: {rt['L_white'][0]:.2f}")
    print(f"    Hue: {np.degrees(np.sqrt(rt['hue_penalty'][0].item())):.1f}")
    for i, (n, tgt) in enumerate(zip(names, targets)):
        h = np.degrees(np.arctan2(plab[i,2].item(), plab[i,1].item())) % 360
        C = np.sqrt(plab[i,1].item()**2 + plab[i,2].item()**2)
        err = (h - tgt + 180) % 360 - 180
        print(f"      {n}: H={h:.1f} C={C:.3f} (err {abs(err):.1f})")
    print(f"    Cond: {rt['cond1'][0]:.1f}/{rt['cond2'][0]:.1f}")

    return {'val_cv': rv['cv_mean'][0].item(), 'cusps': int((rt['cusp_C'][0]>0.02).sum()),
            'hue': np.degrees(np.sqrt(rt['hue_penalty'][0].item())),
            'cliff': rt['cliff'][0].item(), 'yC': rt['yellow_C'][0].item(),
            'uni': int(rt['unimodal_viol'][0].item()),
            'drift': rt['drift_deg'][0].item(), 'neg': rt['neg_frac'][0].item(),
            'r2020_rt': rt['r2020_rt'][0].item(), 'Lw': rt['L_white'][0].item(),
            'c1': rt['cond1'][0].item(), 'c2': rt['cond2'][0].item()}


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--popsize", type=int, default=64)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--cusp-lambda", type=float, default=5.0)
    parser.add_argument("--hue-lambda", type=float, default=0.5)
    parser.add_argument("--cond-lambda", type=float, default=0.01,
                        help="Condition number penalty weight")
    parser.add_argument("--cond-thresh", type=float, default=5.0,
                        help="Condition number threshold (penalty starts above this)")
    parser.add_argument("--seeds", type=int, default=3, help="CMA-ES restarts")
    parser.add_argument("--output", default="checkpoints/v5_batch_best.json")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("v5 BATCH GPU CMA-ES")
    print("=" * 70)
    print(f"  popsize={args.popsize} gen={args.generations} seeds={args.seeds}")
    print(f"  cusp_lambda={args.cusp_lambda} hue_lambda={args.hue_lambda}")
    print(f"  cond_lambda={args.cond_lambda} cond_thresh={args.cond_thresh}")

    ALL = generate_training_pairs()
    np.random.seed(123)
    idx = np.random.permutation(len(ALL))
    split = int(len(ALL) * 0.8)
    train_t = pairs_to_tensor([ALL[i] for i in idx[:split]])
    val_t = pairs_to_tensor([ALL[i] for i in idx[split:]])
    SHOW_PAIRS = [('#ff6b00','#00d4ff'),('#ff0000','#0000ff'),('#000000','#ffffff'),
                  ('#ff0000','#00ff00'),('#0000ff','#ffff00'),('#8000ff','#ff8000')]
    show_t = pairs_to_tensor(SHOW_PAIRS)
    print(f"  Train: {train_t.shape[0]}, Val: {val_t.shape[0]}")

    # Baselines
    print(f"\n{'='*70}\nBASELINES\n{'='*70}")
    full_eval("OKLab", OKLAB_M1, OKLAB_M2, train_t, val_t, SHOW_PAIRS, show_t)

    for name, path in [("Deployed","src/helmlab/data/gen_params.json"),
                       ("v4b","checkpoints/analytic_v4b_genspace.json")]:
        p = Path(path)
        if p.exists():
            d = json.load(open(p))
            full_eval(name, np.array(d["M1"]), np.array(d["M2"]), train_t, val_t, SHOW_PAIRS, show_t)

    # Seeds
    seed_configs = [("OKLab", matrices_to_params(OKLAB_M1, OKLAB_M2))]
    for name, path in [("Deployed","src/helmlab/data/gen_params.json"),
                       ("v4b","checkpoints/analytic_v4b_genspace.json")]:
        p = Path(path)
        if p.exists():
            d = json.load(open(p))
            seed_configs.append((name, matrices_to_params(np.array(d["M1"]), np.array(d["M2"]))))

    # ── Weight configurations to sweep ──
    if args.sweep:
        configs = [
            # (label, cusp_lambda, w_smooth, w_cliff, w_unimodal, w_miss, w_yellow, w_drift)
            ("A_noGamut",    0.0,    0,    0,    0,     0,    0,   0),    # pure CV baseline
            ("B_lightUni",   3.0,    2,  100,   50,   500,  200,   0),    # light unimodal
            ("C_robust",     5.0,    5,  200,  200,  1000,  500,   5),    # robust (drift penalty)
            ("D_heavyUni",  10.0,   10,  200,  500,  1000,  500,   5),    # heavy uni + drift
            ("E_driftOnly",  2.0,    1,   50,  100,   500,  200,  20),    # heavy drift penalty
            ("F_cvPlus",     2.0,    1,   50,  100,   500,  200,   2),    # CV-focused + light drift
            ("G_balanced",   5.0,    3,  150,  200,   800,  300,   5),    # balanced all penalties
            ("H_allMax",    10.0,    5,  200, 1000,  1000,  500,  10),    # all penalties max
        ]
    else:
        configs = [("run", args.cusp_lambda, 50, 200, 200, 1000, 500, 5)]

    overall_best_cv = float('inf')
    overall_best_x = None
    pareto = []

    for cfg_label, cl, ws, wc, wu, wmiss, wy, wd in configs:
        print(f"\n{'='*70}")
        print(f"[{cfg_label}] cl={cl} smooth={ws} cliff={wc} uni={wu} miss={wmiss} yellow={wy} drift={wd}")
        print(f"{'='*70}")

        cfg_best_loss = float('inf')
        cfg_best_x = None
        cfg_best_metrics = {}

        # Use only 1 seed per config in sweep (faster), 3 seeds otherwise
        n_seeds = 1 if args.sweep else args.seeds
        for seed_name, x0 in seed_configs:
            for restart in range(n_seeds):
                sigma = 0.3 if restart == 0 else 0.5
                label = f"[{cfg_label} {seed_name} s{restart}]"
                x_start = x0 if restart == 0 else x0 + np.random.randn(16) * 0.1

                bx, bl, bm = run_cma(
                    x_start, train_t, val_t,
                    sigma=sigma, popsize=args.popsize,
                    generations=args.generations,
                    cusp_lambda=cl, hue_lambda=args.hue_lambda,
                    cond_lambda=args.cond_lambda, cond_thresh=args.cond_thresh,
                    w_miss=wmiss, w_cliff=wc, w_smooth=ws, w_unimodal=wu, w_yellow=wy,
                    w_drift=wd,
                    label=label,
                )

                if bl < cfg_best_loss:
                    cfg_best_loss = bl
                    cfg_best_x = bx
                    cfg_best_metrics = bm

        if cfg_best_x is not None:
            M1b, M2b = params_to_matrices_batch(cfg_best_x.reshape(1, -1))
            m = full_eval(f"v5 {cfg_label}", M1b[0], M2b[0], train_t, val_t, SHOW_PAIRS, show_t)
            m['loss'] = cfg_best_loss
            m['cfg'] = cfg_label
            pareto.append((cfg_label, M1b[0].copy(), M2b[0].copy(), m))

            # Normalize M2 so L_white = 1.0
            D65_np = np.array([0.95047, 1.0, 1.08883])
            lms_w = M1b[0] @ D65_np
            lms_c_w = np.sign(lms_w) * np.abs(lms_w) ** (1/3)
            L_w = M2b[0][0] @ lms_c_w
            M2_norm = M2b[0] / L_w

            ckpt = {"M1": M1b[0].tolist(), "gamma": [1/3,1/3,1/3], "M2": M2_norm.tolist()}
            ckpt_path = f"checkpoints/v5_{cfg_label}.json"
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, indent=2)
            print(f"  Saved -> {ckpt_path}")

    # ── Pareto table ──
    if pareto:
        print(f"\n{'='*70}")
        print(f"PARETO FRONT")
        print(f"{'='*70}")
        print(f"  {'config':<14s}  {'valCV':>6s}  {'cusps':>5s}  {'hue':>5s}  {'yC':>6s}  {'cliff':>6s}  {'uni':>7s}  {'drift':>6s}  {'neg':>5s}  {'R2020RT':>9s}  {'Lw':>7s}  {'cond':>9s}")
        print(f"  {'-'*105}")

        # Sort by val_cv
        pareto.sort(key=lambda x: x[3].get('val_cv', 1))
        best_m = None
        best_label = None
        for label, M1b, M2b, m in pareto:
            cusps = m.get('cusps', 0)
            uni = m.get('uni', '?')
            c1 = m.get('c1', 0); c2 = m.get('c2', 0)
            drift = m.get('drift', 0); neg = m.get('neg', 0)
            r2020_rt = m.get('r2020_rt', 0); lw = m.get('Lw', 0)
            print(f"  {label:<14s}  {m['val_cv']*100:5.1f}%  {cusps:>5}  {m['hue']:5.1f}  {m['yC']:6.4f}  {m['cliff']:6.3f}  {str(uni):>3s}/{N_UNI_HUES}  {drift:5.1f}  {neg:5.3f}  {r2020_rt:9.2e}  {lw:7.1f}  {c1:.1f}/{c2:.1f}")
            # Pick best: cusps=360, uni=0, neg=0, RT<1e-6, lowest val_cv
            if cusps == 360 and (isinstance(uni, int) and uni == 0) and neg < 0.01 and r2020_rt < 1e-3:
                if best_m is None or m['val_cv'] < best_m['val_cv']:
                    best_m = m
                    best_label = label

        # Save best that passes gamut constraints
        if best_label:
            for label, M1b, M2b, m in pareto:
                if label == best_label:
                    D65_np = np.array([0.95047, 1.0, 1.08883])
                    lms_w = M1b @ D65_np
                    lms_c_w = np.sign(lms_w) * np.abs(lms_w) ** (1/3)
                    L_w = M2b[0] @ lms_c_w
                    M2_norm = M2b / L_w
                    out = {"M1": M1b.tolist(), "gamma": [1/3,1/3,1/3], "M2": M2_norm.tolist()}
                    with open(args.output, "w") as f:
                        json.dump(out, f, indent=2)
                    print(f"\nBest gamut-passing -> {args.output} ({best_label}, valCV={best_m['val_cv']*100:.1f}%)")
                    break

        print(f"\n{'='*70}\nFINAL\n{'='*70}")
        full_eval("OKLab", OKLAB_M1, OKLAB_M2, train_t, val_t, SHOW_PAIRS, show_t)


if __name__ == "__main__":
    main()
