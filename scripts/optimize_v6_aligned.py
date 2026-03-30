#!/usr/bin/env python
"""v6 GenSpace — TEST-SUITE-ALIGNED batch GPU CMA-ES.

Key changes from v5:
  - Pair generation matches space-test-project exactly (2512 pairs)
  - Simplified CIEDE2000 (matching test suite — no G/T/RT)
  - CV excludes zero-CV pairs (matching test suite)
  - 26 interpolation steps (matching test suite)
  - Optional delta trick (piecewise-linear near zero, +1 param)
  - Seeds from v5_B_lightUni + OKLab + deployed

Usage:
    python scripts/optimize_v6_aligned.py --sweep
    python scripts/optimize_v6_aligned.py --popsize 64 --generations 300
    python scripts/optimize_v6_aligned.py --delta  # enable delta trick (17 params)
"""

import argparse
import json
import math
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

M_P3_TO_XYZ = np.array([
    [0.4865709486482162, 0.26566769316909306, 0.1982172852343625],
    [0.2289745640697488, 0.6917385218365064, 0.079286914093745],
    [0.0, 0.04511338185890264, 1.0439443689009757],
])
M_P3_T = torch.tensor(M_P3_TO_XYZ, dtype=torch.float64, device=DEVICE)

M_REC2020_TO_XYZ = np.array([
    [0.6369580483012914, 0.14461690358620832, 0.1688809751641721],
    [0.2627002120112671, 0.6779980715188708, 0.05930171646986196],
    [0.0, 0.028072693049087428, 1.0609850577107909],
])
M_R2020_T = torch.tensor(M_REC2020_TO_XYZ, dtype=torch.float64, device=DEVICE)

# OKLab matrices (XYZ domain)
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

# v5_B_lightUni (best from previous sweep)
V5B_M1 = np.array([
    [6.245134503607766, -1.7144872128033921, -0.4984085031504023],
    [-0.35067873392843874, 3.8194731167960145, 0.6821921051865574],
    [-0.07772488543980956, 1.0008781874806263, 2.4502468771982593],
])
V5B_M2 = np.array([
    [0.3838489448608338, 0.5940377948499471, -0.3611096101108259],
    [0.4255812128445003, -0.2581927775760107, -0.15630210445969367],
    [-0.1140739017991063, 0.49547154766841783, -0.40808596462884816],
])

_PRIMARY_SRGB = np.array([[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1]], dtype=np.float64)
_TARGET_HUE_RAD = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])

# Rec2020 boundary for negative LMS check
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

_R2020_BOUNDARY_T = torch.tensor(_build_rec2020_boundary(), dtype=torch.float64, device=DEVICE)

# Dark test points
_DARK_XYZ = torch.tensor([
    [0.001, 0.001, 0.001], [0.005, 0.005, 0.005], [0.01, 0.01, 0.01],
    [0.001, 0.0005, 0.002], [0.002, 0.001, 0.003], [0.003, 0.002, 0.001],
    [0.0001, 0.0001, 0.0001], [0.02, 0.02, 0.02],
], dtype=torch.float64, device=DEVICE)


def srgb_to_linear_np(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

_PRIMARY_XYZ = np.array([M_SRGB_TO_XYZ @ srgb_to_linear_np(c) for c in _PRIMARY_SRGB])
_PRIMARY_XYZ_T = torch.tensor(_PRIMARY_XYZ, dtype=torch.float64, device=DEVICE)
_TARGET_HUE_T = torch.tensor(_TARGET_HUE_RAD, dtype=torch.float64, device=DEVICE)


# ══════════════════════════════════════════════════════════════════════
# Pack / Unpack — BATCH (P candidates)
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

def matrices_to_params(M1, M2, use_delta=False):
    """Encode M1/M2 to parameter vector. 16 params (or 17 with delta)."""
    n = 17 if use_delta else 16
    x = np.zeros(n)
    x[:9] = M1.flatten()
    x[9:12] = M2[0]
    s = signed_cbrt_np(M1 @ D65)
    v1, v2 = ortho_basis(s)
    x[12] = M2[1] @ v1; x[13] = M2[1] @ v2
    x[14] = M2[2] @ v1; x[15] = M2[2] @ v2
    if use_delta:
        x[16] = -6.0  # log10(delta) ≈ 1e-6 (start with almost pure cbrt)
    return x

def params_to_matrices_batch(X, use_delta=False):
    """(P, 16/17) -> (P, 3, 3) M1, (P, 3, 3) M2, (P,) delta. NumPy."""
    P = X.shape[0]
    M1 = X[:, :9].reshape(P, 3, 3)
    M2 = np.zeros((P, 3, 3))
    M2[:, 0, :] = X[:, 9:12]

    for i in range(P):
        s = signed_cbrt_np(M1[i] @ D65)
        v1, v2 = ortho_basis(s)
        M2[i, 1] = X[i, 12] * v1 + X[i, 13] * v2
        M2[i, 2] = X[i, 14] * v1 + X[i, 15] * v2

    delta = None
    if use_delta:
        # delta = 10^x[16], clamped to [1e-10, 0.1]
        delta = np.clip(10.0 ** X[:, 16], 1e-10, 0.1)
    return M1, M2, delta


# ══════════════════════════════════════════════════════════════════════
# GPU helpers
# ══════════════════════════════════════════════════════════════════════

def srgb_to_linear_t(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))

def linear_to_srgb_t(c):
    c = c.clamp(0, 1)
    return torch.where(c <= 0.0031308, 12.92 * c,
                       1.055 * c.clamp(min=1e-12).pow(1/2.4) - 0.055)

def signed_cbrt_t(x):
    """Sign-preserving cube root."""
    return x.sign() * x.abs().pow(1/3)

def transfer_forward_t(lms, delta_t=None):
    """Apply transfer function: signed cbrt, optionally with piecewise-linear delta.

    lms: (..., 3) raw LMS values
    delta_t: (P, 1, 1) or None. If given, applies piecewise-linear below |x| < delta.

    f(x) = sign(x) * g(|x|) where:
      g(t) = t^(1/3)                              if t >= delta
      g(t) = (1/3)*delta^(-2/3)*t + (2/3)*delta^(1/3)  if t < delta
    """
    if delta_t is None:
        return signed_cbrt_t(lms)

    ax = lms.abs()
    # Cbrt branch
    cbrt = ax.pow(1/3)
    # Linear branch: slope = (1/3)*delta^(-2/3), intercept = (2/3)*delta^(1/3)
    d13 = delta_t.pow(1/3)
    slope = d13 / (3.0 * delta_t)  # = (1/3) * delta^(-2/3)
    linear = slope * ax + (2.0/3.0) * d13
    result = torch.where(ax >= delta_t, cbrt, linear)
    return lms.sign() * result

def transfer_inverse_t(lms_c, delta_t=None):
    """Inverse of transfer_forward_t."""
    if delta_t is None:
        return lms_c.sign() * lms_c.abs().pow(3)

    ay = lms_c.abs()
    d13 = delta_t.pow(1/3)
    slope = d13 / (3.0 * delta_t)
    # Cubed branch (for y >= delta^(1/3))
    cubed = ay.pow(3)
    # Linear inverse: t = (y - (2/3)*d13) / slope = (y - (2/3)*d13) * 3*delta / d13
    linear_inv = (ay - (2.0/3.0) * d13) / slope.clamp(min=1e-30)
    linear_inv = linear_inv.clamp(min=0)
    result = torch.where(ay >= d13, cubed, linear_inv)
    return lms_c.sign() * result


def xyz_to_cielab_t(xyz):
    """(..., 3) XYZ -> (..., 3) CIE Lab. Matches test suite exactly."""
    r = xyz / D65_T
    delta3 = (6.0 / 29.0) ** 3
    f = torch.where(r > delta3, r.clamp(min=1e-12).pow(1/3),
                    r / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)


def ciede2000_simplified(cl1, cl2):
    """Simplified CIEDE2000 matching test suite. No G/T/RT terms.

    cl1, cl2: (..., 3) CIE Lab. Returns (...) delta E.
    """
    dL = cl2[..., 0] - cl1[..., 0]
    C1 = (cl1[..., 1] ** 2 + cl1[..., 2] ** 2).sqrt()
    C2 = (cl2[..., 1] ** 2 + cl2[..., 2] ** 2).sqrt()
    dC = C2 - C1
    dH = ((cl2[..., 1] - cl1[..., 1]) ** 2 +
          (cl2[..., 2] - cl1[..., 2]) ** 2 - dC ** 2).clamp(min=0).sqrt()
    SL = 1 + 0.015 * (cl1[..., 0] - 50) ** 2 / (20 + (cl1[..., 0] - 50) ** 2).sqrt()
    SC = 1 + 0.045 * C1
    SH = 1 + 0.015 * C1
    return ((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2).sqrt()


# ══════════════════════════════════════════════════════════════════════
# Pair generation — matches space-test-project/core/pairs.py EXACTLY
# ══════════════════════════════════════════════════════════════════════

def _hsv_to_rgb(h, s, v):
    """HSV [0-1] -> RGB [0-1]."""
    if s == 0:
        return v, v, v
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t_ = v * (1.0 - s * (1.0 - f))
    return [(v, t_, p), (q, v, p), (p, v, t_),
            (p, q, v), (t_, p, v), (v, p, q)][i]


def generate_test_suite_pairs():
    """Generate pairs matching space-test-project exactly. Returns (N, 2, 3) XYZ tensor."""
    device = DEVICE
    pairs = []
    labels = []

    def add_srgb(cat, desc, rgb1, rgb2):
        r1 = torch.tensor(rgb1, device=device, dtype=torch.float64)
        r2 = torch.tensor(rgb2, device=device, dtype=torch.float64)
        x1 = M_S2X_T @ srgb_to_linear_t(r1)
        x2 = M_S2X_T @ srgb_to_linear_t(r2)
        pairs.append(torch.stack([x1, x2]))
        labels.append((cat, desc))

    primaries = {"R": [1,0,0], "G": [0,1,0], "B": [0,0,1],
                 "Y": [1,1,0], "C": [0,1,1], "M": [1,0,1]}
    names = list(primaries.keys())

    # Primary combos (15)
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            add_srgb("primary", f"{names[i]}-{names[j]}",
                     primaries[names[i]], primaries[names[j]])

    # To white / black (12)
    for n, rgb in primaries.items():
        add_srgb("to_white", f"{n}->W", rgb, [1,1,1])
        add_srgb("to_black", f"{n}->K", rgb, [0,0,0])

    # Hue sweep — every 5 degrees (72)
    for h_start in range(0, 360, 5):
        h_end = (h_start + 30) % 360
        r1, g1, b1 = _hsv_to_rgb(h_start / 360, 1.0, 1.0)
        r2, g2, b2 = _hsv_to_rgb(h_end / 360, 1.0, 1.0)
        add_srgb("hue_sweep", f"h{h_start}-h{h_end}", [r1,g1,b1], [r2,g2,b2])

    # Saturation sweep (18)
    for h in [0, 60, 120, 180, 240, 300]:
        for s in [1.0, 0.7, 0.4]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, s, 1.0)
            add_srgb("saturation", f"h{h}s{s:.1f}->W", [r1,g1,b1], [1,1,1])

    # Lightness sweep (24 + 6 = 30)
    for h in [0, 60, 120, 180, 240, 300]:
        for v_lo, v_hi in [(0.2, 0.8), (0.1, 0.5), (0.5, 1.0)]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, 0.8, v_lo)
            r2, g2, b2 = _hsv_to_rgb(h / 360, 0.8, v_hi)
            add_srgb("lightness", f"h{h}v{v_lo}-v{v_hi}", [r1,g1,b1], [r2,g2,b2])
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.8, 0.15)
        r2, g2, b2 = _hsv_to_rgb(h / 360, 0.8, 0.95)
        add_srgb("lightness", f"h{h}dark-light", [r1,g1,b1], [r2,g2,b2])

    # Gray pairs (15)
    grays = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(grays)):
        for j in range(i+2, len(grays)):
            add_srgb("gray", f"g{grays[i]:.1f}-g{grays[j]:.1f}",
                     [grays[i]]*3, [grays[j]]*3)

    # Complementary (3)
    for n1, n2 in [("R", "C"), ("G", "M"), ("B", "Y")]:
        add_srgb("complementary", f"{n1}-{n2}", primaries[n1], primaries[n2])

    # Near-achromatic (72)
    for h in range(0, 360, 15):
        for s in [0.05, 0.10, 0.15]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, s, 0.5)
            r2, g2, b2 = _hsv_to_rgb(((h + 30) % 360) / 360, s, 0.5)
            add_srgb("near_achromatic", f"h{h}s{s:.2f}", [r1,g1,b1], [r2,g2,b2])

    # Dark-to-dark (12)
    for h in range(0, 360, 30):
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.6, 0.15)
        r2, g2, b2 = _hsv_to_rgb(((h + 60) % 360) / 360, 0.6, 0.15)
        add_srgb("dark_dark", f"dark_h{h}", [r1,g1,b1], [r2,g2,b2])

    # Pastel (18)
    for h in range(0, 360, 20):
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.25, 0.95)
        r2, g2, b2 = _hsv_to_rgb(((h + 40) % 360) / 360, 0.25, 0.95)
        add_srgb("pastel", f"pastel_h{h}", [r1,g1,b1], [r2,g2,b2])

    # Hue wrap-around (9)
    for h_start in [350, 355, 358]:
        for h_end in [2, 5, 10]:
            r1, g1, b1 = _hsv_to_rgb(h_start / 360, 1.0, 1.0)
            r2, g2, b2 = _hsv_to_rgb(h_end / 360, 1.0, 1.0)
            add_srgb("hue_wrap", f"h{h_start}-h{h_end}", [r1,g1,b1], [r2,g2,b2])

    # L extremes (8)
    for h in [0, 60, 120, 240]:
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.5, 0.02)
        r2, g2, b2 = _hsv_to_rgb(h / 360, 0.5, 0.08)
        add_srgb("L_extreme_dark", f"vdark_h{h}", [r1,g1,b1], [r2,g2,b2])
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.3, 0.93)
        r2, g2, b2 = _hsv_to_rgb(h / 360, 0.3, 0.99)
        add_srgb("L_extreme_bright", f"vbright_h{h}", [r1,g1,b1], [r2,g2,b2])

    # Gamut boundary stress (96)
    for h in range(0, 360, 15):
        for v in [0.3, 0.5, 0.7, 0.9]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, 0.95, v)
            r2, g2, b2 = _hsv_to_rgb(h / 360, 0.95, min(v + 0.2, 1.0))
            add_srgb("boundary_srgb", f"bnd_h{h}_v{v}", [r1,g1,b1], [r2,g2,b2])

    # Random sRGB (1000) — same seed as test suite
    gen = torch.Generator(device=device).manual_seed(42)
    for k in range(1000):
        rgb1 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        rgb2 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        x1 = M_S2X_T @ srgb_to_linear_t(rgb1)
        x2 = M_S2X_T @ srgb_to_linear_t(rgb2)
        pairs.append(torch.stack([x1, x2]))
        labels.append(("random_srgb", f"rnd_s{k}"))

    # ── P3 pairs ──
    # P3 primaries (15)
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            rgb1_t = torch.tensor(primaries[names[i]], device=device, dtype=torch.float64)
            rgb2_t = torch.tensor(primaries[names[j]], device=device, dtype=torch.float64)
            x1 = M_P3_T @ srgb_to_linear_t(rgb1_t)
            x2 = M_P3_T @ srgb_to_linear_t(rgb2_t)
            pairs.append(torch.stack([x1, x2]))
            labels.append(("p3_primary", f"P3_{names[i]}-{names[j]}"))

    # P3 to sRGB cross-gamut (6)
    for n, rgb in primaries.items():
        rgb_t = torch.tensor(rgb, device=device, dtype=torch.float64)
        x1 = M_P3_T @ srgb_to_linear_t(rgb_t)
        x2 = M_S2X_T @ srgb_to_linear_t(torch.tensor([1.,1.,1.], device=device, dtype=torch.float64))
        pairs.append(torch.stack([x1, x2]))
        labels.append(("p3_to_srgb", f"P3_{n}->sRGB_W"))

    # P3 to_white / to_black (12)
    for n, rgb in primaries.items():
        rgb_t = torch.tensor(rgb, device=device, dtype=torch.float64)
        x1 = M_P3_T @ srgb_to_linear_t(rgb_t)
        x_w = M_P3_T @ srgb_to_linear_t(torch.ones(3, device=device, dtype=torch.float64))
        x_k = M_P3_T @ srgb_to_linear_t(torch.zeros(3, device=device, dtype=torch.float64))
        pairs.append(torch.stack([x1, x_w]))
        labels.append(("p3_to_white", f"P3_{n}->W"))
        pairs.append(torch.stack([x1, x_k]))
        labels.append(("p3_to_black", f"P3_{n}->K"))

    # P3 hue sweep (24)
    for h_start in range(0, 360, 15):
        h_end = (h_start + 30) % 360
        r1, g1, b1 = _hsv_to_rgb(h_start / 360, 1.0, 1.0)
        r2, g2, b2 = _hsv_to_rgb(h_end / 360, 1.0, 1.0)
        x1 = M_P3_T @ srgb_to_linear_t(torch.tensor([r1,g1,b1], device=device, dtype=torch.float64))
        x2 = M_P3_T @ srgb_to_linear_t(torch.tensor([r2,g2,b2], device=device, dtype=torch.float64))
        pairs.append(torch.stack([x1, x2]))
        labels.append(("p3_hue_sweep", f"P3_h{h_start}-h{h_end}"))

    # P3 near-achromatic (24)
    for h in range(0, 360, 30):
        for s in [0.05, 0.10]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, s, 0.5)
            r2, g2, b2 = _hsv_to_rgb(((h + 30) % 360) / 360, s, 0.5)
            x1 = M_P3_T @ srgb_to_linear_t(torch.tensor([r1,g1,b1], device=device, dtype=torch.float64))
            x2 = M_P3_T @ srgb_to_linear_t(torch.tensor([r2,g2,b2], device=device, dtype=torch.float64))
            pairs.append(torch.stack([x1, x2]))
            labels.append(("p3_near_achromatic", f"P3_na_h{h}_s{s}"))

    # Random P3 (500)
    for k in range(500):
        rgb1 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        rgb2 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        x1 = M_P3_T @ srgb_to_linear_t(rgb1)
        x2 = M_P3_T @ srgb_to_linear_t(rgb2)
        pairs.append(torch.stack([x1, x2]))
        labels.append(("random_p3", f"rnd_p3_{k}"))

    # ── Rec.2020 pairs ──
    # Rec2020→P3 cross-gamut (6)
    for n, rgb in primaries.items():
        rgb_t = torch.tensor(rgb, device=device, dtype=torch.float64)
        x1 = M_R2020_T @ srgb_to_linear_t(rgb_t)
        x2 = M_P3_T @ srgb_to_linear_t(rgb_t)
        pairs.append(torch.stack([x1, x2]))
        labels.append(("rec2020_to_p3", f"R2020_{n}->P3_{n}"))

    # Rec2020 primaries (15)
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            rgb1_t = torch.tensor(primaries[names[i]], device=device, dtype=torch.float64)
            rgb2_t = torch.tensor(primaries[names[j]], device=device, dtype=torch.float64)
            x1 = M_R2020_T @ srgb_to_linear_t(rgb1_t)
            x2 = M_R2020_T @ srgb_to_linear_t(rgb2_t)
            pairs.append(torch.stack([x1, x2]))
            labels.append(("rec2020_primary", f"R2020_{names[i]}-{names[j]}"))

    # Rec2020 to_white/to_black (12)
    for n, rgb in primaries.items():
        rgb_t = torch.tensor(rgb, device=device, dtype=torch.float64)
        x1 = M_R2020_T @ srgb_to_linear_t(rgb_t)
        x_w = M_R2020_T @ srgb_to_linear_t(torch.ones(3, device=device, dtype=torch.float64))
        x_k = M_R2020_T @ srgb_to_linear_t(torch.zeros(3, device=device, dtype=torch.float64))
        pairs.append(torch.stack([x1, x_w]))
        labels.append(("rec2020_to_white", f"R2020_{n}->W"))
        pairs.append(torch.stack([x1, x_k]))
        labels.append(("rec2020_to_black", f"R2020_{n}->K"))

    # Rec2020 hue sweep (24)
    for h_start in range(0, 360, 15):
        h_end = (h_start + 30) % 360
        r1, g1, b1 = _hsv_to_rgb(h_start / 360, 1.0, 1.0)
        r2, g2, b2 = _hsv_to_rgb(h_end / 360, 1.0, 1.0)
        x1 = M_R2020_T @ srgb_to_linear_t(torch.tensor([r1,g1,b1], device=device, dtype=torch.float64))
        x2 = M_R2020_T @ srgb_to_linear_t(torch.tensor([r2,g2,b2], device=device, dtype=torch.float64))
        pairs.append(torch.stack([x1, x2]))
        labels.append(("rec2020_hue_sweep", f"R2020_h{h_start}-h{h_end}"))

    # Random Rec2020 (500)
    for k in range(500):
        rgb1 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        rgb2 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        x1 = M_R2020_T @ srgb_to_linear_t(rgb1)
        x2 = M_R2020_T @ srgb_to_linear_t(rgb2)
        pairs.append(torch.stack([x1, x2]))
        labels.append(("random_rec2020", f"rnd_r2020_{k}"))

    result = torch.stack(pairs)  # (N, 2, 3)
    print(f"  Generated {result.shape[0]} pairs ({len(set(c for c,_ in labels))} categories)")
    return result, labels


# ══════════════════════════════════════════════════════════════════════
# BATCH objective: P candidates x N pairs — matches test suite
# ══════════════════════════════════════════════════════════════════════

STEPS = 26  # matches test suite (was 25 in v5)
_T_FRACS = torch.linspace(0, 1, STEPS, dtype=torch.float64, device=DEVICE)

# Gamut boundary (sRGB)
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

_BXYZ = _build_boundary()
N_HUE_BINS = 360
_HUE_EDGES = torch.linspace(-torch.pi, torch.pi, N_HUE_BINS + 1, dtype=torch.float64, device=DEVICE)

# Unimodality
N_UNI_HUES = 72
N_UNI_L = 50
N_UNI_C = 40
UNI_HUE_BATCH = 12


def compute_unimodal(M1_t, M2_t, L_white, delta_t=None):
    """Boundary unimodality violations for P candidates."""
    P = M1_t.shape[0]
    device = M1_t.device

    M2_norm = M2_t / L_white.view(P, 1, 1).clamp(min=1e-10)
    M2_norm_inv = torch.linalg.inv(M2_norm)
    M1_inv = torch.linalg.inv(M1_t)

    Ls = torch.linspace(0.02, 0.998, N_UNI_L, device=device, dtype=torch.float64)
    Cs = torch.linspace(0.001, 0.5, N_UNI_C, device=device, dtype=torch.float64)
    mc_all = torch.zeros(P, N_UNI_HUES, N_UNI_L, device=device, dtype=torch.float64)

    for hs in range(0, N_UNI_HUES, UNI_HUE_BATCH):
        he = min(hs + UNI_HUE_BATCH, N_UNI_HUES)
        nh = he - hs
        angles = torch.arange(hs, he, device=device, dtype=torch.float64) * (2 * torch.pi / N_UNI_HUES)
        ch = torch.cos(angles).view(nh, 1, 1)
        sh = torch.sin(angles).view(nh, 1, 1)
        Le = Ls.view(1, N_UNI_L, 1).expand(nh, N_UNI_L, N_UNI_C)
        Ce = Cs.view(1, 1, N_UNI_C).expand(nh, N_UNI_L, N_UNI_C)
        lab = torch.stack([Le, Ce * ch, Ce * sh], dim=-1)
        lab_flat = lab.reshape(-1, 3)
        K = lab_flat.shape[0]

        lms_c = torch.matmul(lab_flat.unsqueeze(0), M2_norm_inv.transpose(1, 2))
        lms = transfer_inverse_t(lms_c, delta_t)
        xyz = torch.matmul(lms, M1_inv.transpose(1, 2))
        lin = torch.matmul(xyz, M_X2S_T.T)
        ok = ((lin >= -0.002) & (lin <= 1.002)).all(dim=2)
        ok = ok.reshape(P, nh, N_UNI_L, N_UNI_C)
        Cs_exp = Cs.view(1, 1, 1, N_UNI_C).expand(P, nh, N_UNI_L, N_UNI_C)
        mc = torch.where(ok, Cs_exp, torch.zeros_like(Cs_exp)).max(dim=3).values
        mc_all[:, hs:he] = mc

    ci = mc_all.argmax(dim=2)
    mc_diff = mc_all[:, :, 1:] - mc_all[:, :, :-1]
    idx = torch.arange(N_UNI_L - 1, device=device).view(1, 1, N_UNI_L - 1)
    after_cusp = idx >= ci.unsqueeze(2)
    viol_per_hue = ((mc_diff > 0.001) & after_cusp).any(dim=2)
    unimodal_viol = viol_per_hue.sum(dim=1)
    positive_dC = mc_diff.clamp(min=0) * after_cusp.float()
    unimodal_raw = positive_dC.sum(dim=[1, 2]) + unimodal_viol.float() / N_UNI_HUES

    return unimodal_viol, unimodal_raw


def batch_evaluate(M1_t, M2_t, pairs_t, delta_t=None):
    """Evaluate P candidates on N pairs — ALIGNED with test suite.

    M1_t: (P, 3, 3), M2_t: (P, 3, 3), pairs_t: (N, 2, 3)
    delta_t: (P, 1, 1) or None for piecewise-linear transfer

    Returns dict of (P,) tensors.
    """
    P = M1_t.shape[0]
    N = pairs_t.shape[0]
    S = STEPS

    # ── 1. Gradient CV (test-suite-aligned) ───────────────────
    xyz1 = pairs_t[:, 0].unsqueeze(0).expand(P, -1, -1)
    xyz2 = pairs_t[:, 1].unsqueeze(0).expand(P, -1, -1)

    lms1 = transfer_forward_t(torch.bmm(xyz1, M1_t.transpose(1, 2)), delta_t)
    lms2 = transfer_forward_t(torch.bmm(xyz2, M1_t.transpose(1, 2)), delta_t)
    lab1 = torch.bmm(lms1, M2_t.transpose(1, 2))
    lab2 = torch.bmm(lms2, M2_t.transpose(1, 2))

    # Interpolate: (P, N, S, 3)
    t = _T_FRACS.view(1, 1, S, 1)
    lab_interp = lab1.unsqueeze(2) * (1 - t) + lab2.unsqueeze(2) * t

    # Inverse: Lab -> LMS_c -> LMS -> XYZ -> sRGB -> quantize -> CIE Lab
    M2i_t = torch.linalg.inv(M2_t)
    M1i_t = torch.linalg.inv(M1_t)
    flat = lab_interp.reshape(P, N * S, 3)
    lms_c = torch.bmm(flat, M2i_t.transpose(1, 2))
    lms = transfer_inverse_t(lms_c, delta_t)
    xyz = torch.bmm(lms, M1i_t.transpose(1, 2))

    # Test-suite-aligned quantization: clamp linear sRGB to [0,1] first
    rgb_lin = torch.matmul(xyz, M_X2S_T.T).clamp(0, 1)
    rgb_srgb = linear_to_srgb_t(rgb_lin)
    rgb8 = (rgb_srgb * 255).round() / 255.0
    rgb_q = srgb_to_linear_t(rgb8)
    xyz_q = torch.matmul(rgb_q, M_S2X_T.T)
    cielab = xyz_to_cielab_t(xyz_q.clamp(min=1e-10))
    cielab = cielab.reshape(P, N, S, 3)

    # Simplified CIEDE2000 (matching test suite)
    des = ciede2000_simplified(cielab[:, :, :-1], cielab[:, :, 1:])  # (P, N, S-1)

    mean_de = des.mean(dim=2)
    std_de = des.std(dim=2)
    ok = mean_de > 0.001  # test suite threshold
    cvs = torch.where(ok, std_de / mean_de, torch.zeros_like(mean_de))

    # CV aggregation: exclude zero-CV pairs (matching test suite)
    valid_mask = cvs > 0  # (P, N)
    valid_counts = valid_mask.float().sum(dim=1).clamp(min=1)
    cv_mean = (cvs * valid_mask.float()).sum(dim=1) / valid_counts  # (P,)
    # Also compute top-10% of valid CVs
    cv_sorted = cvs.sort(dim=1, descending=True).values
    n_top = max(1, N // 10)
    cv_top10 = cv_sorted[:, :n_top].mean(dim=1)

    # ── 1b. Drift penalty ──
    h_steps = torch.atan2(cielab[:, :, :, 2], cielab[:, :, :, 1])
    C_steps = (cielab[:, :, :, 1]**2 + cielab[:, :, :, 2]**2).sqrt()
    h_start = h_steps[:, :, 0:1]
    h_end = h_steps[:, :, -1:]
    h_diff = torch.atan2((h_end - h_start).sin(), (h_end - h_start).cos())
    t_drift = _T_FRACS.view(1, 1, S)
    h_expected = h_start + t_drift * h_diff
    dev = torch.atan2((h_steps - h_expected).sin(), (h_steps - h_expected).cos()).abs()
    dev = dev * (C_steps > 5.0).float()
    max_dev_per_pair = dev.max(dim=2).values
    drift_raw = max_dev_per_pair.mean(dim=1)

    # ── 2. Cusp profiling ──
    B = _BXYZ.shape[0]
    bxyz = _BXYZ.unsqueeze(0).expand(P, -1, -1)
    blms = transfer_forward_t(torch.bmm(bxyz, M1_t.transpose(1, 2)), delta_t)
    blab = torch.bmm(blms, M2_t.transpose(1, 2))

    # L_white normalization
    d65_batch = D65_T.unsqueeze(0).expand(P, -1).unsqueeze(1)
    lms_w = torch.bmm(d65_batch, M1_t.transpose(1, 2)).squeeze(1)
    lms_w = transfer_forward_t(lms_w.unsqueeze(1), delta_t).squeeze(1) if delta_t is not None else signed_cbrt_t(lms_w)
    L_white = (lms_w * M2_t[:, 0, :]).sum(dim=1)
    scale = L_white.clamp(min=1e-10).unsqueeze(1).unsqueeze(2)
    blab = blab / scale

    ba = blab[:, :, 1]; bb = blab[:, :, 2]
    bL = blab[:, :, 0]
    bC = (ba**2 + bb**2).sqrt()
    bh = torch.atan2(bb, ba)

    bin_idx = torch.bucketize(bh, _HUE_EDGES) - 1
    bin_idx = bin_idx.clamp(0, N_HUE_BINS - 1)

    cusp_C = torch.zeros(P, N_HUE_BINS, dtype=torch.float64, device=DEVICE)
    cusp_L = torch.zeros(P, N_HUE_BINS, dtype=torch.float64, device=DEVICE)
    for i in range(N_HUE_BINS):
        mask = (bin_idx == i)
        masked_C = bC * mask.float()
        max_C, max_idx = masked_C.max(dim=1)
        cusp_C[:, i] = max_C
        cusp_L[:, i] = bL.gather(1, max_idx.unsqueeze(1)).squeeze(1)

    min_C_thresh = 0.02
    deficit = (min_C_thresh - cusp_C).clamp(min=0)
    missing_pen = (deficit ** 2).sum(dim=1)

    shifted = torch.roll(cusp_C, 1, dims=1)
    both_valid = (cusp_C > 0.005) & (shifted > 0.005)
    ratios = torch.where(both_valid, cusp_C / shifted.clamp(min=1e-10), torch.ones_like(cusp_C))
    ratios = torch.where(ratios > 1, 1 / ratios, ratios)
    min_ratio = ratios.min(dim=1).values
    cliff = 1.0 - min_ratio
    cliff_raw = (cliff - 0.30).clamp(min=0) ** 2

    dC = cusp_C[:, 1:] - cusp_C[:, :-1]
    smooth_raw = (dC ** 2).sum(dim=1)

    unimodal_viol, unimodal_raw = compute_unimodal(M1_t, M2_t, L_white, delta_t)

    ybin = int((np.radians(85) + np.pi) / (2*np.pi) * N_HUE_BINS) % N_HUE_BINS
    yellow_C = cusp_C[:, ybin]
    yellow_raw = (0.15 - yellow_C).clamp(min=0) ** 2

    # ── 3. Hue penalty ──
    pxyz = _PRIMARY_XYZ_T.unsqueeze(0).expand(P, -1, -1)
    plms = transfer_forward_t(torch.bmm(pxyz, M1_t.transpose(1, 2)), delta_t)
    plab = torch.bmm(plms, M2_t.transpose(1, 2))
    ph = torch.atan2(plab[:, :, 2], plab[:, :, 1])
    diff = ph - _TARGET_HUE_T.unsqueeze(0)
    angular_err = torch.atan2(diff.sin(), diff.cos())
    hue_penalty = (angular_err ** 2).mean(dim=1)

    # ── 4. Conditioning ──
    s1 = torch.linalg.svdvals(M1_t)
    s2 = torch.linalg.svdvals(M2_t)
    cond1 = s1[:, 0] / s1[:, 2].clamp(min=1e-10)
    cond2 = s2[:, 0] / s2[:, 2].clamp(min=1e-10)

    # ── 5. Rec2020 round-trip ──
    R = _R2020_BOUNDARY_T.shape[0]
    r2020_xyz = _R2020_BOUNDARY_T.unsqueeze(0).expand(P, -1, -1)
    r2020_lms = torch.bmm(r2020_xyz, M1_t.transpose(1, 2))
    r2020_lms_c = transfer_forward_t(r2020_lms, delta_t)
    r2020_lab = torch.bmm(r2020_lms_c, M2_t.transpose(1, 2))
    r2020_lms_c_inv = torch.bmm(r2020_lab, M2i_t.transpose(1, 2))
    r2020_lms_inv = transfer_inverse_t(r2020_lms_c_inv, delta_t)
    r2020_xyz_inv = torch.bmm(r2020_lms_inv, M1i_t.transpose(1, 2))
    rt_err = (r2020_xyz - r2020_xyz_inv).abs().max(dim=2).values.max(dim=1).values

    # ── 6. L_white scale ──
    lw_log = (L_white / 1.0).clamp(min=1e-10).log10()
    scale_raw = (lw_log.abs() - 0.5).clamp(min=0) ** 2

    # ── 7. Dark region stability ──
    dark_xyz = _DARK_XYZ.unsqueeze(0).expand(P, -1, -1)
    dark_lms = torch.bmm(dark_xyz, M1_t.transpose(1, 2))
    min_dark_abs = dark_lms.abs().min(dim=2).values.min(dim=1).values
    dark_lms_raw = (1e-5 - min_dark_abs).clamp(min=0) ** 2

    # Valid mask
    valid = (cond1 < 50) & (cond2 < 50) & (cv_mean < 5.0)

    return {
        'cv_mean': cv_mean, 'cv_top10': cv_top10,
        'hue_penalty': hue_penalty,
        'valid': valid, 'cusp_C': cusp_C, 'cusp_L': cusp_L,
        'cond1': cond1, 'cond2': cond2,
        'cliff': cliff, 'yellow_C': yellow_C,
        'unimodal_viol': unimodal_viol, 'L_white': L_white,
        'missing_pen': missing_pen, 'cliff_raw': cliff_raw,
        'smooth_raw': smooth_raw, 'unimodal_raw': unimodal_raw,
        'yellow_raw': yellow_raw,
        'drift_raw': drift_raw, 'dark_lms_raw': dark_lms_raw,
        'r2020_rt_raw': rt_err, 'scale_raw': scale_raw,
        'drift_deg': drift_raw * (180.0 / torch.pi),
        'r2020_rt': rt_err,
    }


# ══════════════════════════════════════════════════════════════════════
# CMA-ES
# ══════════════════════════════════════════════════════════════════════

def run_cma(x0, pairs_t, sigma=0.3, popsize=64, generations=300,
            cusp_lambda=5.0, hue_lambda=0.5, cond_lambda=0.01, cond_thresh=5.0,
            w_miss=1000.0, w_cliff=200.0, w_smooth=50.0, w_unimodal=100.0, w_yellow=500.0,
            w_drift=0.0, w_dark=1e6, use_delta=False, label=""):
    """Run CMA-ES with batch GPU objective."""

    es = cma.CMAEvolutionStrategy(x0, sigma, {
        'popsize': popsize,
        'maxiter': generations,
        'tolfun': 1e-11,
        'tolx': 1e-11,
        'verbose': -1,
    })

    best_loss = float('inf')
    best_x = None
    best_metrics = {}
    t0 = time.time()
    n_params = len(x0)

    gen = 0
    while not es.stop():
        X = np.array(es.ask())
        P = X.shape[0]

        M1_np, M2_np, delta_np = params_to_matrices_batch(X, use_delta)
        M1_t = torch.tensor(M1_np, dtype=torch.float64, device=DEVICE)
        M2_t = torch.tensor(M2_np, dtype=torch.float64, device=DEVICE)

        delta_t = None
        if use_delta and delta_np is not None:
            delta_t = torch.tensor(delta_np, dtype=torch.float64, device=DEVICE).view(P, 1, 1)

        try:
            r = batch_evaluate(M1_t, M2_t, pairs_t, delta_t)
        except Exception as e:
            if gen == 0:
                print(f"  {label} gen {gen} ERROR: {e}", flush=True)
            es.tell(X, [999.0] * P)
            gen += 1
            continue

        cond_excess1 = (r['cond1'] - cond_thresh).clamp(min=0)
        cond_excess2 = (r['cond2'] - cond_thresh).clamp(min=0)
        cond_pen = cond_lambda * (cond_excess1 ** 2 + cond_excess2 ** 2)

        gamut_pen = (w_miss * r['missing_pen'] +
                     w_cliff * r['cliff_raw'] +
                     w_smooth * r['smooth_raw'] +
                     w_unimodal * r['unimodal_raw'] +
                     w_yellow * r['yellow_raw'])

        robust_pen = (w_dark * r['dark_lms_raw'] +
                      w_drift * r['drift_raw'] +
                      200.0 * r['scale_raw'])

        losses = (r['cv_mean'] + 0.3 * r['cv_top10'] +
                  cusp_lambda * gamut_pen +
                  hue_lambda * r['hue_penalty'] +
                  cond_pen + robust_pen)

        losses = torch.where(r['valid'], losses, torch.full_like(losses, 999.0))
        losses_np = losses.cpu().numpy()

        es.tell(X, losses_np.tolist())

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
                  f"drift={m.get('drift',0):.1f}  "
                  f"RT={m.get('r2020_rt',0):.1e}  Lw={m.get('Lw',0):.1f}  "
                  f"cond={m['c1']:.1f}/{m['c2']:.1f}  ({dt:.0f}s)", flush=True)

    dt = time.time() - t0
    print(f"  {label} DONE in {dt:.0f}s  loss={best_loss:.5f}", flush=True)
    return best_x, best_loss, best_metrics


# ══════════════════════════════════════════════════════════════════════
# Full evaluation
# ══════════════════════════════════════════════════════════════════════

def full_eval(label, M1, M2, pairs_t, delta_val=None, use_delta=False):
    M1_t = torch.tensor(M1, dtype=torch.float64, device=DEVICE).unsqueeze(0)
    M2_t = torch.tensor(M2, dtype=torch.float64, device=DEVICE).unsqueeze(0)

    delta_t = None
    if use_delta and delta_val is not None:
        delta_t = torch.tensor([[delta_val]], dtype=torch.float64, device=DEVICE).view(1, 1, 1)

    r = batch_evaluate(M1_t, M2_t, pairs_t, delta_t)

    # Hue details
    pxyz = _PRIMARY_XYZ_T.unsqueeze(0)
    plms = transfer_forward_t(torch.bmm(pxyz, M1_t.transpose(1, 2)), delta_t)
    plab = torch.bmm(plms, M2_t.transpose(1, 2))[0]
    names = ["Red", "Yellow", "Green", "Cyan", "Blue", "Magenta"]
    targets = [0, 60, 120, 180, 240, 300]

    print(f"\n  {label}:")
    print(f"    CV: {r['cv_mean'][0]*100:.2f}% (top10: {r['cv_top10'][0]*100:.2f}%)")
    print(f"    Cusps: {int((r['cusp_C'][0]>0.02).sum())}/360  cliff={r['cliff'][0]:.3f}  "
          f"yC={r['yellow_C'][0]:.4f}  uni={int(r['unimodal_viol'][0])}/{N_UNI_HUES}")
    print(f"    Drift: {r['drift_deg'][0]:.1f}°  R2020_RT: {r['r2020_rt'][0]:.2e}  Lw: {r['L_white'][0]:.2f}")
    print(f"    Hue RMS: {np.degrees(np.sqrt(r['hue_penalty'][0].item())):.1f}°")
    for i, (n, tgt) in enumerate(zip(names, targets)):
        h = np.degrees(np.arctan2(plab[i,2].item(), plab[i,1].item())) % 360
        C = np.sqrt(plab[i,1].item()**2 + plab[i,2].item()**2)
        err = (h - tgt + 180) % 360 - 180
        print(f"      {n}: H={h:.1f} C={C:.3f} (err {abs(err):.1f}°)")
    print(f"    Cond: {r['cond1'][0]:.1f}/{r['cond2'][0]:.1f}")

    return {
        'cv': r['cv_mean'][0].item(), 'top10': r['cv_top10'][0].item(),
        'cusps': int((r['cusp_C'][0]>0.02).sum()),
        'hue': np.degrees(np.sqrt(r['hue_penalty'][0].item())),
        'cliff': r['cliff'][0].item(), 'yC': r['yellow_C'][0].item(),
        'uni': int(r['unimodal_viol'][0].item()),
        'drift': r['drift_deg'][0].item(),
        'r2020_rt': r['r2020_rt'][0].item(), 'Lw': r['L_white'][0].item(),
        'c1': r['cond1'][0].item(), 'c2': r['cond2'][0].item(),
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--popsize", type=int, default=64)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--hue-lambda", type=float, default=0.5)
    parser.add_argument("--cond-lambda", type=float, default=0.01)
    parser.add_argument("--cond-thresh", type=float, default=5.0)
    parser.add_argument("--output", default="checkpoints/v6_best.json")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--delta", action="store_true",
                        help="Enable piecewise-linear near zero (17 params)")
    args = parser.parse_args()

    use_delta = args.delta

    print("=" * 70)
    print(f"v6 ALIGNED GPU CMA-ES ({'17 params (delta)' if use_delta else '16 params'})")
    print("=" * 70)
    print(f"  popsize={args.popsize} gen={args.generations}")
    print(f"  hue_lambda={args.hue_lambda} cond_lambda={args.cond_lambda}")
    print(f"  Aligned: simplified CIEDE2000, 26 steps, zero-CV excluded, 2500+ pairs")

    # Generate test-suite-aligned pairs
    pairs_t, labels = generate_test_suite_pairs()
    print(f"  Total pairs: {pairs_t.shape[0]}")

    # Baselines
    print(f"\n{'='*70}\nBASELINES (test-suite-aligned CV)\n{'='*70}")
    full_eval("OKLab", OKLAB_M1, OKLAB_M2, pairs_t)
    full_eval("v5_B_lightUni", V5B_M1, V5B_M2, pairs_t)

    for name, path in [("Deployed", "src/helmlab/data/gen_params.json")]:
        p = Path(path)
        if p.exists():
            d = json.load(open(p))
            full_eval(name, np.array(d["M1"]), np.array(d["M2"]), pairs_t)

    # Seeds: v5B + OKLab + deployed
    seed_configs = [
        ("v5B", matrices_to_params(V5B_M1, V5B_M2, use_delta)),
        ("OKLab", matrices_to_params(OKLAB_M1, OKLAB_M2, use_delta)),
    ]
    for name, path in [("Deployed", "src/helmlab/data/gen_params.json")]:
        p = Path(path)
        if p.exists():
            d = json.load(open(p))
            seed_configs.append((name, matrices_to_params(
                np.array(d["M1"]), np.array(d["M2"]), use_delta)))

    # Sweep configs
    if args.sweep:
        configs = [
            # (label, cusp_lambda, w_smooth, w_cliff, w_unimodal, w_miss, w_yellow, w_drift)
            ("A_noGamut",    0.0,    0,    0,    0,     0,    0,   0),
            ("B_lightUni",   3.0,    2,  100,   50,   500,  200,   0),
            ("C_robust",     5.0,    5,  200,  200,  1000,  500,   5),
            ("D_heavyUni",  10.0,   10,  200,  500,  1000,  500,   5),
            ("E_driftOnly",  2.0,    1,   50,  100,   500,  200,  20),
            ("F_cvPlus",     2.0,    1,   50,  100,   500,  200,   2),
            ("G_balanced",   5.0,    3,  150,  200,   800,  300,   5),
            ("H_allMax",    10.0,    5,  200, 1000,  1000,  500,  10),
        ]
    else:
        configs = [("run", 5.0, 3, 150, 200, 800, 300, 5)]

    pareto = []

    for cfg_label, cl, ws, wc, wu, wmiss, wy, wd in configs:
        print(f"\n{'='*70}")
        print(f"[{cfg_label}] cl={cl} smooth={ws} cliff={wc} uni={wu} miss={wmiss} yellow={wy} drift={wd}")
        print(f"{'='*70}")

        cfg_best_loss = float('inf')
        cfg_best_x = None

        n_seeds = 1 if args.sweep else 3
        for seed_name, x0 in seed_configs:
            for restart in range(n_seeds):
                sigma = 0.3 if restart == 0 else 0.5
                lab = f"[{cfg_label} {seed_name} s{restart}]"
                x_start = x0 if restart == 0 else x0 + np.random.randn(len(x0)) * 0.1

                bx, bl, bm = run_cma(
                    x_start, pairs_t,
                    sigma=sigma, popsize=args.popsize,
                    generations=args.generations,
                    cusp_lambda=cl, hue_lambda=args.hue_lambda,
                    cond_lambda=args.cond_lambda, cond_thresh=args.cond_thresh,
                    w_miss=wmiss, w_cliff=wc, w_smooth=ws, w_unimodal=wu, w_yellow=wy,
                    w_drift=wd, use_delta=use_delta,
                    label=lab,
                )

                if bl < cfg_best_loss:
                    cfg_best_loss = bl
                    cfg_best_x = bx

        if cfg_best_x is not None:
            M1b, M2b, delta_b = params_to_matrices_batch(cfg_best_x.reshape(1, -1), use_delta)
            delta_val = delta_b[0] if delta_b is not None else None
            m = full_eval(f"v6 {cfg_label}", M1b[0], M2b[0], pairs_t,
                          delta_val=delta_val, use_delta=use_delta)
            m['loss'] = cfg_best_loss
            m['cfg'] = cfg_label
            pareto.append((cfg_label, M1b[0].copy(), M2b[0].copy(), delta_val, m))

            # Save checkpoint (normalize M2 so L_white = 1)
            lms_w = M1b[0] @ D65
            if use_delta and delta_val is not None:
                ax = np.abs(lms_w)
                d13 = delta_val ** (1/3)
                slope = d13 / (3 * delta_val)
                lms_c_w = np.where(ax >= delta_val,
                                   np.sign(lms_w) * ax ** (1/3),
                                   np.sign(lms_w) * (slope * ax + (2/3) * d13))
            else:
                lms_c_w = np.sign(lms_w) * np.abs(lms_w) ** (1/3)
            L_w = M2b[0][0] @ lms_c_w
            M2_norm = M2b[0] / L_w

            ckpt = {"M1": M1b[0].tolist(), "gamma": [1/3, 1/3, 1/3], "M2": M2_norm.tolist()}
            if use_delta and delta_val is not None:
                ckpt["delta"] = float(delta_val)
            ckpt_path = f"checkpoints/v6_{cfg_label}.json"
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, indent=2)
            print(f"  Saved -> {ckpt_path}")

    # Pareto table
    if pareto:
        print(f"\n{'='*70}")
        print(f"PARETO FRONT (test-suite-aligned)")
        print(f"{'='*70}")
        print(f"  {'config':<14s}  {'CV':>6s}  {'cusps':>5s}  {'hue':>5s}  {'yC':>6s}  {'cliff':>6s}  {'uni':>7s}  {'drift':>6s}  {'RT':>9s}  {'Lw':>7s}  {'cond':>9s}")
        print(f"  {'-'*105}")

        pareto.sort(key=lambda x: x[4].get('cv', 1))
        best_label = None
        best_cv = float('inf')
        for label, M1b, M2b, dv, m in pareto:
            cusps = m.get('cusps', 0)
            uni = m.get('uni', '?')
            c1 = m.get('c1', 0); c2 = m.get('c2', 0)
            drift = m.get('drift', 0)
            r2020_rt = m.get('r2020_rt', 0); lw = m.get('Lw', 0)
            print(f"  {label:<14s}  {m['cv']*100:5.1f}%  {cusps:>5}  {m['hue']:5.1f}  {m['yC']:6.4f}  "
                  f"{m['cliff']:6.3f}  {str(uni):>3s}/{N_UNI_HUES}  {drift:5.1f}  "
                  f"{r2020_rt:9.2e}  {lw:7.1f}  {c1:.1f}/{c2:.1f}")
            if cusps == 360 and (isinstance(uni, int) and uni == 0) and r2020_rt < 1e-3:
                if m['cv'] < best_cv:
                    best_cv = m['cv']
                    best_label = label

        if best_label:
            for label, M1b, M2b, dv, m in pareto:
                if label == best_label:
                    lms_w = M1b @ D65
                    lms_c_w = np.sign(lms_w) * np.abs(lms_w) ** (1/3)
                    L_w = M2b[0] @ lms_c_w
                    M2_norm = M2b / L_w
                    out = {"M1": M1b.tolist(), "gamma": [1/3,1/3,1/3], "M2": M2_norm.tolist()}
                    if use_delta and dv is not None:
                        out["delta"] = float(dv)
                    with open(args.output, "w") as f:
                        json.dump(out, f, indent=2)
                    print(f"\nBest -> {args.output} ({best_label}, CV={best_cv*100:.1f}%)")
                    break

        print(f"\n{'='*70}\nFINAL COMPARISON\n{'='*70}")
        full_eval("OKLab", OKLAB_M1, OKLAB_M2, pairs_t)


if __name__ == "__main__":
    main()
