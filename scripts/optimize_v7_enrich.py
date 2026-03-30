#!/usr/bin/env python
"""v7 GenSpace — ENRICHMENT on top of v6 base.

Takes a v6 checkpoint (M1/M2) and adds:
  - delta: piecewise-linear transfer near zero (1 DOF)
  - L_corr: [c1, c2, c3] cubic L correction (3 DOF)
  - M2 L-row fine-tuning (3 DOF)
Total: 7 DOF (Phase 1) or 20 DOF (Phase 2 with all params)

Strategy:
  Phase 1: Grid delta × CMA-ES(L_corr + M2_L) — 7 DOF, fast
  Phase 2: Joint CMA-ES on all 20 params — refine from Phase 1

Usage:
    python scripts/optimize_v7_enrich.py
    python scripts/optimize_v7_enrich.py --base checkpoints/v6_F_cvPlus.json
    python scripts/optimize_v7_enrich.py --phase 2 --base checkpoints/v7_phase1_best.json
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

# Default v6_B_lightUni (will be overridden by --base)
DEFAULT_M1 = np.array([
    [6.213663274448127, -0.5041794153770129, -0.40416891025666857],
    [-1.1592256796157883, 4.350194381717271, 0.5254938968299478],
    [0.0008170122534259527, 0.7226718820884986, 2.227799849833172],
])
DEFAULT_M2 = np.array([
    [0.6707003764386248, 0.17275693129316558, -0.2824551307487954],
    [0.48396436585569175, -0.36626971564869565, -0.17250847293787694],
    [-0.04414429613347437, 0.39348704882576724, -0.36830343592339543],
])

_PRIMARY_SRGB = np.array([[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1]], dtype=np.float64)
_TARGET_HUE_RAD = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])

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
# GPU helpers
# ══════════════════════════════════════════════════════════════════════

def srgb_to_linear_t(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))

def linear_to_srgb_t(c):
    c = c.clamp(0, 1)
    return torch.where(c <= 0.0031308, 12.92 * c,
                       1.055 * c.clamp(min=1e-12).pow(1/2.4) - 0.055)

def signed_cbrt_t(x):
    return x.sign() * x.abs().pow(1/3)

def transfer_forward_t(lms, delta_t=None):
    """Apply transfer: signed cbrt with optional piecewise-linear near zero."""
    if delta_t is None:
        return signed_cbrt_t(lms)
    ax = lms.abs()
    cbrt = ax.pow(1/3)
    d13 = delta_t.pow(1/3)
    slope = d13 / (3.0 * delta_t)
    linear = slope * ax + (2.0/3.0) * d13
    result = torch.where(ax >= delta_t, cbrt, linear)
    return lms.sign() * result

def transfer_inverse_t(lms_c, delta_t=None):
    if delta_t is None:
        return lms_c.sign() * lms_c.abs().pow(3)
    ay = lms_c.abs()
    d13 = delta_t.pow(1/3)
    slope = d13 / (3.0 * delta_t)
    cubed = ay.pow(3)
    linear_inv = (ay - (2.0/3.0) * d13) / slope.clamp(min=1e-30)
    linear_inv = linear_inv.clamp(min=0)
    result = torch.where(ay >= d13, cubed, linear_inv)
    return lms_c.sign() * result


def L_corr_forward(L, c1, c2, c3):
    """Apply cubic L correction: L' = L + c1*L*(1-L) + c2*L*(1-L)*(2L-1) + c3*L²*(1-L)².

    c1, c2, c3: (P, 1) or scalar
    L: (P, N) or (P, N, 1)
    """
    L1 = L * (1.0 - L)
    return L + c1 * L1 + c2 * L1 * (2.0 * L - 1.0) + c3 * L * L * (1.0 - L) * (1.0 - L)


def L_corr_inverse(L_prime, c1, c2, c3, n_iter=8):
    """Newton iteration to invert L correction.

    L_prime: target L value
    Returns L such that L_corr_forward(L, c1, c2, c3) ≈ L_prime
    """
    L = L_prime.clone()  # initial guess
    for _ in range(n_iter):
        L1 = L * (1.0 - L)
        f = L + c1 * L1 + c2 * L1 * (2.0 * L - 1.0) + c3 * L * L * (1.0 - L) * (1.0 - L) - L_prime
        # Derivative: df/dL
        df = (1.0 + c1 * (1.0 - 2.0 * L)
              + c2 * (6.0 * L * L - 6.0 * L + 1.0)
              + c3 * 2.0 * L * (1.0 - L) * (1.0 - 2.0 * L))
        L = L - f / df.clamp(min=1e-12)
        L = L.clamp(0.0, 1.0 + 0.01)  # small slack for out-of-gamut
    return L


def xyz_to_cielab_t(xyz):
    r = xyz / D65_T
    delta3 = (6.0 / 29.0) ** 3
    f = torch.where(r > delta3, r.clamp(min=1e-12).pow(1/3),
                    r / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)


def ciede2000_simplified(cl1, cl2):
    """Simplified CIEDE2000 matching test suite."""
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
# Pair generation — identical to v6
# ══════════════════════════════════════════════════════════════════════

def _hsv_to_rgb(h, s, v):
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

    for i in range(len(names)):
        for j in range(i+1, len(names)):
            add_srgb("primary", f"{names[i]}-{names[j]}", primaries[names[i]], primaries[names[j]])

    for n, rgb in primaries.items():
        add_srgb("to_white", f"{n}->W", rgb, [1,1,1])
        add_srgb("to_black", f"{n}->K", rgb, [0,0,0])

    for h_start in range(0, 360, 5):
        h_end = (h_start + 30) % 360
        r1, g1, b1 = _hsv_to_rgb(h_start / 360, 1.0, 1.0)
        r2, g2, b2 = _hsv_to_rgb(h_end / 360, 1.0, 1.0)
        add_srgb("hue_sweep", f"h{h_start}-h{h_end}", [r1,g1,b1], [r2,g2,b2])

    for h in [0, 60, 120, 180, 240, 300]:
        for s in [1.0, 0.7, 0.4]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, s, 1.0)
            add_srgb("saturation", f"h{h}s{s:.1f}->W", [r1,g1,b1], [1,1,1])

    for h in [0, 60, 120, 180, 240, 300]:
        for v_lo, v_hi in [(0.2, 0.8), (0.1, 0.5), (0.5, 1.0)]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, 0.8, v_lo)
            r2, g2, b2 = _hsv_to_rgb(h / 360, 0.8, v_hi)
            add_srgb("lightness", f"h{h}v{v_lo}-v{v_hi}", [r1,g1,b1], [r2,g2,b2])
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.8, 0.15)
        r2, g2, b2 = _hsv_to_rgb(h / 360, 0.8, 0.95)
        add_srgb("lightness", f"h{h}dark-light", [r1,g1,b1], [r2,g2,b2])

    grays = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(grays)):
        for j in range(i+2, len(grays)):
            add_srgb("gray", f"g{grays[i]:.1f}-g{grays[j]:.1f}", [grays[i]]*3, [grays[j]]*3)

    for n1, n2 in [("R", "C"), ("G", "M"), ("B", "Y")]:
        add_srgb("complementary", f"{n1}-{n2}", primaries[n1], primaries[n2])

    for h in range(0, 360, 15):
        for s in [0.05, 0.10, 0.15]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, s, 0.5)
            r2, g2, b2 = _hsv_to_rgb(((h + 30) % 360) / 360, s, 0.5)
            add_srgb("near_achromatic", f"h{h}s{s:.2f}", [r1,g1,b1], [r2,g2,b2])

    for h in range(0, 360, 30):
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.6, 0.15)
        r2, g2, b2 = _hsv_to_rgb(((h + 60) % 360) / 360, 0.6, 0.15)
        add_srgb("dark_dark", f"dark_h{h}", [r1,g1,b1], [r2,g2,b2])

    for h in range(0, 360, 20):
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.25, 0.95)
        r2, g2, b2 = _hsv_to_rgb(((h + 40) % 360) / 360, 0.25, 0.95)
        add_srgb("pastel", f"pastel_h{h}", [r1,g1,b1], [r2,g2,b2])

    for h_start in [350, 355, 358]:
        for h_end in [2, 5, 10]:
            r1, g1, b1 = _hsv_to_rgb(h_start / 360, 1.0, 1.0)
            r2, g2, b2 = _hsv_to_rgb(h_end / 360, 1.0, 1.0)
            add_srgb("hue_wrap", f"h{h_start}-h{h_end}", [r1,g1,b1], [r2,g2,b2])

    for h in [0, 60, 120, 240]:
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.5, 0.02)
        r2, g2, b2 = _hsv_to_rgb(h / 360, 0.5, 0.08)
        add_srgb("L_extreme_dark", f"vdark_h{h}", [r1,g1,b1], [r2,g2,b2])
        r1, g1, b1 = _hsv_to_rgb(h / 360, 0.3, 0.93)
        r2, g2, b2 = _hsv_to_rgb(h / 360, 0.3, 0.99)
        add_srgb("L_extreme_bright", f"vbright_h{h}", [r1,g1,b1], [r2,g2,b2])

    for h in range(0, 360, 15):
        for v in [0.3, 0.5, 0.7, 0.9]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, 0.95, v)
            r2, g2, b2 = _hsv_to_rgb(h / 360, 0.95, min(v + 0.2, 1.0))
            add_srgb("boundary_srgb", f"bnd_h{h}_v{v}", [r1,g1,b1], [r2,g2,b2])

    gen = torch.Generator(device=device).manual_seed(42)
    for k in range(1000):
        rgb1 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        rgb2 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        x1 = M_S2X_T @ srgb_to_linear_t(rgb1)
        x2 = M_S2X_T @ srgb_to_linear_t(rgb2)
        pairs.append(torch.stack([x1, x2]))
        labels.append(("random_srgb", f"rnd_s{k}"))

    for i in range(len(names)):
        for j in range(i+1, len(names)):
            rgb1_t = torch.tensor(primaries[names[i]], device=device, dtype=torch.float64)
            rgb2_t = torch.tensor(primaries[names[j]], device=device, dtype=torch.float64)
            x1 = M_P3_T @ srgb_to_linear_t(rgb1_t)
            x2 = M_P3_T @ srgb_to_linear_t(rgb2_t)
            pairs.append(torch.stack([x1, x2]))
            labels.append(("p3_primary", f"P3_{names[i]}-{names[j]}"))

    for n, rgb in primaries.items():
        rgb_t = torch.tensor(rgb, device=device, dtype=torch.float64)
        x1 = M_P3_T @ srgb_to_linear_t(rgb_t)
        x2 = M_S2X_T @ srgb_to_linear_t(torch.tensor([1.,1.,1.], device=device, dtype=torch.float64))
        pairs.append(torch.stack([x1, x2]))
        labels.append(("p3_to_srgb", f"P3_{n}->sRGB_W"))

    for n, rgb in primaries.items():
        rgb_t = torch.tensor(rgb, device=device, dtype=torch.float64)
        x1 = M_P3_T @ srgb_to_linear_t(rgb_t)
        x_w = M_P3_T @ srgb_to_linear_t(torch.ones(3, device=device, dtype=torch.float64))
        x_k = M_P3_T @ srgb_to_linear_t(torch.zeros(3, device=device, dtype=torch.float64))
        pairs.append(torch.stack([x1, x_w]))
        labels.append(("p3_to_white", f"P3_{n}->W"))
        pairs.append(torch.stack([x1, x_k]))
        labels.append(("p3_to_black", f"P3_{n}->K"))

    for h_start in range(0, 360, 15):
        h_end = (h_start + 30) % 360
        r1, g1, b1 = _hsv_to_rgb(h_start / 360, 1.0, 1.0)
        r2, g2, b2 = _hsv_to_rgb(h_end / 360, 1.0, 1.0)
        x1 = M_P3_T @ srgb_to_linear_t(torch.tensor([r1,g1,b1], device=device, dtype=torch.float64))
        x2 = M_P3_T @ srgb_to_linear_t(torch.tensor([r2,g2,b2], device=device, dtype=torch.float64))
        pairs.append(torch.stack([x1, x2]))
        labels.append(("p3_hue_sweep", f"P3_h{h_start}-h{h_end}"))

    for h in range(0, 360, 30):
        for s in [0.05, 0.10]:
            r1, g1, b1 = _hsv_to_rgb(h / 360, s, 0.5)
            r2, g2, b2 = _hsv_to_rgb(((h + 30) % 360) / 360, s, 0.5)
            x1 = M_P3_T @ srgb_to_linear_t(torch.tensor([r1,g1,b1], device=device, dtype=torch.float64))
            x2 = M_P3_T @ srgb_to_linear_t(torch.tensor([r2,g2,b2], device=device, dtype=torch.float64))
            pairs.append(torch.stack([x1, x2]))
            labels.append(("p3_near_achromatic", f"P3_na_h{h}_s{s}"))

    for k in range(500):
        rgb1 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        rgb2 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        x1 = M_P3_T @ srgb_to_linear_t(rgb1)
        x2 = M_P3_T @ srgb_to_linear_t(rgb2)
        pairs.append(torch.stack([x1, x2]))
        labels.append(("random_p3", f"rnd_p3_{k}"))

    for n, rgb in primaries.items():
        rgb_t = torch.tensor(rgb, device=device, dtype=torch.float64)
        x1 = M_R2020_T @ srgb_to_linear_t(rgb_t)
        x2 = M_P3_T @ srgb_to_linear_t(rgb_t)
        pairs.append(torch.stack([x1, x2]))
        labels.append(("rec2020_to_p3", f"R2020_{n}->P3_{n}"))

    for i in range(len(names)):
        for j in range(i+1, len(names)):
            rgb1_t = torch.tensor(primaries[names[i]], device=device, dtype=torch.float64)
            rgb2_t = torch.tensor(primaries[names[j]], device=device, dtype=torch.float64)
            x1 = M_R2020_T @ srgb_to_linear_t(rgb1_t)
            x2 = M_R2020_T @ srgb_to_linear_t(rgb2_t)
            pairs.append(torch.stack([x1, x2]))
            labels.append(("rec2020_primary", f"R2020_{names[i]}-{names[j]}"))

    for n, rgb in primaries.items():
        rgb_t = torch.tensor(rgb, device=device, dtype=torch.float64)
        x1 = M_R2020_T @ srgb_to_linear_t(rgb_t)
        x_w = M_R2020_T @ srgb_to_linear_t(torch.ones(3, device=device, dtype=torch.float64))
        x_k = M_R2020_T @ srgb_to_linear_t(torch.zeros(3, device=device, dtype=torch.float64))
        pairs.append(torch.stack([x1, x_w]))
        labels.append(("rec2020_to_white", f"R2020_{n}->W"))
        pairs.append(torch.stack([x1, x_k]))
        labels.append(("rec2020_to_black", f"R2020_{n}->K"))

    for h_start in range(0, 360, 15):
        h_end = (h_start + 30) % 360
        r1, g1, b1 = _hsv_to_rgb(h_start / 360, 1.0, 1.0)
        r2, g2, b2 = _hsv_to_rgb(h_end / 360, 1.0, 1.0)
        x1 = M_R2020_T @ srgb_to_linear_t(torch.tensor([r1,g1,b1], device=device, dtype=torch.float64))
        x2 = M_R2020_T @ srgb_to_linear_t(torch.tensor([r2,g2,b2], device=device, dtype=torch.float64))
        pairs.append(torch.stack([x1, x2]))
        labels.append(("rec2020_hue_sweep", f"R2020_h{h_start}-h{h_end}"))

    for k in range(500):
        rgb1 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        rgb2 = torch.rand(3, generator=gen, device=device, dtype=torch.float64)
        x1 = M_R2020_T @ srgb_to_linear_t(rgb1)
        x2 = M_R2020_T @ srgb_to_linear_t(rgb2)
        pairs.append(torch.stack([x1, x2]))
        labels.append(("random_rec2020", f"rnd_r2020_{k}"))

    result = torch.stack(pairs)
    print(f"  Generated {result.shape[0]} pairs ({len(set(c for c,_ in labels))} categories)")
    return result, labels


# ══════════════════════════════════════════════════════════════════════
# Pack / Unpack for v7 enrichment
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


def pack_phase1(M2_Lrow, delta, L_corr):
    """Phase 1: 7 DOF = M2_L[3] + delta[1] + L_corr[3]"""
    x = np.zeros(7)
    x[0:3] = M2_Lrow
    x[3] = np.log10(max(delta, 1e-10))  # log-scale delta
    x[4:7] = L_corr
    return x


def unpack_phase1_batch(X, M1_fixed, M2_ab_fixed, s_white):
    """X: (P, 7) -> M2_Lrow (P, 3), delta (P,), L_corr (P, 3)"""
    P = X.shape[0]
    M2_Lrow = X[:, 0:3]
    delta = np.clip(10.0 ** X[:, 3], 1e-10, 0.1)
    L_corr = X[:, 4:7]

    # Reconstruct full M2 with fixed ab-rows
    M2 = np.zeros((P, 3, 3))
    M2[:, 0, :] = M2_Lrow
    M2[:, 1, :] = M2_ab_fixed[0]  # broadcast
    M2[:, 2, :] = M2_ab_fixed[1]  # broadcast

    return M2, delta, L_corr


def pack_phase2(M1, M2, delta, L_corr):
    """Phase 2: 20 DOF = M1[9] + M2_L[3] + M2_ab[4] + delta[1] + L_corr[3]"""
    s = signed_cbrt_np(M1 @ D65)
    v1, v2 = ortho_basis(s)
    x = np.zeros(20)
    x[0:9] = M1.flatten()
    x[9:12] = M2[0]
    x[12] = M2[1] @ v1; x[13] = M2[1] @ v2
    x[14] = M2[2] @ v1; x[15] = M2[2] @ v2
    x[16] = np.log10(max(delta, 1e-10))
    x[17:20] = L_corr
    return x


def unpack_phase2_batch(X):
    """X: (P, 20) -> M1 (P,3,3), M2 (P,3,3), delta (P,), L_corr (P,3)"""
    P = X.shape[0]
    M1 = X[:, :9].reshape(P, 3, 3)
    M2 = np.zeros((P, 3, 3))
    M2[:, 0, :] = X[:, 9:12]
    for i in range(P):
        s = signed_cbrt_np(M1[i] @ D65)
        v1, v2 = ortho_basis(s)
        M2[i, 1] = X[i, 12] * v1 + X[i, 13] * v2
        M2[i, 2] = X[i, 14] * v1 + X[i, 15] * v2
    delta = np.clip(10.0 ** X[:, 16], 1e-10, 0.1)
    L_corr = X[:, 17:20]
    return M1, M2, delta, L_corr


# ══════════════════════════════════════════════════════════════════════
# Gamut checks
# ══════════════════════════════════════════════════════════════════════

STEPS = 26
_T_FRACS = torch.linspace(0, 1, STEPS, dtype=torch.float64, device=DEVICE)

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

N_UNI_HUES = 72
N_UNI_L = 50
N_UNI_C = 40
UNI_HUE_BATCH = 12


def compute_unimodal(M1_t, M2_t, L_white, delta_t=None, c1_t=None, c2_t=None, c3_t=None):
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

        # If L_corr, invert L first
        if c1_t is not None:
            Le_raw = L_corr_inverse(Le, c1_t[:1, :1, :1].expand(1, 1, 1),
                                       c2_t[:1, :1, :1].expand(1, 1, 1),
                                       c3_t[:1, :1, :1].expand(1, 1, 1))
            lab = torch.stack([Le_raw, Ce * ch, Ce * sh], dim=-1)
        else:
            lab = torch.stack([Le, Ce * ch, Ce * sh], dim=-1)

        lab_flat = lab.reshape(-1, 3)

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


# ══════════════════════════════════════════════════════════════════════
# Batch evaluation with L correction
# ══════════════════════════════════════════════════════════════════════

def batch_evaluate_v7(M1_t, M2_t, pairs_t, delta_t, c1_t, c2_t, c3_t):
    """Evaluate P candidates with enrichment.

    M1_t: (P, 3, 3), M2_t: (P, 3, 3)
    delta_t: (P, 1, 1) for transfer
    c1_t, c2_t, c3_t: (P, 1) for L correction
    """
    P = M1_t.shape[0]
    N = pairs_t.shape[0]
    S = STEPS

    # Forward: XYZ → M1 → transfer → M2 → L_corr → Lab
    xyz1 = pairs_t[:, 0].unsqueeze(0).expand(P, -1, -1)
    xyz2 = pairs_t[:, 1].unsqueeze(0).expand(P, -1, -1)

    lms1 = transfer_forward_t(torch.bmm(xyz1, M1_t.transpose(1, 2)), delta_t)
    lms2 = transfer_forward_t(torch.bmm(xyz2, M1_t.transpose(1, 2)), delta_t)
    lab1 = torch.bmm(lms1, M2_t.transpose(1, 2))
    lab2 = torch.bmm(lms2, M2_t.transpose(1, 2))

    # L_white normalization
    d65_batch = D65_T.unsqueeze(0).expand(P, -1).unsqueeze(1)
    lms_w = torch.bmm(d65_batch, M1_t.transpose(1, 2)).squeeze(1)
    lms_w_c = transfer_forward_t(lms_w.unsqueeze(1), delta_t).squeeze(1)
    L_white = (lms_w_c * M2_t[:, 0, :]).sum(dim=1)
    scale = L_white.clamp(min=1e-10).unsqueeze(1).unsqueeze(2)
    lab1 = lab1 / scale
    lab2 = lab2 / scale

    # Apply L correction to endpoints
    L1 = lab1[:, :, 0:1]
    L2 = lab2[:, :, 0:1]
    c1_e = c1_t.unsqueeze(1)  # (P, 1, 1)
    c2_e = c2_t.unsqueeze(1)
    c3_e = c3_t.unsqueeze(1)
    L1_c = L_corr_forward(L1, c1_e, c2_e, c3_e)
    L2_c = L_corr_forward(L2, c1_e, c2_e, c3_e)
    lab1_c = torch.cat([L1_c, lab1[:, :, 1:]], dim=-1)
    lab2_c = torch.cat([L2_c, lab2[:, :, 1:]], dim=-1)

    # Interpolate in enriched Lab
    t = _T_FRACS.view(1, 1, S, 1)
    lab_interp = lab1_c.unsqueeze(2) * (1 - t) + lab2_c.unsqueeze(2) * t

    # Inverse: Lab → L_corr_inv → M2_inv → transfer_inv → M1_inv → XYZ → sRGB → quantize → CIE Lab
    M2_norm = M2_t / scale.squeeze(2).unsqueeze(1)
    M2_norm_inv = torch.linalg.inv(M2_norm)
    M1_inv = torch.linalg.inv(M1_t)

    flat = lab_interp.reshape(P, N * S, 3)
    # Undo L correction
    L_flat = flat[:, :, 0:1]
    L_raw = L_corr_inverse(L_flat, c1_t.unsqueeze(1), c2_t.unsqueeze(1), c3_t.unsqueeze(1))
    flat_raw = torch.cat([L_raw, flat[:, :, 1:]], dim=-1)

    lms_c = torch.bmm(flat_raw, M2_norm_inv.transpose(1, 2))
    lms = transfer_inverse_t(lms_c, delta_t)
    xyz = torch.bmm(lms, M1_inv.transpose(1, 2))

    rgb_lin = torch.matmul(xyz, M_X2S_T.T).clamp(0, 1)
    rgb_srgb = linear_to_srgb_t(rgb_lin)
    rgb8 = (rgb_srgb * 255).round() / 255.0
    rgb_q = srgb_to_linear_t(rgb8)
    xyz_q = torch.matmul(rgb_q, M_S2X_T.T)
    cielab = xyz_to_cielab_t(xyz_q.clamp(min=1e-10))
    cielab = cielab.reshape(P, N, S, 3)

    des = ciede2000_simplified(cielab[:, :, :-1], cielab[:, :, 1:])
    mean_de = des.mean(dim=2)
    std_de = des.std(dim=2)
    ok = mean_de > 0.001
    cvs = torch.where(ok, std_de / mean_de, torch.zeros_like(mean_de))

    valid_mask = cvs > 0
    valid_counts = valid_mask.float().sum(dim=1).clamp(min=1)
    cv_mean = (cvs * valid_mask.float()).sum(dim=1) / valid_counts
    cv_sorted = cvs.sort(dim=1, descending=True).values
    n_top = max(1, N // 10)
    cv_top10 = cv_sorted[:, :n_top].mean(dim=1)

    # Drift
    h_steps = torch.atan2(cielab[:, :, :, 2], cielab[:, :, :, 1])
    C_steps = (cielab[:, :, :, 1]**2 + cielab[:, :, :, 2]**2).sqrt()
    h_start = h_steps[:, :, 0:1]
    h_end = h_steps[:, :, -1:]
    h_diff = torch.atan2((h_end - h_start).sin(), (h_end - h_start).cos())
    t_drift = _T_FRACS.view(1, 1, S)
    h_expected = h_start + t_drift * h_diff
    dev = torch.atan2((h_steps - h_expected).sin(), (h_steps - h_expected).cos()).abs()
    dev = dev * (C_steps > 5.0).float()
    drift_raw = dev.max(dim=2).values.mean(dim=1)

    # Cusp profiling (with L correction)
    B = _BXYZ.shape[0]
    bxyz = _BXYZ.unsqueeze(0).expand(P, -1, -1)
    blms = transfer_forward_t(torch.bmm(bxyz, M1_t.transpose(1, 2)), delta_t)
    blab = torch.bmm(blms, M2_t.transpose(1, 2))
    blab = blab / scale

    # Apply L_corr to boundary L
    bL = blab[:, :, 0:1]
    bL_c = L_corr_forward(bL, c1_t.unsqueeze(1), c2_t.unsqueeze(1), c3_t.unsqueeze(1))
    blab = torch.cat([bL_c, blab[:, :, 1:]], dim=-1)

    ba = blab[:, :, 1]; bb = blab[:, :, 2]
    bL_out = blab[:, :, 0]
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
        cusp_L[:, i] = bL_out.gather(1, max_idx.unsqueeze(1)).squeeze(1)

    deficit = (0.02 - cusp_C).clamp(min=0)
    missing_pen = (deficit ** 2).sum(dim=1)

    shifted = torch.roll(cusp_C, 1, dims=1)
    both_valid = (cusp_C > 0.005) & (shifted > 0.005)
    ratios = torch.where(both_valid, cusp_C / shifted.clamp(min=1e-10), torch.ones_like(cusp_C))
    ratios = torch.where(ratios > 1, 1 / ratios, ratios)
    cliff = 1.0 - ratios.min(dim=1).values
    cliff_raw = (cliff - 0.30).clamp(min=0) ** 2

    dC = cusp_C[:, 1:] - cusp_C[:, :-1]
    smooth_raw = (dC ** 2).sum(dim=1)

    unimodal_viol, unimodal_raw = compute_unimodal(M1_t, M2_t, L_white, delta_t)

    ybin = int((np.radians(85) + np.pi) / (2*np.pi) * N_HUE_BINS) % N_HUE_BINS
    yellow_C = cusp_C[:, ybin]
    yellow_raw = (0.15 - yellow_C).clamp(min=0) ** 2

    # Hue
    pxyz = _PRIMARY_XYZ_T.unsqueeze(0).expand(P, -1, -1)
    plms = transfer_forward_t(torch.bmm(pxyz, M1_t.transpose(1, 2)), delta_t)
    plab = torch.bmm(plms, M2_t.transpose(1, 2))
    plab = plab / scale
    # a,b stay same with L_corr, so hue unchanged
    ph = torch.atan2(plab[:, :, 2], plab[:, :, 1])
    diff = ph - _TARGET_HUE_T.unsqueeze(0)
    angular_err = torch.atan2(diff.sin(), diff.cos())
    hue_penalty = (angular_err ** 2).mean(dim=1)

    # Conditioning
    s1 = torch.linalg.svdvals(M1_t)
    s2 = torch.linalg.svdvals(M2_t)
    cond1 = s1[:, 0] / s1[:, 2].clamp(min=1e-10)
    cond2 = s2[:, 0] / s2[:, 2].clamp(min=1e-10)

    # Rec2020 RT
    M2i_t = torch.linalg.inv(M2_t)
    r2020_xyz = _R2020_BOUNDARY_T.unsqueeze(0).expand(P, -1, -1)
    r2020_lms = torch.bmm(r2020_xyz, M1_t.transpose(1, 2))
    r2020_lms_c = transfer_forward_t(r2020_lms, delta_t)
    r2020_lab = torch.bmm(r2020_lms_c, M2_t.transpose(1, 2))
    # L_corr forward + inverse should cancel
    r2020_lms_c_inv = torch.bmm(r2020_lab, M2i_t.transpose(1, 2))
    r2020_lms_inv = transfer_inverse_t(r2020_lms_c_inv, delta_t)
    r2020_xyz_inv = torch.bmm(r2020_lms_inv, M1_inv.transpose(1, 2))
    rt_err = (r2020_xyz - r2020_xyz_inv).abs().max(dim=2).values.max(dim=1).values

    # L_white scale
    lw_log = (L_white / 1.0).clamp(min=1e-10).log10()
    scale_raw = (lw_log.abs() - 0.5).clamp(min=0) ** 2

    # Dark stability
    dark_xyz = _DARK_XYZ.unsqueeze(0).expand(P, -1, -1)
    dark_lms = torch.bmm(dark_xyz, M1_t.transpose(1, 2))
    min_dark_abs = dark_lms.abs().min(dim=2).values.min(dim=1).values
    dark_lms_raw = (1e-5 - min_dark_abs).clamp(min=0) ** 2

    valid = (cond1 < 50) & (cond2 < 50) & (cv_mean < 5.0)

    return {
        'cv_mean': cv_mean, 'cv_top10': cv_top10,
        'hue_penalty': hue_penalty,
        'valid': valid, 'cusp_C': cusp_C,
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
# CMA-ES for Phase 1 and Phase 2
# ══════════════════════════════════════════════════════════════════════

def run_cma_phase1(x0, M1_fixed, M2_ab_fixed, s_white, pairs_t,
                   sigma=0.1, popsize=48, generations=200, label=""):
    """Phase 1: optimize M2_L[3] + delta[1] + L_corr[3] = 7 DOF."""

    M1_t_fixed = torch.tensor(M1_fixed, dtype=torch.float64, device=DEVICE).unsqueeze(0)

    es = cma.CMAEvolutionStrategy(x0, sigma, {
        'popsize': popsize,
        'maxiter': generations,
        'tolfun': 1e-11,
        'tolx': 1e-11,
        'verbose': -1,
    })

    best_loss = float('inf')
    best_x = None
    best_m = {}
    t0 = time.time()
    gen = 0

    while not es.stop():
        X = np.array(es.ask())
        P = X.shape[0]

        M2_all, delta_all, Lc_all = unpack_phase1_batch(X, M1_fixed, M2_ab_fixed, s_white)

        M1_t = M1_t_fixed.expand(P, -1, -1).contiguous()
        M2_t = torch.tensor(M2_all, dtype=torch.float64, device=DEVICE)
        delta_t = torch.tensor(delta_all, dtype=torch.float64, device=DEVICE).view(P, 1, 1)
        c1_t = torch.tensor(Lc_all[:, 0], dtype=torch.float64, device=DEVICE).view(P, 1)
        c2_t = torch.tensor(Lc_all[:, 1], dtype=torch.float64, device=DEVICE).view(P, 1)
        c3_t = torch.tensor(Lc_all[:, 2], dtype=torch.float64, device=DEVICE).view(P, 1)

        try:
            r = batch_evaluate_v7(M1_t, M2_t, pairs_t, delta_t, c1_t, c2_t, c3_t)
        except Exception as e:
            if gen == 0:
                print(f"  {label} gen {gen} ERROR: {e}", flush=True)
            es.tell(X, [999.0] * P)
            gen += 1
            continue

        # Loss: CV-focused with light gamut constraints
        cond_pen = 0.01 * ((r['cond1'] - 5.0).clamp(min=0)**2 + (r['cond2'] - 5.0).clamp(min=0)**2)
        gamut_pen = (500 * r['missing_pen'] + 100 * r['cliff_raw'] +
                     2 * r['smooth_raw'] + 50 * r['unimodal_raw'] + 200 * r['yellow_raw'])
        robust_pen = 1e6 * r['dark_lms_raw'] + 200 * r['scale_raw']

        losses = (r['cv_mean'] + 0.3 * r['cv_top10'] +
                  3.0 * gamut_pen + 0.5 * r['hue_penalty'] +
                  cond_pen + robust_pen)

        losses = torch.where(r['valid'], losses, torch.full_like(losses, 999.0))
        losses_np = losses.cpu().numpy()
        es.tell(X, losses_np.tolist())

        idx_best = np.argmin(losses_np)
        if losses_np[idx_best] < best_loss:
            best_loss = losses_np[idx_best]
            best_x = X[idx_best].copy()
            best_m = {
                'cv': r['cv_mean'][idx_best].item(),
                'top10': r['cv_top10'][idx_best].item(),
                'hue': np.degrees(np.sqrt(r['hue_penalty'][idx_best].item())),
                'cusps': int((r['cusp_C'][idx_best] > 0.02).sum().item()),
                'cliff': r['cliff'][idx_best].item(),
                'yC': r['yellow_C'][idx_best].item(),
                'uni': int(r['unimodal_viol'][idx_best].item()),
                'drift': r['drift_deg'][idx_best].item(),
                'Lw': r['L_white'][idx_best].item(),
                'delta': delta_all[idx_best],
                'L_corr': Lc_all[idx_best].tolist(),
            }

        gen += 1
        if gen % 10 == 0 or gen == 1:
            dt = time.time() - t0
            m = best_m
            d = m.get('delta', 0)
            lc = m.get('L_corr', [0,0,0])
            print(f"  {label} gen {gen:>4d}  loss={best_loss:.5f}  "
                  f"CV={m['cv']*100:.2f}%  cusps={m['cusps']}/360  "
                  f"hue={m['hue']:.1f}  yC={m['yC']:.3f}  "
                  f"cliff={m['cliff']:.2f}  uni={m['uni']}/72  "
                  f"delta={d:.2e}  Lc=[{lc[0]:.4f},{lc[1]:.4f},{lc[2]:.4f}]  "
                  f"({dt:.0f}s)", flush=True)

    return best_x, best_loss, best_m


def run_cma_phase2(x0, pairs_t, sigma=0.1, popsize=64, generations=300, label=""):
    """Phase 2: optimize all 20 DOF jointly."""

    es = cma.CMAEvolutionStrategy(x0, sigma, {
        'popsize': popsize,
        'maxiter': generations,
        'tolfun': 1e-11,
        'tolx': 1e-11,
        'verbose': -1,
    })

    best_loss = float('inf')
    best_x = None
    best_m = {}
    t0 = time.time()
    gen = 0

    while not es.stop():
        X = np.array(es.ask())
        P = X.shape[0]

        M1_all, M2_all, delta_all, Lc_all = unpack_phase2_batch(X)

        M1_t = torch.tensor(M1_all, dtype=torch.float64, device=DEVICE)
        M2_t = torch.tensor(M2_all, dtype=torch.float64, device=DEVICE)
        delta_t = torch.tensor(delta_all, dtype=torch.float64, device=DEVICE).view(P, 1, 1)
        c1_t = torch.tensor(Lc_all[:, 0], dtype=torch.float64, device=DEVICE).view(P, 1)
        c2_t = torch.tensor(Lc_all[:, 1], dtype=torch.float64, device=DEVICE).view(P, 1)
        c3_t = torch.tensor(Lc_all[:, 2], dtype=torch.float64, device=DEVICE).view(P, 1)

        try:
            r = batch_evaluate_v7(M1_t, M2_t, pairs_t, delta_t, c1_t, c2_t, c3_t)
        except Exception as e:
            if gen == 0:
                print(f"  {label} gen {gen} ERROR: {e}", flush=True)
            es.tell(X, [999.0] * P)
            gen += 1
            continue

        cond_pen = 0.01 * ((r['cond1'] - 5.0).clamp(min=0)**2 + (r['cond2'] - 5.0).clamp(min=0)**2)
        gamut_pen = (500 * r['missing_pen'] + 100 * r['cliff_raw'] +
                     2 * r['smooth_raw'] + 50 * r['unimodal_raw'] + 200 * r['yellow_raw'])
        robust_pen = 1e6 * r['dark_lms_raw'] + 200 * r['scale_raw']

        losses = (r['cv_mean'] + 0.3 * r['cv_top10'] +
                  3.0 * gamut_pen + 0.5 * r['hue_penalty'] +
                  cond_pen + robust_pen)

        losses = torch.where(r['valid'], losses, torch.full_like(losses, 999.0))
        losses_np = losses.cpu().numpy()
        es.tell(X, losses_np.tolist())

        idx_best = np.argmin(losses_np)
        if losses_np[idx_best] < best_loss:
            best_loss = losses_np[idx_best]
            best_x = X[idx_best].copy()
            best_m = {
                'cv': r['cv_mean'][idx_best].item(),
                'top10': r['cv_top10'][idx_best].item(),
                'hue': np.degrees(np.sqrt(r['hue_penalty'][idx_best].item())),
                'cusps': int((r['cusp_C'][idx_best] > 0.02).sum().item()),
                'cliff': r['cliff'][idx_best].item(),
                'yC': r['yellow_C'][idx_best].item(),
                'uni': int(r['unimodal_viol'][idx_best].item()),
                'drift': r['drift_deg'][idx_best].item(),
                'Lw': r['L_white'][idx_best].item(),
                'delta': delta_all[idx_best],
                'L_corr': Lc_all[idx_best].tolist(),
            }

        gen += 1
        if gen % 10 == 0 or gen == 1:
            dt = time.time() - t0
            m = best_m
            d = m.get('delta', 0)
            lc = m.get('L_corr', [0,0,0])
            print(f"  {label} gen {gen:>4d}  loss={best_loss:.5f}  "
                  f"CV={m['cv']*100:.2f}%  cusps={m['cusps']}/360  "
                  f"hue={m['hue']:.1f}  yC={m['yC']:.3f}  "
                  f"cliff={m['cliff']:.2f}  uni={m['uni']}/72  "
                  f"delta={d:.2e}  Lc=[{lc[0]:.4f},{lc[1]:.4f},{lc[2]:.4f}]  "
                  f"({dt:.0f}s)", flush=True)

    return best_x, best_loss, best_m


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=None, help="v6 checkpoint JSON to start from")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--popsize", type=int, default=48)
    parser.add_argument("--generations", type=int, default=200)
    args = parser.parse_args()

    # Load base checkpoint
    if args.base and Path(args.base).exists():
        d = json.load(open(args.base))
        M1_base = np.array(d["M1"])
        M2_base = np.array(d["M2"])
        delta_base = d.get("delta", 1e-7)
        L_corr_base = d.get("L_corr", [0.0, 0.0, 0.0])
        print(f"Base: {args.base}")
    else:
        M1_base = DEFAULT_M1
        M2_base = DEFAULT_M2
        delta_base = 1e-7
        L_corr_base = [0.0, 0.0, 0.0]
        print("Base: v6_B_lightUni (hardcoded)")

    print("=" * 70)
    print(f"v7 ENRICHMENT Phase {args.phase} — {7 if args.phase == 1 else 20} DOF")
    print("=" * 70)

    pairs_t, labels = generate_test_suite_pairs()
    print(f"  Total pairs: {pairs_t.shape[0]}")

    # OKLab baseline
    M1_ok_t = torch.tensor(OKLAB_M1, dtype=torch.float64, device=DEVICE).unsqueeze(0)
    M2_ok_t = torch.tensor(OKLAB_M2, dtype=torch.float64, device=DEVICE).unsqueeze(0)
    delta_zero = torch.tensor([[[1e-10]]], dtype=torch.float64, device=DEVICE)
    c_zero = torch.tensor([[0.0]], dtype=torch.float64, device=DEVICE)
    r_ok = batch_evaluate_v7(M1_ok_t, M2_ok_t, pairs_t, delta_zero, c_zero, c_zero, c_zero)
    print(f"\n  OKLab baseline: CV={r_ok['cv_mean'][0]*100:.2f}%")

    # v6 base baseline
    M1_b_t = torch.tensor(M1_base, dtype=torch.float64, device=DEVICE).unsqueeze(0)
    M2_b_t = torch.tensor(M2_base, dtype=torch.float64, device=DEVICE).unsqueeze(0)
    r_b = batch_evaluate_v7(M1_b_t, M2_b_t, pairs_t, delta_zero, c_zero, c_zero, c_zero)
    print(f"  v6 base (no enrichment): CV={r_b['cv_mean'][0]*100:.2f}%")

    if args.phase == 1:
        # Phase 1: Grid delta × CMA-ES(L_corr + M2_L)
        s_white = signed_cbrt_np(M1_base @ D65)
        M2_ab_fixed = M2_base[1:]  # rows 1 and 2

        deltas_to_try = [1e-7, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

        overall_best_loss = float('inf')
        overall_best_x = None
        overall_best_delta = None
        overall_best_m = {}

        for delta_init in deltas_to_try:
            x0 = pack_phase1(M2_base[0], delta_init, np.array(L_corr_base))

            print(f"\n--- Delta = {delta_init:.4f} ---")

            bx, bl, bm = run_cma_phase1(
                x0, M1_base, M2_ab_fixed, s_white, pairs_t,
                sigma=0.05, popsize=args.popsize, generations=args.generations,
                label=f"[d={delta_init:.4f}]"
            )

            if bl < overall_best_loss:
                overall_best_loss = bl
                overall_best_x = bx
                overall_best_delta = delta_init
                overall_best_m = bm

        # Save best Phase 1 result
        if overall_best_x is not None:
            M2_best, delta_best, Lc_best = unpack_phase1_batch(
                overall_best_x.reshape(1, -1), M1_base, M2_ab_fixed, s_white)

            # Normalize M2 so L_white = 1
            lms_w = M1_base @ D65
            if delta_best[0] > 1e-8:
                ax = np.abs(lms_w)
                d13 = delta_best[0] ** (1/3)
                slope = d13 / (3 * delta_best[0])
                lms_c_w = np.where(ax >= delta_best[0],
                                   np.sign(lms_w) * ax ** (1/3),
                                   np.sign(lms_w) * (slope * ax + (2/3) * d13))
            else:
                lms_c_w = np.sign(lms_w) * np.abs(lms_w) ** (1/3)
            L_w = M2_best[0, 0] @ lms_c_w
            M2_norm = M2_best[0] / L_w

            ckpt = {
                "M1": M1_base.tolist(),
                "gamma": [1/3, 1/3, 1/3],
                "M2": M2_norm.tolist(),
                "delta": float(delta_best[0]),
                "L_corr": Lc_best[0].tolist(),
            }
            ckpt_path = "checkpoints/v7_phase1_best.json"
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, indent=2)

            m = overall_best_m
            print(f"\n{'='*70}")
            print(f"PHASE 1 BEST (delta grid={overall_best_delta}):")
            print(f"  CV: {m['cv']*100:.2f}%  (OKLab: {r_ok['cv_mean'][0]*100:.2f}%)")
            print(f"  Delta: {m['delta']:.4e}")
            print(f"  L_corr: {m['L_corr']}")
            print(f"  Cusps: {m['cusps']}/360  cliff={m['cliff']:.3f}")
            print(f"  Hue RMS: {m['hue']:.1f}°")
            print(f"  Saved: {ckpt_path}")
            print(f"{'='*70}")

    elif args.phase == 2:
        # Phase 2: Joint 20 DOF
        x0 = pack_phase2(M1_base, M2_base, delta_base, np.array(L_corr_base))

        best_x = None
        best_loss = float('inf')
        best_m = {}

        for restart in range(3):
            sigma = 0.05 if restart == 0 else 0.1
            x_start = x0 if restart == 0 else x0 + np.random.randn(len(x0)) * 0.03

            bx, bl, bm = run_cma_phase2(
                x_start, pairs_t,
                sigma=sigma, popsize=args.popsize, generations=args.generations,
                label=f"[phase2 r{restart}]"
            )

            if bl < best_loss:
                best_loss = bl
                best_x = bx
                best_m = bm

        if best_x is not None:
            M1_b, M2_b, delta_b, Lc_b = unpack_phase2_batch(best_x.reshape(1, -1))

            lms_w = M1_b[0] @ D65
            if delta_b[0] > 1e-8:
                ax = np.abs(lms_w)
                d13 = delta_b[0] ** (1/3)
                slope = d13 / (3 * delta_b[0])
                lms_c_w = np.where(ax >= delta_b[0],
                                   np.sign(lms_w) * ax ** (1/3),
                                   np.sign(lms_w) * (slope * ax + (2/3) * d13))
            else:
                lms_c_w = np.sign(lms_w) * np.abs(lms_w) ** (1/3)
            L_w = M2_b[0, 0] @ lms_c_w
            M2_norm = M2_b[0] / L_w

            ckpt = {
                "M1": M1_b[0].tolist(),
                "gamma": [1/3, 1/3, 1/3],
                "M2": M2_norm.tolist(),
                "delta": float(delta_b[0]),
                "L_corr": Lc_b[0].tolist(),
            }
            ckpt_path = "checkpoints/v7_phase2_best.json"
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, indent=2)

            m = best_m
            print(f"\n{'='*70}")
            print(f"PHASE 2 BEST:")
            print(f"  CV: {m['cv']*100:.2f}%  (OKLab: {r_ok['cv_mean'][0]*100:.2f}%)")
            print(f"  Delta: {m['delta']:.4e}")
            print(f"  L_corr: {m['L_corr']}")
            print(f"  Cusps: {m['cusps']}/360  cliff={m['cliff']:.3f}")
            print(f"  Hue RMS: {m['hue']:.1f}°")
            print(f"  Saved: {ckpt_path}")
            print(f"{'='*70}")


if __name__ == "__main__":
    main()
