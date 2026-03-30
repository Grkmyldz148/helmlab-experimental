#!/usr/bin/env python
"""GenSpace parameter optimization — two-phase gradient CV minimization.

Phase 1: Core matrices (16 params) — DE global + L-BFGS-B refinement
Phase 2: Full pipeline (32 params) — L-BFGS-B with hue_lambda Pareto sweep

Target: beat Oklab on CIEDE2000-based gradient CV across diverse sRGB pairs.
Structure: XYZ → M1 → cbrt → M2 → [enrichment] → Lab
"""

import argparse
import json
import sys
import time
import colorsys
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

# Oklab reference matrices (XYZ input)
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

# Helmlab v20b (current gen_params.json M1/M2)
HELMLAB_M1 = np.array([
    [0.7342943222452644, 0.24049249952372878, -0.15763751765949208],
    [-0.3298489081190054, 1.2349200319947062, -0.00011401053364536622],
    [0.08633204542081865, 0.3414453053380001, 0.8320187908189167],
])
HELMLAB_M2 = np.array([
    [-0.4201444325131867, 0.4846067800936122, 0.641671081311568],
    [1.9328411649568151, -2.7178725296509234, 0.7639609115884012],
    [0.0056138814291964425, 1.6281620091666282, -1.2607966751966067],
])

# sRGB primary/secondary XYZ values and target hue angles
_PRIMARY_SRGB = np.array([
    [1, 0, 0],  # Red
    [1, 1, 0],  # Yellow
    [0, 1, 0],  # Green
    [0, 1, 1],  # Cyan
    [0, 0, 1],  # Blue
    [1, 0, 1],  # Magenta
], dtype=np.float64)

_TARGET_HUE_RAD = np.array([
    0,              # Red     = 0
    np.pi / 3,      # Yellow  = 60
    2 * np.pi / 3,  # Green   = 120
    np.pi,          # Cyan    = 180
    4 * np.pi / 3,  # Blue    = 240
    5 * np.pi / 3,  # Magenta = 300
])

_PRIMARY_NAMES = ["Red", "Yellow", "Green", "Cyan", "Blue", "Magenta"]

# Wide-gamut primaries for hue linearity constraint
M_REC2020_TO_XYZ = np.array([
    [0.6369580483012914, 0.14461690358620832, 0.1688809751641721],
    [0.2627002120112671, 0.6779980715188708, 0.05930171646986196],
    [0.0,                0.028072693049087428, 1.0609850577107909],
])
M_P3_TO_XYZ = np.array([
    [0.4865709486482162, 0.26566769316909306, 0.1982172852343625],
    [0.2289745640697488, 0.6917385218365064, 0.079286914093745],
    [0.0,                0.04511338185890264, 1.0439443689009757],
])

_WG_PRIMARIES = np.array([
    [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 1, 1], [0, 0, 1], [1, 0, 1],
], dtype=np.float64)
# rec2020/P3 primaries are already linear (display-referred), direct matrix multiply
_WG_REC2020_XYZ = np.array([M_REC2020_TO_XYZ @ c for c in _WG_PRIMARIES])
_WG_P3_XYZ = np.array([M_P3_TO_XYZ @ c for c in _WG_PRIMARIES])


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
    """Compute orthonormal basis perpendicular to s."""
    sn = s / (np.linalg.norm(s) + 1e-30)
    if abs(sn[0]) < 0.9:
        v1 = np.array([1, 0, 0]) - sn[0] * sn
    else:
        v1 = np.array([0, 1, 0]) - sn[1] * sn
    v1 /= (np.linalg.norm(v1) + 1e-30)
    v2 = np.cross(sn, v1)
    v2 /= (np.linalg.norm(v2) + 1e-30)
    return v1, v2


# Compute primary XYZ now that srgb_to_linear is defined
_PRIMARY_XYZ = np.array([M_SRGB_TO_XYZ @ srgb_to_linear(c) for c in _PRIMARY_SRGB])


# ══════════════════════════════════════════════════════════════════════
# CIE Lab + CIEDE2000 (scalar)
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
# Hue penalty
# ══════════════════════════════════════════════════════════════════════

def compute_hue_penalty_from_matrices(M1, M2, enrichment=None):
    """Mean squared angular error (radians^2) for 6 sRGB primaries.

    If enrichment is provided, apply full pipeline.
    """
    labs = []
    for xyz in _PRIMARY_XYZ:
        lab = M2 @ signed_cbrt(M1 @ xyz)
        L, a, b = lab[0], lab[1], lab[2]
        if enrichment is not None:
            L, a, b = apply_enrichment_scalar(L, a, b, enrichment)
        labs.append((L, a, b))

    errors_sq = []
    for (L, a, b), target_h in zip(labs, _TARGET_HUE_RAD):
        h = np.arctan2(b, a)
        diff = h - target_h
        angular_err = np.arctan2(np.sin(diff), np.cos(diff))
        errors_sq.append(angular_err ** 2)
    return float(np.mean(errors_sq))


def compute_hue_stats_from_matrices(M1, M2, enrichment=None):
    """Detailed hue stats for each primary/secondary."""
    labs = []
    for xyz in _PRIMARY_XYZ:
        lab = M2 @ signed_cbrt(M1 @ xyz)
        L, a, b = lab[0], lab[1], lab[2]
        if enrichment is not None:
            L, a, b = apply_enrichment_scalar(L, a, b, enrichment)
        labs.append((L, a, b))

    H_deg = []
    C_vals = []
    for L, a, b in labs:
        H_deg.append(np.degrees(np.arctan2(b, a)) % 360)
        C_vals.append(np.sqrt(a**2 + b**2))

    targets = [0, 60, 120, 180, 240, 300]
    errors = []
    for h, t in zip(H_deg, targets):
        diff = h - t
        diff = (diff + 180) % 360 - 180
        errors.append(abs(diff))

    errors = np.array(errors)
    return {
        "rms": float(np.sqrt(np.mean(errors**2))),
        "max": float(np.max(errors)),
        "per_color": {
            name: f"H={h:.1f} C={c:.3f} (err {e:.1f})"
            for name, h, c, e in zip(_PRIMARY_NAMES, H_deg, C_vals, errors)
        },
    }


def compute_wide_gamut_hue_penalty(M1, M2):
    """Hue drift when reducing chroma from wide-gamut colors.

    For each rec2020+P3 primary, converts to our space, then samples
    points along the same hue line at reduced chroma levels, inverts
    back to XYZ, and measures CIE Lab hue drift. Penalizes spaces
    where constant-hue chroma reduction changes perceptual hue.
    """
    try:
        M1_inv = np.linalg.inv(M1)
        M2_inv = np.linalg.inv(M2)
    except np.linalg.LinAlgError:
        return 10.0

    fracs = np.array([0.2, 0.4, 0.6, 0.8])
    total_drift = 0.0
    n_tests = 0

    for xyz_wg in np.concatenate([_WG_REC2020_XYZ, _WG_P3_XYZ]):
        # Forward: XYZ → our space
        lms_c = signed_cbrt(M1 @ xyz_wg)
        lab = M2 @ lms_c
        L, a, b = lab[0], lab[1], lab[2]

        h = np.arctan2(b, a)
        C = np.sqrt(a * a + b * b)
        if C < 1e-6:
            continue

        # Reference hue in CIE Lab
        cielab_wg = xyz_to_cielab(np.maximum(xyz_wg, 0))
        h_ref = np.arctan2(cielab_wg[2], cielab_wg[1])

        # Sample chroma reductions along same hue in our space
        cos_h, sin_h = np.cos(h), np.sin(h)
        for frac in fracs:
            C_test = C * frac
            lab_test = np.array([L, C_test * cos_h, C_test * sin_h])

            # Inverse: our space → XYZ
            lms_c_test = M2_inv @ lab_test
            lms_test = np.sign(lms_c_test) * np.abs(lms_c_test) ** 3
            xyz_test = M1_inv @ lms_test

            # CIE Lab hue of inverted point
            cielab_test = xyz_to_cielab(np.maximum(xyz_test, 0))
            h_test = np.arctan2(cielab_test[2], cielab_test[1])

            # Angular drift
            dh = np.arctan2(np.sin(h_test - h_ref), np.cos(h_test - h_ref))
            total_drift += dh * dh
            n_tests += 1

    return total_drift / max(n_tests, 1)


# ══════════════════════════════════════════════════════════════════════
# Training pairs (from opt_grad_cma.py)
# ══════════════════════════════════════════════════════════════════════

def generate_training_pairs():
    pairs = []

    # 1. Key presets
    presets = [
        ('#ff6b00','#00d4ff'), ('#ff0000','#0000ff'), ('#000000','#ffffff'),
        ('#ff0000','#00ff00'), ('#0000ff','#ffff00'), ('#8000ff','#ff8000'),
        ('#ff0000','#ffffff'), ('#0000ff','#ffffff'), ('#00ff00','#ffffff'),
        ('#000000','#ff0000'), ('#000000','#0000ff'), ('#000000','#00ff00'),
        ('#00ff00','#ff00ff'), ('#ff00ff','#ffff00'), ('#00ffff','#ff0000'),
    ]
    pairs.extend(presets)

    # 2. Primary/secondary pairs
    primaries = ['#ff0000','#00ff00','#0000ff','#ffff00','#ff00ff','#00ffff']
    for i in range(len(primaries)):
        for j in range(i+1, len(primaries)):
            p = (primaries[i], primaries[j])
            if p not in pairs and (p[1],p[0]) not in pairs:
                pairs.append(p)

    # 3. Achromatic pairs
    grays = ['#000000','#333333','#666666','#999999','#cccccc','#ffffff']
    for i in range(len(grays)):
        for j in range(i+1, len(grays)):
            if abs(i-j) >= 2:
                pairs.append((grays[i], grays[j]))

    # 4. Hue sweep
    for s, v in [(1.0, 1.0), (0.7, 0.8), (0.5, 0.6)]:
        for h_start in range(0, 360, 30):
            h_end = (h_start + 60) % 360
            r1,g1,b1 = colorsys.hsv_to_rgb(h_start/360, s, v)
            r2,g2,b2 = colorsys.hsv_to_rgb(h_end/360, s, v)
            h1 = f"#{int(r1*255):02x}{int(g1*255):02x}{int(b1*255):02x}"
            h2 = f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"
            pairs.append((h1, h2))

    # 5. Complementary hues
    for s, v in [(1.0, 1.0), (0.6, 0.8)]:
        for h in range(0, 180, 20):
            r1,g1,b1 = colorsys.hsv_to_rgb(h/360, s, v)
            r2,g2,b2 = colorsys.hsv_to_rgb((h+180)/360, s, v)
            h1 = f"#{int(r1*255):02x}{int(g1*255):02x}{int(b1*255):02x}"
            h2 = f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"
            pairs.append((h1, h2))

    # 6. Lightness sweeps
    for h in [0, 30, 60, 120, 180, 240, 300]:
        r1,g1,b1 = colorsys.hsv_to_rgb(h/360, 0.8, 0.95)
        r2,g2,b2 = colorsys.hsv_to_rgb(h/360, 0.8, 0.3)
        h1 = f"#{int(r1*255):02x}{int(g1*255):02x}{int(b1*255):02x}"
        h2 = f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"
        pairs.append((h1, h2))

    # 7. Random pairs
    np.random.seed(42)
    for _ in range(50):
        c1 = np.random.randint(0, 256, 3)
        c2 = np.random.randint(0, 256, 3)
        pairs.append((f"#{c1[0]:02x}{c1[1]:02x}{c1[2]:02x}",
                       f"#{c2[0]:02x}{c2[1]:02x}{c2[2]:02x}"))

    # Deduplicate
    seen = set()
    unique = []
    for a, b in pairs:
        key = tuple(sorted([a.lower(), b.lower()]))
        if key not in seen:
            seen.add(key)
            unique.append((a, b))

    return unique


# ══════════════════════════════════════════════════════════════════════
# Pack / Unpack (core 16 params + enrichment 16 params = 32 total)
# ══════════════════════════════════════════════════════════════════════

def matrices_to_core_params(M1, M2):
    """M1/M2 → 16 params (achromatic-constrained)."""
    x = np.zeros(16)
    x[:9] = M1.flatten()
    x[9:12] = M2[0]
    s = signed_cbrt(M1 @ D65)
    v1, v2 = ortho_basis(s)
    x[12] = M2[1] @ v1; x[13] = M2[1] @ v2
    x[14] = M2[2] @ v1; x[15] = M2[2] @ v2
    return x


def core_params_to_matrices(x):
    """16 params → M1/M2 (achromatic-constrained)."""
    M1 = x[:9].reshape(3, 3)
    s = signed_cbrt(M1 @ D65)
    v1, v2 = ortho_basis(s)
    M2 = np.zeros((3, 3))
    M2[0] = x[9:12]
    M2[1] = x[12]*v1 + x[13]*v2
    M2[2] = x[14]*v1 + x[15]*v2
    return M1, M2


def pack_full_params(M1, M2, enrichment):
    """Pack M1/M2 + enrichment dict → 32-param vector."""
    x = np.zeros(32)
    x[:16] = matrices_to_core_params(M1, M2)
    x[16] = enrichment.get("hue_cos1", 0.0)
    x[17] = enrichment.get("hue_sin1", 0.0)
    x[18] = enrichment.get("hue_cos2", 0.0)
    x[19] = enrichment.get("hue_sin2", 0.0)
    x[20] = enrichment.get("hue_cos3", 0.0)
    x[21] = enrichment.get("hue_sin3", 0.0)
    x[22] = enrichment.get("hue_cos4", 0.0)
    x[23] = enrichment.get("hue_sin4", 0.0)
    x[24] = enrichment.get("L_corr_p1", 0.0)
    x[25] = enrichment.get("L_corr_p2", 0.0)
    x[26] = enrichment.get("L_corr_p3", 0.0)
    x[27] = enrichment.get("lp_dark", 0.0)
    x[28] = enrichment.get("lp_dark_hcos", 0.0)
    x[29] = enrichment.get("lp_dark_hsin", 0.0)
    x[30] = enrichment.get("lc1", 0.0)
    x[31] = enrichment.get("lc2", 0.0)
    return x


def unpack_full_params(x):
    """32-param vector → (M1, M2, enrichment dict)."""
    M1, M2 = core_params_to_matrices(x[:16])
    enrichment = {
        "hue_cos1": x[16], "hue_sin1": x[17],
        "hue_cos2": x[18], "hue_sin2": x[19],
        "hue_cos3": x[20], "hue_sin3": x[21],
        "hue_cos4": x[22], "hue_sin4": x[23],
        "L_corr_p1": x[24], "L_corr_p2": x[25], "L_corr_p3": x[26],
        "lp_dark": x[27], "lp_dark_hcos": x[28], "lp_dark_hsin": x[29],
        "lc1": x[30], "lc2": x[31],
    }
    return M1, M2, enrichment


def params_to_gen_dict(M1, M2, enrichment):
    """Build gen_params.json-compatible dict."""
    d = {
        "M1": M1.tolist(),
        "gamma": [1/3, 1/3, 1/3],
        "M2": M2.tolist(),
    }
    d.update(enrichment)
    return d


# ══════════════════════════════════════════════════════════════════════
# Bounds
# ══════════════════════════════════════════════════════════════════════

def make_core_bounds(center):
    """Bounds for 16 core params."""
    bounds = []
    cap = 5.0  # wide enough for CMA-ES matrices
    # M1[0:9] + M2_L[9:12]
    for i in range(12):
        half = max(abs(center[i]) * 1.0, 0.5)
        lo = max(center[i] - half, -cap)
        hi = min(center[i] + half, cap)
        # Ensure center is within bounds
        if center[i] < lo:
            lo = center[i] - 0.5
        if center[i] > hi:
            hi = center[i] + 0.5
        bounds.append((lo, hi))
    # M2_ab projection[12:16]
    for i in range(12, 16):
        bounds.append((-5.0, 5.0))
    return bounds


def make_full_bounds(center):
    """Bounds for 32 full params."""
    bounds = make_core_bounds(center[:16])
    # hue_cos/sin [16:24]
    for _ in range(8):
        bounds.append((-0.5, 0.5))
    # L_corr_p1, p3 [24, 26]
    bounds.append((-0.8, 0.8))
    # L_corr_p2 [25]
    bounds.append((-1.2, 1.2))
    # L_corr_p3 [26]
    bounds.append((-0.8, 0.8))
    # lp_dark [27]
    bounds.append((-0.5, 1.0))
    # lp_dark_hcos, hsin [28:30]
    bounds.append((-0.5, 0.5))
    bounds.append((-0.5, 0.5))
    # lc1, lc2 [30:32]
    bounds.append((-1.0, 1.0))
    bounds.append((-1.0, 1.0))
    return bounds


# ══════════════════════════════════════════════════════════════════════
# Enrichment inline functions (from gen.py)
# ══════════════════════════════════════════════════════════════════════

def hue_delta(h, enrichment):
    """Hue correction delta (4 harmonics)."""
    return (
        enrichment["hue_cos1"] * np.cos(h) + enrichment["hue_sin1"] * np.sin(h) +
        enrichment["hue_cos2"] * np.cos(2*h) + enrichment["hue_sin2"] * np.sin(2*h) +
        enrichment["hue_cos3"] * np.cos(3*h) + enrichment["hue_sin3"] * np.sin(3*h) +
        enrichment["hue_cos4"] * np.cos(4*h) + enrichment["hue_sin4"] * np.sin(4*h)
    )


def hue_delta_deriv(h, enrichment):
    return (
        -enrichment["hue_cos1"] * np.sin(h) + enrichment["hue_sin1"] * np.cos(h) +
        -2 * enrichment["hue_cos2"] * np.sin(2*h) + 2 * enrichment["hue_sin2"] * np.cos(2*h) +
        -3 * enrichment["hue_cos3"] * np.sin(3*h) + 3 * enrichment["hue_sin3"] * np.cos(3*h) +
        -4 * enrichment["hue_cos4"] * np.sin(4*h) + 4 * enrichment["hue_sin4"] * np.cos(4*h)
    )


def apply_hue_correction(a, b, enrichment):
    h = np.arctan2(b, a)
    delta = hue_delta(h, enrichment)
    h_new = h + delta
    C = np.sqrt(a**2 + b**2)
    return C * np.cos(h_new), C * np.sin(h_new)


def undo_hue_correction(a, b, enrichment):
    h_out = np.arctan2(b, a)
    C = np.sqrt(a**2 + b**2)
    h_raw = h_out
    for _ in range(8):
        f = h_raw + hue_delta(h_raw, enrichment) - h_out
        fp = 1.0 + hue_delta_deriv(h_raw, enrichment)
        fp = fp if abs(fp) > 1e-10 else 1.0
        h_raw = h_raw - f / fp
    return C * np.cos(h_raw), C * np.sin(h_raw)


def L_correct(L, enrichment):
    p1, p2, p3 = enrichment["L_corr_p1"], enrichment["L_corr_p2"], enrichment["L_corr_p3"]
    t = L * (1.0 - L)
    return L + p1 * t + p2 * t * (0.5 - L) + p3 * t * t


def L_correct_inv(L1, enrichment):
    p1, p2, p3 = enrichment["L_corr_p1"], enrichment["L_corr_p2"], enrichment["L_corr_p3"]
    L = L1
    for _ in range(15):
        t = L * (1.0 - L)
        dt = 1.0 - 2.0 * L
        f = L + p1 * t + p2 * t * (0.5 - L) + p3 * t * t - L1
        dfdL = 1.0 + p1 * dt + p2 * (dt * (0.5 - L) - t) + p3 * 2.0 * t * dt
        if abs(dfdL) < 1e-10:
            dfdL = 1.0
        L = L - f / dfdL
    return L


def dark_L_compress(L, h, enrichment):
    coeff = enrichment["lp_dark"]
    coeff += enrichment["lp_dark_hcos"] * np.cos(h) + enrichment["lp_dark_hsin"] * np.sin(h)
    g = coeff * L * (1.0 - L) ** 2
    return L * np.exp(np.clip(g, -30.0, 30.0))


def dark_L_compress_inv(L_new, h, enrichment):
    coeff = enrichment["lp_dark"]
    coeff += enrichment["lp_dark_hcos"] * np.cos(h) + enrichment["lp_dark_hsin"] * np.sin(h)
    L = L_new
    for _ in range(12):
        oml = 1.0 - L
        g = coeff * L * oml ** 2
        eg = np.exp(np.clip(g, -30.0, 30.0))
        f = L * eg - L_new
        gp = coeff * oml * (1.0 - 3.0 * L)
        fp = eg * (1.0 + L * gp)
        if abs(fp) < 1e-10:
            fp = 1.0
        L = L - f / fp
    return L


def L_chroma_scale(L, enrichment):
    dL = L - 0.5
    arg = enrichment["lc1"] * dL + enrichment["lc2"] * dL ** 2
    return np.exp(np.clip(arg, -30.0, 30.0))


def has_enrichment(enrichment):
    """Check if any enrichment parameter is nonzero."""
    return any(abs(v) > 1e-15 for v in enrichment.values())


def has_hue_corr(enrichment):
    return any(abs(enrichment[k]) > 1e-15 for k in
               ["hue_cos1","hue_sin1","hue_cos2","hue_sin2",
                "hue_cos3","hue_sin3","hue_cos4","hue_sin4"])


def has_L_corr(enrichment):
    return any(abs(enrichment[k]) > 1e-15 for k in ["L_corr_p1","L_corr_p2","L_corr_p3"])


def has_dark_L(enrichment):
    return any(abs(enrichment[k]) > 1e-15 for k in ["lp_dark","lp_dark_hcos","lp_dark_hsin"])


def has_L_chroma(enrichment):
    return abs(enrichment["lc1"]) > 1e-15 or abs(enrichment["lc2"]) > 1e-15


def apply_enrichment_scalar(L, a, b, enrichment):
    """Apply enrichment stages (scalar, for one sample)."""
    # Hue correction
    if has_hue_corr(enrichment):
        a, b = apply_hue_correction(a, b, enrichment)

    # Cubic L correction
    if has_L_corr(enrichment):
        L = L_correct(L, enrichment)

    # Dark L compression
    if has_dark_L(enrichment):
        h = np.arctan2(b, a)
        L = dark_L_compress(L, h, enrichment)

    # L-chroma scaling
    if has_L_chroma(enrichment):
        T = L_chroma_scale(L, enrichment)
        a = a * T
        b = b * T

    return L, a, b


def undo_enrichment_scalar(L, a, b, enrichment):
    """Undo enrichment stages (scalar, for one sample)."""
    # Undo L-chroma scaling
    if has_L_chroma(enrichment):
        T = L_chroma_scale(L, enrichment)
        a = a / T
        b = b / T

    # Undo dark L
    if has_dark_L(enrichment):
        h = np.arctan2(b, a)
        L = dark_L_compress_inv(L, h, enrichment)

    # Undo cubic L
    if has_L_corr(enrichment):
        L = L_correct_inv(L, enrichment)

    # Undo hue correction
    if has_hue_corr(enrichment):
        a, b = undo_hue_correction(a, b, enrichment)

    return L, a, b


# ══════════════════════════════════════════════════════════════════════
# Gradient CV computation
# ══════════════════════════════════════════════════════════════════════

STEPS = 25


def compute_cv_core(M1, M2, pairs_xyz):
    """Compute CVs for core-only pipeline (no enrichment)."""
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


def compute_cv_full(M1, M2, enrichment, pairs_xyz):
    """Compute CVs for full pipeline (with enrichment, no NC)."""
    try:
        M2i = np.linalg.inv(M2)
        M1i = np.linalg.inv(M1)
    except np.linalg.LinAlgError:
        return 999.0, []

    use_enrichment = has_enrichment(enrichment)

    cvs = []
    for xyz1, xyz2 in pairs_xyz:
        try:
            # Forward: xyz → Lab
            raw1 = M2 @ signed_cbrt(M1 @ xyz1)
            raw2 = M2 @ signed_cbrt(M1 @ xyz2)
            L1, a1, b1 = raw1[0], raw1[1], raw1[2]
            L2, a2, b2 = raw2[0], raw2[1], raw2[2]

            if use_enrichment:
                L1, a1, b1 = apply_enrichment_scalar(L1, a1, b1, enrichment)
                L2, a2, b2 = apply_enrichment_scalar(L2, a2, b2, enrichment)

            # Interpolate in enriched Lab
            des = []
            prev = None
            for i in range(STEPS):
                t = i / (STEPS - 1)
                L = L1 + (L2 - L1) * t
                a = a1 + (a2 - a1) * t
                b = b1 + (b2 - b1) * t

                # Inverse
                Li, ai, bi = L, a, b
                if use_enrichment:
                    Li, ai, bi = undo_enrichment_scalar(Li, ai, bi, enrichment)

                c = M2i @ np.array([Li, ai, bi])
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
# Objectives
# ══════════════════════════════════════════════════════════════════════

_eval_count = 0
_best_loss = float("inf")
_last_print = 0.0


def reset_counters():
    global _eval_count, _best_loss, _last_print
    _eval_count = 0
    _best_loss = float("inf")
    _last_print = 0.0


def make_phase1_objective(train_xyz, hue_lambda=0.0, wg_lambda=0.0):
    """Phase 1: core matrices only (16 params), optional hue + wide-gamut penalty."""
    def objective(x):
        global _eval_count, _best_loss, _last_print
        try:
            M1, M2 = core_params_to_matrices(x)
            if np.linalg.cond(M1) > 100 or np.linalg.cond(M2) > 100:
                return 999.0

            mean_cv, cvs = compute_cv_core(M1, M2, train_xyz)
            if not cvs or mean_cv > 50:
                return 999.0

            sorted_cvs = sorted(cvs, reverse=True)
            n_top = max(1, len(cvs) // 10)
            top10 = np.mean(sorted_cvs[:n_top])

            # Conditioning penalty
            cond_penalty = 0.0
            c1 = np.linalg.cond(M1)
            c2 = np.linalg.cond(M2)
            if c1 > 20:
                cond_penalty += 0.01 * (c1 - 20)
            if c2 > 20:
                cond_penalty += 0.01 * (c2 - 20)

            # Hue penalty (sRGB primaries)
            hue_pen = 0.0
            if hue_lambda > 0:
                hue_pen = compute_hue_penalty_from_matrices(M1, M2)

            # Wide-gamut hue linearity penalty (rec2020 + P3)
            wg_pen = 0.0
            if wg_lambda > 0:
                wg_pen = compute_wide_gamut_hue_penalty(M1, M2)

            loss = mean_cv + 0.3 * top10 + cond_penalty + hue_lambda * hue_pen + wg_lambda * wg_pen

        except Exception:
            return 999.0

        _eval_count += 1
        if loss < _best_loss:
            _best_loss = loss

        now = time.time()
        if now - _last_print > 15.0:
            _last_print = now
            extras = ""
            if hue_lambda > 0:
                extras += f"  hue={hue_pen:.4f}"
            if wg_lambda > 0:
                extras += f"  wg={wg_pen:.4f}"
            print(f"    eval #{_eval_count:>6d}  loss={loss:.5f}  "
                  f"mean_cv={mean_cv*100:.2f}%  top10={top10*100:.2f}%  "
                  f"best={_best_loss:.5f}{extras}", flush=True)

        return loss

    return objective


def make_phase2_objective(train_xyz, hue_lambda=0.0):
    """Phase 2: full pipeline (32 params) with optional hue penalty."""
    def objective(x):
        global _eval_count, _best_loss, _last_print
        try:
            M1, M2, enrichment = unpack_full_params(x)
            if np.linalg.cond(M1) > 100 or np.linalg.cond(M2) > 100:
                return 999.0

            mean_cv, cvs = compute_cv_full(M1, M2, enrichment, train_xyz)
            if not cvs or mean_cv > 50:
                return 999.0

            sorted_cvs = sorted(cvs, reverse=True)
            n_top = max(1, len(cvs) // 10)
            top10 = np.mean(sorted_cvs[:n_top])

            # Conditioning penalty
            cond_penalty = 0.0
            c1 = np.linalg.cond(M1)
            c2 = np.linalg.cond(M2)
            if c1 > 20:
                cond_penalty += 0.01 * (c1 - 20)
            if c2 > 20:
                cond_penalty += 0.01 * (c2 - 20)

            # Hue penalty
            hue_pen = 0.0
            if hue_lambda > 0:
                hue_pen = compute_hue_penalty_from_matrices(M1, M2, enrichment)

            loss = mean_cv + 0.3 * top10 + cond_penalty + hue_lambda * hue_pen

        except Exception:
            return 999.0

        _eval_count += 1
        if loss < _best_loss:
            _best_loss = loss

        now = time.time()
        if now - _last_print > 15.0:
            _last_print = now
            hue_str = f"  hue_pen={hue_pen:.5f}" if hue_lambda > 0 else ""
            print(f"    eval #{_eval_count:>6d}  loss={loss:.5f}  "
                  f"mean_cv={mean_cv*100:.2f}%  top10={top10*100:.2f}%{hue_str}  "
                  f"best={_best_loss:.5f}", flush=True)

        return loss

    return objective


# ══════════════════════════════════════════════════════════════════════
# Seed preparation
# ══════════════════════════════════════════════════════════════════════

def prepare_seeds():
    """Prepare starting points for Phase 1."""
    seeds = []

    # 1. Oklab
    seeds.append(("Oklab", matrices_to_core_params(OKLAB_M1, OKLAB_M2)))

    # 2. Helmlab v20b
    seeds.append(("Helmlab-v20b", matrices_to_core_params(HELMLAB_M1, HELMLAB_M2)))

    # 3. CMA-ES best (if available)
    cma_path = Path("checkpoints/gradient_matrices_cma.json")
    if cma_path.exists():
        d = json.load(open(cma_path))
        M1_cma = np.array(d["M1_grad"])
        M2_cma = np.array(d["M2_grad"])
        seeds.append(("CMA-ES", matrices_to_core_params(M1_cma, M2_cma)))

    return seeds


# ══════════════════════════════════════════════════════════════════════
# Evaluation + reporting
# ══════════════════════════════════════════════════════════════════════

def evaluate_gen(M1, M2, train_xyz, val_xyz, show_xyz, enrichment=None):
    """Full evaluation of a GenSpace configuration."""
    if enrichment and has_enrichment(enrichment):
        train_cv, _ = compute_cv_full(M1, M2, enrichment, train_xyz)
        val_cv, _ = compute_cv_full(M1, M2, enrichment, val_xyz)
        show_cv, show_cvs = compute_cv_full(M1, M2, enrichment, show_xyz)
        hue = compute_hue_stats_from_matrices(M1, M2, enrichment)
    else:
        train_cv, _ = compute_cv_core(M1, M2, train_xyz)
        val_cv, _ = compute_cv_core(M1, M2, val_xyz)
        show_cv, show_cvs = compute_cv_core(M1, M2, show_xyz)
        hue = compute_hue_stats_from_matrices(M1, M2)

    # Achromatic check
    s = signed_cbrt(M1 @ D65)
    ach_a = float(M2[1] @ s)
    ach_b = float(M2[2] @ s)

    # Round-trip error
    rt_errors = []
    rng = np.random.default_rng(42)
    test_xyz = rng.uniform(0.05, 0.90, (200, 3))
    try:
        M2i = np.linalg.inv(M2)
        M1i = np.linalg.inv(M1)
        for xyz in test_xyz:
            lab_raw = M2 @ signed_cbrt(M1 @ xyz)
            L, a, b = lab_raw[0], lab_raw[1], lab_raw[2]
            if enrichment and has_enrichment(enrichment):
                L, a, b = apply_enrichment_scalar(L, a, b, enrichment)
                L, a, b = undo_enrichment_scalar(L, a, b, enrichment)
            c = M2i @ np.array([L, a, b])
            lms = np.sign(c) * np.abs(c)**3
            xyz_rt = M1i @ lms
            rt_errors.append(np.max(np.abs(xyz - xyz_rt)))
        rt_max = float(max(rt_errors))
    except Exception:
        rt_max = 999.0

    # Wide-gamut hue linearity
    wg_pen = compute_wide_gamut_hue_penalty(M1, M2)

    return {
        "train_cv": train_cv,
        "val_cv": val_cv,
        "show_cv": show_cv,
        "show_cvs": show_cvs,
        "ach_a": ach_a,
        "ach_b": ach_b,
        "rt_max": rt_max,
        "cond_M1": float(np.linalg.cond(M1)),
        "cond_M2": float(np.linalg.cond(M2)),
        "hue_rms": hue["rms"],
        "hue_max": hue["max"],
        "hue_per_color": hue["per_color"],
        "wg_pen": wg_pen,
    }


def print_eval(label, m, show_pairs=None, show_cvs=None):
    print(f"  {label}: train={m['train_cv']*100:.2f}%  val={m['val_cv']*100:.2f}%  "
          f"show={m['show_cv']*100:.2f}%")
    print(f"         ach_a={m['ach_a']:.2e}  ach_b={m['ach_b']:.2e}  "
          f"RT={m['rt_max']:.2e}  cond={m['cond_M1']:.1f}/{m['cond_M2']:.1f}")
    print(f"         hue_rms={m['hue_rms']:.1f}  hue_max={m['hue_max']:.1f}  "
          f"wg_pen={m['wg_pen']:.4f}")
    if show_pairs and show_cvs:
        for (a, b), cv in zip(show_pairs, show_cvs):
            print(f"           {a} -> {b}: {cv*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GenSpace parameter optimization (gradient CV)")
    parser.add_argument("--output", type=str, default="checkpoints/gen_space_best.json",
                        help="Output path for best params")
    parser.add_argument("--phase", type=str, default="both",
                        choices=["1", "1h", "2", "both"],
                        help="Which phase to run (1h = core with hue sweep)")
    parser.add_argument("--init", type=str, default=None,
                        help="Initialize Phase 2 from this JSON (skip Phase 1)")
    parser.add_argument("--de-maxiter", type=int, default=200,
                        help="DE max iterations (Phase 1)")
    parser.add_argument("--de-popsize", type=int, default=20,
                        help="DE population size (Phase 1)")
    parser.add_argument("--de-runs", type=int, default=5,
                        help="Number of DE runs per seed (Phase 1)")
    parser.add_argument("--maxiter", type=int, default=3000,
                        help="L-BFGS-B max iterations")
    parser.add_argument("--restarts", type=int, default=5,
                        help="Restarts per hue lambda (Phase 2)")
    parser.add_argument("--hue-lambdas", type=str, default="0,10,50,100,200",
                        help="Comma-separated hue lambda values (Phase 2)")
    parser.add_argument("--wg-lambda", type=float, default=0.0,
                        help="Wide-gamut hue linearity penalty weight (rec2020+P3)")
    args = parser.parse_args()

    hue_lambdas = [float(v) for v in args.hue_lambdas.split(",")]

    print("=" * 70)
    print("GenSpace Parameter Optimization")
    print("=" * 70)
    print(f"  Phase: {args.phase}")
    print(f"  Output: {args.output}")
    if args.wg_lambda > 0:
        print(f"  Wide-gamut lambda: {args.wg_lambda}")
    if args.phase in ("2", "both"):
        print(f"  Hue lambdas: {hue_lambdas}")
        print(f"  Restarts: {args.restarts}, maxiter: {args.maxiter}")

    # ── Generate training data ────────────────────────────────────────
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
    sys.stdout.flush()

    # ── Baselines ─────────────────────────────────────────────────────
    print("\n--- Baselines ---")
    ok_m = evaluate_gen(OKLAB_M1, OKLAB_M2, TRAIN_XYZ, VAL_XYZ, SHOW_XYZ)
    print_eval("Oklab", ok_m, SHOW_PAIRS, ok_m["show_cvs"])

    hl_m = evaluate_gen(HELMLAB_M1, HELMLAB_M2, TRAIN_XYZ, VAL_XYZ, SHOW_XYZ)
    print_eval("Helmlab-v20b", hl_m, SHOW_PAIRS, hl_m["show_cvs"])

    cma_path = Path("checkpoints/gradient_matrices_cma.json")
    cma_m = None
    if cma_path.exists():
        d = json.load(open(cma_path))
        M1_cma = np.array(d["M1_grad"]); M2_cma = np.array(d["M2_grad"])
        cma_m = evaluate_gen(M1_cma, M2_cma, TRAIN_XYZ, VAL_XYZ, SHOW_XYZ)
        print_eval("CMA-ES", cma_m, SHOW_PAIRS, cma_m["show_cvs"])

    sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════════
    # Phase 1: Core Matrices (16 params)
    # ══════════════════════════════════════════════════════════════════

    best_M1, best_M2 = OKLAB_M1.copy(), OKLAB_M2.copy()
    best_core_val = ok_m["val_cv"]

    if args.phase in ("1", "both"):
        print(f"\n{'='*70}")
        print(f"PHASE 1: Core Matrix Optimization (16 params)")
        print(f"  DE: popsize={args.de_popsize}, maxiter={args.de_maxiter}, "
              f"{args.de_runs} runs/seed")
        print(f"{'='*70}")

        seeds = prepare_seeds()
        objective = make_phase1_objective(TRAIN_XYZ, wg_lambda=args.wg_lambda)
        best_loss = float("inf")
        best_x = None

        for seed_name, x0 in seeds:
            bounds = make_core_bounds(x0)

            for run in range(args.de_runs):
                reset_counters()
                t0 = time.time()
                seed_val = hash(f"{seed_name}_{run}") % 2**31

                print(f"\n  --- DE from {seed_name}, run {run+1}/{args.de_runs} ---")
                sys.stdout.flush()

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
                print(f"    loss={final_loss:.5f} ({dt:.0f}s)")

                if final_loss < best_loss:
                    best_loss = final_loss
                    best_x = final_x.copy()
                    print(f"    ** New best!")

                sys.stdout.flush()

        # Final tight refinement
        print(f"\n  --- Final refinement (tight) ---")
        reset_counters()
        bounds = make_core_bounds(best_x)
        result = minimize(objective, best_x, method="L-BFGS-B",
                          bounds=bounds,
                          options={"maxiter": 3000, "ftol": 1e-14})
        if result.fun < best_loss:
            best_x = result.x.copy()
            best_loss = result.fun
        print(f"    Final loss: {best_loss:.5f}")

        best_M1, best_M2 = core_params_to_matrices(best_x)

        # Phase 1 results
        p1_m = evaluate_gen(best_M1, best_M2, TRAIN_XYZ, VAL_XYZ, SHOW_XYZ)
        print(f"\n  Phase 1 results:")
        print_eval("Phase1-best", p1_m, SHOW_PAIRS, p1_m["show_cvs"])
        best_core_val = p1_m["val_cv"]

        # Comparison table
        print(f"\n  {'Method':<16} {'Train':>8} {'Val':>8} {'Show':>8} {'HueRMS':>8} {'WG':>8}")
        print(f"  {'-'*60}")
        print(f"  {'Oklab':<16} {ok_m['train_cv']*100:>6.2f}%  {ok_m['val_cv']*100:>6.2f}%  "
              f"{ok_m['show_cv']*100:>6.2f}%  {ok_m['hue_rms']:>6.1f}  {ok_m['wg_pen']:.4f}")
        print(f"  {'Helmlab-v20b':<16} {hl_m['train_cv']*100:>6.2f}%  {hl_m['val_cv']*100:>6.2f}%  "
              f"{hl_m['show_cv']*100:>6.2f}%  {hl_m['hue_rms']:>6.1f}  {hl_m['wg_pen']:.4f}")
        if cma_m:
            print(f"  {'CMA-ES':<16} {cma_m['train_cv']*100:>6.2f}%  {cma_m['val_cv']*100:>6.2f}%  "
                  f"{cma_m['show_cv']*100:>6.2f}%  {cma_m['hue_rms']:>6.1f}  {cma_m['wg_pen']:.4f}")
        print(f"  {'Phase1-best':<16} {p1_m['train_cv']*100:>6.2f}%  {p1_m['val_cv']*100:>6.2f}%  "
              f"{p1_m['show_cv']*100:>6.2f}%  {p1_m['hue_rms']:>6.1f}  {p1_m['wg_pen']:.4f}")

        # Save Phase 1 checkpoint
        p1_path = args.output.replace(".json", "_phase1.json")
        p1_dict = params_to_gen_dict(best_M1, best_M2, {
            k: 0.0 for k in ["hue_cos1","hue_sin1","hue_cos2","hue_sin2",
                              "hue_cos3","hue_sin3","hue_cos4","hue_sin4",
                              "L_corr_p1","L_corr_p2","L_corr_p3",
                              "lp_dark","lp_dark_hcos","lp_dark_hsin",
                              "lc1","lc2"]
        })
        Path(p1_path).parent.mkdir(parents=True, exist_ok=True)
        with open(p1_path, "w") as f:
            json.dump(p1_dict, f, indent=2)
        print(f"\n  Saved Phase 1 to {p1_path}")
        sys.stdout.flush()

    elif args.init or args.phase == "1h":
        # Load initial params
        init_path = args.init or args.output.replace(".json", "_phase1.json")
        if not Path(init_path).exists():
            init_path = args.output
        print(f"\n  Loading initial params from {init_path}")
        d = json.load(open(init_path))
        best_M1 = np.array(d["M1"])
        best_M2 = np.array(d["M2"])
        init_m = evaluate_gen(best_M1, best_M2, TRAIN_XYZ, VAL_XYZ, SHOW_XYZ)
        print_eval("Init", init_m, SHOW_PAIRS, init_m["show_cvs"])
        best_core_val = init_m["val_cv"]

    # ══════════════════════════════════════════════════════════════════
    # Phase 1H: Core Matrices with Hue Lambda Sweep (16 params)
    # ══════════════════════════════════════════════════════════════════

    if args.phase == "1h":
        print(f"\n{'='*70}")
        print(f"PHASE 1H: Core Matrix + Hue Sweep (16 params, no enrichment)")
        print(f"  Hue lambdas: {hue_lambdas}")
        print(f"  Restarts: {args.restarts}, DE: popsize={args.de_popsize}, "
              f"maxiter={args.de_maxiter}")
        print(f"{'='*70}")

        x0_core = matrices_to_core_params(best_M1, best_M2)
        ph1h_results = []

        for hue_lambda in hue_lambdas:
            print(f"\n  {'='*60}")
            print(f"  hue_lambda = {hue_lambda}")
            print(f"  {'='*60}")

            objective = make_phase1_objective(TRAIN_XYZ, hue_lambda=hue_lambda,
                                               wg_lambda=args.wg_lambda)
            bounds = make_core_bounds(x0_core)

            best_x = x0_core.copy()
            best_loss = float("inf")

            for restart in range(args.restarts):
                reset_counters()
                t0 = time.time()

                # Perturb starting point for restarts > 0
                if restart > 0:
                    rng = np.random.default_rng(restart * 13 + int(hue_lambda * 7))
                    x_start = x0_core.copy()
                    x_start += rng.normal(0, 0.05, 16)
                    for i in range(16):
                        x_start[i] = np.clip(x_start[i], bounds[i][0], bounds[i][1])
                else:
                    x_start = x0_core.copy()

                # DE for global search
                result = differential_evolution(
                    objective, bounds, seed=restart * 31 + int(hue_lambda),
                    maxiter=args.de_maxiter, popsize=args.de_popsize,
                    tol=1e-8, mutation=(0.5, 1.5), recombination=0.9,
                    x0=x_start, disp=False,
                )

                # L-BFGS-B refinement
                result2 = minimize(objective, result.x, method="L-BFGS-B",
                                   bounds=bounds,
                                   options={"maxiter": args.maxiter,
                                            "ftol": 1e-13, "gtol": 1e-11})

                final_loss = min(result.fun, result2.fun)
                final_x = result2.x if result2.fun <= result.fun else result.x
                dt = time.time() - t0

                try:
                    M1_r, M2_r = core_params_to_matrices(final_x)
                    m_r = evaluate_gen(M1_r, M2_r, TRAIN_XYZ, VAL_XYZ, SHOW_XYZ)
                except Exception as e:
                    print(f"    Restart {restart+1}: FAILED ({e})")
                    continue

                print(f"    Restart {restart+1}/{args.restarts} ({dt:.0f}s): "
                      f"loss={final_loss:.5f}  val_cv={m_r['val_cv']*100:.2f}%  "
                      f"hue_rms={m_r['hue_rms']:.1f}°")

                if final_loss < best_loss:
                    best_loss = final_loss
                    best_x = final_x.copy()
                    print(f"      ** New best!")

                sys.stdout.flush()

            # Evaluate best for this lambda
            M1_best, M2_best = core_params_to_matrices(best_x)
            m_best = evaluate_gen(M1_best, M2_best, TRAIN_XYZ, VAL_XYZ, SHOW_XYZ)

            ph1h_results.append({
                "hue_lambda": hue_lambda,
                "x": best_x.copy(),
                "M1": M1_best,
                "M2": M2_best,
                "m": m_best,
            })

            print(f"\n    Best for hue_lambda={hue_lambda}:")
            print_eval(f"    lambda={hue_lambda}", m_best)

        # ── Phase 1H Summary ─────────────────────────────────────────
        print(f"\n{'='*90}")
        print(f"PHASE 1H PARETO SWEEP SUMMARY")
        print(f"{'='*90}")
        print(f"  {'lambda':>8}  {'Train':>8}  {'Val':>8}  {'Show':>8}  "
              f"{'HueRMS':>8}  {'HueMax':>8}  {'WG_pen':>8}")
        print(f"  {'-'*66}")

        for r in ph1h_results:
            m = r["m"]
            print(f"  {r['hue_lambda']:8.1f}  {m['train_cv']*100:>6.2f}%  "
                  f"{m['val_cv']*100:>6.2f}%  {m['show_cv']*100:>6.2f}%  "
                  f"{m['hue_rms']:>6.1f}°  {m['hue_max']:>6.1f}°  "
                  f"{m['wg_pen']:.4f}")

        # Oklab reference
        print(f"  {'Oklab':>8}  {ok_m['train_cv']*100:>6.2f}%  "
              f"{ok_m['val_cv']*100:>6.2f}%  {ok_m['show_cv']*100:>6.2f}%  "
              f"{ok_m['hue_rms']:>6.1f}°  {ok_m['hue_max']:>6.1f}°  "
              f"{ok_m['wg_pen']:.4f}")

        # Per-color details
        print(f"\n  Per-color hue details:")
        for r in ph1h_results:
            m = r["m"]
            print(f"  hue_lambda={r['hue_lambda']:.1f}:")
            for name, info in m["hue_per_color"].items():
                print(f"    {name:8s}: {info}")

        # Select best trade-off: Val CV < Oklab AND minimize hue_rms
        good = [r for r in ph1h_results if r["m"]["val_cv"] < ok_m["val_cv"]]

        if good:
            best = min(good, key=lambda r: r["m"]["hue_rms"])
            print(f"\n  Best trade-off (val<Oklab, min hue_rms):")
            print(f"    hue_lambda={best['hue_lambda']}, "
                  f"val_cv={best['m']['val_cv']*100:.2f}%, "
                  f"hue_rms={best['m']['hue_rms']:.1f}°")
        else:
            best = min(ph1h_results, key=lambda r: r["m"]["val_cv"])
            print(f"\n  No result beats Oklab val. Best val_cv:")
            print(f"    hue_lambda={best['hue_lambda']}, "
                  f"val_cv={best['m']['val_cv']*100:.2f}%, "
                  f"hue_rms={best['m']['hue_rms']:.1f}°")

        # Save best
        final_dict = params_to_gen_dict(best["M1"], best["M2"], {
            k: 0.0 for k in ["hue_cos1","hue_sin1","hue_cos2","hue_sin2",
                              "hue_cos3","hue_sin3","hue_cos4","hue_sin4",
                              "L_corr_p1","L_corr_p2","L_corr_p3",
                              "lp_dark","lp_dark_hcos","lp_dark_hsin",
                              "lc1","lc2"]
        })
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(final_dict, f, indent=2)
        print(f"\n  Saved best to {args.output}")

        # Save all lambda checkpoints
        for r in ph1h_results:
            ckpt_path = f"checkpoints/gen_1h_lam{r['hue_lambda']:.0f}.json"
            ckpt_dict = params_to_gen_dict(r["M1"], r["M2"], {
                k: 0.0 for k in ["hue_cos1","hue_sin1","hue_cos2","hue_sin2",
                                  "hue_cos3","hue_sin3","hue_cos4","hue_sin4",
                                  "L_corr_p1","L_corr_p2","L_corr_p3",
                                  "lp_dark","lp_dark_hcos","lp_dark_hsin",
                                  "lc1","lc2"]
            })
            with open(ckpt_path, "w") as f:
                json.dump(ckpt_dict, f, indent=2)

        print(f"  Saved {len(ph1h_results)} lambda checkpoints to checkpoints/")

        # Update best M1/M2 for final summary
        best_M1 = best["M1"]
        best_M2 = best["M2"]
        sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════════
    # Phase 2: Full Pipeline (32 params) with hue_lambda sweep
    # ══════════════════════════════════════════════════════════════════

    if args.phase in ("2", "both"):
        print(f"\n{'='*70}")
        print(f"PHASE 2: Full Pipeline Optimization (32 params)")
        print(f"  Hue lambda sweep: {hue_lambdas}")
        print(f"  Restarts: {args.restarts}, maxiter: {args.maxiter}")
        print(f"{'='*70}")

        # Zero enrichment starting point
        zero_enrichment = {
            "hue_cos1": 0.0, "hue_sin1": 0.0,
            "hue_cos2": 0.0, "hue_sin2": 0.0,
            "hue_cos3": 0.0, "hue_sin3": 0.0,
            "hue_cos4": 0.0, "hue_sin4": 0.0,
            "L_corr_p1": 0.0, "L_corr_p2": 0.0, "L_corr_p3": 0.0,
            "lp_dark": 0.0, "lp_dark_hcos": 0.0, "lp_dark_hsin": 0.0,
            "lc1": 0.0, "lc2": 0.0,
        }

        x0_full = pack_full_params(best_M1, best_M2, zero_enrichment)
        results = []

        for hue_lambda in hue_lambdas:
            print(f"\n  {'='*60}")
            print(f"  hue_lambda = {hue_lambda}")
            print(f"  {'='*60}")

            objective = make_phase2_objective(TRAIN_XYZ, hue_lambda=hue_lambda)
            bounds = make_full_bounds(x0_full)

            best_x = x0_full.copy()
            best_val_cv = 999.0

            for restart in range(args.restarts):
                reset_counters()
                t0 = time.time()

                # Perturb starting point slightly for restarts > 0
                if restart > 0:
                    rng = np.random.default_rng(restart * 7 + int(hue_lambda))
                    x_start = best_x.copy()
                    # Perturb enrichment params (16:32) slightly
                    x_start[16:] += rng.normal(0, 0.02, 16)
                    # Clip to bounds
                    for i in range(32):
                        x_start[i] = np.clip(x_start[i], bounds[i][0], bounds[i][1])
                else:
                    x_start = best_x.copy()

                result = minimize(objective, x_start, method="L-BFGS-B",
                                  bounds=bounds,
                                  options={"maxiter": args.maxiter,
                                           "ftol": 1e-13, "gtol": 1e-11})

                dt = time.time() - t0

                try:
                    M1_r, M2_r, enr_r = unpack_full_params(result.x)
                    m_r = evaluate_gen(M1_r, M2_r, TRAIN_XYZ, VAL_XYZ, SHOW_XYZ, enr_r)
                except Exception as e:
                    print(f"    Restart {restart+1}: FAILED ({e})")
                    continue

                print(f"    Restart {restart+1}/{args.restarts} ({dt:.0f}s): "
                      f"loss={result.fun:.5f}  val_cv={m_r['val_cv']*100:.2f}%  "
                      f"hue_rms={m_r['hue_rms']:.1f}  rt={m_r['rt_max']:.2e}")

                if m_r["rt_max"] > 1e-4:
                    print(f"      WARNING: high RT error, skipping")
                    continue

                if m_r["val_cv"] < best_val_cv:
                    best_val_cv = m_r["val_cv"]
                    best_x = result.x.copy()
                    print(f"      ** New best (val_cv={best_val_cv*100:.2f}%)")

                sys.stdout.flush()

            # Evaluate best for this lambda
            M1_best, M2_best, enr_best = unpack_full_params(best_x)
            m_best = evaluate_gen(M1_best, M2_best, TRAIN_XYZ, VAL_XYZ, SHOW_XYZ, enr_best)

            results.append({
                "hue_lambda": hue_lambda,
                "x": best_x.copy(),
                "M1": M1_best,
                "M2": M2_best,
                "enrichment": enr_best,
                "m": m_best,
            })

            print(f"\n    Best for hue_lambda={hue_lambda}:")
            print_eval(f"    lambda={hue_lambda}", m_best)

        # ── Phase 2 Summary ─────────────────────────────────────────
        print(f"\n{'='*90}")
        print(f"PHASE 2 PARETO SWEEP SUMMARY")
        print(f"{'='*90}")
        print(f"  {'lambda':>8}  {'Train':>8}  {'Val':>8}  {'Show':>8}  "
              f"{'HueRMS':>8}  {'HueMax':>8}  {'RT':>10}")
        print(f"  {'-'*72}")

        for r in results:
            m = r["m"]
            print(f"  {r['hue_lambda']:8.0f}  {m['train_cv']*100:>6.2f}%  "
                  f"{m['val_cv']*100:>6.2f}%  {m['show_cv']*100:>6.2f}%  "
                  f"{m['hue_rms']:>6.1f}  {m['hue_max']:>6.1f}  {m['rt_max']:>10.2e}")

        # Oklab reference line
        print(f"  {'Oklab':>8}  {ok_m['train_cv']*100:>6.2f}%  "
              f"{ok_m['val_cv']*100:>6.2f}%  {ok_m['show_cv']*100:>6.2f}%  "
              f"{ok_m['hue_rms']:>6.1f}  {ok_m['hue_max']:>6.1f}  {'—':>10}")

        # Per-color details
        print(f"\n  Per-color hue details:")
        for r in results:
            m = r["m"]
            print(f"  hue_lambda={r['hue_lambda']:.0f}:")
            for name, info in m["hue_per_color"].items():
                print(f"    {name:8s}: {info}")

        # Select best trade-off: Val CV < Oklab AND minimize hue_rms
        good = [r for r in results
                if r["m"]["val_cv"] < ok_m["val_cv"] and r["m"]["rt_max"] <= 1e-4]

        if good:
            best = min(good, key=lambda r: r["m"]["hue_rms"])
            print(f"\n  Best trade-off (val<Oklab, min hue_rms):")
            print(f"    hue_lambda={best['hue_lambda']}, "
                  f"val_cv={best['m']['val_cv']*100:.2f}%, "
                  f"hue_rms={best['m']['hue_rms']:.1f}")
        else:
            # Fallback: lowest val_cv
            valid = [r for r in results if r["m"]["rt_max"] <= 1e-4]
            if valid:
                best = min(valid, key=lambda r: r["m"]["val_cv"])
                print(f"\n  No result beats Oklab val. Best val_cv:")
                print(f"    hue_lambda={best['hue_lambda']}, "
                      f"val_cv={best['m']['val_cv']*100:.2f}%, "
                      f"hue_rms={best['m']['hue_rms']:.1f}")
            else:
                best = results[0]
                print(f"\n  WARNING: No valid results, using first")

        # Save best
        final_dict = params_to_gen_dict(best["M1"], best["M2"], best["enrichment"])
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(final_dict, f, indent=2)
        print(f"\n  Saved best to {args.output}")

        # Save all checkpoints
        for r in results:
            ckpt_path = f"checkpoints/gen_hue{r['hue_lambda']:.0f}.json"
            ckpt_dict = params_to_gen_dict(r["M1"], r["M2"], r["enrichment"])
            with open(ckpt_path, "w") as f:
                json.dump(ckpt_dict, f, indent=2)

        print(f"\n  Saved {len(results)} lambda checkpoints to checkpoints/")

    elif args.phase == "1":
        # Phase 1 only — already saved above
        pass

    # ══════════════════════════════════════════════════════════════════
    # Final Summary
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    # Load the saved output and evaluate
    if Path(args.output).exists():
        d = json.load(open(args.output))
        M1_f = np.array(d["M1"])
        M2_f = np.array(d["M2"])
        enr_f = {k: d.get(k, 0.0) for k in [
            "hue_cos1","hue_sin1","hue_cos2","hue_sin2",
            "hue_cos3","hue_sin3","hue_cos4","hue_sin4",
            "L_corr_p1","L_corr_p2","L_corr_p3",
            "lp_dark","lp_dark_hcos","lp_dark_hsin",
            "lc1","lc2"]}

        m_f = evaluate_gen(M1_f, M2_f, TRAIN_XYZ, VAL_XYZ, SHOW_XYZ, enr_f)

        print(f"\n  {'Method':<16} {'Train':>8} {'Val':>8} {'Show':>8} {'HueRMS':>8} {'RT':>10}")
        print(f"  {'-'*62}")
        print(f"  {'Oklab':<16} {ok_m['train_cv']*100:>6.2f}%  {ok_m['val_cv']*100:>6.2f}%  "
              f"{ok_m['show_cv']*100:>6.2f}%  {ok_m['hue_rms']:>6.1f}  {'—':>10}")
        if cma_m:
            print(f"  {'CMA-ES':<16} {cma_m['train_cv']*100:>6.2f}%  {cma_m['val_cv']*100:>6.2f}%  "
                  f"{cma_m['show_cv']*100:>6.2f}%  {cma_m['hue_rms']:>6.1f}  "
                  f"{cma_m['rt_max']:>10.2e}")
        print(f"  {'GenSpace-opt':<16} {m_f['train_cv']*100:>6.2f}%  {m_f['val_cv']*100:>6.2f}%  "
              f"{m_f['show_cv']*100:>6.2f}%  {m_f['hue_rms']:>6.1f}  "
              f"{m_f['rt_max']:>10.2e}")

        # Verdict
        if m_f["val_cv"] < ok_m["val_cv"]:
            improvement = (ok_m["val_cv"] - m_f["val_cv"]) / ok_m["val_cv"] * 100
            print(f"\n  BEATS Oklab by {improvement:.1f}% on validation set!")
        else:
            print(f"\n  Does NOT beat Oklab on validation set "
                  f"({m_f['val_cv']*100:.2f}% vs {ok_m['val_cv']*100:.2f}%)")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
