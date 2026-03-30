#!/usr/bin/env python3
"""Re-optimize H (Naka-Rushton + Enrichment) with Munsell + hue linearity penalties.

Starting from H_v2_params.json seed. Goal: fix Munsell Value (18% → <5%)
and hue linearity (13° → <10°) while keeping gradient CV advantage.

Usage:
    python scripts/optimize_H_munsell.py --gens 500 --pop 128
"""

import json, math, os, sys, time, argparse
from datetime import datetime
import numpy as np
import torch
import cma

torch.set_default_dtype(torch.float64)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Device: CUDA ({torch.cuda.get_device_name(0)})", flush=True)
else:
    print(f"Device: CPU", flush=True)

pa = argparse.ArgumentParser()
pa.add_argument("--gens", type=int, default=500)
pa.add_argument("--pop", type=int, default=128)
pa.add_argument("--seeds", type=int, default=8)
pa.add_argument("--sigma", type=float, default=0.03)
args = pa.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
CKPT = os.path.join(ROOT, "checkpoints")
os.makedirs(CKPT, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

D65 = torch.tensor([0.95047, 1.0, 1.08883], device=dev)
MS = torch.tensor([[.4124564,.3575761,.1804375],
                    [.2126729,.7151522,.0721750],
                    [.0193339,.1191920,.9503041]], device=dev)
MSi = torch.linalg.inv(MS)

def s2l(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))
def l2s(c):
    return torch.where(c <= 0.0031308, c * 12.92,
                       1.055 * c.clamp(min=1e-12).pow(1./2.4) - 0.055)

# ═══════════════════════════════════════════════════════════════
#  TRAINING PAIRS (2512 from space-test-project, or fallback)
# ═══════════════════════════════════════════════════════════════

_use_imported = False
for _try_dir in [
    os.path.join(ROOT, "space-test-project"),
    os.path.join(SCRIPT_DIR),
]:
    if os.path.isdir(os.path.join(_try_dir, "core")):
        sys.path.insert(0, _try_dir)
        try:
            from core.pairs import generate_all_pairs as _gen_pairs
            PT, _labels = _gen_pairs(dev)
            _use_imported = True
            print(f"Loaded {PT.shape[0]} pairs from space-test-project", flush=True)
            break
        except Exception as e:
            print(f"Pair import failed: {e}", flush=True)

if not _use_imported:
    # Fallback: basic pair set
    _pl = []
    _pr = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[0,0,0]]
    for i in range(len(_pr)):
        for j in range(i+1, len(_pr)):
            _pl.append((_pr[i], _pr[j]))
    _rng = np.random.RandomState(42)
    for _ in range(200):
        _pl.append((_rng.rand(3).tolist(), _rng.rand(3).tolist()))
    PT = torch.zeros(len(_pl), 2, 3, device=dev)
    for i, (c1, c2) in enumerate(_pl):
        PT[i, 0] = MS @ s2l(torch.tensor(c1, device=dev))
        PT[i, 1] = MS @ s2l(torch.tensor(c2, device=dev))
    print(f"Using fallback {PT.shape[0]} pairs", flush=True)

N_PAIRS = PT.shape[0]
N_ST = 25
T_ST = torch.linspace(0, 1, N_ST + 1, device=dev)

# ═══════════════════════════════════════════════════════════════
#  MUNSELL DATA (on GPU)
# ═══════════════════════════════════════════════════════════════

# Munsell Value → Y (ASTM D1535)
MUNSELL_VALUE_Y = {
    1: 0.01221, 2: 0.03126, 3: 0.06552, 4: 0.12000, 5: 0.19770,
    6: 0.30049, 7: 0.43060, 8: 0.59100, 9: 0.78660,
}
# Neutral grays at Munsell V=1..9
MUNSELL_GRAYS = torch.stack([D65 * MUNSELL_VALUE_Y[v] for v in range(1, 10)]).to(dev)  # (9, 3)

# Munsell 10 principal hue chips at V=5, C≈6 (sRGB)
MUNSELL_HUE_CHIPS_SRGB = torch.tensor([
    [176, 103, 101], [169, 117,  82], [155, 135,  80],  # 5R, 5YR, 5Y
    [115, 143,  87], [ 75, 148, 115], [ 58, 146, 140],  # 5GY, 5G, 5BG
    [ 69, 138, 159], [101, 118, 162], [132, 106, 149],  # 5B, 5PB, 5P
    [159,  99, 126],                                      # 5RP
], dtype=torch.float64, device=dev) / 255.0
MUNSELL_HUE_XYZ = (s2l(MUNSELL_HUE_CHIPS_SRGB) @ MS.T)  # (10, 3)

# Primary colors for hue linearity (sRGB primaries + secondaries)
PRIM_SRGB = torch.tensor([
    [1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,0,1],
], dtype=torch.float64, device=dev)
PRIM_XYZ = (s2l(PRIM_SRGB) @ MS.T)  # (6, 3)
WHITE_XYZ = D65.unsqueeze(0)  # (1, 3)

# Expected hue angles for OKLab-like space (roughly 60° apart)
HUE_EXP = torch.tensor([0, 60, 120, 180, 240, 300], dtype=torch.float64, device=dev)

# ═══════════════════════════════════════════════════════════════
#  H ARCHITECTURE: FORWARD / INVERSE
# ═══════════════════════════════════════════════════════════════

def fwd_H(xyz, d):
    """Forward: XYZ → Lab for batch of P candidates. xyz: (N, 3), d has P-batched params.
    Returns (P, N, 3)."""
    M1, M2 = d["M1"], d["M2"]  # (P, 3, 3)
    n, sigma, s_gain = d["n"], d["sigma"], d["s"]  # (P,)
    c1, k, cp = d["c1"], d["k"], d["cp"]  # (P,)

    lms = (xyz.unsqueeze(0) @ M1.transpose(-1, -2)).clamp(min=0)  # (P, N, 3)
    x_n = lms.pow(n.view(-1, 1, 1))
    sig_n = sigma.pow(n).view(-1, 1, 1)
    lms_c = s_gain.view(-1, 1, 1) * x_n / (x_n + sig_n)  # NR transfer

    raw = torch.bmm(lms_c, M2.transpose(-1, -2))  # (P, N, 3)
    L, a, b = raw[..., 0], raw[..., 1], raw[..., 2]

    # Enrichment
    L_out = L + c1.view(-1, 1) * L * (1.0 - L)
    C = torch.sqrt(a * a + b * b + 1e-30)
    f_L = torch.exp(k.view(-1, 1) * (L - 0.5))
    C_out = f_L * C.pow(cp.view(-1, 1))
    a_out = a / C * C_out
    b_out = b / C * C_out

    return torch.stack([L_out, a_out, b_out], dim=-1)


def inv_H(lab, d):
    """Inverse: Lab → XYZ. lab: (P, N, 3)."""
    M1i, M2i = d["M1i"], d["M2i"]
    n, sigma, s_gain = d["n"], d["sigma"], d["s"]
    c1, k, cp = d["c1"], d["k"], d["cp"]

    L_out, a_out, b_out = lab[..., 0], lab[..., 1], lab[..., 2]

    # Undo L enrichment (Newton)
    L = L_out.clone()
    for _ in range(10):
        g = L + c1.view(-1, 1) * L * (1.0 - L) - L_out
        gp = 1.0 + c1.view(-1, 1) * (1.0 - 2.0 * L)
        L = L - g / gp.clamp(min=1e-10)

    # Undo chroma enrichment
    C_out = torch.sqrt(a_out ** 2 + b_out ** 2 + 1e-30)
    f_L = torch.exp(k.view(-1, 1) * (L - 0.5))
    C_in = (C_out / f_L.clamp(min=1e-30)).clamp(min=0).pow(1.0 / cp.view(-1, 1))
    a_in = a_out / C_out * C_in
    b_in = b_out / C_out * C_in

    raw = torch.stack([L, a_in, b_in], dim=-1)
    lms_c = torch.bmm(raw, M2i.transpose(-1, -2))

    # Undo NR
    lms_c = torch.minimum(lms_c.clamp(min=0), s_gain.view(-1, 1, 1) - 1e-10)
    ratio = (lms_c / (s_gain.view(-1, 1, 1) - lms_c).clamp(min=1e-30)).clamp(min=0)
    lms = sigma.view(-1, 1, 1) * ratio.pow(1.0 / n.view(-1, 1, 1))

    return torch.bmm(lms, M1i.transpose(-1, -2))


# ═══════════════════════════════════════════════════════════════
#  PARAMETER PACKING / UNPACKING (19 params)
# ═══════════════════════════════════════════════════════════════

def _pack_m1(M1_np):
    """Pack 3x3 M1 into 6 free params (symmetric parameterization)."""
    x = np.zeros(6)
    x[0] = M1_np[0, 0]; x[1] = M1_np[0, 1]
    x[2] = M1_np[1, 0]; x[3] = M1_np[1, 1]
    x[4] = M1_np[2, 0]; x[5] = M1_np[2, 1]
    return x

def _pack_m2(M2_np):
    """Pack 3x3 M2 into 9 free params (all elements)."""
    x = np.zeros(9)
    x[0] = M2_np[0, 0]; x[1] = M2_np[0, 1]; x[2] = M2_np[0, 2]
    x[3] = M2_np[1, 0]; x[4] = M2_np[1, 1]; x[5] = M2_np[1, 2]
    x[6] = M2_np[2, 0]; x[7] = M2_np[2, 1]; x[8] = M2_np[2, 2]
    return x

def unpack_params(x):
    """x: (P, 21) → dict of batched params, valid mask."""
    P = x.shape[0]
    # M1: 6 params → 3x3 (third column = D65-constrained)
    M1 = torch.zeros(P, 3, 3, device=dev)
    M1[:, 0, 0] = x[:, 0]; M1[:, 0, 1] = x[:, 1]
    M1[:, 1, 0] = x[:, 2]; M1[:, 1, 1] = x[:, 3]
    M1[:, 2, 0] = x[:, 4]; M1[:, 2, 1] = x[:, 5]
    # Third column: M1 @ D65 = [1,1,1] (row sums constrained)
    M1[:, 0, 2] = (1.0 - M1[:, 0, 0] * D65[0] - M1[:, 0, 1] * D65[1]) / D65[2]
    M1[:, 1, 2] = (1.0 - M1[:, 1, 0] * D65[0] - M1[:, 1, 1] * D65[1]) / D65[2]
    M1[:, 2, 2] = (1.0 - M1[:, 2, 0] * D65[0] - M1[:, 2, 1] * D65[1]) / D65[2]

    lms_d65 = (D65.unsqueeze(0).unsqueeze(0) @ M1.transpose(-1, -2)).squeeze(1)  # (P, 3)
    valid = (lms_d65 > 0.01).all(dim=1)

    # M2: 9 params → 3x3 (all free)
    M2 = torch.zeros(P, 3, 3, device=dev)
    M2[:, 0, 0] = x[:, 6]; M2[:, 0, 1] = x[:, 7]; M2[:, 0, 2] = x[:, 8]
    M2[:, 1, 0] = x[:, 9]; M2[:, 1, 1] = x[:, 10]; M2[:, 1, 2] = x[:, 11]
    M2[:, 2, 0] = x[:, 12]; M2[:, 2, 1] = x[:, 13]; M2[:, 2, 2] = x[:, 14]

    # NR params
    n = 0.42 * torch.exp(x[:, 15].clamp(-1.0, 1.0))
    valid &= (n >= 0.15) & (n <= 1.1)
    sigma = torch.exp(x[:, 16].clamp(-3.0, 2.0))
    s_gain = torch.exp(x[:, 17].clamp(-2.0, 2.0))

    # M-cone dominant: soft penalty in evaluate(), not hard constraint here

    # Enrichment
    c1 = x[:, 18].clamp(-0.2, 0.2)
    k = x[:, 19].clamp(-0.5, 0.5)
    cp = 0.85 + 0.15 * torch.sigmoid(x[:, 20])

    # Inverses (per-candidate robust)
    M1i = torch.zeros_like(M1)
    M2i = torch.zeros_like(M2)
    good = valid.nonzero(as_tuple=True)[0]
    if good.numel() > 0:
        # Check determinants to filter singular matrices
        det1 = torch.linalg.det(M1[good])
        det2 = torch.linalg.det(M2[good])
        invertible = (det1.abs() > 1e-10) & (det2.abs() > 1e-10)
        really_good = good[invertible]
        if really_good.numel() > 0:
            M1i[really_good] = torch.linalg.inv(M1[really_good])
            M2i[really_good] = torch.linalg.inv(M2[really_good])
        valid[good[~invertible]] = False

    return {
        "M1": M1, "M2": M2, "M1i": M1i, "M2i": M2i,
        "n": n, "sigma": sigma, "s": s_gain,
        "c1": c1, "k": k, "cp": cp,
    }, valid


# ═══════════════════════════════════════════════════════════════
#  BATCH METRICS (all GPU, all batched over P candidates)
# ═══════════════════════════════════════════════════════════════

def batch_cv(d):
    """Gradient CV (CIEDE2000-based). Returns (P,)."""
    P = d["M1"].shape[0]
    lab1 = fwd_H(PT[:, 0], d)  # (P, N, 3)
    lab2 = fwd_H(PT[:, 1], d)

    t = T_ST.view(1, 1, -1, 1)
    labs = lab1.unsqueeze(2) + t * (lab2 - lab1).unsqueeze(2)  # (P, N, 26, 3)
    lf = labs.reshape(P, -1, 3)  # (P, N*26, 3)

    xyz = inv_H(lf, d)
    lin = (xyz @ MSi.T).clamp(0, 1)
    s8 = (l2s(lin) * 255).round() / 255.0
    xb = (s2l(s8) @ MS.T)

    # CIE Lab
    r = xb.clamp(min=1e-10) / D65.view(1, 1, 3)
    f = torch.where(r > 0.008856, r.pow(1. / 3.), 7.787 * r + 16. / 116.)
    cl = torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                      200 * (f[..., 1] - f[..., 2])], dim=-1)
    cl = cl.reshape(P, N_PAIRS, N_ST + 1, 3)

    c1, c2 = cl[:, :, :-1], cl[:, :, 1:]
    dL = c2[..., 0] - c1[..., 0]
    C1 = (c1[..., 1] ** 2 + c1[..., 2] ** 2).sqrt()
    C2 = (c2[..., 1] ** 2 + c2[..., 2] ** 2).sqrt()
    dC = C2 - C1
    dH = ((c2[..., 1] - c1[..., 1]) ** 2 + (c2[..., 2] - c1[..., 2]) ** 2 - dC ** 2).clamp(min=0).sqrt()
    SL = 1 + 0.015 * (c1[..., 0] - 50) ** 2 / (20 + (c1[..., 0] - 50) ** 2).sqrt()
    SC = 1 + 0.045 * C1
    SH = 1 + 0.015 * C1
    de = ((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2).sqrt()

    md = de.mean(2)
    sd = de.std(2)
    ok = md > 0.001
    cvs = torch.where(ok, sd / md, torch.zeros_like(md))
    cnt = ok.float().sum(1).clamp(min=1)
    return (cvs * ok.float()).sum(1) / cnt


def batch_munsell_value(d):
    """Munsell Value uniformity: CV of L spacing for V=1..9 grays. Returns (P,)."""
    lab = fwd_H(MUNSELL_GRAYS, d)  # (P, 9, 3)
    L = lab[:, :, 0]  # (P, 9)
    dL = L[:, 1:] - L[:, :-1]  # (P, 8)
    mean_dL = dL.abs().mean(dim=1).clamp(min=1e-10)
    std_dL = dL.std(dim=1)
    return std_dL / mean_dL * 100.0  # CV% (P,)


def batch_munsell_hue(d):
    """Munsell Hue spacing uniformity: CV of hue angle gaps for 10 principal hues. Returns (P,)."""
    lab = fwd_H(MUNSELL_HUE_XYZ, d)  # (P, 10, 3)
    h = torch.atan2(lab[:, :, 2], lab[:, :, 1])  # (P, 10) radians

    # Sort by hue for each candidate
    h_sorted, _ = torch.sort(h, dim=1)

    # Compute gaps
    dh = h_sorted[:, 1:] - h_sorted[:, :-1]  # (P, 9)
    # Last gap wraps around
    last_gap = (h_sorted[:, 0] + 2 * math.pi - h_sorted[:, -1]).unsqueeze(1)
    dh = torch.cat([dh, last_gap], dim=1)  # (P, 10)

    mean_dh = dh.mean(dim=1).clamp(min=1e-10)
    std_dh = dh.std(dim=1)
    return std_dh / mean_dh * 100.0  # CV% (P,)


def batch_hue_linearity(d):
    """Hue linearity: RMS hue deviation for primary→white gradients. Returns (P,)."""
    P = d["M1"].shape[0]
    # 6 primaries, 11 steps each
    n_steps = 11
    t = torch.linspace(0, 1, n_steps, device=dev).view(1, 1, -1, 1)  # (1, 1, 11, 1)

    lab_prim = fwd_H(PRIM_XYZ, d)  # (P, 6, 3)
    lab_white = fwd_H(WHITE_XYZ.expand(6, 3), d)  # (P, 6, 3)

    # Interpolate in lab
    lab_start = lab_prim.unsqueeze(2)  # (P, 6, 1, 3)
    lab_end = lab_white.unsqueeze(2)  # (P, 6, 1, 3)
    labs = lab_start + t * (lab_end - lab_start)  # (P, 6, 11, 3)

    # Hue angle at each step
    h = torch.atan2(labs[..., 2], labs[..., 1])  # (P, 6, 11) radians

    # Hue at start and end
    h_start = h[:, :, 0:1]  # (P, 6, 1)
    h_end = h[:, :, -1:]

    # Expected hue at each step (linear interpolation of hue)
    t_lin = torch.linspace(0, 1, n_steps, device=dev).view(1, 1, -1)
    # Handle wrap-around
    dh = h_end - h_start
    dh = torch.where(dh > math.pi, dh - 2 * math.pi, dh)
    dh = torch.where(dh < -math.pi, dh + 2 * math.pi, dh)
    h_expected = h_start + t_lin * dh

    # Deviation (skip endpoints and near-achromatic)
    C = torch.sqrt(labs[..., 1] ** 2 + labs[..., 2] ** 2)
    h_diff = h - h_expected
    h_diff = torch.where(h_diff > math.pi, h_diff - 2 * math.pi, h_diff)
    h_diff = torch.where(h_diff < -math.pi, h_diff + 2 * math.pi, h_diff)

    # Mask out low-chroma points (hue undefined)
    mask = C > 0.01
    h_diff_masked = h_diff * mask.float()
    count = mask.float().sum(dim=(1, 2)).clamp(min=1)
    rms = torch.sqrt((h_diff_masked ** 2).sum(dim=(1, 2)) / count) * (180.0 / math.pi)
    return rms  # degrees, (P,)


def batch_lightness_mono(d):
    """Lightness monotonicity: fraction of gray ramp steps with dL > 0. Returns (P,)."""
    n_gray = 64
    t = torch.linspace(0.001, 0.999, n_gray, device=dev)
    grays_xyz = D65.unsqueeze(0) * t.unsqueeze(1)  # (64, 3)
    lab = fwd_H(grays_xyz, d)  # (P, 64, 3)
    L = lab[:, :, 0]
    dL = L[:, 1:] - L[:, :-1]
    mono = (dL > 0).float().mean(dim=1)  # 1.0 = perfect
    return mono


def batch_info(d):
    """Basic space info: yellow chroma, blue L, B→W G/R, R→W G-B, primary L range."""
    P = d["M1"].shape[0]
    # Primaries in Lab
    lab_p = fwd_H(PRIM_XYZ, d)  # (P, 6, 3) R,G,B,Y,C,M
    lab_w = fwd_H(D65.unsqueeze(0), d)  # (P, 1, 3)
    lab_k = fwd_H(torch.zeros(1, 3, device=dev), d)  # (P, 1, 3)

    L_p = lab_p[:, :, 0]
    yC = torch.sqrt(lab_p[:, 3, 1] ** 2 + lab_p[:, 3, 2] ** 2)  # Yellow chroma
    bL = lab_p[:, 2, 0]  # Blue L

    # B→W midpoint in Lab
    lab_mid_bw = 0.5 * (lab_p[:, 2:3, :] + lab_w)
    xyz_mid = inv_H(lab_mid_bw, d)
    srgb_mid = l2s((xyz_mid @ MSi.T).clamp(0, 1))
    bw_gr = srgb_mid[:, 0, 1] / srgb_mid[:, 0, 0].clamp(min=1e-10)

    # R→W midpoint
    lab_mid_rw = 0.5 * (lab_p[:, 0:1, :] + lab_w)
    xyz_mid_r = inv_H(lab_mid_rw, d)
    srgb_mid_r = l2s((xyz_mid_r @ MSi.T).clamp(0, 1))
    rw_gb = srgb_mid_r[:, 0, 1] - srgb_mid_r[:, 0, 2]

    # Primary L range
    plr = L_p.max(dim=1).values - L_p.min(dim=1).values

    # White L
    wL = lab_w[:, 0, 0]

    return {
        "yC": yC, "bL": bL, "bw": bw_gr.squeeze(), "rw": rw_gb.squeeze(),
        "plr": plr, "wL": wL,
    }


def batch_cond(d):
    """Condition numbers for M1, M2. Returns (P,), (P,)."""
    c1 = torch.linalg.cond(d["M1"])
    c2 = torch.linalg.cond(d["M2"])
    return c1, c2


def batch_cusp(d, hue_degs=None):
    """Cusp scan. Returns cusp_L (P, H), cusp_C (P, H), cliff (P, H)."""
    if hue_degs is None:
        hue_degs = list(range(0, 360, 5))
    P = d["M1"].shape[0]
    H = len(hue_degs)
    _Ls = torch.linspace(0.02, 0.998, 100, device=dev)
    _Cs = torch.linspace(0.001, 0.4, 80, device=dev)

    cusp_L = torch.zeros(P, H, device=dev)
    cusp_C = torch.zeros(P, H, device=dev)

    for hi, hdeg in enumerate(hue_degs):
        h_rad = hdeg * math.pi / 180.0
        cos_h = math.cos(h_rad)
        sin_h = math.sin(h_rad)

        # Build Lab grid: (100*80, 3) for this hue
        L_grid = _Ls.unsqueeze(1).expand(100, 80).reshape(-1)  # (8000,)
        C_grid = _Cs.unsqueeze(0).expand(100, 80).reshape(-1)
        a_grid = C_grid * cos_h
        b_grid = C_grid * sin_h
        lab_grid = torch.stack([L_grid, a_grid, b_grid], dim=-1)  # (8000, 3)

        # Inverse to XYZ for all P candidates
        lab_batch = lab_grid.unsqueeze(0).expand(P, -1, -1)  # (P, 8000, 3)
        xyz = inv_H(lab_batch, d)
        srgb = l2s((xyz @ MSi.T).clamp(0, 1))

        # In-gamut check: all channels in [0, 1] after round-trip
        lin = s2l(srgb)
        xyz_rt = lin @ MS.T
        err = (xyz - xyz_rt).abs().max(dim=-1).values
        in_gamut = (srgb >= -0.001).all(dim=-1) & (srgb <= 1.001).all(dim=-1) & (err < 0.01)

        # Reshape to (P, 100, 80) — L x C
        in_gamut = in_gamut.reshape(P, 100, 80)
        C_vals = _Cs.unsqueeze(0).unsqueeze(0).expand(P, 100, 80)

        # Max in-gamut chroma at each L
        max_C = torch.where(in_gamut, C_vals, torch.zeros_like(C_vals)).max(dim=2).values  # (P, 100)

        # Cusp = L with max chroma
        cusp_idx = max_C.argmax(dim=1)  # (P,)
        cusp_L[:, hi] = _Ls[cusp_idx]
        cusp_C[:, hi] = max_C.gather(1, cusp_idx.unsqueeze(1)).squeeze(1)

    # Cliff: max single-step chroma drop
    cL_shift = torch.cat([cusp_C[:, 1:], cusp_C[:, :1]], dim=1)
    drop = (cusp_C - cL_shift) / cusp_C.clamp(min=1e-10) * 100
    cliff = drop.clamp(min=0)

    return cusp_L, cusp_C, cliff


def batch_ach(d):
    """Achromatic purity: max |a|,|b| for D65 gray ramp. Returns (P,)."""
    n_gray = 32
    t = torch.linspace(0.01, 0.99, n_gray, device=dev)
    grays = D65.unsqueeze(0) * t.unsqueeze(1)
    lab = fwd_H(grays, d)  # (P, 32, 3)
    return torch.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2).max(dim=1).values


# ═══════════════════════════════════════════════════════════════
#  EVALUATE POPULATION
# ═══════════════════════════════════════════════════════════════

def evaluate(x_np):
    """Evaluate P candidates. Returns losses (P,)."""
    x = torch.tensor(x_np, device=dev, dtype=torch.float64)
    P = x.shape[0]
    losses = torch.full((P,), 999.0, device=dev)

    with torch.no_grad():
        d, valid = unpack_params(x)
        if not valid.any():
            return losses.cpu().numpy()

        v0 = valid.sum().item()

        c1_v, c2_v = batch_cond(d)
        valid &= (c1_v < 20) & (c2_v < 30)  # relaxed from 15/25
        v1 = valid.sum().item()

        if not valid.any():
            if not hasattr(evaluate, '_dbg'):
                evaluate._dbg = True
                print(f"    DBG: unpack={v0} cond={v1}/{P} c1_med={c1_v.median():.1f} c2_med={c2_v.median():.1f}", flush=True)
            return losses.cpu().numpy()

        info = batch_info(d)
        valid &= info["yC"] > 0.03  # relaxed from 0.05
        v2 = valid.sum().item()
        # White L check removed — NR+enrichment changes white L

        if not valid.any():
            if not hasattr(evaluate, '_dbg2'):
                evaluate._dbg2 = True
                print(f"    DBG: unpack={v0} cond={v1} yC={v2}/{P}", flush=True)
            return losses.cpu().numpy()

        n_valid = v2

        # Core metrics
        cv = batch_cv(d)
        munsell_v = batch_munsell_value(d)
        munsell_h = batch_munsell_hue(d)
        hue_lin = batch_hue_linearity(d)
        mono = batch_lightness_mono(d)
        ach = batch_ach(d)

        # Cusp
        cusp_L, cusp_C, cliff = batch_cusp(d)

        # ── LOSS ──
        # Core: gradient CV (main objective)
        loss = 5.0 * cv

        # NEW: Munsell Value uniformity (target: <5%, OKLab=2.6%)
        loss += 0.5 * munsell_v  # Penalize high CV

        # NEW: Munsell Hue spacing (target: <20%, OKLab=17.5%)
        loss += 0.2 * munsell_h

        # NEW: Hue linearity (target: <10°, v7b=7°, OKLab=9°)
        loss += 1.0 * hue_lin

        # NEW: Lightness monotonicity (must be 1.0)
        mono_pen = (1.0 - mono).clamp(min=0) * 500.0
        loss += mono_pen

        # Cusp penalties
        cusp_pen = torch.zeros(P, device=dev)
        for hi in range(cusp_L.shape[1]):
            cusp_pen += torch.where(cusp_L[:, hi] > 0.92, (cusp_L[:, hi] - 0.92) ** 2 * 30, 0.0)
            cusp_pen += torch.where(cusp_L[:, hi] < 0.78, (0.78 - cusp_L[:, hi]) ** 2 * 30, 0.0)
            cusp_pen += torch.where(cliff[:, hi] > 40, (cliff[:, hi] - 40) ** 2 * 0.1, 0.0)
            cusp_pen += torch.where(cliff[:, hi] > 80, (cliff[:, hi] - 80) ** 2 * 2.0, 0.0)
        loss += 2.0 * cusp_pen

        # Dead zones
        dead_count = (cusp_C < 0.02).float().sum(dim=1)
        loss += dead_count * 20.0

        # ── SOFT PENALTIES ──
        pen = torch.zeros(P, device=dev)
        pen += torch.where(info["yC"] < 0.12, (0.12 - info["yC"]) ** 2 * 200, 0.0)
        # Blue L: prefer < 0.60 but tolerate up to 0.85 (H_v2 starts at 0.75)
        pen += torch.where(info["bL"] > 0.60, (info["bL"] - 0.60) ** 2 * 50, 0.0)
        pen += torch.where(info["bL"] > 0.85, (info["bL"] - 0.85) ** 2 * 2000, 0.0)
        pen += torch.where(info["bw"] < 1.20, (1.20 - info["bw"]) ** 2 * 50, 0.0)
        pen += torch.where(info["rw"] > 0.08, (info["rw"] - 0.08) ** 2 * 100, 0.0)
        pen += torch.where(info["plr"] < 0.40, (0.40 - info["plr"]) ** 2 * 500, 0.0)
        pen += torch.where(c1_v > 3.5, (c1_v - 3.5) ** 2 * 5, 0.0)
        pen += torch.where(c2_v > 12, (c2_v - 12) ** 2 * 3, 0.0)
        # Achromatic: NR+enrichment has no structural guarantee, use moderate penalty
        pen += torch.where(ach > 0.001, (ach - 0.001) * 100, 0.0)
        pen += torch.where(ach > 0.01, (ach - 0.01) * 500, 0.0)
        pen += torch.where(ach > 0.1, (ach - 0.1) * 2000, 0.0)
        # M-cone dominant L-row: prefer |M2[0,1]| > |M2[0,0]| and |M2[0,2]|
        m_dom = d["M2"][:, 0, 1].abs()
        m_r0 = d["M2"][:, 0, 0].abs()
        m_r2 = d["M2"][:, 0, 2].abs()
        pen += torch.where(m_dom < m_r0, (m_r0 - m_dom) ** 2 * 100, 0.0)
        pen += torch.where(m_dom < m_r2, (m_r2 - m_dom) ** 2 * 100, 0.0)
        loss += pen

        losses = torch.where(valid, loss, torch.full_like(loss, 999.0))

    return losses.cpu().numpy()


# ═══════════════════════════════════════════════════════════════
#  SEEDS FROM H_v2
# ═══════════════════════════════════════════════════════════════

def make_seed_from_checkpoint(path):
    """Load H_v2 checkpoint and pack into 21-param vector."""
    with open(path) as f:
        p = json.load(f)
    M1 = np.array(p["M1"])
    M2 = np.array(p["M2"])
    n = p["n"]
    sigma = p["sigma"]
    s_gain = p["s_gain"]
    c1_val = p["c1"]
    k_val = p["k"]
    cp_val = p["cp"]

    x = np.zeros(21)
    x[:6] = _pack_m1(M1)
    x[6:15] = _pack_m2(M2)
    x[15] = np.log(n / 0.42)
    x[16] = np.log(sigma)
    x[17] = np.log(s_gain)
    x[18] = c1_val
    x[19] = k_val
    # cp = 0.85 + 0.15 * sigmoid(x[20]) → x[20] = logit((cp - 0.85) / 0.15)
    cp_norm = np.clip((cp_val - 0.85) / 0.15, 0.01, 0.99)
    x[20] = np.log(cp_norm / (1 - cp_norm))

    return x


def make_seeds():
    """Generate seeds: H_v2 + perturbations + random."""
    seeds = []
    h_v2_path = os.path.join(CKPT, "H_v2_params.json")

    if os.path.exists(h_v2_path):
        x0 = make_seed_from_checkpoint(h_v2_path)
        seeds.append(("H_v2", x0, args.sigma))

        # Perturbations of H_v2
        rng = np.random.RandomState(42)
        for i in range(4):
            xp = x0 + rng.randn(21) * 0.02
            seeds.append((f"H_v2_pert{i}", xp, args.sigma))
    else:
        print(f"WARNING: {h_v2_path} not found, using random seeds only")

    # Also try OKLab-like M1 seed with NR
    OKn = np.array([[0.8189, 0.3619, -0.1289],
                     [0.0330, 0.9293,  0.0361],
                     [0.0482, 0.2644,  0.6339]])
    OKn2 = np.array([[ 0.2105,  0.7936, -0.0041],
                      [ 1.9780, -2.4286,  0.4506],
                      [ 0.0259,  0.7828, -0.8087]])
    x_ok = np.zeros(21)
    x_ok[:6] = _pack_m1(OKn)
    x_ok[6:15] = _pack_m2(OKn2)
    x_ok[15] = np.log(0.76 / 0.42)  # n=0.76
    x_ok[16] = np.log(0.33)  # sigma
    x_ok[17] = np.log(0.71)  # s_gain
    x_ok[18] = 0.1  # c1
    x_ok[19] = 0.2  # k
    x_ok[20] = 0.0  # cp≈0.925
    seeds.append(("OKLab_NR", x_ok, 0.05))

    # Random seeds
    rng = np.random.RandomState(100)
    for i in range(max(0, args.seeds - len(seeds))):
        xr = np.zeros(21)
        if os.path.exists(h_v2_path):
            xr = x0 + rng.randn(21) * 0.05
        else:
            xr = rng.randn(21) * 0.1
        seeds.append((f"rnd{i}", xr, 0.05))

    return seeds


# ═══════════════════════════════════════════════════════════════
#  CMA-ES MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def save_checkpoint(d_best, loss, seed_label, gen):
    """Save best params as JSON."""
    d = d_best
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"H_munsell_{seed_label}_{ts}.json"
    path = os.path.join(CKPT, fname)

    M1 = d["M1"].cpu().numpy().tolist()
    M2 = d["M2"].cpu().numpy().tolist()
    M1i = np.linalg.inv(np.array(M1)).tolist()
    M2i = np.linalg.inv(np.array(M2)).tolist()

    out = {
        "M1": M1, "M2": M2, "M1_inv": M1i, "M2_inv": M2i,
        "n": d["n"].item(), "sigma": d["sigma"].item(), "s_gain": d["s"].item(),
        "c1": d["c1"].item(), "k": d["k"].item(), "cp": d["cp"].item(),
        "architecture": "H_NakaRushtonCp_Munsell",
        "loss": loss,
        "generation": gen,
        "seed": seed_label,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {fname} (loss={loss:.4f})", flush=True)
    return path


def run_seed(seed_label, x0, sigma):
    """Run CMA-ES for one seed."""
    print(f"\n{'='*60}")
    print(f"  Seed: {seed_label}, sigma={sigma}, pop={args.pop}, gens={args.gens}")
    print(f"{'='*60}", flush=True)

    opts = cma.CMAOptions()
    opts.set("maxiter", args.gens)
    opts.set("popsize", args.pop)
    opts.set("tolfun", 1e-15)
    opts.set("tolx", 1e-15)
    opts.set("verbose", -1)

    es = cma.CMAEvolutionStrategy(x0, sigma, opts)

    best_loss = 999.0
    best_x = x0.copy()
    gen = 0

    while not es.stop():
        sols = es.ask()
        x_batch = np.array(sols)
        fits = evaluate(x_batch)
        es.tell(sols, fits.tolist())

        best_idx = np.argmin(fits)
        if fits[best_idx] < best_loss:
            best_loss = fits[best_idx]
            best_x = x_batch[best_idx].copy()

        gen += 1
        if gen <= 3:
            n_non999 = int(np.sum(fits < 999))
            print(f"  gen {gen} debug: {n_non999}/{len(fits)} valid, min={fits.min():.4f}", flush=True)
        if gen % 20 == 0 or gen == 1:
            # Quick diagnostic
            x_test = torch.tensor(best_x.reshape(1, -1), device=dev)
            d_test, v = unpack_params(x_test)
            if v.any():
                cv_val = batch_cv(d_test).item()
                mv_val = batch_munsell_value(d_test).item()
                mh_val = batch_munsell_hue(d_test).item()
                hl_val = batch_hue_linearity(d_test).item()
                mono_val = batch_lightness_mono(d_test).item()
                print(f"  gen {gen:4d}  loss={best_loss:.4f}  "
                      f"CV={cv_val:.1f}%  MunsV={mv_val:.1f}%  MunsH={mh_val:.1f}%  "
                      f"HueLin={hl_val:.1f}°  Mono={mono_val:.4f}", flush=True)

    # Save best
    x_final = torch.tensor(best_x.reshape(1, -1), device=dev)
    d_final, v = unpack_params(x_final)
    if v.any():
        # Squeeze batch dimension for saving
        d_save = {k: v[0] if isinstance(v, torch.Tensor) and v.dim() > 0 else v
                  for k, v in d_final.items()}
        ckpt_path = save_checkpoint(d_save, best_loss, seed_label, gen)

        # Final metrics
        cv_val = batch_cv(d_final).item()
        mv_val = batch_munsell_value(d_final).item()
        mh_val = batch_munsell_hue(d_final).item()
        hl_val = batch_hue_linearity(d_final).item()
        mono_val = batch_lightness_mono(d_final).item()
        info = batch_info(d_final)
        print(f"\n  FINAL: loss={best_loss:.4f}")
        print(f"    CV={cv_val:.2f}%  MunsV={mv_val:.2f}%  MunsH={mh_val:.2f}%")
        print(f"    HueLin={hl_val:.2f}°  Mono={mono_val:.4f}")
        print(f"    YellowC={info['yC'].item():.3f}  BlueL={info['bL'].item():.3f}")
        print(f"    B-W G/R={info['bw'].item():.3f}  R-W G-B={info['rw'].item():.3f}")
        return best_loss, ckpt_path
    return best_loss, None


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"  H (NR+Enrich) Re-optimization with Munsell + Hue")
    print(f"  Gens={args.gens}, Pop={args.pop}, Seeds={args.seeds}")
    print(f"{'#'*60}\n", flush=True)

    seeds = make_seeds()
    results = []

    for label, x0, sigma in seeds:
        loss, path = run_seed(label, x0, sigma)
        results.append((label, loss, path))

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    results.sort(key=lambda r: r[1])
    for label, loss, path in results:
        print(f"  {label:20s}  loss={loss:.4f}  {path or 'FAILED'}")
    print(f"\nBest: {results[0][0]} (loss={results[0][1]:.4f})")
