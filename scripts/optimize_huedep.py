#!/usr/bin/env python3
"""Hue-Dependent M2 Optimization for GenSpace.

Architecture: XYZ -> M1(6p,D65) -> cbrt -> lms_c -> [L_row fixed, ab_rows hue-rotated] -> L_corr -> Lab
Total: 22 free parameters.

Key idea: M2's L-row is fixed (achromatic guarantee). M2's a,b rows get a
hue-dependent rotation using Fourier coefficients (2 harmonics = 4 params).
This allows the space to redistribute hue angles nonlinearly while keeping
the achromatic axis perfectly clean.

Forward:
  lms_c = cbrt(clamp(XYZ @ M1.T))
  L = lms_c @ M2_L  (fixed L row, 3 params)
  a_raw = lms_c @ M2_a  (base a row, 3 params)
  b_raw = lms_c @ M2_b  (base b row, 3 params)
  h_est = atan2(b_raw, a_raw)
  theta = c1*cos(h) + s1*sin(h) + c2*cos(2h) + s2*sin(2h)
  a = a_raw*cos(theta) - b_raw*sin(theta)
  b = a_raw*sin(theta) + b_raw*cos(theta)
  L' = L + L_corr(L)

Inverse: undo L_corr (Newton), negate theta rotation, undo M2, cube, undo M1.

Usage:
    python scripts/optimize_huedep.py
    python scripts/optimize_huedep.py --gens 300 --pop 128 --seeds 6
"""

import json, math, os, sys, time, argparse
from datetime import datetime
import numpy as np
import torch
import cma

# ================================================================
#  SETUP
# ================================================================

torch.set_default_dtype(torch.float64)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Device: CUDA (%s)" % torch.cuda.get_device_name(0), flush=True)
else:
    print("Device: CPU", flush=True)

pa = argparse.ArgumentParser()
pa.add_argument("--gens", type=int, default=300)
pa.add_argument("--pop", type=int, default=128)
pa.add_argument("--seeds", type=int, default=6)
pa.add_argument("--sigma", type=float, default=0.03)
args = pa.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
CKPT = os.path.join(ROOT, "checkpoints")
os.makedirs(CKPT, exist_ok=True)

N_PARAMS = 22  # M1(6) + M2_L(3) + M2_a(3) + M2_b(3) + rotation(4) + L_corr(3)

# ================================================================
#  CONSTANTS
# ================================================================

D65 = torch.tensor([0.95047, 1.0, 1.08883], device=dev)
MS = torch.tensor([[.4124564, .3575761, .1804375],
                    [.2126729, .7151522, .0721750],
                    [.0193339, .1191920, .9503041]], device=dev)
MSi = torch.linalg.inv(MS)

def s2l(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))

def l2s(c):
    return torch.where(c <= 0.0031308, c * 12.92,
                       1.055 * c.clamp(min=1e-12).pow(1.0 / 2.4) - 0.055)

# ================================================================
#  TRAINING PAIRS
# ================================================================

_use_imported = False
for _try_dir in [
    os.path.join(ROOT, "space-test-project"),
    os.path.join(ROOT, "colorbench"),
    SCRIPT_DIR,
]:
    if os.path.isdir(os.path.join(_try_dir, "core")):
        sys.path.insert(0, _try_dir)
        try:
            from core.pairs import generate_all_pairs as _gen_pairs
            PT, _labels = _gen_pairs(dev)
            _use_imported = True
            print("Loaded %d pairs from %s" % (PT.shape[0], _try_dir), flush=True)
            break
        except Exception as e:
            print("Pair import failed from %s: %s" % (_try_dir, e), flush=True)

if not _use_imported:
    _pl = []
    _pr = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[0,0,0]]
    for i in range(len(_pr)):
        for j in range(i + 1, len(_pr)):
            _pl.append((_pr[i], _pr[j]))
    _rng = np.random.RandomState(42)
    for _ in range(200):
        _pl.append((_rng.rand(3).tolist(), _rng.rand(3).tolist()))
    PT = torch.zeros(len(_pl), 2, 3, device=dev)
    for i, (c1, c2) in enumerate(_pl):
        PT[i, 0] = MS @ s2l(torch.tensor(c1, device=dev))
        PT[i, 1] = MS @ s2l(torch.tensor(c2, device=dev))
    print("Using fallback %d pairs" % PT.shape[0], flush=True)

N_PAIRS = PT.shape[0]
N_ST = 25
T_ST = torch.linspace(0, 1, N_ST + 1, device=dev)

# ================================================================
#  MUNSELL DATA
# ================================================================

MUNSELL_VALUE_Y = {
    1: 0.01221, 2: 0.03126, 3: 0.06552, 4: 0.12000, 5: 0.19770,
    6: 0.30049, 7: 0.43060, 8: 0.59100, 9: 0.78660,
}
MUNSELL_GRAYS = torch.stack([D65 * MUNSELL_VALUE_Y[v] for v in range(1, 10)]).to(dev)

MUNSELL_HUE_CHIPS_SRGB = torch.tensor([
    [176, 103, 101], [169, 117,  82], [155, 135,  80],
    [115, 143,  87], [ 75, 148, 115], [ 58, 146, 140],
    [ 69, 138, 159], [101, 118, 162], [132, 106, 149],
    [159,  99, 126],
], dtype=torch.float64, device=dev) / 255.0
MUNSELL_HUE_XYZ = (s2l(MUNSELL_HUE_CHIPS_SRGB) @ MS.T)

PRIM_SRGB = torch.tensor([
    [1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,0,1],
], dtype=torch.float64, device=dev)
PRIM_XYZ = (s2l(PRIM_SRGB) @ MS.T)
WHITE_XYZ = D65.unsqueeze(0)

# ================================================================
#  HUE-DEPENDENT M2 FORWARD / INVERSE (P-batched)
# ================================================================

def fwd_huedep(xyz, d):
    """Forward: XYZ -> Lab. xyz: (N, 3), d has P-batched params.
    Returns (P, N, 3)."""
    M1 = d["M1"]  # (P, 3, 3)
    M2_L = d["M2_L"]  # (P, 3)  L row
    M2_a = d["M2_a"]  # (P, 3)  a row
    M2_b = d["M2_b"]  # (P, 3)  b row
    fc1, fs1, fc2, fs2 = d["fc1"], d["fs1"], d["fc2"], d["fs2"]  # (P,)
    lc1, lc2, lc3 = d["lc1"], d["lc2"], d["lc3"]  # (P,)

    # XYZ -> LMS -> cbrt
    lms = (xyz.unsqueeze(0) @ M1.transpose(-1, -2)).clamp(min=0)  # (P, N, 3)
    lms_c = lms.pow(1.0 / 3.0)

    # L channel (achromatic guaranteed: for grays, lms_c is on achromatic axis)
    L = (lms_c * M2_L.unsqueeze(1)).sum(dim=-1)  # (P, N)

    # Raw a, b channels
    a_raw = (lms_c * M2_a.unsqueeze(1)).sum(dim=-1)  # (P, N)
    b_raw = (lms_c * M2_b.unsqueeze(1)).sum(dim=-1)  # (P, N)

    # Hue-dependent rotation
    h_est = torch.atan2(b_raw, a_raw)  # (P, N)
    theta = (fc1.view(-1, 1) * torch.cos(h_est)
             + fs1.view(-1, 1) * torch.sin(h_est)
             + fc2.view(-1, 1) * torch.cos(2 * h_est)
             + fs2.view(-1, 1) * torch.sin(2 * h_est))  # (P, N)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    a = a_raw * cos_t - b_raw * sin_t
    b = a_raw * sin_t + b_raw * cos_t

    # L correction
    t = L * (1.0 - L)
    L_out = L + lc1.view(-1, 1) * t + lc2.view(-1, 1) * t * (2.0 * L - 1.0) + lc3.view(-1, 1) * L * L * (1.0 - L) * (1.0 - L)

    return torch.stack([L_out, a, b], dim=-1)


def inv_huedep(lab, d):
    """Inverse: Lab -> XYZ. lab: (P, N, 3)."""
    M1_i = d["M1_i"]
    M2_L = d["M2_L"]
    M2_a = d["M2_a"]
    M2_b = d["M2_b"]
    fc1, fs1, fc2, fs2 = d["fc1"], d["fs1"], d["fc2"], d["fs2"]
    lc1, lc2, lc3 = d["lc1"], d["lc2"], d["lc3"]

    L_out, a_out, b_out = lab[..., 0], lab[..., 1], lab[..., 2]

    # Undo L correction (Newton)
    L = L_out.clone()
    for _ in range(10):
        t = L * (1.0 - L)
        f = L + lc1.view(-1, 1) * t + lc2.view(-1, 1) * t * (2.0 * L - 1.0) + lc3.view(-1, 1) * L * L * (1.0 - L) * (1.0 - L) - L_out
        df = (1.0 + lc1.view(-1, 1) * (1.0 - 2.0 * L)
              + lc2.view(-1, 1) * (6.0 * L * L - 6.0 * L + 1.0)
              + lc3.view(-1, 1) * 2.0 * L * (1.0 - L) * (1.0 - 2.0 * L))
        L = L - f / df.clamp(min=1e-12)

    # Undo hue rotation: need to figure out theta from (a_out, b_out)
    # The rotated hue is atan2(b_out, a_out), but theta depends on the
    # PRE-rotation hue. We use Newton iteration on the rotation.
    # h_out = h_raw + theta(h_raw), solve for h_raw.
    h_out = torch.atan2(b_out, a_out)
    C_out = torch.sqrt(a_out ** 2 + b_out ** 2 + 1e-30)

    # Newton: find h_raw such that h_raw + theta(h_raw) = h_out
    h_raw = h_out.clone()
    for _ in range(15):
        theta_est = (fc1.view(-1, 1) * torch.cos(h_raw)
                     + fs1.view(-1, 1) * torch.sin(h_raw)
                     + fc2.view(-1, 1) * torch.cos(2 * h_raw)
                     + fs2.view(-1, 1) * torch.sin(2 * h_raw))
        dtheta = (-fc1.view(-1, 1) * torch.sin(h_raw)
                  + fs1.view(-1, 1) * torch.cos(h_raw)
                  - 2 * fc2.view(-1, 1) * torch.sin(2 * h_raw)
                  + 2 * fs2.view(-1, 1) * torch.cos(2 * h_raw))
        residual = h_raw + theta_est - h_out
        # Wrap residual to [-pi, pi]
        residual = torch.remainder(residual + math.pi, 2 * math.pi) - math.pi
        h_raw = h_raw - residual / (1.0 + dtheta).clamp(min=0.1)

    # Undo rotation with found h_raw
    theta_final = (fc1.view(-1, 1) * torch.cos(h_raw)
                   + fs1.view(-1, 1) * torch.sin(h_raw)
                   + fc2.view(-1, 1) * torch.cos(2 * h_raw)
                   + fs2.view(-1, 1) * torch.sin(2 * h_raw))
    cos_t = torch.cos(-theta_final)
    sin_t = torch.sin(-theta_final)
    a_raw = a_out * cos_t - b_out * sin_t
    b_raw = a_out * sin_t + b_out * cos_t

    # Reconstruct lms_c from L, a_raw, b_raw using pseudo-inverse of M2 rows
    # M2 = [M2_L; M2_a; M2_b], so lms_c @ M2.T = [L, a, b]
    # lms_c = [L, a, b] @ inv(M2.T)
    M2 = torch.stack([M2_L, M2_a, M2_b], dim=1)  # (P, 3, 3)
    M2_i = d["M2_i"]
    raw = torch.stack([L, a_raw, b_raw], dim=-1)  # (P, N, 3)
    lms_c = torch.bmm(raw, M2_i.transpose(-1, -2))

    # Undo cbrt -> cube
    lms = lms_c.pow(3.0)

    # Undo M1
    xyz = torch.bmm(lms, M1_i.transpose(-1, -2))

    return xyz


# ================================================================
#  PARAMETER PACKING / UNPACKING
# ================================================================

def unpack_params(x):
    """x: (P, 22) -> dict of batched params, valid mask."""
    P = x.shape[0]

    # M1: 6 params -> 3x3 (D65-constrained)
    M1 = torch.zeros(P, 3, 3, device=dev)
    M1[:, 0, 0] = x[:, 0]; M1[:, 0, 1] = x[:, 1]
    M1[:, 1, 0] = x[:, 2]; M1[:, 1, 1] = x[:, 3]
    M1[:, 2, 0] = x[:, 4]; M1[:, 2, 1] = x[:, 5]
    M1[:, 0, 2] = (1.0 - M1[:, 0, 0] * D65[0] - M1[:, 0, 1] * D65[1]) / D65[2]
    M1[:, 1, 2] = (1.0 - M1[:, 1, 0] * D65[0] - M1[:, 1, 1] * D65[1]) / D65[2]
    M1[:, 2, 2] = (1.0 - M1[:, 2, 0] * D65[0] - M1[:, 2, 1] * D65[1]) / D65[2]

    lms_d65 = (D65.unsqueeze(0).unsqueeze(0) @ M1.transpose(-1, -2)).squeeze(1)
    valid = (lms_d65 > 0.01).all(dim=1)

    # M2 rows
    M2_L = x[:, 6:9]    # (P, 3)
    M2_a = x[:, 9:12]   # (P, 3)
    M2_b = x[:, 12:15]  # (P, 3)

    # Achromatic constraint: M2_L must map achromatic direction to positive L
    # achromatic direction in lms_c space = cbrt(M1 @ D65) for each candidate
    ach_dir = lms_d65.clamp(min=1e-10).pow(1.0 / 3.0)  # (P, 3)
    L_at_white = (M2_L * ach_dir).sum(dim=1)
    valid &= L_at_white > 0.1

    # a,b at achromatic should be zero: M2_a @ ach_dir = 0, M2_b @ ach_dir = 0
    # We enforce this as a soft penalty rather than hard constraint
    a_at_white = (M2_a * ach_dir).sum(dim=1)
    b_at_white = (M2_b * ach_dir).sum(dim=1)

    # Rotation Fourier coefficients
    fc1 = x[:, 15].clamp(-0.5, 0.5)
    fs1 = x[:, 16].clamp(-0.5, 0.5)
    fc2 = x[:, 17].clamp(-0.3, 0.3)
    fs2 = x[:, 18].clamp(-0.3, 0.3)

    # L correction
    lc1 = x[:, 19].clamp(-0.5, 0.5)
    lc2 = x[:, 20].clamp(-0.5, 0.5)
    lc3 = x[:, 21].clamp(-1.0, 1.0)

    # Compute inverses
    M1_i = torch.zeros_like(M1)
    M2_full = torch.stack([M2_L, M2_a, M2_b], dim=1)  # (P, 3, 3)
    M2_i = torch.zeros_like(M2_full)

    det1 = torch.linalg.det(M1)
    det2 = torch.linalg.det(M2_full)
    invertible = (det1.abs() > 1e-10) & (det2.abs() > 1e-10)
    valid &= invertible

    good = valid.nonzero(as_tuple=True)[0]
    if good.numel() > 0:
        M1_i[good] = torch.linalg.inv(M1[good])
        M2_i[good] = torch.linalg.inv(M2_full[good])

    return {
        "M1": M1, "M1_i": M1_i,
        "M2_L": M2_L, "M2_a": M2_a, "M2_b": M2_b, "M2_i": M2_i,
        "fc1": fc1, "fs1": fs1, "fc2": fc2, "fs2": fs2,
        "lc1": lc1, "lc2": lc2, "lc3": lc3,
        "a_at_white": a_at_white, "b_at_white": b_at_white,
    }, valid


# ================================================================
#  BATCH METRICS
# ================================================================

def batch_cv(d):
    """Gradient CV. Returns (P,)."""
    P = d["M1"].shape[0]
    lab1 = fwd_huedep(PT[:, 0], d)
    lab2 = fwd_huedep(PT[:, 1], d)

    t = T_ST.view(1, 1, -1, 1)
    labs = lab1.unsqueeze(2) + t * (lab2 - lab1).unsqueeze(2)
    lf = labs.reshape(P, -1, 3)

    xyz = inv_huedep(lf, d)
    lin = (xyz @ MSi.T).clamp(0, 1)
    s8 = (l2s(lin) * 255).round() / 255.0
    xb = (s2l(s8) @ MS.T)

    r = xb.clamp(min=1e-10) / D65.view(1, 1, 3)
    f = torch.where(r > 0.008856, r.pow(1.0 / 3.0), 7.787 * r + 16.0 / 116.0)
    cl = torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                      200 * (f[..., 1] - f[..., 2])], dim=-1)
    cl = cl.reshape(P, N_PAIRS, N_ST + 1, 3)

    c1_, c2_ = cl[:, :, :-1], cl[:, :, 1:]
    dL = c2_[..., 0] - c1_[..., 0]
    C1 = (c1_[..., 1] ** 2 + c1_[..., 2] ** 2).sqrt()
    C2 = (c2_[..., 1] ** 2 + c2_[..., 2] ** 2).sqrt()
    dC = C2 - C1
    dH = ((c2_[..., 1] - c1_[..., 1]) ** 2 + (c2_[..., 2] - c1_[..., 2]) ** 2 - dC ** 2).clamp(min=0).sqrt()
    SL = 1 + 0.015 * (c1_[..., 0] - 50) ** 2 / (20 + (c1_[..., 0] - 50) ** 2).sqrt()
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
    lab = fwd_huedep(MUNSELL_GRAYS, d)
    L = lab[:, :, 0]
    dL = L[:, 1:] - L[:, :-1]
    mean_dL = dL.abs().mean(dim=1).clamp(min=1e-10)
    std_dL = dL.std(dim=1)
    return std_dL / mean_dL * 100.0


def batch_munsell_hue(d):
    lab = fwd_huedep(MUNSELL_HUE_XYZ, d)
    h = torch.atan2(lab[:, :, 2], lab[:, :, 1])
    h_sorted, _ = torch.sort(h, dim=1)
    dh = h_sorted[:, 1:] - h_sorted[:, :-1]
    last_gap = (h_sorted[:, 0] + 2 * math.pi - h_sorted[:, -1]).unsqueeze(1)
    dh = torch.cat([dh, last_gap], dim=1)
    mean_dh = dh.mean(dim=1).clamp(min=1e-10)
    std_dh = dh.std(dim=1)
    return std_dh / mean_dh * 100.0


def batch_hue_linearity(d):
    P = d["M1"].shape[0]
    n_steps = 11
    t = torch.linspace(0, 1, n_steps, device=dev).view(1, 1, -1, 1)

    lab_prim = fwd_huedep(PRIM_XYZ, d)
    lab_white = fwd_huedep(WHITE_XYZ.expand(6, 3), d)

    lab_start = lab_prim.unsqueeze(2)
    lab_end = lab_white.unsqueeze(2)
    labs = lab_start + t * (lab_end - lab_start)

    h = torch.atan2(labs[..., 2], labs[..., 1])
    h_start = h[:, :, 0:1]
    h_end = h[:, :, -1:]

    t_lin = torch.linspace(0, 1, n_steps, device=dev).view(1, 1, -1)
    dh = h_end - h_start
    dh = torch.where(dh > math.pi, dh - 2 * math.pi, dh)
    dh = torch.where(dh < -math.pi, dh + 2 * math.pi, dh)
    h_expected = h_start + t_lin * dh

    C = torch.sqrt(labs[..., 1] ** 2 + labs[..., 2] ** 2)
    h_diff = h - h_expected
    h_diff = torch.where(h_diff > math.pi, h_diff - 2 * math.pi, h_diff)
    h_diff = torch.where(h_diff < -math.pi, h_diff + 2 * math.pi, h_diff)

    mask = C > 0.01
    h_diff_masked = h_diff * mask.float()
    count = mask.float().sum(dim=(1, 2)).clamp(min=1)
    rms = torch.sqrt((h_diff_masked ** 2).sum(dim=(1, 2)) / count) * (180.0 / math.pi)
    return rms


def batch_lightness_mono(d):
    n_gray = 64
    t = torch.linspace(0.001, 0.999, n_gray, device=dev)
    grays_xyz = D65.unsqueeze(0) * t.unsqueeze(1)
    lab = fwd_huedep(grays_xyz, d)
    L = lab[:, :, 0]
    dL = L[:, 1:] - L[:, :-1]
    return (dL > 0).float().mean(dim=1)


def batch_ach(d):
    n_gray = 32
    t = torch.linspace(0.01, 0.99, n_gray, device=dev)
    grays = D65.unsqueeze(0) * t.unsqueeze(1)
    lab = fwd_huedep(grays, d)
    return torch.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2).max(dim=1).values


def batch_info(d):
    P = d["M1"].shape[0]
    lab_p = fwd_huedep(PRIM_XYZ, d)
    lab_w = fwd_huedep(D65.unsqueeze(0), d)

    L_p = lab_p[:, :, 0]
    yC = torch.sqrt(lab_p[:, 3, 1] ** 2 + lab_p[:, 3, 2] ** 2)
    bL = lab_p[:, 2, 0]

    lab_mid_bw = 0.5 * (lab_p[:, 2:3, :] + lab_w)
    xyz_mid = inv_huedep(lab_mid_bw, d)
    srgb_mid = l2s((xyz_mid @ MSi.T).clamp(0, 1))
    bw_gr = srgb_mid[:, 0, 1] / srgb_mid[:, 0, 0].clamp(min=1e-10)

    lab_mid_rw = 0.5 * (lab_p[:, 0:1, :] + lab_w)
    xyz_mid_r = inv_huedep(lab_mid_rw, d)
    srgb_mid_r = l2s((xyz_mid_r @ MSi.T).clamp(0, 1))
    rw_gb = srgb_mid_r[:, 0, 1] - srgb_mid_r[:, 0, 2]

    plr = L_p.max(dim=1).values - L_p.min(dim=1).values
    wL = lab_w[:, 0, 0]

    return {"yC": yC, "bL": bL, "bw": bw_gr.squeeze(), "rw": rw_gb.squeeze(),
            "plr": plr, "wL": wL}


# ================================================================
#  EVALUATE POPULATION
# ================================================================

def evaluate(x_np):
    """Evaluate P candidates. Returns losses (P,)."""
    x = torch.tensor(x_np, device=dev, dtype=torch.float64)
    P = x.shape[0]
    losses = torch.full((P,), 999.0, device=dev)

    with torch.no_grad():
        d, valid = unpack_params(x)
        if not valid.any():
            return losses.cpu().numpy()

        c1 = torch.linalg.cond(d["M1"])
        M2_full = torch.stack([d["M2_L"], d["M2_a"], d["M2_b"]], dim=1)
        c2 = torch.linalg.cond(M2_full)
        valid &= (c1 < 20) & (c2 < 30)

        if not valid.any():
            return losses.cpu().numpy()

        info = batch_info(d)
        valid &= info["yC"] > 0.03

        if not valid.any():
            return losses.cpu().numpy()

        # Core metrics
        cv = batch_cv(d)
        munsell_v = batch_munsell_value(d)
        munsell_h = batch_munsell_hue(d)
        hue_lin = batch_hue_linearity(d)
        mono = batch_lightness_mono(d)
        ach = batch_ach(d)

        # Loss
        loss = 5.0 * cv
        loss += 0.5 * munsell_v
        loss += 0.2 * munsell_h
        loss += 1.0 * hue_lin
        loss += (1.0 - mono).clamp(min=0) * 500.0

        # Soft penalties
        pen = torch.zeros(P, device=dev)
        pen += torch.where(info["yC"] < 0.12, (0.12 - info["yC"]) ** 2 * 200, torch.zeros(1, device=dev))
        pen += torch.where(info["bL"] > 0.60, (info["bL"] - 0.60) ** 2 * 50, torch.zeros(1, device=dev))
        pen += torch.where(info["bw"] < 1.20, (1.20 - info["bw"]) ** 2 * 50, torch.zeros(1, device=dev))
        pen += torch.where(info["rw"] > 0.08, (info["rw"] - 0.08) ** 2 * 100, torch.zeros(1, device=dev))
        pen += torch.where(info["plr"] < 0.40, (0.40 - info["plr"]) ** 2 * 500, torch.zeros(1, device=dev))

        # Achromatic penalty: structural guarantee via M2 row design
        # But the hue rotation could break it slightly, so still penalize
        pen += torch.where(ach > 0.0001, (ach - 0.0001) * 200, torch.zeros(1, device=dev))
        pen += torch.where(ach > 0.001, (ach - 0.001) * 1000, torch.zeros(1, device=dev))
        pen += torch.where(ach > 0.01, (ach - 0.01) * 5000, torch.zeros(1, device=dev))

        # Achromatic axis alignment: M2_a and M2_b should be orthogonal to achromatic
        a_ach = d["a_at_white"].abs()
        b_ach = d["b_at_white"].abs()
        pen += a_ach ** 2 * 500
        pen += b_ach ** 2 * 500

        # Condition number soft penalty
        pen += torch.where(c1 > 8, (c1 - 8) ** 2 * 3, torch.zeros(1, device=dev))
        pen += torch.where(c2 > 12, (c2 - 12) ** 2 * 3, torch.zeros(1, device=dev))

        loss += pen

        losses = torch.where(valid, loss, torch.full_like(loss, 999.0))

    return losses.cpu().numpy()


# ================================================================
#  SEEDS FROM OKLAB
# ================================================================

M_XYZ_TO_SRGB = np.linalg.inv(np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
]))

OKLAB_M1_srgb = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
OKLAB_M1 = OKLAB_M1_srgb @ M_XYZ_TO_SRGB

OKLAB_M2 = np.array([
    [ 0.2104542553,  0.7936177850, -0.0040720468],
    [ 1.9779984951, -2.4285922050,  0.4505937099],
    [ 0.0259040371,  0.7827717662, -0.8086757660],
])


def _pack_m1_d65(M1_np):
    return np.array([M1_np[0, 0], M1_np[0, 1],
                     M1_np[1, 0], M1_np[1, 1],
                     M1_np[2, 0], M1_np[2, 1]])


def make_seeds():
    seeds = []
    rng = np.random.RandomState(42)

    # Seed 0: OKLab direct
    x0 = np.zeros(N_PARAMS)
    x0[:6] = _pack_m1_d65(OKLAB_M1)
    x0[6:9] = OKLAB_M2[0]    # M2_L row
    x0[9:12] = OKLAB_M2[1]   # M2_a row
    x0[12:15] = OKLAB_M2[2]  # M2_b row
    x0[15:19] = [0.0, 0.0, 0.0, 0.0]  # rotation = 0
    x0[19:22] = [0.0, 0.0, 0.0]  # L_corr = 0
    seeds.append(("oklab_base", x0.copy(), args.sigma))

    # Seed 1: OKLab + slight rotation
    x1 = x0.copy()
    x1[15:19] = [0.05, 0.02, 0.01, -0.01]
    seeds.append(("oklab_rot", x1, args.sigma))

    # Seed 2: OKLab + L_corr
    x2 = x0.copy()
    x2[19:22] = [-0.05, 0.05, 0.1]
    seeds.append(("oklab_lcorr", x2, args.sigma))

    # Seed 3: OKLab + both
    x3 = x0.copy()
    x3[15:19] = [0.03, -0.02, 0.01, 0.02]
    x3[19:22] = [-0.03, 0.03, 0.05]
    seeds.append(("oklab_full", x3, 0.04))

    # Seed 4-5: random perturbations
    for i in range(max(0, args.seeds - 4)):
        xr = x0.copy() + rng.randn(N_PARAMS) * 0.04
        # Keep rotation small
        xr[15:19] = np.clip(xr[15:19], -0.3, 0.3)
        seeds.append(("rnd%d" % i, xr, 0.05))

    return seeds[:args.seeds]


# ================================================================
#  CHECKPOINT SAVING
# ================================================================

def save_checkpoint(best_x, loss, seed_label, gen):
    x = torch.tensor(best_x.reshape(1, -1), device=dev)
    d, v = unpack_params(x)
    if not v.any():
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = "huedep_%s_%s.json" % (seed_label, ts)
    path = os.path.join(CKPT, fname)

    M2_full = np.array([d["M2_L"][0].cpu().numpy(),
                        d["M2_a"][0].cpu().numpy(),
                        d["M2_b"][0].cpu().numpy()])

    out = {
        "architecture": "HueDep_M1_cbrt_M2rot_Lcorr",
        "M1": d["M1"][0].cpu().numpy().tolist(),
        "M1_inv": np.linalg.inv(d["M1"][0].cpu().numpy()).tolist(),
        "M2_L": d["M2_L"][0].cpu().numpy().tolist(),
        "M2_a": d["M2_a"][0].cpu().numpy().tolist(),
        "M2_b": d["M2_b"][0].cpu().numpy().tolist(),
        "M2_full": M2_full.tolist(),
        "M2_inv": np.linalg.inv(M2_full).tolist(),
        "rotation_fourier": {
            "c1": d["fc1"][0].item(), "s1": d["fs1"][0].item(),
            "c2": d["fc2"][0].item(), "s2": d["fs2"][0].item(),
        },
        "L_corr": [d["lc1"][0].item(), d["lc2"][0].item(), d["lc3"][0].item()],
        "loss": float(loss),
        "generation": gen,
        "seed": seed_label,
        "n_params": N_PARAMS,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("  Saved: %s (loss=%.4f)" % (fname, loss), flush=True)
    return path


# ================================================================
#  CMA-ES MAIN LOOP
# ================================================================

def run_seed(seed_label, x0, sigma):
    print("\n" + "=" * 60)
    print("  Seed: %s, sigma=%.3f, pop=%d, gens=%d" % (seed_label, sigma, args.pop, args.gens))
    print("=" * 60, flush=True)

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
            print("  gen %d debug: %d/%d valid, min=%.4f" % (gen, n_non999, len(fits), fits.min()), flush=True)

        if gen % 20 == 0 or gen == 1:
            x_test = torch.tensor(best_x.reshape(1, -1), device=dev)
            d_test, v = unpack_params(x_test)
            if v.any():
                cv_val = batch_cv(d_test).item()
                mv_val = batch_munsell_value(d_test).item()
                mh_val = batch_munsell_hue(d_test).item()
                hl_val = batch_hue_linearity(d_test).item()
                mono_val = batch_lightness_mono(d_test).item()
                ach_val = batch_ach(d_test).item()
                print("  gen %4d  loss=%.4f  CV=%.1f%%  MunsV=%.1f%%  MunsH=%.1f%%  "
                      "HueLin=%.1f deg  Mono=%.4f  Ach=%.6f"
                      % (gen, best_loss, cv_val, mv_val, mh_val, hl_val, mono_val, ach_val), flush=True)

    # Save best
    ckpt_path = save_checkpoint(best_x, best_loss, seed_label, gen)

    # Final metrics
    x_final = torch.tensor(best_x.reshape(1, -1), device=dev)
    d_final, v = unpack_params(x_final)
    if v.any():
        cv_val = batch_cv(d_final).item()
        mv_val = batch_munsell_value(d_final).item()
        mh_val = batch_munsell_hue(d_final).item()
        hl_val = batch_hue_linearity(d_final).item()
        mono_val = batch_lightness_mono(d_final).item()
        ach_val = batch_ach(d_final).item()
        info = batch_info(d_final)
        print("\n  FINAL: loss=%.4f" % best_loss)
        print("    CV=%.2f%%  MunsV=%.2f%%  MunsH=%.2f%%" % (cv_val, mv_val, mh_val))
        print("    HueLin=%.2f deg  Mono=%.4f  Ach=%.6f" % (hl_val, mono_val, ach_val))
        print("    YellowC=%.3f  BlueL=%.3f" % (info["yC"].item(), info["bL"].item()))
        print("    B-W G/R=%.3f  R-W G-B=%.3f" % (info["bw"].item(), info["rw"].item()))

    return best_loss, ckpt_path


# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Hue-Dependent M2: M1->cbrt->M2_rows->hue_rotation->L_corr")
    print("  Params=%d, Gens=%d, Pop=%d, Seeds=%d" % (N_PARAMS, args.gens, args.pop, args.seeds))
    print("#" * 60 + "\n", flush=True)

    seeds = make_seeds()
    results = []

    for label, x0, sigma in seeds:
        loss, path = run_seed(label, x0, sigma)
        results.append((label, loss, path))

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    results.sort(key=lambda r: r[1])
    for label, loss, path in results:
        print("  %-20s  loss=%.4f  %s" % (label, loss, path or "FAILED"))
    print("\nBest: %s (loss=%.4f)" % (results[0][0], results[0][1]))
