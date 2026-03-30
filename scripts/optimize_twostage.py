#!/usr/bin/env python3
"""Two-Stage Pipeline Optimization for GenSpace.

Architecture: XYZ -> M1a(6p,D65) -> cbrt -> M1b(9p) -> cbrt -> M2(9p) -> L_corr(3p) -> Lab
Total: 27 free parameters.

Two separate nonlinear stages:
  - M1a: cone fundamentals (D65-constrained, 6 free params)
  - First cbrt: compresses for L uniformity
  - M1b: remixes after first compression (9 free params)
  - Second cbrt: further compression for hue/chroma
  - M2: opponent transform (9 free params)
  - L_corr: cubic L correction (3 params)

Inverse: reverse order. Each cbrt inverts to cube, each matrix inverts.

Usage:
    python scripts/optimize_twostage.py
    python scripts/optimize_twostage.py --gens 300 --pop 128 --seeds 6
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

N_PARAMS = 27  # M1a(6) + M1b(9) + M2(9) + L_corr(3)

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
#  TWO-STAGE FORWARD / INVERSE (P-batched)
# ================================================================

def fwd_twostage(xyz, d):
    """Forward: XYZ -> Lab. xyz: (N, 3), d has P-batched params.
    Returns (P, N, 3)."""
    M1a, M1b, M2 = d["M1a"], d["M1b"], d["M2"]
    c1, c2, c3 = d["c1"], d["c2"], d["c3"]

    # Stage 1: XYZ -> M1a -> cbrt
    lms1 = (xyz.unsqueeze(0) @ M1a.transpose(-1, -2)).clamp(min=0)  # (P, N, 3)
    inter = lms1.pow(1.0 / 3.0)

    # Stage 2: inter -> M1b -> cbrt
    lms2 = (torch.bmm(inter, M1b.transpose(-1, -2))).clamp(min=0)   # (P, N, 3)
    opponent = lms2.pow(1.0 / 3.0)

    # M2: opponent -> Lab
    lab = torch.bmm(opponent, M2.transpose(-1, -2))  # (P, N, 3)

    # L correction
    L = lab[..., 0]
    t = L * (1.0 - L)
    L_out = L + c1.view(-1, 1) * t + c2.view(-1, 1) * t * (2.0 * L - 1.0) + c3.view(-1, 1) * L * L * (1.0 - L) * (1.0 - L)

    return torch.stack([L_out, lab[..., 1], lab[..., 2]], dim=-1)


def inv_twostage(lab, d):
    """Inverse: Lab -> XYZ. lab: (P, N, 3)."""
    M1a_i, M1b_i, M2_i = d["M1a_i"], d["M1b_i"], d["M2_i"]
    c1, c2, c3 = d["c1"], d["c2"], d["c3"]

    L_out = lab[..., 0]

    # Undo L correction (Newton)
    L = L_out.clone()
    for _ in range(10):
        t = L * (1.0 - L)
        f = L + c1.view(-1, 1) * t + c2.view(-1, 1) * t * (2.0 * L - 1.0) + c3.view(-1, 1) * L * L * (1.0 - L) * (1.0 - L) - L_out
        df = (1.0 + c1.view(-1, 1) * (1.0 - 2.0 * L)
              + c2.view(-1, 1) * (6.0 * L * L - 6.0 * L + 1.0)
              + c3.view(-1, 1) * 2.0 * L * (1.0 - L) * (1.0 - 2.0 * L))
        L = L - f / df.clamp(min=1e-12)

    raw = torch.stack([L, lab[..., 1], lab[..., 2]], dim=-1)

    # Undo M2
    opponent = torch.bmm(raw, M2_i.transpose(-1, -2))

    # Undo second cbrt -> cube
    lms2 = opponent.pow(3.0)

    # Undo M1b
    inter = torch.bmm(lms2, M1b_i.transpose(-1, -2))

    # Undo first cbrt -> cube
    lms1 = inter.pow(3.0)

    # Undo M1a
    xyz_out = torch.bmm(lms1, M1a_i.transpose(-1, -2))

    return xyz_out


# ================================================================
#  PARAMETER PACKING / UNPACKING
# ================================================================

def unpack_params(x):
    """x: (P, 27) -> dict of batched params, valid mask."""
    P = x.shape[0]

    # M1a: 6 params -> 3x3 (D65-constrained: M1a @ D65 = [1,1,1])
    M1a = torch.zeros(P, 3, 3, device=dev)
    M1a[:, 0, 0] = x[:, 0]; M1a[:, 0, 1] = x[:, 1]
    M1a[:, 1, 0] = x[:, 2]; M1a[:, 1, 1] = x[:, 3]
    M1a[:, 2, 0] = x[:, 4]; M1a[:, 2, 1] = x[:, 5]
    M1a[:, 0, 2] = (1.0 - M1a[:, 0, 0] * D65[0] - M1a[:, 0, 1] * D65[1]) / D65[2]
    M1a[:, 1, 2] = (1.0 - M1a[:, 1, 0] * D65[0] - M1a[:, 1, 1] * D65[1]) / D65[2]
    M1a[:, 2, 2] = (1.0 - M1a[:, 2, 0] * D65[0] - M1a[:, 2, 1] * D65[1]) / D65[2]

    lms_d65 = (D65.unsqueeze(0).unsqueeze(0) @ M1a.transpose(-1, -2)).squeeze(1)
    valid = (lms_d65 > 0.01).all(dim=1)

    # M1b: 9 params -> 3x3 (all free)
    M1b = torch.zeros(P, 3, 3, device=dev)
    M1b[:, 0, 0] = x[:, 6];  M1b[:, 0, 1] = x[:, 7];  M1b[:, 0, 2] = x[:, 8]
    M1b[:, 1, 0] = x[:, 9];  M1b[:, 1, 1] = x[:, 10]; M1b[:, 1, 2] = x[:, 11]
    M1b[:, 2, 0] = x[:, 12]; M1b[:, 2, 1] = x[:, 13]; M1b[:, 2, 2] = x[:, 14]

    # M2: 9 params -> 3x3 (all free)
    M2 = torch.zeros(P, 3, 3, device=dev)
    M2[:, 0, 0] = x[:, 15]; M2[:, 0, 1] = x[:, 16]; M2[:, 0, 2] = x[:, 17]
    M2[:, 1, 0] = x[:, 18]; M2[:, 1, 1] = x[:, 19]; M2[:, 1, 2] = x[:, 20]
    M2[:, 2, 0] = x[:, 21]; M2[:, 2, 1] = x[:, 22]; M2[:, 2, 2] = x[:, 23]

    # L_corr: 3 params
    c1 = x[:, 24].clamp(-0.5, 0.5)
    c2 = x[:, 25].clamp(-0.5, 0.5)
    c3 = x[:, 26].clamp(-1.0, 1.0)

    # Compute inverses
    M1a_i = torch.zeros_like(M1a)
    M1b_i = torch.zeros_like(M1b)
    M2_i = torch.zeros_like(M2)

    det1a = torch.linalg.det(M1a)
    det1b = torch.linalg.det(M1b)
    det2 = torch.linalg.det(M2)
    invertible = (det1a.abs() > 1e-10) & (det1b.abs() > 1e-10) & (det2.abs() > 1e-10)
    valid &= invertible

    good = valid.nonzero(as_tuple=True)[0]
    if good.numel() > 0:
        M1a_i[good] = torch.linalg.inv(M1a[good])
        M1b_i[good] = torch.linalg.inv(M1b[good])
        M2_i[good] = torch.linalg.inv(M2[good])

    # Achromatic constraint: cbrt(M1a@D65) through M1b, then cbrt, through M2 should give L>0, a~0, b~0
    # We enforce that via soft penalties in evaluate()

    return {
        "M1a": M1a, "M1b": M1b, "M2": M2,
        "M1a_i": M1a_i, "M1b_i": M1b_i, "M2_i": M2_i,
        "c1": c1, "c2": c2, "c3": c3,
    }, valid


# ================================================================
#  BATCH METRICS
# ================================================================

def batch_cv(d):
    """Gradient CV (CIEDE2000-based). Returns (P,)."""
    P = d["M1a"].shape[0]
    lab1 = fwd_twostage(PT[:, 0], d)
    lab2 = fwd_twostage(PT[:, 1], d)

    t = T_ST.view(1, 1, -1, 1)
    labs = lab1.unsqueeze(2) + t * (lab2 - lab1).unsqueeze(2)
    lf = labs.reshape(P, -1, 3)

    xyz = inv_twostage(lf, d)
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
    """Munsell Value uniformity. Returns (P,)."""
    lab = fwd_twostage(MUNSELL_GRAYS, d)
    L = lab[:, :, 0]
    dL = L[:, 1:] - L[:, :-1]
    mean_dL = dL.abs().mean(dim=1).clamp(min=1e-10)
    std_dL = dL.std(dim=1)
    return std_dL / mean_dL * 100.0


def batch_munsell_hue(d):
    """Munsell Hue spacing uniformity. Returns (P,)."""
    lab = fwd_twostage(MUNSELL_HUE_XYZ, d)
    h = torch.atan2(lab[:, :, 2], lab[:, :, 1])
    h_sorted, _ = torch.sort(h, dim=1)
    dh = h_sorted[:, 1:] - h_sorted[:, :-1]
    last_gap = (h_sorted[:, 0] + 2 * math.pi - h_sorted[:, -1]).unsqueeze(1)
    dh = torch.cat([dh, last_gap], dim=1)
    mean_dh = dh.mean(dim=1).clamp(min=1e-10)
    std_dh = dh.std(dim=1)
    return std_dh / mean_dh * 100.0


def batch_hue_linearity(d):
    """Hue linearity: RMS hue deviation for primary->white gradients. Returns (P,)."""
    P = d["M1a"].shape[0]
    n_steps = 11
    t = torch.linspace(0, 1, n_steps, device=dev).view(1, 1, -1, 1)

    lab_prim = fwd_twostage(PRIM_XYZ, d)
    lab_white = fwd_twostage(WHITE_XYZ.expand(6, 3), d)

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
    """Lightness monotonicity. Returns (P,)."""
    n_gray = 64
    t = torch.linspace(0.001, 0.999, n_gray, device=dev)
    grays_xyz = D65.unsqueeze(0) * t.unsqueeze(1)
    lab = fwd_twostage(grays_xyz, d)
    L = lab[:, :, 0]
    dL = L[:, 1:] - L[:, :-1]
    return (dL > 0).float().mean(dim=1)


def batch_ach(d):
    """Achromatic purity. Returns (P,)."""
    n_gray = 32
    t = torch.linspace(0.01, 0.99, n_gray, device=dev)
    grays = D65.unsqueeze(0) * t.unsqueeze(1)
    lab = fwd_twostage(grays, d)
    return torch.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2).max(dim=1).values


def batch_info(d):
    """Yellow chroma, blue L, midpoint info."""
    P = d["M1a"].shape[0]
    lab_p = fwd_twostage(PRIM_XYZ, d)
    lab_w = fwd_twostage(D65.unsqueeze(0), d)

    L_p = lab_p[:, :, 0]
    yC = torch.sqrt(lab_p[:, 3, 1] ** 2 + lab_p[:, 3, 2] ** 2)
    bL = lab_p[:, 2, 0]

    lab_mid_bw = 0.5 * (lab_p[:, 2:3, :] + lab_w)
    xyz_mid = inv_twostage(lab_mid_bw, d)
    srgb_mid = l2s((xyz_mid @ MSi.T).clamp(0, 1))
    bw_gr = srgb_mid[:, 0, 1] / srgb_mid[:, 0, 0].clamp(min=1e-10)

    lab_mid_rw = 0.5 * (lab_p[:, 0:1, :] + lab_w)
    xyz_mid_r = inv_twostage(lab_mid_rw, d)
    srgb_mid_r = l2s((xyz_mid_r @ MSi.T).clamp(0, 1))
    rw_gb = srgb_mid_r[:, 0, 1] - srgb_mid_r[:, 0, 2]

    plr = L_p.max(dim=1).values - L_p.min(dim=1).values
    wL = lab_w[:, 0, 0]

    return {"yC": yC, "bL": bL, "bw": bw_gr.squeeze(), "rw": rw_gb.squeeze(),
            "plr": plr, "wL": wL}


def batch_cond(d):
    """Condition numbers."""
    c1a = torch.linalg.cond(d["M1a"])
    c1b = torch.linalg.cond(d["M1b"])
    c2 = torch.linalg.cond(d["M2"])
    return c1a, c1b, c2


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

        c1a, c1b, c2 = batch_cond(d)
        valid &= (c1a < 25) & (c1b < 25) & (c2 < 30)

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

        # Achromatic penalty (two-stage has no structural guarantee)
        pen += torch.where(ach > 0.001, (ach - 0.001) * 100, torch.zeros(1, device=dev))
        pen += torch.where(ach > 0.01, (ach - 0.01) * 500, torch.zeros(1, device=dev))
        pen += torch.where(ach > 0.1, (ach - 0.1) * 2000, torch.zeros(1, device=dev))

        # Condition number soft penalty
        pen += torch.where(c1a > 10, (c1a - 10) ** 2 * 2, torch.zeros(1, device=dev))
        pen += torch.where(c1b > 10, (c1b - 10) ** 2 * 2, torch.zeros(1, device=dev))
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
    """Pack 3x3 M1 with D65 constraint into 6 free params."""
    return np.array([M1_np[0, 0], M1_np[0, 1],
                     M1_np[1, 0], M1_np[1, 1],
                     M1_np[2, 0], M1_np[2, 1]])


def _pack_m_full(M_np):
    """Pack 3x3 matrix into 9 params."""
    return M_np.ravel()


def make_seeds():
    """Generate seeds from OKLab base."""
    seeds = []
    rng = np.random.RandomState(42)

    # Seed 0: OKLab M1a + Identity M1b + OKLab M2
    x0 = np.zeros(N_PARAMS)
    x0[:6] = _pack_m1_d65(OKLAB_M1)
    x0[6:15] = _pack_m_full(np.eye(3))    # M1b = identity
    x0[15:24] = _pack_m_full(OKLAB_M2)
    x0[24:27] = [0.0, 0.0, 0.0]  # L_corr
    seeds.append(("oklab_id", x0.copy(), args.sigma))

    # Seed 1: OKLab M1a, slight perturbation of M1b
    x1 = x0.copy()
    x1[6:15] = _pack_m_full(np.eye(3) + rng.randn(3, 3) * 0.05)
    seeds.append(("oklab_pert_m1b", x1, args.sigma))

    # Seed 2: Split OKLab M1 into two stages (sqrt decomposition idea)
    # M1a ~ sqrt-ish of OKLab_M1, M1b captures the rest
    x2 = x0.copy()
    x2[6:15] = _pack_m_full(np.eye(3) + rng.randn(3, 3) * 0.1)
    x2[24:27] = [-0.05, 0.05, 0.1]  # non-zero L_corr
    seeds.append(("oklab_split", x2, 0.05))

    # Seed 3-5: random perturbations
    for i in range(max(0, args.seeds - 3)):
        xr = x0.copy() + rng.randn(N_PARAMS) * 0.04
        seeds.append(("rnd%d" % i, xr, 0.05))

    return seeds[:args.seeds]


# ================================================================
#  CHECKPOINT SAVING
# ================================================================

def save_checkpoint(best_x, loss, seed_label, gen):
    """Save best params as JSON."""
    x = torch.tensor(best_x.reshape(1, -1), device=dev)
    d, v = unpack_params(x)
    if not v.any():
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = "twostage_%s_%s.json" % (seed_label, ts)
    path = os.path.join(CKPT, fname)

    out = {
        "architecture": "TwoStage_M1a_cbrt_M1b_cbrt_M2_Lcorr",
        "M1a": d["M1a"][0].cpu().numpy().tolist(),
        "M1b": d["M1b"][0].cpu().numpy().tolist(),
        "M2": d["M2"][0].cpu().numpy().tolist(),
        "M1a_inv": np.linalg.inv(np.array(d["M1a"][0].cpu().numpy())).tolist(),
        "M1b_inv": np.linalg.inv(np.array(d["M1b"][0].cpu().numpy())).tolist(),
        "M2_inv": np.linalg.inv(np.array(d["M2"][0].cpu().numpy())).tolist(),
        "L_corr": [d["c1"][0].item(), d["c2"][0].item(), d["c3"][0].item()],
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
    """Run CMA-ES for one seed."""
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
    print("  Two-Stage Pipeline: M1a->cbrt->M1b->cbrt->M2->L_corr")
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
