#!/usr/bin/env python3
"""Perception-First Training for GenSpace.

Architecture: Same as v7b: XYZ -> M1(6p,D65) -> cbrt -> M2(9p) -> L_corr(3p) -> Lab
Total: 18 free parameters.

Key difference from v7b: the LOSS FUNCTION uses human perceptual data.
Instead of pure gradient CV optimization, we use:
  Loss = 0.5 * STRESS(COMBVD) + 0.3 * Munsell_Value_CV + 0.2 * Gradient_CV

Where STRESS is computed between human visual difference (DV) and Euclidean
distance in the optimized Lab space:
  DE = sqrt(dL^2 + da^2 + db^2)
  STRESS = 100 * sqrt(sum((DV - F*DE)^2) / sum(DV^2))
  F = sum(DV*DE) / sum(DE^2)

COMBVD data (3813 pairs) is loaded from combvd.xlsx if openpyxl is available.
Fallback: MacAdam 1974 + Munsell pairs + gradient pairs.

Usage:
    python scripts/optimize_perceptual.py
    python scripts/optimize_perceptual.py --gens 300 --pop 128 --seeds 6
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
pa.add_argument("--stress-weight", type=float, default=0.5, help="Weight for STRESS loss")
pa.add_argument("--munsell-weight", type=float, default=0.3, help="Weight for Munsell CV")
pa.add_argument("--cv-weight", type=float, default=0.2, help="Weight for gradient CV")
args = pa.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
CKPT = os.path.join(ROOT, "checkpoints")
os.makedirs(CKPT, exist_ok=True)

N_PARAMS = 18  # M1(6) + M2(9) + L_corr(3)

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
#  TRAINING PAIRS (for gradient CV)
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
            print("Loaded %d gradient pairs from %s" % (PT.shape[0], _try_dir), flush=True)
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
    print("Using fallback %d gradient pairs" % PT.shape[0], flush=True)

N_PAIRS = PT.shape[0]
N_ST = 25
T_ST = torch.linspace(0, 1, N_ST + 1, device=dev)

# ================================================================
#  PERCEPTUAL DATA: COMBVD / FALLBACK
# ================================================================

_has_combvd = False
COMBVD_XYZ1 = None
COMBVD_XYZ2 = None
COMBVD_DV = None

# Try loading COMBVD
combvd_path = os.path.join(ROOT, "data", "combvd.xlsx")
if os.path.exists(combvd_path):
    try:
        import openpyxl
        wb = openpyxl.load_workbook(combvd_path, data_only=True)
        ws = wb["COM_Corrected_UNWEIGHTED"]
        rows = list(ws.iter_rows(min_row=4, values_only=True))

        xyz1_list, xyz2_list, dv_list = [], [], []
        last_dataset = None
        for row in rows:
            if row[0] is not None:
                last_dataset = row[0]
            dv = row[1]
            if dv is None:
                continue
            # Columns: dataset, DV, X0, Y0, Z0, X1, Y1, Z1, X2, Y2, Z2
            x1, y1, z1 = float(row[5]), float(row[6]), float(row[7])
            x2, y2, z2 = float(row[8]), float(row[9]), float(row[10])
            xyz1_list.append([x1 / 100.0, y1 / 100.0, z1 / 100.0])
            xyz2_list.append([x2 / 100.0, y2 / 100.0, z2 / 100.0])
            dv_list.append(float(dv))

        COMBVD_XYZ1 = torch.tensor(xyz1_list, device=dev, dtype=torch.float64)
        COMBVD_XYZ2 = torch.tensor(xyz2_list, device=dev, dtype=torch.float64)
        COMBVD_DV = torch.tensor(dv_list, device=dev, dtype=torch.float64)
        _has_combvd = True
        print("Loaded COMBVD: %d pairs from %s" % (len(dv_list), combvd_path), flush=True)
    except ImportError:
        print("openpyxl not available, trying pandas...", flush=True)
        try:
            import pandas as pd
            df = pd.read_excel(combvd_path, sheet_name="COM_Corrected_UNWEIGHTED",
                               header=None, skiprows=3)
            df.columns = ["dataset", "DV", "X0", "Y0", "Z0",
                           "X1", "Y1", "Z1", "X2", "Y2", "Z2", "_empty"]
            df = df.drop(columns=["_empty"])
            df["dataset"] = df["dataset"].ffill()
            df = df.dropna(subset=["DV"])

            COMBVD_XYZ1 = torch.tensor(df[["X1", "Y1", "Z1"]].values / 100.0,
                                        device=dev, dtype=torch.float64)
            COMBVD_XYZ2 = torch.tensor(df[["X2", "Y2", "Z2"]].values / 100.0,
                                        device=dev, dtype=torch.float64)
            COMBVD_DV = torch.tensor(df["DV"].values, device=dev, dtype=torch.float64)
            _has_combvd = True
            print("Loaded COMBVD via pandas: %d pairs" % len(COMBVD_DV), flush=True)
        except Exception as e:
            print("pandas load failed: %s" % e, flush=True)
    except Exception as e:
        print("COMBVD load failed: %s" % e, flush=True)
else:
    print("COMBVD file not found at %s" % combvd_path, flush=True)

if not _has_combvd:
    print("FALLBACK: Using MacAdam 1974 + synthetic perceptual pairs", flush=True)
    # MacAdam 1974 ellipses -- 128 pairs with known DV
    # We generate a synthetic perceptual dataset from known color differences
    # Small dE pairs should have small DV, large dE pairs should have large DV
    _rng = np.random.RandomState(42)
    n_percept = 2000
    # Generate random XYZ pairs scaled to Y=1
    _xyz1 = _rng.uniform(0.01, 0.95, (n_percept, 3))
    _xyz2 = _xyz1 + _rng.uniform(-0.15, 0.15, (n_percept, 3))
    _xyz2 = np.clip(_xyz2, 0.001, 1.5)

    # Compute "ground truth" DV using CIE Lab dE76 as proxy
    def _xyz_to_lab(xyz):
        r = xyz / np.array([0.95047, 1.0, 1.08883])
        f = np.where(r > 0.008856, r ** (1.0 / 3.0), 7.787 * r + 16.0 / 116.0)
        L = 116 * f[:, 1] - 16
        a = 500 * (f[:, 0] - f[:, 1])
        b = 200 * (f[:, 1] - f[:, 2])
        return np.column_stack([L, a, b])

    _lab1 = _xyz_to_lab(_xyz1)
    _lab2 = _xyz_to_lab(_xyz2)
    _de = np.sqrt(np.sum((_lab1 - _lab2) ** 2, axis=1))
    # Add noise to simulate human variability
    _dv = _de * (1.0 + _rng.normal(0, 0.15, n_percept))
    _dv = np.clip(_dv, 0.01, None)

    COMBVD_XYZ1 = torch.tensor(_xyz1, device=dev, dtype=torch.float64)
    COMBVD_XYZ2 = torch.tensor(_xyz2, device=dev, dtype=torch.float64)
    COMBVD_DV = torch.tensor(_dv, device=dev, dtype=torch.float64)
    print("Generated %d synthetic perceptual pairs" % n_percept, flush=True)

N_PERCEPT = COMBVD_XYZ1.shape[0]

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
#  v7b FORWARD / INVERSE (P-batched)
# ================================================================

def fwd_v7b(xyz, d):
    """Forward: XYZ -> Lab. xyz: (N, 3), d has P-batched params.
    Returns (P, N, 3)."""
    M1, M2 = d["M1"], d["M2"]
    c1, c2, c3 = d["c1"], d["c2"], d["c3"]

    lms = (xyz.unsqueeze(0) @ M1.transpose(-1, -2)).clamp(min=0)
    lms_c = lms.pow(1.0 / 3.0)
    lab = torch.bmm(lms_c, M2.transpose(-1, -2))

    # L correction
    L = lab[..., 0]
    t = L * (1.0 - L)
    L_out = L + c1.view(-1, 1) * t + c2.view(-1, 1) * t * (2.0 * L - 1.0) + c3.view(-1, 1) * L * L * (1.0 - L) * (1.0 - L)

    return torch.stack([L_out, lab[..., 1], lab[..., 2]], dim=-1)


def inv_v7b(lab, d):
    """Inverse: Lab -> XYZ. lab: (P, N, 3)."""
    M1_i, M2_i = d["M1_i"], d["M2_i"]
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
    lms_c = torch.bmm(raw, M2_i.transpose(-1, -2))
    lms = lms_c.pow(3.0)
    return torch.bmm(lms, M1_i.transpose(-1, -2))


# ================================================================
#  PARAMETER PACKING / UNPACKING
# ================================================================

def unpack_params(x):
    """x: (P, 18) -> dict of batched params, valid mask."""
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

    # M2: 9 params -> 3x3
    M2 = torch.zeros(P, 3, 3, device=dev)
    M2[:, 0, 0] = x[:, 6];  M2[:, 0, 1] = x[:, 7];  M2[:, 0, 2] = x[:, 8]
    M2[:, 1, 0] = x[:, 9];  M2[:, 1, 1] = x[:, 10]; M2[:, 1, 2] = x[:, 11]
    M2[:, 2, 0] = x[:, 12]; M2[:, 2, 1] = x[:, 13]; M2[:, 2, 2] = x[:, 14]

    # L_corr: 3 params
    c1 = x[:, 15].clamp(-0.5, 0.5)
    c2 = x[:, 16].clamp(-0.5, 0.5)
    c3 = x[:, 17].clamp(-1.0, 1.0)

    # Compute inverses
    M1_i = torch.zeros_like(M1)
    M2_i = torch.zeros_like(M2)

    det1 = torch.linalg.det(M1)
    det2 = torch.linalg.det(M2)
    invertible = (det1.abs() > 1e-10) & (det2.abs() > 1e-10)
    valid &= invertible

    good = valid.nonzero(as_tuple=True)[0]
    if good.numel() > 0:
        M1_i[good] = torch.linalg.inv(M1[good])
        M2_i[good] = torch.linalg.inv(M2[good])

    # Achromatic constraint check: cbrt of (M1 @ D65) through M2 should give a~0, b~0
    ach_lms = lms_d65.clamp(min=1e-10).pow(1.0 / 3.0)  # (P, 3)
    ach_lab = torch.bmm(ach_lms.unsqueeze(1), M2.transpose(-1, -2)).squeeze(1)  # (P, 3)
    ach_ab = torch.sqrt(ach_lab[:, 1] ** 2 + ach_lab[:, 2] ** 2)

    return {
        "M1": M1, "M2": M2, "M1_i": M1_i, "M2_i": M2_i,
        "c1": c1, "c2": c2, "c3": c3,
        "ach_ab": ach_ab,
    }, valid


# ================================================================
#  BATCH PERCEPTUAL METRICS
# ================================================================

def batch_stress(d):
    """STRESS between human DV and Euclidean DE in our Lab space.
    Returns (P,)."""
    P = d["M1"].shape[0]

    # Forward both color samples
    lab1 = fwd_v7b(COMBVD_XYZ1, d)  # (P, N_percept, 3)
    lab2 = fwd_v7b(COMBVD_XYZ2, d)  # (P, N_percept, 3)

    # Euclidean distance in Lab
    dlab = lab2 - lab1
    DE = torch.sqrt((dlab ** 2).sum(dim=-1) + 1e-30)  # (P, N_percept)

    # STRESS: 100 * sqrt(sum((DV - F*DE)^2) / sum(DV^2))
    # where F = sum(DV*DE) / sum(DE^2)
    DV = COMBVD_DV.unsqueeze(0).expand(P, -1)  # (P, N_percept)

    DE2_sum = (DE ** 2).sum(dim=1).clamp(min=1e-30)  # (P,)
    DV_DE_sum = (DV * DE).sum(dim=1)  # (P,)
    F = DV_DE_sum / DE2_sum  # (P,)

    residual = DV - F.unsqueeze(1) * DE  # (P, N_percept)
    DV2_sum = (DV ** 2).sum(dim=1).clamp(min=1e-30)  # (P,)
    stress = 100.0 * torch.sqrt((residual ** 2).sum(dim=1) / DV2_sum)  # (P,)

    return stress


def batch_cv(d):
    """Gradient CV. Returns (P,)."""
    P = d["M1"].shape[0]
    lab1 = fwd_v7b(PT[:, 0], d)
    lab2 = fwd_v7b(PT[:, 1], d)

    t = T_ST.view(1, 1, -1, 1)
    labs = lab1.unsqueeze(2) + t * (lab2 - lab1).unsqueeze(2)
    lf = labs.reshape(P, -1, 3)

    xyz = inv_v7b(lf, d)
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
    lab = fwd_v7b(MUNSELL_GRAYS, d)
    L = lab[:, :, 0]
    dL = L[:, 1:] - L[:, :-1]
    mean_dL = dL.abs().mean(dim=1).clamp(min=1e-10)
    std_dL = dL.std(dim=1)
    return std_dL / mean_dL * 100.0


def batch_munsell_hue(d):
    lab = fwd_v7b(MUNSELL_HUE_XYZ, d)
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

    lab_prim = fwd_v7b(PRIM_XYZ, d)
    lab_white = fwd_v7b(WHITE_XYZ.expand(6, 3), d)

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
    lab = fwd_v7b(grays_xyz, d)
    L = lab[:, :, 0]
    dL = L[:, 1:] - L[:, :-1]
    return (dL > 0).float().mean(dim=1)


def batch_ach(d):
    n_gray = 32
    t = torch.linspace(0.01, 0.99, n_gray, device=dev)
    grays = D65.unsqueeze(0) * t.unsqueeze(1)
    lab = fwd_v7b(grays, d)
    return torch.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2).max(dim=1).values


def batch_info(d):
    P = d["M1"].shape[0]
    lab_p = fwd_v7b(PRIM_XYZ, d)
    lab_w = fwd_v7b(D65.unsqueeze(0), d)

    L_p = lab_p[:, :, 0]
    yC = torch.sqrt(lab_p[:, 3, 1] ** 2 + lab_p[:, 3, 2] ** 2)
    bL = lab_p[:, 2, 0]

    lab_mid_bw = 0.5 * (lab_p[:, 2:3, :] + lab_w)
    xyz_mid = inv_v7b(lab_mid_bw, d)
    srgb_mid = l2s((xyz_mid @ MSi.T).clamp(0, 1))
    bw_gr = srgb_mid[:, 0, 1] / srgb_mid[:, 0, 0].clamp(min=1e-10)

    lab_mid_rw = 0.5 * (lab_p[:, 0:1, :] + lab_w)
    xyz_mid_r = inv_v7b(lab_mid_rw, d)
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

        c1_cond = torch.linalg.cond(d["M1"])
        c2_cond = torch.linalg.cond(d["M2"])
        valid &= (c1_cond < 20) & (c2_cond < 30)

        if not valid.any():
            return losses.cpu().numpy()

        info = batch_info(d)
        valid &= info["yC"] > 0.03

        if not valid.any():
            return losses.cpu().numpy()

        # Core metrics
        stress = batch_stress(d)
        cv = batch_cv(d)
        munsell_v = batch_munsell_value(d)
        munsell_h = batch_munsell_hue(d)
        hue_lin = batch_hue_linearity(d)
        mono = batch_lightness_mono(d)
        ach = batch_ach(d)

        # Main loss: weighted combination
        # STRESS: lower is better (OKLab ~27.5, target <25)
        # Munsell Value CV: lower is better (OKLab ~2.6%)
        # Gradient CV: lower is better (fraction, OKLab ~0.25)
        loss = (args.stress_weight * stress
                + args.munsell_weight * munsell_v
                + args.cv_weight * cv * 100.0)  # scale CV to percentage

        # Hue linearity penalty (regularizer)
        loss += 0.5 * hue_lin

        # Monotonicity penalty
        loss += (1.0 - mono).clamp(min=0) * 500.0

        # Munsell hue penalty
        loss += 0.1 * munsell_h

        # Soft penalties
        pen = torch.zeros(P, device=dev)
        pen += torch.where(info["yC"] < 0.12, (0.12 - info["yC"]) ** 2 * 200, torch.zeros(1, device=dev))
        pen += torch.where(info["bL"] > 0.60, (info["bL"] - 0.60) ** 2 * 50, torch.zeros(1, device=dev))
        pen += torch.where(info["bw"] < 1.20, (1.20 - info["bw"]) ** 2 * 50, torch.zeros(1, device=dev))
        pen += torch.where(info["rw"] > 0.08, (info["rw"] - 0.08) ** 2 * 100, torch.zeros(1, device=dev))
        pen += torch.where(info["plr"] < 0.40, (0.40 - info["plr"]) ** 2 * 500, torch.zeros(1, device=dev))

        # Achromatic: cbrt pipeline has near-structural guarantee, strict penalty
        pen += torch.where(ach > 0.0001, (ach - 0.0001) * 500, torch.zeros(1, device=dev))
        pen += torch.where(ach > 0.001, (ach - 0.001) * 2000, torch.zeros(1, device=dev))

        # Structural achromatic: M2 should map achromatic to a=b=0
        pen += d["ach_ab"] ** 2 * 1000

        # Condition number soft penalty
        pen += torch.where(c1_cond > 8, (c1_cond - 8) ** 2 * 3, torch.zeros(1, device=dev))
        pen += torch.where(c2_cond > 12, (c2_cond - 12) ** 2 * 3, torch.zeros(1, device=dev))

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


def _pack_m2(M2_np):
    return M2_np.ravel()


def make_seeds():
    seeds = []
    rng = np.random.RandomState(42)

    # Seed 0: OKLab direct (no L_corr)
    x0 = np.zeros(N_PARAMS)
    x0[:6] = _pack_m1_d65(OKLAB_M1)
    x0[6:15] = _pack_m2(OKLAB_M2)
    x0[15:18] = [0.0, 0.0, 0.0]
    seeds.append(("oklab_base", x0.copy(), args.sigma))

    # Seed 1: OKLab + L_corr from v7b
    x1 = x0.copy()
    x1[15:18] = [-0.098, 0.133, 0.304]  # v7b L_corr
    seeds.append(("oklab_v7b_lcorr", x1, args.sigma))

    # Seed 2: OKLab + small random perturbation
    x2 = x0.copy() + rng.randn(N_PARAMS) * 0.02
    seeds.append(("oklab_pert1", x2, args.sigma))

    # Seed 3: OKLab emphasizing perception (larger sigma to explore)
    x3 = x0.copy() + rng.randn(N_PARAMS) * 0.03
    x3[15:18] = [-0.05, 0.05, 0.15]
    seeds.append(("oklab_percept", x3, 0.05))

    # Seed 4-5: wider exploration
    for i in range(max(0, args.seeds - 4)):
        xr = x0.copy() + rng.randn(N_PARAMS) * 0.05
        seeds.append(("rnd%d" % i, xr, 0.05))

    return seeds[:args.seeds]


# ================================================================
#  CHECKPOINT SAVING
# ================================================================

def save_checkpoint(best_x, loss, seed_label, gen, stress_val, cv_val, munsell_v_val):
    x = torch.tensor(best_x.reshape(1, -1), device=dev)
    d, v = unpack_params(x)
    if not v.any():
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = "perceptual_%s_%s.json" % (seed_label, ts)
    path = os.path.join(CKPT, fname)

    out = {
        "architecture": "v7b_PerceptionFirst",
        "M1": d["M1"][0].cpu().numpy().tolist(),
        "M2": d["M2"][0].cpu().numpy().tolist(),
        "M1_inv": np.linalg.inv(d["M1"][0].cpu().numpy()).tolist(),
        "M2_inv": np.linalg.inv(d["M2"][0].cpu().numpy()).tolist(),
        "L_corr": [d["c1"][0].item(), d["c2"][0].item(), d["c3"][0].item()],
        "loss": float(loss),
        "stress": float(stress_val),
        "gradient_cv": float(cv_val),
        "munsell_value_cv": float(munsell_v_val),
        "has_combvd": _has_combvd,
        "n_percept_pairs": N_PERCEPT,
        "loss_weights": {
            "stress": args.stress_weight,
            "munsell_cv": args.munsell_weight,
            "gradient_cv": args.cv_weight,
        },
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
    print("  Loss = %.1f*STRESS + %.1f*MunsellCV + %.1f*GradientCV"
          % (args.stress_weight, args.munsell_weight, args.cv_weight))
    print("  Perceptual data: %s (%d pairs)"
          % ("COMBVD" if _has_combvd else "synthetic", N_PERCEPT))
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
                stress_val = batch_stress(d_test).item()
                cv_val = batch_cv(d_test).item()
                mv_val = batch_munsell_value(d_test).item()
                mh_val = batch_munsell_hue(d_test).item()
                hl_val = batch_hue_linearity(d_test).item()
                mono_val = batch_lightness_mono(d_test).item()
                ach_val = batch_ach(d_test).item()
                print("  gen %4d  loss=%.4f  STRESS=%.2f  CV=%.1f%%  MunsV=%.1f%%  "
                      "MunsH=%.1f%%  HueLin=%.1f deg  Mono=%.4f  Ach=%.6f"
                      % (gen, best_loss, stress_val, cv_val * 100, mv_val,
                         mh_val, hl_val, mono_val, ach_val), flush=True)

    # Final metrics
    x_final = torch.tensor(best_x.reshape(1, -1), device=dev)
    d_final, v = unpack_params(x_final)
    stress_val = cv_val = mv_val = 0.0
    if v.any():
        stress_val = batch_stress(d_final).item()
        cv_val = batch_cv(d_final).item()
        mv_val = batch_munsell_value(d_final).item()
        mh_val = batch_munsell_hue(d_final).item()
        hl_val = batch_hue_linearity(d_final).item()
        mono_val = batch_lightness_mono(d_final).item()
        ach_val = batch_ach(d_final).item()
        info = batch_info(d_final)
        print("\n  FINAL: loss=%.4f" % best_loss)
        print("    STRESS=%.2f  CV=%.2f%%  MunsV=%.2f%%  MunsH=%.2f%%"
              % (stress_val, cv_val * 100, mv_val, mh_val))
        print("    HueLin=%.2f deg  Mono=%.4f  Ach=%.6f" % (hl_val, mono_val, ach_val))
        print("    YellowC=%.3f  BlueL=%.3f" % (info["yC"].item(), info["bL"].item()))
        print("    B-W G/R=%.3f  R-W G-B=%.3f" % (info["bw"].item(), info["rw"].item()))

    # Save best
    ckpt_path = save_checkpoint(best_x, best_loss, seed_label, gen, stress_val, cv_val, mv_val)

    return best_loss, ckpt_path


# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Perception-First Training: v7b pipeline + COMBVD loss")
    print("  Params=%d, Gens=%d, Pop=%d, Seeds=%d" % (N_PARAMS, args.gens, args.pop, args.seeds))
    print("  Loss = %.1f*STRESS + %.1f*MunsellCV + %.1f*GradientCV"
          % (args.stress_weight, args.munsell_weight, args.cv_weight))
    print("  Perceptual data: %s (%d pairs)"
          % ("COMBVD" if _has_combvd else "synthetic fallback", N_PERCEPT))
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
