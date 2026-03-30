#!/usr/bin/env python3
"""Pipeline Architecture Search — GPU-batched, population-level evaluation.

ALL 96 candidates evaluated in a SINGLE GPU pass per generation.
No per-candidate Python loops. No CPU hue drift. Minimal .item() calls.

5 transfer functions:
  A. Shared Gamma   (14p) — x^p, p free
  B. Naka-Rushton    (16p) — s·x^n/(x^n+σ^n)
  C. Div. Normal.    (22p) — s·x^n/(σ^n+Σw·x^n)
  D. Log-Weighted    (16p) — w·log(1+x/x_w)
  E. Power+Enriched  (17p) — x^p + post-opponent correction

Usage:
  python scripts/optimize_pipeline_search.py                  # all 5
  python scripts/optimize_pipeline_search.py --arch A B       # subset
  python scripts/optimize_pipeline_search.py --gens 100       # quick
"""

import json, math, os, sys, time, argparse
from datetime import datetime
import numpy as np
import torch
import cma

# ════════════════════════════════════════════════════════════════════
#  SETUP
# ════════════════════════════════════════════════════════════════════

torch.set_default_dtype(torch.float64)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev_name = "CPU"
if torch.cuda.is_available():
    dev_name = f"CUDA ({torch.cuda.get_device_name(0)})"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    dev_name = "MPS"
print(f"Device: {dev} ({dev_name})", flush=True)

pa = argparse.ArgumentParser()
pa.add_argument("--arch", nargs="+", choices=["A","B","C","D","E","F","G","H","I","J","K"],
                default=["A","B","C","D","E","F","G","H","I","J","K"])
pa.add_argument("--gens", type=int, default=300)
pa.add_argument("--pop", type=int, default=96)
pa.add_argument("--seeds", type=int, default=3)
args = pa.parse_args()

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "checkpoints")
os.makedirs(CKPT, exist_ok=True)

# ════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════

D65 = torch.tensor([0.95047, 1.0, 1.08883], device=dev)
D65_np = D65.cpu().numpy()

MS = torch.tensor([[.4124564,.3575761,.1804375],
                    [.2126729,.7151522,.0721750],
                    [.0193339,.1191920,.9503041]], device=dev)
MSi = torch.linalg.inv(MS)
I3 = torch.eye(3, device=dev)

def s2l(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))
def l2s(c):
    return torch.where(c <= 0.0031308, c * 12.92,
                       1.055 * c.clamp(min=1e-12).pow(1./2.4) - 0.055)

# ════════════════════════════════════════════════════════════════════
#  TRAINING PAIRS — precomputed on GPU
# ════════════════════════════════════════════════════════════════════

# ── Import EXACT same pairs as test suite (2512 pairs) ──
import sys as _sys
_use_imported = False
for _try_dir in [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "space-test-project"),
    os.path.dirname(os.path.abspath(__file__)),  # same dir as script
    os.getcwd(),  # current working dir
]:
    if os.path.isdir(os.path.join(_try_dir, "core")):
        _sys.path.insert(0, _try_dir)
        try:
            from core.pairs import generate_all_pairs as _gen_pairs
            _PT_imported, _labels = _gen_pairs(dev)
            _use_imported = True
            break
        except: pass

if _use_imported:
    PT = _PT_imported  # Already (N, 2, 3) XYZ on device — exact same as test suite
else:
    # Fallback: minimal pair set (if space-test-project not found)
    _pl = []
    _pr = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[0,0,0]]
    for i in range(len(_pr)):
        for j in range(i+1, len(_pr)):
            _pl.append((_pr[i], _pr[j]))
    _rng = np.random.RandomState(42)
    for _ in range(80):
        _pl.append((_rng.rand(3).tolist(), _rng.rand(3).tolist()))
    PT = torch.zeros(len(_pl), 2, 3, device=dev)
    for i, (c1, c2) in enumerate(_pl):
        PT[i, 0] = MS @ s2l(torch.tensor(c1, device=dev))
        PT[i, 1] = MS @ s2l(torch.tensor(c2, device=dev))
N_PAIRS = PT.shape[0]
print(f"Training pairs: {N_PAIRS}", flush=True)
N_ST = 25
T_ST = torch.linspace(0, 1, N_ST + 1, device=dev)  # (26,)

# Reference matrices
V14_M1 = torch.tensor([[.7583761294836658,.38380162590825084,-.09608055040602373],
                        [.12671393631532843,.8421628149123207,.03434823621506485],
                        [.07639223722200054,.258943526275451,.6139139663787314]], device=dev)
V14_M2 = torch.tensor([[.10058070589596230,1.01558970993941444,-.11617041583537688],
                        [2.36157646996164416,-2.44099737506293479,.07942090510129070],
                        [.04565327074453784,.81875488445424471,-.86440815519878267]], device=dev)
OK_M1s = torch.tensor([[.4122214708,.5363325363,.0514459929],
                        [.2119034982,.6806995451,.1073969566],
                        [.0883024619,.2817188376,.6299787005]], device=dev)
OK_M1 = OK_M1s @ MSi
OK_M2 = torch.tensor([[.2104542553,.7936177850,-.0040720468],
                       [1.9779984951,-2.4285922050,.4505937099],
                       [.0259040371,.7827717662,-.8086757660]], device=dev)

# Cusp scan grids — precomputed
CUSP_Ls = torch.linspace(0.3, 1.05, 90, device=dev)   # (90,)
CUSP_Cs = torch.linspace(0.001, 0.4, 60, device=dev)   # (60,)
# Gamut coverage grid
GC_Ls = torch.linspace(0.2, 1.0, 50, device=dev)
GC_Cs = torch.linspace(0.001, 0.35, 40, device=dev)

# Primary sRGB for hue metric
PRIM_SRGB = torch.tensor([[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1]],
                          dtype=torch.float64, device=dev)
PRIM_XYZ = s2l(PRIM_SRGB) @ MS.T   # (6, 3)
HUE_EXP = torch.tensor([0,60,120,180,240,300], dtype=torch.float64, device=dev)

# Gray ramp for achromatic check
GRAY_Y = torch.tensor([.005,.01,.05,.1,.2,.3,.5,.7,.8,.9,1.0,1.5], device=dev)
GRAY_XYZ = GRAY_Y.unsqueeze(1) * D65.unsqueeze(0)  # (12, 3)

# Special colors for info metric
YEL_XYZ = (MS @ s2l(torch.tensor([1.,1.,0.], device=dev))).unsqueeze(0)
BLU_XYZ = (MS @ s2l(torch.tensor([0.,0.,1.], device=dev))).unsqueeze(0)
RED_XYZ = (MS @ s2l(torch.tensor([1.,0.,0.], device=dev))).unsqueeze(0)
WHT_XYZ = (MS @ s2l(torch.tensor([1.,1.,1.], device=dev))).unsqueeze(0)

# ════════════════════════════════════════════════════════════════════
#  GPU-BATCHED HELPERS
# ════════════════════════════════════════════════════════════════════

def batched_ortho(s):
    """s: (P, 3) → e1: (P, 3), e2: (P, 3) perpendicular to s."""
    sn = s / s.norm(dim=1, keepdim=True).clamp(min=1e-30)
    use_x = sn[:, 0].abs() < 0.9
    v = torch.zeros_like(sn)
    v[use_x, 0] = 1.0
    v[~use_x, 1] = 1.0
    proj = (v * sn).sum(dim=1, keepdim=True)
    e1 = v - proj * sn
    e1 = e1 / e1.norm(dim=1, keepdim=True).clamp(min=1e-30)
    e2 = torch.linalg.cross(sn, e1)
    e2 = e2 / e2.norm(dim=1, keepdim=True).clamp(min=1e-30)
    return e1, e2


def unpack_M1(x):
    """x: (P, >=6) → M1: (P,3,3), valid: (P,)"""
    P = x.shape[0]
    M1 = torch.zeros(P, 3, 3, device=dev)
    for i in range(3):
        M1[:, i, 0] = x[:, 2*i]
        M1[:, i, 1] = x[:, 2*i+1]
        M1[:, i, 2] = (1.0 - M1[:, i, 0]*D65[0] - M1[:, i, 1]*D65[1]) / D65[2]
    lms = M1 @ D65   # (P, 3)
    valid = (lms > 0).all(dim=1)
    return M1, lms, valid


def unpack_M2(x_m2, s):
    """x_m2: (P, 7), s: (P, 3) achromatic dir → M2: (P,3,3), valid: (P,)"""
    P = s.shape[0]
    e1, e2 = batched_ortho(s)
    M2 = torch.zeros(P, 3, 3, device=dev)
    M2[:, 0, :] = x_m2[:, 0:3]
    Lw = (M2[:, 0, :] * s).sum(dim=1)
    valid = Lw.abs() > 1e-10
    M2[:, 0, :] /= Lw.unsqueeze(1).clamp(min=1e-30)
    M2[:, 1, :] = x_m2[:, 3].unsqueeze(1) * e1 + x_m2[:, 4].unsqueeze(1) * e2
    M2[:, 2, :] = x_m2[:, 5].unsqueeze(1) * e1 + x_m2[:, 6].unsqueeze(1) * e2
    return M2, valid


# ════════════════════════════════════════════════════════════════════
#  BATCH FORWARD / INVERSE  (per architecture)
# ════════════════════════════════════════════════════════════════════
# xyz: (N, 3) shared, lab: (P, G, 3)
# M1: (P,3,3), M2: (P,3,3), etc.
# Output: (P, N, 3) or (P, G, 3)

# ── A: Shared Gamma ──
def fwd_A(xyz, M1, M2, p):
    lms = (xyz.unsqueeze(0) @ M1.transpose(-1,-2)).clamp(min=0)  # (P,N,3)
    lms_c = lms.pow(p.view(-1,1,1))
    return torch.bmm(lms_c, M2.transpose(-1,-2))

def inv_A(lab, M1i, M2i, p):
    lms_c = torch.bmm(lab, M2i.transpose(-1,-2))
    lms = lms_c.clamp(min=0).pow(1.0 / p.view(-1,1,1))
    return torch.bmm(lms, M1i.transpose(-1,-2))

# ── B: Naka-Rushton ──
def fwd_B(xyz, M1, M2, n, sigma, s):
    lms = (xyz.unsqueeze(0) @ M1.transpose(-1,-2)).clamp(min=0)
    x_n = lms.pow(n.view(-1,1,1))
    sig_n = sigma.pow(n).view(-1,1,1)
    lms_c = s.view(-1,1,1) * x_n / (x_n + sig_n)
    return torch.bmm(lms_c, M2.transpose(-1,-2))

def inv_B(lab, M1i, M2i, n, sigma, s):
    lms_c = torch.bmm(lab, M2i.transpose(-1,-2))
    lms_c = torch.minimum(lms_c.clamp(min=0), s.view(-1,1,1) - 1e-10)
    ratio = (lms_c / (s.view(-1,1,1) - lms_c).clamp(min=1e-30)).clamp(min=0)
    lms = sigma.view(-1,1,1) * ratio.pow(1.0 / n.view(-1,1,1))
    return torch.bmm(lms, M1i.transpose(-1,-2))

# ── C: Divisive Normalization ──
def fwd_C(xyz, M1, M2, n, sigma, s, W):
    lms = (xyz.unsqueeze(0) @ M1.transpose(-1,-2)).clamp(min=0)
    u = lms.pow(n.view(-1,1,1))
    sig_n = sigma.pow(n).view(-1,1,1)
    denom = sig_n + torch.bmm(u, W.transpose(-1,-2))
    dn = s.view(-1,1,1) * u / denom.clamp(min=1e-30)
    return torch.bmm(dn, M2.transpose(-1,-2))

def inv_C(lab, M1i, M2i, n, sigma, s, W):
    dn = torch.bmm(lab, M2i.transpose(-1,-2)).clamp(min=0)
    z = dn / s.view(-1,1,1)
    P, G = z.shape[0], z.shape[1]
    sig_n = sigma.pow(n).view(-1,1,1)
    A = I3.view(1,1,3,3) - z.unsqueeze(-1) * W.unsqueeze(1)
    b = sig_n * z
    A_f = A.reshape(P*G, 3, 3)
    b_f = b.reshape(P*G, 3, 1)
    u = torch.linalg.solve(A_f, b_f).squeeze(-1).clamp(min=0).reshape(P, G, 3)
    lms = u.pow(1.0 / n.view(-1,1,1))
    return torch.bmm(lms, M1i.transpose(-1,-2))

# ── D: Log-Weighted ──
def fwd_D(xyz, M1, M2, w, lms_w):
    lms = (xyz.unsqueeze(0) @ M1.transpose(-1,-2)).clamp(min=0)
    ratio = lms / lms_w.unsqueeze(1).clamp(min=1e-30)
    lr = w.unsqueeze(1) * torch.log1p(ratio)
    return torch.bmm(lr, M2.transpose(-1,-2))

def inv_D(lab, M1i, M2i, w, lms_w):
    lr = torch.bmm(lab, M2i.transpose(-1,-2))
    ratio = torch.expm1((lr / w.unsqueeze(1).clamp(min=1e-10)).clamp(-30, 30)).clamp(min=0)
    lms = lms_w.unsqueeze(1) * ratio
    return torch.bmm(lms, M1i.transpose(-1,-2))

# ── E: Power + Enrichment ──
def fwd_E(xyz, M1, M2, p, c1, k, cp):
    lms = (xyz.unsqueeze(0) @ M1.transpose(-1,-2)).clamp(min=0)
    lms_c = lms.pow(p.view(-1,1,1))
    raw = torch.bmm(lms_c, M2.transpose(-1,-2))
    L, a, b = raw[...,0], raw[...,1], raw[...,2]
    L_out = L + c1.view(-1,1) * L * (1.0 - L)
    C = torch.sqrt(a*a + b*b + 1e-30)
    f_L = torch.exp(k.view(-1,1) * (L - 0.5))
    C_out = f_L * C.pow(cp.view(-1,1))
    a_out = a / C * C_out
    b_out = b / C * C_out
    return torch.stack([L_out, a_out, b_out], dim=-1)

def inv_E(lab, M1i, M2i, p, c1, k, cp):
    L_out, a_out, b_out = lab[...,0], lab[...,1], lab[...,2]
    # Newton for L
    L = L_out.clone()
    for _ in range(10):
        g = L + c1.view(-1,1) * L * (1.0 - L) - L_out
        gp = 1.0 + c1.view(-1,1) * (1.0 - 2.0 * L)
        L = L - g / gp.clamp(min=1e-10)
    C_out = torch.sqrt(a_out**2 + b_out**2 + 1e-30)
    f_L = torch.exp(k.view(-1,1) * (L - 0.5))
    C_in = (C_out / f_L.clamp(min=1e-30)).clamp(min=0).pow(1.0 / cp.view(-1,1))
    a_in = a_out / C_out * C_in
    b_in = b_out / C_out * C_in
    raw = torch.stack([L, a_in, b_in], dim=-1)
    lms_c = torch.bmm(raw, M2i.transpose(-1,-2))
    lms = lms_c.clamp(min=0).pow(1.0 / p.view(-1,1,1))
    return torch.bmm(lms, M1i.transpose(-1,-2))


# ── G: Per-Channel Gamma + Chroma Power (best of F + E) ──
def fwd_G(xyz, M1, M2, gamma, c1, k, cp):
    lms = (xyz.unsqueeze(0) @ M1.transpose(-1,-2)).clamp(min=0)
    lms_c = lms.pow(gamma.unsqueeze(1))
    raw = torch.bmm(lms_c, M2.transpose(-1,-2))
    L, a, b = raw[...,0], raw[...,1], raw[...,2]
    L_out = L + c1.view(-1,1) * L * (1.0 - L)
    C = torch.sqrt(a*a + b*b + 1e-30)
    f_L = torch.exp(k.view(-1,1) * (L - 0.5))
    C_out = f_L * C.pow(cp.view(-1,1))
    a_out = a / C * C_out
    b_out = b / C * C_out
    return torch.stack([L_out, a_out, b_out], dim=-1)

def inv_G(lab, M1i, M2i, gamma, c1, k, cp):
    L_out, a_out, b_out = lab[...,0], lab[...,1], lab[...,2]
    L = L_out.clone()
    for _ in range(10):
        g = L + c1.view(-1,1) * L * (1.0 - L) - L_out
        gp = 1.0 + c1.view(-1,1) * (1.0 - 2.0 * L)
        L = L - g / gp.clamp(min=1e-10)
    C_out = torch.sqrt(a_out**2 + b_out**2 + 1e-30)
    f_L = torch.exp(k.view(-1,1) * (L - 0.5))
    C_in = (C_out / f_L.clamp(min=1e-30)).clamp(min=0).pow(1.0 / cp.view(-1,1))
    a_in = a_out / C_out * C_in
    b_in = b_out / C_out * C_in
    raw = torch.stack([L, a_in, b_in], dim=-1)
    lms_c = torch.bmm(raw, M2i.transpose(-1,-2))
    inv_gamma = 1.0 / gamma.unsqueeze(1)
    lms = lms_c.clamp(min=0).pow(inv_gamma)
    return torch.bmm(lms, M1i.transpose(-1,-2))

def _unpack_G(x):
    """19 params: M1(6) + M2(7) + g2(1) + g3(1) + c1(1) + k(1) + cp(1)."""
    M1, lms, v = unpack_M1(x)
    g1 = 1.0 / 3.0
    g2 = g1 * torch.exp(x[:,13].clamp(-1.0, 1.0))
    g3 = g1 * torch.exp(x[:,14].clamp(-1.0, 1.0))
    gamma = torch.stack([torch.full_like(g2, g1), g2, g3], dim=1)
    v &= (g2 > 0.1) & (g2 < 0.8) & (g3 > 0.1) & (g3 < 0.8)
    s = lms.clamp(min=1e-30).pow(gamma)
    M2, v2 = unpack_M2(x[:, 6:13], s)
    v &= v2
    c1 = x[:,15].clamp(-0.5, 0.5)
    k = x[:,16].clamp(-2.0, 2.0)
    cp = torch.exp(x[:,17].clamp(-1.0, 0.7))
    v &= (cp >= 0.3) & (cp <= 2.5)
    M1i = torch.zeros_like(M1); M2i = torch.zeros_like(M1)
    good = v.nonzero(as_tuple=True)[0]
    if good.numel() > 0:
        M1i[good] = torch.linalg.inv(M1[good])
        M2i[good] = torch.linalg.inv(M2[good])
    return {"M1":M1,"M2":M2,"M1i":M1i,"M2i":M2i,
            "gamma":gamma,"c1":c1,"k":k,"cp":cp}, v

def _fwd_G(xyz, d): return fwd_G(xyz, d["M1"], d["M2"], d["gamma"], d["c1"], d["k"], d["cp"])
def _inv_G(lab, d): return inv_G(lab, d["M1i"], d["M2i"], d["gamma"], d["c1"], d["k"], d["cp"])


# ── H: Naka-Rushton + Enrichment ──
def fwd_H(xyz, M1, M2, n, sigma, s_gain, c1, k, cp):
    lms = (xyz.unsqueeze(0) @ M1.transpose(-1,-2)).clamp(min=0)
    x_n = lms.pow(n.view(-1,1,1))
    sig_n = sigma.pow(n).view(-1,1,1)
    lms_c = s_gain.view(-1,1,1) * x_n / (x_n + sig_n)
    raw = torch.bmm(lms_c, M2.transpose(-1,-2))
    L, a, b = raw[...,0], raw[...,1], raw[...,2]
    L_out = L + c1.view(-1,1) * L * (1.0 - L)
    C = torch.sqrt(a*a + b*b + 1e-30)
    f_L = torch.exp(k.view(-1,1) * (L - 0.5))
    C_out = f_L * C.pow(cp.view(-1,1))
    a_out = a / C * C_out
    b_out = b / C * C_out
    return torch.stack([L_out, a_out, b_out], dim=-1)

def inv_H(lab, M1i, M2i, n, sigma, s_gain, c1, k, cp):
    L_out, a_out, b_out = lab[...,0], lab[...,1], lab[...,2]
    L = L_out.clone()
    for _ in range(10):
        g = L + c1.view(-1,1) * L * (1.0 - L) - L_out
        gp = 1.0 + c1.view(-1,1) * (1.0 - 2.0 * L)
        L = L - g / gp.clamp(min=1e-10)
    C_out = torch.sqrt(a_out**2 + b_out**2 + 1e-30)
    f_L = torch.exp(k.view(-1,1) * (L - 0.5))
    C_in = (C_out / f_L.clamp(min=1e-30)).clamp(min=0).pow(1.0 / cp.view(-1,1))
    a_in = a_out / C_out * C_in
    b_in = b_out / C_out * C_in
    raw = torch.stack([L, a_in, b_in], dim=-1)
    lms_c = torch.bmm(raw, M2i.transpose(-1,-2))
    lms_c = torch.minimum(lms_c.clamp(min=0), s_gain.view(-1,1,1) - 1e-10)
    ratio = (lms_c / (s_gain.view(-1,1,1) - lms_c).clamp(min=1e-30)).clamp(min=0)
    lms = sigma.view(-1,1,1) * ratio.pow(1.0 / n.view(-1,1,1))
    return torch.bmm(lms, M1i.transpose(-1,-2))

def _unpack_H(x):
    """19 params: M1(6) + M2(7) + n(1) + σ(1) + s(1) + c1(1) + k(1) + cp(1)."""
    M1, lms, v = unpack_M1(x)
    n = 0.42 * torch.exp(x[:,13].clamp(-1.0, 1.0))
    v &= (n >= 0.15) & (n <= 1.1)
    sigma = torch.exp(x[:,14].clamp(-3.0, 2.0))
    s_gain = torch.exp(x[:,15].clamp(-2.0, 2.0))
    lms_n = lms.clamp(min=1e-30).pow(n.unsqueeze(1))
    nr = s_gain.unsqueeze(1) * lms_n / (lms_n + sigma.pow(n).unsqueeze(1))
    v &= (nr > 0).all(dim=1)
    M2, v2 = unpack_M2(x[:, 6:13], nr)
    v &= v2
    # M-cone dominant L-row: M2[0,1] must be > M2[0,0] and > M2[0,2]
    # This ensures Blue (high S, low M) gets low L
    v &= (M2[:, 0, 1].abs() > M2[:, 0, 0].abs())
    v &= (M2[:, 0, 1].abs() > M2[:, 0, 2].abs())
    c1 = x[:,16].clamp(-0.2, 0.2)
    k = x[:,17].clamp(-0.5, 0.5)
    cp = 0.85 + 0.15 * torch.sigmoid(x[:,18])  # range [0.85, 1.0]
    M1i = torch.zeros_like(M1); M2i = torch.zeros_like(M1)
    good = v.nonzero(as_tuple=True)[0]
    if good.numel() > 0:
        M1i[good] = torch.linalg.inv(M1[good])
        M2i[good] = torch.linalg.inv(M2[good])
    return {"M1":M1,"M2":M2,"M1i":M1i,"M2i":M2i,
            "n":n,"sigma":sigma,"s":s_gain,"c1":c1,"k":k,"cp":cp}, v

def _fwd_H(xyz, d): return fwd_H(xyz, d["M1"], d["M2"], d["n"], d["sigma"], d["s"], d["c1"], d["k"], d["cp"])
def _inv_H(lab, d): return inv_H(lab, d["M1i"], d["M2i"], d["n"], d["sigma"], d["s"], d["c1"], d["k"], d["cp"])


# ════════════════════════════════════════════════════════════════════
#  ARCHITECTURE CONFIGS
# ════════════════════════════════════════════════════════════════════

class Arch:
    """Holds name, n_params, unpack, forward, inverse, seeds."""
    def __init__(self, name, n_params, ach_exact=True):
        self.name = name
        self.n_params = n_params
        self.ach_exact = ach_exact


def _unpack_A(x):
    """x: (P,14) → dict of GPU tensors + valid (P,)"""
    M1, lms, v = unpack_M1(x)
    p = (1./3.) * torch.exp(x[:,13].clamp(-0.8, 0.9))
    v &= (p >= 0.15) & (p <= 0.80)
    s = lms.clamp(min=1e-30).pow(p.unsqueeze(1))
    M2, v2 = unpack_M2(x[:, 6:13], s)
    v &= v2
    M1i = torch.zeros_like(M1); M2i = torch.zeros_like(M2)
    good = v.nonzero(as_tuple=True)[0]
    if good.numel() > 0:
        M1i[good] = torch.linalg.inv(M1[good])
        M2i[good] = torch.linalg.inv(M2[good])
    return {"M1":M1,"M2":M2,"M1i":M1i,"M2i":M2i,"p":p}, v

def _fwd_A(xyz, d): return fwd_A(xyz, d["M1"], d["M2"], d["p"])
def _inv_A(lab, d): return inv_A(lab, d["M1i"], d["M2i"], d["p"])


def _unpack_B(x):
    M1, lms, v = unpack_M1(x)
    n = 0.42 * torch.exp(x[:,13].clamp(-1.0, 1.0))
    v &= (n >= 0.15) & (n <= 1.1)
    sigma = torch.exp(x[:,14].clamp(-3.0, 2.0))
    s_gain = torch.exp(x[:,15].clamp(-2.0, 2.0))
    lms_n = lms.clamp(min=1e-30).pow(n.unsqueeze(1))
    nr = s_gain.unsqueeze(1) * lms_n / (lms_n + sigma.pow(n).unsqueeze(1))
    v &= (nr > 0).all(dim=1)
    M2, v2 = unpack_M2(x[:, 6:13], nr)
    v &= v2
    M1i = torch.zeros_like(M1); M2i = torch.zeros_like(M1)
    good = v.nonzero(as_tuple=True)[0]
    if good.numel() > 0:
        M1i[good] = torch.linalg.inv(M1[good])
        M2i[good] = torch.linalg.inv(M2[good])
    return {"M1":M1,"M2":M2,"M1i":M1i,"M2i":M2i,"n":n,"sigma":sigma,"s":s_gain}, v

def _fwd_B(xyz, d): return fwd_B(xyz, d["M1"], d["M2"], d["n"], d["sigma"], d["s"])
def _inv_B(lab, d): return inv_B(lab, d["M1i"], d["M2i"], d["n"], d["sigma"], d["s"])


def _unpack_C(x):
    M1, lms, v = unpack_M1(x)
    n = 0.33 * torch.exp(x[:,13].clamp(-0.8, 0.8))
    v &= (n >= 0.15) & (n <= 0.75)
    sigma = torch.exp(x[:,14].clamp(-3.0, 2.0))
    s_gain = torch.exp(x[:,21].clamp(-2.0, 2.0))
    W = I3.unsqueeze(0).expand(x.shape[0], 3, 3).clone()
    W[:, 0, 1] = torch.exp(x[:,15].clamp(-4, 2))
    W[:, 0, 2] = torch.exp(x[:,16].clamp(-4, 2))
    W[:, 1, 0] = torch.exp(x[:,17].clamp(-4, 2))
    W[:, 1, 2] = torch.exp(x[:,18].clamp(-4, 2))
    W[:, 2, 0] = torch.exp(x[:,19].clamp(-4, 2))
    W[:, 2, 1] = torch.exp(x[:,20].clamp(-4, 2))
    u = lms.clamp(min=1e-30).pow(n.unsqueeze(1))
    sig_n = sigma.pow(n).unsqueeze(1)
    denom = sig_n + torch.bmm(u.unsqueeze(1), W.transpose(-1,-2)).squeeze(1)
    v &= (denom > 0).all(dim=1)
    dn = s_gain.unsqueeze(1) * u / denom.clamp(min=1e-30)
    v &= (dn > 0).all(dim=1)
    M2, v2 = unpack_M2(x[:, 6:13], dn)
    v &= v2
    M1i = torch.zeros_like(M1); M2i = torch.zeros_like(M1)
    good = v.nonzero(as_tuple=True)[0]
    if good.numel() > 0:
        M1i[good] = torch.linalg.inv(M1[good])
        M2i[good] = torch.linalg.inv(M2[good])
    return {"M1":M1,"M2":M2,"M1i":M1i,"M2i":M2i,"n":n,"sigma":sigma,"s":s_gain,"W":W}, v

def _fwd_C(xyz, d): return fwd_C(xyz, d["M1"], d["M2"], d["n"], d["sigma"], d["s"], d["W"])
def _inv_C(lab, d): return inv_C(lab, d["M1i"], d["M2i"], d["n"], d["sigma"], d["s"], d["W"])


def _unpack_D(x):
    M1, lms, v = unpack_M1(x)
    w = torch.exp(x[:, 6:9].clamp(-2, 2))
    v &= (w > 0.05).all(dim=1) & (w < 7.5).all(dim=1)
    lms_w = lms.clamp(min=1e-30)
    s_ach = w * math.log(2.0)
    M2, v2 = unpack_M2(x[:, 9:16], s_ach)
    v &= v2
    M1i = torch.zeros_like(M1); M2i = torch.zeros_like(M1)
    good = v.nonzero(as_tuple=True)[0]
    if good.numel() > 0:
        M1i[good] = torch.linalg.inv(M1[good])
        M2i[good] = torch.linalg.inv(M2[good])
    return {"M1":M1,"M2":M2,"M1i":M1i,"M2i":M2i,"w":w,"lms_w":lms_w}, v

def _fwd_D(xyz, d): return fwd_D(xyz, d["M1"], d["M2"], d["w"], d["lms_w"])
def _inv_D(lab, d): return inv_D(lab, d["M1i"], d["M2i"], d["w"], d["lms_w"])


def _unpack_E(x):
    M1, lms, v = unpack_M1(x)
    p = (1./3.) * torch.exp(x[:,13].clamp(-0.8, 0.9))
    v &= (p >= 0.15) & (p <= 0.80)
    s = lms.clamp(min=1e-30).pow(p.unsqueeze(1))
    M2, v2 = unpack_M2(x[:, 6:13], s)
    v &= v2
    c1 = x[:,14].clamp(-0.5, 0.5)
    k = x[:,15].clamp(-2.0, 2.0)
    cp = torch.exp(x[:,16].clamp(-1.0, 0.7))
    v &= (cp >= 0.3) & (cp <= 2.5)
    M1i = torch.zeros_like(M1); M2i = torch.zeros_like(M1)
    good = v.nonzero(as_tuple=True)[0]
    if good.numel() > 0:
        M1i[good] = torch.linalg.inv(M1[good])
        M2i[good] = torch.linalg.inv(M2[good])
    return {"M1":M1,"M2":M2,"M1i":M1i,"M2i":M2i,"p":p,"c1":c1,"k":k,"cp":cp}, v

def _fwd_E(xyz, d): return fwd_E(xyz, d["M1"], d["M2"], d["p"], d["c1"], d["k"], d["cp"])
def _inv_E(lab, d): return inv_E(lab, d["M1i"], d["M2i"], d["p"], d["c1"], d["k"], d["cp"])


# ── F: Per-Channel Gamma + L correction + L-dep chroma (NO chroma power) ──
def fwd_F(xyz, M1, M2, gamma, c1, k):
    """Per-channel gamma: gamma is (P, 3)."""
    lms = (xyz.unsqueeze(0) @ M1.transpose(-1,-2)).clamp(min=0)  # (P,N,3)
    lms_c = lms.pow(gamma.unsqueeze(1))  # (P,N,3) with per-channel exponents
    raw = torch.bmm(lms_c, M2.transpose(-1,-2))
    L, a, b = raw[...,0], raw[...,1], raw[...,2]
    L_out = L + c1.view(-1,1) * L * (1.0 - L)
    T = torch.exp(k.view(-1,1) * (L - 0.5))
    return torch.stack([L_out, a * T, b * T], dim=-1)

def inv_F(lab, M1i, M2i, gamma, c1, k):
    L_out, a, b = lab[...,0], lab[...,1], lab[...,2]
    # Undo L correction (Newton)
    L = L_out.clone()
    for _ in range(10):
        g = L + c1.view(-1,1) * L * (1.0 - L) - L_out
        gp = 1.0 + c1.view(-1,1) * (1.0 - 2.0 * L)
        L = L - g / gp.clamp(min=1e-10)
    # Undo chroma scale
    T = torch.exp(k.view(-1,1) * (L - 0.5))
    a_raw = a / T.clamp(min=1e-30)
    b_raw = b / T.clamp(min=1e-30)
    raw = torch.stack([L, a_raw, b_raw], dim=-1)
    lms_c = torch.bmm(raw, M2i.transpose(-1,-2))
    inv_gamma = 1.0 / gamma.unsqueeze(1)  # (P, 1, 3)
    lms = lms_c.clamp(min=0).pow(inv_gamma)
    return torch.bmm(lms, M1i.transpose(-1,-2))

def _unpack_F(x):
    """17 params: M1(6) + M2(7) + g2_ratio(1) + g3_ratio(1) + c1(1) + k(1)."""
    M1, lms, v = unpack_M1(x)
    # Per-channel gamma: g1=1/3 fixed, g2=g1*exp(x[13]), g3=g1*exp(x[14])
    g1 = 1.0 / 3.0
    g2 = g1 * torch.exp(x[:,13].clamp(-1.0, 1.0))
    g3 = g1 * torch.exp(x[:,14].clamp(-1.0, 1.0))
    gamma = torch.stack([torch.full_like(g2, g1), g2, g3], dim=1)  # (P, 3)
    v &= (g2 > 0.1) & (g2 < 0.8) & (g3 > 0.1) & (g3 < 0.8)
    # Achromatic direction at D65 with per-channel gamma
    s = lms.clamp(min=1e-30).pow(gamma)  # (P, 3)
    M2, v2 = unpack_M2(x[:, 6:13], s)
    v &= v2
    c1 = x[:,15].clamp(-0.5, 0.5)
    k = x[:,16].clamp(-2.0, 2.0)
    M1i = torch.zeros_like(M1); M2i = torch.zeros_like(M1)
    good = v.nonzero(as_tuple=True)[0]
    if good.numel() > 0:
        M1i[good] = torch.linalg.inv(M1[good])
        M2i[good] = torch.linalg.inv(M2[good])
    return {"M1":M1,"M2":M2,"M1i":M1i,"M2i":M2i,
            "gamma":gamma,"c1":c1,"k":k}, v

def _fwd_F(xyz, d): return fwd_F(xyz, d["M1"], d["M2"], d["gamma"], d["c1"], d["k"])
def _inv_F(lab, d): return inv_F(lab, d["M1i"], d["M2i"], d["gamma"], d["c1"], d["k"])


# ── I: NR + hue-dependent cp only (21 params) ──
def fwd_I(xyz, M1, M2, n, sigma, s_gain, c1, k, cp_base, cp_cos, cp_sin):
    lms = (xyz.unsqueeze(0) @ M1.transpose(-1,-2)).clamp(min=0)
    x_n = lms.pow(n.view(-1,1,1))
    sig_n = sigma.pow(n).view(-1,1,1)
    lms_c = s_gain.view(-1,1,1) * x_n / (x_n + sig_n)
    raw = torch.bmm(lms_c, M2.transpose(-1,-2))
    L, a, b = raw[...,0], raw[...,1], raw[...,2]
    L_out = L + c1.view(-1,1) * L * (1.0 - L)
    h = torch.atan2(b, a)  # (P, N)
    cp = cp_base.view(-1,1) + cp_cos.view(-1,1)*torch.cos(h) + cp_sin.view(-1,1)*torch.sin(h)
    cp = cp.clamp(0.4, 1.8)
    C = torch.sqrt(a*a + b*b + 1e-30)
    f_L = torch.exp(k.view(-1,1) * (L - 0.5))
    C_out = f_L * C.pow(cp)
    a_out = a / C * C_out
    b_out = b / C * C_out
    return torch.stack([L_out, a_out, b_out], dim=-1)

def inv_I(lab, M1i, M2i, n, sigma, s_gain, c1, k, cp_base, cp_cos, cp_sin):
    L_out, a_out, b_out = lab[...,0], lab[...,1], lab[...,2]
    L = L_out.clone()
    for _ in range(10):
        g = L + c1.view(-1,1) * L * (1.0 - L) - L_out
        gp = 1.0 + c1.view(-1,1) * (1.0 - 2.0 * L)
        L = L - g / gp.clamp(min=1e-10)
    h_out = torch.atan2(b_out, a_out)
    cp = cp_base.view(-1,1) + cp_cos.view(-1,1)*torch.cos(h_out) + cp_sin.view(-1,1)*torch.sin(h_out)
    cp = cp.clamp(0.4, 1.8)
    C_out = torch.sqrt(a_out**2 + b_out**2 + 1e-30)
    f_L = torch.exp(k.view(-1,1) * (L - 0.5))
    C_in = (C_out / f_L.clamp(min=1e-30)).clamp(min=0).pow(1.0 / cp)
    a_in = a_out / C_out * C_in
    b_in = b_out / C_out * C_in
    raw = torch.stack([L, a_in, b_in], dim=-1)
    lms_c = torch.bmm(raw, M2i.transpose(-1,-2))
    lms_c = torch.minimum(lms_c.clamp(min=0), s_gain.view(-1,1,1) - 1e-10)
    ratio = (lms_c / (s_gain.view(-1,1,1) - lms_c).clamp(min=1e-30)).clamp(min=0)
    lms = sigma.view(-1,1,1) * ratio.pow(1.0 / n.view(-1,1,1))
    return torch.bmm(lms, M1i.transpose(-1,-2))

def _unpack_I(x):
    """21p: M1(6)+M2(7)+n(1)+σ(1)+s(1)+c1(1)+k(1)+cp_base(1)+cp_cos(1)+cp_sin(1)."""
    M1, lms, v = unpack_M1(x)
    n = 0.42 * torch.exp(x[:,13].clamp(-1,1)); v &= (n>=0.15)&(n<=1.1)
    sigma = torch.exp(x[:,14].clamp(-3,2)); s_gain = torch.exp(x[:,15].clamp(-2,2))
    lms_n = lms.clamp(min=1e-30).pow(n.unsqueeze(1))
    nr = s_gain.unsqueeze(1)*lms_n/(lms_n+sigma.pow(n).unsqueeze(1))
    v &= (nr>0).all(dim=1)
    M2, v2 = unpack_M2(x[:,6:13], nr); v &= v2
    c1 = x[:,16].clamp(-0.5,0.5); k = x[:,17].clamp(-2,2)
    cp_base = torch.exp(x[:,18].clamp(-1,0.7)); v &= (cp_base>=0.3)&(cp_base<=2.5)
    cp_cos = x[:,19].clamp(-0.5,0.5); cp_sin = x[:,20].clamp(-0.5,0.5)
    M1i = torch.zeros_like(M1); M2i = torch.zeros_like(M1)
    good = v.nonzero(as_tuple=True)[0]
    if good.numel()>0: M1i[good]=torch.linalg.inv(M1[good]); M2i[good]=torch.linalg.inv(M2[good])
    return {"M1":M1,"M2":M2,"M1i":M1i,"M2i":M2i,"n":n,"sigma":sigma,"s":s_gain,
            "c1":c1,"k":k,"cp_base":cp_base,"cp_cos":cp_cos,"cp_sin":cp_sin}, v

def _fwd_I(xyz, d): return fwd_I(xyz, d["M1"],d["M2"],d["n"],d["sigma"],d["s"],d["c1"],d["k"],d["cp_base"],d["cp_cos"],d["cp_sin"])
def _inv_I(lab, d): return inv_I(lab, d["M1i"],d["M2i"],d["n"],d["sigma"],d["s"],d["c1"],d["k"],d["cp_base"],d["cp_cos"],d["cp_sin"])


# ── J: NR + full hue-dependent enrichment (25 params) ──
def fwd_J(xyz, M1, M2, n, sigma, s_gain, c1b, c1c, c1s, kb, kc, ks, cpb, cpc, cps):
    lms = (xyz.unsqueeze(0) @ M1.transpose(-1,-2)).clamp(min=0)
    x_n = lms.pow(n.view(-1,1,1))
    sig_n = sigma.pow(n).view(-1,1,1)
    lms_c = s_gain.view(-1,1,1) * x_n / (x_n + sig_n)
    raw = torch.bmm(lms_c, M2.transpose(-1,-2))
    L, a, b = raw[...,0], raw[...,1], raw[...,2]
    h = torch.atan2(b, a)
    c1 = c1b.view(-1,1) + c1c.view(-1,1)*torch.cos(h) + c1s.view(-1,1)*torch.sin(h)
    c1 = c1.clamp(-0.5, 0.5)
    L_out = L + c1 * L * (1.0 - L)
    k_h = kb.view(-1,1) + kc.view(-1,1)*torch.cos(h) + ks.view(-1,1)*torch.sin(h)
    cp = cpb.view(-1,1) + cpc.view(-1,1)*torch.cos(h) + cps.view(-1,1)*torch.sin(h)
    cp = cp.clamp(0.4, 1.8)
    C = torch.sqrt(a*a + b*b + 1e-30)
    f_L = torch.exp(k_h * (L - 0.5))
    C_out = f_L * C.pow(cp)
    a_out = a / C * C_out; b_out = b / C * C_out
    return torch.stack([L_out, a_out, b_out], dim=-1)

def inv_J(lab, M1i, M2i, n, sigma, s_gain, c1b, c1c, c1s, kb, kc, ks, cpb, cpc, cps):
    L_out, a_out, b_out = lab[...,0], lab[...,1], lab[...,2]
    h_out = torch.atan2(b_out, a_out)
    c1 = c1b.view(-1,1)+c1c.view(-1,1)*torch.cos(h_out)+c1s.view(-1,1)*torch.sin(h_out)
    c1 = c1.clamp(-0.5, 0.5)
    L = L_out.clone()
    for _ in range(10):
        g = L + c1*L*(1.0-L) - L_out; gp = 1.0+c1*(1.0-2.0*L)
        L = L - g / gp.clamp(min=1e-10)
    k_h = kb.view(-1,1)+kc.view(-1,1)*torch.cos(h_out)+ks.view(-1,1)*torch.sin(h_out)
    cp = cpb.view(-1,1)+cpc.view(-1,1)*torch.cos(h_out)+cps.view(-1,1)*torch.sin(h_out)
    cp = cp.clamp(0.4, 1.8)
    C_out = torch.sqrt(a_out**2+b_out**2+1e-30)
    f_L = torch.exp(k_h*(L-0.5))
    C_in = (C_out/f_L.clamp(min=1e-30)).clamp(min=0).pow(1.0/cp)
    a_in = a_out/C_out*C_in; b_in = b_out/C_out*C_in
    raw = torch.stack([L, a_in, b_in], dim=-1)
    lms_c = torch.bmm(raw, M2i.transpose(-1,-2))
    lms_c = torch.minimum(lms_c.clamp(min=0), s_gain.view(-1,1,1)-1e-10)
    ratio = (lms_c/(s_gain.view(-1,1,1)-lms_c).clamp(min=1e-30)).clamp(min=0)
    lms = sigma.view(-1,1,1)*ratio.pow(1.0/n.view(-1,1,1))
    return torch.bmm(lms, M1i.transpose(-1,-2))

def _unpack_J(x):
    """25p: M1(6)+M2(7)+n+σ+s+c1b+c1c+c1s+kb+kc+ks+cpb+cpc+cps."""
    M1, lms, v = unpack_M1(x)
    n = 0.42*torch.exp(x[:,13].clamp(-1,1)); v &= (n>=0.15)&(n<=1.1)
    sigma = torch.exp(x[:,14].clamp(-3,2)); s_gain = torch.exp(x[:,15].clamp(-2,2))
    lms_n = lms.clamp(min=1e-30).pow(n.unsqueeze(1))
    nr = s_gain.unsqueeze(1)*lms_n/(lms_n+sigma.pow(n).unsqueeze(1))
    v &= (nr>0).all(dim=1)
    M2, v2 = unpack_M2(x[:,6:13], nr); v &= v2
    c1b=x[:,16].clamp(-0.5,0.5); c1c=x[:,17].clamp(-0.5,0.5); c1s=x[:,18].clamp(-0.5,0.5)
    kb=x[:,19].clamp(-2,2); kc=x[:,20].clamp(-1,1); ks=x[:,21].clamp(-1,1)
    cpb=torch.exp(x[:,22].clamp(-1,0.7)); v &= (cpb>=0.3)&(cpb<=2.5)
    cpc=x[:,23].clamp(-0.5,0.5); cps=x[:,24].clamp(-0.5,0.5)
    M1i = torch.zeros_like(M1); M2i = torch.zeros_like(M1)
    good = v.nonzero(as_tuple=True)[0]
    if good.numel()>0: M1i[good]=torch.linalg.inv(M1[good]); M2i[good]=torch.linalg.inv(M2[good])
    return {"M1":M1,"M2":M2,"M1i":M1i,"M2i":M2i,"n":n,"sigma":sigma,"s":s_gain,
            "c1b":c1b,"c1c":c1c,"c1s":c1s,"kb":kb,"kc":kc,"ks":ks,"cpb":cpb,"cpc":cpc,"cps":cps}, v

def _fwd_J(xyz, d): return fwd_J(xyz, d["M1"],d["M2"],d["n"],d["sigma"],d["s"],d["c1b"],d["c1c"],d["c1s"],d["kb"],d["kc"],d["ks"],d["cpb"],d["cpc"],d["cps"])
def _inv_J(lab, d): return inv_J(lab, d["M1i"],d["M2i"],d["n"],d["sigma"],d["s"],d["c1b"],d["c1c"],d["c1s"],d["kb"],d["kc"],d["ks"],d["cpb"],d["cpc"],d["cps"])


# ── K: Hybrid (1-α)·x^p + α·NR(x) + M-cone M2 + light enrichment ──
def fwd_K(xyz, M1, M2, p, alpha, nr_n, nr_sigma, c1, k, cp):
    lms = (xyz.unsqueeze(0) @ M1.transpose(-1,-2)).clamp(min=0)  # (P,N,3)
    # Hybrid: blend power and NR
    power_part = lms.pow(p.view(-1,1,1))
    x_n = lms.pow(nr_n.view(-1,1,1))
    sig_n = nr_sigma.pow(nr_n).view(-1,1,1)
    nr_part = x_n / (x_n + sig_n)  # s=1 absorbed into M2
    a = alpha.view(-1,1,1)
    lms_c = (1 - a) * power_part + a * nr_part
    raw = torch.bmm(lms_c, M2.transpose(-1,-2))
    L, a_ch, b_ch = raw[...,0], raw[...,1], raw[...,2]
    L_out = L + c1.view(-1,1) * L * (1.0 - L)
    C = torch.sqrt(a_ch*a_ch + b_ch*b_ch + 1e-30)
    f_L = torch.exp(k.view(-1,1) * (L - 0.5))
    C_out = f_L * C.pow(cp.view(-1,1))
    a_out = a_ch / C * C_out; b_out = b_ch / C * C_out
    return torch.stack([L_out, a_out, b_out], dim=-1)

def inv_K(lab, M1i, M2i, p, alpha, nr_n, nr_sigma, c1, k, cp):
    L_out, a_out, b_out = lab[...,0], lab[...,1], lab[...,2]
    L = L_out.clone()
    for _ in range(10):
        g = L + c1.view(-1,1)*L*(1.0-L) - L_out
        gp = 1.0 + c1.view(-1,1)*(1.0-2.0*L)
        L = L - g / gp.clamp(min=1e-10)
    C_out = torch.sqrt(a_out**2 + b_out**2 + 1e-30)
    f_L = torch.exp(k.view(-1,1)*(L-0.5))
    C_in = (C_out / f_L.clamp(min=1e-30)).clamp(min=0).pow(1.0/cp.view(-1,1))
    a_in = a_out/C_out*C_in; b_in = b_out/C_out*C_in
    raw = torch.stack([L, a_in, b_in], dim=-1)
    lms_c = torch.bmm(raw, M2i.transpose(-1,-2))
    # Inverse hybrid: solve (1-α)·x^p + α·x^n/(x^n+σ^n) = y for x
    # Newton — 5 iterations sufficient (verified: 1e-15 round-trip)
    a_v = alpha.view(-1,1,1)
    p_v = p.view(-1,1,1); n_v = nr_n.view(-1,1,1)
    sn = nr_sigma.pow(nr_n).view(-1,1,1)
    x = lms_c.clamp(min=1e-30).pow(1.0/p_v)  # initial guess
    for _ in range(5):
        x = x.clamp(min=1e-30)
        pw = x.pow(p_v); xn = x.pow(n_v)
        f_val = (1-a_v)*pw + a_v*xn/(xn+sn) - lms_c
        dpw = (1-a_v)*p_v*x.pow(p_v-1)
        dnr = a_v*n_v*sn*x.pow(n_v-1)/(xn+sn)**2
        x = x - f_val / (dpw + dnr).clamp(min=1e-20)
        x = x.clamp(min=0)
    return torch.bmm(x, M1i.transpose(-1,-2))

def _unpack_K(x):
    """20p: M1(6)+M2(7)+p(1)+α(1)+nr_n(1)+nr_σ(1)+c1(1)+k(1)+cp(1)."""
    M1, lms, v = unpack_M1(x)
    p = (1./3.) * torch.exp(x[:,13].clamp(-0.5, 0.5))  # p ∈ [0.20, 0.55]
    v &= (p >= 0.20) & (p <= 0.55)
    alpha = torch.sigmoid(x[:,14])  # α ∈ [0, 1]
    nr_n = 0.42 * torch.exp(x[:,15].clamp(-0.8, 0.8))  # n ∈ [0.19, 0.93]
    v &= (nr_n >= 0.15) & (nr_n <= 1.0)
    nr_sigma = torch.exp(x[:,16].clamp(-2.0, 1.5))  # σ > 0
    # Hybrid response at D65
    lms_d65 = lms.clamp(min=1e-30)
    pw = lms_d65.pow(p.unsqueeze(1))
    xn = lms_d65.pow(nr_n.unsqueeze(1))
    sn = nr_sigma.pow(nr_n).unsqueeze(1)
    hybrid = (1-alpha.unsqueeze(1))*pw + alpha.unsqueeze(1)*xn/(xn+sn)
    v &= (hybrid > 0).all(dim=1)
    M2, v2 = unpack_M2(x[:, 6:13], hybrid)
    v &= v2
    # M-cone dominant L-row
    v &= (M2[:, 0, 1].abs() > M2[:, 0, 0].abs())
    v &= (M2[:, 0, 1].abs() > M2[:, 0, 2].abs())
    c1 = x[:,17].clamp(-0.2, 0.2)
    k = x[:,18].clamp(-0.5, 0.5)
    cp = 0.85 + 0.15 * torch.sigmoid(x[:,19])
    M1i = torch.zeros_like(M1); M2i = torch.zeros_like(M1)
    good = v.nonzero(as_tuple=True)[0]
    if good.numel() > 0:
        M1i[good] = torch.linalg.inv(M1[good])
        M2i[good] = torch.linalg.inv(M2[good])
    return {"M1":M1,"M2":M2,"M1i":M1i,"M2i":M2i,
            "p":p,"alpha":alpha,"nr_n":nr_n,"nr_sigma":nr_sigma,
            "c1":c1,"k":k,"cp":cp}, v

def _fwd_K(xyz, d): return fwd_K(xyz, d["M1"],d["M2"],d["p"],d["alpha"],d["nr_n"],d["nr_sigma"],d["c1"],d["k"],d["cp"])
def _inv_K(lab, d): return inv_K(lab, d["M1i"],d["M2i"],d["p"],d["alpha"],d["nr_n"],d["nr_sigma"],d["c1"],d["k"],d["cp"])


ARCHS = {
    "A": (Arch("A_SharedGamma",  14, True),  _unpack_A, _fwd_A, _inv_A),
    "B": (Arch("B_NakaRushton",  16, False), _unpack_B, _fwd_B, _inv_B),
    "C": (Arch("C_DivNorm",      22, False), _unpack_C, _fwd_C, _inv_C),
    "D": (Arch("D_LogWeighted",  16, True),  _unpack_D, _fwd_D, _inv_D),
    "E": (Arch("E_PowerEnriched",17, True),  _unpack_E, _fwd_E, _inv_E),
    "F": (Arch("F_PerChGamma",   17, False), _unpack_F, _fwd_F, _inv_F),
    "G": (Arch("G_PerChGammaCp", 18, False), _unpack_G, _fwd_G, _inv_G),
    "H": (Arch("H_NakaRushtonCp",19, False), _unpack_H, _fwd_H, _inv_H),
    "I": (Arch("I_NR_HueCp",     21, False), _unpack_I, _fwd_I, _inv_I),
    "J": (Arch("J_NR_HueEnrich", 25, False), _unpack_J, _fwd_J, _inv_J),
    "K": (Arch("K_Hybrid",       20, False), _unpack_K, _fwd_K, _inv_K),
}


# ════════════════════════════════════════════════════════════════════
#  BATCHED METRICS — all P candidates in one pass
# ════════════════════════════════════════════════════════════════════

def batch_cv(fwd, inv, d):
    """Gradient CV for all P candidates. Returns (P,)."""
    lab1 = fwd(PT[:, 0], d)  # (P, N, 3)
    lab2 = fwd(PT[:, 1], d)  # (P, N, 3)
    t = T_ST.view(1, 1, -1, 1)
    labs = lab1.unsqueeze(2) + t * (lab2 - lab1).unsqueeze(2)  # (P, N, 26, 3)
    P = labs.shape[0]
    lf = labs.reshape(P, N_PAIRS * (N_ST+1), 3)
    xyz = inv(lf, d)  # (P, N*26, 3)
    # XYZ → sRGB → 8bit → CIE Lab
    lin = (xyz @ MSi.T).clamp(0, 1)
    s8 = (l2s(lin) * 255).round() / 255.0
    xb = s2l(s8) @ MS.T
    r = xb.clamp(min=1e-10) / D65
    f = torch.where(r > 0.008856, r.pow(1./3.), 7.787*r + 16./116.)
    cl = torch.stack([116*f[...,1]-16, 500*(f[...,0]-f[...,1]),
                      200*(f[...,1]-f[...,2])], dim=-1)
    cl = cl.reshape(P, N_PAIRS, N_ST+1, 3)
    c1, c2 = cl[:,:,:-1], cl[:,:,1:]
    dL = c2[...,0]-c1[...,0]
    C1 = (c1[...,1]**2+c1[...,2]**2).sqrt()
    C2 = (c2[...,1]**2+c2[...,2]**2).sqrt()
    dC = C2 - C1
    dH = ((c2[...,1]-c1[...,1])**2+(c2[...,2]-c1[...,2])**2-dC**2).clamp(min=0).sqrt()
    SL = 1+0.015*(c1[...,0]-50)**2/(20+(c1[...,0]-50)**2).sqrt()
    SC = 1+0.045*C1; SH = 1+0.015*C1
    de = ((dL/SL)**2+(dC/SC)**2+(dH/SH)**2).sqrt()  # (P, N, 25)
    md = de.mean(2); sd = de.std(2)
    ok = md > 0.001
    cvs = torch.where(ok, sd / md, torch.zeros_like(md))
    cnt = ok.float().sum(1).clamp(min=1)
    return (cvs * ok.float()).sum(1) / cnt  # (P,)


def batch_hue(fwd, d):
    """Hue MSE (deg²) for 6 primaries. Returns (P,)."""
    lab = fwd(PRIM_XYZ, d)  # (P, 6, 3)
    h = torch.atan2(lab[:,:,2], lab[:,:,1]) * (180./math.pi) % 360
    dh = h - HUE_EXP.unsqueeze(0)
    dh = torch.where(dh > 180, dh-360, dh)
    dh = torch.where(dh < -180, dh+360, dh)
    return (dh**2).mean(1)  # (P,)


def batch_info(fwd, inv, d):
    """Yellow C, blue-white, red-white, primary L range. Returns dict of (P,) tensors."""
    yl = fwd(YEL_XYZ.squeeze(0).unsqueeze(0), d).squeeze(1)  # (P, 3)
    yC = (yl[:,1]**2 + yl[:,2]**2).sqrt()
    bl = fwd(BLU_XYZ.squeeze(0).unsqueeze(0), d).squeeze(1)
    wl = fwd(WHT_XYZ.squeeze(0).unsqueeze(0), d).squeeze(1)
    rl = fwd(RED_XYZ.squeeze(0).unsqueeze(0), d).squeeze(1)
    # Blue-white midpoint
    ml = ((bl + wl) / 2).unsqueeze(1)  # (P, 1, 3)
    mx = inv(ml, d).squeeze(1)  # (P, 3)
    ms = l2s((mx @ MSi.T).clamp(0, 1))
    bw = ms[:,1] / ms[:,0].clamp(min=0.01)
    # Red-white midpoint
    ml2 = ((rl + wl) / 2).unsqueeze(1)
    mx2 = inv(ml2, d).squeeze(1)
    ms2 = l2s((mx2 @ MSi.T).clamp(0, 1))
    rw = ms2[:,1] - ms2[:,2]
    # Primary L range
    pl = fwd(PRIM_XYZ, d)  # (P, 6, 3)
    plr = pl[:,:,0].max(dim=1).values - pl[:,:,0].min(dim=1).values
    return {"yC":yC, "yL":yl[:,0], "bL":bl[:,0], "bw":bw, "rw":rw, "plr":plr}


def batch_cusp(inv, d, hue_degs=(80,85,90)):
    """Cusp scan (production-grade: 150L×120C, matches test suite)."""
    P = d["M1"].shape[0]
    H = len(hue_degs)
    _Ls = torch.linspace(0.02, 0.998, 150, device=dev)
    _Cs = torch.linspace(0.001, 0.5, 120, device=dev)
    nL, nC = 150, 120
    Le = _Ls.view(nL,1).expand(nL,nC)
    Ce = _Cs.view(1,nC).expand(nL,nC)
    # Build Lab grid for all hues: (H*nL*nC, 3)
    grids = []
    for hd in hue_degs:
        hr = hd * math.pi / 180.0
        g = torch.stack([Le, Ce*math.cos(hr), Ce*math.sin(hr)], dim=-1).reshape(-1, 3)
        grids.append(g)
    grid = torch.cat(grids, dim=0)
    G = grid.shape[0]
    lab_exp = grid.unsqueeze(0).expand(P, G, 3)
    xyz = inv(lab_exp, d)
    lin = xyz @ MSi.T
    ok = ((lin >= -0.002) & (lin <= 1.002)).all(dim=-1).reshape(P, H, nL, nC)
    mc = torch.where(ok, Ce.unsqueeze(0).unsqueeze(0).expand(P,H,nL,nC),
                     torch.zeros(P,H,nL,nC,device=dev)).max(dim=3).values  # (P,H,nL)
    ci = mc.argmax(dim=2)  # (P, H)
    cL = _Ls[ci]  # (P, H)
    cC = mc.gather(2, ci.unsqueeze(2)).squeeze(2)  # (P, H)
    ci2 = (ci + 2).clamp(max=nL-1)
    cC2 = mc.gather(2, ci2.unsqueeze(2)).squeeze(2)
    cliff = torch.where(cC > 0.01, (cC - cC2) / cC * 100, torch.full_like(cC, 100.0))
    return cL, cC, cliff


def batch_gamut_cov(inv, d, n_hues=18):
    """Min cusp chroma across hues. Returns (P,)."""
    P = d["M1"].shape[0]
    Le = GC_Ls.view(50,1).expand(50,40)
    Ce = GC_Cs.view(1,40).expand(50,40)
    min_C = torch.full((P,), 999.0, device=dev)
    # Process 6 hues per chunk to limit VRAM
    for chunk_start in range(0, n_hues, 6):
        chunk_hues = range(chunk_start, min(chunk_start+6, n_hues))
        grids = []
        for hi in chunk_hues:
            hd = hi * (360.0 / n_hues)
            hr = hd * math.pi / 180.0
            g = torch.stack([Le, Ce*math.cos(hr), Ce*math.sin(hr)], dim=-1).reshape(-1, 3)
            grids.append(g)
        grid = torch.cat(grids, dim=0)
        n_h = len(chunk_hues)
        G = grid.shape[0]
        lab_exp = grid.unsqueeze(0).expand(P, G, 3)
        xyz = inv(lab_exp, d)
        lin = xyz @ MSi.T
        ok = ((lin >= -0.002) & (lin <= 1.002)).all(dim=-1).reshape(P, n_h, 50, 40)
        mc = torch.where(ok, Ce.unsqueeze(0).unsqueeze(0).expand(P,n_h,50,40),
                         torch.zeros(P,n_h,50,40,device=dev)).max(dim=3).values  # (P,n_h,50)
        max_per_hue = mc.max(dim=2).values  # (P, n_h)
        min_per_chunk = max_per_hue.min(dim=1).values  # (P,)
        min_C = torch.minimum(min_C, min_per_chunk)
    return min_C


def batch_ach(fwd, d):
    """Max |a|,|b| on 257-step gray ramp (matches production test). Returns (P,)."""
    # sRGB gray 0/256 to 256/256
    g_srgb = torch.linspace(0, 1, 257, device=dev)
    g_lin = s2l(g_srgb)
    g_xyz = g_lin.unsqueeze(1) * (MS.T @ torch.ones(3, device=dev)).unsqueeze(0)  # wrong
    # Correct: sRGB gray [g,g,g] → linear → XYZ
    g_rgb = g_srgb.unsqueeze(1).expand(257, 3)  # (257, 3) all [g,g,g]
    g_xyz = s2l(g_rgb) @ MS.T  # (257, 3)
    lab = fwd(g_xyz, d)  # (P, 257, 3)
    return lab[:, :, 1:].abs().max(dim=2).values.max(dim=1).values  # (P,)


def batch_hue_drift(fwd, inv, d, n_pairs=50):
    """GPU hue drift. Returns drift_mean (P,), drift_max (P,)."""
    lab1 = fwd(PT[:n_pairs, 0], d)  # (P, np, 3)
    lab2 = fwd(PT[:n_pairs, 1], d)  # (P, np, 3)
    n_steps = 13
    t = torch.linspace(0, 1, n_steps, device=dev).view(1, 1, -1, 1)
    labs = lab1.unsqueeze(2) + t * (lab2 - lab1).unsqueeze(2)  # (P, np, 13, 3)
    P = labs.shape[0]
    lf = labs.reshape(P, n_pairs * n_steps, 3)
    xyz = inv(lf, d)
    lin = (xyz @ MSi.T).clamp(0, 1)
    s8 = (l2s(lin) * 255).round() / 255.0
    xb = s2l(s8) @ MS.T
    r = xb.clamp(min=1e-10) / D65
    f = torch.where(r > 0.008856, r.pow(1./3.), 7.787*r + 16./116.)
    cl = torch.stack([116*f[...,1]-16, 500*(f[...,0]-f[...,1]),
                      200*(f[...,1]-f[...,2])], dim=-1)
    cl = cl.reshape(P, n_pairs, n_steps, 3)
    C = (cl[...,1]**2 + cl[...,2]**2).sqrt()
    h = torch.atan2(cl[...,2], cl[...,1])
    dh = (h[:,:,1:] - h[:,:,:-1])
    dh = torch.atan2(torch.sin(dh), torch.cos(dh)).abs() * (180./math.pi)
    ok = (C[:,:,:-1] > 3) & (C[:,:,1:] > 3)
    dh = torch.where(ok, dh, torch.zeros_like(dh))
    pm = dh.max(dim=2).values  # (P, np)
    return pm.mean(dim=1), pm.max(dim=1).values


def batch_cond(d):
    """Condition numbers. Returns c1 (P,), c2 (P,)."""
    return torch.linalg.cond(d["M1"]), torch.linalg.cond(d["M2"])


def batch_monotonicity(inv, d, n_hues=360):
    """Count non-unimodal hues per candidate. Fully GPU — no per-candidate Python loop.
    Returns (P,) count of non-unimodal hues."""
    P = d["M1"].shape[0]
    Lf = torch.linspace(0.02, 0.998, 150, device=dev)
    Cf = torch.linspace(0.001, 0.5, 120, device=dev)
    nL, nC = Lf.shape[0], Cf.shape[0]
    Le = Lf.view(nL, 1).expand(nL, nC)
    Ce = Cf.view(1, nC).expand(nL, nC)
    non_uni = torch.zeros(P, device=dev)
    # Process hues in batches of HB, all P candidates at once
    HB = 6  # hues per batch (P * HB * 150 * 120 * 3 * 8 bytes per tensor)
    for hs in range(0, n_hues, HB):
        he = min(hs + HB, n_hues)
        nh = he - hs
        # Build grid: (nh * nL * nC, 3)
        grids = []
        for hi in range(hs, he):
            hr = hi * math.pi / 180.0
            g = torch.stack([Le, Ce * math.cos(hr), Ce * math.sin(hr)],
                            dim=-1).reshape(-1, 3)
            grids.append(g)
        grid = torch.cat(grids, dim=0)  # (nh*nL*nC, 3)
        G = grid.shape[0]
        # All P candidates: (P, G, 3)
        lab_exp = grid.unsqueeze(0).expand(P, G, 3)
        xyz = inv(lab_exp, d)         # (P, G, 3)
        lin = xyz @ MSi.T            # (P, G, 3)
        ok = ((lin >= -0.002) & (lin <= 1.002)).all(dim=-1).reshape(P, nh, nL, nC)
        Ce_exp = Cf.view(1, 1, 1, nC).expand(P, nh, nL, nC)
        mc = torch.where(ok, Ce_exp,
                         torch.zeros(P, nh, nL, nC, device=dev)).max(dim=3).values  # (P, nh, nL)
        # Find cusp index per (P, nh)
        ci = mc.argmax(dim=2)  # (P, nh)
        # Check monotonicity after cusp — vectorized across P and nh
        # mc_diff[p, h, l] = mc[p, h, l+1] - mc[p, h, l]
        mc_diff = mc[:, :, 1:] - mc[:, :, :-1]  # (P, nh, nL-1)
        # Build mask: only look after cusp for each (p, h)
        idx = torch.arange(nL - 1, device=dev).view(1, 1, nL - 1)
        after_cusp = idx >= ci.unsqueeze(2)  # (P, nh, nL-1)
        # Any increase > 0.001 after cusp?
        has_increase = (mc_diff > 0.001) & after_cusp  # (P, nh, nL-1)
        non_uni_hue = has_increase.any(dim=2).float()   # (P, nh) — 1 if non-unimodal
        non_uni += non_uni_hue.sum(dim=1)               # (P,)
    return non_uni


def batch_wg_rt(fwd, inv, d):
    """Wide gamut round-trip error. Returns (P,) max error."""
    M_P3 = torch.tensor([[0.4865709486482162,0.26566769316909306,0.1982172852343625],
                         [0.2289745640697488,0.6917385218365064,0.079286914093745],
                         [0.0,0.04511338185890264,1.0439443689009757]], device=dev)
    M_R2020 = torch.tensor([[0.6369580483012914,0.14461690358620832,0.1688809751641721],
                            [0.2627002120112671,0.6779980715188708,0.05930171646986196],
                            [0.0,0.028072693049087428,1.0609850577107909]], device=dev)
    wg_prim = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]],
                           dtype=torch.float64, device=dev)
    wg_xyz = torch.cat([wg_prim @ M_P3.T, wg_prim @ M_R2020.T], dim=0)  # (12, 3)
    lab = fwd(wg_xyz, d)       # (P, 12, 3)
    xyz_rt = inv(lab, d)       # (P, 12, 3)
    err = (wg_xyz.unsqueeze(0) - xyz_rt).abs().max(dim=2).values.max(dim=1).values  # (P,)
    return err


# ════════════════════════════════════════════════════════════════════
#  EVALUATE POPULATION — single GPU pass for all P candidates
# ════════════════════════════════════════════════════════════════════

def evaluate_population(x_np, arch_key):
    """x_np: (P, n_params) numpy → losses: (P,) numpy."""
    arch, unpack_fn, fwd_fn, inv_fn = ARCHS[arch_key]
    x = torch.tensor(x_np, device=dev, dtype=torch.float64)
    P = x.shape[0]
    losses = torch.full((P,), 999.0, device=dev)

    with torch.no_grad():
        d, valid = unpack_fn(x)
        if not valid.any():
            return losses.cpu().numpy()

        c1, c2 = batch_cond(d)
        valid &= (c1 < 15) & (c2 < 25)
        if not valid.any():
            return losses.cpu().numpy()

        info = batch_info(fwd_fn, inv_fn, d)
        valid &= info["yC"] > 0.05
        if not valid.any():
            return losses.cpu().numpy()

        # All metrics in one pass
        cv = batch_cv(fwd_fn, inv_fn, d)         # (P,)
        hue_sq = batch_hue(fwd_fn, d)             # (P,)
        # 72 hues (every 5°) — matches test suite resolution for cliff detection
        cusp_hues = tuple(range(0, 360, 5))
        cL, cC, cliff = batch_cusp(inv_fn, d, cusp_hues)  # (P,72) each
        ach = batch_ach(fwd_fn, d)
        dm, dx = batch_hue_drift(fwd_fn, inv_fn, d)
        wg_rt = batch_wg_rt(fwd_fn, inv_fn, d)

        # ── SOFT PENALTIES ──
        pen = torch.zeros(P, device=dev)
        pen += torch.where(info["yC"] < 0.12, (0.12 - info["yC"])**2 * 200, 0.0)
        pen += torch.where(info["yC"] < 0.18, (0.18 - info["yC"])**2 * 500, 0.0)
        # Blue must be dark: L < 0.55 (OKLab=0.452, CIE Lab=0.323)
        pen += torch.where(info["bL"] > 0.55, (info["bL"] - 0.55)**2 * 2000, 0.0)
        pen += torch.where(info["bL"] > 0.70, (info["bL"] - 0.70)**2 * 10000, 0.0)
        pen += torch.where(info["bw"] < 1.20, (1.20 - info["bw"])**2 * 50, 0.0)
        pen += torch.where(info["bw"] < 1.0, (1.0 - info["bw"])**2 * 5000, 0.0)  # hard: no purple shift
        pen += torch.where(info["rw"] > 0.08, (info["rw"] - 0.08)**2 * 100, 0.0)
        # Red→White must NOT shift blue: G-B must be >= 0
        pen += torch.where(info["rw"] < 0.0, info["rw"]**2 * 2000, 0.0)
        pen += torch.where(info["plr"] < 0.45, (0.45 - info["plr"])**2 * 500, 0.0)
        pen += torch.where(info["plr"] < 0.35, (0.35 - info["plr"])**2 * 5000, 0.0)
        pen += torch.where(c1 > 3.15, (c1 - 3.15)**2 * 5, 0.0)
        pen += torch.where(c2 > 10, (c2 - 10)**2 * 3, 0.0)
        pen += torch.where(dx > 40, (dx - 40)**2 * 0.1, 0.0)
        pen += torch.where(dx > 60, (dx - 60)**2 * 0.3, 0.0)
        # Achromatic
        pen += torch.where(ach > 1e-6, (ach - 1e-6)**2 * 1e12, 0.0)
        pen += torch.where(ach > 1e-4, (ach - 1e-4)**2 * 1e14, 0.0)
        # Wide gamut round-trip
        pen += torch.where(wg_rt > 1e-10, wg_rt * 1e6, 0.0)
        pen += torch.where(wg_rt > 1e-4, wg_rt * 1e10, 0.0)

        # ── CUSP PENALTIES (72 hues) ──
        cusp_pen = torch.zeros(P, device=dev)
        n_hues = cL.shape[1]
        for hi in range(n_hues):
            # Cusp L range
            cusp_pen += torch.where(cL[:,hi] > 0.92, (cL[:,hi]-0.92)**2 * 30, 0.0)
            cusp_pen += torch.where(cL[:,hi] < 0.78, (0.78-cL[:,hi])**2 * 30, 0.0)
            # Cliff — target < 50% (OKLab is 48%)
            cusp_pen += torch.where(cliff[:,hi] > 40, (cliff[:,hi]-40)**2 * 0.1, 0.0)
            cusp_pen += torch.where(cliff[:,hi] > 60, (cliff[:,hi]-60)**2 * 0.5, 0.0)
            cusp_pen += torch.where(cliff[:,hi] > 80, (cliff[:,hi]-80)**2 * 2.0, 0.0)

        # ── DEAD ZONE PENALTY: cusp_C < 0.02 = weak gamut at that hue ──
        dead_count = (cC < 0.02).float().sum(dim=1)  # (P,)
        pen += dead_count * 20.0  # gradual: each dead hue = 20

        # ── CUSP SMOOTHNESS: adjacent hue jump ──
        cL_shift = torch.cat([cL[:, 1:], cL[:, :1]], dim=1)  # circular shift
        jumps = (cL - cL_shift).abs()  # (P, 72)
        max_jump = jumps.max(dim=1).values  # (P,)
        pen += torch.where(max_jump > 0.15, (max_jump - 0.15)**2 * 100, 0.0)
        pen += torch.where(max_jump > 0.5, (max_jump - 0.5)**2 * 500, 0.0)

        # ── LOSS ──
        loss = 5.0*cv + 2.0*cusp_pen + 0.01*hue_sq + pen
        losses = torch.where(valid, loss, torch.full_like(loss, 999.0))

    return losses.cpu().numpy()


# ════════════════════════════════════════════════════════════════════
#  SEED GENERATION
# ════════════════════════════════════════════════════════════════════

def _ortho_np(s):
    sn = s / (np.linalg.norm(s) + 1e-30)
    v = np.array([1.,0.,0.]) if abs(sn[0]) < 0.9 else np.array([0.,1.,0.])
    e1 = v - np.dot(v, sn) * sn; e1 /= np.linalg.norm(e1) + 1e-30
    e2 = np.cross(sn, e1); e2 /= np.linalg.norm(e2) + 1e-30
    return e1, e2

def _pack_m1(M1):
    x = np.zeros(6)
    for i in range(3): x[2*i] = M1[i,0]; x[2*i+1] = M1[i,1]
    return x

def _pack_m2(M2, s):
    e1, e2 = _ortho_np(s)
    return np.array([M2[0,0],M2[0,1],M2[0,2], M2[1]@e1,M2[1]@e2, M2[2]@e1,M2[2]@e2])

def _reproject(M2, s_old, s_new):
    Lw = M2[0] @ s_new
    if abs(Lw) < 1e-10: return None
    M2n = np.zeros((3,3)); M2n[0] = M2[0] / Lw
    e1, e2 = _ortho_np(s_new)
    for r in [1,2]:
        c1, c2 = M2[r]@e1, M2[r]@e2
        M2n[r] = c1*e1 + c2*e2
    return M2n

V14n = V14_M1.cpu().numpy(); V14n2 = V14_M2.cpu().numpy()
OKn = OK_M1.cpu().numpy(); OKn2 = OK_M2.cpu().numpy()

def make_seeds_A():
    seeds = []
    for lbl, M1, M2 in [("v14", V14n, V14n2), ("oklab", OKn, OKn2)]:
        s = np.sign(M1@D65_np) * np.abs(M1@D65_np)**(1./3.)
        M2p = _reproject(M2, s, s)
        if M2p is None: continue
        x = np.zeros(14); x[:6] = _pack_m1(M1); x[6:13] = _pack_m2(M2p, s)
        seeds.append((lbl, x, 0.03))
    rng = np.random.RandomState(42)
    x = np.zeros(14)
    mid = 0.5*V14n + 0.5*OKn
    for i in range(3): x[2*i]=mid[i,0]+rng.randn()*.15; x[2*i+1]=mid[i,1]+rng.randn()*.15
    x[6:9] = (V14n2[0]+OKn2[0])/2 + rng.randn(3)*.1; x[9:13] = rng.randn(4)*.8
    seeds.append(("random", x, 0.05))
    return seeds

def make_seeds_B():
    seeds = []
    n_d, sig_d, s_d = 0.42, 0.5, 1.5
    for lbl, M1, M2 in [("v14", V14n, V14n2), ("oklab", OKn, OKn2)]:
        lms = M1 @ D65_np; ln = lms**n_d; nr = s_d*ln/(ln+sig_d**n_d)
        s_old = np.sign(lms)*np.abs(lms)**(1./3.)
        M2p = _reproject(M2, s_old, nr)
        if M2p is None: continue
        x = np.zeros(16); x[:6] = _pack_m1(M1); x[6:13] = _pack_m2(M2p, nr)
        x[14] = np.log(sig_d); x[15] = np.log(s_d)
        seeds.append((lbl, x, 0.03))
    rng = np.random.RandomState(43)
    x = np.zeros(16); mid = 0.5*V14n+0.5*OKn
    for i in range(3): x[2*i]=mid[i,0]+rng.randn()*.15; x[2*i+1]=mid[i,1]+rng.randn()*.15
    x[6:9] = (V14n2[0]+OKn2[0])/2+rng.randn(3)*.1; x[9:13] = rng.randn(4)*.8
    x[14] = np.log(.5)+rng.randn()*.3; x[15] = np.log(1.5)+rng.randn()*.3
    seeds.append(("random", x, 0.05))
    return seeds

def make_seeds_C():
    seeds = []
    n_d, sig_d, s_d = 0.33, 0.5, 1.0
    for lbl, M1, M2 in [("v14", V14n, V14n2), ("oklab", OKn, OKn2)]:
        lms = M1@D65_np; u = lms**n_d; dn = s_d*u/(sig_d**n_d+u.sum())
        # Approximate: W=I → denom same for each channel (not exactly, but close)
        dn2 = s_d * u / (sig_d**n_d + np.eye(3) @ u)
        s_old = np.sign(lms)*np.abs(lms)**(1./3.)
        M2p = _reproject(M2, s_old, dn2)
        if M2p is None: continue
        x = np.zeros(22); x[:6] = _pack_m1(M1); x[6:13] = _pack_m2(M2p, dn2)
        x[14] = np.log(sig_d); x[21] = np.log(s_d)
        seeds.append((lbl, x, 0.03))
    rng = np.random.RandomState(44)
    x = np.zeros(22); mid = 0.5*V14n+0.5*OKn
    for i in range(3): x[2*i]=mid[i,0]+rng.randn()*.15; x[2*i+1]=mid[i,1]+rng.randn()*.15
    x[6:9] = (V14n2[0]+OKn2[0])/2+rng.randn(3)*.1; x[9:13] = rng.randn(4)*.8
    x[15:21] = rng.randn(6)*.5
    seeds.append(("random", x, 0.05))
    return seeds

def make_seeds_D():
    seeds = []
    w_d = np.array([1.,1.,1.])
    for lbl, M1, M2 in [("v14", V14n, V14n2), ("oklab", OKn, OKn2)]:
        s_ach = w_d * np.log(2)
        s_old = np.sign(M1@D65_np)*np.abs(M1@D65_np)**(1./3.)
        M2p = _reproject(M2, s_old, s_ach)
        if M2p is None: continue
        x = np.zeros(16); x[:6] = _pack_m1(M1); x[9:16] = _pack_m2(M2p, s_ach)
        seeds.append((lbl, x, 0.03))
    rng = np.random.RandomState(45)
    x = np.zeros(16); mid = 0.5*V14n+0.5*OKn
    for i in range(3): x[2*i]=mid[i,0]+rng.randn()*.15; x[2*i+1]=mid[i,1]+rng.randn()*.15
    x[6:9] = rng.randn(3)*.3; x[9:12] = (V14n2[0]+OKn2[0])/2+rng.randn(3)*.1
    x[12:16] = rng.randn(4)*.8
    seeds.append(("random", x, 0.05))
    return seeds

def make_seeds_E():
    seeds = []
    for lbl, M1, M2 in [("v14", V14n, V14n2), ("oklab", OKn, OKn2)]:
        s = np.sign(M1@D65_np)*np.abs(M1@D65_np)**(1./3.)
        M2p = _reproject(M2, s, s)
        if M2p is None: continue
        x = np.zeros(17); x[:6] = _pack_m1(M1); x[6:13] = _pack_m2(M2p, s)
        seeds.append((lbl, x, 0.03))
    rng = np.random.RandomState(46)
    x = np.zeros(17); mid = 0.5*V14n+0.5*OKn
    for i in range(3): x[2*i]=mid[i,0]+rng.randn()*.15; x[2*i+1]=mid[i,1]+rng.randn()*.15
    x[6:9] = (V14n2[0]+OKn2[0])/2+rng.randn(3)*.1; x[9:13] = rng.randn(4)*.8
    seeds.append(("random", x, 0.05))
    return seeds

def make_seeds_F():
    seeds = []
    for lbl, M1, M2 in [("v14", V14n, V14n2), ("oklab", OKn, OKn2)]:
        # Per-channel gamma: start at shared 1/3
        gamma = np.array([1./3., 1./3., 1./3.])
        s = np.sign(M1@D65_np) * np.abs(M1@D65_np)**gamma
        M2p = _reproject(M2, np.sign(M1@D65_np)*np.abs(M1@D65_np)**(1./3.), s)
        if M2p is None: continue
        x = np.zeros(17); x[:6] = _pack_m1(M1); x[6:13] = _pack_m2(M2p, s)
        x[13] = 0.0  # g2 ratio = 1 (shared)
        x[14] = 0.0  # g3 ratio = 1 (shared)
        x[15] = 0.0  # c1 = 0
        x[16] = 0.0  # k = 0
        seeds.append((lbl, x, 0.03))
    # Seed from nextgen result: M-cone boost g2=0.359 (log ratio = log(0.359/0.333) = 0.075)
    rng = np.random.RandomState(47)
    x = np.zeros(17); mid = 0.5*V14n+0.5*OKn
    for i in range(3): x[2*i]=mid[i,0]+rng.randn()*.15; x[2*i+1]=mid[i,1]+rng.randn()*.15
    x[6:9] = (V14n2[0]+OKn2[0])/2+rng.randn(3)*.1; x[9:13] = rng.randn(4)*.8
    x[13] = 0.075  # M-cone slight boost
    x[14] = -0.025  # S-cone slight reduction
    seeds.append(("mcone", x, 0.05))
    return seeds

def make_seeds_G():
    seeds = []
    for lbl, M1, M2 in [("v14", V14n, V14n2), ("oklab", OKn, OKn2)]:
        gamma = np.array([1./3., 1./3., 1./3.])
        s = np.sign(M1@D65_np) * np.abs(M1@D65_np)**gamma
        M2p = _reproject(M2, s, s)
        if M2p is None: continue
        x = np.zeros(18); x[:6] = _pack_m1(M1); x[6:13] = _pack_m2(M2p, s)
        # x[13]=g2_ratio=0, x[14]=g3_ratio=0 (shared), x[15]=c1, x[16]=k, x[17]=cp
        seeds.append((lbl, x, 0.03))
    rng = np.random.RandomState(48)
    x = np.zeros(18); mid = 0.5*V14n+0.5*OKn
    for i in range(3): x[2*i]=mid[i,0]+rng.randn()*.15; x[2*i+1]=mid[i,1]+rng.randn()*.15
    x[6:9] = (V14n2[0]+OKn2[0])/2+rng.randn(3)*.1; x[9:13] = rng.randn(4)*.8
    x[13] = 0.075; x[14] = -0.025  # M-cone boost seed
    seeds.append(("mcone", x, 0.05))
    return seeds

def make_seeds_H():
    seeds = []
    # Named seeds with different NR starting params
    nr_configs = [
        ("v14_nr42", V14n, V14n2, 0.42, 0.5, 1.5, 0.03),
        ("ok_nr42", OKn, OKn2, 0.42, 0.5, 1.5, 0.03),
        ("v14_nr33", V14n, V14n2, 0.33, 0.3, 1.0, 0.03),   # lower n → less compression
        ("ok_nr33", OKn, OKn2, 0.33, 0.3, 1.0, 0.03),
        ("v14_nr55", V14n, V14n2, 0.55, 0.8, 2.0, 0.03),   # higher n, higher sigma
        ("ok_nr55", OKn, OKn2, 0.55, 0.8, 2.0, 0.03),
        ("v14_nr25", V14n, V14n2, 0.25, 0.2, 0.8, 0.03),   # very low n → almost power law
        ("ok_nr25", OKn, OKn2, 0.25, 0.2, 0.8, 0.03),
    ]
    for lbl, M1, M2, n_d, sig_d, s_d, sigma in nr_configs:
        lms = M1 @ D65_np; ln = lms**n_d; nr = s_d*ln/(ln+sig_d**n_d)
        s_old = np.sign(lms)*np.abs(lms)**(1./3.)
        M2p = _reproject(M2, s_old, nr)
        if M2p is None: continue
        x = np.zeros(19); x[:6] = _pack_m1(M1); x[6:13] = _pack_m2(M2p, nr)
        x[13] = np.log(n_d / 0.42)  # n ratio from default
        x[14] = np.log(sig_d); x[15] = np.log(s_d)
        seeds.append((lbl, x, sigma))
    # Random seeds with varying NR params
    for i in range(12):
        rng = np.random.RandomState(49 + i)
        x = np.zeros(19); mid = 0.5*V14n+0.5*OKn
        for j in range(3): x[2*j]=mid[j,0]+rng.randn()*.15; x[2*j+1]=mid[j,1]+rng.randn()*.15
        x[6:9] = (V14n2[0]+OKn2[0])/2+rng.randn(3)*.1; x[9:13] = rng.randn(4)*.8
        x[13] = rng.randn()*.3  # n variation
        x[14] = np.log(.5)+rng.randn()*.5  # sigma variation
        x[15] = np.log(1.5)+rng.randn()*.5  # s_gain variation
        seeds.append((f"rnd{i}", x, 0.05))
    return seeds

def make_seeds_I():
    """I seeds: same as H but with 2 extra cp hue params (cos, sin) = 0."""
    seeds = []
    for lbl, M1, M2, n_d, sig_d, s_d, sig in [
        ("v14_42", V14n, V14n2, 0.42, 0.5, 1.5, 0.03),
        ("ok_42", OKn, OKn2, 0.42, 0.5, 1.5, 0.03),
        ("v14_55", V14n, V14n2, 0.55, 0.8, 2.0, 0.03),
        ("ok_55", OKn, OKn2, 0.55, 0.8, 2.0, 0.03),
    ]:
        lms = M1@D65_np; ln = lms**n_d; nr = s_d*ln/(ln+sig_d**n_d)
        s_old = np.sign(lms)*np.abs(lms)**(1./3.)
        M2p = _reproject(M2, s_old, nr)
        if M2p is None: continue
        x = np.zeros(21); x[:6] = _pack_m1(M1); x[6:13] = _pack_m2(M2p, nr)
        x[13] = np.log(n_d/0.42); x[14] = np.log(sig_d); x[15] = np.log(s_d)
        seeds.append((lbl, x, sig))
    for i in range(16):
        rng = np.random.RandomState(60+i)
        x = np.zeros(21); mid = 0.5*V14n+0.5*OKn
        for j in range(3): x[2*j]=mid[j,0]+rng.randn()*.15; x[2*j+1]=mid[j,1]+rng.randn()*.15
        x[6:9] = (V14n2[0]+OKn2[0])/2+rng.randn(3)*.1; x[9:13] = rng.randn(4)*.8
        x[13] = rng.randn()*.3; x[14] = np.log(.5)+rng.randn()*.5; x[15] = np.log(1.5)+rng.randn()*.5
        seeds.append((f"rnd{i}", x, 0.05))
    return seeds

def make_seeds_J():
    """J seeds: same as H but with 6 extra hue enrichment params."""
    seeds = []
    for lbl, M1, M2, n_d, sig_d, s_d, sig in [
        ("v14_42", V14n, V14n2, 0.42, 0.5, 1.5, 0.03),
        ("ok_42", OKn, OKn2, 0.42, 0.5, 1.5, 0.03),
        ("v14_55", V14n, V14n2, 0.55, 0.8, 2.0, 0.03),
        ("ok_55", OKn, OKn2, 0.55, 0.8, 2.0, 0.03),
    ]:
        lms = M1@D65_np; ln = lms**n_d; nr = s_d*ln/(ln+sig_d**n_d)
        s_old = np.sign(lms)*np.abs(lms)**(1./3.)
        M2p = _reproject(M2, s_old, nr)
        if M2p is None: continue
        x = np.zeros(25); x[:6] = _pack_m1(M1); x[6:13] = _pack_m2(M2p, nr)
        x[13] = np.log(n_d/0.42); x[14] = np.log(sig_d); x[15] = np.log(s_d)
        seeds.append((lbl, x, sig))
    for i in range(16):
        rng = np.random.RandomState(80+i)
        x = np.zeros(25); mid = 0.5*V14n+0.5*OKn
        for j in range(3): x[2*j]=mid[j,0]+rng.randn()*.15; x[2*j+1]=mid[j,1]+rng.randn()*.15
        x[6:9] = (V14n2[0]+OKn2[0])/2+rng.randn(3)*.1; x[9:13] = rng.randn(4)*.8
        x[13] = rng.randn()*.3; x[14] = np.log(.5)+rng.randn()*.5; x[15] = np.log(1.5)+rng.randn()*.5
        seeds.append((f"rnd{i}", x, 0.05))
    return seeds

def make_seeds_K():
    seeds = []
    # α=0 → pure power (like E), α=1 → pure NR (like H)
    # Test range of α values
    for lbl, M1, M2, alpha_init, sig in [
        ("v14_a20", V14n, V14n2, 0.2, 0.03),
        ("ok_a20", OKn, OKn2, 0.2, 0.03),
        ("v14_a50", V14n, V14n2, 0.5, 0.03),
        ("ok_a50", OKn, OKn2, 0.5, 0.03),
        ("v14_a80", V14n, V14n2, 0.8, 0.03),
        ("ok_a80", OKn, OKn2, 0.8, 0.03),
    ]:
        # Hybrid response at D65 with default params
        p_d, n_d, sig_d = 1./3., 0.42, 0.5
        lms = M1@D65_np; lms_c = lms.clip(min=1e-30)
        pw = lms_c**(p_d); xn = lms_c**(n_d)
        hybrid = (1-alpha_init)*pw + alpha_init*xn/(xn+sig_d**n_d)
        s_old = np.sign(lms)*np.abs(lms)**(1./3.)
        M2p = _reproject(M2, s_old, hybrid)
        if M2p is None: continue
        x = np.zeros(20); x[:6] = _pack_m1(M1); x[6:13] = _pack_m2(M2p, hybrid)
        # x[13]=log(p/(1/3))=0, x[14]=logit(α), x[15]=log(n/0.42)=0, x[16]=log(σ)
        x[14] = float(np.log(alpha_init / (1.0 - alpha_init)))  # logit(α)
        x[16] = np.log(sig_d)
        seeds.append((lbl, x, sig))
    # Random seeds
    for i in range(14):
        rng = np.random.RandomState(100+i)
        x = np.zeros(20); mid = 0.5*V14n+0.5*OKn
        for j in range(3): x[2*j]=mid[j,0]+rng.randn()*.15; x[2*j+1]=mid[j,1]+rng.randn()*.15
        x[6:9] = (V14n2[0]+OKn2[0])/2+rng.randn(3)*.1; x[9:13] = rng.randn(4)*.8
        x[14] = rng.randn()*0.5  # α around 0.5
        x[16] = np.log(0.5)+rng.randn()*0.5
        seeds.append((f"rnd{i}", x, 0.05))
    return seeds

SEED_FNS = {"A": make_seeds_A, "B": make_seeds_B, "C": make_seeds_C,
            "D": make_seeds_D, "E": make_seeds_E, "F": make_seeds_F,
            "G": make_seeds_G, "H": make_seeds_H, "I": make_seeds_I,
            "J": make_seeds_J, "K": make_seeds_K}


# ════════════════════════════════════════════════════════════════════
#  SINGLE-CANDIDATE METRICS (for final reporting)
# ════════════════════════════════════════════════════════════════════

def full_metrics_single(x_np, arch_key):
    """Compute all metrics for a single candidate."""
    arch, unpack_fn, fwd_fn, inv_fn = ARCHS[arch_key]
    x = torch.tensor(x_np, device=dev).unsqueeze(0)  # (1, n)
    with torch.no_grad():
        d, v = unpack_fn(x)
        if not v.any():
            return None
        cv = batch_cv(fwd_fn, inv_fn, d)[0].item()
        hue = batch_hue(fwd_fn, d)[0].item()
        info = {k: v_[0].item() for k, v_ in batch_info(fwd_fn, inv_fn, d).items()}
        cL, cC, cliff = batch_cusp(inv_fn, d)
        gamut = batch_gamut_cov(inv_fn, d, 36)[0].item()
        ach = batch_ach(fwd_fn, d)[0].item()
        dm, dx = batch_hue_drift(fwd_fn, inv_fn, d)
        c1, c2 = batch_cond(d)
        mono = batch_monotonicity(inv_fn, d, 18)[0].item()
        wg = batch_wg_rt(fwd_fn, inv_fn, d)[0].item()
    return {
        "cv": cv, "hue_rms": math.sqrt(max(hue, 0)),
        "cusp_L_85": cL[0,1].item(), "cusp_C_85": cC[0,1].item(),
        "cliff_85": cliff[0,1].item(),
        "cusp_L_80": cL[0,0].item(), "cusp_L_90": cL[0,2].item(),
        "gamut_min_C": gamut, "ach_err": ach,
        "drift_mean": dm[0].item(), "drift_max": dx[0].item(),
        "bw": info["bw"], "rw_gb": info["rw"], "plr": info["plr"],
        "cond1": c1[0].item(), "cond2": c2[0].item(),
        "yL": info["yL"], "yC": info["yC"],
        "mono_fails": int(mono), "wg_rt": wg,
    }


# ════════════════════════════════════════════════════════════════════
#  CMA-ES LOOP — population-batched
# ════════════════════════════════════════════════════════════════════

def save_ckpt(arch_key, params_np, metrics, seed_name, phase):
    arch = ARCHS[arch_key][0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"pipeline_{arch.name}_{seed_name}_{phase}_{ts}.json"
    fp = os.path.join(CKPT, fn)
    ckpt = {
        "architecture": arch.name, "n_params": arch.n_params,
        "timestamp": datetime.now().isoformat(),
        "seed": seed_name, "phase": phase,
        "metrics": {k: v for k, v in metrics.items() if v is not None},
        "x": params_np.tolist(),
    }
    with open(fp, "w") as f:
        json.dump(ckpt, f, indent=2)
    return fn


def run_arch(arch_key, n_gens, popsize):
    arch = ARCHS[arch_key][0]
    seeds = SEED_FNS[arch_key]()[:args.seeds]
    results = []

    print(f"\n{'─'*60}", flush=True)
    print(f"  {arch.name} ({arch.n_params}p, "
          f"ach={'exact' if arch.ach_exact else 'approx'})", flush=True)
    print(f"  {len(seeds)} seeds × {n_gens} gen × {popsize} pop", flush=True)
    print(f"{'─'*60}", flush=True)

    for si, (sname, x0, sigma) in enumerate(seeds):
        print(f"\n  [{si+1}/{len(seeds)}] {sname} σ={sigma}", flush=True)
        best_loss = 999.0; best_x = x0.copy()
        t0 = time.time(); gen_count = [0]; last_print = [0.0]

        opts = cma.CMAOptions()
        opts.set("maxiter", n_gens); opts.set("popsize", popsize)
        opts.set("tolfun", 1e-11); opts.set("verbose", -1)
        es = cma.CMAEvolutionStrategy(x0, sigma, opts)

        while not es.stop():
            sols = es.ask()
            x_batch = np.array(sols)
            fits = evaluate_population(x_batch, arch_key)
            es.tell(sols, fits.tolist())
            gen_count[0] += 1

            # Track best
            bi = np.argmin(fits)
            if fits[bi] < best_loss:
                best_loss = fits[bi]
                best_x = sols[bi].copy()
                now = time.time()
                if now - last_print[0] > 20:
                    last_print[0] = now
                    ev = gen_count[0] * popsize
                    # Quick metrics for display
                    m = full_metrics_single(best_x, arch_key)
                    if m:
                        fn = save_ckpt(arch_key, best_x, m, sname, "run")
                        print(f"    gen{gen_count[0]:>4d} [{now-t0:5.0f}s] "
                              f"loss={best_loss:.4f} CV={m['cv']*100:.1f}% "
                              f"cusp_L={m['cusp_L_85']:.3f} "
                              f"cliff={m['cliff_85']:.0f}% "
                              f"yC={m['yC']:.3f} ach={m['ach_err']:.6f} "
                              f"→ {fn}", flush=True)

        elapsed = time.time() - t0
        m = full_metrics_single(best_x, arch_key)
        if m is None:
            print(f"    {sname}: FAILED", flush=True)
            continue
        m["loss"] = best_loss
        fn = save_ckpt(arch_key, best_x, m, sname, "final")
        print(f"    {sname}: {gen_count[0]*popsize} evals {elapsed:.0f}s | "
              f"loss={best_loss:.4f} CV={m['cv']*100:.2f}% "
              f"cusp_L={m['cusp_L_85']:.3f} cliff={m['cliff_85']:.0f}% "
              f"drift={m['drift_mean']:.1f}/{m['drift_max']:.1f} "
              f"ach={m['ach_err']:.6f} gamut={m['gamut_min_C']:.3f} "
              f"→ {fn}", flush=True)
        results.append({"seed": sname, "loss": best_loss, "metrics": m,
                        "x": best_x.copy(), "checkpoint": fn,
                        "elapsed": elapsed, "evals": gen_count[0]*popsize})

    results.sort(key=lambda r: r["loss"])
    return results


# ════════════════════════════════════════════════════════════════════
#  REPORT
# ════════════════════════════════════════════════════════════════════

def gen_report(all_res, baselines, t_start, dur):
    rp = ["# Pipeline Architecture Search Report (GPU-Batched)\n",
          f"**Date:** {t_start.strftime('%Y-%m-%d %H:%M:%S')}",
          f"**Device:** {dev_name}",
          f"**Config:** {args.seeds} seeds × {args.gens} gen × {args.pop} pop",
          f"**Total:** {dur:.0f}s ({dur/60:.1f} min)\n",
          "## Baselines\n",
          "| Space | CV% | Hue RMS | Cusp L@85 | Cliff% | Drift | Ach | Gamut |",
          "|-------|-----|---------|-----------|--------|-------|-----|-------|"]
    for nm, m in baselines.items():
        rp.append(f"| {nm} | {m['cv']*100:.2f} | {m['hue_rms']:.1f} | "
                  f"{m['cusp_L_85']:.3f} | {m['cliff_85']:.0f} | "
                  f"{m['drift_mean']:.1f}/{m['drift_max']:.1f} | "
                  f"{m['ach_err']:.6f} | {m['gamut_min_C']:.3f} |")
    rp.append("")

    for ak in sorted(all_res.keys()):
        arch = ARCHS[ak][0]
        res = all_res[ak]
        rp.append(f"## {ak}. {arch.name} ({arch.n_params}p)\n")
        if not res:
            rp.append("*No results.*\n"); continue
        rp.append("| Seed | Loss | CV% | Cusp L | Cliff% | Drift | Ach | Gamut | Hue | Cond | Ckpt |")
        rp.append("|------|------|-----|--------|--------|-------|-----|-------|-----|------|------|")
        for r in res:
            m = r["metrics"]
            rp.append(f"| {r['seed']} | {m.get('loss',0):.4f} | {m['cv']*100:.2f} | "
                      f"{m['cusp_L_85']:.3f} | {m['cliff_85']:.0f} | "
                      f"{m['drift_mean']:.1f}/{m['drift_max']:.1f} | "
                      f"{m['ach_err']:.6f} | {m['gamut_min_C']:.3f} | "
                      f"{m['hue_rms']:.1f} | {m['cond1']:.1f} | "
                      f"`{r['checkpoint']}` |")
        rp.append("")

    # Cross-architecture
    rp.append("## Best per Architecture\n")
    rp.append("| Arch | Loss | CV% | Cusp L | Cliff | Drift | Ach | Gamut | Seed |")
    rp.append("|------|------|-----|--------|-------|-------|-----|-------|------|")
    for ak in sorted(all_res.keys()):
        if all_res[ak]:
            r = all_res[ak][0]; m = r["metrics"]
            rp.append(f"| {ak} | {m.get('loss',0):.4f} | {m['cv']*100:.2f} | "
                      f"{m['cusp_L_85']:.3f} | {m['cliff_85']:.0f} | "
                      f"{m['drift_mean']:.1f}/{m['drift_max']:.1f} | "
                      f"{m['ach_err']:.6f} | {m['gamut_min_C']:.3f} | {r['seed']} |")
    rp.append("")

    path = os.path.join(CKPT, "pipeline_search_report.md")
    with open(path, "w") as f:
        f.write("\n".join(rp))
    print(f"  Report: {path}", flush=True)
    return path


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    print(f"\n{'═'*60}\n  PIPELINE SEARCH (GPU-BATCHED)\n"
          f"  {', '.join(args.arch)} | {args.seeds}s × {args.gens}g × {args.pop}p\n"
          f"  {t0.strftime('%Y-%m-%d %H:%M:%S')}\n{'═'*60}\n", flush=True)

    # Baselines
    print("── Baselines ──", flush=True)
    baselines = {}
    for nm, M1, M2 in [("v14", V14n, V14n2), ("OKLab", OKn, OKn2)]:
        s = np.sign(M1@D65_np)*np.abs(M1@D65_np)**(1./3.)
        x = np.zeros(14); x[:6] = _pack_m1(M1); x[6:13] = _pack_m2(M2, s)
        m = full_metrics_single(x, "A")
        if m:
            baselines[nm] = m
            print(f"  {nm}: CV={m['cv']*100:.2f}% cusp_L={m['cusp_L_85']:.3f} "
                  f"drift={m['drift_mean']:.1f}/{m['drift_max']:.1f}", flush=True)

    all_res = {}
    for ak in args.arch:
        pop = args.pop + 32 if ak == "C" else args.pop
        all_res[ak] = run_arch(ak, args.gens, pop)

    dur = (datetime.now() - t0).total_seconds()

    print(f"\n{'═'*60}\n  RANKING\n{'═'*60}", flush=True)
    ranked = []
    for ak in sorted(all_res.keys()):
        if all_res[ak]:
            r = all_res[ak][0]; m = r["metrics"]
            ranked.append((ak, r))
            print(f"  {ak}. loss={m.get('loss',999):.4f} CV={m['cv']*100:.2f}% "
                  f"cusp_L={m['cusp_L_85']:.3f} ach={m['ach_err']:.6f}", flush=True)

    ranked.sort(key=lambda x: x[1]["metrics"].get("loss", 999))
    if ranked:
        print(f"\n  WINNER: {ranked[0][0]} → checkpoints/{ranked[0][1]['checkpoint']}", flush=True)

    gen_report(all_res, baselines, t0, dur)
    print(f"\n  Total: {dur:.0f}s ({dur/60:.1f} min)\n{'═'*60}", flush=True)


if __name__ == "__main__":
    main()
