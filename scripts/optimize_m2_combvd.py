#!/usr/bin/env python3
"""Direct COMBVD Fit: Fixed M1 (physiological) + M2 fit to human data.

Ottosson's approach but with 7x more data:
- M1: FIXED (OKLab's or CIE2006 LMS — not optimized)
- Transfer: cbrt (fixed)
- M2: FIT to COMBVD 3813 human perception pairs
- L_corr: fine-tuning (3 params)
- Loss: STRESS(COMBVD) primary + hue ordering + achromatic + gradient CV secondary

Total free params: M2(9) + L_corr(3) = 12 (vs v7b's 18)
"""

import json, math, os, sys, time, argparse
from datetime import datetime
import numpy as np
import torch
import cma

torch.set_default_dtype(torch.float64)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {dev} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

pa = argparse.ArgumentParser()
pa.add_argument("--gens", type=int, default=300)
pa.add_argument("--pop", type=int, default=128)
pa.add_argument("--seeds", type=int, default=8)
pa.add_argument("--m1", type=str, default="oklab", choices=["oklab", "v7b", "cie2006", "hpe", "m1blend85", "newm1", "newm1v3", "nlm1"],
                help="Which M1 to use (fixed)")
args = pa.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
CKPT = os.path.join(ROOT, "checkpoints")
os.makedirs(CKPT, exist_ok=True)

# ================================================================
#  CONSTANTS
# ================================================================

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

# ================================================================
#  FIXED M1 (not optimized)
# ================================================================

M1_OPTIONS = {
    "oklab": torch.tensor([[0.8189330101, 0.3618667424, -0.1288597137],
                            [0.0329845436, 0.9293118715,  0.0361456387],
                            [0.0482003018, 0.2643662691,  0.6338517070]], device=dev),
    "v7b": torch.tensor([[6.213663274448127, -0.5041794153770129, -0.40416891025666857],
                          [-1.1592256796157883, 4.350194381717271, 0.5254938968299478],
                          [0.0008170122534259527, 0.7226718820884986, 2.227799849833172]], device=dev),
    "cie2006": torch.tensor([[0.4002, 0.7076, -0.0808],
                              [-0.2263, 1.1653, 0.0457],
                              [0.0, 0.0, 0.9182]], device=dev),
    "hpe": torch.tensor([[0.38971, 0.68898, -0.07868],
                          [-0.22981, 1.18340, 0.04641],
                          [0.00000, 0.00000, 1.00000]], device=dev),
    "m1blend85": torch.tensor([[1.6281425, 0.2319601, -0.1701564],
                                [-0.1458467, 1.4424444, 0.1095482],
                                [0.0410926, 0.3331119, 0.8729442]], device=dev),
    "newm1": torch.tensor([[0.99796429, 0.09658433, -0.04143847],
                            [-0.05062582, 0.848038, 0.18375718],
                            [0.11160787, -0.17686994, 0.98343177]], device=dev),
    "nlm1": torch.tensor([[6.213663, -0.504179, -0.404169],
                          [-1.159226, 4.350194, 0.525494],
                          [0.000817, 0.722672, 2.227800]], device=dev),
    "newm1v3": torch.tensor([[0.85315236, 0.39816738, -0.19200711],
                              [0.00457175, 0.74825091, 0.22721983],
                              [0.04609206, 0.07654393, 0.80788272]], device=dev),
}

M1_FIXED = M1_OPTIONS[args.m1]
M1_FIXED_INV = torch.linalg.inv(M1_FIXED)
LMS_D65 = M1_FIXED @ D65
if args.m1 == "nlm1":
    NLM1_D = -0.60
    LMS_D65[0] = LMS_D65[0] + NLM1_D * (1.0 - D65[1]) * D65[2]
    print(f"Nonlinear M1: cross term d={NLM1_D}, (1-Y)*Z@D65 = {(1-D65[1].item())*D65[2].item():.6f}")
LMS_D65_CBRT = LMS_D65.pow(1./3.)
print(f"M1: {args.m1}, LMS@D65={LMS_D65.tolist()}, cbrt={LMS_D65_CBRT.tolist()}", flush=True)

# ================================================================
#  COMBVD DATA
# ================================================================

combvd_loaded = False
for _path in [os.path.join(ROOT, "data", "combvd_pairs.json"),
              os.path.join(os.path.dirname(ROOT), "data", "combvd_pairs.json")]:
    if os.path.exists(_path):
        with open(_path) as f:
            _pairs = json.load(f)
        # Chromatic adaptation: adapt each pair's XYZ to D65
        xyz1_list, xyz2_list, dv_list = [], [], []
        for p in _pairs:
            w = p["white"]
            # Von Kries adaptation to D65
            scale = [D65[i].item() / max(w[i], 1e-10) for i in range(3)]
            xyz1_list.append([p["xyz1"][i] * scale[i] for i in range(3)])
            xyz2_list.append([p["xyz2"][i] * scale[i] for i in range(3)])
            dv_list.append(p["dv"])
        COMBVD_XYZ1 = torch.tensor(xyz1_list, device=dev, dtype=torch.float64)
        COMBVD_XYZ2 = torch.tensor(xyz2_list, device=dev, dtype=torch.float64)
        COMBVD_DV = torch.tensor(dv_list, device=dev, dtype=torch.float64)
        combvd_loaded = True
        print(f"COMBVD: {len(_pairs)} pairs (D65-adapted)", flush=True)
        break

if not combvd_loaded:
    print("ERROR: COMBVD data not found!", flush=True)
    sys.exit(1)

# Pre-compute LMS cbrt for COMBVD pairs (M1 is fixed)
COMBVD_LMS1 = (COMBVD_XYZ1 @ M1_FIXED.T)
if args.m1 == "nlm1":
    cross1 = NLM1_D * (1.0 - COMBVD_XYZ1[:, 1]) * COMBVD_XYZ1[:, 2]
    COMBVD_LMS1[:, 0] = COMBVD_LMS1[:, 0] + cross1
COMBVD_LMS1_CBRT = torch.sign(COMBVD_LMS1) * COMBVD_LMS1.abs().pow(1./3.)
COMBVD_LMS2 = (COMBVD_XYZ2 @ M1_FIXED.T)
if args.m1 == "nlm1":
    cross2 = NLM1_D * (1.0 - COMBVD_XYZ2[:, 1]) * COMBVD_XYZ2[:, 2]
    COMBVD_LMS2[:, 0] = COMBVD_LMS2[:, 0] + cross2
COMBVD_LMS2_CBRT = torch.sign(COMBVD_LMS2) * COMBVD_LMS2.abs().pow(1./3.)

# ================================================================
#  GRADIENT PAIRS
# ================================================================

_use_imported = False
for _try_dir in [os.path.join(ROOT, "colorbench"),
                 os.path.join(ROOT, "space-test-project")]:
    if os.path.isdir(os.path.join(_try_dir, "core")):
        sys.path.insert(0, _try_dir)
        try:
            from core.pairs import generate_all_pairs
            PT, _ = generate_all_pairs(dev)
            _use_imported = True
            print(f"Gradient pairs: {PT.shape[0]}", flush=True)
            break
        except Exception as e:
            print(f"Pairs failed: {e}", flush=True)

if not _use_imported:
    print("ERROR: Could not load gradient pairs!", flush=True)
    sys.exit(1)

N_PAIRS = PT.shape[0]
N_ST = 25
T_ST = torch.linspace(0, 1, N_ST + 1, device=dev)

# Pre-compute LMS cbrt for gradient pairs
PT_LMS1 = (PT[:,0] @ M1_FIXED.T)
if args.m1 == "nlm1":
    pt_cross1 = NLM1_D * (1.0 - PT[:,0,1]) * PT[:,0,2]
    PT_LMS1[:, 0] = PT_LMS1[:, 0] + pt_cross1
PT_LMS1_CBRT = torch.sign(PT_LMS1) * PT_LMS1.abs().pow(1./3.)
PT_LMS2 = (PT[:,1] @ M1_FIXED.T)
if args.m1 == "nlm1":
    pt_cross2 = NLM1_D * (1.0 - PT[:,1,1]) * PT[:,1,2]
    PT_LMS2[:, 0] = PT_LMS2[:, 0] + pt_cross2
PT_LMS2_CBRT = torch.sign(PT_LMS2) * PT_LMS2.abs().pow(1./3.)

# Munsell data
MUNSELL_Y = {1:0.01221,2:0.03126,3:0.06552,4:0.12000,5:0.19770,
             6:0.30049,7:0.43060,8:0.59100,9:0.78660}
MUNSELL_GRAYS = torch.stack([D65 * MUNSELL_Y[v] for v in range(1,10)]).to(dev)
_MG_LMS = MUNSELL_GRAYS @ M1_FIXED.T
if args.m1 == "nlm1":
    mg_cross = NLM1_D * (1.0 - MUNSELL_GRAYS[:, 1]) * MUNSELL_GRAYS[:, 2]
    _MG_LMS[:, 0] = _MG_LMS[:, 0] + mg_cross
MUNSELL_GRAYS_LMS_CBRT = torch.sign(_MG_LMS) * _MG_LMS.abs().pow(1./3.)

# ================================================================
#  FORWARD (M1 fixed, only M2 + L_corr vary)
# ================================================================

def fwd_lms(lms_cbrt, M2, lc):
    """lms_cbrt: (N,3), M2: (P,3,3), lc: (P,3) -> (P,N,3)"""
    lab = lms_cbrt.unsqueeze(0) @ M2.transpose(-1,-2)  # (P,N,3)
    L = lab[..., 0:1]
    c1 = lc[:, 0:1].unsqueeze(1)
    c2 = lc[:, 1:2].unsqueeze(1)
    c3 = lc[:, 2:3].unsqueeze(1)
    t = L * (1.0 - L)
    L_new = L + c1*t + c2*t*(2.0*L - 1.0) + c3*L**2*(1.0-L)**2
    return torch.cat([L_new, lab[..., 1:2], lab[..., 2:3]], dim=-1)

def fwd_xyz(xyz, M2, lc):
    """Full forward: XYZ -> Lab"""
    lms = (xyz.unsqueeze(0) @ M1_FIXED.T).clamp(min=0)
    lms_cbrt = torch.sign(lms) * lms.abs().pow(1./3.)
    return fwd_lms(lms_cbrt.squeeze(0), M2, lc)

# ================================================================
#  UNPACK (12 params: M2(9) + L_corr(3))
# ================================================================

def unpack(x):
    P = x.shape[0]
    M2 = x[:, 0:9].reshape(P, 3, 3)

    lc = torch.stack([x[:,9].clamp(-0.3, 0.3),
                      x[:,10].clamp(-0.3, 0.3),
                      x[:,11].clamp(-0.5, 0.5)], dim=1)

    # Achromatic constraint: M2 @ LMS_D65_cbrt should have a=b near 0
    lab_d65 = (LMS_D65_CBRT.unsqueeze(0) @ M2.transpose(-1,-2)).squeeze(1)  # (P, 3)
    ach_err = (lab_d65[:, 1]**2 + lab_d65[:, 2]**2).sqrt()
    white_L = lab_d65[:, 0]

    # L_corr applied to white
    t_w = white_L * (1.0 - white_L)
    white_L_corr = white_L + lc[:,0]*t_w + lc[:,1]*t_w*(2*white_L-1) + lc[:,2]*white_L**2*(1-white_L)**2

    # Validity
    valid = (white_L_corr > 0.9) & (white_L_corr < 1.1)
    valid &= (ach_err < 0.05)

    # M2 inverse
    M2i = torch.zeros_like(M2)
    det = torch.linalg.det(M2)
    invertible = det.abs() > 1e-10
    valid &= invertible
    good = valid.nonzero(as_tuple=True)[0]
    if good.numel() > 0:
        M2i[good] = torch.linalg.inv(M2[good])

    return M2, M2i, lc, valid, ach_err, white_L_corr

# ================================================================
#  BATCH METRICS
# ================================================================

def batch_stress(M2, lc):
    """COMBVD STRESS — primary objective."""
    lab1 = fwd_lms(COMBVD_LMS1_CBRT, M2, lc)  # (P, 3813, 3)
    lab2 = fwd_lms(COMBVD_LMS2_CBRT, M2, lc)
    dlab = lab2 - lab1
    DE = (dlab**2).sum(dim=-1).sqrt()  # (P, 3813)
    DV = COMBVD_DV.unsqueeze(0)  # (1, 3813)
    F = (DV * DE).sum(dim=1) / (DE**2).sum(dim=1).clamp(min=1e-10)
    residual = DV - F.unsqueeze(1) * DE
    return 100.0 * (residual**2).sum(dim=1).sqrt() / (DV**2).sum(dim=1).sqrt().clamp(min=1e-10)

def batch_cv(M2, M2i, lc):
    """Gradient CV."""
    P = M2.shape[0]
    lab1 = fwd_lms(PT_LMS1_CBRT, M2, lc)
    lab2 = fwd_lms(PT_LMS2_CBRT, M2, lc)
    t = T_ST.view(1,1,-1,1)
    labs = lab1.unsqueeze(2) + t * (lab2-lab1).unsqueeze(2)
    lf = labs.reshape(P, -1, 3)
    # Inverse: undo L_corr, undo M2, cube, undo M1
    L1 = lf[..., 0:1]
    c1 = lc[:, 0:1].unsqueeze(1); c2 = lc[:, 1:2].unsqueeze(1); c3 = lc[:, 2:3].unsqueeze(1)
    L = L1.clone()
    for _ in range(10):
        t_lc = L * (1.0 - L)
        f = L + c1*t_lc + c2*t_lc*(2*L-1) + c3*L**2*(1-L)**2 - L1
        df = 1.0 + c1*(1-2*L) + c2*(6*L**2-6*L+1) + c3*2*L*(1-L)*(1-2*L)
        L = L - f / df.clamp(min=1e-12)
    raw = torch.cat([L, lf[..., 1:2], lf[..., 2:3]], dim=-1)
    lms_c = torch.bmm(raw, M2i.transpose(-1,-2))
    lms = torch.sign(lms_c) * lms_c.abs().pow(3.0)
    xyz = (lms @ M1_FIXED_INV.T)
    lin = (xyz @ MSi.T).clamp(0,1)
    s8 = (l2s(lin)*255).round()/255.0
    xb = s2l(s8) @ MS.T
    r = xb.clamp(min=1e-10) / D65.view(1,1,3)
    f = torch.where(r>0.008856, r.pow(1./3.), 7.787*r+16./116.)
    cl = torch.stack([116*f[...,1]-16, 500*(f[...,0]-f[...,1]), 200*(f[...,1]-f[...,2])], dim=-1)
    cl = cl.reshape(P, N_PAIRS, N_ST+1, 3)
    c1d,c2d = cl[:,:,:-1], cl[:,:,1:]
    dL=c2d[...,0]-c1d[...,0]; C1=(c1d[...,1]**2+c1d[...,2]**2).sqrt()
    C2=(c2d[...,1]**2+c2d[...,2]**2).sqrt(); dC=C2-C1
    dH=((c2d[...,1]-c1d[...,1])**2+(c2d[...,2]-c1d[...,2])**2-dC**2).clamp(min=0).sqrt()
    SL=1+0.015*(c1d[...,0]-50)**2/(20+(c1d[...,0]-50)**2).sqrt(); SC=1+0.045*C1; SH=1+0.015*C1
    de=((dL/SL)**2+(dC/SC)**2+(dH/SH)**2).sqrt()
    md=de.mean(2); sd=de.std(2); ok=md>0.001
    cvs=torch.where(ok, sd/md, torch.zeros_like(md))
    cnt=ok.float().sum(1).clamp(min=1)
    return (cvs*ok.float()).sum(1)/cnt

def batch_munsell_v(M2, lc):
    lab = fwd_lms(MUNSELL_GRAYS_LMS_CBRT, M2, lc)
    L = lab[:,:,0]
    dL = L[:,1:] - L[:,:-1]
    return dL.std(dim=1) / (dL.abs().mean(dim=1) + 1e-10) * 100

def batch_hue_order(M2, lc):
    """Check primary hue ordering: R < Y < G < C < B < M."""
    prim_xyz = torch.stack([MS @ s2l(torch.tensor(c, device=dev, dtype=torch.float64))
                            for c in [[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1]]])
    prim_lms = (prim_xyz @ M1_FIXED.T).clamp(min=0)
    prim_lms_cbrt = torch.sign(prim_lms) * prim_lms.abs().pow(1./3.)
    lab = fwd_lms(prim_lms_cbrt, M2, lc)  # (P, 6, 3)
    h = torch.atan2(lab[:,:,2], lab[:,:,1])  # (P, 6)
    # Hue should increase: R(~0) < Y(~60) < G(~120) < C(~180) < B(~240) < M(~300)
    # Check ordering violations
    dh = h[:,1:] - h[:,:-1]
    # Wrap
    dh = torch.where(dh < -math.pi, dh + 2*math.pi, dh)
    dh = torch.where(dh > math.pi, dh - 2*math.pi, dh)
    violations = (dh <= 0).float().sum(dim=1)
    return violations

def batch_munsell_hue(M2, lc):
    """Munsell Hue spacing uniformity."""
    import colorsys as _cs
    chips_rgb = [(176,103,101),(169,117,82),(155,135,80),(115,143,87),(75,148,115),
                 (58,146,140),(69,138,159),(101,118,162),(132,106,149),(159,99,126)]
    xyzs = []
    for r,g,b in chips_rgb:
        rgb_t = torch.tensor([r/255.,g/255.,b/255.], device=dev, dtype=torch.float64)
        xyzs.append(MS @ s2l(rgb_t))
    xyzs = torch.stack(xyzs)
    lms = xyzs @ M1_FIXED.T
    if args.m1 == "nlm1":
        cross = NLM1_D * (1.0 - xyzs[:, 1]) * xyzs[:, 2]
        lms[:, 0] = lms[:, 0] + cross
    lms_cbrt = torch.sign(lms) * lms.abs().pow(1./3.)
    lab = fwd_lms(lms_cbrt, M2, lc)
    h = torch.atan2(lab[:,:,2], lab[:,:,1])
    h_sorted, _ = torch.sort(h, dim=1)
    dh = h_sorted[:,1:] - h_sorted[:,:-1]
    last = h_sorted[:,0:1] + 2*3.14159 - h_sorted[:,-1:]
    dh = torch.cat([dh, last], dim=1)
    return dh.std(dim=1) / (dh.mean(dim=1) + 1e-10) * 100

def batch_shade_hue(M2, lc):
    """Shade palette hue drift for 6 colors."""
    shade_rgbs = [[0.2,0.4,1.0],[1.0,0.2,0.2],[0.0,0.7,0.2],[1.0,0.5,0.0],[0.6,0.2,0.8],[0.0,0.8,0.9]]
    white_lms = M1_FIXED @ D65
    if args.m1 == "nlm1":
        white_lms[0] = white_lms[0] + NLM1_D * (1.0 - D65[1]) * D65[2]
    white_lc = torch.sign(white_lms) * white_lms.abs().pow(1./3.)
    lab_white = fwd_lms(white_lc.unsqueeze(0), M2, lc)
    
    max_drifts = []
    for rgb in shade_rgbs:
        xyz = MS @ s2l(torch.tensor(rgb, device=dev, dtype=torch.float64))
        lms = M1_FIXED @ xyz
        if args.m1 == "nlm1":
            lms[0] = lms[0] + NLM1_D * (1.0 - xyz[1]) * xyz[2]
        lms_cbrt = torch.sign(lms) * lms.abs().pow(1./3.)
        lab_base = fwd_lms(lms_cbrt.unsqueeze(0), M2, lc)
        h_base = torch.atan2(lab_base[:, 0, 2], lab_base[:, 0, 1])
        
        max_d = torch.zeros(M2.shape[0], device=dev)
        for frac in [0.15, 0.3, 0.5, 0.7]:
            lab_s = lab_base + frac * (lab_white - lab_base)
            h_s = torch.atan2(lab_s[:, 0, 2], lab_s[:, 0, 1])
            C_s = (lab_s[:, 0, 1]**2 + lab_s[:, 0, 2]**2).sqrt()
            dh = torch.atan2(torch.sin(h_s - h_base), torch.cos(h_s - h_base)).abs()
            dh = torch.where(C_s > 0.01, dh, torch.zeros_like(dh))
            max_d = torch.maximum(max_d, dh)
        max_drifts.append(max_d)
    
    return torch.stack(max_drifts).mean(dim=0) * (180.0/3.14159)  # degrees

# ================================================================
#  EVALUATE
# ================================================================

def evaluate(x_np):
    x = torch.tensor(x_np, device=dev, dtype=torch.float64)
    P = x.shape[0]
    losses = torch.full((P,), 999.0, device=dev)

    with torch.no_grad():
        M2, M2i, lc, valid, ach_err, wL = unpack(x)
        if not valid.any():
            return losses.cpu().numpy()

        stress = batch_stress(M2, lc)
        cv = batch_cv(M2, M2i, lc)
        munsell_v = batch_munsell_v(M2, lc)
        hue_viol = batch_hue_order(M2, lc)

        munsell_h = batch_munsell_hue(M2, lc)
        shade_h = batch_shade_hue(M2, lc)

        # PRIMARY: COMBVD STRESS (human perception fit)
        loss = 1.0 * stress

        # SECONDARY: gradient CV
        loss += 2.0 * cv

        # Munsell Value uniformity
        loss += 0.2 * munsell_v

        # Munsell Hue spacing (NEW)
        loss += 0.1 * munsell_h

        # Shade hue drift (NEW)
        loss += 0.5 * shade_h

        # Hue ordering penalty
        loss += hue_viol * 50.0

        # Achromatic penalty
        loss += torch.where(ach_err > 0.001, (ach_err - 0.001) * 200, 0.0)
        loss += torch.where(ach_err > 0.01, (ach_err - 0.01) * 1000, 0.0)

        # White L penalty
        loss += torch.where(wL < 0.95, (0.95 - wL)**2 * 500, 0.0)
        loss += torch.where(wL > 1.05, (wL - 1.05)**2 * 500, 0.0)

        losses = torch.where(valid, loss, torch.full_like(loss, 999.0))

    return losses.cpu().numpy()

# ================================================================
#  SEEDS (only M2 + L_corr, M1 is fixed)
# ================================================================

OKM2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                  [1.9779984951, -2.4285922050, 0.4505937099],
                  [0.0259040371, 0.7827717662, -0.8086757660]])

# IPT M2 (Ebner-Fairchild 1998)
IPT_M2 = np.array([[0.4000, 0.4000, 0.2000],
                    [4.4550, -4.8510, 0.3960],
                    [0.8056, 0.3572, -1.1628]])

def make_seeds():
    seeds = []
    # 1. OKLab M2
    x0 = np.zeros(12)
    x0[0:9] = OKM2.flatten()
    seeds.append(("oklab_m2", x0, 0.03))

    # 2. IPT M2
    x1 = np.zeros(12)
    x1[0:9] = IPT_M2.flatten()
    seeds.append(("ipt_m2", x1, 0.05))

    # 2b. v7b M2
    V7B_M2 = np.array([[0.4675499211910323, 0.20915320090703618, -0.08488334505679182],
                        [0.4843952725673558, -0.3665958307304812, -0.17266206907852755],
                        [-0.04418360083197623, 0.39383739736845824, -0.36863136176600936]])
    x1b = np.zeros(12)
    x1b[0:9] = V7B_M2.flatten()
    seeds.append(("v7b_m2", x1b, 0.03))

    # 3. OKLab M2 + small L_corr
    x2 = x0.copy()
    x2[9:12] = [-0.05, 0.01, 0.1]
    seeds.append(("oklab_lcorr", x2, 0.03))

    # 4-8. Random perturbations of OKLab M2
    rng = np.random.RandomState(42)
    for i in range(max(0, args.seeds - len(seeds))):
        xr = x0.copy()
        xr[0:9] += rng.randn(9) * 0.1
        xr[9:12] = rng.randn(3) * 0.05
        seeds.append((f"rnd{i}", xr, 0.05))

    return seeds

# ================================================================
#  CMA-ES
# ================================================================

def save_ckpt(x_best, loss, seed_label, gen):
    x = torch.tensor(x_best.reshape(1, 12), device=dev)
    M2, M2i, lc, valid, _, wL = unpack(x)
    if not valid.any():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"m2fit_{args.m1}_{seed_label}_{ts}.json"
    path = os.path.join(CKPT, fname)
    out = {
        "M1": M1_FIXED.cpu().tolist(),
        "M2": M2[0].cpu().tolist(),
        "M1_inv": M1_FIXED_INV.cpu().tolist(),
        "M2_inv": M2i[0].cpu().tolist(),
        "gamma": [1/3, 1/3, 1/3],
        "L_corr": lc[0].cpu().tolist(),
        "architecture": f"M2fit_COMBVD_{args.m1}",
        "m1_source": args.m1,
        "loss": float(loss),
        "generation": gen,
        "seed": seed_label,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path

def run_seed(label, x0, sigma):
    print(f"\n{'='*60}")
    print(f"  Seed: {label}, sigma={sigma}")
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
        fits = evaluate(np.array(sols))
        es.tell(sols, fits.tolist())

        idx = np.argmin(fits)
        if fits[idx] < best_loss:
            best_loss = fits[idx]
            best_x = np.array(sols[idx]).copy()

        gen += 1
        if gen % 20 == 0 or gen <= 3:
            x_t = torch.tensor(best_x.reshape(1,12), device=dev)
            M2,M2i,lc,v,ach,wL = unpack(x_t)
            if v.any():
                st = batch_stress(M2, lc).item()
                cv = batch_cv(M2, M2i, lc).item()
                mv = batch_munsell_v(M2, lc).item()
                hv = batch_hue_order(M2, lc).item()
                print(f"  gen {gen:4d}  loss={best_loss:.2f}  STRESS={st:.1f}  CV={cv:.3f}  "
                      f"MunsV={mv:.1f}%  HueViol={hv:.0f}  Ach={ach.item():.4f}  WhiteL={wL.item():.3f}",
                      flush=True)

    path = save_ckpt(best_x, best_loss, label, gen)
    x_t = torch.tensor(best_x.reshape(1,12), device=dev)
    M2,M2i,lc,v,ach,wL = unpack(x_t)
    if v.any():
        st = batch_stress(M2, lc).item()
        cv = batch_cv(M2, M2i, lc).item()
        mv = batch_munsell_v(M2, lc).item()
        print(f"\n  FINAL: loss={best_loss:.4f}")
        print(f"    STRESS={st:.2f}  CV={cv:.4f}  MunsV={mv:.2f}%")
        print(f"    Ach={ach.item():.6f}  WhiteL={wL.item():.4f}", flush=True)
    return best_loss, path

# ================================================================

if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"  Direct COMBVD Fit: M1={args.m1} (fixed) + M2 fit")
    print(f"  12 params (M2=9, L_corr=3), {len(COMBVD_DV)} human pairs")
    print(f"{'#'*60}\n", flush=True)

    seeds = make_seeds()
    results = []
    for label, x0, sigma in seeds:
        loss, path = run_seed(label, x0, sigma)
        results.append((label, loss, path))

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    results.sort(key=lambda r: r[1])
    for label, loss, path in results:
        print(f"  {label:20s}  loss={loss:.4f}  {path or 'FAILED'}")
    print(f"\nBest: {results[0][0]} (loss={results[0][1]:.4f})")
