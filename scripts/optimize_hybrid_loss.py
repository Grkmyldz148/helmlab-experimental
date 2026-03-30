#!/usr/bin/env python3
"""Hybrid Loss: v7b pipeline + combined objective (hue + COMBVD + Munsell + CV).

Same 18-param pipeline as v7b (cbrt + L_corr), but four-objective loss:
  - Hue linearity (from v7b's strength)
  - COMBVD STRESS (from perceptual model's strength)
  - Munsell Value/Hue uniformity (from perceptual model's strength)
  - Gradient CV (baseline quality)

Goal: combine v7b's hue 3.5 deg with Perceptual's Munsell 0.53%.
"""

import json, math, os, sys, time, argparse
from datetime import datetime
import numpy as np
import torch
import cma

torch.set_default_dtype(torch.float64)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev_name = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
print(f"Device: {dev_name}", flush=True)

pa = argparse.ArgumentParser()
pa.add_argument("--gens", type=int, default=300)
pa.add_argument("--pop", type=int, default=128)
pa.add_argument("--seeds", type=int, default=8)
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
#  GRADIENT PAIRS (from colorbench)
# ================================================================

_use_imported = False
for _try_dir in [os.path.join(ROOT, "colorbench"),
                 os.path.join(ROOT, "space-test-project")]:
    if os.path.isdir(os.path.join(_try_dir, "core")):
        sys.path.insert(0, _try_dir)
        try:
            from core.pairs import generate_all_pairs
            PT, _labels = generate_all_pairs(dev)
            _use_imported = True
            print(f"Loaded {PT.shape[0]} gradient pairs", flush=True)
            break
        except Exception as e:
            print(f"Pair import failed: {e}", flush=True)

if not _use_imported:
    print("ERROR: Could not load gradient pairs!", flush=True)
    sys.exit(1)

N_PAIRS = PT.shape[0]
N_ST = 25
T_ST = torch.linspace(0, 1, N_ST + 1, device=dev)

# ================================================================
#  COMBVD PERCEPTUAL DATA
# ================================================================

combvd_loaded = False
for _try_path in [os.path.join(ROOT, "data", "combvd_pairs.json"),
                  os.path.join(os.path.dirname(SCRIPT_DIR), "data", "combvd_pairs.json")]:
    if os.path.exists(_try_path):
        with open(_try_path) as f:
            _combvd = json.load(f)
        COMBVD_XYZ1 = torch.tensor([p["xyz1"] for p in _combvd], device=dev, dtype=torch.float64)
        COMBVD_XYZ2 = torch.tensor([p["xyz2"] for p in _combvd], device=dev, dtype=torch.float64)
        COMBVD_DV = torch.tensor([p["dv"] for p in _combvd], device=dev, dtype=torch.float64)
        combvd_loaded = True
        print(f"Loaded {len(_combvd)} COMBVD pairs", flush=True)
        break

if not combvd_loaded:
    print("WARNING: COMBVD not found, using gradient-only loss", flush=True)

# ================================================================
#  MUNSELL DATA
# ================================================================

MUNSELL_VALUE_Y = {1:0.01221,2:0.03126,3:0.06552,4:0.12000,5:0.19770,
                   6:0.30049,7:0.43060,8:0.59100,9:0.78660}
MUNSELL_GRAYS = torch.stack([D65 * MUNSELL_VALUE_Y[v] for v in range(1,10)]).to(dev)

MUNSELL_HUE_CHIPS = {
    '5R':(176,103,101),'5YR':(169,117,82),'5Y':(155,135,80),
    '5GY':(115,143,87),'5G':(75,148,115),'5BG':(58,146,140),
    '5B':(69,138,159),'5PB':(101,118,162),'5P':(132,106,149),'5RP':(159,99,126)}
_hue_xyzs = []
for name, (r,g,b) in MUNSELL_HUE_CHIPS.items():
    rgb = torch.tensor([r/255., g/255., b/255.], device=dev, dtype=torch.float64)
    _hue_xyzs.append(MS @ s2l(rgb))
MUNSELL_HUE_XYZ = torch.stack(_hue_xyzs)

# Primary XYZ for hue linearity
def _hsv_to_rgb(h, s, v):
    if s == 0: return v, v, v
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v*(1-s); q = v*(1-s*f); t = v*(1-s*(1-f))
    return [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][i]

PRIM_XYZ = torch.stack([MS @ s2l(torch.tensor([1,0,0.],device=dev)),
                         MS @ s2l(torch.tensor([0,1,0.],device=dev)),
                         MS @ s2l(torch.tensor([0,0,1.],device=dev)),
                         MS @ s2l(torch.tensor([1,1,0.],device=dev)),
                         MS @ s2l(torch.tensor([0,1,1.],device=dev)),
                         MS @ s2l(torch.tensor([1,0,1.],device=dev))])
WHITE_XYZ = D65.unsqueeze(0)

# ================================================================
#  FORWARD / INVERSE (batched over P candidates)
# ================================================================

def fwd(xyz, M1, M2, lc):
    """xyz: (N,3), M1/M2: (P,3,3), lc: (P,3) -> (P,N,3)"""
    lms = (xyz.unsqueeze(0) @ M1.transpose(-1,-2)).clamp(min=0)
    lms_c = torch.sign(lms) * lms.abs().pow(1./3.)
    lab = torch.bmm(lms_c, M2.transpose(-1,-2))
    # L_corr: Bernstein basis
    L = lab[..., 0:1]
    c1 = lc[:, 0:1].unsqueeze(1)
    c2 = lc[:, 1:2].unsqueeze(1)
    c3 = lc[:, 2:3].unsqueeze(1)
    t = L * (1.0 - L)
    L_new = L + c1*t + c2*t*(2.0*L - 1.0) + c3*L**2*(1.0-L)**2
    return torch.cat([L_new, lab[..., 1:2], lab[..., 2:3]], dim=-1)

def inv(lab, M1i, M2i, lc):
    """lab: (P,N,3) -> (P,N,3) XYZ"""
    L1 = lab[..., 0:1]
    c1 = lc[:, 0:1].unsqueeze(1)
    c2 = lc[:, 1:2].unsqueeze(1)
    c3 = lc[:, 2:3].unsqueeze(1)
    L = L1.clone()
    for _ in range(12):
        t = L * (1.0 - L)
        f = L + c1*t + c2*t*(2.0*L-1.0) + c3*L**2*(1.0-L)**2 - L1
        df = 1.0 + c1*(1.0-2.0*L) + c2*(6.0*L**2-6.0*L+1.0) + c3*2.0*L*(1.0-L)*(1.0-2.0*L)
        L = L - f / df.clamp(min=1e-12)
    raw = torch.cat([L, lab[..., 1:2], lab[..., 2:3]], dim=-1)
    lms_c = torch.bmm(raw, M2i.transpose(-1,-2))
    lms = torch.sign(lms_c) * lms_c.abs().pow(3.0)
    return torch.bmm(lms, M1i.transpose(-1,-2))

# ================================================================
#  PARAMETER PACKING (18 params: M1(6) + M2(9) + L_corr(3))
# ================================================================

def unpack(x):
    P = x.shape[0]
    M1 = torch.zeros(P, 3, 3, device=dev)
    M1[:,0,0] = x[:,0]; M1[:,0,1] = x[:,1]
    M1[:,1,0] = x[:,2]; M1[:,1,1] = x[:,3]
    M1[:,2,0] = x[:,4]; M1[:,2,1] = x[:,5]
    M1[:,0,2] = (1.0 - M1[:,0,0]*D65[0] - M1[:,0,1]*D65[1]) / D65[2]
    M1[:,1,2] = (1.0 - M1[:,1,0]*D65[0] - M1[:,1,1]*D65[1]) / D65[2]
    M1[:,2,2] = (1.0 - M1[:,2,0]*D65[0] - M1[:,2,1]*D65[1]) / D65[2]
    lms_d65 = (D65.unsqueeze(0).unsqueeze(0) @ M1.transpose(-1,-2)).squeeze(1)
    valid = (lms_d65 > 0.01).all(dim=1)

    M2 = torch.zeros(P, 3, 3, device=dev)
    M2[:,0,0]=x[:,6]; M2[:,0,1]=x[:,7]; M2[:,0,2]=x[:,8]
    M2[:,1,0]=x[:,9]; M2[:,1,1]=x[:,10]; M2[:,1,2]=x[:,11]
    M2[:,2,0]=x[:,12]; M2[:,2,1]=x[:,13]; M2[:,2,2]=x[:,14]

    lc = torch.stack([x[:,15].clamp(-0.5,0.5),
                      x[:,16].clamp(-0.5,0.5),
                      x[:,17].clamp(-1.0,1.0)], dim=1)

    # Structural achromatic: M2 @ cbrt(M1 @ D65) should have a=b=0
    lms_c_d65 = lms_d65.pow(1./3.)
    lab_d65 = torch.bmm(lms_c_d65.unsqueeze(1), M2.transpose(-1,-2)).squeeze(1)
    ach_err = (lab_d65[:, 1]**2 + lab_d65[:, 2]**2).sqrt()

    M1i = torch.zeros_like(M1); M2i = torch.zeros_like(M2)
    det1 = torch.linalg.det(M1); det2 = torch.linalg.det(M2)
    invertible = (det1.abs() > 1e-10) & (det2.abs() > 1e-10) & valid
    good = invertible.nonzero(as_tuple=True)[0]
    if good.numel() > 0:
        M1i[good] = torch.linalg.inv(M1[good])
        M2i[good] = torch.linalg.inv(M2[good])
    valid = invertible

    return M1, M2, M1i, M2i, lc, valid, ach_err

# ================================================================
#  BATCH METRICS
# ================================================================

def batch_cv(M1, M2, M1i, M2i, lc):
    P = M1.shape[0]
    lab1 = fwd(PT[:,0], M1, M2, lc)
    lab2 = fwd(PT[:,1], M1, M2, lc)
    t = T_ST.view(1,1,-1,1)
    labs = lab1.unsqueeze(2) + t * (lab2-lab1).unsqueeze(2)
    lf = labs.reshape(P, -1, 3)
    xyz = inv(lf, M1i, M2i, lc)
    lin = (xyz @ MSi.T).clamp(0,1)
    s8 = (l2s(lin)*255).round()/255.0
    xb = s2l(s8) @ MS.T
    r = xb.clamp(min=1e-10) / D65.view(1,1,3)
    f = torch.where(r>0.008856, r.pow(1./3.), 7.787*r+16./116.)
    cl = torch.stack([116*f[...,1]-16, 500*(f[...,0]-f[...,1]), 200*(f[...,1]-f[...,2])], dim=-1)
    cl = cl.reshape(P, N_PAIRS, N_ST+1, 3)
    c1,c2 = cl[:,:,:-1], cl[:,:,1:]
    dL=c2[...,0]-c1[...,0]; C1=(c1[...,1]**2+c1[...,2]**2).sqrt(); C2=(c2[...,1]**2+c2[...,2]**2).sqrt()
    dC=C2-C1; dH=((c2[...,1]-c1[...,1])**2+(c2[...,2]-c1[...,2])**2-dC**2).clamp(min=0).sqrt()
    SL=1+0.015*(c1[...,0]-50)**2/(20+(c1[...,0]-50)**2).sqrt(); SC=1+0.045*C1; SH=1+0.015*C1
    de=((dL/SL)**2+(dC/SC)**2+(dH/SH)**2).sqrt()
    md=de.mean(2); sd=de.std(2); ok=md>0.001
    cvs=torch.where(ok, sd/md, torch.zeros_like(md))
    cnt=ok.float().sum(1).clamp(min=1)
    return (cvs*ok.float()).sum(1)/cnt

def batch_stress(M1, M2, lc):
    if not combvd_loaded:
        return torch.zeros(M1.shape[0], device=dev)
    lab1 = fwd(COMBVD_XYZ1, M1, M2, lc)
    lab2 = fwd(COMBVD_XYZ2, M1, M2, lc)
    dlab = lab2 - lab1
    DE = (dlab**2).sum(dim=-1).sqrt()
    DV = COMBVD_DV.unsqueeze(0).expand_as(DE)
    F = (DV * DE).sum(dim=1) / (DE**2).sum(dim=1).clamp(min=1e-10)
    residual = DV - F.unsqueeze(1) * DE
    stress = 100.0 * (residual**2).sum(dim=1).sqrt() / (DV**2).sum(dim=1).sqrt().clamp(min=1e-10)
    return stress

def batch_munsell_v(M1, M2, lc):
    lab = fwd(MUNSELL_GRAYS, M1, M2, lc)
    L = lab[:,:,0]
    dL = L[:,1:] - L[:,:-1]
    return dL.std(dim=1) / (dL.abs().mean(dim=1) + 1e-10) * 100

def batch_munsell_h(M1, M2, lc):
    lab = fwd(MUNSELL_HUE_XYZ, M1, M2, lc)
    h = torch.atan2(lab[:,:,2], lab[:,:,1])
    h_sorted, _ = torch.sort(h, dim=1)
    dh = h_sorted[:,1:] - h_sorted[:,:-1]
    last = h_sorted[:,0:1] + 2*math.pi - h_sorted[:,-1:]
    dh = torch.cat([dh, last], dim=1)
    return dh.std(dim=1) / (dh.mean(dim=1) + 1e-10) * 100

def batch_hue_lin(M1, M2, lc):
    P = M1.shape[0]
    n_steps = 11
    t = torch.linspace(0,1,n_steps,device=dev).view(1,1,-1,1)
    lab_p = fwd(PRIM_XYZ, M1, M2, lc)
    lab_w = fwd(WHITE_XYZ.expand(6,3), M1, M2, lc)
    labs = lab_p.unsqueeze(2) + t*(lab_w.unsqueeze(2)-lab_p.unsqueeze(2))
    h = torch.atan2(labs[...,2], labs[...,1])
    h_s = h[:,:,0:1]; h_e = h[:,:,-1:]
    dh = h_e - h_s
    dh = torch.where(dh>math.pi, dh-2*math.pi, dh)
    dh = torch.where(dh<-math.pi, dh+2*math.pi, dh)
    t_lin = torch.linspace(0,1,n_steps,device=dev).view(1,1,-1)
    h_exp = h_s + t_lin * dh
    C = (labs[...,1]**2 + labs[...,2]**2).sqrt()
    h_diff = h - h_exp
    h_diff = torch.where(h_diff>math.pi, h_diff-2*math.pi, h_diff)
    h_diff = torch.where(h_diff<-math.pi, h_diff+2*math.pi, h_diff)
    mask = C > 0.01
    h_diff_m = h_diff * mask.float()
    count = mask.float().sum(dim=(1,2)).clamp(min=1)
    return (h_diff_m**2).sum(dim=(1,2)).sqrt() / count.sqrt() * (180./math.pi)

def batch_mono(M1, M2, lc):
    t = torch.linspace(0.001, 0.999, 64, device=dev)
    grays = D65.unsqueeze(0) * t.unsqueeze(1)
    lab = fwd(grays, M1, M2, lc)
    L = lab[:,:,0]
    dL = L[:,1:] - L[:,:-1]
    return (dL > 0).float().mean(dim=1)

def batch_ach(M1, M2, lc):
    t = torch.linspace(0.01, 0.99, 32, device=dev)
    grays = D65.unsqueeze(0) * t.unsqueeze(1)
    lab = fwd(grays, M1, M2, lc)
    return (lab[:,:,1]**2 + lab[:,:,2]**2).sqrt().max(dim=1).values

def batch_info(M1, M2, M1i, M2i, lc):
    lab_p = fwd(PRIM_XYZ, M1, M2, lc)
    lab_w = fwd(D65.unsqueeze(0), M1, M2, lc)
    yC = (lab_p[:,3,1]**2 + lab_p[:,3,2]**2).sqrt()
    bL = lab_p[:,2,0]
    # B-W midpoint
    lab_mid = 0.5*(lab_p[:,2:3,:] + lab_w)
    xyz_mid = inv(lab_mid, M1i, M2i, lc)
    srgb_mid = l2s((xyz_mid @ MSi.T).clamp(0,1))
    bw_gr = srgb_mid[:,0,1] / srgb_mid[:,0,0].clamp(min=1e-10)
    # R-W midpoint
    lab_mid_r = 0.5*(lab_p[:,0:1,:] + lab_w)
    xyz_mid_r = inv(lab_mid_r, M1i, M2i, lc)
    srgb_mid_r = l2s((xyz_mid_r @ MSi.T).clamp(0,1))
    rw_gb = srgb_mid_r[:,0,1] - srgb_mid_r[:,0,2]
    wL = lab_w[:,0,0]
    return yC, bL, bw_gr.squeeze(), rw_gb.squeeze(), wL

# ================================================================
#  EVALUATE
# ================================================================

def evaluate(x_np):
    x = torch.tensor(x_np, device=dev, dtype=torch.float64)
    P = x.shape[0]
    losses = torch.full((P,), 999.0, device=dev)

    with torch.no_grad():
        M1, M2, M1i, M2i, lc, valid, ach_err = unpack(x)
        if not valid.any():
            return losses.cpu().numpy()

        yC, bL, bw, rw, wL = batch_info(M1, M2, M1i, M2i, lc)
        valid &= yC > 0.03
        valid &= wL > 0.85  # White L MUST be near 1.0

        if not valid.any():
            return losses.cpu().numpy()

        cv = batch_cv(M1, M2, M1i, M2i, lc)
        stress = batch_stress(M1, M2, lc)
        munsell_v = batch_munsell_v(M1, M2, lc)
        munsell_h = batch_munsell_h(M1, M2, lc)
        hue_lin = batch_hue_lin(M1, M2, lc)
        mono = batch_mono(M1, M2, lc)

        # ======== HYBRID LOSS ========
        # Four objectives balanced:
        loss = torch.zeros(P, device=dev)

        # 1. Gradient CV (weight 3.0) — baseline quality
        loss += 3.0 * cv

        # 2. COMBVD STRESS (weight 0.3) — perceptual accuracy
        if combvd_loaded:
            loss += 0.3 * stress

        # 3. Munsell uniformity (weight 0.3 each)
        loss += 0.3 * munsell_v
        loss += 0.15 * munsell_h

        # 4. Hue linearity (weight 1.5) — v7b's strength
        loss += 1.5 * hue_lin

        # Monotonicity
        mono_pen = (1.0 - mono).clamp(min=0) * 500
        loss += mono_pen

        # Soft penalties
        pen = torch.zeros(P, device=dev)
        pen += torch.where(ach_err > 0.001, (ach_err - 0.001) * 100, 0.0)
        pen += torch.where(ach_err > 0.01, (ach_err - 0.01) * 500, 0.0)
        pen += torch.where(bL > 0.55, (bL - 0.55)**2 * 50, 0.0)
        pen += torch.where(bw < 1.0, (1.0 - bw)**2 * 50, 0.0)
        pen += torch.where(rw > 0.1, (rw - 0.1)**2 * 100, 0.0)
        pen += torch.where(rw < -0.05, (rw + 0.05)**2 * 200, 0.0)
        pen += torch.where(yC < 0.08, (0.08 - yC)**2 * 200, 0.0)
        loss += pen

        losses = torch.where(valid, loss, torch.full_like(loss, 999.0))

    return losses.cpu().numpy()

# ================================================================
#  SEEDS
# ================================================================

OKM1 = np.array([[0.8189330101,0.3618667424,-0.1288597137],
                  [0.0329845436,0.9293118715,0.0361456387],
                  [0.0482003018,0.2643662691,0.6338517070]])
OKM2 = np.array([[0.2104542553,0.7936177850,-0.0040720468],
                  [1.9779984951,-2.4285922050,0.4505937099],
                  [0.0259040371,0.7827717662,-0.8086757660]])

def pack_m1(M1):
    return [M1[0,0],M1[0,1],M1[1,0],M1[1,1],M1[2,0],M1[2,1]]

def make_seeds():
    seeds = []
    # 1. OKLab seed (no L_corr)
    x0 = np.zeros(18)
    x0[:6] = pack_m1(OKM1)
    x0[6:15] = OKM2.flatten()
    seeds.append(("oklab", x0, 0.03))

    # 2. v7b seed (with its L_corr)
    v7b_path = os.path.join(CKPT, "v7b_nodelta.json")
    if os.path.exists(v7b_path):
        with open(v7b_path) as f:
            v7b = json.load(f)
        xv = np.zeros(18)
        M1v = np.array(v7b["M1"])
        M2v = np.array(v7b["M2"])
        xv[:6] = pack_m1(M1v)
        xv[6:15] = M2v.flatten()
        lc = v7b.get("L_corr", [0,0,0])
        if isinstance(lc, list) and len(lc) == 3:
            xv[15:18] = lc
        else:
            xv[15] = v7b.get("L_corr_p1", 0)
            xv[16] = v7b.get("L_corr_p2", 0)
            xv[17] = v7b.get("L_corr_p3", 0)
        seeds.append(("v7b", xv, 0.02))

    # 3. Perceptual seed
    perc_path = os.path.join(CKPT, "perceptual_rnd0_20260324_080852.json")
    if os.path.exists(perc_path):
        with open(perc_path) as f:
            perc = json.load(f)
        xp = np.zeros(18)
        xp[:6] = pack_m1(np.array(perc["M1"]))
        xp[6:15] = np.array(perc["M2"]).flatten()
        lc = perc.get("L_corr", [0,0,0])
        xp[15:18] = lc
        seeds.append(("perceptual", xp, 0.02))

    # 4. Midpoint of v7b and perceptual
    if len(seeds) >= 3:
        xmid = 0.5 * seeds[1][1] + 0.5 * seeds[2][1]
        seeds.append(("v7b_perc_mid", xmid, 0.03))

    # 5-8. Random perturbations
    rng = np.random.RandomState(42)
    for i in range(max(0, args.seeds - len(seeds))):
        base = seeds[1][1] if len(seeds) > 1 else seeds[0][1]
        xr = base + rng.randn(18) * 0.03
        seeds.append((f"rnd{i}", xr, 0.04))

    return seeds

# ================================================================
#  CMA-ES
# ================================================================

def save_ckpt(x_best, loss, seed_label, gen):
    x = torch.tensor(x_best.reshape(1,18), device=dev)
    M1,M2,M1i,M2i,lc,valid,_ = unpack(x)
    if not valid.any():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"hybrid_{seed_label}_{ts}.json"
    path = os.path.join(CKPT, fname)
    out = {
        "M1": M1[0].cpu().tolist(),
        "M2": M2[0].cpu().tolist(),
        "M1_inv": M1i[0].cpu().tolist(),
        "M2_inv": M2i[0].cpu().tolist(),
        "gamma": [1/3, 1/3, 1/3],
        "L_corr": lc[0].cpu().tolist(),
        "architecture": "GenSpace_HybridLoss",
        "loss": float(loss), "generation": gen, "seed": seed_label,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path

def run_seed(label, x0, sigma):
    print(f"\n{'='*60}")
    print(f"  Seed: {label}, sigma={sigma}, pop={args.pop}, gens={args.gens}")
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
            x_t = torch.tensor(best_x.reshape(1,18), device=dev)
            M1,M2,M1i,M2i,lc,v,_ = unpack(x_t)
            if v.any():
                cv_v = batch_cv(M1,M2,M1i,M2i,lc).item()
                st_v = batch_stress(M1,M2,lc).item() if combvd_loaded else 0
                mv_v = batch_munsell_v(M1,M2,lc).item()
                mh_v = batch_munsell_h(M1,M2,lc).item()
                hl_v = batch_hue_lin(M1,M2,lc).item()
                mo_v = batch_mono(M1,M2,lc).item()
                print(f"  gen {gen:4d}  loss={best_loss:.2f}  CV={cv_v:.1f}%  STRESS={st_v:.1f}  "
                      f"MunsV={mv_v:.1f}%  MunsH={mh_v:.1f}%  HueLin={hl_v:.1f} deg  Mono={mo_v:.4f}",
                      flush=True)

    path = save_ckpt(best_x, best_loss, label, gen)
    x_t = torch.tensor(best_x.reshape(1,18), device=dev)
    M1,M2,M1i,M2i,lc,v,_ = unpack(x_t)
    if v.any():
        cv_v = batch_cv(M1,M2,M1i,M2i,lc).item()
        st_v = batch_stress(M1,M2,lc).item() if combvd_loaded else 0
        mv_v = batch_munsell_v(M1,M2,lc).item()
        mh_v = batch_munsell_h(M1,M2,lc).item()
        hl_v = batch_hue_lin(M1,M2,lc).item()
        yC,bL,bw,rw,wL = batch_info(M1,M2,M1i,M2i,lc)
        print(f"\n  FINAL: loss={best_loss:.4f}")
        print(f"    CV={cv_v:.2f}%  STRESS={st_v:.2f}  MunsV={mv_v:.2f}%  MunsH={mh_v:.2f}%")
        print(f"    HueLin={hl_v:.2f} deg  Mono={batch_mono(M1,M2,lc).item():.4f}")
        print(f"    YellowC={yC.item():.3f}  BlueL={bL.item():.3f}  WhiteL={wL.item():.3f}")
        print(f"    B-W G/R={bw.item():.3f}  R-W G-B={rw.item():.3f}", flush=True)
    return best_loss, path

# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"  Hybrid Loss: Hue + COMBVD + Munsell + CV")
    print(f"  Pipeline: cbrt + L_corr (18 params, same as v7b)")
    print(f"  Gens={args.gens}, Pop={args.pop}, Seeds={args.seeds}")
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
