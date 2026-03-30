#!/usr/bin/env python3
"""Two-Phase Adam Optimizer — differentiable, gradient-based.

Phase 1: M1 fixed (OKLab), optimize M2+k+L_corr with Adam (13 params)
Phase 2: M1 also free, fine-tune all 19 params with small lr

Loss: midpoint_quality + gradient_CV + munsell_V + achromatic
All differentiable through torch autograd.
"""

import json, math, os, sys, time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_default_dtype(torch.float64)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {dev}", flush=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
CKPT = os.path.join(ROOT, "checkpoints")
os.makedirs(CKPT, exist_ok=True)

D65 = torch.tensor([0.95047, 1.0, 1.08883], device=dev)
MS = torch.tensor([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],
                    [.0193339,.1191920,.9503041]], device=dev)
MSi = torch.linalg.inv(MS)

def s2l(c): return torch.where(c<=0.04045, c/12.92, ((c+0.055)/1.055).pow(2.4))
def l2s(c): return torch.where(c<=0.0031308, c*12.92, 1.055*c.clamp(min=1e-12).pow(1./2.4)-0.055)

# ================================================================
#  DATA
# ================================================================

# Midpoint pairs
import colorsys
_mp = []
for h in range(0,360,30):
    r1,g1,b1 = colorsys.hsv_to_rgb(h/360,1,1)
    r2,g2,b2 = colorsys.hsv_to_rgb(((h+180)%360)/360,1,1)
    _mp.append(([r1,g1,b1],[r2,g2,b2]))
for rgb in [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]]:
    _mp.append((rgb,[1,1,1]))
_mp.append(([1,0.6,0.2],[0.2,0.4,1]))
_mp.append(([1,0.5,0.5],[0.5,1,1]))
_mp.append(([0.6,0.2,0.8],[0.9,0.8,0.2]))

MID_XYZ1 = torch.stack([MS@s2l(torch.tensor(r,device=dev,dtype=torch.float64)) for r,_ in _mp])
MID_XYZ2 = torch.stack([MS@s2l(torch.tensor(r,device=dev,dtype=torch.float64)) for _,r in _mp])

def _cielab(xyz):
    r=xyz/D65; d3=(6./29.)**3
    f=torch.where(r>d3, r.pow(1./3.), r/(3*(6./29.)**2)+4./29.)
    return torch.stack([116*f[...,1]-16, 500*(f[...,0]-f[...,1]), 200*(f[...,1]-f[...,2])], dim=-1)

CL1=_cielab(MID_XYZ1); CL2=_cielab(MID_XYZ2)
C_END1=(CL1[:,1]**2+CL1[:,2]**2).sqrt(); C_END2=(CL2[:,1]**2+CL2[:,2]**2).sqrt()
C_END_AVG=0.5*(C_END1+C_END2)

# Gradient pairs
for _d in [os.path.join(ROOT,"colorbench"), os.path.join(ROOT,"space-test-project")]:
    if os.path.isdir(os.path.join(_d,"core")):
        sys.path.insert(0,_d)
        from core.pairs import generate_all_pairs
        PT,_ = generate_all_pairs(dev)
        print(f"Pairs: {PT.shape[0]}", flush=True)
        break
N_PAIRS=PT.shape[0]; N_ST=25; T_ST=torch.linspace(0,1,N_ST+1,device=dev)

# Munsell
MY={1:0.01221,2:0.03126,3:0.06552,4:0.12,5:0.1977,6:0.30049,7:0.4306,8:0.591,9:0.7866}
MUNSELL_GRAYS=torch.stack([D65*MY[v] for v in range(1,10)]).to(dev)

# ================================================================
#  MODEL (differentiable)
# ================================================================

class HybridColorSpace(nn.Module):
    def __init__(self):
        super().__init__()
        # M1: OKLab as init (6 free params, col3 = D65 constrained)
        M1_ok = torch.tensor([[0.818933,0.361867],
                               [0.032985,0.929312],
                               [0.048200,0.264366]], device=dev)
        self.m1_free = nn.Parameter(M1_ok)

        # M2: OKLab as init (9 free params)
        M2_ok = torch.tensor([[0.2104542553,0.7936177850,-0.0040720468],
                               [1.9779984951,-2.4285922050,0.4505937099],
                               [0.0259040371,0.7827717662,-0.8086757660]], device=dev)
        self.m2 = nn.Parameter(M2_ok)

        # Transfer param k (log part)
        self.log_k = nn.Parameter(torch.tensor(1.0, device=dev))  # k = exp(log_k), range ~1-5

        # Blend weight w
        self.logit_w = nn.Parameter(torch.tensor(-0.5, device=dev))  # w = sigmoid(logit_w), start ~0.38

        # L_corr
        self.lc = nn.Parameter(torch.zeros(3, device=dev))

    @property
    def M1(self):
        m = torch.zeros(3, 3, device=dev, dtype=torch.float64)
        m[:, :2] = self.m1_free
        m[:, 2] = (1.0 - m[:, 0]*D65[0] - m[:, 1]*D65[1]) / D65[2]
        return m

    @property
    def k(self):
        return self.log_k.exp().clamp(0.5, 8.0)

    @property
    def w(self):
        return torch.sigmoid(self.logit_w).clamp(0.05, 0.6)

    def transfer(self, x):
        k = self.k; w = self.w
        cbrt = torch.sign(x) * x.abs().clamp(min=1e-30).pow(1./3.)
        log = torch.log1p(k * x.clamp(min=0)) / torch.log1p(k)
        return (1-w) * cbrt + w * log

    def forward_lab(self, xyz):
        M1 = self.M1; M2 = self.m2
        lms = (xyz @ M1.T).clamp(min=0)
        lms_c = self.transfer(lms)

        # Structural achromatic: project M2 a,b rows (no in-place ops for autograd)
        lms_c_d65 = self.transfer((D65 @ M1.T).clamp(min=0).unsqueeze(0)).squeeze(0)
        norm2 = (lms_c_d65 * lms_c_d65).sum()
        proj1 = (M2[1] * lms_c_d65).sum() / norm2
        proj2 = (M2[2] * lms_c_d65).sum() / norm2
        row0 = M2[0]
        row1 = M2[1] - proj1 * lms_c_d65
        row2 = M2[2] - proj2 * lms_c_d65
        M2_fixed = torch.stack([row0, row1, row2])

        lab = lms_c @ M2_fixed.T
        L = lab[:, 0:1]
        c1, c2, c3 = self.lc[0], self.lc[1], self.lc[2]
        t = L * (1 - L)
        L_new = L + c1*t + c2*t*(2*L-1) + c3*L**2*(1-L)**2
        return torch.cat([L_new, lab[:, 1:2], lab[:, 2:3]], dim=1)

    def compute_loss(self):
        # 1. Midpoint quality (chroma + hue)
        lab1 = self.forward_lab(MID_XYZ1)
        lab2 = self.forward_lab(MID_XYZ2)
        lab_mid = 0.5 * (lab1 + lab2)

        # CIE Lab of midpoint (approximate via forward)
        # Use space's own chroma as proxy (faster than full CIE Lab inverse)
        C_mid = (lab_mid[:, 1]**2 + lab_mid[:, 2]**2).sqrt()
        C1 = (lab1[:, 1]**2 + lab1[:, 2]**2).sqrt()
        C2 = (lab2[:, 1]**2 + lab2[:, 2]**2).sqrt()
        C_avg = 0.5 * (C1 + C2)
        mask = C_avg > 0.01
        chroma_ratio = torch.where(mask, C_mid / C_avg.clamp(min=0.001), torch.ones_like(C_mid))
        chroma_loss = (1.0 - chroma_ratio).clamp(min=0).mean()

        # Hue preservation
        h_mid = torch.atan2(lab_mid[:, 2], lab_mid[:, 1])
        h1 = torch.atan2(lab1[:, 2], lab1[:, 1])
        h2 = torch.atan2(lab2[:, 2], lab2[:, 1])
        dh = h2 - h1
        dh = torch.where(dh > math.pi, dh - 2*math.pi, dh)
        dh = torch.where(dh < -math.pi, dh + 2*math.pi, dh)
        h_expected = h1 + 0.5 * dh
        h_err = torch.atan2(torch.sin(h_mid - h_expected), torch.cos(h_mid - h_expected)).abs()
        hue_loss = torch.where(C_mid > 0.01, h_err, torch.zeros_like(h_err)).mean()

        # 2. Gradient CV (simplified — use space's own Lab, not CIEDE2000)
        # Subsample pairs for speed (full 3038 is too slow for gradient)
        idx = torch.randperm(N_PAIRS, device=dev)[:200]
        lab_s = self.forward_lab(PT[idx, 0])
        lab_e = self.forward_lab(PT[idx, 1])
        t = T_ST.view(-1, 1, 1)  # (26, 1, 1)
        labs = lab_s.unsqueeze(0) + t * (lab_e - lab_s).unsqueeze(0)  # (26, 200, 3)
        dlab = labs[1:] - labs[:-1]  # (25, 200, 3)
        de = (dlab**2).sum(dim=-1).sqrt()  # (25, 200)
        md = de.mean(dim=0)
        sd = de.std(dim=0)
        ok = md > 0.001
        cvs = torch.where(ok, sd / md, torch.zeros_like(md))
        cv_loss = cvs[ok].mean() if ok.any() else torch.tensor(0.0, device=dev)

        # 3. Munsell V
        lab_g = self.forward_lab(MUNSELL_GRAYS)
        dL = lab_g[1:, 0] - lab_g[:-1, 0]
        munsv_loss = dL.std() / (dL.abs().mean() + 1e-10)

        # 4. White L
        lab_w = self.forward_lab(D65.unsqueeze(0))
        white_loss = (lab_w[0, 0] - 1.0).pow(2) * 50

        # 5. L monotonicity
        mono_loss = (-dL).clamp(min=0).sum() * 100

        # Combined
        loss = (4.0 * chroma_loss +
                2.0 * hue_loss +
                3.0 * cv_loss +
                1.5 * munsv_loss +
                white_loss +
                mono_loss)
        return loss, {
            'chroma': chroma_loss.item(),
            'hue': hue_loss.item(),
            'cv': cv_loss.item(),
            'munsv': munsv_loss.item(),
            'whiteL': lab_w[0,0].item(),
            'k': self.k.item(),
            'w': self.w.item(),
        }

# ================================================================
#  TRAINING
# ================================================================

model = HybridColorSpace().to(dev)

# Phase 1: freeze M1
print("\n=== PHASE 1: M1 frozen, optimize M2+k+w+L_corr ===", flush=True)
model.m1_free.requires_grad = False
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.003)

for step in range(2000):
    optimizer.zero_grad()
    loss, metrics = model.compute_loss()
    loss.backward()
    optimizer.step()

    # Clamp L_corr
    with torch.no_grad():
        model.lc.clamp_(-0.3, 0.3)

    if step % 100 == 0 or step < 5:
        print(f"  step {step:5d}  loss={loss.item():.4f}  "
              f"chroma={metrics['chroma']:.4f}  hue={metrics['hue']:.3f}  "
              f"cv={metrics['cv']:.3f}  munsv={metrics['munsv']:.3f}  "
              f"wL={metrics['whiteL']:.3f}  k={metrics['k']:.2f}  w={metrics['w']:.3f}",
              flush=True)

# Phase 2: unfreeze M1, small lr
print("\n=== PHASE 2: All params free, fine-tune ===", flush=True)
model.m1_free.requires_grad = True
optimizer2 = optim.Adam(model.parameters(), lr=0.0005)

for step in range(1000):
    optimizer2.zero_grad()
    loss, metrics = model.compute_loss()
    loss.backward()
    optimizer2.step()

    with torch.no_grad():
        model.lc.clamp_(-0.3, 0.3)

    if step % 100 == 0:
        print(f"  step {step:5d}  loss={loss.item():.4f}  "
              f"chroma={metrics['chroma']:.4f}  hue={metrics['hue']:.3f}  "
              f"cv={metrics['cv']:.3f}  munsv={metrics['munsv']:.3f}  "
              f"wL={metrics['whiteL']:.3f}  k={metrics['k']:.2f}  w={metrics['w']:.3f}",
              flush=True)

# Save
print("\n=== SAVING ===", flush=True)
M1 = model.M1.detach()
M2 = model.m2.detach()
# Apply achromatic fix permanently
lms_c_d65 = model.transfer((D65 @ M1.T).clamp(min=0).unsqueeze(0)).squeeze(0).detach()
norm2 = (lms_c_d65 * lms_c_d65).sum()
for i in [1, 2]:
    proj = (M2[i] * lms_c_d65).sum() / norm2
    M2[i] = M2[i] - proj * lms_c_d65

out = {
    "M1": M1.cpu().tolist(),
    "M2": M2.cpu().tolist(),
    "transfer": "hybrid",
    "transfer_param": model.k.item(),
    "blend_w": model.w.item(),
    "L_corr": model.lc.detach().cpu().tolist(),
    "architecture": "HybridTransfer_Adam_TwoPhase",
}

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
path = os.path.join(CKPT, f"adam_hybrid_{ts}.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved: {path}")

# Final metrics
loss, m = model.compute_loss()
print(f"\nFINAL: loss={loss.item():.4f}")
print(f"  chroma={m['chroma']:.4f} hue={m['hue']:.3f} cv={m['cv']:.3f}")
print(f"  munsv={m['munsv']:.3f} whiteL={m['whiteL']:.4f}")
print(f"  k={m['k']:.3f} w={m['w']:.3f}")
