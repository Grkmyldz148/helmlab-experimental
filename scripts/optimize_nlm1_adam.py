#!/usr/bin/env python3
"""Adam M2 optimizer for Nonlinear M1 (d=-0.60).

M1 fixed (v7b + cross term). Only M2(9) + L_corr(3) = 12 params.
Differentiable loss: gradient CV + Munsell V + achromatic + hue metrics.
"""

import json, math, os, sys, time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_default_dtype(torch.float64)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {dev}", flush=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")

D65 = torch.tensor([0.95047, 1.0, 1.08883], device=dev)
MS = torch.tensor([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],
                    [.0193339,.1191920,.9503041]], device=dev)
MSi = torch.linalg.inv(MS)
def s2l(c): return torch.where(c<=0.04045, c/12.92, ((c+0.055)/1.055).pow(2.4))

# Fixed nonlinear M1
M1 = torch.tensor([[6.213663,-0.504179,-0.404169],[-1.159226,4.350194,0.525494],
                    [0.000817,0.722672,2.227800]], device=dev)
CROSS_D = -0.60

def nlm1_fwd(xyz):
    lms = xyz @ M1.T
    cross = CROSS_D * (1.0 - xyz[..., 1]) * xyz[..., 2]
    lms = torch.cat([lms[..., 0:1] + cross.unsqueeze(-1), lms[..., 1:2], lms[..., 2:3]], dim=-1)
    return torch.sign(lms) * lms.abs().clamp(min=1e-30).pow(1./3.)

# Precompute D65 LMS cbrt
LMS_C_D65 = nlm1_fwd(D65.unsqueeze(0)).squeeze(0)

# Data
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

MID_XYZ1 = torch.stack([MS@s2l(torch.tensor(r,device=dev,dtype=torch.float64)) for r,_ in _mp])
MID_XYZ2 = torch.stack([MS@s2l(torch.tensor(r,device=dev,dtype=torch.float64)) for _,r in _mp])
MID_LMS_C1 = nlm1_fwd(MID_XYZ1)
MID_LMS_C2 = nlm1_fwd(MID_XYZ2)

MY = {1:0.01221,2:0.03126,3:0.06552,4:0.12,5:0.1977,6:0.30049,7:0.4306,8:0.591,9:0.7866}
MG = torch.stack([D65*MY[v] for v in range(1,10)]).to(dev)
MG_LMS_C = nlm1_fwd(MG)

# Gradient pairs (subsample)
for _d in [os.path.join(ROOT,"colorbench"), os.path.join(ROOT,"space-test-project")]:
    if os.path.isdir(os.path.join(_d,"core")):
        sys.path.insert(0,_d)
        from core.pairs import generate_all_pairs
        PT_full,_ = generate_all_pairs(dev)
        idx = torch.randperm(PT_full.shape[0], device=dev)[:300]
        PT = PT_full[idx]
        PT_LMS_C1 = nlm1_fwd(PT[:,0])
        PT_LMS_C2 = nlm1_fwd(PT[:,1])
        print(f"Pairs: {PT.shape[0]}", flush=True)
        break

# Shade hue test colors
SHADE_COLORS = []
for name, rgb in [("Blue",[0.2,0.4,1.0]),("Red",[1.0,0.2,0.2]),("Green",[0.0,0.7,0.2]),
                   ("Orange",[1.0,0.5,0.0]),("Purple",[0.6,0.2,0.8]),("Cyan",[0.0,0.8,0.9])]:
    xyz = MS @ s2l(torch.tensor(rgb, device=dev, dtype=torch.float64))
    SHADE_COLORS.append(xyz)

# ================================================================

class M2Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Start from v7b M2
        M2_v7b = torch.tensor([[0.4675499211910323, 0.20915320090703618, -0.08488334505679182],
                                [0.4843952725673558, -0.3665958307304812, -0.17266206907852755],
                                [-0.04418360083197623, 0.39383739736845824, -0.36863136176600936]], device=dev)
        self.m2 = nn.Parameter(M2_v7b)
        self.lc = nn.Parameter(torch.zeros(3, device=dev))

    def get_M2_fixed(self):
        """M2 with achromatic projection."""
        M2 = self.m2
        norm2 = (LMS_C_D65 * LMS_C_D65).sum()
        proj1 = (M2[1] * LMS_C_D65).sum() / norm2
        proj2 = (M2[2] * LMS_C_D65).sum() / norm2
        return torch.stack([M2[0], M2[1] - proj1*LMS_C_D65, M2[2] - proj2*LMS_C_D65])

    def fwd_lab(self, lms_c):
        M2 = self.get_M2_fixed()
        lab = lms_c @ M2.T
        L = lab[..., 0:1]
        c1,c2,c3 = self.lc[0],self.lc[1],self.lc[2]
        t = L*(1-L)
        L_new = L + c1*t + c2*t*(2*L-1) + c3*L**2*(1-L)**2
        return torch.cat([L_new, lab[..., 1:2], lab[..., 2:3]], dim=-1)

    def compute_loss(self):
        # 1. Gradient CV (Lab-space step sizes)
        lab1 = self.fwd_lab(PT_LMS_C1); lab2 = self.fwd_lab(PT_LMS_C2)
        N_ST = 16
        t = torch.linspace(0,1,N_ST,device=dev).view(-1,1,1)
        labs = lab1.unsqueeze(0) + t*(lab2-lab1).unsqueeze(0)  # (16, 300, 3)
        # Scale to CIE Lab-like range for meaningful step sizes
        labs_scaled = labs.clone()
        labs_scaled[..., 0] = labs_scaled[..., 0] * 100  # L: 0-1 -> 0-100
        labs_scaled[..., 1:] = labs_scaled[..., 1:] * 200  # a,b: scale up
        dlab = labs_scaled[1:]-labs_scaled[:-1]
        de = (dlab**2).sum(dim=-1).sqrt()
        md=de.mean(dim=0); sd=de.std(dim=0); ok=md>0.1  # threshold for scaled
        cvs = torch.where(ok, sd/md, torch.zeros_like(md))
        cv_loss = cvs[ok].mean() if ok.any() else torch.tensor(0.0, device=dev)

        # 2. Munsell V
        lab_g = self.fwd_lab(MG_LMS_C)
        dL = lab_g[1:,0]-lab_g[:-1,0]
        munsv_loss = dL.std()/(dL.abs().mean()+1e-10)
        mono_loss = (-dL).clamp(min=0).sum()*100

        # 3. White
        lab_w = self.fwd_lab(LMS_C_D65.unsqueeze(0))
        white_loss = (lab_w[0,0]-1.0).pow(2)*200

        # 4. Midpoint quality (chroma + hue)
        lab_m1 = self.fwd_lab(MID_LMS_C1); lab_m2 = self.fwd_lab(MID_LMS_C2)
        lab_mid = 0.5*(lab_m1+lab_m2)
        C_mid = (lab_mid[:,1]**2+lab_mid[:,2]**2).sqrt()
        C1 = (lab_m1[:,1]**2+lab_m1[:,2]**2).sqrt()
        C2 = (lab_m2[:,1]**2+lab_m2[:,2]**2).sqrt()
        C_avg = 0.5*(C1+C2)
        mask = C_avg>0.01
        chroma_ratio = torch.where(mask, C_mid/C_avg.clamp(min=0.001), torch.ones_like(C_mid))
        chroma_loss = (1-chroma_ratio).clamp(min=0).mean()

        h_mid = torch.atan2(lab_mid[:,2], lab_mid[:,1])
        h1 = torch.atan2(lab_m1[:,2], lab_m1[:,1])
        h2 = torch.atan2(lab_m2[:,2], lab_m2[:,1])
        dh = h2-h1
        dh = torch.where(dh>math.pi, dh-2*math.pi, dh)
        dh = torch.where(dh<-math.pi, dh+2*math.pi, dh)
        h_exp = h1+0.5*dh
        h_err = torch.atan2(torch.sin(h_mid-h_exp), torch.cos(h_mid-h_exp)).abs()
        hue_loss = torch.where(C_mid>0.01, h_err, torch.zeros_like(h_err)).mean()

        # 5. Shade hue consistency
        lab_white = self.fwd_lab(LMS_C_D65.unsqueeze(0))
        shade_drifts = []
        for xyz_base in SHADE_COLORS:
            lms_c_base = nlm1_fwd(xyz_base.unsqueeze(0))
            lab_base = self.fwd_lab(lms_c_base)
            h_base = torch.atan2(lab_base[0,2], lab_base[0,1])
            C_base = (lab_base[0,1]**2+lab_base[0,2]**2).sqrt()
            for frac in [0.15, 0.3, 0.5, 0.7, 0.85]:
                lab_shade = lab_base + frac*(lab_white - lab_base)
                h_shade = torch.atan2(lab_shade[0,2], lab_shade[0,1])
                dh = torch.atan2(torch.sin(h_shade-h_base), torch.cos(h_shade-h_base)).abs()
                C_shade = (lab_shade[0,1]**2+lab_shade[0,2]**2).sqrt()
                # Weight by chroma (achromatic samples don't count)
                shade_drifts.append(dh * C_shade.clamp(max=1.0) / max(C_base.item(), 0.01))
        shade_loss = torch.stack(shade_drifts).mean()

        # Combined
        loss = (3.0*cv_loss + 1.0*munsv_loss + mono_loss + white_loss +
                2.0*chroma_loss + 3.0*hue_loss + 4.0*shade_loss)

        return loss, {
            'cv': cv_loss.item(), 'munsv': munsv_loss.item(),
            'chroma': chroma_loss.item(), 'hue': hue_loss.item(),
            'shade': shade_loss.item(), 'wL': lab_w[0,0].item(),
        }

# ================================================================

model = M2Model().to(dev)
optimizer = optim.Adam(model.parameters(), lr=0.002)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000)

print("Training M2 for Nonlinear M1...", flush=True)
for step in range(3000):
    optimizer.zero_grad()
    loss, m = model.compute_loss()
    if torch.isnan(loss): continue
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    with torch.no_grad():
        model.lc.clamp_(-0.3, 0.3)
    if step % 200 == 0 or step < 5:
        print(f"  step {step:5d}  loss={loss.item():.3f}  cv={m['cv']:.4f}  munsv={m['munsv']:.4f}  "
              f"chroma={m['chroma']:.4f}  hue={m['hue']:.3f}  shade={m['shade']:.3f}  wL={m['wL']:.3f}", flush=True)

# Save
M2_final = model.get_M2_fixed().detach()
lc_final = model.lc.detach().clamp(-0.3, 0.3)

out = {
    "M1": M1.cpu().tolist(),
    "M2": M2_final.cpu().tolist(),
    "gamma": [1/3, 1/3, 1/3],
    "L_corr": lc_final.cpu().tolist(),
    "cross_term_d": CROSS_D,
    "cross_term_formula": "lms[0] += d * (1 - Y) * Z",
    "architecture": "NonlinearM1_Adam_M2",
}
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
path = os.path.join(CKPT, f"nlm1_adam_{ts}.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
print(f"FINAL: loss={loss.item():.4f}")
for k,v in m.items():
    print(f"  {k}={v:.4f}")
