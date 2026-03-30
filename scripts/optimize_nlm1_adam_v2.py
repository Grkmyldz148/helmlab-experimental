#!/usr/bin/env python3
"""Adam M2 optimizer for Nonlinear M1 — v2 (bug-free)."""

import json, math, os, sys, time
from datetime import datetime
import torch
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

M1 = torch.tensor([[6.213663,-0.504179,-0.404169],[-1.159226,4.350194,0.525494],
                    [0.000817,0.722672,2.227800]], device=dev)
CROSS_D = -0.60

def nlm1(xyz):
    lms = xyz @ M1.T
    cross = CROSS_D * (1.0 - xyz[..., 1]) * xyz[..., 2]
    return torch.cat([lms[..., 0:1] + cross.unsqueeze(-1), lms[..., 1:2], lms[..., 2:3]], dim=-1)

def to_lms_c(xyz):
    lms = nlm1(xyz)
    return torch.sign(lms) * lms.abs().clamp(min=1e-30).pow(1./3.)

D65_LC = to_lms_c(D65.unsqueeze(0)).squeeze(0).detach()

# Data
import colorsys
prims_rgb = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]]
PRIMS = torch.stack([MS@s2l(torch.tensor(r,device=dev,dtype=torch.float64)) for r in prims_rgb])
PRIMS_LC = to_lms_c(PRIMS).detach()
WHITE_LC = D65_LC.unsqueeze(0).detach()

MY = {1:0.01221,2:0.03126,3:0.06552,4:0.12,5:0.1977,6:0.30049,7:0.4306,8:0.591,9:0.7866}
MG = torch.stack([D65*MY[v] for v in range(1,10)]).to(dev)
MG_LC = to_lms_c(MG).detach()

# Shade colors
shade_rgbs = [[0.2,0.4,1.0],[1.0,0.2,0.2],[0.0,0.7,0.2],[1.0,0.5,0.0],[0.6,0.2,0.8],[0.0,0.8,0.9]]
SHADE_XYZ = torch.stack([MS@s2l(torch.tensor(r,device=dev,dtype=torch.float64)) for r in shade_rgbs])
SHADE_LC = to_lms_c(SHADE_XYZ).detach()

# Midpoint pairs
_mp = []
for h in range(0,360,30):
    r1,g1,b1 = colorsys.hsv_to_rgb(h/360,1,1)
    r2,g2,b2 = colorsys.hsv_to_rgb(((h+180)%360)/360,1,1)
    _mp.append(([r1,g1,b1],[r2,g2,b2]))
for rgb in prims_rgb:
    _mp.append((rgb,[1,1,1]))
MID1 = torch.stack([MS@s2l(torch.tensor(r,device=dev,dtype=torch.float64)) for r,_ in _mp])
MID2 = torch.stack([MS@s2l(torch.tensor(r,device=dev,dtype=torch.float64)) for _,r in _mp])
MID1_LC = to_lms_c(MID1).detach()
MID2_LC = to_lms_c(MID2).detach()

# Gradient pairs
for _d in [os.path.join(ROOT,"colorbench"), os.path.join(ROOT,"space-test-project")]:
    if os.path.isdir(os.path.join(_d,"core")):
        sys.path.insert(0,_d)
        from core.pairs import generate_all_pairs
        PT_full,_ = generate_all_pairs(dev)
        torch.manual_seed(42)
        idx = torch.randperm(PT_full.shape[0], device=dev)[:200]
        GP1_LC = to_lms_c(PT_full[idx,0]).detach()
        GP2_LC = to_lms_c(PT_full[idx,1]).detach()
        print(f"Gradient pairs: 200", flush=True)
        break

# Parameters
M2_v7b = torch.tensor([[0.4675499211910323, 0.20915320090703618, -0.08488334505679182],
                        [0.4843952725673558, -0.3665958307304812, -0.17266206907852755],
                        [-0.04418360083197623, 0.39383739736845824, -0.36863136176600936]], device=dev)
m2_param = torch.nn.Parameter(M2_v7b.clone())
lc_param = torch.nn.Parameter(torch.zeros(3, device=dev))

def get_M2():
    norm2 = (D65_LC * D65_LC).sum()
    p1 = (m2_param[1] * D65_LC).sum() / norm2
    p2 = (m2_param[2] * D65_LC).sum() / norm2
    return torch.stack([m2_param[0], m2_param[1]-p1*D65_LC, m2_param[2]-p2*D65_LC])

def fwd(lms_c):
    M2 = get_M2()
    lab = lms_c @ M2.T
    L = lab[..., 0:1]; c1,c2,c3 = lc_param[0],lc_param[1],lc_param[2]
    t = L*(1-L)
    return torch.cat([L+c1*t+c2*t*(2*L-1)+c3*L**2*(1-L)**2, lab[...,1:2], lab[...,2:3]], dim=-1)

def loss_fn():
    # 1. Gradient CV
    lab1 = fwd(GP1_LC); lab2 = fwd(GP2_LC)
    n = 16; ts = torch.linspace(0,1,n,device=dev)
    de_list = []
    for i in range(n-1):
        t1, t2 = ts[i], ts[i+1]
        p1 = lab1 + t1*(lab2-lab1)
        p2 = lab1 + t2*(lab2-lab1)
        dl = (p2-p1); de = (dl[...,0]**2*10000 + dl[...,1]**2*40000 + dl[...,2]**2*40000).sqrt()
        de_list.append(de)
    de_stack = torch.stack(de_list)  # (15, 200)
    md = de_stack.mean(dim=0); sd = de_stack.std(dim=0)
    ok = md > 0.001
    cv_loss = torch.where(ok, sd/md, torch.zeros_like(md)).mean()

    # 2. Munsell V
    lab_g = fwd(MG_LC)
    dL = lab_g[1:,0]-lab_g[:-1,0]
    munsv = dL.std()/(dL.abs().mean()+1e-10)
    mono = (-dL).clamp(min=0).sum()*100

    # 3. White
    lab_w = fwd(D65_LC.unsqueeze(0))
    w_loss = (lab_w[0,0]-1.0).pow(2)*200

    # 4. Midpoint chroma+hue
    lm1 = fwd(MID1_LC); lm2 = fwd(MID2_LC)
    lmid = 0.5*(lm1+lm2)
    Cm = (lmid[:,1]**2+lmid[:,2]**2).sqrt()
    C1 = (lm1[:,1]**2+lm1[:,2]**2).sqrt()
    C2 = (lm2[:,1]**2+lm2[:,2]**2).sqrt()
    Ca = 0.5*(C1+C2)
    mask = Ca>0.01
    cr = torch.where(mask, Cm/Ca.clamp(min=0.001), torch.ones_like(Cm))
    chroma = (1-cr).clamp(min=0).mean()

    hm = torch.atan2(lmid[:,2],lmid[:,1])
    h1 = torch.atan2(lm1[:,2],lm1[:,1]); h2 = torch.atan2(lm2[:,2],lm2[:,1])
    dh = h2-h1; dh = torch.where(dh>3.14159,dh-6.28318,dh); dh = torch.where(dh<-3.14159,dh+6.28318,dh)
    he = h1+0.5*dh
    herr = torch.atan2(torch.sin(hm-he),torch.cos(hm-he)).abs()
    hue = torch.where(Cm>0.01, herr, torch.zeros_like(herr)).mean()

    # 5. Shade hue (fully differentiable)
    lab_bases = fwd(SHADE_LC)  # (6, 3)
    lab_white = fwd(D65_LC.unsqueeze(0)).expand(6, -1)  # (6, 3)
    h_bases = torch.atan2(lab_bases[:,2], lab_bases[:,1])  # (6,)
    shade_total = torch.tensor(0.0, device=dev)
    for frac in [0.2, 0.5, 0.8]:
        lab_s = lab_bases + frac * (lab_white - lab_bases)
        h_s = torch.atan2(lab_s[:,2], lab_s[:,1])
        C_s = (lab_s[:,1]**2+lab_s[:,2]**2).sqrt()
        dhs = torch.atan2(torch.sin(h_s-h_bases), torch.cos(h_s-h_bases)).abs()
        # Weight by chroma
        shade_total = shade_total + (dhs * C_s.clamp(min=0, max=0.5)).mean()
    shade = shade_total / 3.0

    return (3.0*cv_loss + 1.0*munsv + mono + w_loss +
            2.0*chroma + 3.0*hue + 5.0*shade), {
        "cv":cv_loss.item(), "munsv":munsv.item(), "chroma":chroma.item(),
        "hue":hue.item(), "shade":shade.item(), "wL":lab_w[0,0].item()}

# Train
opt = optim.Adam([m2_param, lc_param], lr=0.002)
sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000)
print("Training...", flush=True)

for step in range(5000):
    opt.zero_grad()
    loss, m = loss_fn()
    if torch.isnan(loss): continue
    loss.backward()
    torch.nn.utils.clip_grad_norm_([m2_param, lc_param], 1.0)
    opt.step()
    sch.step()
    with torch.no_grad(): lc_param.clamp_(-0.3, 0.3)
    if step % 200 == 0 or step < 5:
        print("  step %5d  loss=%.3f  cv=%.4f  munsv=%.4f  chroma=%.4f  hue=%.3f  shade=%.4f  wL=%.3f" % (step, loss.item(), m["cv"], m["munsv"], m["chroma"], m["hue"], m["shade"], m["wL"]),

              flush=True)

# Save
M2f = get_M2().detach()
out = {"M1":M1.cpu().tolist(), "M2":M2f.cpu().tolist(), "gamma":[1/3,1/3,1/3],
       "L_corr":lc_param.detach().cpu().tolist(), "cross_term_d":CROSS_D,
       "architecture":"NLM1_Adam_v2"}
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
p = os.path.join(CKPT, f"nlm1_adam_v2_{ts}.json")
with open(p,"w") as f: json.dump(out, f, indent=2)
print(f"Saved: {p}")
_, m = loss_fn()
print("FINAL:", m)
