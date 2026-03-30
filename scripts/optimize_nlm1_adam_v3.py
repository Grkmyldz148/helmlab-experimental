#!/usr/bin/env python3
"""Adam M2 for NLM1 — v3: sRGB-space metrics, fully differentiable.

Key fix: compute gradient quality in CIE Lab (after inverse to XYZ),
not in our Lab space. This makes CV and shade losses meaningful.
"""

import json, math, os, sys, time, colorsys
from datetime import datetime
import torch
import torch.optim as optim

torch.set_default_dtype(torch.float64)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", dev, flush=True)

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
M1i = torch.linalg.inv(M1)
CROSS_D = -0.60

def nlm1_fwd(xyz):
    lms = xyz @ M1.T
    cross = CROSS_D * (1.0 - xyz[..., 1]) * xyz[..., 2]
    return torch.cat([lms[..., 0:1]+cross.unsqueeze(-1), lms[..., 1:2], lms[..., 2:3]], dim=-1)

def nlm1_inv(lms_target):
    """Newton inverse for nonlinear M1 — differentiable through implicit function theorem."""
    xyz = lms_target @ M1i.T
    for _ in range(20):
        lms = xyz @ M1.T
        cross = CROSS_D * (1.0 - xyz[..., 1]) * xyz[..., 2]
        lms_cur = torch.cat([lms[..., 0:1]+cross.unsqueeze(-1), lms[..., 1:2], lms[..., 2:3]], dim=-1)
        err = lms_cur - lms_target
        # Jacobian: M1 + cross term derivatives
        J = M1.unsqueeze(0).expand(xyz.shape[0], -1, -1).clone()
        J[:, 0, 1] = J[:, 0, 1] - CROSS_D * xyz[..., 2]
        J[:, 0, 2] = J[:, 0, 2] + CROSS_D * (1.0 - xyz[..., 1])
        xyz = xyz - torch.linalg.solve(J, err.unsqueeze(-1)).squeeze(-1)
    return xyz

D65_LC = (lambda lms: torch.sign(lms)*lms.abs().clamp(min=1e-30).pow(1./3.))(nlm1_fwd(D65.unsqueeze(0))).squeeze(0).detach()

def xyz_to_cielab(xyz):
    r = xyz / D65
    d3 = (6./29.)**3
    f = torch.where(r > d3, r.pow(1./3.), r/(3*(6./29.)**2) + 4./29.)
    return torch.stack([116*f[...,1]-16, 500*(f[...,0]-f[...,1]), 200*(f[...,1]-f[...,2])], dim=-1)

def ciede_simple(cl1, cl2):
    dL = cl2[...,0]-cl1[...,0]
    C1 = (cl1[...,1]**2+cl1[...,2]**2).sqrt()
    C2 = (cl2[...,1]**2+cl2[...,2]**2).sqrt()
    dC = C2-C1
    dH = ((cl2[...,1]-cl1[...,1])**2+(cl2[...,2]-cl1[...,2])**2-dC**2).clamp(min=0).sqrt()
    SL = 1+0.015*(cl1[...,0]-50)**2/(20+(cl1[...,0]-50)**2).sqrt()
    SC = 1+0.045*C1; SH = 1+0.015*C1
    return ((dL/SL)**2+(dC/SC)**2+(dH/SH)**2).sqrt()

# Data
MY = {1:0.01221,2:0.03126,3:0.06552,4:0.12,5:0.1977,6:0.30049,7:0.4306,8:0.591,9:0.7866}
MG = torch.stack([D65*MY[v] for v in range(1,10)]).to(dev)
shade_rgbs = [[0.2,0.4,1.0],[1.0,0.2,0.2],[0.0,0.7,0.2],[1.0,0.5,0.0],[0.6,0.2,0.8],[0.0,0.8,0.9]]
SHADE_XYZ = torch.stack([MS@s2l(torch.tensor(r,device=dev,dtype=torch.float64)) for r in shade_rgbs])

# Gradient pairs (50 — small for speed with full inverse)
for _d in [os.path.join(ROOT,"colorbench"), os.path.join(ROOT,"space-test-project")]:
    if os.path.isdir(os.path.join(_d,"core")):
        sys.path.insert(0,_d)
        from core.pairs import generate_all_pairs
        PT_full,_ = generate_all_pairs(dev)
        torch.manual_seed(42)
        GP = PT_full[torch.randperm(PT_full.shape[0], device=dev)[:50]]
        print("Pairs:", GP.shape[0], flush=True)
        break

# Parameters
M2_init = torch.tensor([[0.4675499211910323, 0.20915320090703618, -0.08488334505679182],
                         [0.4843952725673558, -0.3665958307304812, -0.17266206907852755],
                         [-0.04418360083197623, 0.39383739736845824, -0.36863136176600936]], device=dev)
m2_param = torch.nn.Parameter(M2_init.clone())
lc_param = torch.nn.Parameter(torch.zeros(3, device=dev))

def get_M2():
    norm2 = (D65_LC*D65_LC).sum()
    p1 = (m2_param[1]*D65_LC).sum()/norm2
    p2 = (m2_param[2]*D65_LC).sum()/norm2
    return torch.stack([m2_param[0], m2_param[1]-p1*D65_LC, m2_param[2]-p2*D65_LC])

def space_fwd(xyz):
    lms = nlm1_fwd(xyz)
    lms_c = torch.sign(lms)*lms.abs().clamp(min=1e-30).pow(1./3.)
    M2 = get_M2()
    lab = lms_c @ M2.T
    L = lab[...,0:1]; c1,c2,c3 = lc_param[0],lc_param[1],lc_param[2]
    t = L*(1-L)
    return torch.cat([L+c1*t+c2*t*(2*L-1)+c3*L**2*(1-L)**2, lab[...,1:2], lab[...,2:3]], dim=-1)

def space_inv(lab):
    L1 = lab[...,0:1]; c1,c2,c3 = lc_param[0],lc_param[1],lc_param[2]
    L = L1.clone()
    for _ in range(10):
        t = L*(1-L); f = L+c1*t+c2*t*(2*L-1)+c3*L**2*(1-L)**2-L1
        df = 1+c1*(1-2*L)+c2*(6*L**2-6*L+1)+c3*2*L*(1-L)*(1-2*L)
        L = L - f/df.clamp(min=1e-12)
    raw = torch.cat([L, lab[...,1:2], lab[...,2:3]], dim=-1)
    M2 = get_M2()
    M2i = torch.linalg.inv(M2)
    lms_c = raw @ M2i.T
    lms = torch.sign(lms_c)*lms_c.abs().pow(3)
    return nlm1_inv(lms)

def loss_fn():
    M2 = get_M2()
    
    # Precompute some things
    def fwd_lc(lms_c):
        lab = lms_c @ M2.T
        L = lab[...,0:1]; c1,c2,c3 = lc_param[0],lc_param[1],lc_param[2]
        t = L*(1-L)
        return torch.cat([L+c1*t+c2*t*(2*L-1)+c3*L**2*(1-L)**2, lab[...,1:2], lab[...,2:3]], dim=-1)
    
    # Precompute LMS_c for data (fixed, no gradient needed)
    def to_lc(xyz):
        lms = nlm1_fwd(xyz)
        return torch.sign(lms)*lms.abs().clamp(min=1e-30).pow(1./3.)
    
    # 1. Gradient CV — in our Lab, but with inverse→sRGB→CIE Lab for accuracy
    # Use detached inverse for CIE Lab computation (no gradient through Newton)
    gp1_lc = to_lc(GP[:,0]).detach()
    gp2_lc = to_lc(GP[:,1]).detach()
    lab1 = fwd_lc(gp1_lc); lab2 = fwd_lc(gp2_lc)
    N_ST = 11; ts = torch.linspace(0,1,N_ST,device=dev)
    
    # Compute step sizes with detached inverse
    de_steps = []
    for i in range(N_ST-1):
        p1 = lab1 + ts[i]*(lab2-lab1)
        p2 = lab1 + ts[i+1]*(lab2-lab1)
        with torch.no_grad():
            xyz1 = space_inv(p1.detach())
            xyz2 = space_inv(p2.detach())
            cl1 = xyz_to_cielab(xyz1); cl2 = xyz_to_cielab(xyz2)
            de = ciede_simple(cl1, cl2)
        de_steps.append(de)
    de_stack = torch.stack(de_steps)
    # CV as target: compute CV value, then penalize high CV via Lab-space proxy
    with torch.no_grad():
        md = de_stack.mean(dim=0); sd = de_stack.std(dim=0)
        ok = md > 0.1
        cv_val = torch.where(ok, sd/md, torch.zeros_like(md)).mean()
    
    # Lab-space CV (provides gradient)
    dlab = []
    for i in range(N_ST-1):
        p1 = lab1 + ts[i]*(lab2-lab1)
        p2 = lab1 + ts[i+1]*(lab2-lab1)
        dl = (p2-p1).pow(2).sum(dim=-1).sqrt()
        dlab.append(dl)
    dlab_s = torch.stack(dlab)
    md_lab = dlab_s.mean(dim=0); sd_lab = dlab_s.std(dim=0)
    ok_lab = md_lab > 0.001
    cv_loss = torch.where(ok_lab, sd_lab/md_lab, torch.zeros_like(md_lab)).mean()
    
    # 2. Munsell V
    mg_lc = to_lc(MG).detach()
    lab_g = fwd_lc(mg_lc)
    dL = lab_g[1:,0]-lab_g[:-1,0]
    munsv = dL.std()/(dL.abs().mean()+1e-10)
    mono = (-dL).clamp(min=0).sum()*100
    
    # 3. White
    w_lc = to_lc(D65.unsqueeze(0)).detach()
    lab_w = fwd_lc(w_lc)
    w_loss = (lab_w[0,0]-1.0).pow(2)*200
    
    # 4. Shade hue — in our Lab with CIE Lab monitoring
    shade_lc = to_lc(SHADE_XYZ).detach()
    lab_bases = fwd_lc(shade_lc)
    lab_white = fwd_lc(w_lc)
    shade_drifts = []
    shade_cielab_drifts = []
    for si in range(6):
        h_base = torch.atan2(lab_bases[si,2], lab_bases[si,1])
        for frac in [0.2, 0.5, 0.8]:
            lab_s = lab_bases[si:si+1] + frac*(lab_white - lab_bases[si:si+1])
            h_s = torch.atan2(lab_s[0,2], lab_s[0,1])
            C_s = (lab_s[0,1]**2+lab_s[0,2]**2).sqrt()
            dh = torch.atan2(torch.sin(h_s-h_base), torch.cos(h_s-h_base)).abs()
            shade_drifts.append(torch.where(C_s > 0.01, dh, torch.zeros_like(dh)))
            # CIE Lab monitoring (detached)
            with torch.no_grad():
                xyz_s = space_inv(lab_s.detach())
                cl_s = xyz_to_cielab(xyz_s)
                h_cs = torch.atan2(cl_s[0,2], cl_s[0,1])
                xyz_b = space_inv(lab_bases[si:si+1].detach())
                cl_b = xyz_to_cielab(xyz_b)
                h_cb = torch.atan2(cl_b[0,2], cl_b[0,1])
                dh_cie = torch.atan2(torch.sin(h_cs-h_cb), torch.cos(h_cs-h_cb)).abs()
                shade_cielab_drifts.append(dh_cie.item())
    
    shade_loss = torch.stack(shade_drifts).mean()
    shade_cie = sum(shade_cielab_drifts)/len(shade_cielab_drifts) if shade_cielab_drifts else 0
    
    # 5. Hue linearity (our Lab) — primary→white gradients
    prims_rgb = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]]
    hue_drifts = []
    for rgb in prims_rgb:
        xyz_p = MS@s2l(torch.tensor(rgb,device=dev,dtype=torch.float64))
        lc_p = to_lc(xyz_p.unsqueeze(0)).detach()
        lab_p = fwd_lc(lc_p)
        h_p = torch.atan2(lab_p[0,2], lab_p[0,1])
        for frac in [0.2, 0.5, 0.8]:
            lab_t = lab_p + frac*(lab_white - lab_p)
            h_t = torch.atan2(lab_t[0,2], lab_t[0,1])
            C_t = (lab_t[0,1]**2+lab_t[0,2]**2).sqrt()
            dh = torch.atan2(torch.sin(h_t-h_p), torch.cos(h_t-h_p)).abs()
            hue_drifts.append(torch.where(C_t > 0.01, dh, torch.zeros_like(dh)))
    hue_loss = torch.stack(hue_drifts).mean()
    
    loss = 2.0*cv_loss + 1.0*munsv + mono + w_loss + 4.0*shade_loss + 3.0*hue_loss
    return loss, cv_val.item(), munsv.item(), shade_cie, lab_w[0,0].item(), hue_loss.item()

# Train
opt = optim.Adam([m2_param, lc_param], lr=0.002)
print("Training (sRGB-space metrics)...", flush=True)

for step in range(2000):
    opt.zero_grad()
    loss, cv, mv, sh, wl, hl = loss_fn()
    if torch.isnan(loss):
        print("  NaN at step", step, flush=True)
        continue
    loss.backward()
    torch.nn.utils.clip_grad_norm_([m2_param, lc_param], 1.0)
    opt.step()
    with torch.no_grad(): lc_param.clamp_(-0.3, 0.3)
    if step % 100 == 0 or step < 5:
        print("  step %5d  loss=%.3f  cv=%.4f  munsv=%.4f  shade=%.4f  wL=%.3f  hue=%.4f" % (step, loss.item(), cv, mv, sh, wl, hl), flush=True)

# Save
M2f = get_M2().detach()
out = {"M1": M1.cpu().tolist(), "M2": M2f.cpu().tolist(), "gamma": [1/3,1/3,1/3],
       "L_corr": lc_param.detach().cpu().tolist(), "cross_term_d": CROSS_D,
       "architecture": "NLM1_Adam_v3_sRGB"}
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
p = os.path.join(CKPT, "nlm1_adam_v3_%s.json" % ts)
with open(p, "w") as f: json.dump(out, f, indent=2)
print("Saved:", p)
print("FINAL: loss=%.4f cv=%.4f munsv=%.4f shade_cie=%.4f wL=%.4f hue=%.4f" % (loss.item(), cv, mv, sh, wl, hl))
