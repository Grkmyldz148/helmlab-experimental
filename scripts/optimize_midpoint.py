#!/usr/bin/env python3
"""Midpoint Quality Optimization — novel approach.

Optimize M1+M2+L_corr not for distance or uniformity,
but for MIDPOINT APPEARANCE: the interpolated color between two endpoints
should preserve chroma, maintain hue, and look perceptually correct.

Nobody has done this. Every existing space optimizes for distance (STRESS),
uniformity (Munsell), or gradient step size (CV). This optimizes for what
users actually see: the midpoint of a gradient.

Loss = midpoint_chroma_preservation + gradient_CV + munsell_V + achromatic
"""

import json, math, os, sys, time, argparse
from datetime import datetime
import numpy as np
import torch
import cma

torch.set_default_dtype(torch.float64)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {dev}", flush=True)

pa = argparse.ArgumentParser()
pa.add_argument("--gens", type=int, default=300)
pa.add_argument("--pop", type=int, default=128)
pa.add_argument("--seeds", type=int, default=8)
args = pa.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
CKPT = os.path.join(ROOT, "checkpoints")
os.makedirs(CKPT, exist_ok=True)

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
#  MIDPOINT TEST PAIRS — diverse, including problematic ones
# ================================================================

def _hsv(h, s, v):
    if s == 0: return v, v, v
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v*(1-s); q = v*(1-s*f); t = v*(1-s*(1-f))
    return [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][i]

# Build midpoint test pairs on GPU
_mid_pairs = []
# Complementary (worst muddy)
for h in range(0, 360, 30):
    r1,g1,b1 = _hsv(h/360, 1.0, 1.0)
    r2,g2,b2 = _hsv(((h+180)%360)/360, 1.0, 1.0)
    _mid_pairs.append(([r1,g1,b1], [r2,g2,b2]))
# Primary to white (blue-white!)
for rgb in [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]]:
    _mid_pairs.append((rgb, [1,1,1]))
# Primary to primary
for i, c1 in enumerate([[1,0,0],[0,1,0],[0,0,1]]):
    for c2 in [[1,0,0],[0,1,0],[0,0,1]]:
        if c1 != c2:
            _mid_pairs.append((c1, c2))
# Warm-cool
_mid_pairs.append(([1,0.6,0.2], [0.2,0.4,1]))
_mid_pairs.append(([1,0.5,0.5], [0.5,1,1]))
_mid_pairs.append(([0.6,0.2,0.8], [0.9,0.8,0.2]))
# Skin tones
_mid_pairs.append(([0.96,0.91,0.84], [0.36,0.21,0.13]))
# Dark-dark
_mid_pairs.append(([0.6,0,0], [0,0,0.6]))

# Convert to XYZ tensors
MID_XYZ1 = torch.zeros(len(_mid_pairs), 3, device=dev)
MID_XYZ2 = torch.zeros(len(_mid_pairs), 3, device=dev)
for i, (rgb1, rgb2) in enumerate(_mid_pairs):
    MID_XYZ1[i] = MS @ s2l(torch.tensor(rgb1, device=dev, dtype=torch.float64))
    MID_XYZ2[i] = MS @ s2l(torch.tensor(rgb2, device=dev, dtype=torch.float64))

# CIE Lab chroma of endpoints (reference)
def _xyz_to_cielab(xyz):
    r = xyz / D65
    delta3 = (6.0/29.0)**3
    f = torch.where(r > delta3, r.pow(1./3.), r/(3*(6.0/29.0)**2) + 4.0/29.0)
    L = 116.0*f[...,1]-16.0; a = 500.0*(f[...,0]-f[...,1]); b = 200.0*(f[...,1]-f[...,2])
    return torch.stack([L,a,b], dim=-1)

CL1 = _xyz_to_cielab(MID_XYZ1)
CL2 = _xyz_to_cielab(MID_XYZ2)
C_END1 = (CL1[:,1]**2 + CL1[:,2]**2).sqrt()
C_END2 = (CL2[:,1]**2 + CL2[:,2]**2).sqrt()
C_END_AVG = 0.5 * (C_END1 + C_END2)  # (N,)

N_MID = len(_mid_pairs)
print(f"Midpoint pairs: {N_MID}", flush=True)

# ================================================================
#  GRADIENT PAIRS (from colorbench)
# ================================================================
for _d in [os.path.join(ROOT, "colorbench"), os.path.join(ROOT, "space-test-project")]:
    if os.path.isdir(os.path.join(_d, "core")):
        sys.path.insert(0, _d)
        from core.pairs import generate_all_pairs
        PT, _ = generate_all_pairs(dev)
        print(f"Gradient pairs: {PT.shape[0]}", flush=True)
        break

N_PAIRS = PT.shape[0]
N_ST = 25
T_ST = torch.linspace(0, 1, N_ST + 1, device=dev)

# Munsell
MY = {1:0.01221,2:0.03126,3:0.06552,4:0.12000,5:0.19770,6:0.30049,7:0.43060,8:0.59100,9:0.78660}
MUNSELL_GRAYS = torch.stack([D65*MY[v] for v in range(1,10)]).to(dev)

# ================================================================
#  FORWARD / INVERSE (batched P candidates)
# ================================================================

def fwd(xyz, M1, M2, lc):
    lms = (xyz.unsqueeze(0) @ M1.transpose(-1,-2)).clamp(min=0)
    lms_c = torch.sign(lms) * lms.abs().pow(1./3.)
    lab = torch.bmm(lms_c, M2.transpose(-1,-2))
    L = lab[...,0:1]
    c1=lc[:,0:1].unsqueeze(1); c2=lc[:,1:2].unsqueeze(1); c3=lc[:,2:3].unsqueeze(1)
    t = L*(1.0-L)
    L_new = L + c1*t + c2*t*(2.0*L-1.0) + c3*L**2*(1.0-L)**2
    return torch.cat([L_new, lab[...,1:2], lab[...,2:3]], dim=-1)

def inv(lab, M1i, M2i, lc):
    L1 = lab[...,0:1]
    c1=lc[:,0:1].unsqueeze(1); c2=lc[:,1:2].unsqueeze(1); c3=lc[:,2:3].unsqueeze(1)
    L = L1.clone()
    for _ in range(12):
        t=L*(1-L); f=L+c1*t+c2*t*(2*L-1)+c3*L**2*(1-L)**2-L1
        df=1+c1*(1-2*L)+c2*(6*L**2-6*L+1)+c3*2*L*(1-L)*(1-2*L)
        L = L - f/df.clamp(min=1e-12)
    raw = torch.cat([L, lab[...,1:2], lab[...,2:3]], dim=-1)
    lms_c = torch.bmm(raw, M2i.transpose(-1,-2))
    lms = torch.sign(lms_c)*lms_c.abs().pow(3.0)
    return torch.bmm(lms, M1i.transpose(-1,-2))

# ================================================================
#  UNPACK (18 params: M1(6) + M2(9) + L_corr(3))
# ================================================================

def unpack(x):
    P = x.shape[0]
    M1 = torch.zeros(P,3,3,device=dev)
    M1[:,0,0]=x[:,0]; M1[:,0,1]=x[:,1]; M1[:,1,0]=x[:,2]; M1[:,1,1]=x[:,3]
    M1[:,2,0]=x[:,4]; M1[:,2,1]=x[:,5]
    M1[:,0,2]=(1-M1[:,0,0]*D65[0]-M1[:,0,1]*D65[1])/D65[2]
    M1[:,1,2]=(1-M1[:,1,0]*D65[0]-M1[:,1,1]*D65[1])/D65[2]
    M1[:,2,2]=(1-M1[:,2,0]*D65[0]-M1[:,2,1]*D65[1])/D65[2]
    lms_d65=(D65.unsqueeze(0).unsqueeze(0)@M1.transpose(-1,-2)).squeeze(1)
    valid=(lms_d65>0.01).all(dim=1)

    M2=torch.zeros(P,3,3,device=dev)
    M2[:,0,0]=x[:,6];M2[:,0,1]=x[:,7];M2[:,0,2]=x[:,8]
    M2[:,1,0]=x[:,9];M2[:,1,1]=x[:,10];M2[:,1,2]=x[:,11]
    M2[:,2,0]=x[:,12];M2[:,2,1]=x[:,13];M2[:,2,2]=x[:,14]

    lc=torch.stack([x[:,15].clamp(-0.3,0.3),x[:,16].clamp(-0.3,0.3),x[:,17].clamp(-0.5,0.5)],dim=1)

    # Achromatic: M2 a,b rows orthogonal to D65 cbrt
    lms_c_d65 = lms_d65.pow(1./3.)
    lab_d65 = torch.bmm(lms_c_d65.unsqueeze(1), M2.transpose(-1,-2)).squeeze(1)
    ach_err = (lab_d65[:,1]**2 + lab_d65[:,2]**2).sqrt()
    white_L = lab_d65[:,0]
    t_w = white_L*(1-white_L)
    white_L_corr = white_L + lc[:,0]*t_w + lc[:,1]*t_w*(2*white_L-1) + lc[:,2]*white_L**2*(1-white_L)**2

    valid &= (white_L_corr > 0.9) & (white_L_corr < 1.1)
    valid &= (ach_err < 0.02)

    M1i=torch.zeros_like(M1); M2i=torch.zeros_like(M2)
    det1=torch.linalg.det(M1); det2=torch.linalg.det(M2)
    invertible=(det1.abs()>1e-10)&(det2.abs()>1e-10)&valid
    good=invertible.nonzero(as_tuple=True)[0]
    if good.numel()>0:
        M1i[good]=torch.linalg.inv(M1[good])
        M2i[good]=torch.linalg.inv(M2[good])
    return M1,M2,M1i,M2i,lc,invertible,ach_err

# ================================================================
#  BATCH METRICS
# ================================================================

def batch_midpoint_quality(M1, M2, M1i, M2i, lc):
    """Midpoint chroma preservation — THE novel metric."""
    P = M1.shape[0]
    lab1 = fwd(MID_XYZ1, M1, M2, lc)  # (P, N_MID, 3)
    lab2 = fwd(MID_XYZ2, M1, M2, lc)
    lab_mid = 0.5 * (lab1 + lab2)

    xyz_mid = inv(lab_mid, M1i, M2i, lc)  # (P, N_MID, 3)
    # CIE Lab chroma of midpoint
    cielab_mid = _xyz_to_cielab(xyz_mid.reshape(P*N_MID, 3)).reshape(P, N_MID, 3)
    C_mid = (cielab_mid[:,:,1]**2 + cielab_mid[:,:,2]**2).sqrt()  # (P, N_MID)

    # Preservation ratio: C_mid / C_endpoint_avg
    # Only for pairs where endpoints have chroma (skip achromatic pairs)
    mask = C_END_AVG > 5.0  # (N_MID,)
    C_avg = C_END_AVG.unsqueeze(0).expand(P, -1)
    ratio = torch.where(mask.unsqueeze(0), C_mid / C_avg.clamp(min=1), torch.ones(P, N_MID, device=dev))

    # We want ratio close to 1 (or higher). Penalize ratio < 0.5 strongly.
    # Loss: mean of max(0, 1 - ratio) — lower is better
    loss = (1.0 - ratio).clamp(min=0).mean(dim=1)
    return loss

def batch_cv(M1, M2, M1i, M2i, lc):
    P=M1.shape[0]
    lab1=fwd(PT[:,0],M1,M2,lc); lab2=fwd(PT[:,1],M1,M2,lc)
    t=T_ST.view(1,1,-1,1)
    labs=lab1.unsqueeze(2)+t*(lab2-lab1).unsqueeze(2)
    lf=labs.reshape(P,-1,3)
    xyz=inv(lf,M1i,M2i,lc)
    lin=(xyz@MSi.T).clamp(0,1); s8=(l2s(lin)*255).round()/255.0; xb=s2l(s8)@MS.T
    r=xb.clamp(min=1e-10)/D65.view(1,1,3)
    f=torch.where(r>0.008856,r.pow(1./3.),7.787*r+16./116.)
    cl=torch.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],dim=-1)
    cl=cl.reshape(P,N_PAIRS,N_ST+1,3)
    c1d,c2d=cl[:,:,:-1],cl[:,:,1:]
    dL=c2d[...,0]-c1d[...,0];C1=(c1d[...,1]**2+c1d[...,2]**2).sqrt();C2=(c2d[...,1]**2+c2d[...,2]**2).sqrt()
    dC=C2-C1;dH=((c2d[...,1]-c1d[...,1])**2+(c2d[...,2]-c1d[...,2])**2-dC**2).clamp(min=0).sqrt()
    SL=1+0.015*(c1d[...,0]-50)**2/(20+(c1d[...,0]-50)**2).sqrt();SC=1+0.045*C1;SH=1+0.015*C1
    de=((dL/SL)**2+(dC/SC)**2+(dH/SH)**2).sqrt()
    md=de.mean(2);sd=de.std(2);ok=md>0.001
    cvs=torch.where(ok,sd/md,torch.zeros_like(md))
    cnt=ok.float().sum(1).clamp(min=1)
    return (cvs*ok.float()).sum(1)/cnt

def batch_munsell_v(M1, M2, lc):
    lab=fwd(MUNSELL_GRAYS,M1,M2,lc)
    L=lab[:,:,0]; dL=L[:,1:]-L[:,:-1]
    return dL.std(dim=1)/(dL.abs().mean(dim=1)+1e-10)*100

# ================================================================
#  EVALUATE
# ================================================================

def evaluate(x_np):
    x=torch.tensor(x_np,device=dev,dtype=torch.float64)
    P=x.shape[0]; losses=torch.full((P,),999.0,device=dev)
    with torch.no_grad():
        M1,M2,M1i,M2i,lc,valid,ach=unpack(x)
        if not valid.any(): return losses.cpu().numpy()

        mid_q = batch_midpoint_quality(M1,M2,M1i,M2i,lc)
        cv = batch_cv(M1,M2,M1i,M2i,lc)
        munsv = batch_munsell_v(M1,M2,lc)

        # Hard constraints: reject if these fail
        # RT check on a small sample
        test_xyz = MID_XYZ1[:10]  # 10 colors
        test_lab = fwd(test_xyz, M1, M2, lc)
        test_rt = inv(test_lab, M1i, M2i, lc)
        rt_err = (test_rt - test_xyz.unsqueeze(0)).abs().max(dim=-1).values.max(dim=-1).values
        valid &= rt_err < 1e-6  # RT must be < 1e-6
        valid &= ach < 0.005  # achromatic must be tight

        if not valid.any():
            return losses.cpu().numpy()

        # Combined loss — balanced
        loss = 5.0*mid_q + 3.0*cv + 0.5*munsv

        # Soft penalties
        loss += torch.where(ach>0.0001, (ach-0.0001)*5000, 0.0)
        loss += torch.where(rt_err>1e-10, rt_err*1e8, 0.0)

        losses=torch.where(valid,loss,torch.full_like(loss,999.0))
    return losses.cpu().numpy()

# ================================================================
#  SEEDS
# ================================================================

OKM1=np.array([[0.818933,0.361867,-0.128860],[0.032985,0.929312,0.036146],[0.048200,0.264366,0.633852]])
OKM2=np.array([[0.2104542553,0.7936177850,-0.0040720468],[1.9779984951,-2.4285922050,0.4505937099],[0.0259040371,0.7827717662,-0.8086757660]])

def pack_m1(M1): return [M1[0,0],M1[0,1],M1[1,0],M1[1,1],M1[2,0],M1[2,1]]

def make_seeds():
    seeds=[]
    x0=np.zeros(18); x0[:6]=pack_m1(OKM1); x0[6:15]=OKM2.flatten()
    seeds.append(("oklab",x0,0.03))
    # v7b seed
    try:
        with open(os.path.join(CKPT,"v7b_nodelta.json")) as f: v7b=json.load(f)
        xv=np.zeros(18); xv[:6]=pack_m1(np.array(v7b["M1"])); xv[6:15]=np.array(v7b["M2"]).flatten()
        seeds.append(("v7b",xv,0.03))
    except: pass
    # Perturbations
    rng=np.random.RandomState(42)
    for i in range(max(0,args.seeds-len(seeds))):
        xr=x0+rng.randn(18)*0.05
        seeds.append((f"rnd{i}",xr,0.05))
    return seeds

# ================================================================
#  CMA-ES
# ================================================================

def run_seed(label,x0,sigma):
    print(f"\n  Seed: {label}",flush=True)
    opts=cma.CMAOptions()
    opts.set("maxiter",args.gens);opts.set("popsize",args.pop)
    opts.set("tolfun",1e-15);opts.set("tolx",1e-15);opts.set("verbose",-1)
    es=cma.CMAEvolutionStrategy(x0,sigma,opts)
    best_loss,best_x=999.0,x0.copy()
    gen=0
    while not es.stop():
        sols=es.ask(); fits=evaluate(np.array(sols))
        es.tell(sols,fits.tolist())
        idx=np.argmin(fits)
        if fits[idx]<best_loss: best_loss=fits[idx]; best_x=np.array(sols[idx]).copy()
        gen+=1
        if gen%20==0 or gen<=3:
            x_t=torch.tensor(best_x.reshape(1,18),device=dev)
            M1,M2,M1i,M2i,lc,v,_=unpack(x_t)
            if v.any():
                mq=batch_midpoint_quality(M1,M2,M1i,M2i,lc).item()
                cv=batch_cv(M1,M2,M1i,M2i,lc).item()
                mv=batch_munsell_v(M1,M2,lc).item()
                print(f"  gen {gen:4d}  loss={best_loss:.2f}  MidQ={mq:.4f}  CV={cv:.3f}  MunsV={mv:.1f}%",flush=True)

    # Save
    x_t=torch.tensor(best_x.reshape(1,18),device=dev)
    M1,M2,M1i,M2i,lc,v,_=unpack(x_t)
    if v.any():
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        out={"M1":M1[0].cpu().tolist(),"M2":M2[0].cpu().tolist(),
             "gamma":[1/3,1/3,1/3],"L_corr":lc[0].cpu().tolist(),
             "architecture":"MidpointOptimized","loss":float(best_loss),
             "generation":gen,"seed":label}
        path=os.path.join(CKPT,f"midpoint_{label}_{ts}.json")
        with open(path,"w") as f: json.dump(out,f,indent=2)
        mq=batch_midpoint_quality(M1,M2,M1i,M2i,lc).item()
        cv=batch_cv(M1,M2,M1i,M2i,lc).item()
        mv=batch_munsell_v(M1,M2,lc).item()
        print(f"  FINAL: loss={best_loss:.4f} MidQ={mq:.4f} CV={cv:.4f} MunsV={mv:.2f}%",flush=True)
        return best_loss,path
    return best_loss,None

if __name__=="__main__":
    print(f"\n  Midpoint Quality Optimization (18 params)",flush=True)
    seeds=make_seeds(); results=[]
    for label,x0,sigma in seeds:
        loss,path=run_seed(label,x0,sigma); results.append((label,loss,path))
    results.sort(key=lambda r:r[1])
    print(f"\nBest: {results[0][0]} (loss={results[0][1]:.4f})")
