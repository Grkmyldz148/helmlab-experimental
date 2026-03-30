#!/usr/bin/env python3
"""Novel Transfer Function Search — beyond cbrt.

Tests 3 fundamentally different transfer functions:
  A. log(1+k*x)/log(1+k)  — logarithmic, k controls compression
  B. x/(x+k)              — rational/Michaelis-Menten, k controls saturation
  C. sinh(a*x)/sinh(a)    — S-curve, a controls shape

Each with full M1(6)+M2(9)+transfer_param(1)+L_corr(3) = 19 params.
Optimized for: midpoint_quality + gradient_CV + munsell_V + achromatic.

This is NOT cbrt. This is genuinely different.
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
pa.add_argument("--seeds", type=int, default=6)
pa.add_argument("--arch", type=str, default="all", choices=["log","rational","sinh","hybrid","all"])
args = pa.parse_args()

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

# Pairs
for _d in [os.path.join(ROOT,"colorbench"), os.path.join(ROOT,"space-test-project")]:
    if os.path.isdir(os.path.join(_d,"core")):
        sys.path.insert(0,_d)
        from core.pairs import generate_all_pairs
        PT,_ = generate_all_pairs(dev)
        print(f"Pairs: {PT.shape[0]}", flush=True)
        break

N_PAIRS=PT.shape[0]; N_ST=25; T_ST=torch.linspace(0,1,N_ST+1,device=dev)

# Midpoint test pairs
_mp = []
for h in range(0,360,30):
    r1,g1,b1 = [1,0,0] if h==0 else [0,0,0]  # placeholder
    import colorsys
    r1,g1,b1 = colorsys.hsv_to_rgb(h/360,1,1)
    r2,g2,b2 = colorsys.hsv_to_rgb(((h+180)%360)/360,1,1)
    _mp.append(([r1,g1,b1],[r2,g2,b2]))
for rgb in [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]]:
    _mp.append((rgb,[1,1,1]))
_mp.append(([1,0.6,0.2],[0.2,0.4,1]))
_mp.append(([1,0.5,0.5],[0.5,1,1]))
_mp.append(([0.6,0.2,0.8],[0.9,0.8,0.2]))

MID_XYZ1=torch.zeros(len(_mp),3,device=dev)
MID_XYZ2=torch.zeros(len(_mp),3,device=dev)
for i,(rgb1,rgb2) in enumerate(_mp):
    MID_XYZ1[i]=MS@s2l(torch.tensor(rgb1,device=dev,dtype=torch.float64))
    MID_XYZ2[i]=MS@s2l(torch.tensor(rgb2,device=dev,dtype=torch.float64))

def _xyz_to_cielab(xyz):
    r=xyz/D65; delta3=(6./29.)**3
    f=torch.where(r>delta3,r.pow(1./3.),r/(3*(6./29.)**2)+4./29.)
    return torch.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],dim=-1)

CL1=_xyz_to_cielab(MID_XYZ1); CL2=_xyz_to_cielab(MID_XYZ2)
C_END1=(CL1[:,1]**2+CL1[:,2]**2).sqrt(); C_END2=(CL2[:,1]**2+CL2[:,2]**2).sqrt()
C_END_AVG=0.5*(C_END1+C_END2); N_MID=len(_mp)

# Munsell
MY={1:0.01221,2:0.03126,3:0.06552,4:0.12,5:0.1977,6:0.30049,7:0.4306,8:0.591,9:0.7866}
MUNSELL_GRAYS=torch.stack([D65*MY[v] for v in range(1,10)]).to(dev)

print(f"Midpoint pairs: {N_MID}", flush=True)

# ================================================================
#  TRANSFER FUNCTIONS (the novel part!)
# ================================================================

def log_fwd(x, k):
    """f(x) = log(1 + k*x) / log(1 + k). k>0 controls compression."""
    k = k.view(-1,1,1)
    return torch.log1p(k * x.clamp(min=0)) / torch.log1p(k)

def log_inv(y, k):
    k = k.view(-1,1,1)
    return (torch.expm1(y * torch.log1p(k))) / k

def rational_fwd(x, k):
    """f(x) = x/(x+k) * (1+k). Normalized so f(1)=1. k>0."""
    k = k.view(-1,1,1)
    return x.clamp(min=0) / (x.clamp(min=0) + k) * (1 + k)

def rational_inv(y, k):
    k = k.view(-1,1,1)
    y_norm = y / (1 + k)
    return k * y_norm / (1.0 - y_norm).clamp(min=1e-15)

def sinh_fwd(x, a):
    """f(x) = sinh(a*x) / sinh(a). a>0."""
    a = a.view(-1,1,1)
    return torch.sinh(a * x.clamp(min=0)) / torch.sinh(a)

def sinh_inv(y, a):
    a = a.view(-1,1,1)
    return torch.arcsinh(y * torch.sinh(a)) / a

def hybrid_fwd(x, params):
    """f(x) = (1-w)*x^(1/3) + w*log(1+k*x)/log(1+k). Best of both worlds."""
    # params encodes both w and k: params = w * 10 + k_encoded
    # Split: w = sigmoid(first_half), k = exp(second_half)
    # Actually simpler: use params as k, fix w=0.3
    k = params.view(-1,1,1)
    w = 0.3  # fixed blend weight
    cbrt_part = torch.sign(x) * x.abs().clamp(min=1e-30).pow(1.0/3.0)
    log_part = torch.log1p(k * x.clamp(min=0)) / torch.log1p(k)
    return (1-w) * cbrt_part + w * log_part

def hybrid_inv(y, params):
    k = params.view(-1,1,1)
    w = 0.3
    # Newton iteration: solve (1-w)*x^(1/3) + w*log(1+kx)/log(1+k) = y
    x = y.clamp(min=0).pow(3)  # initial guess from cbrt inverse
    for _ in range(20):
        x = x.clamp(min=1e-30)
        f_val = (1-w) * x.pow(1.0/3.0) + w * torch.log1p(k*x) / torch.log1p(k) - y
        df = (1-w) / (3 * x.pow(2.0/3.0).clamp(min=1e-30)) + w * k / ((1+k*x) * torch.log1p(k))
        x = (x - f_val / df.clamp(min=1e-20)).clamp(min=0)
    return x

ARCHS = {
    "log": (log_fwd, log_inv, "Log"),
    "rational": (rational_fwd, rational_inv, "Rational"),
    "sinh": (sinh_fwd, sinh_inv, "Sinh"),
    "hybrid": (hybrid_fwd, hybrid_inv, "Hybrid"),
}

# ================================================================
#  FORWARD / INVERSE
# ================================================================

def fwd(xyz, M1, M2, lc, tf_fwd, tf_param):
    lms = (xyz.unsqueeze(0) @ M1.transpose(-1,-2)).clamp(min=0)
    lms_c = tf_fwd(lms, tf_param)
    lab = torch.bmm(lms_c, M2.transpose(-1,-2))
    L=lab[...,0:1]
    c1=lc[:,0:1].unsqueeze(1);c2=lc[:,1:2].unsqueeze(1);c3=lc[:,2:3].unsqueeze(1)
    t=L*(1-L); L_new=L+c1*t+c2*t*(2*L-1)+c3*L**2*(1-L)**2
    return torch.cat([L_new,lab[...,1:2],lab[...,2:3]],dim=-1)

def inv(lab, M1i, M2i, lc, tf_inv, tf_param):
    L1=lab[...,0:1]
    c1=lc[:,0:1].unsqueeze(1);c2=lc[:,1:2].unsqueeze(1);c3=lc[:,2:3].unsqueeze(1)
    L=L1.clone()
    for _ in range(12):
        t=L*(1-L); f=L+c1*t+c2*t*(2*L-1)+c3*L**2*(1-L)**2-L1
        df=1+c1*(1-2*L)+c2*(6*L**2-6*L+1)+c3*2*L*(1-L)*(1-2*L)
        L=L-f/df.clamp(min=1e-12)
    raw=torch.cat([L,lab[...,1:2],lab[...,2:3]],dim=-1)
    lms_c=torch.bmm(raw,M2i.transpose(-1,-2))
    lms=tf_inv(lms_c, tf_param).clamp(min=0)
    return torch.bmm(lms, M1i.transpose(-1,-2))

# ================================================================
#  UNPACK (19 params: M1(6)+M2(9)+k(1)+L_corr(3))
# ================================================================

def unpack(x, arch_key):
    P=x.shape[0]
    M1=torch.zeros(P,3,3,device=dev)
    M1[:,0,0]=x[:,0];M1[:,0,1]=x[:,1];M1[:,1,0]=x[:,2];M1[:,1,1]=x[:,3]
    M1[:,2,0]=x[:,4];M1[:,2,1]=x[:,5]
    M1[:,0,2]=(1-M1[:,0,0]*D65[0]-M1[:,0,1]*D65[1])/D65[2]
    M1[:,1,2]=(1-M1[:,1,0]*D65[0]-M1[:,1,1]*D65[1])/D65[2]
    M1[:,2,2]=(1-M1[:,2,0]*D65[0]-M1[:,2,1]*D65[1])/D65[2]
    lms_d65=(D65.unsqueeze(0).unsqueeze(0)@M1.transpose(-1,-2)).squeeze(1)
    valid=(lms_d65>0.01).all(dim=1)

    M2=torch.zeros(P,3,3,device=dev)
    M2[:,0,0]=x[:,6];M2[:,0,1]=x[:,7];M2[:,0,2]=x[:,8]
    M2[:,1,0]=x[:,9];M2[:,1,1]=x[:,10];M2[:,1,2]=x[:,11]
    M2[:,2,0]=x[:,12];M2[:,2,1]=x[:,13];M2[:,2,2]=x[:,14]

    # Transfer param: must be positive
    if arch_key == "log":
        k = torch.exp(x[:,15].clamp(0, 1.6))  # k in [1.0, 5.0]
    elif arch_key == "rational":
        k = torch.exp(x[:,15].clamp(-3, 1))  # k in [0.05, 2.7]
    elif arch_key == "sinh":
        k = torch.exp(x[:,15].clamp(-1, 2))  # a in [0.37, 7.4]
    elif arch_key == "hybrid":
        k = torch.exp(x[:,15].clamp(-0.5, 1.6))  # k in [0.6, 5.0]
    else:
        k = torch.ones(P, device=dev)

    lc=torch.stack([x[:,16].clamp(-0.3,0.3),x[:,17].clamp(-0.3,0.3),x[:,18].clamp(-0.5,0.5)],dim=1)

    # Apply transfer to D65
    tf_fwd = ARCHS[arch_key][0]
    lms_c_d65 = tf_fwd(lms_d65.unsqueeze(1), k).squeeze(1)  # (P, 3)
    
    # STRUCTURAL ACHROMATIC: project M2 a,b rows orthogonal to lms_c_d65
    for row_idx in [1, 2]:
        row = M2[:, row_idx, :]  # (P, 3)
        dot = (row * lms_c_d65).sum(dim=1, keepdim=True)
        norm2 = (lms_c_d65 * lms_c_d65).sum(dim=1, keepdim=True)
        proj = dot / norm2.clamp(min=1e-20)
        M2[:, row_idx, :] = row - proj * lms_c_d65

    lab_d65 = torch.bmm(lms_c_d65.unsqueeze(1), M2.transpose(-1,-2)).squeeze(1)
    ach = (lab_d65[:,1]**2+lab_d65[:,2]**2).sqrt()
    white_L = lab_d65[:,0]
    t_w=white_L*(1-white_L)
    white_L_corr=white_L+lc[:,0]*t_w+lc[:,1]*t_w*(2*white_L-1)+lc[:,2]*white_L**2*(1-white_L)**2

    valid &= (white_L_corr>0.95)&(white_L_corr<1.05)

    M1i=torch.zeros_like(M1);M2i=torch.zeros_like(M2)
    det1=torch.linalg.det(M1);det2=torch.linalg.det(M2)
    inv_ok=(det1.abs()>1e-10)&(det2.abs()>1e-10)&valid
    good=inv_ok.nonzero(as_tuple=True)[0]
    if good.numel()>0:
        M1i[good]=torch.linalg.inv(M1[good])
        M2i[good]=torch.linalg.inv(M2[good])
    return M1,M2,M1i,M2i,k,lc,inv_ok,ach

# ================================================================
#  METRICS
# ================================================================

def batch_midpoint(M1,M2,M1i,M2i,k,lc,arch_key):
    P=M1.shape[0]; tf_f,tf_i,_=ARCHS[arch_key]
    lab1=fwd(MID_XYZ1,M1,M2,lc,tf_f,k)
    lab2=fwd(MID_XYZ2,M1,M2,lc,tf_f,k)
    lab_mid=0.5*(lab1+lab2)
    xyz_mid=inv(lab_mid,M1i,M2i,lc,tf_i,k)

    # CIE Lab of midpoint and endpoints
    cl_mid=_xyz_to_cielab(xyz_mid.reshape(P*N_MID,3)).reshape(P,N_MID,3)
    cl1=_xyz_to_cielab(MID_XYZ1).unsqueeze(0).expand(P,-1,-1)
    cl2=_xyz_to_cielab(MID_XYZ2).unsqueeze(0).expand(P,-1,-1)

    C_mid=(cl_mid[:,:,1]**2+cl_mid[:,:,2]**2).sqrt()
    C1=(cl1[:,:,1]**2+cl1[:,:,2]**2).sqrt()
    C2=(cl2[:,:,1]**2+cl2[:,:,2]**2).sqrt()
    C_avg=0.5*(C1+C2)

    # 1. Chroma preservation (existing)
    mask=C_avg>5.0
    chroma_ratio=torch.where(mask,C_mid/C_avg.clamp(min=1),torch.ones_like(C_mid))
    chroma_loss=(1.0-chroma_ratio).clamp(min=0)

    # 2. HUE preservation (NEW): midpoint hue should be between endpoint hues
    h_mid=torch.atan2(cl_mid[:,:,2],cl_mid[:,:,1])
    h1=torch.atan2(cl1[:,:,2],cl1[:,:,1])
    h2=torch.atan2(cl2[:,:,2],cl2[:,:,1])
    # Expected hue: linear interp of endpoint hues (shortest arc)
    dh=h2-h1
    dh=torch.where(dh>3.14159,dh-2*3.14159,dh)
    dh=torch.where(dh<-3.14159,dh+2*3.14159,dh)
    h_expected=h1+0.5*dh
    # Hue error
    h_err=torch.atan2(torch.sin(h_mid-h_expected),torch.cos(h_mid-h_expected)).abs()
    h_err_deg=h_err*180/3.14159
    # Only penalize chromatic midpoints (achromatic midpoint hue is undefined)
    h_loss=torch.where(C_mid>3.0,h_err_deg/90.0,torch.zeros_like(h_err_deg))  # normalize to ~1

    # Combined: both chroma AND hue must be good
    return (chroma_loss + 0.5*h_loss).mean(dim=1)

def batch_cv(M1,M2,M1i,M2i,k,lc,arch_key):
    P=M1.shape[0];tf_f,tf_i,_=ARCHS[arch_key]
    lab1=fwd(PT[:,0],M1,M2,lc,tf_f,k);lab2=fwd(PT[:,1],M1,M2,lc,tf_f,k)
    t=T_ST.view(1,1,-1,1)
    labs=lab1.unsqueeze(2)+t*(lab2-lab1).unsqueeze(2)
    lf=labs.reshape(P,-1,3)
    xyz=inv(lf,M1i,M2i,lc,tf_i,k)
    lin=(xyz@MSi.T).clamp(0,1);s8=(l2s(lin)*255).round()/255.0;xb=s2l(s8)@MS.T
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

def batch_munsell(M1,M2,k,lc,arch_key):
    tf_f,_,_=ARCHS[arch_key]
    lab=fwd(MUNSELL_GRAYS,M1,M2,lc,tf_f,k)
    L=lab[:,:,0];dL=L[:,1:]-L[:,:-1]
    return dL.std(dim=1)/(dL.abs().mean(dim=1)+1e-10)*100

# ================================================================
#  EVALUATE
# ================================================================

def evaluate(x_np, arch_key):
    x=torch.tensor(x_np,device=dev,dtype=torch.float64)
    P=x.shape[0];losses=torch.full((P,),999.0,device=dev)
    with torch.no_grad():
        M1,M2,M1i,M2i,k,lc,valid,ach=unpack(x,arch_key)
        if not valid.any(): return losses.cpu().numpy()

        mid_q=batch_midpoint(M1,M2,M1i,M2i,k,lc,arch_key)
        cv=batch_cv(M1,M2,M1i,M2i,k,lc,arch_key)
        munsv=batch_munsell(M1,M2,k,lc,arch_key)

        loss=4.0*mid_q+3.0*cv+1.5*munsv
        loss+=torch.where(ach>0.001,ach*500,0.0)

        losses=torch.where(valid,loss,torch.full_like(loss,999.0))
    return losses.cpu().numpy()

# ================================================================
#  SEEDS + RUN
# ================================================================

OKM1=np.array([[0.818933,0.361867,-0.128860],[0.032985,0.929312,0.036146],[0.048200,0.264366,0.633852]])
OKM2=np.array([[0.2104542553,0.7936177850,-0.0040720468],[1.9779984951,-2.4285922050,0.4505937099],[0.0259040371,0.7827717662,-0.8086757660]])

def pack_m1(M): return [M[0,0],M[0,1],M[1,0],M[1,1],M[2,0],M[2,1]]

def make_seeds(arch_key):
    seeds=[]
    x0=np.zeros(19);x0[:6]=pack_m1(OKM1);x0[6:15]=OKM2.flatten()
    if arch_key=="log": x0[15]=np.log(3.0)  # k=3
    elif arch_key=="rational": x0[15]=np.log(0.5)  # k=0.5
    elif arch_key=="sinh": x0[15]=np.log(1.0)  # a=1
    elif arch_key=="hybrid": x0[15]=np.log(2.0)  # k=2
    seeds.append(("oklab",x0,0.08))
    rng=np.random.RandomState(42)
    for i in range(args.seeds-1):
        xr=x0+rng.randn(19)*0.05;seeds.append((f"rnd{i}",xr,0.10))
    return seeds

def run_arch(arch_key):
    _,_,arch_name=ARCHS[arch_key]
    print(f"\n{'='*60}\n  Architecture: {arch_name} ({arch_key})\n{'='*60}",flush=True)
    seeds=make_seeds(arch_key);results=[]
    for label,x0,sigma in seeds:
        print(f"\n  Seed: {label}",flush=True)
        opts=cma.CMAOptions()
        opts.set("maxiter",args.gens);opts.set("popsize",args.pop)
        opts.set("tolfun",1e-15);opts.set("tolx",1e-15);opts.set("verbose",-1)
        es=cma.CMAEvolutionStrategy(x0,sigma,opts)
        best_loss,best_x=999.0,x0.copy();gen=0
        while not es.stop():
            sols=es.ask();fits=evaluate(np.array(sols),arch_key)
            es.tell(sols,fits.tolist())
            idx=np.argmin(fits)
            if fits[idx]<best_loss:best_loss=fits[idx];best_x=np.array(sols[idx]).copy()
            gen+=1
            if gen%20==0 or gen<=3:
                x_t=torch.tensor(best_x.reshape(1,19),device=dev)
                M1,M2,M1i,M2i,k,lc,v,_=unpack(x_t,arch_key)
                if v.any():
                    mq=batch_midpoint(M1,M2,M1i,M2i,k,lc,arch_key).item()
                    cv=batch_cv(M1,M2,M1i,M2i,k,lc,arch_key).item()
                    mv=batch_munsell(M1,M2,k,lc,arch_key).item()
                    print(f"  gen {gen:4d}  loss={best_loss:.2f}  MidQ={mq:.4f}  CV={cv:.3f}  MunsV={mv:.1f}%  k={k.item():.3f}",flush=True)

        # Save
        x_t=torch.tensor(best_x.reshape(1,19),device=dev)
        M1,M2,M1i,M2i,k,lc,v,_=unpack(x_t,arch_key)
        if v.any():
            ts=datetime.now().strftime("%Y%m%d_%H%M%S")
            out={"M1":M1[0].cpu().tolist(),"M2":M2[0].cpu().tolist(),
                 "transfer":arch_key,"transfer_param":k.item(),
                 "gamma":[1/3,1/3,1/3],  # dummy for compat
                 "L_corr":lc[0].cpu().tolist(),
                 "architecture":f"NovelTransfer_{arch_name}",
                 "loss":float(best_loss),"seed":label}
            path=os.path.join(CKPT,f"transfer_{arch_key}_{label}_{ts}.json")
            with open(path,"w") as f:json.dump(out,f,indent=2)
            mq=batch_midpoint(M1,M2,M1i,M2i,k,lc,arch_key).item()
            cv=batch_cv(M1,M2,M1i,M2i,k,lc,arch_key).item()
            mv=batch_munsell(M1,M2,k,lc,arch_key).item()
            print(f"  FINAL: loss={best_loss:.4f} MidQ={mq:.4f} CV={cv:.4f} MunsV={mv:.2f}% k={k.item():.4f}",flush=True)
            results.append((label,best_loss,path))
        else:
            results.append((label,best_loss,None))
    return results

# ================================================================

if __name__=="__main__":
    print(f"\n  Novel Transfer Function Search\n  Gens={args.gens} Pop={args.pop}",flush=True)
    archs = ["log","rational","sinh","hybrid"] if args.arch=="all" else [args.arch]
    all_results={}
    for arch in archs:
        all_results[arch]=run_arch(arch)
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for arch,results in all_results.items():
        results.sort(key=lambda r:r[1])
        print(f"\n  {arch}: best={results[0][0]} loss={results[0][1]:.4f}")
