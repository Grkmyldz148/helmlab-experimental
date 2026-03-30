"""Next-gen GenSpace optimization: comprehensive objective from day 1.

Key differences from all previous attempts:
- ALL metrics in objective simultaneously (not patching after)
- Multiple seeds: v14, OKLab, HPE, random
- Cusp scan at physical yellow (not space hue)
- Hue drift on ALL pairs (not just primaries)
- Grid extends to L=1.05
- Hierarchical constraints: hard reject -> soft penalty -> primary objective
- Both shared gamma (13 params) and per-channel gamma (15 params) modes
- Phase 2 refinement of top 3 seeds with smaller sigma
- Immediate checkpoint saving on improvement
- Comprehensive .md report generation

Usage:
  python scripts/optimize_nextgen.py                    # 13-param shared gamma
  python scripts/optimize_nextgen.py --per-channel-gamma  # 15-param per-channel gamma

This script does NOT promise to solve yellow cusp.
It explores whether comprehensive optimization finds a better trade-off.
"""
import json, time, math, numpy as np, torch, sys, os, argparse, platform
from datetime import datetime

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = "CPU"
if torch.cuda.is_available():
    device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device_name = "MPS (Apple Silicon)"
print(f"Device: {device} ({device_name})", flush=True)
import cma

# ── CLI args ──
parser = argparse.ArgumentParser(description="Next-gen GenSpace optimization")
parser.add_argument("--per-channel-gamma", action="store_true",
                    help="Use per-channel gamma (15 params) instead of shared gamma (13 params)")
args = parser.parse_args()

PER_CHANNEL_GAMMA = args.per_channel_gamma
N_PARAMS = 15 if PER_CHANNEL_GAMMA else 13
MODE_STR = "per-channel gamma (15 params)" if PER_CHANNEL_GAMMA else "shared gamma (13 params)"
print(f"Mode: {MODE_STR}", flush=True)

# ── Checkpoint directory ──
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

D65 = np.array([0.95047, 1.0, 1.08883])
M_S = torch.tensor([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]], device=device)
M_Si = torch.linalg.inv(M_S)
D65_T = torch.tensor(D65, device=device)
M_S_np = np.array([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]])
M_Si_np = np.linalg.inv(M_S_np)

V14_M1 = np.array([[0.7583761294836658,0.38380162590825084,-0.09608055040602373],[0.12671393631532843,0.8421628149123207,0.03434823621506485],[0.07639223722200054,0.258943526275451,0.6139139663787314]])
V14_M2 = np.array([[0.10058070589596230,1.01558970993941444,-0.11617041583537688],[2.36157646996164416,-2.44099737506293479,0.07942090510129070],[0.04565327074453784,0.81875488445424471,-0.86440815519878267]])
OK_M1s = np.array([[0.4122214708,0.5363325363,0.0514459929],[0.2119034982,0.6806995451,0.1073969566],[0.0883024619,0.2817188376,0.6299787005]])
OK_M1 = OK_M1s @ np.linalg.inv(M_S_np)
OK_M2 = np.array([[0.2104542553,0.7936177850,-0.0040720468],[1.9779984951,-2.4285922050,0.4505937099],[0.0259040371,0.7827717662,-0.8086757660]])
# Hunt-Pointer-Estevez (physiological starting point)
HPE_M1 = np.array([[0.38971,0.68898,-0.07868],[-0.22981,1.18340,0.04641],[0.00000,0.00000,1.00000]])

def scbrt(x): return torch.sign(x)*torch.abs(x).pow(1./3.)
def s2l(c): return torch.where(c<=0.04045,c/12.92,((c+0.055)/1.055).pow(2.4))
def l2s(c): return torch.where(c<=0.0031308,c*12.92,1.055*c.clamp(min=1e-10).pow(1./2.4)-0.055)
def s2l_np(c): return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def l2s_np(c): return np.where(c<=0.0031308,c*12.92,1.055*np.maximum(c,1e-10)**(1/2.4)-0.055)

# Per-channel gamma application (torch)
def apply_gamma(lms, gamma):
    """Apply per-channel or shared gamma to LMS values."""
    if isinstance(gamma, (list, tuple, np.ndarray, torch.Tensor)) and len(gamma) == 3:
        gamma_t = torch.tensor(gamma, device=lms.device, dtype=lms.dtype) if not isinstance(gamma, torch.Tensor) else gamma
        return torch.sign(lms) * torch.abs(lms).pow(gamma_t)
    else:
        return torch.sign(lms) * torch.abs(lms).pow(gamma)

def apply_gamma_inv(lms_c, gamma):
    """Inverse of apply_gamma."""
    if isinstance(gamma, (list, tuple, np.ndarray, torch.Tensor)) and len(gamma) == 3:
        gamma_t = torch.tensor(gamma, device=lms_c.device, dtype=lms_c.dtype) if not isinstance(gamma, torch.Tensor) else gamma
        inv_gamma = 1.0 / gamma_t
        return torch.sign(lms_c) * torch.abs(lms_c).pow(inv_gamma)
    else:
        return torch.sign(lms_c) * torch.abs(lms_c).pow(1.0 / gamma)

def apply_gamma_np(lms, gamma):
    """Apply per-channel or shared gamma (numpy)."""
    if isinstance(gamma, (list, tuple, np.ndarray)) and len(gamma) == 3:
        gamma_a = np.array(gamma)
        return np.sign(lms) * np.abs(lms)**gamma_a
    else:
        return np.sign(lms) * np.abs(lms)**gamma

def apply_gamma_inv_np(lms_c, gamma):
    """Inverse of apply_gamma (numpy)."""
    if isinstance(gamma, (list, tuple, np.ndarray)) and len(gamma) == 3:
        inv_gamma = 1.0 / np.array(gamma)
        return np.sign(lms_c) * np.abs(lms_c)**inv_gamma
    else:
        return np.sign(lms_c) * np.abs(lms_c)**(1.0/gamma)


# ── Training pairs (same as production test) ──
pairs_list = []
prims = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[0,0,0]]
for i in range(len(prims)):
    for j in range(i+1,len(prims)): pairs_list.append((prims[i],prims[j]))
for g1 in [0.0,0.2,0.4,0.6,0.8,1.0]:
    for g2 in [g1+0.2,g1+0.4]:
        if g2<=1.0: pairs_list.append(([g1]*3,[g2]*3))
rng=np.random.RandomState(42)
for _ in range(80): pairs_list.append((rng.rand(3).tolist(),rng.rand(3).tolist()))
pt=torch.zeros(len(pairs_list),2,3,device=device)
pair_xyz_np = []
for i,(c1,c2) in enumerate(pairs_list):
    x1 = M_S@s2l(torch.tensor(c1,device=device))
    x2 = M_S@s2l(torch.tensor(c2,device=device))
    pt[i,0]=x1; pt[i,1]=x2
    pair_xyz_np.append((M_S_np@s2l_np(np.array(c1)), M_S_np@s2l_np(np.array(c2))))

N_ST=25; T_ST=torch.linspace(0,1,N_ST+1,device=device)

# ── GPU Metrics ──
def gpu_cv(M1, M2, gamma=1./3.):
    M1i,M2i=torch.linalg.inv(M1),torch.linalg.inv(M2)
    N=pt.shape[0]
    l1=apply_gamma(pt[:,0]@M1.T, gamma)@M2.T
    l2=apply_gamma(pt[:,1]@M1.T, gamma)@M2.T
    t=T_ST.view(1,-1,1); labs=l1.unsqueeze(1)+t*(l2-l1).unsqueeze(1)
    lf=labs.reshape(-1,3); lc=lf@M2i.T
    lm=apply_gamma_inv(lc, gamma)
    lin=(lm@M1i.T)@M_Si.T; s8=(l2s(lin.clamp(0,1))*255).round()/255.
    xb=s2l(s8)@M_S.T; r=xb.clamp(min=1e-10)/D65_T
    f=torch.where(r>0.008856,r.pow(1./3.),7.787*r+16./116.)
    cl=torch.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],dim=-1).reshape(N,N_ST+1,3)
    c1,c2=cl[:,:-1],cl[:,1:]
    dL=c2[...,0]-c1[...,0]; C1=(c1[...,1]**2+c1[...,2]**2).sqrt(); C2=(c2[...,1]**2+c2[...,2]**2).sqrt()
    dC=C2-C1; dH=((c2[...,1]-c1[...,1])**2+(c2[...,2]-c1[...,2])**2-dC**2).clamp(min=0).sqrt()
    SL=1+0.015*(c1[...,0]-50)**2/(20+(c1[...,0]-50)**2).sqrt(); SC=1+0.045*C1; SH=1+0.015*C1
    de=((dL/SL)**2+(dC/SC)**2+(dH/SH)**2).sqrt()
    md=de.mean(1); sd=de.std(1); v=md>0.001
    cvs=torch.where(v,sd/md,torch.zeros_like(md))
    return cvs[v].mean().item() if v.any() else 0.99

def gpu_hue(M1, M2, gamma=1./3.):
    prs=torch.tensor([[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1]],dtype=torch.float64,device=device)
    exp=torch.tensor([0,60,120,180,240,300],dtype=torch.float64,device=device)
    lab=apply_gamma(s2l(prs)@M_S.T@M1.T, gamma)@M2.T
    h=torch.atan2(lab[:,2],lab[:,1])*(180/3.14159265)%360
    dh=h-exp; dh=torch.where(dh>180,dh-360,dh); dh=torch.where(dh<-180,dh+360,dh)
    return (dh**2).mean().item()

def gpu_info(M1, M2, gamma=1./3.):
    M1i,M2i=torch.linalg.inv(M1),torch.linalg.inv(M2)
    # Yellow
    yl=apply_gamma((M_S@s2l(torch.tensor([1.,1.,0.],device=device)))@M1.T, gamma)@M2.T
    yL,yC=yl[0].item(),(yl[1]**2+yl[2]**2).sqrt().item()
    # Blue->White midpoint
    bx=M_S@s2l(torch.tensor([0.,0.,1.],device=device))
    wx=M_S@s2l(torch.tensor([1.,1.,1.],device=device))
    bl=apply_gamma(bx@M1.T, gamma)@M2.T; wl=apply_gamma(wx@M1.T, gamma)@M2.T
    ml=(bl+wl)/2; lc=ml@M2i.T; lm=apply_gamma_inv(lc, gamma); mx=lm@M1i.T
    ms=l2s((M_Si@mx).clamp(0,1))
    bw=ms[1].item()/max(ms[0].item(),0.01)
    # Red->White midpoint G-B
    rx=M_S@s2l(torch.tensor([1.,0.,0.],device=device))
    rl=apply_gamma(rx@M1.T, gamma)@M2.T; ml2=(rl+wl)/2
    lc2=ml2@M2i.T; lm2=apply_gamma_inv(lc2, gamma); mx2=lm2@M1i.T
    ms2=l2s((M_Si@mx2).clamp(0,1))
    rw_gb=ms2[1].item()-ms2[2].item()
    # Primary L range
    ps=torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]],dtype=torch.float64,device=device)
    pl=apply_gamma(s2l(ps)@M_S.T@M1.T, gamma)@M2.T
    plr=(pl[:,0].max()-pl[:,0].min()).item()
    return {'yL':yL,'yC':yC,'bw':bw,'rw_gb':rw_gb,'plr':plr}

# ── Cusp scan (yellow region, GPU) ──
CUSP_Ls = torch.linspace(0.3, 1.05, 90, device=device)
CUSP_Cs = torch.linspace(0.001, 0.4, 60, device=device)
CUSP_Le = CUSP_Ls.view(90,1).expand(90,60)
CUSP_Ce = CUSP_Cs.view(1,60).expand(90,60)
CUSP_Ce_v = CUSP_Cs.view(1,60).expand(90,60)

def gpu_cusp(M1, M2, hue_degs=[75,80,85,90,95], gamma=1./3.):
    """Returns dict: {hue_deg: (cusp_L, cusp_C, cliff%)}"""
    M1i = torch.linalg.inv(M1); M2i = torch.linalg.inv(M2)
    results = {}
    for hd in hue_degs:
        hr = hd * 3.14159265 / 180
        ch, sh = np.cos(hr), np.sin(hr)
        lab = torch.stack([CUSP_Le, CUSP_Ce*ch, CUSP_Ce*sh], dim=-1).reshape(-1, 3)
        lc = lab @ M2i.T
        lm = apply_gamma_inv(lc, gamma)
        lin = (lm @ M1i.T) @ M_Si.T
        ok = ((lin >= -0.002).all(dim=1) & (lin <= 1.002).all(dim=1)).reshape(90, 60)
        mc, _ = torch.where(ok, CUSP_Ce_v, torch.zeros(90,60,device=device)).max(dim=1)
        ci = mc.argmax().item()
        cL = CUSP_Ls[ci].item()
        cC = mc[ci].item()
        # Cliff: chroma 2 L-steps after cusp
        if ci < 88 and cC > 0.01:
            post_C = mc[min(ci+2, 89)].item()
            cliff = (cC - post_C) / cC * 100
        else:
            cliff = 100.0
        results[hd] = (cL, cC, cliff)
    return results

# ── Hue drift (CPU, ALL pairs) ──
def cpu_hue_drift(M1_np, M2_np, n_pairs=None, gamma=1./3.):
    M1i_np, M2i_np = np.linalg.inv(M1_np), np.linalg.inv(M2_np)
    drifts = []
    pairs_to_test = pair_xyz_np if n_pairs is None else pair_xyz_np[:n_pairs]
    for x1, x2 in pairs_to_test:
        l1 = M2_np @ apply_gamma_np(M1_np@x1, gamma)
        l2 = M2_np @ apply_gamma_np(M1_np@x2, gamma)
        prev_h = None; pair_max = 0
        for t in np.linspace(0, 1, 13):
            lab = l1 + t * (l2 - l1)
            lc = M2i_np @ lab; xyz = M1i_np @ apply_gamma_inv_np(lc, gamma)
            rgb8 = np.round(l2s_np(np.clip(M_Si_np @ xyz, 0, 1)) * 255) / 255
            xyz_q = M_S_np @ s2l_np(rgb8)
            r = np.maximum(xyz_q, 1e-10) / D65
            f = np.where(r > 0.008856, r**(1/3), 7.787*r + 16/116)
            cl = np.array([116*f[1]-16, 500*(f[0]-f[1]), 200*(f[1]-f[2])])
            C_val = math.sqrt(cl[1]**2 + cl[2]**2)
            if C_val < 3.0: prev_h = None; continue
            h = math.atan2(cl[2], cl[1])
            if prev_h is not None:
                dh = abs(math.atan2(math.sin(h-prev_h), math.cos(h-prev_h))) * 180 / math.pi
                pair_max = max(pair_max, dh)
            prev_h = h
        drifts.append(pair_max)
    return np.mean(drifts), np.max(drifts)

# ── Parameterization ──
def ortho(s):
    sn=s/np.linalg.norm(s)
    v=np.array([1,0,0.]) if abs(sn[0])<0.9 else np.array([0,1,0.])
    e1=v-np.dot(v,sn)*sn; e1/=np.linalg.norm(e1); e2=np.cross(sn,e1)
    return e1,e2

def unpack(x):
    """Unpack parameter vector to M1, M2, gamma.

    Shared gamma (13 params): x[0:12] = M1+M2, gamma = 1/3
    Per-channel gamma (15 params): x[0:12] = M1+M2, x[13]=log(g2/g1), x[14]=log(g3/g1)
      gamma = [1/3, (1/3)*exp(x[13]), (1/3)*exp(x[14])]
    """
    M1=np.zeros((3,3))
    for i in range(3):
        M1[i,0]=x[2*i]; M1[i,1]=x[2*i+1]
        M1[i,2]=(1-M1[i,0]*D65[0]-M1[i,1]*D65[1])/D65[2]
    lms=M1@D65
    if np.any(lms<=0): return None,None,None

    # Determine gamma
    if PER_CHANNEL_GAMMA and len(x) >= 15:
        g1 = 1./3.
        g2 = g1 * np.exp(np.clip(x[13], -2.0, 2.0))  # clamp for stability
        g3 = g1 * np.exp(np.clip(x[14], -2.0, 2.0))
        gamma = np.array([g1, g2, g3])
        # Validate gamma range (0.05 to 2.0)
        if np.any(gamma < 0.05) or np.any(gamma > 2.0):
            return None, None, None
    else:
        gamma = 1./3.

    s=lms**( np.array(gamma) if isinstance(gamma, np.ndarray) else gamma )
    if np.linalg.norm(s)<1e-10: return None,None,None
    e1,e2=ortho(s)
    M2=np.zeros((3,3)); M2[0]=x[6:9]
    Lw=M2[0]@s
    if abs(Lw)<1e-10: return None,None,None
    M2[0]/=Lw
    M2[1]=x[9]*e1+x[10]*e2; M2[2]=x[11]*e1+x[12]*e2
    return M1, M2, gamma

def pack(M1, M2, gamma=1./3.):
    """Pack M1, M2, gamma into parameter vector."""
    n = 15 if PER_CHANNEL_GAMMA else 13
    x=np.zeros(n)
    for i in range(3): x[2*i]=M1[i,0]; x[2*i+1]=M1[i,1]
    x[6:9]=M2[0]
    lms=M1@D65
    s=lms**( np.array(gamma) if isinstance(gamma, np.ndarray) else gamma )
    e1,e2=ortho(s)
    x[9]=M2[1]@e1; x[10]=M2[1]@e2; x[11]=M2[2]@e1; x[12]=M2[2]@e2
    if PER_CHANNEL_GAMMA:
        if isinstance(gamma, (list, tuple, np.ndarray)) and len(gamma) == 3:
            g1 = gamma[0] if isinstance(gamma, np.ndarray) else gamma[0]
            x[13] = np.log(gamma[1] / g1) if g1 > 0 else 0.0
            x[14] = np.log(gamma[2] / g1) if g1 > 0 else 0.0
        else:
            x[13] = 0.0  # g2/g1 = 1 => log(1) = 0
            x[14] = 0.0  # g3/g1 = 1 => log(1) = 0
    return x

# ── Full metrics computation ──
def compute_full_metrics(M1_np, M2_np, gamma=1./3.):
    """Compute all metrics for a given M1, M2, gamma. Returns dict."""
    M1t = torch.tensor(M1_np, device=device)
    M2t = torch.tensor(M2_np, device=device)
    gamma_t = gamma
    if isinstance(gamma, np.ndarray):
        gamma_t = torch.tensor(gamma, device=device)

    with torch.no_grad():
        cv = gpu_cv(M1t, M2t, gamma_t)
        hue = gpu_hue(M1t, M2t, gamma_t)
        info = gpu_info(M1t, M2t, gamma_t)
        cusps = gpu_cusp(M1t, M2t, [80,85,90], gamma_t)
        cond1 = torch.linalg.cond(M1t).item()
        cond2 = torch.linalg.cond(M2t).item()
    dm, dx = cpu_hue_drift(M1_np, M2_np, 50, gamma)
    c85 = cusps[85]
    return {
        'cv': cv,
        'hue_rms': math.sqrt(hue),
        'cusp_L_85': c85[0], 'cusp_C_85': c85[1], 'cliff_85': c85[2],
        'cusp_L_80': cusps[80][0], 'cusp_L_90': cusps[90][0],
        'drift_mean': dm, 'drift_max': dx,
        'bw': info['bw'], 'rw_gb': info['rw_gb'], 'plr': info['plr'],
        'cond1': cond1, 'cond2': cond2,
        'yL': info['yL'], 'yC': info['yC'],
    }

# ── Checkpoint saving ──
def save_checkpoint(M1_np, M2_np, gamma, metrics, seed_name, phase="p1"):
    """Save a checkpoint to checkpoints/ with timestamp. Returns filename."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    gamma_tag = "pcg" if PER_CHANNEL_GAMMA else "sg"
    fn = f"nextgen_{seed_name}_{gamma_tag}_{ts}.json"
    fp = os.path.join(CKPT_DIR, fn)

    M1i = np.linalg.inv(M1_np)
    M2i = np.linalg.inv(M2_np)

    gamma_serializable = gamma.tolist() if isinstance(gamma, np.ndarray) else gamma

    ckpt = {
        "version": f"nextgen-{seed_name}-{phase}",
        "mode": MODE_STR,
        "timestamp": datetime.now().isoformat(),
        "M1": M1_np.tolist(),
        "M2": M2_np.tolist(),
        "M1_inv": M1i.tolist(),
        "M2_inv": M2i.tolist(),
        "gamma": gamma_serializable,
        "metrics": metrics,
    }
    with open(fp, "w") as f:
        json.dump(ckpt, f, indent=2)
    return fn

# ── COMPREHENSIVE OBJECTIVE ──
call_count = [0]

def make_objective(gamma_mode):
    """Create objective function for the given gamma mode."""
    def objective(x):
        try:
            M1n, M2n, gamma = unpack(x)
            if M1n is None: return 999.
            M1t = torch.tensor(M1n, device=device)
            M2t = torch.tensor(M2n, device=device)
            gamma_t = gamma
            if isinstance(gamma, np.ndarray):
                gamma_t = torch.tensor(gamma, device=device)

            with torch.no_grad():
                cond1 = torch.linalg.cond(M1t).item()
                cond2 = torch.linalg.cond(M2t).item()

                # ── HARD CONSTRAINTS (immediate reject) ──
                if cond1 > 15 or cond2 > 25: return 999.

                info = gpu_info(M1t, M2t, gamma_t)
                if info['yC'] < 0.05: return 500.  # yellow must exist

                cv = gpu_cv(M1t, M2t, gamma_t)
                hue_rms = gpu_hue(M1t, M2t, gamma_t)

                # Cusp at yellow hues
                cusps = gpu_cusp(M1t, M2t, [80, 85, 90], gamma_t)

            # Hue drift (every 3rd eval to save time, ALL pairs)
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                drift_mean, drift_max = cpu_hue_drift(M1n, M2n, n_pairs=50, gamma=gamma)
            else:
                drift_mean, drift_max = 0, 0

            # ── SOFT PENALTIES (smooth, proportional) ──
            pen = 0.0

            # Yellow chroma > 0.12
            if info['yC'] < 0.12: pen += (0.12 - info['yC'])**2 * 200
            # Blue-white G/R >= 1.20
            if info['bw'] < 1.20: pen += (1.20 - info['bw'])**2 * 50
            # Red-white G-B <= 0.08
            if info['rw_gb'] > 0.08: pen += (info['rw_gb'] - 0.08)**2 * 100
            # Primary L range > 0.40
            if info['plr'] < 0.40: pen += (0.40 - info['plr'])**2 * 50
            # Condition M1 < 3.15
            if cond1 > 3.15: pen += (cond1 - 3.15)**2 * 5
            if cond2 > 10: pen += (cond2 - 10)**2 * 3

            # Hue drift
            if drift_max > 40: pen += (drift_max - 40)**2 * 0.1
            if drift_max > 60: pen += (drift_max - 60)**2 * 0.3

            # ── CUSP PENALTIES (the key addition) ──
            cusp_pen = 0
            for hd in [80, 85, 90]:
                cL, cC, cliff = cusps[hd]
                # Cusp L should be 0.80-0.92
                if cL > 0.92: cusp_pen += (cL - 0.92)**2 * 30
                if cL < 0.78: cusp_pen += (0.78 - cL)**2 * 30
                # Cliff < 50% (ambitious but not extreme)
                if cliff > 50: cusp_pen += (cliff - 50)**2 * 0.01

            # ── PRIMARY OBJECTIVE ──
            # CV is important but not king -- balance with cusp
            loss = 5.0*cv + 2.0*cusp_pen + 0.01*hue_rms + pen

            return loss
        except: return 999.
    return objective

objective = make_objective(PER_CHANNEL_GAMMA)

# ── Configuration ──
PHASE1_GENS = 500
PHASE1_POP = 96
PHASE2_GENS = 300
PHASE2_POP = 96
PHASE2_SIGMA = 0.005
PHASE2_TOP_N = 3

# ── RUN ──
run_start = datetime.now()
print(f"\n{'='*60}", flush=True)
print("  NEXT-GEN GenSpace Optimization", flush=True)
print(f"  {MODE_STR}", flush=True)
print(f"  {run_start.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(f"{'='*60}\n", flush=True)

# ── Baselines ──
print("--- Baselines ---", flush=True)
baselines = {}
for name, M1, M2 in [("v14", V14_M1, V14_M2), ("OKLab", OK_M1, OK_M2)]:
    metrics = compute_full_metrics(M1, M2, gamma=1./3.)
    x_test = pack(M1, M2, gamma=1./3.)
    loss = objective(x_test)
    metrics['loss'] = loss
    baselines[name] = metrics
    print(f"  {name}: loss={loss:.2f} CV={metrics['cv']*100:.2f}% hue={metrics['hue_rms']:.1f} "
          f"yL={metrics['yL']:.3f} cusp_L={metrics['cusp_L_85']:.3f} "
          f"cliff={metrics['cliff_85']:.0f}% drift={metrics['drift_mean']:.1f}/{metrics['drift_max']:.1f} "
          f"bw={metrics['bw']:.2f} rw={metrics['rw_gb']:+.3f} plr={metrics['plr']:.3f} "
          f"cond={metrics['cond1']:.1f}", flush=True)

# ── Seeds ──
x_v14 = pack(V14_M1, V14_M2)
x_ok = pack(OK_M1, OK_M2)

seeds = [
    ("v14", x_v14, 0.02),
    ("OKLab", x_ok, 0.02),
    ("mid", pack((V14_M1+OK_M1)/2, (V14_M2+OK_M2)/2), 0.03),
]
# Random seeds near v14-OKLab midpoint
rng2 = np.random.RandomState(2026)
for i in range(5):
    x = np.zeros(N_PARAMS)
    base_M1 = 0.5*V14_M1 + 0.5*OK_M1
    for r in range(3):
        x[2*r] = base_M1[r,0] + rng2.randn()*0.15
        x[2*r+1] = base_M1[r,1] + rng2.randn()*0.15
    x[6:9] = (V14_M2[0]+OK_M2[0])/2 + rng2.randn(3)*0.1
    x[9:13] = rng2.randn(4)*0.8
    if PER_CHANNEL_GAMMA:
        x[13] = rng2.randn()*0.1  # small perturbation around 0 (shared gamma)
        x[14] = rng2.randn()*0.1
    seeds.append((f"rnd{i}", x, 0.05))

# ══════════════════════════════════════════════════════════════
#  PHASE 1
# ══════════════════════════════════════════════════════════════
print(f"\n--- Phase 1: {len(seeds)} seeds x {PHASE1_GENS} gen x {PHASE1_POP} pop ---", flush=True)
phase1_results = []

for seed_idx, (name, x0, sigma) in enumerate(seeds):
    print(f"\n  [{seed_idx+1}/{len(seeds)}] Seed: {name} (sigma={sigma})", flush=True)
    best_loss = 999.; best_x = x0.copy()
    t0 = time.time(); ev = [0]; lp = [0]
    call_count[0] = 0
    best_ckpt_file = [None]

    def obj_fn(x, _name=name, _t0=t0, _ev=ev, _lp=lp, _best=[best_loss, best_x, best_ckpt_file]):
        loss = objective(x); _ev[0] += 1
        if loss < _best[0]:
            _best[0] = loss; _best[1] = x.copy()
            now = time.time()

            # Save checkpoint immediately on improvement
            M1n, M2n, gamma = unpack(x)
            if M1n is not None:
                metrics_quick = {
                    'loss': loss,
                    'cv': None, 'hue_rms': None,
                    'cusp_L_85': None, 'cliff_85': None,
                }
                M1t = torch.tensor(M1n, device=device); M2t = torch.tensor(M2n, device=device)
                gamma_t = gamma
                if isinstance(gamma, np.ndarray):
                    gamma_t = torch.tensor(gamma, device=device)
                with torch.no_grad():
                    cv = gpu_cv(M1t, M2t, gamma_t)
                    cusps = gpu_cusp(M1t, M2t, [85], gamma_t)
                    inf = gpu_info(M1t, M2t, gamma_t)
                c85 = cusps[85]

                # Save checkpoint with available metrics
                ckpt_metrics = {
                    'loss': loss, 'cv': cv, 'hue_rms': None,
                    'cusp_L_85': c85[0], 'cusp_C_85': c85[1], 'cliff_85': c85[2],
                    'bw': inf['bw'], 'rw_gb': inf['rw_gb'], 'plr': inf['plr'],
                    'yL': inf['yL'], 'yC': inf['yC'],
                }
                fn = save_checkpoint(M1n, M2n, gamma, ckpt_metrics, _name, "p1")
                _best[2][0] = fn

                # Print progress (throttled to every 30s)
                if now - _lp[0] > 30:
                    _lp[0] = now
                    print(f"    #{_ev[0]:>6d} [{now-_t0:5.0f}s] loss={loss:.4f} "
                          f"CV={cv*100:.1f}% cusp_L={c85[0]:.3f} cliff={c85[2]:.0f}% "
                          f"yC={inf['yC']:.3f} bw={inf['bw']:.2f} rw={inf['rw_gb']:+.3f} "
                          f"plr={inf['plr']:.3f} -> {fn}", flush=True)
        return loss

    opts = cma.CMAOptions()
    opts.set("maxiter", PHASE1_GENS); opts.set("popsize", PHASE1_POP)
    opts.set("tolfun", 1e-11); opts.set("verbose", -1)
    es = cma.CMAEvolutionStrategy(x0, sigma, opts)
    while not es.stop():
        sols = es.ask(); fits = [obj_fn(x) for x in sols]; es.tell(sols, fits)
    el = time.time() - t0

    # Extract best from closure (_best is 5th default param, index 4)
    best_loss = obj_fn.__defaults__[4][0]
    best_x = obj_fn.__defaults__[4][1]
    last_ckpt = obj_fn.__defaults__[4][2][0]

    M1f, M2f, gamma_f = unpack(best_x)
    if M1f is None:
        print(f"    {name}: FAILED (unpack returned None)", flush=True)
        continue

    # Compute full metrics for final result
    full_metrics = compute_full_metrics(M1f, M2f, gamma_f)
    full_metrics['loss'] = best_loss

    # Save final checkpoint with full metrics
    final_ckpt = save_checkpoint(M1f, M2f, gamma_f, full_metrics, name, "p1_final")

    print(f"    {name}: {ev[0]} evals {el:.0f}s | loss={best_loss:.4f} "
          f"CV={full_metrics['cv']*100:.2f}% cusp_L={full_metrics['cusp_L_85']:.3f} "
          f"cliff={full_metrics['cliff_85']:.0f}% "
          f"drift={full_metrics['drift_mean']:.1f}/{full_metrics['drift_max']:.1f} "
          f"bw={full_metrics['bw']:.2f} rw={full_metrics['rw_gb']:+.3f} "
          f"plr={full_metrics['plr']:.3f} cond={full_metrics['cond1']:.1f} "
          f"-> {final_ckpt}", flush=True)

    phase1_results.append({
        'name': name,
        'x': best_x.copy(),
        'loss': best_loss,
        'metrics': full_metrics,
        'M1': M1f.copy(),
        'M2': M2f.copy(),
        'gamma': gamma_f if not isinstance(gamma_f, np.ndarray) else gamma_f.copy(),
        'elapsed': el,
        'evals': ev[0],
        'checkpoint': final_ckpt,
    })

# Sort Phase 1 by loss
phase1_results.sort(key=lambda r: r['loss'])

print(f"\n{'='*60}", flush=True)
print(f"  PHASE 1 RANKING", flush=True)
print(f"{'='*60}", flush=True)
for i, r in enumerate(phase1_results):
    m = r['metrics']
    print(f"  {i+1}. {r['name']:>6}: loss={r['loss']:.4f} CV={m['cv']*100:.2f}% "
          f"cusp_L={m['cusp_L_85']:.3f} cliff={m['cliff_85']:.0f}% "
          f"drift={m['drift_mean']:.1f}/{m['drift_max']:.1f} "
          f"bw={m['bw']:.2f} rw={m['rw_gb']:+.3f} plr={m['plr']:.3f} "
          f"cond={m['cond1']:.1f}", flush=True)

# ══════════════════════════════════════════════════════════════
#  PHASE 2: Refine top N with smaller sigma
# ══════════════════════════════════════════════════════════════
top_n = min(PHASE2_TOP_N, len(phase1_results))
print(f"\n{'='*60}", flush=True)
print(f"  PHASE 2: Refining top {top_n} with sigma={PHASE2_SIGMA}, {PHASE2_GENS} gen", flush=True)
print(f"{'='*60}", flush=True)

phase2_results = []

for ri, r in enumerate(phase1_results[:top_n]):
    name = r['name']
    x0 = r['x']
    print(f"\n  [{ri+1}/{top_n}] Refining: {name} (Phase 1 loss={r['loss']:.4f})", flush=True)

    best_loss = r['loss']; best_x = x0.copy()
    t0 = time.time(); ev = [0]; lp = [0]
    call_count[0] = 0
    best_ckpt_file = [None]

    def obj_fn_p2(x, _name=name, _t0=t0, _ev=ev, _lp=lp, _best=[best_loss, best_x, best_ckpt_file]):
        loss = objective(x); _ev[0] += 1
        if loss < _best[0]:
            _best[0] = loss; _best[1] = x.copy()
            now = time.time()

            M1n, M2n, gamma = unpack(x)
            if M1n is not None:
                M1t = torch.tensor(M1n, device=device); M2t = torch.tensor(M2n, device=device)
                gamma_t = gamma
                if isinstance(gamma, np.ndarray):
                    gamma_t = torch.tensor(gamma, device=device)
                with torch.no_grad():
                    cv = gpu_cv(M1t, M2t, gamma_t)
                    cusps = gpu_cusp(M1t, M2t, [85], gamma_t)
                    inf = gpu_info(M1t, M2t, gamma_t)
                c85 = cusps[85]

                ckpt_metrics = {
                    'loss': loss, 'cv': cv, 'hue_rms': None,
                    'cusp_L_85': c85[0], 'cusp_C_85': c85[1], 'cliff_85': c85[2],
                    'bw': inf['bw'], 'rw_gb': inf['rw_gb'], 'plr': inf['plr'],
                    'yL': inf['yL'], 'yC': inf['yC'],
                }
                fn = save_checkpoint(M1n, M2n, gamma, ckpt_metrics, _name, "p2")
                _best[2][0] = fn

                if now - _lp[0] > 30:
                    _lp[0] = now
                    print(f"    #{_ev[0]:>6d} [{now-_t0:5.0f}s] loss={loss:.4f} "
                          f"CV={cv*100:.1f}% cusp_L={c85[0]:.3f} cliff={c85[2]:.0f}% "
                          f"yC={inf['yC']:.3f} bw={inf['bw']:.2f} rw={inf['rw_gb']:+.3f} "
                          f"plr={inf['plr']:.3f} -> {fn}", flush=True)
        return loss

    opts = cma.CMAOptions()
    opts.set("maxiter", PHASE2_GENS); opts.set("popsize", PHASE2_POP)
    opts.set("tolfun", 1e-12); opts.set("verbose", -1)
    es = cma.CMAEvolutionStrategy(x0, PHASE2_SIGMA, opts)
    while not es.stop():
        sols = es.ask(); fits = [obj_fn_p2(x) for x in sols]; es.tell(sols, fits)
    el = time.time() - t0

    best_loss = obj_fn_p2.__defaults__[4][0]
    best_x = obj_fn_p2.__defaults__[4][1]

    M1f, M2f, gamma_f = unpack(best_x)
    if M1f is None:
        print(f"    {name}: FAILED in Phase 2", flush=True)
        continue

    full_metrics = compute_full_metrics(M1f, M2f, gamma_f)
    full_metrics['loss'] = best_loss
    final_ckpt = save_checkpoint(M1f, M2f, gamma_f, full_metrics, name, "p2_final")

    improvement = r['loss'] - best_loss
    print(f"    {name}: {ev[0]} evals {el:.0f}s | loss={best_loss:.4f} "
          f"(improved by {improvement:.4f}) "
          f"CV={full_metrics['cv']*100:.2f}% cusp_L={full_metrics['cusp_L_85']:.3f} "
          f"cliff={full_metrics['cliff_85']:.0f}% "
          f"drift={full_metrics['drift_mean']:.1f}/{full_metrics['drift_max']:.1f} "
          f"bw={full_metrics['bw']:.2f} rw={full_metrics['rw_gb']:+.3f} "
          f"plr={full_metrics['plr']:.3f} cond={full_metrics['cond1']:.1f} "
          f"-> {final_ckpt}", flush=True)

    phase2_results.append({
        'name': name,
        'x': best_x.copy(),
        'loss': best_loss,
        'metrics': full_metrics,
        'M1': M1f.copy(),
        'M2': M2f.copy(),
        'gamma': gamma_f if not isinstance(gamma_f, np.ndarray) else gamma_f.copy(),
        'elapsed': el,
        'evals': ev[0],
        'checkpoint': final_ckpt,
        'p1_loss': r['loss'],
    })

phase2_results.sort(key=lambda r: r['loss'])

# ══════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════

# Determine overall best (from Phase 2 if available, else Phase 1)
all_results = phase2_results if phase2_results else phase1_results
all_results.sort(key=lambda r: r['loss'])
best = all_results[0] if all_results else None

run_end = datetime.now()
run_duration = (run_end - run_start).total_seconds()

print(f"\n{'='*60}", flush=True)
print(f"  FINAL RANKING (Phase 2)", flush=True)
print(f"{'='*60}", flush=True)
for i, r in enumerate(phase2_results):
    m = r['metrics']
    print(f"  {i+1}. {r['name']:>6}: loss={r['loss']:.4f} (was {r['p1_loss']:.4f}) "
          f"CV={m['cv']*100:.2f}% cusp_L={m['cusp_L_85']:.3f} "
          f"cliff={m['cliff_85']:.0f}% drift={m['drift_mean']:.1f}/{m['drift_max']:.1f} "
          f"bw={m['bw']:.2f} rw={m['rw_gb']:+.3f} plr={m['plr']:.3f} "
          f"cond={m['cond1']:.1f}", flush=True)

if best:
    print(f"\n{'='*60}", flush=True)
    print(f"  BEST vs v14 BASELINE", flush=True)
    print(f"{'='*60}", flush=True)
    bm = best['metrics']
    v14m = baselines['v14']

    def fmt_change(val, ref, fmt=".2f", lower_better=True):
        diff = val - ref
        pct = (diff / abs(ref) * 100) if abs(ref) > 1e-10 else 0
        arrow = "v" if (diff < 0 and lower_better) or (diff > 0 and not lower_better) else "^"
        if (diff < 0 and lower_better) or (diff > 0 and not lower_better):
            arrow = "BETTER"
        else:
            arrow = "WORSE"
        return f"{val:{fmt}} (v14: {ref:{fmt}}, {diff:+{fmt}} {arrow})"

    print(f"  Best seed: {best['name']}")
    print(f"  Loss:      {fmt_change(bm['loss'], v14m['loss'], '.4f')}")
    print(f"  CV:        {fmt_change(bm['cv']*100, v14m['cv']*100, '.2f')}%")
    print(f"  Cusp L@85: {fmt_change(bm['cusp_L_85'], v14m['cusp_L_85'], '.3f')}")
    print(f"  Cliff@85:  {fmt_change(bm['cliff_85'], v14m['cliff_85'], '.0f')}%")
    print(f"  Drift:     {bm['drift_mean']:.1f}/{bm['drift_max']:.1f} (v14: {v14m['drift_mean']:.1f}/{v14m['drift_max']:.1f})")
    print(f"  B->W:      {fmt_change(bm['bw'], v14m['bw'], '.3f', lower_better=False)}")
    print(f"  R->W:      {bm['rw_gb']:+.4f} (v14: {v14m['rw_gb']:+.4f})")
    print(f"  PLR:       {fmt_change(bm['plr'], v14m['plr'], '.3f', lower_better=False)}")
    print(f"  Cond M1:   {fmt_change(bm['cond1'], v14m['cond1'], '.1f')}")
    print(f"  Hue RMS:   {fmt_change(bm['hue_rms'], v14m['hue_rms'], '.1f')}")
    print(f"  Checkpoint: {best['checkpoint']}", flush=True)

# ══════════════════════════════════════════════════════════════
#  GENERATE MARKDOWN REPORT
# ══════════════════════════════════════════════════════════════
report_path = os.path.join(CKPT_DIR, "nextgen_report.md")

def fmt_metric(val, fmt=".2f"):
    if val is None: return "N/A"
    return f"{val:{fmt}}"

report_lines = []
report_lines.append("# Next-Gen GenSpace Optimization Report\n")
report_lines.append(f"**Date:** {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"**Device:** {device_name}")
report_lines.append(f"**Mode:** {MODE_STR}")
report_lines.append(f"**Seeds:** {len(seeds)}")
report_lines.append(f"**Phase 1:** {PHASE1_GENS} gen x {PHASE1_POP} pop")
report_lines.append(f"**Phase 2:** {PHASE2_GENS} gen x {PHASE2_POP} pop (top {PHASE2_TOP_N})")
report_lines.append(f"**Total time:** {run_duration:.0f}s ({run_duration/60:.1f} min)")
report_lines.append("")

# Baselines table
report_lines.append("## Baselines\n")
report_lines.append("| Space | Loss | CV% | Cusp L@85 | Cliff% | Drift | B->W | R->W | PLR | Cond M1 | Hue RMS |")
report_lines.append("|-------|------|-----|-----------|--------|-------|------|------|-----|---------|---------|")
for bname, bm in baselines.items():
    report_lines.append(
        f"| {bname} | {fmt_metric(bm['loss'],'.2f')} | {fmt_metric(bm['cv']*100,'.2f')} | "
        f"{fmt_metric(bm['cusp_L_85'],'.3f')} | {fmt_metric(bm['cliff_85'],'.0f')} | "
        f"{fmt_metric(bm['drift_mean'],'.1f')}/{fmt_metric(bm['drift_max'],'.1f')} | "
        f"{fmt_metric(bm['bw'],'.2f')} | {bm['rw_gb']:+.3f} | "
        f"{fmt_metric(bm['plr'],'.3f')} | {fmt_metric(bm['cond1'],'.1f')} | "
        f"{fmt_metric(bm['hue_rms'],'.1f')} |"
    )
report_lines.append("")

# Phase 1 results table
report_lines.append("## Phase 1 Results\n")
report_lines.append("| Rank | Seed | Loss | CV% | Cusp L@85 | Cliff% | Drift (mean/max) | B->W | R->W | PLR | Cond M1 | Hue RMS | Time | Evals | Checkpoint |")
report_lines.append("|------|------|------|-----|-----------|--------|------------------|------|------|-----|---------|---------|------|-------|------------|")
for i, r in enumerate(phase1_results):
    m = r['metrics']
    report_lines.append(
        f"| {i+1} | {r['name']} | {fmt_metric(m['loss'],'.4f')} | {fmt_metric(m['cv']*100,'.2f')} | "
        f"{fmt_metric(m['cusp_L_85'],'.3f')} | {fmt_metric(m['cliff_85'],'.0f')} | "
        f"{fmt_metric(m['drift_mean'],'.1f')}/{fmt_metric(m['drift_max'],'.1f')} | "
        f"{fmt_metric(m['bw'],'.2f')} | {m['rw_gb']:+.3f} | "
        f"{fmt_metric(m['plr'],'.3f')} | {fmt_metric(m['cond1'],'.1f')} | "
        f"{fmt_metric(m['hue_rms'],'.1f')} | {r['elapsed']:.0f}s | {r['evals']} | "
        f"`{r['checkpoint']}` |"
    )
report_lines.append("")

# Phase 2 results table
report_lines.append("## Phase 2 Refinement\n")
if phase2_results:
    report_lines.append(f"Top {top_n} seeds refined with sigma={PHASE2_SIGMA}, {PHASE2_GENS} generations.\n")
    report_lines.append("| Rank | Seed | P1 Loss | P2 Loss | Improvement | CV% | Cusp L@85 | Cliff% | Drift | B->W | R->W | PLR | Cond M1 | Hue RMS | Checkpoint |")
    report_lines.append("|------|------|---------|---------|-------------|-----|-----------|--------|-------|------|------|-----|---------|---------|------------|")
    for i, r in enumerate(phase2_results):
        m = r['metrics']
        imp = r['p1_loss'] - r['loss']
        report_lines.append(
            f"| {i+1} | {r['name']} | {fmt_metric(r['p1_loss'],'.4f')} | {fmt_metric(m['loss'],'.4f')} | "
            f"{imp:+.4f} | {fmt_metric(m['cv']*100,'.2f')} | "
            f"{fmt_metric(m['cusp_L_85'],'.3f')} | {fmt_metric(m['cliff_85'],'.0f')} | "
            f"{fmt_metric(m['drift_mean'],'.1f')}/{fmt_metric(m['drift_max'],'.1f')} | "
            f"{fmt_metric(m['bw'],'.2f')} | {m['rw_gb']:+.3f} | "
            f"{fmt_metric(m['plr'],'.3f')} | {fmt_metric(m['cond1'],'.1f')} | "
            f"{fmt_metric(m['hue_rms'],'.1f')} | `{r['checkpoint']}` |"
        )
else:
    report_lines.append("No Phase 2 results (all seeds failed in Phase 1).")
report_lines.append("")

# Best vs v14 comparison
if best:
    report_lines.append("## Best Result vs v14\n")
    bm = best['metrics']
    v14m = baselines['v14']

    report_lines.append("| Metric | Best | v14 | Change |")
    report_lines.append("|--------|------|-----|--------|")

    comparisons = [
        ("Loss", bm['loss'], v14m['loss'], ".4f", True),
        ("CV%", bm['cv']*100, v14m['cv']*100, ".2f", True),
        ("Cusp L@85", bm['cusp_L_85'], v14m['cusp_L_85'], ".3f", None),
        ("Cliff@85%", bm['cliff_85'], v14m['cliff_85'], ".0f", True),
        ("Drift mean", bm['drift_mean'], v14m['drift_mean'], ".1f", True),
        ("Drift max", bm['drift_max'], v14m['drift_max'], ".1f", True),
        ("B->W G/R", bm['bw'], v14m['bw'], ".3f", False),
        ("R->W G-B", bm['rw_gb'], v14m['rw_gb'], "+.4f", None),
        ("PLR", bm['plr'], v14m['plr'], ".3f", False),
        ("Cond M1", bm['cond1'], v14m['cond1'], ".1f", True),
        ("Cond M2", bm['cond2'], v14m['cond2'], ".1f", True),
        ("Hue RMS", bm['hue_rms'], v14m['hue_rms'], ".1f", True),
        ("Yellow L", bm['yL'], v14m['yL'], ".3f", None),
        ("Yellow C", bm['yC'], v14m['yC'], ".3f", False),
    ]

    for metric_name, best_val, ref_val, fmt, lower_better in comparisons:
        diff = best_val - ref_val
        if lower_better is None:
            change_str = f"{diff:{fmt}}"
        elif (diff < 0 and lower_better) or (diff > 0 and not lower_better):
            change_str = f"{diff:{fmt}} (better)"
        elif abs(diff) < 1e-10:
            change_str = "same"
        else:
            change_str = f"{diff:{fmt}} (worse)"
        report_lines.append(f"| {metric_name} | {best_val:{fmt}} | {ref_val:{fmt}} | {change_str} |")

    report_lines.append("")

    # Gamma info for per-channel mode
    if PER_CHANNEL_GAMMA and isinstance(best['gamma'], np.ndarray):
        report_lines.append("### Per-Channel Gamma\n")
        g = best['gamma']
        report_lines.append(f"- Channel 1 (L): {g[0]:.6f}")
        report_lines.append(f"- Channel 2 (M): {g[1]:.6f}")
        report_lines.append(f"- Channel 3 (S): {g[2]:.6f}")
        report_lines.append(f"- Ratios: g2/g1={g[1]/g[0]:.4f}, g3/g1={g[2]/g[0]:.4f}")
        report_lines.append("")

    report_lines.append("## Checkpoint\n")
    report_lines.append(f"Best: `checkpoints/{best['checkpoint']}`")
    report_lines.append("")

report_text = "\n".join(report_lines)

with open(report_path, "w") as f:
    f.write(report_text)

print(f"\n{'='*60}", flush=True)
print(f"  Report saved to: {report_path}", flush=True)
if best:
    print(f"  Best checkpoint: checkpoints/{best['checkpoint']}", flush=True)
print(f"  Total time: {run_duration:.0f}s ({run_duration/60:.1f} min)", flush=True)
print(f"{'='*60}", flush=True)
