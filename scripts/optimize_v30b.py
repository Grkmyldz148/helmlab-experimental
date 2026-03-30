"""v30b: Fixed weights. Key lesson from v30:
- CV must stay < OKLab (22.9%). It's our main advantage.
- Cusp L must be 0.80-0.93 (not too high, not too low)
- Hue drift must be checked EVERY eval with strong weight
- No Phase 2, just 4 seeds × 500 gen, fast iteration
"""
import json, time, math, numpy as np, torch, subprocess, sys
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
import cma

D65 = np.array([0.95047, 1.0, 1.08883])
M_S = torch.tensor([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]], device=device)
M_Si = torch.linalg.inv(M_S)
D65_T = torch.tensor(D65, device=device)
M_S_np = np.array([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]])
M_Si_np = np.linalg.inv(M_S_np)
D65_np = np.array([0.95047, 1.0, 1.08883])

V14_M1 = np.array([[0.7583761294836658,0.38380162590825084,-0.09608055040602373],[0.12671393631532843,0.8421628149123207,0.03434823621506485],[0.07639223722200054,0.258943526275451,0.6139139663787314]])
V14_M2 = np.array([[0.10058070589596230,1.01558970993941444,-0.11617041583537688],[2.36157646996164416,-2.44099737506293479,0.07942090510129070],[0.04565327074453784,0.81875488445424471,-0.86440815519878267]])
OK_M1s = np.array([[0.4122214708,0.5363325363,0.0514459929],[0.2119034982,0.6806995451,0.1073969566],[0.0883024619,0.2817188376,0.6299787005]])
OK_M1 = OK_M1s @ np.linalg.inv(M_S_np)
OK_M2 = np.array([[0.2104542553,0.7936177850,-0.0040720468],[1.9779984951,-2.4285922050,0.4505937099],[0.0259040371,0.7827717662,-0.8086757660]])

def scbrt(x): return torch.sign(x)*torch.abs(x).pow(1./3.)
def s2l(c): return torch.where(c<=0.04045,c/12.92,((c+0.055)/1.055).pow(2.4))
def l2s(c): return torch.where(c<=0.0031308,c*12.92,1.055*c.clamp(min=1e-10).pow(1./2.4)-0.055)
def s2l_np(c): return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def l2s_np(c): return np.where(c<=0.0031308,c*12.92,1.055*np.maximum(c,1e-10)**(1/2.4)-0.055)

# Training pairs
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
pair_xyz = []
for i,(c1,c2) in enumerate(pairs_list):
    x1_t = M_S@s2l(torch.tensor(c1,device=device))
    x2_t = M_S@s2l(torch.tensor(c2,device=device))
    pt[i,0]=x1_t; pt[i,1]=x2_t
    pair_xyz.append((M_S_np@s2l_np(np.array(c1)), M_S_np@s2l_np(np.array(c2))))

N_ST=25; T_ST=torch.linspace(0,1,N_ST+1,device=device)

def gpu_cv(M1,M2):
    M1i,M2i=torch.linalg.inv(M1),torch.linalg.inv(M2)
    N=pt.shape[0]
    l1=scbrt(pt[:,0]@M1.T)@M2.T; l2=scbrt(pt[:,1]@M1.T)@M2.T
    t=T_ST.view(1,-1,1); labs=l1.unsqueeze(1)+t*(l2-l1).unsqueeze(1)
    lf=labs.reshape(-1,3); lc=lf@M2i.T; lm=torch.sign(lc)*torch.abs(lc).pow(3.)
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

def gpu_hue(M1,M2):
    prs=torch.tensor([[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1]],dtype=torch.float64,device=device)
    exp=torch.tensor([0,60,120,180,240,300],dtype=torch.float64,device=device)
    lab=scbrt(s2l(prs)@M_S.T@M1.T)@M2.T
    h=torch.atan2(lab[:,2],lab[:,1])*(180/3.14159265)%360
    dh=h-exp; dh=torch.where(dh>180,dh-360,dh); dh=torch.where(dh<-180,dh+360,dh)
    return (dh**2).mean().item()

def gpu_info(M1,M2):
    M1i,M2i=torch.linalg.inv(M1),torch.linalg.inv(M2)
    yl=scbrt((M_S@torch.tensor([1.,1.,0.],device=device))@M1.T)@M2.T
    yL,yC=yl[0].item(),(yl[1]**2+yl[2]**2).sqrt().item()
    bx=M_S@s2l(torch.tensor([0.,0.,1.],device=device))
    wx=M_S@s2l(torch.tensor([1.,1.,1.],device=device))
    bl=scbrt(bx@M1.T)@M2.T; wl=scbrt(wx@M1.T)@M2.T
    ml=(bl+wl)/2; lc=ml@M2i.T; lm=torch.sign(lc)*torch.abs(lc).pow(3.); mx=lm@M1i.T
    ms=l2s((M_Si@mx).clamp(0,1))
    bw=ms[1].item()/max(ms[0].item(),0.01)
    ps=torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]],dtype=torch.float64,device=device)
    pl=scbrt(s2l(ps)@M_S.T@M1.T)@M2.T
    plr=(pl[:,0].max()-pl[:,0].min()).item()
    return {'yL':yL,'yC':yC,'bw':bw,'plr':plr}

# Cusp scan (yellow region)
CUSP_Ls = torch.linspace(0.3, 0.998, 80, device=device)
CUSP_Cs = torch.linspace(0.001, 0.4, 60, device=device)
CUSP_Le = CUSP_Ls.view(80,1).expand(80,60)
CUSP_Ce = CUSP_Cs.view(1,60).expand(80,60)
CUSP_Ce_v = CUSP_Cs.view(1,60).expand(80,60)

def gpu_cusp_yellow(M1, M2):
    M1i = torch.linalg.inv(M1); M2i = torch.linalg.inv(M2)
    penalty = 0.0
    for hd in [75, 80, 85, 90, 95]:
        hr = hd * 3.14159265 / 180
        ch, sh = np.cos(hr), np.sin(hr)
        lab = torch.stack([CUSP_Le, CUSP_Ce*ch, CUSP_Ce*sh], dim=-1).reshape(-1, 3)
        lc = lab @ M2i.T; lm = torch.sign(lc) * torch.abs(lc).pow(3.)
        lin = (lm @ M1i.T) @ M_Si.T
        ok = ((lin >= -0.002).all(dim=1) & (lin <= 1.002).all(dim=1)).reshape(80, 60)
        mc, _ = torch.where(ok, CUSP_Ce_v, torch.zeros(80,60,device=device)).max(dim=1)
        ci = mc.argmax().item()
        cL = CUSP_Ls[ci].item()
        cC = mc[ci].item()
        # Penalize cusp L > 0.93 (too high = cliff)
        if cL > 0.93: penalty += (cL - 0.93) ** 2 * 10
        # Penalize cusp L < 0.80 (too low = wrong shape)
        if cL < 0.80: penalty += (0.80 - cL) ** 2 * 10
        # Cliff steepness
        if ci < 78 and cC > 0.01:
            post_C = mc[ci+2].item()
            drop = (cC - post_C) / cC
            if drop > 0.50: penalty += (drop - 0.50) ** 2 * 5
    return penalty / 5

# Hue drift (fast: 10 primary pairs only, EVERY eval)
DRIFT_PAIRS = []
for i in range(len(prims)):
    for j in range(i+1, len(prims)):
        DRIFT_PAIRS.append((M_S_np@s2l_np(np.array(prims[i],dtype=float)), M_S_np@s2l_np(np.array(prims[j],dtype=float))))

def cpu_hue_drift_fast(M1_np, M2_np):
    M1i_np, M2i_np = np.linalg.inv(M1_np), np.linalg.inv(M2_np)
    max_drift = 0.0
    for x1, x2 in DRIFT_PAIRS:
        l1 = M2_np @ (np.sign(M1_np@x1)*np.abs(M1_np@x1)**(1/3))
        l2 = M2_np @ (np.sign(M1_np@x2)*np.abs(M1_np@x2)**(1/3))
        prev_h = None
        for t in np.linspace(0, 1, 13):  # fewer steps for speed
            lab = l1 + t * (l2 - l1)
            lc = M2i_np @ lab; xyz = M1i_np @ (np.sign(lc)*np.abs(lc)**3)
            rgb8 = np.round(l2s_np(np.clip(M_Si_np @ xyz, 0, 1)) * 255) / 255
            xyz_q = M_S_np @ s2l_np(rgb8)
            r = np.maximum(xyz_q, 1e-10) / D65_np
            f = np.where(r > 0.008856, r**(1/3), 7.787*r + 16/116)
            cl = np.array([116*f[1]-16, 500*(f[0]-f[1]), 200*(f[1]-f[2])])
            C_val = math.sqrt(cl[1]**2 + cl[2]**2)
            if C_val < 3.0: prev_h = None; continue
            h = math.atan2(cl[2], cl[1])
            if prev_h is not None:
                dh = abs(math.atan2(math.sin(h-prev_h), math.cos(h-prev_h))) * 180 / math.pi
                if dh > max_drift: max_drift = dh
            prev_h = h
    return max_drift

# Parameterization
def ortho(s):
    sn=s/np.linalg.norm(s)
    v=np.array([1,0,0.]) if abs(sn[0])<0.9 else np.array([0,1,0.])
    e1=v-np.dot(v,sn)*sn; e1/=np.linalg.norm(e1); e2=np.cross(sn,e1)
    return e1,e2
def unpack(x):
    M1=np.zeros((3,3))
    for i in range(3):
        M1[i,0]=x[2*i]; M1[i,1]=x[2*i+1]
        M1[i,2]=(1-M1[i,0]*D65_np[0]-M1[i,1]*D65_np[1])/D65_np[2]
    lms=M1@D65_np
    if np.any(lms<=0): return None,None
    s=lms**(1/3)
    if np.linalg.norm(s)<1e-10: return None,None
    e1,e2=ortho(s)
    M2=np.zeros((3,3)); M2[0]=x[6:9]
    Lw=M2[0]@s
    if abs(Lw)<1e-10: return None,None
    M2[0]/=Lw
    M2[1]=x[9]*e1+x[10]*e2; M2[2]=x[11]*e1+x[12]*e2
    return M1,M2
def pack(M1,M2):
    x=np.zeros(13)
    for i in range(3): x[2*i]=M1[i,0]; x[2*i+1]=M1[i,1]
    x[6:9]=M2[0]
    lms=M1@D65_np; s=lms**(1/3)
    e1,e2=ortho(s)
    x[9]=M2[1]@e1; x[10]=M2[1]@e2; x[11]=M2[2]@e1; x[12]=M2[2]@e2
    return x

# ══ OBJECTIVE ══
def objective(x):
    try:
        M1n, M2n = unpack(x)
        if M1n is None: return 999.
        M1t = torch.tensor(M1n, device=device)
        M2t = torch.tensor(M2n, device=device)
        with torch.no_grad():
            cond1 = torch.linalg.cond(M1t).item()
            cond2 = torch.linalg.cond(M2t).item()
            if cond1 > 10 or cond2 > 20: return 999.
            info = gpu_info(M1t, M2t)
            cv = gpu_cv(M1t, M2t)
            hue_err = gpu_hue(M1t, M2t)
            cusp = gpu_cusp_yellow(M1t, M2t)

        # Hue drift (EVERY eval, fast version)
        drift_max = cpu_hue_drift_fast(M1n, M2n)

        # ── HARD REJECT ──
        if info['yC'] < 0.08: return 100
        if info['bw'] < 1.0: return 100
        if cv > 0.30: return 50 + (cv - 0.30)**2 * 100  # CV must not explode

        # ── SOFT PENALTIES ──
        pen = 0.0
        if info['yC'] < 0.12: pen += (0.12 - info['yC'])**2 * 200
        if info['bw'] < 1.20: pen += (1.20 - info['bw'])**2 * 50
        if info['plr'] < 0.40: pen += (0.40 - info['plr'])**2 * 50
        if cond1 > 3.15: pen += (cond1 - 3.15)**2 * 10
        if drift_max > 45: pen += (drift_max - 45)**2 * 0.1
        if drift_max > 90: pen += (drift_max - 90)**2 * 0.5  # extra strong above 90

        # ── MAIN LOSS (CV-dominant!) ──
        # CV is king. cusp and drift are constraints, not primary objectives.
        loss = 5.0*cv + 1.0*cusp + 0.01*hue_err + pen
        return loss
    except: return 999.

# ══ RUN ══
print(f"\n{'='*60}", flush=True)
print("  v30b: CV-dominant + cusp + drift constraints", flush=True)
print(f"{'='*60}\n", flush=True)

# Baselines
x_v14 = pack(V14_M1, V14_M2); x_ok = pack(OK_M1, OK_M2)
l_v14 = objective(x_v14); l_ok = objective(x_ok)
print(f"  v14 loss={l_v14:.4f}, OKLab loss={l_ok:.4f}", flush=True)

seeds = [
    ("v14", x_v14, 0.01),
    ("OKLab", x_ok, 0.01),
    ("mid", pack((V14_M1+OK_M1)/2, (V14_M2+OK_M2)/2), 0.02),
    ("v14w", x_v14, 0.03),
]

results = []
for name, x0, sigma in seeds:
    best_loss = 999.; best_x = x0.copy()
    t0 = time.time(); ev = [0]; lp = [0]

    def obj_fn(x):
        global best_loss, best_x
        loss = objective(x); ev[0] += 1
        if loss < best_loss:
            best_loss = loss; best_x = x.copy()
            now = time.time()
            if now - lp[0] > 20:
                lp[0] = now
                M1n, M2n = unpack(x)
                if M1n is not None:
                    M1t = torch.tensor(M1n, device=device); M2t = torch.tensor(M2n, device=device)
                    with torch.no_grad():
                        cv = gpu_cv(M1t, M2t); cusp = gpu_cusp_yellow(M1t, M2t); inf = gpu_info(M1t, M2t)
                    drift = cpu_hue_drift_fast(M1n, M2n)
                    print(f"  #{ev[0]:>5d} [{now-t0:4.0f}s] loss={loss:.4f} CV={cv*100:.1f}% cusp={cusp:.3f} yL={inf['yL']:.3f} drift={drift:.0f} bw={inf['bw']:.2f} plr={inf['plr']:.3f}", flush=True)
        return loss

    opts = cma.CMAOptions()
    opts.set("maxiter", 500); opts.set("popsize", 96); opts.set("tolfun", 1e-11); opts.set("verbose", -1)
    es = cma.CMAEvolutionStrategy(x0, sigma, opts)
    while not es.stop():
        sols = es.ask(); fits = [obj_fn(x) for x in sols]; es.tell(sols, fits)
    el = time.time() - t0

    M1f, M2f = unpack(best_x)
    M1t = torch.tensor(M1f, device=device); M2t = torch.tensor(M2f, device=device)
    with torch.no_grad():
        cv = gpu_cv(M1t, M2t); cusp = gpu_cusp_yellow(M1t, M2t); inf = gpu_info(M1t, M2t)
    drift = cpu_hue_drift_fast(M1f, M2f)
    c1n = np.linalg.cond(M1f)
    print(f"  {name}: {ev[0]} evals {el:.0f}s | CV={cv*100:.2f}% cusp={cusp:.3f} yL={inf['yL']:.3f} drift={drift:.0f} bw={inf['bw']:.2f} plr={inf['plr']:.3f} cond={c1n:.1f}", flush=True)
    results.append((name, best_x.copy(), best_loss, cv, cusp, inf, drift, c1n))

    # Save
    M1i, M2i = np.linalg.inv(M1f), np.linalg.inv(M2f)
    fn = f"/root/gen_v30b_{name}.json"
    ckpt = {"version":f"v30b-{name}","M1":M1f.tolist(),"M2":M2f.tolist(),"M1_inv":M1i.tolist(),"M2_inv":M2i.tolist()}
    with open(fn,"w") as f: json.dump(ckpt,f,indent=2)

# Summary
results.sort(key=lambda r: r[2])
print(f"\n{'='*60}")
print(f"  RANKING")
print(f"{'='*60}")
for i,(name,_,loss,cv,cusp,inf,drift,c1n) in enumerate(results):
    print(f"  {i+1}. {name:>6}: loss={loss:.4f} CV={cv*100:.2f}% cusp={cusp:.3f} yL={inf['yL']:.3f} drift={drift:.0f} bw={inf['bw']:.2f} plr={inf['plr']:.3f} cond={c1n:.1f}")

# Production test on best
best_name = results[0][0]
fn = f"/root/gen_v30b_{best_name}.json"
print(f"\n{'='*60}")
print(f"  PRODUCTION TEST ({best_name})")
print(f"{'='*60}\n")
subprocess.run([sys.executable, "/root/production_test_gpu.py", "--json", fn])
