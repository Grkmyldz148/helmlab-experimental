"""v29: Direct cusp shape optimization.
NOT LMS balance (indirect proxy) — directly penalize:
  1. Yellow cusp L > 0.93 (too high = sharp cliff)
  2. Cliff steepness (chroma drop rate post-cusp)
  3. Cusp L jumps between adjacent hues
Keep CV + mono + hue in objective. Start from v14.
"""
import json, time, numpy as np, torch, subprocess, sys
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
import cma

D65 = np.array([0.95047, 1.0, 1.08883])
M_S = torch.tensor([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]], device=device)
M_Si = torch.linalg.inv(M_S)
D65_T = torch.tensor(D65, device=device)

V14_M1 = np.array([[0.7583761294836658,0.38380162590825084,-0.09608055040602373],[0.12671393631532843,0.8421628149123207,0.03434823621506485],[0.07639223722200054,0.258943526275451,0.6139139663787314]])
V14_M2 = np.array([[0.10058070589596230,1.01558970993941444,-0.11617041583537688],[2.36157646996164416,-2.44099737506293479,0.07942090510129070],[0.04565327074453784,0.81875488445424471,-0.86440815519878267]])

def scbrt(x): return torch.sign(x)*torch.abs(x).pow(1./3.)
def s2l(c): return torch.where(c<=0.04045,c/12.92,((c+0.055)/1.055).pow(2.4))
def l2s(c): return torch.where(c<=0.0031308,c*12.92,1.055*c.clamp(min=1e-10).pow(1./2.4)-0.055)

# ── Training pairs ──
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
for i,(c1,c2) in enumerate(pairs_list):
    pt[i,0]=M_S@s2l(torch.tensor(c1,device=device))
    pt[i,1]=M_S@s2l(torch.tensor(c2,device=device))

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

# ══════════════════════════════════════════════════════════
#  DIRECT CUSP SHAPE PENALTY (the right approach)
# ══════════════════════════════════════════════════════════
# Scan cusp at 24 hues (every 15°), focus on yellow region
CUSP_HUES = list(range(0, 360, 15))  # 24 hues
YELLOW_HUES = [75, 80, 85, 90]  # yellow region specifically

# Pre-compute scan grids
CUSP_Ls = torch.linspace(0.3, 0.998, 80, device=device)
CUSP_Cs = torch.linspace(0.001, 0.4, 60, device=device)
CUSP_Le = CUSP_Ls.view(80,1).expand(80,60)
CUSP_Ce = CUSP_Cs.view(1,60).expand(80,60)
CUSP_Ce_v = CUSP_Cs.view(1,60).expand(80,60)

def gpu_cusp_penalty(M1, M2):
    """Direct cusp shape penalty. Cheap: 24 hues × 80L × 60C grid.
    Penalizes:
      1. Yellow cusp L > 0.93 (too high → sharp cliff)
      2. Cliff steepness at yellow (chroma drop rate post-cusp)
      3. Cusp L variance across all hues (smoothness)
    """
    M1i = torch.linalg.inv(M1)
    M2i = torch.linalg.inv(M2)

    cusp_Ls = []
    yellow_penalty = 0.0

    for hd in CUSP_HUES:
        hr = hd * 3.14159265 / 180
        ch, sh = np.cos(hr), np.sin(hr)
        lab = torch.stack([CUSP_Le, CUSP_Ce*ch, CUSP_Ce*sh], dim=-1).reshape(-1, 3)
        lc = lab @ M2i.T
        lm = torch.sign(lc) * torch.abs(lc).pow(3.)
        lin = (lm @ M1i.T) @ M_Si.T
        ok = ((lin >= -0.002).all(dim=1) & (lin <= 1.002).all(dim=1)).reshape(80, 60)
        mc, _ = torch.where(ok, CUSP_Ce_v, torch.zeros(80,60,device=device)).max(dim=1)
        ci = mc.argmax().item()
        cL = CUSP_Ls[ci].item()
        cC = mc[ci].item()
        cusp_Ls.append(cL)

        # Yellow-specific: penalize cusp L > 0.93
        if hd in YELLOW_HUES:
            if cL > 0.93:
                yellow_penalty += (cL - 0.93) ** 2 * 50

            # Cliff steepness: how fast does chroma drop after cusp?
            if ci < 78:  # room to check post-cusp
                post_cusp_C = mc[ci+2].item()  # 2 L-steps after cusp
                drop_rate = (cC - post_cusp_C) / max(cC, 0.01)
                # OKLab drops ~30% over similar range; penalize > 50%
                if drop_rate > 0.50:
                    yellow_penalty += (drop_rate - 0.50) ** 2 * 20

    # Cusp smoothness: penalize large jumps between adjacent scanned hues
    smoothness_pen = 0.0
    for i in range(len(cusp_Ls)):
        j = (i + 1) % len(cusp_Ls)
        jump = abs(cusp_Ls[i] - cusp_Ls[j])
        if jump > 0.05:  # 15° apart, max 0.05 L jump
            smoothness_pen += (jump - 0.05) ** 2

    return yellow_penalty + smoothness_pen

# ── Parameterization ──
def ortho(s):
    sn=s/np.linalg.norm(s)
    v=np.array([1,0,0.]) if abs(sn[0])<0.9 else np.array([0,1,0.])
    e1=v-np.dot(v,sn)*sn; e1/=np.linalg.norm(e1); e2=np.cross(sn,e1)
    return e1,e2

def unpack(x):
    M1=np.zeros((3,3))
    for i in range(3):
        M1[i,0]=x[2*i]; M1[i,1]=x[2*i+1]
        M1[i,2]=(1-M1[i,0]*D65[0]-M1[i,1]*D65[1])/D65[2]
    lms=M1@D65
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
    lms=M1@D65; s=lms**(1/3)
    e1,e2=ortho(s)
    x[9]=M2[1]@e1; x[10]=M2[1]@e2; x[11]=M2[2]@e1; x[12]=M2[2]@e2
    return x

# ── Baseline ──
print("\n--- v14 baseline cusp analysis ---", flush=True)
M1t=torch.tensor(V14_M1,device=device); M2t=torch.tensor(V14_M2,device=device)
with torch.no_grad():
    cp0 = gpu_cusp_penalty(M1t, M2t)
    cv0 = gpu_cv(M1t, M2t)
    inf0 = gpu_info(M1t, M2t)
print(f"  v14: CV={cv0*100:.2f}% cusp_pen={cp0:.4f} yL={inf0['yL']:.3f} yC={inf0['yC']:.3f} bw={inf0['bw']:.2f}", flush=True)

# ── Sweep cusp weight ──
x_v14 = pack(V14_M1, V14_M2)
all_results = []

for cusp_w in [1.0, 3.0, 10.0]:
    print(f"\n{'='*60}", flush=True)
    print(f"  cusp_w={cusp_w}", flush=True)
    print(f"{'='*60}", flush=True)

    best_loss = 999.; best_x = x_v14.copy()
    t0 = time.time(); ev = [0]; lp = [0]

    def objective(x):
        global best_loss, best_x
        try:
            M1n,M2n = unpack(x)
            if M1n is None: return 999.
            M1t=torch.tensor(M1n,device=device); M2t=torch.tensor(M2n,device=device)
            with torch.no_grad():
                c1,c2=torch.linalg.cond(M1t).item(),torch.linalg.cond(M2t).item()
                if c1>20 or c2>30: return 999.
                info=gpu_info(M1t,M2t)
                cv=gpu_cv(M1t,M2t)
                hue=gpu_hue(M1t,M2t)
                cusp=gpu_cusp_penalty(M1t,M2t)

                pen=0.
                if info['yC']<0.12: pen+=(0.12-info['yC'])**2*500
                if info['bw']<1.15: pen+=(1.15-info['bw'])**2*500
                if info['plr']<0.35: pen+=(0.35-info['plr'])**2*500
                if c1>5: pen+=(c1-5)**2*5
                if c2>12: pen+=(c2-12)**2*5

                loss = cv + 0.01*hue + cusp_w*cusp + pen
        except: return 999.
        ev[0]+=1
        if loss<best_loss:
            best_loss=loss; best_x=x.copy()
            now=time.time()
            if now-lp[0]>20:
                lp[0]=now
                print(f"  #{ev[0]:>5d} [{now-t0:4.0f}s] loss={loss:.4f} CV={cv*100:.1f}% cusp={cusp:.3f} yL={info['yL']:.3f} yC={info['yC']:.3f} bw={info['bw']:.2f} plr={info['plr']:.3f}", flush=True)
        return loss

    opts=cma.CMAOptions()
    opts.set("maxiter",500); opts.set("popsize",96); opts.set("tolfun",1e-11); opts.set("verbose",-1)
    es=cma.CMAEvolutionStrategy(x_v14.copy(),0.02,opts)
    while not es.stop():
        sols=es.ask(); fits=[objective(x) for x in sols]; es.tell(sols,fits)
    el=time.time()-t0

    M1f,M2f=unpack(best_x)
    M1t=torch.tensor(M1f,device=device); M2t=torch.tensor(M2f,device=device)
    with torch.no_grad():
        cv=gpu_cv(M1t,M2t); hue=gpu_hue(M1t,M2t)
        cusp=gpu_cusp_penalty(M1t,M2t); inf=gpu_info(M1t,M2t)
    print(f"  DONE: {ev[0]} evals {el:.0f}s | loss={best_loss:.4f} CV={cv*100:.2f}% cusp={cusp:.3f} hue={hue:.1f} yL={inf['yL']:.3f} yC={inf['yC']:.3f} bw={inf['bw']:.2f} plr={inf['plr']:.3f}", flush=True)

    # Yellow cusp L scan
    M2it=torch.linalg.inv(M2t); M1it=torch.linalg.inv(M1t)
    print(f"  Yellow cusp (h=85°):", flush=True)
    hr=85*3.14159265/180; ch,sh=np.cos(hr),np.sin(hr)
    for Lv in [0.80,0.85,0.90,0.93,0.95,0.97,0.99]:
        Cs=torch.linspace(0.001,0.4,80,device=device)
        lab=torch.stack([torch.full((80,),Lv,device=device),Cs*ch,Cs*sh],dim=1)
        lc=lab@M2it.T; lm=torch.sign(lc)*torch.abs(lc).pow(3.)
        lin=(lm@M1it.T)@M_Si.T
        ok=(lin>=-0.002).all(dim=1)&(lin<=1.002).all(dim=1)
        mc=Cs[ok].max().item() if ok.any() else 0
        print(f"    L={Lv:.2f} C={mc:.4f}", flush=True)

    # Save
    M1i,M2i=np.linalg.inv(M1f),np.linalg.inv(M2f)
    fn=f"/root/gen_v29_cw{cusp_w:.0f}.json"
    ckpt={"version":f"v29-cw{cusp_w}","M1":M1f.tolist(),"M2":M2f.tolist(),
          "M1_inv":M1i.tolist(),"M2_inv":M2i.tolist()}
    with open(fn,"w") as f: json.dump(ckpt,f,indent=2)
    all_results.append((cusp_w, best_loss, cv, cusp, inf, fn))

# ── Summary + best production test ──
print(f"\n{'='*60}")
print(f"  SWEEP SUMMARY")
print(f"{'='*60}")
for cw, loss, cv, cusp, inf, fn in all_results:
    print(f"  cw={cw:>4.0f}: loss={loss:.4f} CV={cv*100:.2f}% cusp={cusp:.3f} yL={inf['yL']:.3f} yC={inf['yC']:.3f}")

# Test best (lowest cusp penalty with cv < 26%)
viable = [(cw,l,cv,cp,inf,fn) for cw,l,cv,cp,inf,fn in all_results if cv < 0.26]
if viable:
    viable.sort(key=lambda r: r[3])  # sort by cusp penalty
    best = viable[0]
    print(f"\n  BEST: cw={best[0]} CV={best[2]*100:.2f}% cusp={best[3]:.3f} yL={best[4]['yL']:.3f}")
    print(f"\n{'='*60}")
    print(f"  PRODUCTION TEST")
    print(f"{'='*60}\n")
    subprocess.run([sys.executable, "/root/production_test_gpu.py", "--json", best[5]])

if __name__ == "__main__":
    pass
