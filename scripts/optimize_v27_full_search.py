"""v27: Full search from scratch with cusp-aware constraints.
No starting bias from v14 or OKLab. Find OUR optimal M1/M2.

Multi-seed: 10 random starting points + v14 + OKLab neighborhoods.
Best seed → long CMA-ES refinement.
"""
import json, time, numpy as np, torch
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
import cma

D65 = np.array([0.95047, 1.0, 1.08883])
M_S = torch.tensor([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]], device=device)
M_Si = torch.linalg.inv(M_S)

# References for comparison
V14_M1 = np.array([[0.7583761294836658,0.38380162590825084,-0.09608055040602373],[0.12671393631532843,0.8421628149123207,0.03434823621506485],[0.07639223722200054,0.258943526275451,0.6139139663787314]])
V14_M2 = np.array([[0.10058070589596230,1.01558970993941444,-0.11617041583537688],[2.36157646996164416,-2.44099737506293479,0.07942090510129070],[0.04565327074453784,0.81875488445424471,-0.86440815519878267]])
OK_M1s = np.array([[0.4122214708,0.5363325363,0.0514459929],[0.2119034982,0.6806995451,0.1073969566],[0.0883024619,0.2817188376,0.6299787005]])
OK_M1 = OK_M1s @ np.linalg.inv(np.array([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]]))
OK_M2 = np.array([[0.2104542553,0.7936177850,-0.0040720468],[1.9779984951,-2.4285922050,0.4505937099],[0.0259040371,0.7827717662,-0.8086757660]])

def scbrt_t(x): return torch.sign(x)*torch.abs(x).pow(1./3.)
def s2l_t(c): return torch.where(c<=0.04045,c/12.92,((c+0.055)/1.055).pow(2.4))
def l2s_t(c): return torch.where(c<=0.0031308,c*12.92,1.055*c.clamp(min=1e-10).pow(1./2.4)-0.055)

# Training pairs
def build_pairs():
    pairs=[]
    prims=[[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[0,0,0]]
    for i in range(len(prims)):
        for j in range(i+1,len(prims)):pairs.append((prims[i],prims[j]))
    for g1 in [0.0,0.2,0.4,0.6,0.8,1.0]:
        for g2 in [g1+0.2,g1+0.4]:
            if g2<=1.0:pairs.append(([g1]*3,[g2]*3))
    rng=np.random.RandomState(42)
    for _ in range(80):pairs.append((rng.rand(3).tolist(),rng.rand(3).tolist()))
    pt=torch.zeros(len(pairs),2,3,device=device)
    for i,(c1,c2) in enumerate(pairs):
        pt[i,0]=M_S@s2l_t(torch.tensor(c1,device=device))
        pt[i,1]=M_S@s2l_t(torch.tensor(c2,device=device))
    return pt

N_ST=25;T_ST=torch.linspace(0,1,N_ST+1,device=device)
D65_T=torch.tensor(D65,device=device)

def gpu_cv(M1,M2,pairs):
    M1i,M2i=torch.linalg.inv(M1),torch.linalg.inv(M2)
    N=pairs.shape[0]
    l1=scbrt_t(pairs[:,0]@M1.T)@M2.T;l2=scbrt_t(pairs[:,1]@M1.T)@M2.T
    t=T_ST.view(1,-1,1);labs=l1.unsqueeze(1)+t*(l2-l1).unsqueeze(1)
    lf=labs.reshape(-1,3);lc=lf@M2i.T;lm=torch.sign(lc)*torch.abs(lc).pow(3.)
    lin=(lm@M1i.T)@M_Si.T;s8=(l2s_t(lin.clamp(0,1))*255).round()/255.
    xb=s2l_t(s8)@M_S.T;r=xb.clamp(min=1e-10)/D65_T
    f=torch.where(r>0.008856,r.pow(1./3.),7.787*r+16./116.)
    cl=torch.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],dim=-1).reshape(N,N_ST+1,3)
    c1,c2=cl[:,:-1],cl[:,1:]
    dL=c2[...,0]-c1[...,0];C1=(c1[...,1]**2+c1[...,2]**2).sqrt();C2=(c2[...,1]**2+c2[...,2]**2).sqrt()
    dC=C2-C1;dH=((c2[...,1]-c1[...,1])**2+(c2[...,2]-c1[...,2])**2-dC**2).clamp(min=0).sqrt()
    SL=1+0.015*(c1[...,0]-50)**2/(20+(c1[...,0]-50)**2).sqrt();SC=1+0.045*C1;SH=1+0.015*C1
    de=((dL/SL)**2+(dC/SC)**2+(dH/SH)**2).sqrt()
    md=de.mean(1);sd=de.std(1);v=md>0.001
    cvs=torch.where(v,sd/md,torch.zeros_like(md))
    return cvs[v].mean().item() if v.any() else 999.

# Mono: dense grid, GPU batch
Lf=torch.linspace(0.80,0.998,50,device=device)
Cf=torch.linspace(0.001,0.4,50,device=device)
Le=Lf.view(50,1).expand(50,50);Ce=Cf.view(1,50).expand(50,50)
Cf_e=Cf.view(1,50).expand(50,50)
MONO_H=[h*3.14159265/180 for h in range(60,121,5)]

def gpu_mono(M1,M2):
    M1i,M2i=torch.linalg.inv(M1),torch.linalg.inv(M2)
    pen=0.
    for hr in MONO_H:
        ch,sh=np.cos(hr),np.sin(hr)
        lab=torch.stack([Le,Ce*ch,Ce*sh],dim=-1).reshape(-1,3)
        lc=lab@M2i.T;lm=torch.sign(lc)*torch.abs(lc).pow(3.);lin=(lm@M1i.T)@M_Si.T
        ok=((lin>=-0.002).all(dim=1)&(lin<=1.002).all(dim=1)).reshape(50,50)
        mc,_=torch.where(ok,Cf_e,torch.zeros(50,50,device=device)).max(dim=1)
        ci=mc.argmax().item();cL=Lf[ci].item()
        if cL>0.95:pen+=(cL-0.95)**2*200
        d=mc[1:]-mc[:-1];p=d[d>0.002]
        if p.numel()>0:pen+=p.pow(2).sum().item()*100
    return pen/len(MONO_H)

def gpu_hue(M1,M2):
    prims=torch.tensor([[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1]],dtype=torch.float64,device=device)
    exp=torch.tensor([0,60,120,180,240,300],dtype=torch.float64,device=device)
    lab=scbrt_t(s2l_t(prims)@M_S.T@M1.T)@M2.T
    h=torch.atan2(lab[:,2],lab[:,1])*(180/3.14159265)%360
    dh=h-exp;dh=torch.where(dh>180,dh-360,dh);dh=torch.where(dh<-180,dh+360,dh)
    return (dh**2).mean().item()

def gpu_constraints(M1,M2):
    M1i,M2i=torch.linalg.inv(M1),torch.linalg.inv(M2)
    yl=scbrt_t((M_S@torch.tensor([1.,1.,0.],device=device))@M1.T)@M2.T
    yL,yC=yl[0].item(),(yl[1]**2+yl[2]**2).sqrt().item()
    bx=M_S@s2l_t(torch.tensor([0.,0.,1.],device=device));wx=M_S@torch.tensor([1.,1.,1.],device=device)
    bl=scbrt_t(bx@M1.T)@M2.T;wl=scbrt_t(wx@M1.T)@M2.T;ml=(bl+wl)/2
    lc=ml@M2i.T;lm=torch.sign(lc)*torch.abs(lc).pow(3.);mx=lm@M1i.T
    ms=l2s_t((M_Si@mx).clamp(0,1))
    bw=ms[1].item()/max(ms[0].item(),1e-10)
    ps=torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]],dtype=torch.float64,device=device)
    pl=scbrt_t(s2l_t(ps)@M_S.T@M1.T)@M2.T
    plr=(pl[:,0].max()-pl[:,0].min()).item()
    return yL,yC,bw,plr,torch.linalg.cond(M1).item(),torch.linalg.cond(M2).item()

# Parameterization
def ortho(s):
    sn=s/np.linalg.norm(s)
    v=np.array([1,0,0.]) if abs(sn[0])<0.9 else np.array([0,1,0.])
    e1=v-np.dot(v,sn)*sn;e1/=np.linalg.norm(e1);e2=np.cross(sn,e1)
    return e1,e2

def unpack(x):
    M1=np.zeros((3,3))
    for i in range(3):
        M1[i,0]=x[2*i];M1[i,1]=x[2*i+1]
        M1[i,2]=(1-M1[i,0]*D65[0]-M1[i,1]*D65[1])/D65[2]
    s=np.sign(M1@D65)*np.abs(M1@D65)**(1/3)
    if np.linalg.norm(s)<1e-10:return None,None
    e1,e2=ortho(s)
    M2=np.zeros((3,3));M2[0]=x[6:9]
    Lw=M2[0]@s
    if abs(Lw)<1e-10:return None,None
    M2[0]/=Lw
    M2[1]=x[9]*e1+x[10]*e2;M2[2]=x[11]*e1+x[12]*e2
    return M1,M2

def pack(M1,M2):
    x=np.zeros(13)
    for i in range(3):x[2*i]=M1[i,0];x[2*i+1]=M1[i,1]
    x[6:9]=M2[0]
    s=np.sign(M1@D65)*np.abs(M1@D65)**(1/3)
    e1,e2=ortho(s)
    x[9]=M2[1]@e1;x[10]=M2[1]@e2;x[11]=M2[2]@e1;x[12]=M2[2]@e2
    return x

def make_objective(pairs):
    def objective(x):
        try:
            M1n,M2n=unpack(x)
            if M1n is None:return 999.
            M1t=torch.tensor(M1n,device=device);M2t=torch.tensor(M2n,device=device)
            with torch.no_grad():
                c1,c2=torch.linalg.cond(M1t).item(),torch.linalg.cond(M2t).item()
                if c1>8 or c2>15:return 100+c1+c2
                yL,yC,bw,plr,_,_=gpu_constraints(M1t,M2t)
                viol=0
                if yC<0.15:viol+=(0.15-yC)**2*2000
                if yL<0.90:viol+=(0.90-yL)**2*2000
                if bw<1.20:viol+=(1.20-bw)**2*2000
                if plr<0.40:viol+=(0.40-plr)**2*2000
                if c1>4:viol+=(c1-4)**2*20
                if viol>0:return 50+viol
                cv=gpu_cv(M1t,M2t,pairs)
                if cv>0.28:return 30+(cv-0.28)**2*200
                mono=gpu_mono(M1t,M2t)
                hue=gpu_hue(M1t,M2t)
                return cv+0.3*cv+3.0*mono+0.3*hue
        except:return 999.
    return objective

def run_seed(name,x0,sigma,gens,popsize,pairs,obj_fn):
    best={'loss':999,'x':x0.copy()};ev=[0];t0=time.time()
    def obj(x):
        loss=obj_fn(x);ev[0]+=1
        if loss<best['loss']:
            best['loss']=loss;best['x']=x.copy()
        return loss
    opts=cma.CMAOptions()
    opts.set("maxiter",gens);opts.set("popsize",popsize);opts.set("tolfun",1e-10);opts.set("verbose",-1)
    es=cma.CMAEvolutionStrategy(x0,sigma,opts)
    while not es.stop():
        sols=es.ask();fits=[obj(x) for x in sols];es.tell(sols,fits)
    el=time.time()-t0
    # Get info
    M1n,M2n=unpack(best['x'])
    M1t,M2t=torch.tensor(M1n,device=device),torch.tensor(M2n,device=device)
    with torch.no_grad():
        cv=gpu_cv(M1t,M2t,pairs);mono=gpu_mono(M1t,M2t);hue=gpu_hue(M1t,M2t)
        yL,yC,bw,plr,c1,c2=gpu_constraints(M1t,M2t)
    print(f"  {name}: {ev[0]} evals {el:.0f}s | loss={best['loss']:.4f} CV={cv*100:.2f}% mono={mono:.4f} hue={hue:.1f} yL={yL:.3f} yC={yC:.3f} bw={bw:.2f} cond=({c1:.1f},{c2:.1f})")
    return best['x'],best['loss'],{'cv':cv,'mono':mono,'hue':hue,'yL':yL,'yC':yC,'bw':bw,'c1':c1,'c2':c2,'plr':plr}


def main():
    print(f"\n{'='*60}")
    print("  v27: FULL SEARCH FROM SCRATCH")
    print("  Multi-seed, cusp-aware, GPU-native")
    print(f"{'='*60}\n")

    pairs=build_pairs()
    print(f"Training pairs: {pairs.shape[0]}")
    obj_fn=make_objective(pairs)

    # ── Phase 1: Multi-seed exploration (short runs) ──
    print(f"\n--- Phase 1: 12 seeds x 200 gen x 64 pop ---")
    seeds=[]

    # Seed from v14
    seeds.append(("v14", pack(V14_M1, V14_M2)))
    # Seed from OKLab
    seeds.append(("OKLab", pack(OK_M1, OK_M2)))
    # Seed: midpoint v14-OKLab
    mid_M1=(V14_M1+OK_M1)/2;mid_M2=(V14_M2+OK_M2)/2
    seeds.append(("mid", pack(mid_M1, mid_M2)))

    # Random seeds (D65-normalized)
    rng=np.random.RandomState(123)
    for i in range(9):
        x=np.zeros(13)
        # Random M1 (2 free per row, D65 normalized)
        for r in range(3):
            x[2*r]=rng.randn()*0.3+(V14_M1[r,0]+OK_M1[r,0])/2
            x[2*r+1]=rng.randn()*0.3+(V14_M1[r,1]+OK_M1[r,1])/2
        # M2 from v14 with noise
        x[6:9]=V14_M2[0]+rng.randn(3)*0.1
        x[9:13]=rng.randn(4)*1.0
        seeds.append((f"rnd{i}", x))

    phase1_results=[]
    for name,x0 in seeds:
        xb,loss,info=run_seed(name,x0,sigma=0.05,gens=200,popsize=64,pairs=pairs,obj_fn=obj_fn)
        phase1_results.append((name,xb,loss,info))

    # Sort by loss
    phase1_results.sort(key=lambda r:r[2])
    print(f"\n--- Phase 1 ranking ---")
    for i,(name,_,loss,info) in enumerate(phase1_results):
        flag="***" if i<3 else ""
        print(f"  {i+1}. {name:>6}: loss={loss:.4f} CV={info['cv']*100:.2f}% mono={info['mono']:.4f} hue={info['hue']:.1f} yL={info['yL']:.3f} yC={info['yC']:.3f} {flag}")

    # ── Phase 2: Top 3 seeds, long refinement ──
    print(f"\n--- Phase 2: Top 3 x 1000 gen x 96 pop ---")
    phase2_results=[]
    for name,x0,_,_ in phase1_results[:3]:
        xb,loss,info=run_seed(f"{name}_ref",x0,sigma=0.01,gens=1000,popsize=96,pairs=pairs,obj_fn=obj_fn)
        phase2_results.append((f"{name}_ref",xb,loss,info))

    phase2_results.sort(key=lambda r:r[2])
    print(f"\n--- Phase 2 ranking ---")
    for i,(name,_,loss,info) in enumerate(phase2_results):
        print(f"  {i+1}. {name}: loss={loss:.4f} CV={info['cv']*100:.2f}% mono={info['mono']:.4f} hue={info['hue']:.1f} yL={info['yL']:.3f} yC={info['yC']:.3f}")

    # ── Phase 3: Winner, final polish ──
    winner_name,winner_x,_,_=phase2_results[0]
    print(f"\n--- Phase 3: Winner '{winner_name}' x 500 gen x 128 pop, sigma=0.003 ---")
    final_x,final_loss,final_info=run_seed("FINAL",winner_x,sigma=0.003,gens=500,popsize=128,pairs=pairs,obj_fn=obj_fn)

    # Extract matrices
    M1f,M2f=unpack(final_x)
    M1i,M2i=np.linalg.inv(M1f),np.linalg.inv(M2f)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULT")
    print(f"{'='*60}")
    print(f"  CV={final_info['cv']*100:.2f}% mono={final_info['mono']:.4f} hue={final_info['hue']:.1f}")
    print(f"  yL={final_info['yL']:.3f} yC={final_info['yC']:.3f} bw={final_info['bw']:.2f} plr={final_info['plr']:.3f}")
    print(f"  cond=({final_info['c1']:.1f},{final_info['c2']:.1f})")

    # Yellow boundary
    M1t,M2t=torch.tensor(M1f,device=device),torch.tensor(M2f,device=device)
    M1it,M2it=torch.linalg.inv(M1t),torch.linalg.inv(M2t)
    print(f"\n  Yellow boundary (h=85deg):")
    h=85*3.14159265/180;ch,sh=np.cos(h),np.sin(h)
    prev=None
    for Lv in [0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.97,0.98,0.99,1.0]:
        Ls2=torch.linspace(Lv-0.001,Lv+0.001,1,device=device)
        Cs2=torch.linspace(0.001,0.4,60,device=device)
        Le2=torch.full((60,),Lv,device=device);Ce2=Cs2
        lab=torch.stack([Le2,Ce2*ch,Ce2*sh],dim=1)
        lc=lab@M2it.T;lm=torch.sign(lc)*torch.abs(lc).pow(3.);lin=(lm@M1it.T)@M_Si.T
        ok=(lin>=-0.002).all(dim=1)&(lin<=1.002).all(dim=1)
        mc=Cs2[ok].max().item() if ok.any() else 0
        arrow=""
        if prev is not None:arrow=" UP" if mc>prev+0.001 else " DN" if mc<prev-0.001 else " =="
        prev=mc
        print(f"    L={Lv:.2f} C={mc:.4f}{arrow}")

    print(f"\nM1 =")
    for r in M1f:print(f"  [{r[0]:>22.16f}, {r[1]:>22.16f}, {r[2]:>22.16f}],")
    print(f"M2 =")
    for r in M2f:print(f"  [{r[0]:>22.16f}, {r[1]:>22.16f}, {r[2]:>22.16f}],")
    print(f"M1_inv =")
    for r in M1i:print(f"  [{r[0]:>22.16f}, {r[1]:>22.16f}, {r[2]:>22.16f}],")
    print(f"M2_inv =")
    for r in M2i:print(f"  [{r[0]:>22.16f}, {r[1]:>22.16f}, {r[2]:>22.16f}],")

    ckpt={"version":"v27","M1":M1f.tolist(),"M2":M2f.tolist(),
          "M1_inv":M1i.tolist(),"M2_inv":M2i.tolist(),"metrics":final_info}
    with open("gen_v27.json","w") as f:json.dump(ckpt,f,indent=2)
    print(f"\nSaved: gen_v27.json")


if __name__=="__main__":main()
