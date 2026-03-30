"""v27 fixed: Full search with soft constraints, progress files, bug fixes.
Run on H100 or any CUDA GPU.
"""
import json,time,numpy as np,torch,os
torch.set_default_dtype(torch.float64)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory//1024**2} MB")
import cma

D65=np.array([0.95047,1.0,1.08883])
M_S=torch.tensor([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]],device=device)
M_Si=torch.linalg.inv(M_S)
D65_T=torch.tensor(D65,device=device)

V14_M1=np.array([[0.7583761294836658,0.38380162590825084,-0.09608055040602373],[0.12671393631532843,0.8421628149123207,0.03434823621506485],[0.07639223722200054,0.258943526275451,0.6139139663787314]])
V14_M2=np.array([[0.10058070589596230,1.01558970993941444,-0.11617041583537688],[2.36157646996164416,-2.44099737506293479,0.07942090510129070],[0.04565327074453784,0.81875488445424471,-0.86440815519878267]])
OK_M1s=np.array([[0.4122214708,0.5363325363,0.0514459929],[0.2119034982,0.6806995451,0.1073969566],[0.0883024619,0.2817188376,0.6299787005]])
OK_M1=OK_M1s@np.linalg.inv(np.array([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]]))
OK_M2=np.array([[0.2104542553,0.7936177850,-0.0040720468],[1.9779984951,-2.4285922050,0.4505937099],[0.0259040371,0.7827717662,-0.8086757660]])

def scbrt(x):return torch.sign(x)*torch.abs(x).pow(1./3.)
def s2l(c):return torch.where(c<=0.04045,c/12.92,((c+0.055)/1.055).pow(2.4))
def l2s(c):return torch.where(c<=0.0031308,c*12.92,1.055*c.clamp(min=1e-10).pow(1./2.4)-0.055)

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
        pt[i,0]=M_S@s2l(torch.tensor(c1,device=device))
        pt[i,1]=M_S@s2l(torch.tensor(c2,device=device))
    return pt

N_ST=25;T_ST=torch.linspace(0,1,N_ST+1,device=device)

def gpu_cv(M1,M2,pairs):
    try:
        M1i,M2i=torch.linalg.inv(M1),torch.linalg.inv(M2)
    except:return 0.99
    N=pairs.shape[0]
    l1=scbrt(pairs[:,0]@M1.T)@M2.T;l2=scbrt(pairs[:,1]@M1.T)@M2.T
    t=T_ST.view(1,-1,1);labs=l1.unsqueeze(1)+t*(l2-l1).unsqueeze(1)
    lf=labs.reshape(-1,3);lc=lf@M2i.T;lm=torch.sign(lc)*torch.abs(lc).pow(3.)
    lin=(lm@M1i.T)@M_Si.T;s8=(l2s(lin.clamp(0,1))*255).round()/255.
    xb=s2l(s8)@M_S.T;r=xb.clamp(min=1e-10)/D65_T
    f=torch.where(r>0.008856,r.pow(1./3.),7.787*r+16./116.)
    cl=torch.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],dim=-1).reshape(N,N_ST+1,3)
    c1,c2=cl[:,:-1],cl[:,1:]
    dL=c2[...,0]-c1[...,0];C1=(c1[...,1]**2+c1[...,2]**2).sqrt();C2=(c2[...,1]**2+c2[...,2]**2).sqrt()
    dC=C2-C1;dH=((c2[...,1]-c1[...,1])**2+(c2[...,2]-c1[...,2])**2-dC**2).clamp(min=0).sqrt()
    SL=1+0.015*(c1[...,0]-50)**2/(20+(c1[...,0]-50)**2).sqrt();SC=1+0.045*C1;SH=1+0.015*C1
    de=((dL/SL)**2+(dC/SC)**2+(dH/SH)**2).sqrt()
    md=de.mean(1);sd=de.std(1);v=md>0.001
    cvs=torch.where(v,sd/md,torch.zeros_like(md))
    return cvs[v].mean().item() if v.any() else 0.99

Lf=torch.linspace(0.80,0.998,50,device=device)
Cf=torch.linspace(0.001,0.4,50,device=device)
Le=Lf.view(50,1).expand(50,50);Ce=Cf.view(1,50).expand(50,50)
Cf_e=Cf.view(1,50).expand(50,50)
MONO_H=[h*3.14159265/180 for h in range(60,121,5)]

def gpu_mono(M1,M2):
    try:M1i,M2i=torch.linalg.inv(M1),torch.linalg.inv(M2)
    except:return 99.
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
    lab=scbrt(s2l(prims)@M_S.T@M1.T)@M2.T
    h=torch.atan2(lab[:,2],lab[:,1])*(180/3.14159265)%360
    dh=h-exp;dh=torch.where(dh>180,dh-360,dh);dh=torch.where(dh<-180,dh+360,dh)
    return (dh**2).mean().item()

def gpu_info(M1,M2):
    M1i,M2i=torch.linalg.inv(M1),torch.linalg.inv(M2)
    # Yellow
    yl=scbrt((M_S@torch.tensor([1.,1.,0.],device=device))@M1.T)@M2.T
    yL,yC=yl[0].item(),(yl[1]**2+yl[2]**2).sqrt().item()
    # Blue->White midpoint (FIXED: both endpoints properly converted)
    bx=M_S@s2l(torch.tensor([0.,0.,1.],device=device))
    wx=M_S@s2l(torch.tensor([1.,1.,1.],device=device))  # FIX: s2l on white too
    bl=scbrt(bx@M1.T)@M2.T;wl=scbrt(wx@M1.T)@M2.T
    ml=(bl+wl)/2;lc=ml@M2i.T;lm=torch.sign(lc)*torch.abs(lc).pow(3.);mx=lm@M1i.T
    ms=l2s((M_Si@mx).clamp(0,1))
    bw=ms[1].item()/max(ms[0].item(),0.01)  # FIX: clamp denominator
    # Primary L range
    ps=torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]],dtype=torch.float64,device=device)
    pl=scbrt(s2l(ps)@M_S.T@M1.T)@M2.T
    plr=(pl[:,0].max()-pl[:,0].min()).item()
    c1,c2=torch.linalg.cond(M1).item(),torch.linalg.cond(M2).item()
    return {'yL':yL,'yC':yC,'bw':bw,'plr':plr,'c1':c1,'c2':c2}

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
    lms=M1@D65
    if np.any(lms<=0):return None,None  # FIX: prevent negative LMS
    s=lms**(1/3)
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
    lms=M1@D65;s=lms**(1/3)
    e1,e2=ortho(s)
    x[9]=M2[1]@e1;x[10]=M2[1]@e2;x[11]=M2[2]@e1;x[12]=M2[2]@e2
    return x

def write_progress(phase, data):
    """Write progress file so we can monitor remotely."""
    with open(f"progress_phase{phase}.json","w") as f:
        json.dump(data, f, indent=2)

def make_objective(pairs):
    """Soft constraints - no hard cliffs!"""
    def objective(x):
        try:
            M1n,M2n=unpack(x)
            if M1n is None:return 999.
            M1t=torch.tensor(M1n,device=device);M2t=torch.tensor(M2n,device=device)
            with torch.no_grad():
                c1,c2=torch.linalg.cond(M1t).item(),torch.linalg.cond(M2t).item()
                if c1>20 or c2>30:return 999.  # only reject truly degenerate

                info=gpu_info(M1t,M2t)
                cv=gpu_cv(M1t,M2t,pairs)
                mono=gpu_mono(M1t,M2t)
                hue=gpu_hue(M1t,M2t)

                # SOFT penalties (smooth, no cliffs)
                pen=0.
                if info['yC']<0.15:pen+=(0.15-info['yC'])**2*500
                if info['yL']<0.90:pen+=(0.90-info['yL'])**2*500
                if info['bw']<1.15:pen+=(1.15-info['bw'])**2*500
                if info['plr']<0.40:pen+=(0.40-info['plr'])**2*500
                if c1>4:pen+=(c1-4)**2*5
                if c2>12:pen+=(c2-12)**2*5
                if cv>0.25:pen+=(cv-0.25)**2*50  # soft ramp, not cliff

                loss=cv+0.3*cv+3.0*mono+0.01*hue+pen
                return loss
        except:return 999.
    return objective

def run_seed(name,x0,sigma,gens,popsize,pairs,obj_fn):
    best={'loss':999,'x':x0.copy()};ev=[0];t0=time.time()
    last_print=[0]
    def obj(x):
        loss=obj_fn(x);ev[0]+=1
        if loss<best['loss']:
            best['loss']=loss;best['x']=x.copy()
            now=time.time()
            if now-last_print[0]>10:
                last_print[0]=now
                # Quick info for printing
                M1n,M2n=unpack(x)
                if M1n is not None:
                    M1t=torch.tensor(M1n,device=device);M2t=torch.tensor(M2n,device=device)
                    with torch.no_grad():
                        cv=gpu_cv(M1t,M2t,pairs);mono=gpu_mono(M1t,M2t)
                        inf=gpu_info(M1t,M2t)
                    print(f"    #{ev[0]:>6d} [{now-t0:5.0f}s] loss={loss:.4f} CV={cv*100:.1f}% mono={mono:.4f} yL={inf['yL']:.3f} yC={inf['yC']:.3f} bw={inf['bw']:.2f} cond=({inf['c1']:.1f},{inf['c2']:.1f})",flush=True)
        return loss
    opts=cma.CMAOptions()
    opts.set("maxiter",gens);opts.set("popsize",popsize);opts.set("tolfun",1e-11);opts.set("verbose",-1)
    es=cma.CMAEvolutionStrategy(x0,sigma,opts)
    while not es.stop():
        sols=es.ask();fits=[obj(x) for x in sols];es.tell(sols,fits)
    el=time.time()-t0
    M1n,M2n=unpack(best['x'])
    M1t,M2t=torch.tensor(M1n,device=device),torch.tensor(M2n,device=device)
    with torch.no_grad():
        cv=gpu_cv(M1t,M2t,pairs);mono=gpu_mono(M1t,M2t);hue=gpu_hue(M1t,M2t)
        inf=gpu_info(M1t,M2t)
    result={'name':name,'evals':ev[0],'time':el,'loss':best['loss'],
            'cv':cv,'mono':mono,'hue':hue,**inf}
    print(f"  {name}: {ev[0]} evals {el:.0f}s | loss={best['loss']:.4f} CV={cv*100:.2f}% mono={mono:.4f} hue={hue:.1f} yL={inf['yL']:.3f} yC={inf['yC']:.3f} bw={inf['bw']:.2f} cond=({inf['c1']:.1f},{inf['c2']:.1f})",flush=True)
    return best['x'],best['loss'],result

def main():
    print(f"\n{'='*60}")
    print("  v27 FIXED: Soft constraints, no cliffs")
    print(f"{'='*60}\n",flush=True)

    pairs=build_pairs()
    obj_fn=make_objective(pairs)

    # Verify v14 baseline
    print("--- Baseline verification ---",flush=True)
    x_v14=pack(V14_M1,V14_M2)
    M1c,M2c=unpack(x_v14)
    M1t,M2t=torch.tensor(M1c,device=device),torch.tensor(M2c,device=device)
    with torch.no_grad():
        cv0=gpu_cv(M1t,M2t,pairs);mono0=gpu_mono(M1t,M2t);hue0=gpu_hue(M1t,M2t)
        inf0=gpu_info(M1t,M2t)
    print(f"  v14: CV={cv0*100:.2f}% mono={mono0:.4f} hue={hue0:.1f} yL={inf0['yL']:.3f} yC={inf0['yC']:.3f} bw={inf0['bw']:.2f} cond=({inf0['c1']:.1f},{inf0['c2']:.1f})")
    loss0=obj_fn(x_v14)
    print(f"  v14 loss={loss0:.4f} (should NOT be 30 or 999!)")

    x_ok=pack(OK_M1,OK_M2)
    loss_ok=obj_fn(x_ok)
    M1c2,M2c2=unpack(x_ok)
    M1t2,M2t2=torch.tensor(M1c2,device=device),torch.tensor(M2c2,device=device)
    with torch.no_grad():
        cv_ok=gpu_cv(M1t2,M2t2,pairs)
    print(f"  OKLab: CV={cv_ok*100:.2f}% loss={loss_ok:.4f}")
    print(flush=True)

    if loss0>50:
        print(f"ERROR: v14 baseline loss={loss0:.2f} > 50, something is still broken!")
        return

    # ── Phase 1: 12 seeds x 300 gen x 64 pop ──
    print("--- Phase 1: 12 seeds x 300 gen x 64 pop ---",flush=True)
    seeds=[
        ("v14",pack(V14_M1,V14_M2),0.01),
        ("OKLab",pack(OK_M1,OK_M2),0.01),
        ("mid",pack((V14_M1+OK_M1)/2,(V14_M2+OK_M2)/2),0.02),
    ]
    rng=np.random.RandomState(777)
    for i in range(9):
        x=np.zeros(13)
        base_M1=(V14_M1+OK_M1)/2
        for r in range(3):
            x[2*r]=base_M1[r,0]+rng.randn()*0.15
            x[2*r+1]=base_M1[r,1]+rng.randn()*0.15
        x[6:9]=(V14_M2[0]+OK_M2[0])/2+rng.randn(3)*0.1
        x[9:13]=rng.randn(4)*0.8
        seeds.append((f"rnd{i}",x,0.05))

    p1_results=[]
    for name,x0,sigma in seeds:
        xb,loss,info=run_seed(name,x0,sigma,gens=300,popsize=64,pairs=pairs,obj_fn=obj_fn)
        p1_results.append((name,xb,loss,info))

    p1_results.sort(key=lambda r:r[2])
    print(f"\n--- Phase 1 ranking ---",flush=True)
    for i,(name,_,loss,info) in enumerate(p1_results):
        flag="***" if i<3 else ""
        print(f"  {i+1:>2}. {name:>6}: loss={loss:.4f} CV={info['cv']*100:.2f}% mono={info['mono']:.4f} hue={info['hue']:.1f} yL={info['yL']:.3f} yC={info['yC']:.3f} bw={info['bw']:.2f} {flag}")
    write_progress(1,[{'name':n,'loss':l,'metrics':m} for n,_,l,m in p1_results])
    print(flush=True)

    # ── Phase 2: Top 3 x 1500 gen x 96 pop ──
    print("--- Phase 2: Top 3 x 1500 gen x 96 pop ---",flush=True)
    p2_results=[]
    for name,x0,_,_ in p1_results[:3]:
        xb,loss,info=run_seed(f"{name}+",x0,sigma=0.005,gens=1500,popsize=96,pairs=pairs,obj_fn=obj_fn)
        p2_results.append((f"{name}+",xb,loss,info))

    p2_results.sort(key=lambda r:r[2])
    print(f"\n--- Phase 2 ranking ---",flush=True)
    for i,(name,_,loss,info) in enumerate(p2_results):
        print(f"  {i+1}. {name}: loss={loss:.4f} CV={info['cv']*100:.2f}% mono={info['mono']:.4f} hue={info['hue']:.1f} yL={info['yL']:.3f} yC={info['yC']:.3f}")
    write_progress(2,[{'name':n,'loss':l,'metrics':m} for n,_,l,m in p2_results])
    print(flush=True)

    # ── Phase 3: Winner polish ──
    wn,wx,_,_=p2_results[0]
    print(f"--- Phase 3: '{wn}' x 800 gen x 128 pop, sigma=0.002 ---",flush=True)
    fx,fl,fi=run_seed("FINAL",wx,sigma=0.002,gens=800,popsize=128,pairs=pairs,obj_fn=obj_fn)

    M1f,M2f=unpack(fx)
    M1i,M2i=np.linalg.inv(M1f),np.linalg.inv(M2f)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULT")
    print(f"{'='*60}")
    print(f"  CV={fi['cv']*100:.2f}% mono={fi['mono']:.4f} hue={fi['hue']:.1f}")
    print(f"  yL={fi['yL']:.3f} yC={fi['yC']:.3f} bw={fi['bw']:.2f} plr={fi['plr']:.3f}")
    print(f"  cond=({fi['c1']:.1f},{fi['c2']:.1f})")

    # Yellow boundary
    M1t,M2t=torch.tensor(M1f,device=device),torch.tensor(M2f,device=device)
    M1it,M2it=torch.linalg.inv(M1t),torch.linalg.inv(M2t)
    print(f"\n  Yellow boundary (h=85deg):")
    h=85*3.14159265/180;ch,sh=np.cos(h),np.sin(h)
    prev=None
    for Lv in [0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.97,0.98,0.99,1.0]:
        Cs=torch.linspace(0.001,0.4,80,device=device)
        Le2=torch.full((80,),Lv,device=device)
        lab=torch.stack([Le2,Cs*ch,Cs*sh],dim=1)
        lc=lab@M2it.T;lm=torch.sign(lc)*torch.abs(lc).pow(3.);lin=(lm@M1it.T)@M_Si.T
        ok=(lin>=-0.002).all(dim=1)&(lin<=1.002).all(dim=1)
        mc=Cs[ok].max().item() if ok.any() else 0
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
          "M1_inv":M1i.tolist(),"M2_inv":M2i.tolist(),"metrics":fi}
    with open("gen_v27.json","w") as f:json.dump(ckpt,f,indent=2)
    print(f"\nSaved: gen_v27.json",flush=True)
    write_progress(3,fi)

if __name__=="__main__":main()
