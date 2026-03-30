"""v26 Blitz: 4 fast experiments on GPU, aggressive parallelism.

Exp A: M2 L-row only (2 params), v14 M1+M2 ab-rows fixed
Exp B: Ab-plane rotation (1 param), v14 M1/M2 fixed
Exp C: OKLab M1 + v14 M2 (direct combine, no optimization)
Exp D: OKLab basin micro-tune (sigma=0.001, 500 gen, pop=96)

All experiments run sequentially, results compared at end.
"""

import json, time, sys, numpy as np
import torch
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory//1024**2} MB")

import cma

D65 = torch.tensor([0.95047, 1.0, 1.08883], device=device)
D65_NP = np.array([0.95047, 1.0, 1.08883])

M_SRGB = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], device=device)
M_SRGB_INV = torch.linalg.inv(M_SRGB)

OKLAB_M1_SRGB = torch.tensor([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], device=device)
OKLAB_M1 = OKLAB_M1_SRGB @ M_SRGB_INV
OKLAB_M2 = torch.tensor([
    [ 0.2104542553,  0.7936177850, -0.0040720468],
    [ 1.9779984951, -2.4285922050,  0.4505937099],
    [ 0.0259040371,  0.7827717662, -0.8086757660],
], device=device)

V14_M1 = torch.tensor([
    [0.7583761294836658, 0.38380162590825084, -0.09608055040602373],
    [0.12671393631532843, 0.8421628149123207, 0.03434823621506485],
    [0.07639223722200054, 0.258943526275451, 0.6139139663787314],
], device=device)
V14_M2 = torch.tensor([
    [0.10058070589596230, 1.01558970993941444, -0.11617041583537688],
    [2.36157646996164416, -2.44099737506293479, 0.07942090510129070],
    [0.04565327074453784, 0.81875488445424471, -0.86440815519878267],
], device=device)

def signed_cbrt(x):
    return torch.sign(x) * torch.abs(x).pow(1.0/3.0)

def srgb_to_linear(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))

def linear_to_srgb(c):
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c.clamp(min=1e-10).pow(1.0/2.4) - 0.055)

def xyz_to_cielab_batch(xyz):
    r = xyz / D65
    f = torch.where(r > 0.008856, r.pow(1.0/3.0), 7.787 * r + 16.0/116.0)
    return torch.stack([116*f[...,1]-16, 500*(f[...,0]-f[...,1]), 200*(f[...,1]-f[...,2])], dim=-1)

def forward_batch(M1, M2, xyz):
    return signed_cbrt(xyz @ M1.T) @ M2.T

def inverse_batch(M1i, M2i, lab):
    lc = lab @ M2i.T
    return (torch.sign(lc)*torch.abs(lc).pow(3.0)) @ M1i.T

def build_pairs():
    pairs = []
    prims = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[0,0,0]]
    for i in range(len(prims)):
        for j in range(i+1, len(prims)):
            pairs.append((prims[i], prims[j]))
    for g1 in [0.0,0.2,0.4,0.6,0.8,1.0]:
        for g2 in [g1+0.2, g1+0.4]:
            if g2 <= 1.0: pairs.append(([g1]*3, [g2]*3))
    rng = np.random.RandomState(42)
    for _ in range(80):
        pairs.append((rng.rand(3).tolist(), rng.rand(3).tolist()))
    pt = torch.zeros(len(pairs), 2, 3, device=device)
    for i, (c1, c2) in enumerate(pairs):
        pt[i,0] = M_SRGB @ srgb_to_linear(torch.tensor(c1, device=device))
        pt[i,1] = M_SRGB @ srgb_to_linear(torch.tensor(c2, device=device))
    return pt

N_STEPS = 25
T_STEPS = torch.linspace(0, 1, N_STEPS+1, device=device)

def compute_cv(M1, M2, pairs):
    M1i, M2i = torch.linalg.inv(M1), torch.linalg.inv(M2)
    N = pairs.shape[0]
    lab1, lab2 = forward_batch(M1, M2, pairs[:,0]), forward_batch(M1, M2, pairs[:,1])
    t = T_STEPS.view(1,-1,1)
    labs = lab1.unsqueeze(1) + t*(lab2-lab1).unsqueeze(1)
    lf = labs.reshape(-1,3)
    xf = inverse_batch(M1i, M2i, lf)
    lin = (xf @ M_SRGB_INV.T).clamp(0,1)
    s8 = (linear_to_srgb(lin)*255).round()/255.0
    xb = srgb_to_linear(s8) @ M_SRGB.T
    cl = xyz_to_cielab_batch(xb.clamp(min=1e-10)).reshape(N, N_STEPS+1, 3)
    c1, c2 = cl[:,:-1], cl[:,1:]
    dL = c2[...,0]-c1[...,0]
    C1 = (c1[...,1]**2+c1[...,2]**2).sqrt()
    C2 = (c2[...,1]**2+c2[...,2]**2).sqrt()
    dC = C2-C1
    dH = ((c2[...,1]-c1[...,1])**2+(c2[...,2]-c1[...,2])**2-dC**2).clamp(min=0).sqrt()
    SL = 1+0.015*(c1[...,0]-50)**2/(20+(c1[...,0]-50)**2).sqrt()
    SC, SH = 1+0.045*C1, 1+0.015*C1
    de = ((dL/SL)**2+(dC/SC)**2+(dH/SH)**2).sqrt()
    md = de.mean(dim=1); sd = de.std(dim=1)
    v = md > 0.001
    cvs = torch.where(v, sd/md, torch.zeros_like(md))
    return cvs[v].mean().item() if v.any() else 999.0

MONO_HUES = torch.arange(60, 121, 3, device=device, dtype=torch.float64) * (3.14159265358979/180.0)
MONO_LS = torch.arange(0.80, 1.001, 0.004, device=device)

def compute_mono(M1, M2):
    M1i, M2i = torch.linalg.inv(M1), torch.linalg.inv(M2)
    pen = 0.0
    for hi in range(MONO_HUES.shape[0]):
        h = MONO_HUES[hi]; ch, sh = torch.cos(h), torch.sin(h)
        lo = torch.zeros(MONO_LS.shape[0], device=device)
        hi_c = torch.full_like(lo, 0.5)
        for _ in range(35):
            mid = (lo+hi_c)/2
            lab = torch.stack([MONO_LS, mid*ch, mid*sh], dim=1)
            lc = lab @ M2i.T; lm = torch.sign(lc)*torch.abs(lc).pow(3.0)
            lin = (lm @ M1i.T) @ M_SRGB_INV.T
            ok = (lin>=-0.001).all(dim=1) & (lin<=1.001).all(dim=1)
            lo = torch.where(ok, mid, lo); hi_c = torch.where(ok, hi_c, mid)
        ci = lo.argmax(); cL = MONO_LS[ci].item()
        if cL > 0.975: pen += (cL-0.975)**2*100
        diffs = lo[1:]-lo[:-1]
        pos = diffs[diffs > 1e-5]
        if pos.numel() > 0: pen += (pos/0.004).pow(2).sum().item()
    return pen / MONO_HUES.shape[0]

def compute_hue(M1, M2):
    prims = torch.tensor([[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1]], dtype=torch.float64, device=device)
    exp = torch.tensor([0,60,120,180,240,300], dtype=torch.float64, device=device)
    lab = forward_batch(M1, M2, srgb_to_linear(prims) @ M_SRGB.T)
    h = torch.atan2(lab[:,2], lab[:,1])*(180/3.14159265358979) % 360
    dh = h - exp
    dh = torch.where(dh>180, dh-360, dh)
    dh = torch.where(dh<-180, dh+360, dh)
    return (dh**2).mean().item()

def get_info(M1, M2, pairs):
    with torch.no_grad():
        cv = compute_cv(M1, M2, pairs)
        mono = compute_mono(M1, M2)
        hue = compute_hue(M1, M2)
        # Yellow
        yl = forward_batch(M1, M2, (M_SRGB@torch.tensor([1.,1.,0.],device=device)).unsqueeze(0)).squeeze()
        yL, yC = yl[0].item(), (yl[1]**2+yl[2]**2).sqrt().item()
        # Blue->White
        M1i, M2i = torch.linalg.inv(M1), torch.linalg.inv(M2)
        bx = M_SRGB@srgb_to_linear(torch.tensor([0.,0.,1.],device=device))
        wx = M_SRGB@torch.tensor([1.,1.,1.],device=device)
        bl = forward_batch(M1, M2, bx.unsqueeze(0)).squeeze()
        wl = forward_batch(M1, M2, wx.unsqueeze(0)).squeeze()
        ml = (bl+wl)/2
        mx = inverse_batch(M1i, M2i, ml.unsqueeze(0)).squeeze()
        ms = linear_to_srgb((M_SRGB_INV@mx).clamp(0,1))
        bwgr = (ms[1]/ms[0].clamp(min=1e-10)).item()
        c1, c2 = torch.linalg.cond(M1).item(), torch.linalg.cond(M2).item()
    return {'cv':cv,'mono':mono,'hue':hue,'yL':yL,'yC':yC,'bwgr':bwgr,'c1':c1,'c2':c2}

def print_info(label, info):
    print(f"  {label}: CV={info['cv']*100:.2f}% mono={info['mono']:.4f} hue={info['hue']:.1f} "
          f"yL={info['yL']:.3f} yC={info['yC']:.3f} B->W={info['bwgr']:.2f} cond=({info['c1']:.1f},{info['c2']:.1f})")

def yellow_boundary(M1, M2, hue_deg=85):
    M1i, M2i = torch.linalg.inv(M1), torch.linalg.inv(M2)
    h = torch.tensor(hue_deg*3.14159265358979/180, device=device)
    ch, sh = torch.cos(h), torch.sin(h)
    print(f"    Yellow boundary (h={hue_deg}deg):")
    prev = None
    for Lv in [0.70,0.80,0.85,0.90,0.93,0.95,0.97,0.98,0.99,1.0]:
        L = torch.tensor(Lv, device=device)
        lo, hi2 = torch.tensor(0., device=device), torch.tensor(0.5, device=device)
        for _ in range(40):
            mid = (lo+hi2)/2
            lab = torch.stack([L, mid*ch, mid*sh])
            lc = M2i@lab; lm = torch.sign(lc)*torch.abs(lc).pow(3.)
            lin = M_SRGB_INV@(M1i@lm)
            if (lin>=-0.001).all() and (lin<=1.001).all(): lo=mid
            else: hi2=mid
        arrow = ""
        if prev is not None:
            arrow = " UP" if lo.item()>prev+0.0005 else " DOWN" if lo.item()<prev-0.0005 else " FLAT"
        prev = lo.item()
        print(f"      L={Lv:.2f} C={lo.item():.6f}{arrow}")


def main():
    print(f"\n{'='*60}")
    print("  v26 BLITZ: 4 experiments")
    print(f"{'='*60}\n")

    pairs = build_pairs()
    print(f"Training pairs: {pairs.shape[0]}")

    # Baselines
    print("\n--- Baselines ---")
    v14_info = get_info(V14_M1, V14_M2, pairs)
    ok_info = get_info(OKLAB_M1, OKLAB_M2, pairs)
    print_info("v14", v14_info)
    yellow_boundary(V14_M1, V14_M2)
    print_info("OKLab", ok_info)
    yellow_boundary(OKLAB_M1, OKLAB_M2)

    results = {}

    # ══════════════════════════════════════════════════════════════
    # Exp A: M2 L-row only (2 params)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  Exp A: M2 L-row optimization (2 free params)")
    print(f"{'='*60}")

    v14_m1_np = V14_M1.cpu().numpy()
    v14_m2_np = V14_M2.cpu().numpy()
    s = np.sign(v14_m1_np @ D65_NP) * np.abs(v14_m1_np @ D65_NP)**(1/3)

    # Current L-row
    L_row_0 = v14_m2_np[0].copy()
    # Parameterize: L_row = L_row_0 + delta * perturbation_basis
    # We'll optimize L_row[0] and L_row[1], L_row[2] determined by L(D65)=1
    x0_a = np.array([v14_m2_np[0, 0], v14_m2_np[0, 1]])

    def unpack_a(x2):
        M2 = v14_m2_np.copy()
        M2[0, 0] = x2[0]
        M2[0, 1] = x2[1]
        # L(D65)=1: M2[0] @ s = 1 => M2[0,2] = (1 - M2[0,0]*s[0] - M2[0,1]*s[1]) / s[2]
        M2[0, 2] = (1.0 - M2[0,0]*s[0] - M2[0,1]*s[1]) / s[2]
        return M2

    best_a = {'loss': 999, 'x': x0_a.copy()}
    t0 = time.time()
    ea = [0]

    def obj_a(x):
        M2_np = unpack_a(x)
        M1t, M2t = V14_M1, torch.tensor(M2_np, device=device)
        with torch.no_grad():
            try:
                cv = compute_cv(M1t, M2t, pairs)
                if cv > 0.30: return 50 + cv
                mono = compute_mono(M1t, M2t)
                hue = compute_hue(M1t, M2t)
                # Yellow check
                yl = forward_batch(M1t, M2t, (M_SRGB@torch.tensor([1.,1.,0.],device=device)).unsqueeze(0)).squeeze()
                yC = (yl[1]**2+yl[2]**2).sqrt().item()
                if yC < 0.10: return 80 + (0.10-yC)*100
                loss = cv + 0.3*cv + 2.0*mono + 0.3*hue
            except: return 999
        ea[0] += 1
        if loss < best_a['loss']:
            best_a['loss'] = loss; best_a['x'] = x.copy()
            if ea[0] % 50 == 1:
                print(f"    #{ea[0]:>5d} [{time.time()-t0:5.0f}s] loss={loss:.4f} CV={cv*100:.1f}% mono={mono:.4f}")
        return loss

    opts = cma.CMAOptions()
    opts.set("maxiter", 500); opts.set("popsize", 64); opts.set("verbose", -1)
    es = cma.CMAEvolutionStrategy(x0_a, 0.05, opts)
    while not es.stop():
        sols = es.ask(); fits = [obj_a(x) for x in sols]; es.tell(sols, fits)

    M2_a = unpack_a(best_a['x'])
    M1t_a, M2t_a = V14_M1, torch.tensor(M2_a, device=device)
    results['A'] = get_info(M1t_a, M2t_a, pairs)
    print(f"  Done: {ea[0]} evals in {time.time()-t0:.0f}s")
    print_info("Exp A", results['A'])
    yellow_boundary(M1t_a, M2t_a)

    # ══════════════════════════════════════════════════════════════
    # Exp B: Ab-plane rotation (1 param)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  Exp B: Ab-plane rotation scan (1 param)")
    print(f"{'='*60}")

    best_b = {'loss': 999, 'theta': 0}
    for theta_deg in np.arange(-45, 46, 1):
        theta = theta_deg * np.pi / 180
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Rotate M2 a,b rows
        M2_rot = v14_m2_np.copy()
        M2_rot[1] = cos_t * v14_m2_np[1] - sin_t * v14_m2_np[2]
        M2_rot[2] = sin_t * v14_m2_np[1] + cos_t * v14_m2_np[2]
        M2t_rot = torch.tensor(M2_rot, device=device)
        with torch.no_grad():
            mono = compute_mono(V14_M1, M2t_rot)
            cv = compute_cv(V14_M1, M2t_rot, pairs)
            hue = compute_hue(V14_M1, M2t_rot)
            yl = forward_batch(V14_M1, M2t_rot, (M_SRGB@torch.tensor([1.,1.,0.],device=device)).unsqueeze(0)).squeeze()
            yC = (yl[1]**2+yl[2]**2).sqrt().item()
        loss = cv + 2.0*mono
        if loss < best_b['loss']:
            best_b['loss'] = loss; best_b['theta'] = theta_deg
            print(f"    theta={theta_deg:+3d}deg loss={loss:.4f} CV={cv*100:.1f}% mono={mono:.4f} yC={yC:.3f} hue={hue:.0f}")

    # Apply best rotation
    theta = best_b['theta'] * np.pi / 180
    M2_b = v14_m2_np.copy()
    M2_b[1] = np.cos(theta)*v14_m2_np[1] - np.sin(theta)*v14_m2_np[2]
    M2_b[2] = np.sin(theta)*v14_m2_np[1] + np.cos(theta)*v14_m2_np[2]
    M2t_b = torch.tensor(M2_b, device=device)
    results['B'] = get_info(V14_M1, M2t_b, pairs)
    print_info("Exp B", results['B'])
    yellow_boundary(V14_M1, M2t_b)

    # ══════════════════════════════════════════════════════════════
    # Exp C: OKLab M1 + v14 M2 (direct combine)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  Exp C: OKLab M1 + v14 M2 (direct, no optimization)")
    print(f"{'='*60}")

    # Need to re-normalize v14 M2 L-row for OKLab M1's gray axis
    ok_m1_np = OKLAB_M1.cpu().numpy()
    s_ok = np.sign(ok_m1_np @ D65_NP) * np.abs(ok_m1_np @ D65_NP)**(1/3)
    M2_c = v14_m2_np.copy()
    Lw = M2_c[0] @ s_ok
    M2_c[0] /= Lw
    # a,b rows need to be re-projected onto OKLab's orthonormal basis
    # Actually, just use them as-is and see what happens
    M2t_c = torch.tensor(M2_c, device=device)
    try:
        results['C'] = get_info(OKLAB_M1, M2t_c, pairs)
        print_info("Exp C", results['C'])
        yellow_boundary(OKLAB_M1, M2t_c)
    except Exception as e:
        print(f"  FAILED: {e}")
        results['C'] = None

    # ══════════════════════════════════════════════════════════════
    # Exp D: OKLab basin micro-tune (very small sigma)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  Exp D: OKLab basin micro-tune (sigma=0.001, 500 gen, pop=96)")
    print(f"{'='*60}")

    def ortho_np(s):
        sn = s/np.linalg.norm(s)
        v = np.array([1,0,0.]) if abs(sn[0])<0.9 else np.array([0,1,0.])
        e1 = v - np.dot(v,sn)*sn; e1 /= np.linalg.norm(e1)
        e2 = np.cross(sn,e1)
        return e1, e2

    def unpack_13(x):
        M1 = np.zeros((3,3))
        for i in range(3):
            M1[i,0]=x[2*i]; M1[i,1]=x[2*i+1]
            M1[i,2]=(1.0-M1[i,0]*D65_NP[0]-M1[i,1]*D65_NP[1])/D65_NP[2]
        s = np.sign(M1@D65_NP)*np.abs(M1@D65_NP)**(1/3)
        e1, e2 = ortho_np(s)
        M2 = np.zeros((3,3)); M2[0]=x[6:9]
        Lw = M2[0]@s
        if abs(Lw)<1e-10: return None, None
        M2[0]/=Lw
        M2[1]=x[9]*e1+x[10]*e2; M2[2]=x[11]*e1+x[12]*e2
        return M1, M2

    def pack_13(M1, M2):
        x = np.zeros(13)
        for i in range(3): x[2*i]=M1[i,0]; x[2*i+1]=M1[i,1]
        x[6:9]=M2[0]
        s = np.sign(M1@D65_NP)*np.abs(M1@D65_NP)**(1/3)
        e1, e2 = ortho_np(s)
        x[9]=M2[1]@e1; x[10]=M2[1]@e2; x[11]=M2[2]@e1; x[12]=M2[2]@e2
        return x

    ok_m2_np = OKLAB_M2.cpu().numpy()
    x0_d = pack_13(ok_m1_np, ok_m2_np)

    best_d = {'loss': 999, 'x': x0_d.copy()}
    t0 = time.time()
    ed = [0]

    def obj_d(x):
        M1_np, M2_np = unpack_13(x)
        if M1_np is None: return 999
        M1t, M2t = torch.tensor(M1_np, device=device), torch.tensor(M2_np, device=device)
        with torch.no_grad():
            try:
                c1 = torch.linalg.cond(M1t).item()
                c2 = torch.linalg.cond(M2t).item()
                if c1 > 5 or c2 > 12: return 100 + c1 + c2
                cv = compute_cv(M1t, M2t, pairs)
                if cv > 0.28: return 50 + (cv-0.28)**2*100
                mono = compute_mono(M1t, M2t)
                hue = compute_hue(M1t, M2t)
                yl = forward_batch(M1t, M2t, (M_SRGB@torch.tensor([1.,1.,0.],device=device)).unsqueeze(0)).squeeze()
                yC = (yl[1]**2+yl[2]**2).sqrt().item()
                yL = yl[0].item()
                if yC < 0.15: return 80 + (0.15-yC)*500
                if yL < 0.90: return 80 + (0.90-yL)*500
                # Blue->White
                M1i, M2i = torch.linalg.inv(M1t), torch.linalg.inv(M2t)
                bx = M_SRGB@srgb_to_linear(torch.tensor([0.,0.,1.],device=device))
                wx = M_SRGB@torch.tensor([1.,1.,1.],device=device)
                bl = forward_batch(M1t, M2t, bx.unsqueeze(0)).squeeze()
                wl = forward_batch(M1t, M2t, wx.unsqueeze(0)).squeeze()
                ml = (bl+wl)/2
                mx = inverse_batch(M1i, M2i, ml.unsqueeze(0)).squeeze()
                ms = linear_to_srgb((M_SRGB_INV@mx).clamp(0,1))
                bwgr = (ms[1]/ms[0].clamp(min=1e-10)).item()
                if bwgr < 1.20: return 80 + (1.20-bwgr)*500
                loss = cv + 0.3*cv + 2.0*mono + 0.3*hue
            except: return 999
        ed[0] += 1
        if loss < best_d['loss']:
            best_d['loss'] = loss; best_d['x'] = x.copy()
            if ed[0] % 100 < 2:
                el = time.time()-t0
                print(f"    #{ed[0]:>5d} [{el:5.0f}s] loss={loss:.4f} CV={cv*100:.1f}% mono={mono:.4f} yL={yL:.3f} yC={yC:.3f}")
        return loss

    opts = cma.CMAOptions()
    opts.set("maxiter", 500); opts.set("popsize", 96); opts.set("tolfun", 1e-10); opts.set("verbose", -1)
    es = cma.CMAEvolutionStrategy(x0_d, 0.001, opts)
    while not es.stop():
        sols = es.ask(); fits = [obj_d(x) for x in sols]; es.tell(sols, fits)

    M1_d, M2_d = unpack_13(best_d['x'])
    M1t_d, M2t_d = torch.tensor(M1_d, device=device), torch.tensor(M2_d, device=device)
    results['D'] = get_info(M1t_d, M2t_d, pairs)
    print(f"  Done: {ed[0]} evals in {time.time()-t0:.0f}s")
    print_info("Exp D", results['D'])
    yellow_boundary(M1t_d, M2t_d)

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print_info("v14 (baseline)", v14_info)
    print_info("OKLab (ref)", ok_info)
    for k in ['A','B','C','D']:
        if results.get(k):
            print_info(f"Exp {k}", results[k])

    # Save best
    best_key = min([k for k in results if results[k]], key=lambda k: results[k]['mono'] + results[k]['cv']*2)
    print(f"\n  Best overall: Exp {best_key}")

    if best_key in ['A','B']:
        M1_save = v14_m1_np
        M2_save = M2_a if best_key=='A' else M2_b
    elif best_key == 'C':
        M1_save = ok_m1_np; M2_save = M2_c
    else:
        M1_save = M1_d; M2_save = M2_d

    ckpt = {"version":f"v26-{best_key}",
            "M1":M1_save.tolist(),"M2":M2_save.tolist(),
            "M1_inv":np.linalg.inv(M1_save).tolist(),
            "M2_inv":np.linalg.inv(M2_save).tolist(),
            "metrics":results[best_key]}
    with open("gen_v26.json","w") as f: json.dump(ckpt,f,indent=2)
    print(f"  Saved: gen_v26.json")


if __name__ == "__main__":
    main()
