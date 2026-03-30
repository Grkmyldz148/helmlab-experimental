"""v26 GPU-native: Zero binary search, pure tensor ops.

Cusp scan via dense grid + argmax instead of binary search.
All 360 hues x 200 L points x 200 C points = 14.4M points in one batch.
GPU utilization should be 80%+.
"""

import json, time, numpy as np
import torch
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory//1024**2} MB")

import cma

# Constants
D65 = torch.tensor([0.95047, 1.0, 1.08883], device=device)
D65_NP = np.array([0.95047, 1.0, 1.08883])
M_SRGB = torch.tensor([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]], device=device)
M_SRGB_INV = torch.linalg.inv(M_SRGB)

OKLAB_M1 = torch.tensor([[0.4122214708,0.5363325363,0.0514459929],[0.2119034982,0.6806995451,0.1073969566],[0.0883024619,0.2817188376,0.6299787005]], device=device) @ M_SRGB_INV
OKLAB_M2 = torch.tensor([[0.2104542553,0.7936177850,-0.0040720468],[1.9779984951,-2.4285922050,0.4505937099],[0.0259040371,0.7827717662,-0.8086757660]], device=device)

V14_M1 = torch.tensor([[0.7583761294836658,0.38380162590825084,-0.09608055040602373],[0.12671393631532843,0.8421628149123207,0.03434823621506485],[0.07639223722200054,0.258943526275451,0.6139139663787314]], device=device)
V14_M2 = torch.tensor([[0.10058070589596230,1.01558970993941444,-0.11617041583537688],[2.36157646996164416,-2.44099737506293479,0.07942090510129070],[0.04565327074453784,0.81875488445424471,-0.86440815519878267]], device=device)

def signed_cbrt(x): return torch.sign(x) * torch.abs(x).pow(1./3.)
def srgb_to_linear(c): return torch.where(c<=0.04045, c/12.92, ((c+0.055)/1.055).pow(2.4))
def linear_to_srgb(c): return torch.where(c<=0.0031308, c*12.92, 1.055*c.clamp(min=1e-10).pow(1./2.4)-0.055)

# ── GPU-NATIVE CUSP SCAN ──
# Pre-build grids
N_HUES = 72   # every 5 degrees
N_L = 100
N_C = 100

HUES_RAD = torch.linspace(0, 2*3.14159265358979*(1-1/N_HUES), N_HUES, device=device)
L_GRID = torch.linspace(0.05, 0.999, N_L, device=device)
C_GRID = torch.linspace(0.001, 0.5, N_C, device=device)

def gpu_cusp_scan(M1, M2):
    """Find cusp (max chroma) at each hue using dense grid. Pure GPU, no loops.
    Returns cusp_L (N_HUES,), cusp_C (N_HUES,)
    """
    M1_inv = torch.linalg.inv(M1)
    M2_inv = torch.linalg.inv(M2)

    # Build all (hue, L, C) combinations: (N_HUES, N_L, N_C, 3)
    cos_h = torch.cos(HUES_RAD)  # (H,)
    sin_h = torch.sin(HUES_RAD)  # (H,)

    # For each hue and L, find max C that's in gamut
    # Strategy: build Lab grid, inverse to sRGB, check gamut
    cusp_L = torch.zeros(N_HUES, device=device)
    cusp_C = torch.zeros(N_HUES, device=device)

    # Process in hue batches to manage memory
    BATCH = 12
    for h_start in range(0, N_HUES, BATCH):
        h_end = min(h_start + BATCH, N_HUES)
        n_h = h_end - h_start

        # Lab points: (n_h, N_L, N_C, 3)
        L_exp = L_GRID.view(1, N_L, 1).expand(n_h, N_L, N_C)
        C_exp = C_GRID.view(1, 1, N_C).expand(n_h, N_L, N_C)
        ch = cos_h[h_start:h_end].view(n_h, 1, 1)
        sh = sin_h[h_start:h_end].view(n_h, 1, 1)

        a = C_exp * ch
        b = C_exp * sh
        lab = torch.stack([L_exp, a, b], dim=-1)  # (n_h, N_L, N_C, 3)

        # Inverse: lab -> xyz -> linear sRGB
        flat = lab.reshape(-1, 3)  # (n_h*N_L*N_C, 3)
        lms_c = flat @ M2_inv.T
        lms = torch.sign(lms_c) * torch.abs(lms_c).pow(3.)
        xyz = lms @ M1_inv.T
        lin = xyz @ M_SRGB_INV.T  # (n_h*N_L*N_C, 3)

        # Gamut check
        in_gamut = (lin >= -0.002).all(dim=1) & (lin <= 1.002).all(dim=1)
        in_gamut = in_gamut.reshape(n_h, N_L, N_C)  # (n_h, N_L, N_C)

        # Max C in gamut at each (hue, L): find highest C index that's in gamut
        # Mask out-of-gamut with 0 chroma
        c_vals = C_GRID.view(1, 1, N_C).expand(n_h, N_L, N_C)
        masked_c = torch.where(in_gamut, c_vals, torch.zeros_like(c_vals))
        max_c_per_L, _ = masked_c.max(dim=2)  # (n_h, N_L)

        # Cusp = L with highest max_c
        cusp_idx = max_c_per_L.argmax(dim=1)  # (n_h,)
        for i in range(n_h):
            ci = cusp_idx[i].item()
            cusp_L[h_start + i] = L_GRID[ci]
            cusp_C[h_start + i] = max_c_per_L[i, ci]

    return cusp_L, cusp_C


def gpu_mono_penalty(M1, M2, yellow_hues_deg=range(70, 101, 5)):
    """Monotonicity penalty using dense grid scan. Pure GPU."""
    M1_inv, M2_inv = torch.linalg.inv(M1), torch.linalg.inv(M2)
    L_fine = torch.linspace(0.80, 0.999, 80, device=device)
    pen = 0.0

    hues_rad = torch.tensor([h*3.14159265358979/180 for h in yellow_hues_deg], device=device)
    n_h = len(hues_rad)

    # Build grid: (n_h, 80, N_C, 3)
    N_C_fine = 80
    C_fine = torch.linspace(0.001, 0.4, N_C_fine, device=device)

    ch = torch.cos(hues_rad).view(n_h, 1, 1)
    sh = torch.sin(hues_rad).view(n_h, 1, 1)
    L_exp = L_fine.view(1, 80, 1).expand(n_h, 80, N_C_fine)
    C_exp = C_fine.view(1, 1, N_C_fine).expand(n_h, 80, N_C_fine)

    lab = torch.stack([L_exp, C_exp*ch, C_exp*sh], dim=-1)
    flat = lab.reshape(-1, 3)
    lms_c = flat @ M2_inv.T
    lms = torch.sign(lms_c) * torch.abs(lms_c).pow(3.)
    lin = (lms @ M1_inv.T) @ M_SRGB_INV.T
    in_gamut = ((lin >= -0.002).all(dim=1) & (lin <= 1.002).all(dim=1)).reshape(n_h, 80, N_C_fine)

    c_vals = C_fine.view(1, 1, N_C_fine).expand(n_h, 80, N_C_fine)
    max_c, _ = torch.where(in_gamut, c_vals, torch.zeros_like(c_vals)).max(dim=2)  # (n_h, 80)

    # Per hue: check monotonicity after cusp
    for hi in range(n_h):
        mc = max_c[hi]  # (80,)
        ci = mc.argmax().item()
        cL = L_fine[ci].item()
        if cL > 0.975: pen += (cL - 0.975)**2 * 100
        diffs = mc[1:] - mc[:-1]
        pos = diffs[diffs > 0.002]
        if pos.numel() > 0:
            pen += pos.pow(2).sum().item() * 100

    return pen / n_h


def gpu_gradient_cv(M1, M2, pairs):
    """Gradient CV - same as before but kept for compatibility."""
    M1i, M2i = torch.linalg.inv(M1), torch.linalg.inv(M2)
    N = pairs.shape[0]
    N_STEPS = 25
    T = torch.linspace(0, 1, N_STEPS+1, device=device)

    lab1 = signed_cbrt(pairs[:,0] @ M1.T) @ M2.T
    lab2 = signed_cbrt(pairs[:,1] @ M1.T) @ M2.T
    t = T.view(1,-1,1)
    labs = lab1.unsqueeze(1) + t*(lab2-lab1).unsqueeze(1)
    lf = labs.reshape(-1,3)
    lc = lf @ M2i.T; lm = torch.sign(lc)*torch.abs(lc).pow(3.)
    lin = (lm @ M1i.T) @ M_SRGB_INV.T
    s8 = (linear_to_srgb(lin.clamp(0,1))*255).round()/255.
    xb = srgb_to_linear(s8) @ M_SRGB.T
    r = xb.clamp(min=1e-10) / D65
    f = torch.where(r>0.008856, r.pow(1./3.), 7.787*r+16./116.)
    cl = torch.stack([116*f[...,1]-16, 500*(f[...,0]-f[...,1]), 200*(f[...,1]-f[...,2])], dim=-1)
    cl = cl.reshape(N, N_STEPS+1, 3)
    c1, c2 = cl[:,:-1], cl[:,1:]
    dL=c2[...,0]-c1[...,0]; C1=(c1[...,1]**2+c1[...,2]**2).sqrt(); C2=(c2[...,1]**2+c2[...,2]**2).sqrt()
    dC=C2-C1; dH=((c2[...,1]-c1[...,1])**2+(c2[...,2]-c1[...,2])**2-dC**2).clamp(min=0).sqrt()
    SL=1+0.015*(c1[...,0]-50)**2/(20+(c1[...,0]-50)**2).sqrt(); SC=1+0.045*C1; SH=1+0.015*C1
    de=((dL/SL)**2+(dC/SC)**2+(dH/SH)**2).sqrt()
    md=de.mean(1); sd=de.std(1); v=md>0.001
    cvs=torch.where(v, sd/md, torch.zeros_like(md))
    return cvs[v].mean().item() if v.any() else 999.


def get_full_info(M1, M2, pairs):
    with torch.no_grad():
        cv = gpu_gradient_cv(M1, M2, pairs)
        mono = gpu_mono_penalty(M1, M2)
        cusp_L, cusp_C = gpu_cusp_scan(M1, M2)

        # Yellow hue index (85 deg ~ index 85/5 = 17)
        yellow_idx = 17  # 85 degrees at 5-degree steps
        yL_cusp = cusp_L[yellow_idx].item()
        yC_cusp = cusp_C[yellow_idx].item()

        # Yellow primary
        yl = (signed_cbrt((M_SRGB@torch.tensor([1.,1.,0.],device=device)) @ M1.T)) @ M2.T
        yL_prim = yl[0].item()
        yC_prim = (yl[1]**2+yl[2]**2).sqrt().item()

        # Blue->White
        M1i, M2i = torch.linalg.inv(M1), torch.linalg.inv(M2)
        bx = M_SRGB@srgb_to_linear(torch.tensor([0.,0.,1.],device=device))
        wx = M_SRGB@torch.tensor([1.,1.,1.],device=device)
        bl = (signed_cbrt(bx@M1.T))@M2.T; wl = (signed_cbrt(wx@M1.T))@M2.T
        ml = (bl+wl)/2; lc=ml@M2i.T; lm=torch.sign(lc)*torch.abs(lc).pow(3.); mx=lm@M1i.T
        ms = linear_to_srgb((M_SRGB_INV@mx).clamp(0,1))
        bwgr = (ms[1]/ms[0].clamp(min=1e-10)).item()

        c1, c2 = torch.linalg.cond(M1).item(), torch.linalg.cond(M2).item()

    return {'cv':cv, 'mono':mono, 'yL_cusp':yL_cusp, 'yC_cusp':yC_cusp,
            'yL_prim':yL_prim, 'yC_prim':yC_prim, 'bwgr':bwgr, 'c1':c1, 'c2':c2,
            'cusp_L':cusp_L.cpu().numpy(), 'cusp_C':cusp_C.cpu().numpy()}

def print_info(label, info):
    print(f"  {label}: CV={info['cv']*100:.2f}% mono={info['mono']:.4f} "
          f"yL_cusp={info['yL_cusp']:.3f} yC_cusp={info['yC_cusp']:.3f} "
          f"yL_prim={info['yL_prim']:.3f} yC_prim={info['yC_prim']:.3f} "
          f"B->W={info['bwgr']:.2f} cond=({info['c1']:.1f},{info['c2']:.1f})")


def main():
    print(f"\n{'='*60}")
    print("  v26 GPU-NATIVE: Dense grid scan, zero binary search")
    print(f"{'='*60}\n")

    # Warmup GPU
    t0 = time.time()
    _ = torch.randn(1000, 1000, device=device) @ torch.randn(1000, 1000, device=device)
    torch.cuda.synchronize()
    print(f"GPU warmup: {time.time()-t0:.2f}s")

    pairs = []
    prims = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[0,0,0]]
    for i in range(len(prims)):
        for j in range(i+1, len(prims)):
            pairs.append((prims[i], prims[j]))
    for g1 in [0.0,0.2,0.4,0.6,0.8,1.0]:
        for g2 in [g1+0.2,g1+0.4]:
            if g2<=1.0: pairs.append(([g1]*3,[g2]*3))
    rng = np.random.RandomState(42)
    for _ in range(80): pairs.append((rng.rand(3).tolist(), rng.rand(3).tolist()))
    pt = torch.zeros(len(pairs), 2, 3, device=device)
    for i,(c1,c2) in enumerate(pairs):
        pt[i,0] = M_SRGB@srgb_to_linear(torch.tensor(c1,device=device))
        pt[i,1] = M_SRGB@srgb_to_linear(torch.tensor(c2,device=device))
    print(f"Training pairs: {pt.shape[0]}")

    # Benchmark cusp scan speed
    t0 = time.time()
    cL, cC = gpu_cusp_scan(V14_M1, V14_M2)
    torch.cuda.synchronize()
    print(f"Cusp scan (72 hues x 100L x 100C = 720K points): {time.time()-t0:.2f}s")

    t0 = time.time()
    mono = gpu_mono_penalty(V14_M1, V14_M2)
    torch.cuda.synchronize()
    print(f"Mono penalty: {time.time()-t0:.2f}s (value={mono:.4f})")

    # Baselines
    print("\n--- Baselines ---")
    t0 = time.time()
    v14_info = get_full_info(V14_M1, V14_M2, pt)
    print(f"  v14 computed in {time.time()-t0:.1f}s")
    print_info("v14", v14_info)

    t0 = time.time()
    ok_info = get_full_info(OKLAB_M1, OKLAB_M2, pt)
    print(f"  OKLab computed in {time.time()-t0:.1f}s")
    print_info("OKLab", ok_info)

    # OKLab cusp L at yellow for reference
    ok_yellow_cusp = ok_info['yL_cusp']
    print(f"\n  OKLab yellow cusp L = {ok_yellow_cusp:.3f}")
    print(f"  v14 yellow cusp L = {v14_info['yL_cusp']:.3f}")

    # ── Rotation sweep (91 angles, should be FAST now) ──
    print(f"\n{'='*60}")
    print("  ROTATION SWEEP: -45 to +45 deg")
    print(f"{'='*60}")
    t0 = time.time()
    v14_m2_np = V14_M2.cpu().numpy()

    best_rot = {'loss': 999, 'theta': 0, 'mono': 999}
    for theta_deg in range(-45, 46, 1):
        theta = theta_deg * np.pi / 180
        ct, st = np.cos(theta), np.sin(theta)
        M2_r = v14_m2_np.copy()
        M2_r[1] = ct*v14_m2_np[1] - st*v14_m2_np[2]
        M2_r[2] = st*v14_m2_np[1] + ct*v14_m2_np[2]
        M2t = torch.tensor(M2_r, device=device)
        with torch.no_grad():
            mono = gpu_mono_penalty(V14_M1, M2t)
            cv = gpu_gradient_cv(V14_M1, M2t, pt)
        loss = cv + 2*mono
        if mono < best_rot['mono']:
            best_rot = {'loss':loss, 'theta':theta_deg, 'mono':mono, 'cv':cv}
            print(f"  theta={theta_deg:+3d} mono={mono:.4f} CV={cv*100:.1f}%")

    print(f"  Sweep done in {time.time()-t0:.1f}s")
    print(f"  Best rotation: theta={best_rot['theta']} mono={best_rot['mono']:.4f}")

    # Apply best rotation and get full info
    theta = best_rot['theta'] * np.pi / 180
    M2_best_rot = v14_m2_np.copy()
    M2_best_rot[1] = np.cos(theta)*v14_m2_np[1] - np.sin(theta)*v14_m2_np[2]
    M2_best_rot[2] = np.sin(theta)*v14_m2_np[1] + np.cos(theta)*v14_m2_np[2]
    M2t_rot = torch.tensor(M2_best_rot, device=device)
    rot_info = get_full_info(V14_M1, M2t_rot, pt)
    print_info(f"Rotated ({best_rot['theta']}deg)", rot_info)

    # ── OKLab M1 + CMA-ES M2, micro sigma, high pop ──
    print(f"\n{'='*60}")
    print("  CMA-ES: OKLab M1 fixed, optimize M2 (pop=128, gen=300)")
    print(f"{'='*60}")

    ok_m1_np = OKLAB_M1.cpu().numpy()
    ok_m2_np = OKLAB_M2.cpu().numpy()
    s = np.sign(ok_m1_np @ D65_NP) * np.abs(ok_m1_np @ D65_NP)**(1/3)
    sn = s/np.linalg.norm(s)
    v = np.array([1,0,0.]) if abs(sn[0])<0.9 else np.array([0,1,0.])
    e1 = v - np.dot(v,sn)*sn; e1 /= np.linalg.norm(e1)
    e2 = np.cross(sn,e1)

    def unpack_m2(x4):
        M2 = np.zeros((3,3))
        M2[0] = ok_m2_np[0].copy()
        Lw = M2[0]@s
        M2[0] /= Lw
        M2[1] = x4[0]*e1 + x4[1]*e2
        M2[2] = x4[2]*e1 + x4[3]*e2
        return M2

    x0 = np.array([ok_m2_np[1]@e1, ok_m2_np[1]@e2, ok_m2_np[2]@e1, ok_m2_np[2]@e2])

    best_cma = {'loss':999, 'x':x0.copy()}
    evals = [0]
    t0 = time.time()

    def obj_cma(x):
        M2_np = unpack_m2(x)
        M2t = torch.tensor(M2_np, device=device)
        with torch.no_grad():
            cv = gpu_gradient_cv(OKLAB_M1, M2t, pt)
            mono = gpu_mono_penalty(OKLAB_M1, M2t)
            # Yellow check
            yl = (signed_cbrt((M_SRGB@torch.tensor([1.,1.,0.],device=device))@OKLAB_M1.T))@M2t.T
            yC = (yl[1]**2+yl[2]**2).sqrt().item()
            if yC < 0.10: return 80 + (0.10-yC)*500
            loss = cv + 0.3*cv + 2.0*mono
        evals[0] += 1
        if loss < best_cma['loss']:
            best_cma['loss'] = loss; best_cma['x'] = x.copy()
            if evals[0] % 200 < 2:
                print(f"    #{evals[0]:>5d} [{time.time()-t0:5.0f}s] loss={loss:.4f} CV={cv*100:.1f}% mono={mono:.4f} yC={yC:.3f}")
        return loss

    opts = cma.CMAOptions()
    opts.set("maxiter", 300); opts.set("popsize", 128); opts.set("tolfun", 1e-10); opts.set("verbose", -1)
    es = cma.CMAEvolutionStrategy(x0, 0.1, opts)
    while not es.stop():
        sols = es.ask(); fits = [obj_cma(x) for x in sols]; es.tell(sols, fits)

    M2_cma = unpack_m2(best_cma['x'])
    M2t_cma = torch.tensor(M2_cma, device=device)
    cma_info = get_full_info(OKLAB_M1, M2t_cma, pt)
    print(f"\n  CMA-ES done: {evals[0]} evals in {time.time()-t0:.0f}s")
    print_info("CMA-ES best", cma_info)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print_info("v14 (current)", v14_info)
    print_info("OKLab (ref)", ok_info)
    print_info(f"Rotation ({best_rot['theta']}deg)", rot_info)
    print_info("CMA-ES (OKLab M1)", cma_info)

    # Save best
    results = {'rotation': rot_info, 'cma': cma_info}
    best_key = min(results, key=lambda k: results[k]['mono'])
    print(f"\n  Best mono: {best_key}")

    if best_key == 'rotation':
        M1s, M2s = V14_M1.cpu().numpy(), M2_best_rot
    else:
        M1s, M2s = ok_m1_np, M2_cma

    ckpt = {"version":"v26-native","strategy":best_key,
            "M1":M1s.tolist(),"M2":M2s.tolist(),
            "M1_inv":np.linalg.inv(M1s).tolist(),"M2_inv":np.linalg.inv(M2s).tolist(),
            "metrics":{k:v for k,v in results[best_key].items() if not isinstance(v, np.ndarray)}}
    with open("gen_v26_native.json","w") as f: json.dump(ckpt,f,indent=2)
    print(f"  Saved: gen_v26_native.json")


if __name__ == "__main__":
    main()
