"""v25: Start from OKLab's M1, optimize M2 + small M1 perturbation.

OKLab's M1 has correct cusp behavior. We keep it (or perturb slightly)
and optimize M2 for better gradient CV.

Two stages:
  Stage 1: M2 only (4 params), M1 = OKLab's M1
  Stage 2: M1 perturbation + M2 (13 params), starting from Stage 1 best
"""

import json
import time
import sys
import numpy as np

import torch
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

import cma

# Constants
D65 = torch.tensor([0.95047, 1.0, 1.08883], device=device)
D65_NP = np.array([0.95047, 1.0, 1.08883])

M_SRGB = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], device=device)
M_SRGB_INV = torch.linalg.inv(M_SRGB)

# OKLab M1 (in XYZ space)
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


# Utilities
def signed_cbrt(x):
    return torch.sign(x) * torch.abs(x).pow(1.0/3.0)

def srgb_to_linear(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))

def linear_to_srgb(c):
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c.clamp(min=1e-10).pow(1.0/2.4) - 0.055)

def xyz_to_cielab_batch(xyz):
    r = xyz / D65
    mask = r > 0.008856
    f = torch.where(mask, r.pow(1.0/3.0), 7.787 * r + 16.0/116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)

def forward_batch(M1, M2, xyz):
    return (signed_cbrt(xyz @ M1.T)) @ M2.T

def inverse_batch(M1_inv, M2_inv, lab):
    lms_c = lab @ M2_inv.T
    return (torch.sign(lms_c) * torch.abs(lms_c).pow(3.0)) @ M1_inv.T


# Training pairs
def build_training_pairs():
    pairs = []
    prims = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[0,0,0]]
    for i in range(len(prims)):
        for j in range(i+1, len(prims)):
            pairs.append((prims[i], prims[j]))
    for g1 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for g2 in [g1+0.2, g1+0.4]:
            if g2 <= 1.0: pairs.append(([g1]*3, [g2]*3))
    rng = np.random.RandomState(42)
    for _ in range(80):
        pairs.append((rng.rand(3).tolist(), rng.rand(3).tolist()))
    pt = torch.zeros(len(pairs), 2, 3, device=device)
    for i, (c1, c2) in enumerate(pairs):
        pt[i, 0] = M_SRGB @ srgb_to_linear(torch.tensor(c1, device=device))
        pt[i, 1] = M_SRGB @ srgb_to_linear(torch.tensor(c2, device=device))
    return pt

N_STEPS = 25
T_STEPS = torch.linspace(0, 1, N_STEPS + 1, device=device)

def compute_cv_gpu(M1, M2, pairs):
    M1_inv, M2_inv = torch.linalg.inv(M1), torch.linalg.inv(M2)
    N = pairs.shape[0]
    lab1 = forward_batch(M1, M2, pairs[:, 0])
    lab2 = forward_batch(M1, M2, pairs[:, 1])
    t = T_STEPS.view(1, -1, 1)
    labs = lab1.unsqueeze(1) + t * (lab2 - lab1).unsqueeze(1)
    labs_flat = labs.reshape(-1, 3)
    xyz_flat = inverse_batch(M1_inv, M2_inv, labs_flat)
    lin_flat = (xyz_flat @ M_SRGB_INV.T).clamp(0, 1)
    srgb8 = (linear_to_srgb(lin_flat) * 255).round() / 255.0
    xyz_back = srgb_to_linear(srgb8) @ M_SRGB.T
    cielab = xyz_to_cielab_batch(xyz_back.clamp(min=1e-10)).reshape(N, N_STEPS+1, 3)
    cl1, cl2 = cielab[:, :-1], cielab[:, 1:]
    dL = cl2[...,0]-cl1[...,0]
    C1 = (cl1[...,1]**2+cl1[...,2]**2).sqrt()
    C2 = (cl2[...,1]**2+cl2[...,2]**2).sqrt()
    dC = C2-C1
    dH = ((cl2[...,1]-cl1[...,1])**2+(cl2[...,2]-cl1[...,2])**2-dC**2).clamp(min=0).sqrt()
    SL = 1+0.015*(cl1[...,0]-50)**2/(20+(cl1[...,0]-50)**2).sqrt()
    SC, SH = 1+0.045*C1, 1+0.015*C1
    de = ((dL/SL)**2+(dC/SC)**2+(dH/SH)**2).sqrt()
    mean_de = de.mean(dim=1)
    std_de = de.std(dim=1)
    valid = mean_de > 0.001
    cvs = torch.where(valid, std_de/mean_de, torch.zeros_like(mean_de))
    mean_cv = cvs[valid].mean() if valid.any() else torch.tensor(999.0, device=device)
    top10 = cvs[valid].topk(max(1, valid.sum().item()//10)).values.mean() if valid.any() else mean_cv
    return mean_cv, top10

MONO_HUES = torch.arange(60, 121, 5, device=device, dtype=torch.float64) * (3.14159265358979/180.0)
MONO_LS = torch.arange(0.80, 1.001, 0.005, device=device)

def compute_mono_gpu(M1, M2):
    M1_inv, M2_inv = torch.linalg.inv(M1), torch.linalg.inv(M2)
    pen = torch.tensor(0.0, device=device)
    for hi in range(MONO_HUES.shape[0]):
        h = MONO_HUES[hi]
        ch, sh = torch.cos(h), torch.sin(h)
        lo = torch.zeros(MONO_LS.shape[0], device=device)
        hi_c = torch.full_like(lo, 0.5)
        for _ in range(35):
            mid = (lo + hi_c) / 2
            lab = torch.stack([MONO_LS, mid*ch, mid*sh], dim=1)
            lms_c = lab @ M2_inv.T
            lms = torch.sign(lms_c) * torch.abs(lms_c).pow(3.0)
            lin = (lms @ M1_inv.T) @ M_SRGB_INV.T
            ok = (lin >= -0.001).all(dim=1) & (lin <= 1.001).all(dim=1)
            lo = torch.where(ok, mid, lo)
            hi_c = torch.where(ok, hi_c, mid)
        ci = lo.argmax()
        cL = MONO_LS[ci]
        if cL > 0.975: pen = pen + (cL - 0.975)**2 * 100
        diffs = lo[1:] - lo[:-1]
        pos = diffs[diffs > 1e-5]
        if pos.numel() > 0: pen = pen + (pos/0.005).pow(2).sum()
    return pen / MONO_HUES.shape[0]

def compute_hue_gpu(M1, M2):
    prims = torch.tensor([[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1]], dtype=torch.float64, device=device)
    expected = torch.tensor([0,60,120,180,240,300], dtype=torch.float64, device=device)
    lab = forward_batch(M1, M2, srgb_to_linear(prims) @ M_SRGB.T)
    h = torch.atan2(lab[:,2], lab[:,1]) * (180.0/3.14159265358979) % 360
    dh = h - expected
    dh = torch.where(dh > 180, dh-360, dh)
    dh = torch.where(dh < -180, dh+360, dh)
    return (dh**2).mean()

def compute_constraints(M1, M2):
    M1_inv, M2_inv = torch.linalg.inv(M1), torch.linalg.inv(M2)
    yl = forward_batch(M1, M2, (M_SRGB @ torch.tensor([1.,1.,0.], device=device)).unsqueeze(0)).squeeze()
    yL, yC = yl[0].item(), (yl[1]**2+yl[2]**2).sqrt().item()
    bxyz = M_SRGB @ srgb_to_linear(torch.tensor([0.,0.,1.], device=device))
    wxyz = M_SRGB @ torch.tensor([1.,1.,1.], device=device)
    blab = forward_batch(M1, M2, bxyz.unsqueeze(0)).squeeze()
    wlab = forward_batch(M1, M2, wxyz.unsqueeze(0)).squeeze()
    mlab = (blab+wlab)/2
    mxyz = inverse_batch(M1_inv, M2_inv, mlab.unsqueeze(0)).squeeze()
    ms = linear_to_srgb((M_SRGB_INV @ mxyz).clamp(0,1))
    bwgr = (ms[1]/ms[0].clamp(min=1e-10)).item()
    ps = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]], dtype=torch.float64, device=device)
    plab = forward_batch(M1, M2, srgb_to_linear(ps) @ M_SRGB.T)
    plr = (plab[:,0].max() - plab[:,0].min()).item()
    return {'yL':yL,'yC':yC,'bwgr':bwgr,'plr':plr,
            'c1':torch.linalg.cond(M1).item(),'c2':torch.linalg.cond(M2).item()}


# Parameterization
def ortho_np(s):
    sn = s/np.linalg.norm(s)
    v = np.array([1,0,0.]) if abs(sn[0])<0.9 else np.array([0,1,0.])
    e1 = v - np.dot(v,sn)*sn; e1 /= np.linalg.norm(e1)
    e2 = np.cross(sn,e1)
    return e1, e2

def m2_params_to_matrix(x4, M1_np):
    """4 params -> M2 (M1 fixed). x=[a1,a2,b1,b2]. L-row = OKLab's."""
    s = np.sign(M1_np @ D65_NP) * np.abs(M1_np @ D65_NP)**(1/3)
    e1, e2 = ortho_np(s)
    M2 = np.zeros((3,3))
    # L-row: normalized so L(D65)=1
    M2[0] = OKLAB_M2.cpu().numpy()[0]  # start from OKLab L-row
    Lw = M2[0] @ s
    if abs(Lw) < 1e-10: return None
    M2[0] /= Lw
    M2[1] = x4[0]*e1 + x4[1]*e2
    M2[2] = x4[2]*e1 + x4[3]*e2
    return M2

def m2_matrix_to_params(M2, M1_np):
    s = np.sign(M1_np @ D65_NP) * np.abs(M1_np @ D65_NP)**(1/3)
    e1, e2 = ortho_np(s)
    return np.array([M2[1]@e1, M2[1]@e2, M2[2]@e1, M2[2]@e2])

def full_params_to_matrices(x13):
    """13 params -> D65-normalized M1 + achromatic M2."""
    M1 = np.zeros((3,3))
    for i in range(3):
        M1[i,0] = x13[2*i]; M1[i,1] = x13[2*i+1]
        M1[i,2] = (1.0 - M1[i,0]*D65_NP[0] - M1[i,1]*D65_NP[1]) / D65_NP[2]
    s = np.sign(M1 @ D65_NP) * np.abs(M1 @ D65_NP)**(1/3)
    e1, e2 = ortho_np(s)
    M2 = np.zeros((3,3))
    M2[0] = x13[6:9]
    Lw = M2[0] @ s
    if abs(Lw) < 1e-10: return None, None
    M2[0] /= Lw
    M2[1] = x13[9]*e1 + x13[10]*e2
    M2[2] = x13[11]*e1 + x13[12]*e2
    return M1, M2

def full_matrices_to_params(M1, M2):
    x = np.zeros(13)
    for i in range(3): x[2*i]=M1[i,0]; x[2*i+1]=M1[i,1]
    x[6:9] = M2[0]
    s = np.sign(M1 @ D65_NP) * np.abs(M1 @ D65_NP)**(1/3)
    e1, e2 = ortho_np(s)
    x[9]=M2[1]@e1; x[10]=M2[1]@e2; x[11]=M2[2]@e1; x[12]=M2[2]@e2
    return x


def run_stage(name, x0, unpack_fn, sigma, gens, popsize, pairs):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  params={len(x0)} sigma={sigma} gens={gens} pop={popsize}")
    print(f"{'='*60}")

    best_loss = float("inf")
    best_x = x0.copy()
    evals = [0]
    t0 = time.time()

    def objective(x):
        try:
            result = unpack_fn(x)
            if result is None: return 999.0
            if isinstance(result, tuple):
                M1_np, M2_np = result
                if M1_np is None: return 999.0
            else:
                M1_np = OKLAB_M1.cpu().numpy()
                M2_np = result
                if M2_np is None: return 999.0

            M1 = torch.tensor(M1_np, device=device)
            M2 = torch.tensor(M2_np, device=device)

            with torch.no_grad():
                c = compute_constraints(M1, M2)
                # Hard constraints
                viol = 0
                if c['yC'] < 0.12: viol += (0.12-c['yC'])**2*1000
                if c['yL'] < 0.90: viol += (0.90-c['yL'])**2*1000
                if c['bwgr'] < 1.20: viol += (1.20-c['bwgr'])**2*1000
                if c['c1'] > 5: viol += (c['c1']-5)**2*10
                if c['c2'] > 12: viol += (c['c2']-12)**2*10
                if c['plr'] < 0.40: viol += (0.40-c['plr'])**2*1000
                if viol > 0: return 100 + viol

                cv, top10 = compute_cv_gpu(M1, M2, pairs)
                cv_v = cv.item()
                if cv_v > 0.30: return 50 + (cv_v-0.30)**2*100

                mono = compute_mono_gpu(M1, M2).item()
                hue = compute_hue_gpu(M1, M2).item()
                loss = cv_v + 0.3*top10.item() + 2.0*mono + 0.3*hue

        except: return 999.0

        evals[0] += 1
        nonlocal best_loss, best_x
        if loss < best_loss:
            best_loss = loss
            best_x = x.copy()
            el = time.time()-t0
            print(f"  #{evals[0]:>5d} [{el:5.0f}s] loss={loss:.4f} CV={cv_v*100:.1f}% "
                  f"mono={mono:.4f} hue={hue:.1f} yL={c['yL']:.3f} yC={c['yC']:.3f} "
                  f"B->W={c['bwgr']:.2f} cond=({c['c1']:.1f},{c['c2']:.1f})")
        return loss

    opts = cma.CMAOptions()
    opts.set("maxiter", gens)
    opts.set("popsize", popsize)
    opts.set("tolfun", 1e-9)
    opts.set("verbose", -1)
    es = cma.CMAEvolutionStrategy(x0, sigma, opts)
    while not es.stop():
        sols = es.ask()
        fits = [objective(x) for x in sols]
        es.tell(sols, fits)

    print(f"\n  Done: {evals[0]} evals in {time.time()-t0:.0f}s")
    return best_x, best_loss


def main():
    print("=== v25: Start from OKLab M1, optimize M2 ===\n")
    pairs = build_training_pairs()
    print(f"Training pairs: {pairs.shape[0]}")

    # Baselines
    with torch.no_grad():
        v14_cv, _ = compute_cv_gpu(V14_M1, V14_M2, pairs)
        ok_cv, _ = compute_cv_gpu(OKLAB_M1, OKLAB_M2, pairs)
        v14_mono = compute_mono_gpu(V14_M1, V14_M2)
        ok_mono = compute_mono_gpu(OKLAB_M1, OKLAB_M2)
        v14_c = compute_constraints(V14_M1, V14_M2)
        ok_c = compute_constraints(OKLAB_M1, OKLAB_M2)
    print(f"v14:   CV={v14_cv.item()*100:.2f}% mono={v14_mono.item():.4f} yL={v14_c['yL']:.3f} yC={v14_c['yC']:.3f} B->W={v14_c['bwgr']:.2f}")
    print(f"OKLab: CV={ok_cv.item()*100:.2f}% mono={ok_mono.item():.4f} yL={ok_c['yL']:.3f} yC={ok_c['yC']:.3f} B->W={ok_c['bwgr']:.2f}")

    # ── Stage 1: M2 only, M1 = OKLab ──
    ok_m1_np = OKLAB_M1.cpu().numpy()
    ok_m2_np = OKLAB_M2.cpu().numpy()
    x0_s1 = m2_matrix_to_params(ok_m2_np, ok_m1_np)

    best_s1, _ = run_stage(
        "Stage 1: M2 only (OKLab M1 fixed)", x0_s1,
        lambda x: m2_params_to_matrix(x, ok_m1_np),
        sigma=0.1, gens=200, popsize=48, pairs=pairs
    )

    # Get stage 1 result
    M2_s1 = m2_params_to_matrix(best_s1, ok_m1_np)
    M1_s1 = ok_m1_np

    # ── Stage 2: Small M1 perturbation + M2 ──
    x0_s2 = full_matrices_to_params(M1_s1, M2_s1)

    best_s2, _ = run_stage(
        "Stage 2: M1 perturbation + M2 (from Stage 1)", x0_s2,
        full_params_to_matrices,
        sigma=0.005, gens=300, popsize=48, pairs=pairs
    )

    # Final result
    M1_final, M2_final = full_params_to_matrices(best_s2)
    M1_inv = np.linalg.inv(M1_final)
    M2_inv = np.linalg.inv(M2_final)

    M1t = torch.tensor(M1_final, device=device)
    M2t = torch.tensor(M2_final, device=device)
    with torch.no_grad():
        fcv, _ = compute_cv_gpu(M1t, M2t, pairs)
        fmono = compute_mono_gpu(M1t, M2t)
        fhue = compute_hue_gpu(M1t, M2t)
        fc = compute_constraints(M1t, M2t)

    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"CV={fcv.item()*100:.2f}% mono={fmono.item():.4f} hue={fhue.item():.1f}")
    print(f"yL={fc['yL']:.3f} yC={fc['yC']:.3f} B->W={fc['bwgr']:.2f} "
          f"Lrange={fc['plr']:.3f} cond=({fc['c1']:.1f},{fc['c2']:.1f})")

    print("\nM1 =")
    for r in M1_final: print(f"  [{r[0]:>22.16f}, {r[1]:>22.16f}, {r[2]:>22.16f}],")
    print("M2 =")
    for r in M2_final: print(f"  [{r[0]:>22.16f}, {r[1]:>22.16f}, {r[2]:>22.16f}],")
    print("M1_inv =")
    for r in M1_inv: print(f"  [{r[0]:>22.16f}, {r[1]:>22.16f}, {r[2]:>22.16f}],")
    print("M2_inv =")
    for r in M2_inv: print(f"  [{r[0]:>22.16f}, {r[1]:>22.16f}, {r[2]:>22.16f}],")

    # Yellow boundary
    M2it = torch.linalg.inv(M2t); M1it = torch.linalg.inv(M1t)
    h = torch.tensor(85.0*3.14159265358979/180.0, device=device)
    ch, sh = torch.cos(h), torch.sin(h)
    print(f"\nYellow boundary (hue 85deg):")
    for Lv in [0.70,0.75,0.80,0.85,0.90,0.93,0.95,0.97,0.98,0.99,0.995,1.0]:
        L = torch.tensor(Lv, device=device)
        lo, hi2 = torch.tensor(0., device=device), torch.tensor(0.5, device=device)
        for _ in range(40):
            mid = (lo+hi2)/2
            lab = torch.stack([L, mid*ch, mid*sh])
            lc = M2it @ lab; lm = torch.sign(lc)*torch.abs(lc).pow(3.0)
            lin = M_SRGB_INV @ (M1it @ lm)
            if (lin>=-0.001).all() and (lin<=1.001).all(): lo=mid
            else: hi2=mid
        print(f"  L={Lv:.3f} C={lo.item():.6f}")

    ckpt = {"version":"v25","M1":M1_final.tolist(),"M2":M2_final.tolist(),
            "M1_inv":M1_inv.tolist(),"M2_inv":M2_inv.tolist(),
            "metrics":{"cv":fcv.item(),"mono":fmono.item(),"hue":fhue.item(),**fc}}
    with open("gen_v25.json","w") as f: json.dump(ckpt,f,indent=2)
    print(f"\nSaved: gen_v25.json")

if __name__ == "__main__":
    main()
