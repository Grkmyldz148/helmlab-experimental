#!/usr/bin/env python3
"""v8 GenSpace -- GPU-BATCHED CMA-ES (v4: white norm + hue penalty).

True GPU batching: all candidates evaluated simultaneously via bmm.
Pipeline: XYZ -> M1 -> cbrt -> M2 -> L_corr -> Lab
21 DOF: M1(9) + M2(9) + L_corr(3)

v4 fixes:
- White normalization: white sRGB(1,1,1) must map to L~=1.0
- Hue drift penalty: penalizes hue rotation along gradients
- Better cusp coverage: penalizes dead zones in gamut boundary
"""
import json, time, sys
from datetime import datetime
import numpy as np
import torch
import cma

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
sys.stdout.flush()

# sRGB <-> XYZ
SRGB_TO_XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]], dtype=DTYPE, device=DEVICE)
XYZ_TO_SRGB = torch.linalg.inv(SRGB_TO_XYZ)

# Known parameters
V7B_M1 = np.array([
    [6.213663274448127, -0.5041794153770129, -0.40416891025666857],
    [-1.1592256796157883, 4.350194381717271, 0.5254938968299478],
    [0.0008170122534259527, 0.7226718820884986, 2.227799849833172]])
V7B_M2 = np.array([
    [0.4675499211910323, 0.20915320090703618, -0.08488334505679182],
    [0.4843952725673558, -0.3665958307304812, -0.17266206907852755],
    [-0.04418360083197623, 0.39383739736845824, -0.36863136176600936]])
V7B_LC = np.array([-0.09792777021381058, -0.26695959819582816, 0.30350768100715936])

OK_M1 = np.array([
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715, 0.0361456387],
    [0.0482003018, 0.2643662691, 0.6338517070]])
OK_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660]])


# ================================================================
# sRGB helpers (work on any shape, last dim=3)
# ================================================================
def srgb_to_linear(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(c):
    c = c.clamp(min=0)
    return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c.pow(1/2.4) - 0.055)

def srgb_to_xyz(rgb):
    return srgb_to_linear(rgb) @ SRGB_TO_XYZ.T

def xyz_to_srgb(xyz):
    return linear_to_srgb(xyz @ XYZ_TO_SRGB.T).clamp(0, 1)


# ================================================================
# GenSpace forward/inverse -- BATCHED across candidates
# ================================================================
EPS_EYE = torch.eye(3, dtype=DTYPE, device=DEVICE) * 1e-10

def forward_shared(xyz, M1, M2, lc):
    """Shared xyz (N,3) + batched params -> (C,N,3) Lab."""
    C = M1.shape[0]
    x = xyz.unsqueeze(0).expand(C, -1, -1)  # (C,N,3)
    lms = torch.bmm(x, M1.transpose(1, 2)).clamp(min=0).pow(1.0/3.0)
    lab = torch.bmm(lms, M2.transpose(1, 2))
    L = lab[:, :, 0]
    t = L * (1 - L)
    lab = lab.clone()
    lab[:, :, 0] = L + lc[:, 0:1]*t + lc[:, 1:2]*t*(0.5-L) + lc[:, 2:3]*t*t
    return lab

def forward_per(xyz, M1, M2, lc):
    """Per-candidate xyz (C,N,3) + batched params -> (C,N,3) Lab."""
    lms = torch.bmm(xyz, M1.transpose(1, 2)).clamp(min=0).pow(1.0/3.0)
    lab = torch.bmm(lms, M2.transpose(1, 2))
    L = lab[:, :, 0]
    t = L * (1 - L)
    lab = lab.clone()
    lab[:, :, 0] = L + lc[:, 0:1]*t + lc[:, 1:2]*t*(0.5-L) + lc[:, 2:3]*t*t
    return lab

def inverse_batch(lab, M1, M2, lc, n_iter=10):
    """(C,N,3) Lab -> (C,N,3) XYZ."""
    M1i = torch.linalg.inv(M1 + EPS_EYE)
    M2i = torch.linalg.inv(M2 + EPS_EYE)
    lab = lab.clone()
    L0 = lab[:, :, 0].clone()
    L = L0.clone()
    lc0, lc1, lc2 = lc[:, 0:1], lc[:, 1:2], lc[:, 2:3]
    for _ in range(n_iter):
        t = L * (1 - L)
        dt = 1 - 2*L
        f = L + lc0*t + lc1*t*(0.5-L) + lc2*t*t - L0
        df = 1 + lc0*dt + lc1*(dt*(0.5-L) - t) + lc2*2*t*dt
        df = torch.where(df.abs() < 1e-12, torch.ones_like(df), df)
        L = L - f / df
    lab[:, :, 0] = L
    lms_c = torch.bmm(lab, M2i.transpose(1, 2))
    lms = lms_c.clamp(min=0).pow(3)
    return torch.bmm(lms, M1i.transpose(1, 2))


# ================================================================
# Test data (cached on GPU)
# ================================================================
rng = np.random.default_rng(42)
_r1 = torch.tensor(rng.uniform(0, 1, (600, 3)), dtype=DTYPE, device=DEVICE)
_r2 = torch.tensor(rng.uniform(0, 1, (600, 3)), dtype=DTYPE, device=DEVICE)
mask = (_r1 - _r2).pow(2).sum(-1).sqrt() > 0.3
_r1, _r2 = _r1[mask], _r2[mask]
N_P = min(200, _r1.shape[0])
pairs1, pairs2 = _r1[:N_P], _r2[:N_P]
xyz_p1 = srgb_to_xyz(pairs1)
xyz_p2 = srgb_to_xyz(pairs2)
print(f"Gradient pairs: {N_P}")

BLUE_XYZ = srgb_to_xyz(torch.tensor([[0., 0., 1.]], dtype=DTYPE, device=DEVICE))
WHITE_XYZ = srgb_to_xyz(torch.tensor([[1., 1., 1.]], dtype=DTYPE, device=DEVICE))
RED_XYZ = srgb_to_xyz(torch.tensor([[1., 0., 0.]], dtype=DTYPE, device=DEVICE))
BLACK_XYZ = srgb_to_xyz(torch.tensor([[0., 0., 0.]], dtype=DTYPE, device=DEVICE))
GRAY_XYZ = srgb_to_xyz(
    torch.linspace(0.05, 0.95, 20, dtype=DTYPE, device=DEVICE).unsqueeze(1).expand(-1, 3))

# Primary colors for hue drift check
PRIMARIES_RGB = torch.tensor([
    [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 1, 1], [0, 0, 1], [1, 0, 1]], dtype=DTYPE, device=DEVICE)
PRIMARIES_XYZ = srgb_to_xyz(PRIMARIES_RGB)  # (6,3)

# Cusp scan: 36 hue angles
CUSP_N_HUES = 36
cusp_angles = torch.linspace(0, 2*torch.pi*(1 - 1/CUSP_N_HUES), CUSP_N_HUES,
                              dtype=DTYPE, device=DEVICE)
cusp_a_dir = torch.cos(cusp_angles)  # (36,)
cusp_b_dir = torch.sin(cusp_angles)  # (36,)

N_S = 24
t_steps = torch.linspace(0, 1, N_S, dtype=DTYPE, device=DEVICE)
sys.stdout.flush()


# ================================================================
# GPU-batched evaluation
# ================================================================
def eval_batch(params_list):
    """Evaluate all candidates in one GPU pass. Returns list of floats."""
    C = len(params_list)
    arr = np.stack([np.asarray(x, dtype=np.float64) for x in params_list])
    M1 = torch.tensor(arr[:, :9].reshape(C, 3, 3), dtype=DTYPE, device=DEVICE)
    M2 = torch.tensor(arr[:, 9:18].reshape(C, 3, 3), dtype=DTYPE, device=DEVICE)
    lc = torch.tensor(arr[:, 18:21], dtype=DTYPE, device=DEVICE)

    loss = torch.full((C,), 1e6, dtype=DTYPE, device=DEVICE)

    # Validity checks
    det1 = torch.linalg.det(M1).abs()
    det2 = torch.linalg.det(M2).abs()
    cond1 = torch.linalg.cond(M1)
    cond2 = torch.linalg.cond(M2)
    lc_mag = lc.abs().max(dim=1).values  # (C,) max |lc_i|
    valid = (det1 > 0.001) & (det2 > 0.0001) & (cond1 < 20) & (cond2 < 20) & (lc_mag < 1.0)

    if not valid.any():
        return loss.cpu().numpy().tolist()

    try:
        # ===== WHITE NORMALIZATION (CRITICAL) =====
        # White sRGB(1,1,1) must map to L~=1.0, a~=0, b~=0
        lab_w = forward_shared(WHITE_XYZ, M1, M2, lc)   # (C,1,3)
        white_L = lab_w[:, 0, 0]  # (C,)
        white_L_err = (white_L - 1.0).abs()
        # Hard reject if white L deviates by more than 0.5
        valid = valid & (white_L_err < 0.5)

        # Black sRGB(0,0,0) must map to L~=0.0
        lab_k = forward_shared(BLACK_XYZ, M1, M2, lc)   # (C,1,3)
        black_L = lab_k[:, 0, 0]  # (C,)
        black_L_err = black_L.abs()

        if not valid.any():
            return loss.cpu().numpy().tolist()

        # ===== Blue->White midpoint =====
        lab_b = forward_shared(BLUE_XYZ, M1, M2, lc)   # (C,1,3)
        mid_bw = inverse_batch((lab_b + lab_w) / 2, M1, M2, lc)
        srgb_bw = xyz_to_srgb(mid_bw)
        blue_gr = srgb_bw[:, 0, 1] / srgb_bw[:, 0, 0].clamp(min=1e-8)  # (C,)

        # ===== Red->White midpoint =====
        lab_r = forward_shared(RED_XYZ, M1, M2, lc)
        mid_rw = inverse_batch((lab_r + lab_w) / 2, M1, M2, lc)
        srgb_rw = xyz_to_srgb(mid_rw)
        red_gb = srgb_rw[:, 0, 1] - srgb_rw[:, 0, 2]  # (C,)

        # ===== Achromatic =====
        lab_g = forward_shared(GRAY_XYZ, M1, M2, lc)    # (C,20,3)
        ach = lab_g[:, :, 1:].abs().amax(dim=(1, 2))    # (C,)

        # ===== Gradient CV (main cost) =====
        l1 = forward_shared(xyz_p1, M1, M2, lc)  # (C,N,3)
        l2 = forward_shared(xyz_p2, M1, M2, lc)  # (C,N,3)
        l1e = l1.unsqueeze(2)  # (C,N,1,3)
        l2e = l2.unsqueeze(2)  # (C,N,1,3)
        te = t_steps.view(1, 1, -1, 1)  # (1,1,S,1)
        interp = l1e + (l2e - l1e) * te  # (C,N,S,3)
        flat = interp.reshape(C, -1, 3)  # (C, N*S, 3)
        fxyz = inverse_batch(flat, M1, M2, lc, n_iter=15)
        fsrgb = xyz_to_srgb(fxyz)
        flin = srgb_to_linear(fsrgb).reshape(C, N_P, N_S, 3)
        d = (flin[:, :, 1:] - flin[:, :, :-1]).pow(2).sum(-1).sqrt()  # (C,N,S-1)
        cv = (d.std(2) / d.mean(2).clamp(min=1e-10)).mean(1)  # (C,)

        # ===== Primary hue linearity (6 primary→white gradients) =====
        lab_prims = forward_shared(PRIMARIES_XYZ, M1, M2, lc)  # (C,6,3)
        lab_w_exp = lab_w.expand(-1, 6, -1)  # (C,6,3)
        # 10-step interpolation for hue check
        t10 = torch.linspace(0, 1, 10, dtype=DTYPE, device=DEVICE)
        t10e = t10.view(1, 1, 10, 1)
        prim_interp = lab_prims.unsqueeze(2) + (lab_w_exp.unsqueeze(2) - lab_prims.unsqueeze(2)) * t10e  # (C,6,10,3)
        prim_hue = torch.atan2(prim_interp[..., 2], prim_interp[..., 1])  # (C,6,10)
        prim_hue_diff = prim_hue[:, :, 1:] - prim_hue[:, :, :-1]
        prim_hue_diff = torch.atan2(torch.sin(prim_hue_diff), torch.cos(prim_hue_diff))
        prim_hue_cum = torch.cumsum(prim_hue_diff, dim=2)
        prim_max_drift = prim_hue_cum.abs().amax(dim=2) * (180/torch.pi)  # (C,6) max drift per primary
        prim_hue_rms = (prim_hue_cum.pow(2).mean(2).mean(1)).sqrt() * (180/torch.pi)  # (C,)
        # Blue-specific hue drift (index 4 = blue)
        blue_hue_drift = prim_max_drift[:, 4]  # (C,)

        # ===== Compose loss =====
        L = cv * 5.0

        # ----- White normalization (HIGHEST PRIORITY) -----
        L = L + 200.0 * white_L_err ** 2  # Strong: L=15.8 would add 200*(14.8)^2 = 43800
        L = L + 50.0 * black_L_err ** 2   # Black should map to L~=0

        # ----- Blue->White sky-blue (HIGH PRIORITY) -----
        blue_r = srgb_bw[:, 0, 0]
        # Ramp penalty: 0 at G/R=1.30+, increasing below
        L = L + torch.where(blue_gr < 1.30,
                            200.0 * (1.30 - blue_gr) ** 2,
                            torch.zeros_like(L))
        L = L + torch.where(blue_gr < 1.15,
                            500.0 * (1.15 - blue_gr) ** 2,
                            torch.zeros_like(L))
        # Hard reject if blue is lavender (G/R < 1.05)
        valid = valid & (blue_gr > 1.05)
        # Penalize too-cyan (R too low — needs some warmth)
        L = L + torch.where(blue_r < 0.35,
                            100.0 * (0.35 - blue_r) ** 2,
                            torch.zeros_like(L))
        # Penalize overshoot (G/R > 1.60 means too cyan-shifted)
        L = L + torch.where(blue_gr > 1.60,
                            20.0 * (blue_gr - 1.60) ** 2,
                            torch.zeros_like(L))
        # Blue hue drift: penalize if blue→white gradient shifts hue
        L = L + torch.where(blue_hue_drift > 15.0,
                            5.0 * (blue_hue_drift - 15.0) ** 2,
                            torch.zeros_like(L))

        # ----- Red->White orange -----
        rabs = red_gb.abs()
        L = L + torch.where(rabs > 0.08,
                            5.0 * (rabs - 0.08) ** 2,
                            torch.zeros_like(L))

        # ----- Achromatic -----
        ach_log = torch.log10(ach.clamp(min=1e-30))
        L = L + torch.where(ach > 1e-6,
                            2.0 * (ach_log + 6) ** 2,
                            torch.zeros_like(L))

        # ----- Primary hue linearity (focused, not random pairs) -----
        L = L + torch.where(prim_hue_rms > 30.0,
                            0.3 * (prim_hue_rms - 30.0) ** 2,
                            torch.zeros_like(L))

        # ----- Condition regularization (TIGHT) -----
        L = L + torch.where(cond1 > 3, 0.5 * (cond1 - 3) ** 2, torch.zeros_like(L))
        L = L + torch.where(cond2 > 8, 0.3 * (cond2 - 8) ** 2, torch.zeros_like(L))

        # ----- L_corr magnitude penalty -----
        L = L + torch.where(lc_mag > 0.35,
                            10.0 * (lc_mag - 0.35) ** 2,
                            torch.zeros_like(L))

        # ----- Round-trip check -----
        rt_xyz = srgb_to_xyz(torch.rand(30, 3, dtype=DTYPE, device=DEVICE))
        rt_lab = forward_shared(rt_xyz, M1, M2, lc)
        rt_back = inverse_batch(rt_lab, M1, M2, lc, n_iter=20)
        rt_err = (rt_xyz.unsqueeze(0) - rt_back).abs().amax(dim=(1, 2))  # (C,)
        L = torch.where(rt_err > 1e-6, torch.tensor(1e6, dtype=DTYPE, device=DEVICE), L)
        rt_log = torch.log10(rt_err.clamp(min=1e-30))
        L = L + torch.where(rt_err > 1e-10,
                            0.5 * (rt_log + 10) ** 2,
                            torch.zeros_like(L))

        loss = torch.where(valid, L, loss)

    except Exception as e:
        print(f"  Batch error: {e}", file=sys.stderr)
        sys.stderr.flush()

    return loss.cpu().numpy().tolist()


# ================================================================
# Helpers
# ================================================================
def pack(M1, M2, lc):
    return np.concatenate([M1.ravel(), M2.ravel(), lc])

def unpack(x):
    x = np.asarray(x, dtype=np.float64)
    return x[:9].reshape(3, 3), x[9:18].reshape(3, 3), x[18:21]

def report(x, label=""):
    M1n, M2n, lcn = unpack(x)
    M1t = torch.tensor(M1n, dtype=DTYPE, device=DEVICE).unsqueeze(0)
    M2t = torch.tensor(M2n, dtype=DTYPE, device=DEVICE).unsqueeze(0)
    lct = torch.tensor(lcn, dtype=DTYPE, device=DEVICE).unsqueeze(0)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # White/Black normalization check
    lab_w = forward_shared(WHITE_XYZ, M1t, M2t, lct)
    lab_k = forward_shared(BLACK_XYZ, M1t, M2t, lct)
    print(f"  White L={lab_w[0,0,0].item():.6f}  a={lab_w[0,0,1].item():.2e}  b={lab_w[0,0,2].item():.2e}")
    print(f"  Black L={lab_k[0,0,0].item():.6f}  a={lab_k[0,0,1].item():.2e}  b={lab_k[0,0,2].item():.2e}")

    names = ["Red", "Yellow", "Green", "Cyan", "Blue", "Magenta"]
    prims = torch.tensor([
        [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 1, 1], [0, 0, 1], [1, 0, 1]], dtype=DTYPE, device=DEVICE)
    wh = torch.ones(1, 3, dtype=DTYPE, device=DEVICE)

    hue_drifts = []
    for i, nm in enumerate(names):
        p_xyz = srgb_to_xyz(prims[i:i+1])
        w_xyz = srgb_to_xyz(wh)
        lp = forward_shared(p_xyz, M1t, M2t, lct)
        lw = forward_shared(w_xyz, M1t, M2t, lct)

        # 10-step interpolation for hue drift
        t10 = torch.linspace(0, 1, 10, dtype=DTYPE, device=DEVICE).view(1, 10, 1)
        interp10 = lp + (lw - lp) * t10  # (1,10,3)
        h10 = torch.atan2(interp10[0, :, 2], interp10[0, :, 1])
        hd = h10[1:] - h10[:-1]
        hd = torch.atan2(torch.sin(hd), torch.cos(hd))
        max_drift = torch.cumsum(hd, dim=0).abs().max().item() * 180 / torch.pi
        hue_drifts.append(max_drift)

        mid = inverse_batch((lp + lw) / 2, M1t, M2t, lct, n_iter=30)
        s = xyz_to_srgb(mid)[0, 0].cpu().numpy()
        extra = ""
        if nm == "Blue":
            extra = f" G/R={s[1]/max(s[0], 1e-8):.3f}"
        if nm == "Red":
            extra = f" G-B={s[1]-s[2]:.3f}"
        print(f"    {nm:8s}->W: #{int(s[0]*255):02x}{int(s[1]*255):02x}{int(s[2]*255):02x}"
              f" R={s[0]:.3f} G={s[1]:.3f} B={s[2]:.3f}{extra}"
              f"  hue_drift={max_drift:.1f}deg")

    print(f"  Hue drift: mean={np.mean(hue_drifts):.1f} max={np.max(hue_drifts):.1f} deg")

    loss_val = eval_batch([x])[0]
    print(f"  Loss: {loss_val:.5f}")
    print(f"  Cond: M1={np.linalg.cond(M1n):.2f} M2={np.linalg.cond(M2n):.2f}")

    lab_g = forward_shared(GRAY_XYZ, M1t, M2t, lct)
    print(f"  Achromatic: {lab_g[:, :, 1:].abs().max().item():.2e}")

    rt_x = srgb_to_xyz(torch.rand(500, 3, dtype=DTYPE, device=DEVICE))
    rt_lab = forward_shared(rt_x, M1t, M2t, lct)
    rt_back = inverse_batch(rt_lab, M1t, M2t, lct, n_iter=30)
    print(f"  Round-trip: {(rt_x.unsqueeze(0) - rt_back).abs().max().item():.2e}")

    print(f"\n  M1 = {M1n.tolist()}")
    print(f"  M2 = {M2n.tolist()}")
    print(f"  L_corr = {lcn.tolist()}")
    sys.stdout.flush()
    return loss_val


# ================================================================
# CMA-ES runner
# ================================================================
def run_seed(name, x0, sigma, n_gens=300, popsize=96):
    print(f"\n{'='*60}")
    print(f"  Seed: {name} | sigma={sigma} | pop={popsize} | gens={n_gens}")
    print(f"{'='*60}")
    sys.stdout.flush()

    opts = cma.CMAOptions()
    opts['popsize'] = popsize
    opts['maxiter'] = n_gens
    opts['verb_disp'] = 0
    opts['verb_log'] = 0
    opts['tolfun'] = 1e-9
    opts['seed'] = abs(hash(name)) % (2**31)

    es = cma.CMAEvolutionStrategy(x0, sigma, opts)
    best_loss = 1e9
    best_x = x0.copy()
    t0 = time.time()
    gen = 0
    stale = 0

    while not es.stop() and gen < n_gens:
        solutions = es.ask()
        fitnesses = eval_batch(solutions)  # ALL candidates in one GPU pass
        es.tell(solutions, fitnesses)

        idx = int(np.argmin(fitnesses))
        if fitnesses[idx] < best_loss - 1e-6:
            best_loss = fitnesses[idx]
            best_x = np.array(solutions[idx]).copy()
            stale = 0
        else:
            stale += 1

        gen += 1
        if gen % 10 == 0 or gen == 1:
            M1n, M2n, lcn = unpack(best_x)
            M1t = torch.tensor(M1n, dtype=DTYPE, device=DEVICE).unsqueeze(0)
            M2t = torch.tensor(M2n, dtype=DTYPE, device=DEVICE).unsqueeze(0)
            lct = torch.tensor(lcn, dtype=DTYPE, device=DEVICE).unsqueeze(0)
            try:
                lb = forward_shared(BLUE_XYZ, M1t, M2t, lct)
                lw = forward_shared(WHITE_XYZ, M1t, M2t, lct)
                wL = lw[0, 0, 0].item()
                mid = inverse_batch((lb + lw) / 2, M1t, M2t, lct)
                s = xyz_to_srgb(mid)
                gr = (s[0, 0, 1] / s[0, 0, 0].clamp(min=1e-8)).item()
                el = time.time() - t0
                print(f"  Gen {gen:4d} | loss={best_loss:.5f} B-W G/R={gr:.3f} white_L={wL:.4f}"
                      f" | {el:.0f}s ({el/gen:.2f}s/gen)")
            except:
                print(f"  Gen {gen:4d} | loss={best_loss:.5f}")
            sys.stdout.flush()

        # Early stop
        if stale >= 100:
            print(f"  Early stop: no improvement for 100 gens")
            break

    el = time.time() - t0
    print(f"  Done: {gen} gens in {el:.1f}s ({el/gen:.2f}s/gen)")
    sys.stdout.flush()
    return best_x, best_loss


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print("v8 GenSpace Optimizer v5 -- White Norm + Strong Blue + Tight Cond (GPU-Batched)")
    print(f"Pairs: {N_P}, Steps: {N_S}")
    sys.stdout.flush()

    # v8-oklab from previous run (great blue, bad white norm)
    V8OK_M1 = np.array([
        [1.0370803567498818, 0.41706427750501835, -0.14209855427541408],
        [-0.14234637826770403, 1.1270539513927563, 0.18903613432870972],
        [0.30221174768723685, 0.012457270779735455, 0.7537277399620994]])
    V8OK_M2 = np.array([
        [0.7819766780549597, 1.1297085069614834, 0.783917296723455],
        [1.9964497289994135, -2.530965606327474, 0.5183547019095403],
        [0.029534519556990783, 0.636449174413001, -0.6905960437908145]])
    V8OK_LC = np.array([0.16391476505709447, 0.35846893461768116, 0.3281879637032988])

    # Baselines
    print("\n--- BASELINES ---")
    report(pack(V7B_M1, V7B_M2, V7B_LC), "v7b (current deployed)")
    report(pack(OK_M1, OK_M2, np.zeros(3)), "OKLab")
    report(pack(V8OK_M1, V8OK_M2, V8OK_LC), "v8-oklab (white_L broken)")
    sys.stdout.flush()

    # Seeds:
    # 1. OKLab: already has good blue G/R=1.408 AND white_L=1.0
    # 2. v7b: good CV, needs blue fix
    # 3. v8-oklab: great blue but white_L=15.8, CMA-ES should be able to fix M2 L-row
    # 4. hybrid: mix of v7b structure with OKLab normalization
    # 5. random perturbation of OKLab
    seeds = [
        ("oklab",     pack(OK_M1, OK_M2, np.zeros(3)),                            0.05),
        ("v7b",       pack(V7B_M1, V7B_M2, V7B_LC),                              0.08),
        ("hybrid",    pack(V7B_M1, OK_M2, np.zeros(3)),                           0.06),
        ("rnd1",      pack(OK_M1, OK_M2, np.zeros(3))
                      + np.random.RandomState(42).randn(21) * 0.03,              0.05),
        ("rnd2",      pack(V7B_M1, V7B_M2, V7B_LC)
                      + np.random.RandomState(99).randn(21) * 0.05,              0.06),
    ]

    all_results = {}
    for name, x0, sigma in seeds:
        bx, bl = run_seed(name, x0, sigma, n_gens=500, popsize=96)
        report(bx, f"Best from {name}")
        all_results[name] = {"x": bx, "loss": bl}

        # Save per-seed checkpoint
        M1n, M2n, lcn = unpack(bx)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt = {
            "version": "v8v4",
            "seed": name,
            "loss": float(bl),
            "M1": M1n.tolist(),
            "gamma": [1.0/3, 1.0/3, 1.0/3],
            "M2": M2n.tolist(),
            "L_corr": lcn.tolist(),
        }
        with open(f"v8v4_{name}_{ts}.json", "w") as f:
            json.dump(ckpt, f, indent=2)
        print(f"  Saved: v8v4_{name}_{ts}.json")
        sys.stdout.flush()

    # Overall winner
    best_name = min(all_results, key=lambda k: all_results[k]["loss"])
    print(f"\n{'#'*60}")
    print(f"  WINNER: {best_name} (loss={all_results[best_name]['loss']:.5f})")
    print(f"{'#'*60}")
    report(all_results[best_name]["x"], "WINNER")
    sys.stdout.flush()
