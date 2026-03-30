#!/usr/bin/env python3
"""v24 GPU: GenSpace CMA-ES with CUDA-accelerated fitness evaluation.

All fitness computations (gradient CV, monotonicity, hue, constraints)
are batched on GPU. Population members evaluated in parallel.

RTX 3080 Ti target: ~50-100x speedup over CPU.
"""

import json
import time
import sys
import numpy as np

import torch
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

import cma

# ══════════════════════════════════════════════════════════════════════
# Constants (GPU tensors)
# ══════════════════════════════════════════════════════════════════════

D65 = torch.tensor([0.95047, 1.0, 1.08883], device=device)

M_SRGB = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], device=device)
M_SRGB_INV = torch.linalg.inv(M_SRGB)

# OKLab
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

# v14 starting point
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


# ══════════════════════════════════════════════════════════════════════
# GPU utilities
# ══════════════════════════════════════════════════════════════════════

def signed_cbrt(x):
    return torch.sign(x) * torch.abs(x).pow(1.0/3.0)

def srgb_to_linear(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))

def linear_to_srgb(c):
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c.clamp(min=1e-10).pow(1.0/2.4) - 0.055)

def xyz_to_cielab_batch(xyz):
    """Batch XYZ -> CIE Lab. xyz: (..., 3)"""
    r = xyz / D65
    mask = r > 0.008856
    f = torch.where(mask, r.pow(1.0/3.0), 7.787 * r + 16.0/116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)

def forward_batch(M1, M2, xyz):
    """Batch forward: xyz (..., 3) -> lab (..., 3)"""
    lms = xyz @ M1.T
    lms_c = signed_cbrt(lms)
    return lms_c @ M2.T

def inverse_batch(M1_inv, M2_inv, lab):
    """Batch inverse: lab (..., 3) -> xyz (..., 3)"""
    lms_c = lab @ M2_inv.T
    lms = torch.sign(lms_c) * torch.abs(lms_c).pow(3.0)
    return lms @ M1_inv.T


# ══════════════════════════════════════════════════════════════════════
# Training pairs (pre-computed on GPU)
# ══════════════════════════════════════════════════════════════════════

def build_training_pairs():
    """Generate training pairs as GPU tensor. Returns (N, 2, 3) XYZ tensor."""
    pairs = []
    prims = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[0,0,0]]
    for i in range(len(prims)):
        for j in range(i+1, len(prims)):
            pairs.append((prims[i], prims[j]))
    for g1 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for g2 in [g1+0.2, g1+0.4]:
            if g2 <= 1.0:
                pairs.append(([g1]*3, [g2]*3))
    rng = np.random.RandomState(42)
    for _ in range(80):
        pairs.append((rng.rand(3).tolist(), rng.rand(3).tolist()))

    # Convert to XYZ tensors
    pair_tensor = torch.zeros(len(pairs), 2, 3, device=device)
    for i, (c1, c2) in enumerate(pairs):
        lin1 = srgb_to_linear(torch.tensor(c1, device=device))
        lin2 = srgb_to_linear(torch.tensor(c2, device=device))
        pair_tensor[i, 0] = M_SRGB @ lin1
        pair_tensor[i, 1] = M_SRGB @ lin2
    return pair_tensor


# Pre-build interpolation weights
N_STEPS = 25
T_STEPS = torch.linspace(0, 1, N_STEPS + 1, device=device)  # (26,)


def compute_gradient_cv_gpu(M1, M2, pairs):
    """Compute gradient CV for all pairs simultaneously on GPU.

    pairs: (N_pairs, 2, 3) XYZ
    Returns: mean_cv (scalar)
    """
    M1_inv = torch.linalg.inv(M1)
    M2_inv = torch.linalg.inv(M2)
    N = pairs.shape[0]

    # Forward transform both endpoints: (N, 3)
    lab1 = forward_batch(M1, M2, pairs[:, 0])
    lab2 = forward_batch(M1, M2, pairs[:, 1])

    # Interpolate: (N, 26, 3)
    t = T_STEPS.view(1, -1, 1)  # (1, 26, 1)
    labs = lab1.unsqueeze(1) + t * (lab2 - lab1).unsqueeze(1)  # (N, 26, 3)

    # Inverse -> XYZ -> sRGB 8-bit -> back to XYZ -> CIE Lab
    labs_flat = labs.reshape(-1, 3)  # (N*26, 3)
    xyz_flat = inverse_batch(M1_inv, M2_inv, labs_flat)
    lin_flat = (xyz_flat @ M_SRGB_INV.T).clamp(0, 1)
    srgb_flat = linear_to_srgb(lin_flat)
    srgb8 = (srgb_flat * 255).round() / 255.0
    lin_back = srgb_to_linear(srgb8)
    xyz_back = lin_back @ M_SRGB.T
    cielab_flat = xyz_to_cielab_batch(xyz_back.clamp(min=1e-10))

    # Reshape back: (N, 26, 3)
    cielab = cielab_flat.reshape(N, N_STEPS + 1, 3)

    # CIEDE2000 simplified between consecutive steps: (N, 25)
    cl1 = cielab[:, :-1]  # (N, 25, 3)
    cl2 = cielab[:, 1:]   # (N, 25, 3)

    dL = cl2[..., 0] - cl1[..., 0]
    C1 = (cl1[..., 1]**2 + cl1[..., 2]**2).sqrt()
    C2 = (cl2[..., 1]**2 + cl2[..., 2]**2).sqrt()
    dC = C2 - C1
    da = cl2[..., 1] - cl1[..., 1]
    db = cl2[..., 2] - cl1[..., 2]
    dH2 = (da**2 + db**2 - dC**2).clamp(min=0)
    dH = dH2.sqrt()

    SL = 1 + 0.015 * (cl1[..., 0] - 50)**2 / (20 + (cl1[..., 0] - 50)**2).sqrt()
    SC = 1 + 0.045 * C1
    SH = 1 + 0.015 * C1

    de = ((dL/SL)**2 + (dC/SC)**2 + (dH/SH)**2).sqrt()  # (N, 25)

    # CV per pair
    mean_de = de.mean(dim=1)  # (N,)
    std_de = de.std(dim=1)    # (N,)
    valid = mean_de > 0.001
    cvs = torch.where(valid, std_de / mean_de, torch.zeros_like(mean_de))

    mean_cv = cvs[valid].mean() if valid.any() else torch.tensor(999.0, device=device)
    top10 = cvs[valid].topk(max(1, valid.sum().item() // 10)).values.mean() if valid.any() else torch.tensor(999.0, device=device)

    return mean_cv, top10


# ══════════════════════════════════════════════════════════════════════
# Monotonicity penalty (GPU batch)
# ══════════════════════════════════════════════════════════════════════

# Pre-build hue and L grids
MONO_HUES = torch.arange(70, 101, 5, device=device, dtype=torch.float64) * (3.14159265358979 / 180.0)
MONO_LS = torch.arange(0.85, 1.001, 0.005, device=device)


def compute_monotonicity_gpu(M1, M2, cusp_L_max=0.975):
    """GPU-batched monotonicity penalty."""
    M1_inv = torch.linalg.inv(M1)
    M2_inv = torch.linalg.inv(M2)

    n_hues = MONO_HUES.shape[0]
    n_Ls = MONO_LS.shape[0]

    penalty = torch.tensor(0.0, device=device)

    for hi in range(n_hues):
        h = MONO_HUES[hi]
        cos_h, sin_h = torch.cos(h), torch.sin(h)

        # Binary search max chroma at each L — vectorized over L
        lo = torch.zeros(n_Ls, device=device)
        hi_c = torch.full((n_Ls,), 0.5, device=device)

        for _ in range(35):
            mid = (lo + hi_c) / 2  # (n_Ls,)
            # Build lab: (n_Ls, 3)
            lab = torch.stack([MONO_LS, mid * cos_h, mid * sin_h], dim=1)
            # Inverse
            lms_c = lab @ M2_inv.T
            lms = torch.sign(lms_c) * torch.abs(lms_c).pow(3.0)
            xyz = lms @ M1_inv.T
            lin = xyz @ M_SRGB_INV.T
            in_gamut = (lin >= -0.001).all(dim=1) & (lin <= 1.001).all(dim=1)
            lo = torch.where(in_gamut, mid, lo)
            hi_c = torch.where(in_gamut, hi_c, mid)

        max_chromas = lo  # (n_Ls,)

        # Find cusp
        cusp_idx = max_chromas.argmax()
        cusp_L = MONO_LS[cusp_idx]

        # Penalty: cusp too high
        if cusp_L > cusp_L_max:
            penalty = penalty + (cusp_L - cusp_L_max)**2 * 100

        # Penalty: positive slopes
        diffs = max_chromas[1:] - max_chromas[:-1]
        positive = diffs[diffs > 1e-5]
        if positive.numel() > 0:
            dL = 0.005
            penalty = penalty + (positive / dL).pow(2).sum()

    return penalty / n_hues


# ══════════════════════════════════════════════════════════════════════
# Other penalties (GPU)
# ══════════════════════════════════════════════════════════════════════

def compute_hue_penalty_gpu(M1, M2):
    """Hue ordering for 6 sRGB primaries."""
    prims_srgb = torch.tensor([
        [1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1]
    ], dtype=torch.float64, device=device)
    expected = torch.tensor([0, 60, 120, 180, 240, 300], dtype=torch.float64, device=device)

    lin = srgb_to_linear(prims_srgb)
    xyz = lin @ M_SRGB.T
    lab = forward_batch(M1, M2, xyz)  # (6, 3)

    h = torch.atan2(lab[:, 2], lab[:, 1]) * (180.0 / 3.14159265358979)
    h = h % 360

    dh = h - expected
    dh = torch.where(dh > 180, dh - 360, dh)
    dh = torch.where(dh < -180, dh + 360, dh)

    return (dh**2).mean()


def compute_constraints_gpu(M1, M2):
    """Compute all hard constraints. Returns dict of values."""
    # Yellow primary
    yellow_lin = torch.tensor([1.0, 1.0, 0.0], device=device)
    yellow_xyz = M_SRGB @ yellow_lin
    yellow_lab = forward_batch(M1, M2, yellow_xyz.unsqueeze(0)).squeeze()
    yellow_L = yellow_lab[0]
    yellow_C = (yellow_lab[1]**2 + yellow_lab[2]**2).sqrt()

    # Blue->White midpoint
    blue_xyz = M_SRGB @ srgb_to_linear(torch.tensor([0.0, 0.0, 1.0], device=device))
    white_xyz = M_SRGB @ torch.tensor([1.0, 1.0, 1.0], device=device)
    blue_lab = forward_batch(M1, M2, blue_xyz.unsqueeze(0)).squeeze()
    white_lab = forward_batch(M1, M2, white_xyz.unsqueeze(0)).squeeze()
    mid_lab = (blue_lab + white_lab) / 2
    M1_inv = torch.linalg.inv(M1)
    M2_inv = torch.linalg.inv(M2)
    mid_xyz = inverse_batch(M1_inv, M2_inv, mid_lab.unsqueeze(0)).squeeze()
    mid_lin = (M_SRGB_INV @ mid_xyz).clamp(0, 1)
    mid_srgb = linear_to_srgb(mid_lin)
    blue_white_gr = mid_srgb[1] / mid_srgb[0].clamp(min=1e-10)

    # Primary L range
    prims = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]],
                         dtype=torch.float64, device=device)
    prim_lin = srgb_to_linear(prims)
    prim_xyz = prim_lin @ M_SRGB.T
    prim_lab = forward_batch(M1, M2, prim_xyz)
    prim_L_range = prim_lab[:, 0].max() - prim_lab[:, 0].min()

    # Condition numbers
    cond_M1 = torch.linalg.cond(M1)
    cond_M2 = torch.linalg.cond(M2)

    return {
        'yellow_L': yellow_L.item(),
        'yellow_C': yellow_C.item(),
        'blue_white_gr': blue_white_gr.item(),
        'prim_L_range': prim_L_range.item(),
        'cond_M1': cond_M1.item(),
        'cond_M2': cond_M2.item(),
    }


# ══════════════════════════════════════════════════════════════════════
# Parameterization (D65-normalized M1)
# ══════════════════════════════════════════════════════════════════════

def ortho_basis_np(s):
    s_n = s / np.linalg.norm(s)
    v = np.array([1,0,0], dtype=np.float64) if abs(s_n[0]) < 0.9 else np.array([0,1,0], dtype=np.float64)
    e1 = v - np.dot(v, s_n) * s_n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s_n, e1)
    return e1, e2

def params_to_matrices_np(x):
    """13 params -> M1, M2 as numpy."""
    d65 = np.array([0.95047, 1.0, 1.08883])
    M1 = np.zeros((3, 3))
    for i in range(3):
        M1[i, 0] = x[2*i]
        M1[i, 1] = x[2*i+1]
        M1[i, 2] = (1.0 - M1[i,0]*d65[0] - M1[i,1]*d65[1]) / d65[2]
    s = np.sign(M1 @ d65) * np.abs(M1 @ d65)**(1/3)
    e1, e2 = ortho_basis_np(s)
    M2 = np.zeros((3, 3))
    M2[0] = x[6:9]
    Lw = M2[0] @ s
    if abs(Lw) < 1e-10: return None, None
    M2[0] /= Lw
    M2[1] = x[9]*e1 + x[10]*e2
    M2[2] = x[11]*e1 + x[12]*e2
    return M1, M2

def matrices_to_params_np(M1, M2):
    d65 = np.array([0.95047, 1.0, 1.08883])
    x = np.zeros(13)
    for i in range(3):
        x[2*i] = M1[i,0]
        x[2*i+1] = M1[i,1]
    x[6:9] = M2[0]
    s = np.sign(M1 @ d65) * np.abs(M1 @ d65)**(1/3)
    e1, e2 = ortho_basis_np(s)
    x[9], x[10] = M2[1]@e1, M2[1]@e2
    x[11], x[12] = M2[2]@e1, M2[2]@e2
    return x


# ══════════════════════════════════════════════════════════════════════
# CMA-ES with GPU fitness
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=== v24 GPU: GenSpace CMA-ES with 7 hard constraints ===")
    print()

    pairs = build_training_pairs()
    print(f"Training pairs: {pairs.shape[0]}")

    # Baselines
    with torch.no_grad():
        v14_cv, v14_top10 = compute_gradient_cv_gpu(V14_M1, V14_M2, pairs)
        v14_mono = compute_monotonicity_gpu(V14_M1, V14_M2)
        v14_hue = compute_hue_penalty_gpu(V14_M1, V14_M2)
        v14_cons = compute_constraints_gpu(V14_M1, V14_M2)
        print(f"v14: CV={v14_cv.item()*100:.2f}% mono={v14_mono.item():.4f} hue={v14_hue.item():.1f}deg2")
        print(f"     yL={v14_cons['yellow_L']:.3f} yC={v14_cons['yellow_C']:.3f} "
              f"B->W={v14_cons['blue_white_gr']:.3f} Lrange={v14_cons['prim_L_range']:.3f} "
              f"cond=({v14_cons['cond_M1']:.1f},{v14_cons['cond_M2']:.1f})")

        ok_cv, _ = compute_gradient_cv_gpu(OKLAB_M1, OKLAB_M2, pairs)
        ok_mono = compute_monotonicity_gpu(OKLAB_M1, OKLAB_M2)
        print(f"OKLab: CV={ok_cv.item()*100:.2f}% mono={ok_mono.item():.4f}")
    print()

    # Pack v14
    v14_m1_np = V14_M1.cpu().numpy()
    v14_m2_np = V14_M2.cpu().numpy()
    x0 = matrices_to_params_np(v14_m1_np, v14_m2_np)

    # Verify
    M1_check, M2_check = params_to_matrices_np(x0)
    assert np.allclose(M1_check, v14_m1_np, atol=1e-10), "Round-trip failed"
    print(f"Param round-trip OK (13 params)")

    # ── Hard constraints ──
    YELLOW_C_MIN = 0.15
    YELLOW_L_MIN = 0.95
    BLUE_WHITE_GR_MIN = 1.20
    COND_M1_MAX = 4.0
    COND_M2_MAX = 10.0
    PRIM_L_RANGE_MIN = 0.45
    CV_MAX = ok_cv.item() * 1.1  # max 10% worse than OKLab

    print(f"\nHard constraints:")
    print(f"  Yellow C > {YELLOW_C_MIN}")
    print(f"  Yellow L > {YELLOW_L_MIN}")
    print(f"  Blue->White G/R > {BLUE_WHITE_GR_MIN}")
    print(f"  cond(M1) < {COND_M1_MAX}")
    print(f"  cond(M2) < {COND_M2_MAX}")
    print(f"  Primary L range > {PRIM_L_RANGE_MIN}")
    print(f"  CV < {CV_MAX*100:.2f}%")
    print()

    best_loss = float("inf")
    best_x = x0.copy()
    best_info = {}
    eval_count = 0
    t0 = time.time()

    def objective(x):
        nonlocal eval_count, best_loss, best_x, best_info

        try:
            M1_np, M2_np = params_to_matrices_np(x)
            if M1_np is None:
                return 999.0

            M1 = torch.tensor(M1_np, device=device)
            M2 = torch.tensor(M2_np, device=device)

            with torch.no_grad():
                cons = compute_constraints_gpu(M1, M2)

                # Hard constraint violations -> huge penalty
                violations = 0
                if cons['yellow_C'] < YELLOW_C_MIN: violations += (YELLOW_C_MIN - cons['yellow_C'])**2 * 1000
                if cons['yellow_L'] < YELLOW_L_MIN: violations += (YELLOW_L_MIN - cons['yellow_L'])**2 * 1000
                if cons['blue_white_gr'] < BLUE_WHITE_GR_MIN: violations += (BLUE_WHITE_GR_MIN - cons['blue_white_gr'])**2 * 1000
                if cons['cond_M1'] > COND_M1_MAX: violations += (cons['cond_M1'] - COND_M1_MAX)**2 * 10
                if cons['cond_M2'] > COND_M2_MAX: violations += (cons['cond_M2'] - COND_M2_MAX)**2 * 10
                if cons['prim_L_range'] < PRIM_L_RANGE_MIN: violations += (PRIM_L_RANGE_MIN - cons['prim_L_range'])**2 * 1000

                if violations > 0:
                    return 100.0 + violations

                cv, top10 = compute_gradient_cv_gpu(M1, M2, pairs)
                cv_val = cv.item()

                if cv_val > CV_MAX:
                    return 50.0 + (cv_val - CV_MAX)**2 * 100

                mono = compute_monotonicity_gpu(M1, M2).item()
                hue = compute_hue_penalty_gpu(M1, M2).item()

                loss = cv_val + 0.3 * top10.item() + 2.0 * mono + 0.3 * hue

        except Exception as e:
            return 999.0

        eval_count += 1
        if loss < best_loss:
            best_loss = loss
            best_x = x.copy()
            best_info = {**cons, 'cv': cv_val, 'mono': mono, 'hue': hue}
            elapsed = time.time() - t0
            print(f"  #{eval_count:>5d} [{elapsed:6.1f}s] loss={loss:.4f} "
                  f"CV={cv_val*100:.1f}% mono={mono:.4f} hue={hue:.1f}deg2 "
                  f"yL={cons['yellow_L']:.3f} yC={cons['yellow_C']:.3f} "
                  f"B->W={cons['blue_white_gr']:.2f} cond=({cons['cond_M1']:.1f},{cons['cond_M2']:.1f})")

        return loss

    # CMA-ES
    print("Starting CMA-ES...")
    opts = cma.CMAOptions()
    opts.set("maxiter", 300)
    opts.set("popsize", 48)
    opts.set("tolfun", 1e-9)
    opts.set("verbose", -1)

    es = cma.CMAEvolutionStrategy(x0, 0.008, opts)
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)

    print(f"\n{'='*60}")
    print(f"Finished: {eval_count} evals in {time.time()-t0:.1f}s")
    print()

    # Final results
    M1_np, M2_np = params_to_matrices_np(best_x)
    M1 = torch.tensor(M1_np, device=device)
    M2 = torch.tensor(M2_np, device=device)

    with torch.no_grad():
        final_cv, _ = compute_gradient_cv_gpu(M1, M2, pairs)
        final_mono = compute_monotonicity_gpu(M1, M2)
        final_hue = compute_hue_penalty_gpu(M1, M2)
        final_cons = compute_constraints_gpu(M1, M2)

    print(f"Final: CV={final_cv.item()*100:.2f}% mono={final_mono.item():.4f} hue={final_hue.item():.1f}deg2")
    print(f"       yL={final_cons['yellow_L']:.3f} yC={final_cons['yellow_C']:.3f} "
          f"B->W={final_cons['blue_white_gr']:.3f} Lrange={final_cons['prim_L_range']:.3f} "
          f"cond=({final_cons['cond_M1']:.1f},{final_cons['cond_M2']:.1f})")
    print()

    M1_inv = np.linalg.inv(M1_np)
    M2_inv = np.linalg.inv(M2_np)

    print("M1 =")
    for row in M1_np:
        print(f"  [{row[0]:>22.16f}, {row[1]:>22.16f}, {row[2]:>22.16f}],")
    print("M2 =")
    for row in M2_np:
        print(f"  [{row[0]:>22.16f}, {row[1]:>22.16f}, {row[2]:>22.16f}],")
    print("M1_inv =")
    for row in M1_inv:
        print(f"  [{row[0]:>22.16f}, {row[1]:>22.16f}, {row[2]:>22.16f}],")
    print("M2_inv =")
    for row in M2_inv:
        print(f"  [{row[0]:>22.16f}, {row[1]:>22.16f}, {row[2]:>22.16f}],")

    # Yellow boundary
    print(f"\nYellow boundary (hue 85deg):")
    M2_inv_t = torch.linalg.inv(M2)
    M1_inv_t = torch.linalg.inv(M1)
    h = torch.tensor(85.0 * 3.14159265358979 / 180.0, device=device)
    ch, sh = torch.cos(h), torch.sin(h)
    for L_val in [0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0]:
        L = torch.tensor(L_val, device=device)
        lo_t = torch.tensor(0.0, device=device)
        hi_t = torch.tensor(0.5, device=device)
        for _ in range(40):
            mid = (lo_t + hi_t) / 2
            lab = torch.stack([L, mid*ch, mid*sh])
            lc = M2_inv_t @ lab
            lm = torch.sign(lc) * torch.abs(lc).pow(3.0)
            lin = M_SRGB_INV @ (M1_inv_t @ lm)
            if (lin >= -0.001).all() and (lin <= 1.001).all():
                lo_t = mid
            else:
                hi_t = mid
        print(f"  L={L_val:.3f} C={lo_t.item():.6f}")

    # Save
    ckpt = {
        "version": "v24-gpu",
        "M1": M1_np.tolist(), "M2": M2_np.tolist(),
        "M1_inv": M1_inv.tolist(), "M2_inv": M2_inv.tolist(),
        "metrics": {
            "cv": final_cv.item(), "mono": final_mono.item(),
            "hue": final_hue.item(), **final_cons
        }
    }
    with open("gen_v24_gpu.json", "w") as f:
        json.dump(ckpt, f, indent=2)
    print(f"\nSaved: gen_v24_gpu.json")


if __name__ == "__main__":
    main()
