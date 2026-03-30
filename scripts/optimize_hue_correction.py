#!/usr/bin/env python3
"""Hue Correction on top of v7bblend30_achfix.

Fixed: M1, M2, L_corr (from blend30_achfix)
Free: hue_cos1..4, hue_sin1..4 (8 params)

Goal: fix hue RMS from 28.6 -> <10 without touching Munsell/CV.
"""

import json, math, os, sys, time, argparse
from datetime import datetime
import numpy as np
import torch
import cma

torch.set_default_dtype(torch.float64)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {dev}", flush=True)

pa = argparse.ArgumentParser()
pa.add_argument("--gens", type=int, default=300)
pa.add_argument("--pop", type=int, default=96)
pa.add_argument("--seeds", type=int, default=6)
args = pa.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
CKPT = os.path.join(ROOT, "checkpoints")

# Load base model
base_path = os.path.join(CKPT, "v7bblend_30_achfix.json")
with open(base_path) as f:
    BASE = json.load(f)

M1 = torch.tensor(BASE["M1"], device=dev, dtype=torch.float64)
M2 = torch.tensor(BASE["M2"], device=dev, dtype=torch.float64)
M1i = torch.linalg.inv(M1)
M2i = torch.linalg.inv(M2)
LC = torch.tensor(BASE.get("L_corr", [0,0,0]), device=dev, dtype=torch.float64)

D65 = torch.tensor([0.95047, 1.0, 1.08883], device=dev)
MS = torch.tensor([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],
                    [.0193339,.1191920,.9503041]], device=dev)
MSi = torch.linalg.inv(MS)

def s2l(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))
def l2s(c):
    return torch.where(c <= 0.0031308, c * 12.92,
                       1.055 * c.clamp(min=1e-12).pow(1./2.4) - 0.055)

# Gradient pairs
for _d in [os.path.join(ROOT, "colorbench"), os.path.join(ROOT, "space-test-project")]:
    if os.path.isdir(os.path.join(_d, "core")):
        sys.path.insert(0, _d)
        from core.pairs import generate_all_pairs
        PT, _ = generate_all_pairs(dev)
        print(f"Pairs: {PT.shape[0]}", flush=True)
        break

N_PAIRS = PT.shape[0]
N_ST = 25
T_ST = torch.linspace(0, 1, N_ST + 1, device=dev)

# Primaries for hue linearity
PRIM_SRGB = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]],
                          dtype=torch.float64, device=dev)
PRIM_XYZ = (s2l(PRIM_SRGB) @ MS.T)
WHITE_XYZ = D65.unsqueeze(0)

# Munsell
MUNSELL_Y = {1:0.01221,2:0.03126,3:0.06552,4:0.12000,5:0.19770,
             6:0.30049,7:0.43060,8:0.59100,9:0.78660}
MUNSELL_GRAYS = torch.stack([D65*MUNSELL_Y[v] for v in range(1,10)]).to(dev)

# ================================================================
#  FORWARD/INVERSE with hue correction
# ================================================================

def _base_fwd(xyz):
    """Base forward without hue correction. Returns (N, 3)."""
    lms = (xyz @ M1.T).clamp(min=0)
    lms_c = torch.sign(lms) * lms.abs().pow(1./3.)
    lab = lms_c @ M2.T
    L = lab[:, 0:1]
    c1, c2, c3 = LC[0], LC[1], LC[2]
    t = L * (1.0 - L)
    L_new = L + c1*t + c2*t*(2*L-1) + c3*L**2*(1-L)**2
    return torch.cat([L_new, lab[:, 1:2], lab[:, 2:3]], dim=1)

def _apply_hue_corr(lab, hue_cos, hue_sin):
    """Apply Fourier hue correction. hue_cos/sin: (P, 4)."""
    a, b = lab[..., 1], lab[..., 2]
    h = torch.atan2(b, a)  # (...,)
    C = (a**2 + b**2).sqrt()
    ks = torch.arange(1, 5, device=dev, dtype=torch.float64)  # [1,2,3,4]
    # delta_h = sum(cos_k * cos(k*h) + sin_k * sin(k*h))
    h_exp = h.unsqueeze(-1) * ks  # (..., 4)
    if hue_cos.dim() == 1:
        delta_h = (hue_cos * torch.cos(h_exp) + hue_sin * torch.sin(h_exp)).sum(-1)
    else:
        # Batched: hue_cos (P, 4), h_exp (P, N, 4)
        delta_h = (hue_cos.unsqueeze(1) * torch.cos(h_exp) +
                   hue_sin.unsqueeze(1) * torch.sin(h_exp)).sum(-1)
    h_new = h + delta_h
    a_new = C * torch.cos(h_new)
    b_new = C * torch.sin(h_new)
    return torch.stack([lab[..., 0], a_new, b_new], dim=-1)

def _undo_hue_corr(lab, hue_cos, hue_sin):
    """Newton iteration to invert hue correction."""
    a_out, b_out = lab[..., 1], lab[..., 2]
    h_out = torch.atan2(b_out, a_out)
    C = (a_out**2 + b_out**2).sqrt()
    ks = torch.arange(1, 5, device=dev, dtype=torch.float64)
    h_raw = h_out.clone()
    for _ in range(15):
        h_exp = h_raw.unsqueeze(-1) * ks
        if hue_cos.dim() == 1:
            delta = (hue_cos * torch.cos(h_exp) + hue_sin * torch.sin(h_exp)).sum(-1)
        else:
            delta = (hue_cos.unsqueeze(1) * torch.cos(h_exp) +
                     hue_sin.unsqueeze(1) * torch.sin(h_exp)).sum(-1)
        h_raw = h_out - delta
    a_raw = C * torch.cos(h_raw)
    b_raw = C * torch.sin(h_raw)
    return torch.stack([lab[..., 0], a_raw, b_raw], dim=-1)

# ================================================================
#  BATCHED FORWARD (P candidates)
# ================================================================

def fwd_batch(xyz, hue_cos, hue_sin):
    """xyz: (N, 3), hue_cos/sin: (P, 4) -> (P, N, 3)"""
    lab_base = _base_fwd(xyz)  # (N, 3)
    lab_exp = lab_base.unsqueeze(0).expand(hue_cos.shape[0], -1, -1)
    return _apply_hue_corr(lab_exp, hue_cos, hue_sin)

# ================================================================
#  METRICS
# ================================================================

def batch_hue_lin(hue_cos, hue_sin):
    P = hue_cos.shape[0]
    n_steps = 11
    t = torch.linspace(0, 1, n_steps, device=dev).view(1, 1, -1, 1)
    lab_p = fwd_batch(PRIM_XYZ, hue_cos, hue_sin)  # (P, 6, 3)
    lab_w = fwd_batch(WHITE_XYZ.expand(6, 3), hue_cos, hue_sin)
    labs = lab_p.unsqueeze(2) + t * (lab_w.unsqueeze(2) - lab_p.unsqueeze(2))
    h = torch.atan2(labs[..., 2], labs[..., 1])
    h_s = h[:, :, 0:1]; h_e = h[:, :, -1:]
    dh = h_e - h_s
    dh = torch.where(dh > math.pi, dh - 2*math.pi, dh)
    dh = torch.where(dh < -math.pi, dh + 2*math.pi, dh)
    t_lin = torch.linspace(0, 1, n_steps, device=dev).view(1, 1, -1)
    h_exp = h_s + t_lin * dh
    C = (labs[..., 1]**2 + labs[..., 2]**2).sqrt()
    h_diff = h - h_exp
    h_diff = torch.where(h_diff > math.pi, h_diff - 2*math.pi, h_diff)
    h_diff = torch.where(h_diff < -math.pi, h_diff + 2*math.pi, h_diff)
    mask = C > 0.01
    count = mask.float().sum(dim=(1, 2)).clamp(min=1)
    return ((h_diff * mask.float())**2).sum(dim=(1, 2)).sqrt() / count.sqrt() * (180/math.pi)

def batch_hue_rms(hue_cos, hue_sin):
    lab = fwd_batch(PRIM_XYZ, hue_cos, hue_sin)  # (P, 6, 3)
    h = torch.atan2(lab[:, :, 2], lab[:, :, 1]) * (180/math.pi) % 360
    expected = torch.tensor([0, 120, 240, 60, 180, 300], device=dev, dtype=torch.float64)
    dh = h - expected.unsqueeze(0)
    dh = torch.where(dh > 180, dh - 360, dh)
    dh = torch.where(dh < -180, dh + 360, dh)
    return (dh**2).mean(dim=1).sqrt()

def batch_munsell_v(hue_cos, hue_sin):
    lab = fwd_batch(MUNSELL_GRAYS, hue_cos, hue_sin)
    L = lab[:, :, 0]
    dL = L[:, 1:] - L[:, :-1]
    return dL.std(dim=1) / (dL.abs().mean(dim=1) + 1e-10) * 100

def batch_ach(hue_cos, hue_sin):
    t = torch.linspace(0.01, 0.99, 32, device=dev)
    grays = D65.unsqueeze(0) * t.unsqueeze(1)
    lab = fwd_batch(grays, hue_cos, hue_sin)
    return (lab[:, :, 1]**2 + lab[:, :, 2]**2).sqrt().max(dim=1).values

# ================================================================
#  EVALUATE
# ================================================================

def evaluate(x_np):
    x = torch.tensor(x_np, device=dev, dtype=torch.float64)
    P = x.shape[0]
    hue_cos = x[:, 0:4].clamp(-0.3, 0.3)
    hue_sin = x[:, 4:8].clamp(-0.3, 0.3)

    with torch.no_grad():
        hue_rms = batch_hue_rms(hue_cos, hue_sin)
        hue_lin = batch_hue_lin(hue_cos, hue_sin)
        munsell_v = batch_munsell_v(hue_cos, hue_sin)
        ach = batch_ach(hue_cos, hue_sin)

        # Primary: hue linearity
        loss = 2.0 * hue_lin + 1.0 * hue_rms

        # Don't break Munsell
        loss += 0.5 * munsell_v

        # Don't break achromatic
        loss += torch.where(ach > 0.001, (ach - 0.001) * 500, 0.0)

    return loss.cpu().numpy()

# ================================================================
#  SEEDS & RUN
# ================================================================

def make_seeds():
    seeds = [("zero", np.zeros(8), 0.05)]  # Start with no correction
    rng = np.random.RandomState(42)
    for i in range(args.seeds - 1):
        seeds.append((f"rnd{i}", rng.randn(8) * 0.02, 0.05))
    return seeds

def run_seed(label, x0, sigma):
    print(f"\n  Seed: {label}", flush=True)
    opts = cma.CMAOptions()
    opts.set("maxiter", args.gens); opts.set("popsize", args.pop)
    opts.set("tolfun", 1e-15); opts.set("tolx", 1e-15); opts.set("verbose", -1)
    es = cma.CMAEvolutionStrategy(x0, sigma, opts)
    best_loss, best_x = 999.0, x0.copy()
    gen = 0
    while not es.stop():
        sols = es.ask()
        fits = evaluate(np.array(sols))
        es.tell(sols, fits.tolist())
        idx = np.argmin(fits)
        if fits[idx] < best_loss:
            best_loss = fits[idx]; best_x = np.array(sols[idx]).copy()
        gen += 1
        if gen % 20 == 0 or gen <= 3:
            x_t = torch.tensor(best_x.reshape(1, 8), device=dev)
            hc, hs = x_t[:, :4].clamp(-0.3, 0.3), x_t[:, 4:].clamp(-0.3, 0.3)
            hr = batch_hue_rms(hc, hs).item()
            hl = batch_hue_lin(hc, hs).item()
            mv = batch_munsell_v(hc, hs).item()
            ac = batch_ach(hc, hs).item()
            print(f"  gen {gen:4d}  loss={best_loss:.2f}  HueRMS={hr:.1f} deg  "
                  f"HueLin={hl:.1f} deg  MunsV={mv:.1f}%  Ach={ac:.4f}", flush=True)

    # Save
    hc = np.clip(best_x[:4], -0.3, 0.3).tolist()
    hs = np.clip(best_x[4:], -0.3, 0.3).tolist()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = dict(BASE)
    out["hue_cos1"] = hc[0]; out["hue_cos2"] = hc[1]
    out["hue_cos3"] = hc[2]; out["hue_cos4"] = hc[3]
    out["hue_sin1"] = hs[0]; out["hue_sin2"] = hs[1]
    out["hue_sin3"] = hs[2]; out["hue_sin4"] = hs[3]
    out["architecture"] = "v7bblend30_achfix_huecorr"
    out["loss"] = float(best_loss)
    fname = f"blend30_huecorr_{label}_{ts}.json"
    path = os.path.join(CKPT, fname)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  FINAL: loss={best_loss:.2f} HueRMS={hr:.1f} HueLin={hl:.1f} MunsV={mv:.1f}%", flush=True)
    return best_loss, path

if __name__ == "__main__":
    print(f"Hue Correction: 8 params on top of v7bblend30_achfix", flush=True)
    seeds = make_seeds()
    results = []
    for label, x0, sigma in seeds:
        loss, path = run_seed(label, x0, sigma)
        results.append((label, loss, path))
    results.sort(key=lambda r: r[1])
    print(f"\nBest: {results[0][0]} (loss={results[0][1]:.2f})")
