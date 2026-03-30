"""Optimize v7b + hue-dependent rotation + chroma enrichment.

Architecture search winner: cbrt + v7b + L_corr + chroma + hue_rot_dep
Best CV=0.361 out of 2058 variants tested.

Previous attempt with hue_rot_dep FAILED because Fourier amplitudes were
too large (c1=0.5, s1=-0.5) → inverse diverged on 47/1000 colors.
This time: amplitude clamped to ±0.15.

Pipeline: M1(fixed) → cbrt → M2(free ab-rows) → chroma(free) → hue_rot_dep(free, clamped) → L_corr(free)

Free params (17):
  M2 ab-rows:    6  (L-row fixed for white point)
  L_corr:        3  (p1, p2, p3)
  hue_rot_dep:   4  (c1, s1, c2, s2 — Fourier, amplitude < 0.15)
  chroma_power:  1
  chroma_k:      1  (L-dependent)
  --
  spare:         2  more rotation harmonics if needed

Total: 15 free params
"""

import json
import sys
import time
import os
import numpy as np
import torch
import cma

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from colorbench.core.spaces import ColorSpace, OKLab
from colorbench.core import gpu_metrics, gpu_metrics_advanced, gpu_metrics_perceptual
from colorbench.core import pairs as pairs_mod
from colorbench.core.comparison import METRIC_DEFS, _extract_score

# Load v7b base
with open("checkpoints/v7b_nodelta.json") as f:
    BASE = json.load(f)

M1 = torch.tensor(BASE["M1"], device=device, dtype=torch.float64)
M1_inv = torch.linalg.inv(M1)
M2_base = torch.tensor(BASE["M2"], device=device, dtype=torch.float64)
M2_L_ROW = M2_base[0].cpu().tolist()  # Fixed L row

# Amplitude limit for hue rotation (CRITICAL — inverse diverges above ~0.2)
HUE_ROT_MAX = 0.15

print(f"M1 cond: {float(torch.linalg.cond(M1)):.1f}")
print(f"M2 cond: {float(torch.linalg.cond(M2_base)):.1f}")
print(f"Hue rotation amplitude limit: ±{HUE_ROT_MAX}")


class V7bHueRotDep(ColorSpace):
    """v7b + chroma enrichment + hue-dependent rotation + L correction."""
    name = "V7bHueRotDep"

    def __init__(self, M2, L_corr, hue_fourier, cp, lk):
        self.M1 = M1
        self.M1_inv = M1_inv
        self.M2 = M2
        self.M2_inv = torch.linalg.inv(M2)
        self.L_corr = L_corr      # [p1, p2, p3]
        self.hue_f = hue_fourier  # [c1, s1, c2, s2]
        self.cp = cp              # chroma power
        self.lk = lk              # L-dep chroma k

    def forward(self, xyz):
        # M1 → cbrt → M2
        lms = xyz @ self.M1.T
        lms_c = torch.sign(lms) * torch.abs(lms).pow(1.0 / 3.0)
        raw = lms_c @ self.M2.T
        L, a, b = raw[:, 0], raw[:, 1], raw[:, 2]

        # Chroma enrichment
        if abs(self.cp - 1.0) > 1e-10 or abs(self.lk) > 1e-10:
            C = torch.sqrt(a * a + b * b + 1e-30)
            scale = torch.ones_like(C)
            if abs(self.cp - 1.0) > 1e-10:
                scale = scale * C.pow(self.cp - 1.0)
            if abs(self.lk) > 1e-10:
                scale = scale * torch.exp(self.lk * (L - 0.5))
            a = a * scale
            b = b * scale

        # Hue-dependent rotation (Fourier)
        c1, s1, c2, s2 = self.hue_f
        if abs(c1) + abs(s1) + abs(c2) + abs(s2) > 1e-10:
            h = torch.atan2(b, a)
            dh = (c1 * torch.cos(h) + s1 * torch.sin(h)
                  + c2 * torch.cos(2 * h) + s2 * torch.sin(2 * h))
            cos_dh = torch.cos(dh)
            sin_dh = torch.sin(dh)
            a_new = a * cos_dh - b * sin_dh
            b_new = a * sin_dh + b * cos_dh
            a, b = a_new, b_new

        # L correction
        p1, p2, p3 = self.L_corr
        if abs(p1) + abs(p2) + abs(p3) > 1e-15:
            t = L * (1.0 - L)
            L = L + p1 * t + p2 * t * (2.0 * L - 1.0) + p3 * t * t

        return torch.stack([L, a, b], dim=-1)

    def inverse(self, lab):
        L_out, a_out, b_out = lab[:, 0], lab[:, 1], lab[:, 2]

        # Undo L correction (Newton)
        p1, p2, p3 = self.L_corr
        L = L_out.clone()
        if abs(p1) + abs(p2) + abs(p3) > 1e-15:
            for _ in range(15):
                t = L * (1.0 - L)
                dt = 1.0 - 2.0 * L
                f = L + p1 * t + p2 * t * (2.0 * L - 1.0) + p3 * t * t - L_out
                df = 1.0 + p1 * dt + p2 * (dt * (2.0 * L - 1.0) + t * 2.0) + p3 * 2.0 * t * dt
                df_safe = torch.where(df.abs() < 1e-12, torch.ones_like(df), df)
                L = L - f / df_safe

        a, b = a_out.clone(), b_out.clone()

        # Undo hue rotation (fixed-point iteration)
        c1, s1, c2, s2 = self.hue_f
        if abs(c1) + abs(s1) + abs(c2) + abs(s2) > 1e-10:
            for _ in range(12):
                h = torch.atan2(b, a)
                dh = (c1 * torch.cos(h) + s1 * torch.sin(h)
                      + c2 * torch.cos(2 * h) + s2 * torch.sin(2 * h))
                cos_dh = torch.cos(-dh)
                sin_dh = torch.sin(-dh)
                a = a_out * cos_dh - b_out * sin_dh
                b = a_out * sin_dh + b_out * cos_dh

        # Undo chroma enrichment
        if abs(self.cp - 1.0) > 1e-10 or abs(self.lk) > 1e-10:
            C_out = torch.sqrt(a * a + b * b + 1e-30)
            inv_scale = torch.ones_like(C_out)
            if abs(self.lk) > 1e-10:
                inv_scale = inv_scale * torch.exp(-self.lk * (L - 0.5))
            if abs(self.cp - 1.0) > 1e-10:
                C_mid = C_out * inv_scale
                C_orig = C_mid.pow(1.0 / self.cp)
                total_inv = C_orig / C_out.clamp(min=1e-30)
            else:
                total_inv = inv_scale
            a = a * total_inv
            b = b * total_inv

        # Undo M2 → cube → M1
        raw = torch.stack([L, a, b], dim=-1)
        lms_c = raw @ self.M2_inv.T
        lms = torch.sign(lms_c) * lms_c.abs().pow(3.0)
        return lms @ self.M1_inv.T


# Pre-generate pairs
print("Generating gradient pairs...")
pairs_xyz, pair_labels = pairs_mod.generate_all_pairs(device)

# Pre-compute OKLab baseline
print("Computing OKLab baseline...")
oklab = OKLab(device)

# Run all metrics for OKLab (once)
ok_results = {}
ok_results["gradients"] = gpu_metrics.measure_gradients(oklab, pairs_xyz, pair_labels, device)
ok_results["gamut"] = gpu_metrics.measure_gamut(oklab, device)
ok_results["hue"] = gpu_metrics.measure_hue(oklab, device)
ok_results["specials"] = gpu_metrics.measure_special_gradients(oklab, device)
ok_results["cvd"] = gpu_metrics_advanced.measure_cvd(oklab, device)
ok_results["hue_leaf"] = gpu_metrics_advanced.measure_hue_leaf(oklab, device)
ok_results["jacobian"] = gpu_metrics_advanced.measure_jacobian(oklab, device)
ok_results["banding"] = gpu_metrics_advanced.measure_perceptual_banding(oklab, device)
ok_results["animation"] = gpu_metrics_advanced.measure_animation(oklab, device)
ok_results["3color"] = gpu_metrics_advanced.measure_3color_gradients(oklab, device)
ok_results["munsell_value"] = gpu_metrics_perceptual.measure_munsell_value(oklab, device)
ok_results["munsell_hue"] = gpu_metrics_perceptual.measure_munsell_hue(oklab, device)
ok_results["macadam_isotropy"] = gpu_metrics_perceptual.measure_macadam_isotropy(oklab, device)
ok_results["hue_agreement"] = gpu_metrics_perceptual.measure_hue_agreement(oklab, device)
ok_results["palette_uniformity"] = gpu_metrics_perceptual.measure_palette_uniformity(oklab, device)
ok_results["tint_shade_hue"] = gpu_metrics_perceptual.measure_tint_shade_hue(oklab, device)
ok_results["dataviz_distinguish"] = gpu_metrics_perceptual.measure_dataviz_distinguishability(oklab, device)
ok_results["multistop_gradient"] = gpu_metrics_perceptual.measure_multistop_gradient(oklab, device)
ok_results["wcag_midpoint"] = gpu_metrics_perceptual.measure_wcag_midpoint_contrast(oklab, device)
ok_results["harmony_accuracy"] = gpu_metrics_perceptual.measure_harmony_accuracy(oklab, device)
ok_results["photo_gamut_map"] = gpu_metrics_perceptual.measure_photo_gamut_map(oklab, device)
ok_results["eased_animation"] = gpu_metrics_perceptual.measure_eased_animation(oklab, device)
ok_results["shade_hue_consistency"] = gpu_metrics_perceptual.measure_shade_hue_consistency(oklab, device)
ok_results["chroma_preservation"] = gpu_metrics_perceptual.measure_chroma_preservation(oklab, device)

# Extract OKLab scores
EVAL_KEYS = list(ok_results.keys())  # Which metric groups to run per candidate
ok_scores = {}
for mdef in METRIC_DEFS:
    s = _extract_score(ok_results, mdef.result_key, mdef.score_path)
    if s is not None:
        ok_scores[mdef.name] = (s, mdef.lower_is_better)

print(f"OKLab: {len(ok_scores)} metric scores cached")

# Metric functions (same as baseline, run per candidate)
def eval_candidate(space):
    results = {}
    try:
        results["gradients"] = gpu_metrics.measure_gradients(space, pairs_xyz, pair_labels, device)
        results["gamut"] = gpu_metrics.measure_gamut(space, device)
        results["hue"] = gpu_metrics.measure_hue(space, device)
        results["specials"] = gpu_metrics.measure_special_gradients(space, device)
        results["cvd"] = gpu_metrics_advanced.measure_cvd(space, device)
        results["hue_leaf"] = gpu_metrics_advanced.measure_hue_leaf(space, device)
        results["jacobian"] = gpu_metrics_advanced.measure_jacobian(space, device)
        results["banding"] = gpu_metrics_advanced.measure_perceptual_banding(space, device)
        results["animation"] = gpu_metrics_advanced.measure_animation(space, device)
        results["3color"] = gpu_metrics_advanced.measure_3color_gradients(space, device)
        results["munsell_value"] = gpu_metrics_perceptual.measure_munsell_value(space, device)
        results["munsell_hue"] = gpu_metrics_perceptual.measure_munsell_hue(space, device)
        results["macadam_isotropy"] = gpu_metrics_perceptual.measure_macadam_isotropy(space, device)
        results["hue_agreement"] = gpu_metrics_perceptual.measure_hue_agreement(space, device)
        results["palette_uniformity"] = gpu_metrics_perceptual.measure_palette_uniformity(space, device)
        results["tint_shade_hue"] = gpu_metrics_perceptual.measure_tint_shade_hue(space, device)
        results["dataviz_distinguish"] = gpu_metrics_perceptual.measure_dataviz_distinguishability(space, device)
        results["multistop_gradient"] = gpu_metrics_perceptual.measure_multistop_gradient(space, device)
        results["wcag_midpoint"] = gpu_metrics_perceptual.measure_wcag_midpoint_contrast(space, device)
        results["harmony_accuracy"] = gpu_metrics_perceptual.measure_harmony_accuracy(space, device)
        results["photo_gamut_map"] = gpu_metrics_perceptual.measure_photo_gamut_map(space, device)
        results["eased_animation"] = gpu_metrics_perceptual.measure_eased_animation(space, device)
        results["shade_hue_consistency"] = gpu_metrics_perceptual.measure_shade_hue_consistency(space, device)
        results["chroma_preservation"] = gpu_metrics_perceptual.measure_chroma_preservation(space, device)
    except Exception as e:
        print(f"    eval error: {e}")
    return results


def count_wins(results):
    """Count wins vs OKLab."""
    wins = losses = ties = 0
    for mdef in METRIC_DEFS:
        if mdef.name not in ok_scores:
            continue
        ok_val, lower = ok_scores[mdef.name]
        my_val = _extract_score(results, mdef.result_key, mdef.score_path)
        if my_val is None:
            losses += 1
            continue
        if ok_val != 0:
            rel = abs(my_val - ok_val) / (abs(ok_val) + 1e-30)
            if rel <= 0.01:
                ties += 1
                continue
        if lower:
            if my_val < ok_val:
                wins += 1
            else:
                losses += 1
        else:
            if my_val > ok_val:
                wins += 1
            else:
                losses += 1
    return wins, losses, ties


# ── CMA-ES ──────────────────────────────────────────────────────

# Parameter layout (15 free):
# [0:3]   M2 ab-row1
# [3:6]   M2 ab-row2
# [6:9]   L_corr [p1, p2, p3]
# [9:13]  hue_rot [c1, s1, c2, s2]
# [13]    chroma_power
# [14]    chroma_k (L-dep)

x0 = (
    M2_base[1].cpu().tolist() +
    M2_base[2].cpu().tolist() +
    [0.0, 0.0, 0.0] +           # L_corr start from zero
    [0.0, 0.0, 0.0, 0.0] +      # hue_rot start from zero
    [1.0] +                       # chroma_power (no change)
    [0.0]                         # chroma_k (no L-dep)
)

best_wins = 0
best_wl = (0, 999, 0)  # (wins, losses, ties)
best_x = None

def objective(x):
    global best_wins, best_wl, best_x

    M2_ab1 = list(x[0:3])
    M2_ab2 = list(x[3:6])
    L_corr = list(x[6:9])
    hue_f = list(x[9:13])
    cp = float(x[13])
    lk = float(x[14])

    # Bounds
    if cp < 0.5 or cp > 1.0:
        return 100.0
    if abs(lk) > 1.5:
        return 100.0

    # CRITICAL: clamp hue rotation amplitude
    hue_f = [max(-HUE_ROT_MAX, min(HUE_ROT_MAX, h)) for h in hue_f]

    # Clamp L_corr to reasonable range
    L_corr = [max(-0.3, min(0.3, c)) for c in L_corr]

    M2 = torch.tensor([M2_L_ROW, M2_ab1, M2_ab2], device=device, dtype=torch.float64)

    # Check M2 invertibility
    try:
        det = float(torch.linalg.det(M2).abs())
        if det < 1e-6:
            return 100.0
    except:
        return 100.0

    # Check achromatic + white point
    D65 = torch.tensor([[0.95047, 1.0, 1.08883]], device=device, dtype=torch.float64)
    try:
        space = V7bHueRotDep(M2, L_corr, hue_f, cp, lk)
        white_lab = space.forward(D65)
        white_L = float(white_lab[0, 0])
        white_ab = float(torch.sqrt(white_lab[0, 1]**2 + white_lab[0, 2]**2))
        if white_L < 0.9 or white_L > 1.1 or white_ab > 0.01:
            return 100.0
    except:
        return 100.0

    # Quick RT check (50 colors)
    try:
        torch.manual_seed(42)
        M_srgb = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ], device=device, dtype=torch.float64)
        test_rgb = torch.rand(50, 3, device=device, dtype=torch.float64)
        test_xyz = test_rgb @ M_srgb.T
        test_lab = space.forward(test_xyz)
        test_rt = space.inverse(test_lab)
        rt_err = float((test_xyz - test_rt).abs().max())
        if rt_err > 1e-4:
            return 90.0 + rt_err  # Penalize but don't kill
    except:
        return 100.0

    # Full metric eval
    try:
        results = eval_candidate(space)
    except:
        return 100.0

    # Count wins
    w, l, t = count_wins(results)

    # Loss: maximize wins, minimize losses
    margin_penalty = 0.0
    for mdef in METRIC_DEFS:
        if mdef.name not in ok_scores:
            continue
        ok_val, lower = ok_scores[mdef.name]
        my_val = _extract_score(results, mdef.result_key, mdef.score_path)
        if my_val is None:
            continue
        if lower and my_val > ok_val:
            margin_penalty += (my_val - ok_val) / (abs(ok_val) + 1e-30)
        elif not lower and my_val < ok_val:
            margin_penalty += (ok_val - my_val) / (abs(ok_val) + 1e-30)

    loss = -w * 3.0 + l * 2.0 + margin_penalty * 0.05 + rt_err * 1000

    if w > best_wins or (w == best_wins and l < best_wl[1]):
        best_wins = w
        best_wl = (w, l, t)
        best_x = [float(v) for v in x]
        print(f"  NEW BEST: {w}-{l} ({t} tie) RT={rt_err:.2e}")

        # Save checkpoint
        ckpt = {
            "M1": BASE["M1"],
            "M2": [M2_L_ROW, [float(v) for v in M2_ab1], [float(v) for v in M2_ab2]],
            "gamma": [1/3, 1/3, 1/3],
            "L_corr": [float(v) for v in L_corr],
            "hue_correction": [float(v) for v in hue_f],
            "chroma_power": float(cp),
            "chroma_k": float(lk),
            "architecture": "v7b_cbrt_huerotdep_chroma_lcorr",
            "wins_vs_oklab": w,
            "losses_vs_oklab": l,
            "rt_err": float(rt_err),
        }
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"checkpoints/v7b_hrd_{w}w_{ts}.json"
        with open(path, "w") as f:
            json.dump(ckpt, f, indent=2)
        print(f"  Saved: {path}")

    return loss


# Run
print(f"\nCMA-ES: 15 params, pop=64, gen=200")
print(f"Pipeline: M1(fixed) → cbrt → M2(free ab) → chroma → hue_rot_dep → L_corr")
print(f"Hue rotation amplitude limit: ±{HUE_ROT_MAX}\n")

opts = cma.CMAOptions()
opts["maxiter"] = 200
opts["popsize"] = 64
opts["tolfun"] = 1e-12
opts["tolx"] = 1e-12
opts["verbose"] = -1

es = cma.CMAEvolutionStrategy(x0, 0.03, opts)

t0 = time.time()
gen = 0
while not es.stop():
    X = es.ask()
    losses = [objective(x) for x in X]
    es.tell(X, losses)
    gen += 1
    if gen % 5 == 0:
        elapsed = time.time() - t0
        rate = gen / elapsed * 60
        print(f"  gen {gen}: best_loss={min(losses):.2f}, "
              f"best={best_wl[0]}-{best_wl[1]}, "
              f"{rate:.0f} gen/min")

elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"DONE: {elapsed:.0f}s ({gen} gen)")
print(f"Best: {best_wl[0]}-{best_wl[1]} ({best_wl[2]} tie) vs OKLab")

# Final: run full ColorBench on best model
if best_x is not None:
    print(f"\nRunning full ColorBench on best model...")
    M2_best = torch.tensor([M2_L_ROW, best_x[0:3], best_x[3:6]],
                           device=device, dtype=torch.float64)
    best_space = V7bHueRotDep(M2_best, best_x[6:9], best_x[9:13],
                               best_x[13], best_x[14])
    best_space.name = f"V7b_HRD_{best_wl[0]}w"

    # Find latest checkpoint
    import glob
    latest = sorted(glob.glob("checkpoints/v7b_hrd_*w_*.json"))[-1]
    print(f"Best checkpoint: {latest}")
    os.system(f"python3 colorbench/run.py oklab genenriched --json {latest} 2>&1 | tail -30")
