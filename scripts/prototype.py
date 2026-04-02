"""HelmGen v2.0 Prototype — Log-Chroma Architecture.

Pipeline: M1 → depcubic(α) → M2 → log_chroma(k) → Lab

Log-chroma transform:
  Forward: C' = log(1 + k·C) / k,  a' = (C'/C)·a,  b' = (C'/C)·b
  Inverse: C = (exp(k·C') - 1) / k, a = (C/C')·a', b = (C/C')·b'

Properties:
- Compresses high chroma logarithmically → more uniform gradient steps
- Bounded derivative: d(C')/dC = 1/(1+kC) ≤ 1 → caps amplification
- Exact analytical inverse via exp
- L channel untouched (preserves Munsell)
- Neutrals exact (C=0 → C'=0)
"""

import numpy as np
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from helmlab.utils.srgb_convert import sRGB_to_XYZ

D65 = np.array([0.95047, 1.0, 1.08883])
XYZ_TO_SRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
])

PROJ = Path(__file__).parent.parent.parent
prod = json.loads((PROJ / "src/helmlab/data/gen_params.json").read_text())
M1 = np.array(prod["M1"])
M2 = np.array(prod["M2"])
M1_INV = np.linalg.inv(M1)
M2_INV = np.linalg.inv(M2)
ALPHA = prod.get("depcubic_alpha", 0.02)

PRIMARIES = [[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,0,1],[1,1,0]]


def depcubic_fwd(x, alpha):
    s = np.sqrt(alpha / 3.0)
    t = x / (2.0 * s**3)
    y = 2.0 * s * np.sinh(np.arcsinh(t) / 3.0)
    f = y**3 + alpha * y - x
    fp = 3.0 * y**2 + alpha
    fpp = 6.0 * y
    d = 2.0 * fp * fp - f * fpp
    mask = np.abs(d) > 1e-30
    y = np.where(mask, y - 2.0 * f * fp / np.where(mask, d, 1.0), y)
    return y


def depcubic_inv(y, alpha):
    return y**3 + alpha * y


def log_chroma_fwd(lab, k):
    """Log-chroma compression. C' = log(1+kC)/k. L untouched. Neutrals exact."""
    L, a, b = lab[0], lab[1], lab[2]
    C = np.sqrt(a**2 + b**2)
    if C < 1e-30:
        return lab.copy()
    C_new = np.log1p(k * C) / k
    scale = C_new / C
    return np.array([L, a * scale, b * scale])


def log_chroma_inv(lab, k):
    """Inverse: C = (exp(kC') - 1) / k. Exact analytical."""
    L, a, b = lab[0], lab[1], lab[2]
    C_prime = np.sqrt(a**2 + b**2)
    if C_prime < 1e-30:
        return lab.copy()
    C = (np.exp(k * C_prime) - 1.0) / k
    scale = C / C_prime
    return np.array([L, a * scale, b * scale])


def forward(xyz, k):
    lms = M1 @ xyz
    lms_c = depcubic_fwd(lms, ALPHA)
    lab = M2 @ lms_c
    if k > 1e-10:
        lab = log_chroma_fwd(lab, k)
    return lab


def inverse(lab, k):
    if k > 1e-10:
        lab = log_chroma_inv(lab, k)
    lms_c = M2_INV @ lab
    lms = depcubic_inv(lms_c, ALPHA)
    return M1_INV @ lms


def measure_quick(k, verbose=True):
    """Quick proxy metrics."""
    results = {}

    # RT
    np.random.seed(42)
    max_rt = 0
    for _ in range(3000):
        rgb = np.random.rand(3)
        xyz = sRGB_to_XYZ(rgb)
        lab = forward(xyz, k)
        xyz2 = inverse(lab, k)
        max_rt = max(max_rt, np.max(np.abs(xyz - xyz2)))
    results["rt"] = max_rt

    # Blue G/R
    b_xyz = sRGB_to_XYZ(np.array([0,0,1.0]))
    w_xyz = sRGB_to_XYZ(np.array([1,1,1.0]))
    lab_b = forward(b_xyz, k)
    lab_w = forward(w_xyz, k)
    lab_mid = (lab_b + lab_w) / 2
    xyz_mid = inverse(lab_mid, k)
    rgb_mid = np.clip(XYZ_TO_SRGB @ xyz_mid, 0, 1)
    results["bgr"] = rgb_mid[1] / max(rgb_mid[0], 1e-10)

    # Gradient CV proxy (10 representative pairs, 26 steps)
    pairs = [
        ([1,0,0], [0,0,1]),  # R→B
        ([1,0,0], [1,1,1]),  # R→W
        ([0,0,1], [1,1,1]),  # B→W
        ([0,1,0], [1,1,1]),  # G→W
        ([1,1,0], [1,1,1]),  # Y→W
        ([1,0,0], [0,1,0]),  # R→G
        ([0,1,1], [1,0,0]),  # C→R
        ([0.5,0,0], [0,0,0.5]),  # dark R→B
        ([0.8,0.6,0.4], [0.4,0.6,0.8]),  # pastel
        ([0,0,0], [1,1,1]),  # K→W
    ]
    cvs = []
    for rgb1, rgb2 in pairs:
        xyz1 = sRGB_to_XYZ(np.array(rgb1, dtype=float))
        xyz2 = sRGB_to_XYZ(np.array(rgb2, dtype=float))
        lab1 = forward(xyz1, k)
        lab2 = forward(xyz2, k)

        # 26 steps, linear lerp in Lab
        steps_lab = [lab1 + t * (lab2 - lab1) for t in np.linspace(0, 1, 26)]
        # Convert back to XYZ → sRGB 8-bit → XYZ → CIE Lab → CIEDE2000
        des = []
        for i in range(len(steps_lab) - 1):
            xyz_a = inverse(steps_lab[i], k)
            xyz_b = inverse(steps_lab[i+1], k)
            # Simplified CIE Lab
            def xyz_to_cielab(xyz):
                r = xyz / D65
                f = np.where(r > 0.008856, r**(1/3), 7.787 * r + 16/116)
                return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])
            clab_a = xyz_to_cielab(np.maximum(xyz_a, 1e-10))
            clab_b = xyz_to_cielab(np.maximum(xyz_b, 1e-10))
            de = np.sqrt(np.sum((clab_a - clab_b)**2))
            des.append(de)
        des = np.array(des)
        if des.mean() > 0.01:
            cvs.append(des.std() / des.mean())
    results["cv_proxy"] = np.mean(cvs) * 100

    # Achromatic
    max_ach = 0
    for v in np.linspace(0.01, 1.0, 50):
        xyz = sRGB_to_XYZ(np.array([v, v, v]))
        lab = forward(xyz, k)
        max_ach = max(max_ach, abs(lab[1]) + abs(lab[2]))
    results["ach"] = max_ach

    # Chroma amp proxy
    max_amp = 0
    for rgb in PRIMARIES:
        xyz = sRGB_to_XYZ(np.array(rgb, dtype=float))
        lab = forward(xyz, k)
        for dx in [0.001]:
            for dim in range(3):
                rgb2 = np.array(rgb, dtype=float)
                rgb2[dim] = min(rgb2[dim] + dx, 1.0)
                xyz2 = sRGB_to_XYZ(rgb2)
                lab2 = forward(xyz2, k)
                dlab = np.sqrt(np.sum((lab - lab2)**2))
                dxyz = np.sqrt(np.sum((xyz - xyz2)**2))
                if dxyz > 1e-10:
                    max_amp = max(max_amp, dlab / dxyz)
    results["amp"] = max_amp

    if verbose:
        print(f"  k={k:.3f}: RT={results['rt']:.2e}  G/R={results['bgr']:.3f}  "
              f"CV≈{results['cv_proxy']:.1f}%  Ach={results['ach']:.2e}  Amp={results['amp']:.1f}x")

    return results


def main():
    print("=" * 60)
    print("HelmGen v2.0 — Log-Chroma Scan")
    print("Pipeline: M1 → depcubic → M2 → log_chroma(k) → Lab")
    print("=" * 60)

    print("\nBaseline (k=0, no log-chroma = production):")
    r0 = measure_quick(0.0)

    print("\nLog-chroma k scan:")
    best_k = 0
    best_score = 0
    for k in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
        r = measure_quick(k)
        # Score: lower CV is better, but must keep G/R≥1.50 and RT<1e-14
        if r["bgr"] >= 1.50 and r["rt"] < 1e-12:
            score = (40 - r["cv_proxy"])  # higher = better
            if score > best_score:
                best_score = score
                best_k = k

    print(f"\nBest k: {best_k} (CV improvement: {best_score:.1f}%)")
    if best_k > 0:
        print(f"\nBest detailed:")
        measure_quick(best_k)


if __name__ == "__main__":
    main()
