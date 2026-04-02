"""Rational Transfer Function Experiment.

f(x) = x · (a + b·x) / (1 + c·x)

Properties:
- f(0) = 0, f'(0) = a
- f'(∞) → b/c (bounded!)
- Monotone if a > 0, b > 0, c > 0, a·c ≥ b (sufficient)
- Inverse: quadratic → exact analytical
- Zero gamut fold guarantee (bounded derivative)

Inverse derivation:
  y = x(a + bx) / (1 + cx)
  y + cxy = ax + bx²
  bx² + (a - cy)x - y = 0
  x = [-(a-cy) + sqrt((a-cy)² + 4by)] / (2b)

Pipeline: M1 → rational(a,b,c) → M2 → Lab
"""

import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from helmlab.utils.srgb_convert import sRGB_to_XYZ

D65 = np.array([0.95047, 1.0, 1.08883])
XYZ_TO_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [0.0556434, -0.2040259,  1.0572252]
])

OKLAB_M1 = np.array([
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715,  0.0361456387],
    [0.0482003018, 0.2643662691,  0.633851707]
])
OKLAB_M2 = np.array([
    [0.2104542553,  0.793617785, -0.0040720468],
    [1.9779984951, -2.428592205,  0.4505937099],
    [0.0259040371, 0.7827717662, -0.808675766]
])

PRIMARIES = [[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,0,1],[1,1,0]]


def rational_fwd(x, a, b, c):
    """f(x) = x·(a + b·x) / (1 + c·x). Element-wise, handles negative x."""
    sign = np.sign(x)
    ax = np.abs(x)
    y = ax * (a + b * ax) / (1.0 + c * ax)
    return sign * y


def rational_inv(y, a, b, c):
    """Inverse: quadratic formula. x = [-(a-cy) + sqrt((a-cy)²+4by)] / (2b)."""
    sign = np.sign(y)
    ay = np.abs(y)
    if b < 1e-30:
        # Linear case: y = ax/(1+cx), x = y/(a-cy)
        return sign * ay / np.maximum(a - c * ay, 1e-30)
    disc = (a - c * ay)**2 + 4.0 * b * ay
    disc = np.maximum(disc, 0.0)
    x = (-(a - c * ay) + np.sqrt(disc)) / (2.0 * b)
    return sign * x


def measure(M1, M2, a, b, c, verbose=True):
    """Quick metrics: RT, holes, cusps, G/R, chroma amp, gradient CV proxy."""
    M1_inv = np.linalg.inv(M1)
    M2_inv = np.linalg.inv(M2)
    results = {}

    # RT
    np.random.seed(42)
    max_rt = 0
    for _ in range(2000):
        rgb = np.random.rand(3)
        xyz = sRGB_to_XYZ(rgb)
        lms = M1 @ xyz
        lms_c = rational_fwd(lms, a, b, c)
        lab = M2 @ lms_c
        # Inverse
        lms_c2 = M2_inv @ lab
        lms2 = rational_inv(lms_c2, a, b, c)
        xyz2 = M1_inv @ lms2
        max_rt = max(max_rt, np.max(np.abs(xyz - xyz2)))
    results['rt'] = max_rt

    # Achromatic
    max_ach = 0
    for v in np.linspace(0.01, 1.0, 50):
        xyz = sRGB_to_XYZ(np.array([v, v, v]))
        lms = M1 @ xyz
        lms_c = rational_fwd(lms, a, b, c)
        lab = M2 @ lms_c
        max_ach = max(max_ach, abs(lab[1]) + abs(lab[2]))
    results['ach'] = max_ach

    # Cusps (coarse)
    valid = 0
    for hue_deg in range(0, 360, 10):
        h = np.radians(hue_deg)
        ch, sh = np.cos(h), np.sin(h)
        best_C, best_L = 0, 0
        for L in np.arange(0.05, 0.95, 0.02):
            lo, hi = 0.0, 0.5
            for _ in range(16):
                mid = (lo + hi) / 2
                lab = np.array([L, mid * ch, mid * sh])
                lms_c = M2_inv @ lab
                lms = rational_inv(lms_c, a, b, c)
                xyz = M1_inv @ lms
                rgb = XYZ_TO_SRGB @ xyz
                if np.all(rgb >= -0.001) and np.all(rgb <= 1.001):
                    lo = mid
                else:
                    hi = mid
            if lo > best_C:
                best_C, best_L = lo, L
        if 0.05 < best_L < 0.99:
            valid += 1
    results['cusps_36'] = valid

    # Blue G/R
    b_xyz = sRGB_to_XYZ(np.array([0, 0, 1.0]))
    w_xyz = sRGB_to_XYZ(np.array([1, 1, 1.0]))
    lab_b = M2 @ rational_fwd(M1 @ b_xyz, a, b, c)
    lab_w = M2 @ rational_fwd(M1 @ w_xyz, a, b, c)
    lab_mid = (lab_b + lab_w) / 2
    lms_c_mid = M2_inv @ lab_mid
    lms_mid = rational_inv(lms_c_mid, a, b, c)
    xyz_mid = M1_inv @ lms_mid
    rgb_mid = np.clip(XYZ_TO_SRGB @ xyz_mid, 0, 1)
    results['bgr'] = rgb_mid[1] / max(rgb_mid[0], 1e-10)

    # Chroma amp proxy
    max_amp = 0
    for rgb in PRIMARIES:
        xyz = sRGB_to_XYZ(np.array(rgb, dtype=float))
        lab = M2 @ rational_fwd(M1 @ xyz, a, b, c)
        for dx in [0.001]:
            for dim in range(3):
                rgb2 = np.array(rgb, dtype=float)
                rgb2[dim] = min(rgb2[dim] + dx, 1.0)
                xyz2 = sRGB_to_XYZ(rgb2)
                lab2 = M2 @ rational_fwd(M1 @ xyz2, a, b, c)
                dlab = np.sqrt(np.sum((lab - lab2)**2))
                dxyz = np.sqrt(np.sum((xyz - xyz2)**2))
                if dxyz > 1e-10:
                    max_amp = max(max_amp, dlab / dxyz)
    results['amp'] = max_amp

    # Gamut holes at h=264° (blue region)
    holes_264 = 0
    h = np.radians(264)
    ch, sh = np.cos(h), np.sin(h)
    for L_int in range(10, 90):
        L = L_int / 100
        prev_ok = None
        for C_int in range(5, 400):
            C = C_int * 0.001
            lab = np.array([L, C * ch, C * sh])
            lms_c = M2_inv @ lab
            lms = rational_inv(lms_c, a, b, c)
            xyz = M1_inv @ lms
            rgb = XYZ_TO_SRGB @ xyz
            ok = np.all(rgb >= -0.001) and np.all(rgb <= 1.001)
            if prev_ok is not None and not prev_ok and ok:
                holes_264 += 1
            prev_ok = ok if not ok else True
    results['holes_264'] = holes_264

    # Gradient CV proxy (5 pairs)
    pairs = [([1,0,0],[0,0,1]), ([0,0,1],[1,1,1]), ([1,0,0],[1,1,1]),
             ([0,1,0],[1,1,1]), ([1,0,0],[0,1,0])]
    cvs = []
    for rgb1, rgb2 in pairs:
        xyz1 = sRGB_to_XYZ(np.array(rgb1, dtype=float))
        xyz2 = sRGB_to_XYZ(np.array(rgb2, dtype=float))
        lab1 = M2 @ rational_fwd(M1 @ xyz1, a, b, c)
        lab2 = M2 @ rational_fwd(M1 @ xyz2, a, b, c)
        des = []
        for i in range(25):
            t1, t2 = i/25, (i+1)/25
            p1 = lab1 + t1*(lab2-lab1)
            p2 = lab1 + t2*(lab2-lab1)
            x1 = M1_inv @ rational_inv(M2_inv @ p1, a, b, c)
            x2 = M1_inv @ rational_inv(M2_inv @ p2, a, b, c)
            r1 = x1/D65; f1 = np.where(r1>0.008856, r1**(1/3), 7.787*r1+16/116)
            r2 = x2/D65; f2 = np.where(r2>0.008856, r2**(1/3), 7.787*r2+16/116)
            cl1 = np.array([116*f1[1]-16, 500*(f1[0]-f1[1]), 200*(f1[1]-f1[2])])
            cl2 = np.array([116*f2[1]-16, 500*(f2[0]-f2[1]), 200*(f2[1]-f2[2])])
            de = np.sqrt(np.sum((cl1-cl2)**2))
            des.append(de)
        des = np.array(des)
        if des.mean() > 0.01:
            cvs.append(des.std()/des.mean())
    results['cv_proxy'] = np.mean(cvs)*100

    if verbose:
        bgr_s = '✓' if results['bgr'] >= 1.50 else '✗'
        amp_s = '✓' if results['amp'] < 3.0 else '✗'
        rt_s = '✓' if results['rt'] < 1e-14 else '✗'
        print(f"  a={a:.2f} b={b:.3f} c={c:.3f}: "
              f"RT={results['rt']:.1e}{rt_s} "
              f"BGR={results['bgr']:.2f}{bgr_s} "
              f"Amp={results['amp']:.1f}x{amp_s} "
              f"Cusps={results['cusps_36']}/36 "
              f"Holes264={results['holes_264']} "
              f"CV≈{results['cv_proxy']:.1f}% "
              f"Ach={results['ach']:.1e}")
    return results


def main():
    print("=" * 70)
    print("Rational Transfer Experiment")
    print("f(x) = x·(a + b·x) / (1 + c·x)")
    print("=" * 70)

    M1 = OKLAB_M1
    M2 = OKLAB_M2

    # Baseline: cbrt equivalent
    # cbrt ≈ rational with a→∞, special case
    # Let's find rational params that approximate cbrt behavior
    # For cbrt: f(1) = 1, f'(0) = ∞
    # Rational: f(1) = (a+b)/(1+c) = 1, f'(0) = a

    print("\n--- cbrt reference (OKLab baseline) ---")
    # cbrt can't be represented as rational, test depcubic instead
    print("  (Using depcubic α=0.02 as reference)")

    print("\n--- Rational transfer scan ---")
    print("Searching for: RT<1e-14, BGR≥1.50, Amp<3x, Cusps=36/36, Holes=0\n")

    # Strategy: a controls near-zero slope (like 1/α in depcubic)
    # b/c controls large-x behavior
    # Need f(1) ≈ 1 for normalization: (a+b)/(1+c) ≈ 1
    # So b ≈ 1 + c - a

    best_score = 0
    best_params = None

    for a in [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
        for c in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
            b = 1.0 + c - a  # normalize f(1)≈1
            if b <= 0:
                continue
            # Check monotonicity: need a + 2bx > 0 and denominator positive
            # Sufficient: a > 0, b > 0
            r = measure(M1, M2, a, b, c, verbose=True)
            # Score: prioritize low holes + high cusps + low amp
            score = (r['cusps_36'] / 36) * 100 - r['holes_264'] * 10 - max(0, r['amp'] - 3.0) * 20
            if score > best_score and r['rt'] < 1e-10:
                best_score = score
                best_params = (a, b, c)

    if best_params:
        a, b, c = best_params
        print(f"\nBest: a={a}, b={b:.3f}, c={c}")
        print("Detailed:")
        measure(M1, M2, a, b, c)

    # Also try production M1
    print("\n--- With production M1 ---")
    prod = json.loads((Path(__file__).parent.parent.parent / "src/helmlab/data/gen_params.json").read_text())
    M1_prod = np.array(prod["M1"])
    M2_prod = np.array(prod["M2"])

    if best_params:
        a, b, c = best_params
        print(f"Using best rational (a={a}, b={b:.3f}, c={c}) with production M1/M2:")
        measure(M1_prod, M2_prod, a, b, c)


if __name__ == "__main__":
    main()
