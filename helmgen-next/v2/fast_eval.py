"""Fast evaluator using REAL ColorBench kernels — ~3s per eval.

Runs only the 12 critical metrics that determine WIN/LOSS/TIE.
Skips: RT 16.7M (5.5s), CVD (0.1s), double RT (1.3s), Jacobian (0.5s).

Usage:
    from fast_eval import fast_score
    wins, losses, ties, details = fast_score(checkpoint_path)
"""

import sys, json, time, torch
from pathlib import Path

COLORBENCH = Path(__file__).parent.parent.parent / "colorbench"
sys.path.insert(0, str(COLORBENCH))

from core.spaces import OKLab, HelmCT
from core.pairs import generate_all_pairs
from core.gpu_metrics import measure_gradients, measure_gamut, measure_special_gradients, measure_achromatic
from core.gpu_metrics_perceptual import measure_munsell_value, measure_munsell_hue, measure_macadam_isotropy, measure_chroma_preservation
from core.gpu_metrics_advanced import measure_hue_reversal, measure_extreme_chroma_stability

DEVICE = torch.device('cpu')

# Pre-compute OKLab baseline (cached)
_oklab_cache = {}

def _get_oklab_baseline():
    if _oklab_cache:
        return _oklab_cache

    oklab = OKLab(DEVICE)
    pairs, labels = generate_all_pairs(DEVICE)

    g = measure_gradients(oklab, pairs, labels, DEVICE)
    _oklab_cache['grad_cv'] = g['overall']['cv_mean']
    _oklab_cache['grad_p95'] = g['overall']['cv_p95']
    _oklab_cache['grad_cv3'] = g['overall'].get('cv_3color_mean', 0)
    _oklab_cache['banding'] = g['overall']['banding_mean']
    _oklab_cache['worst_cv'] = g['overall']['cv_max']
    _oklab_cache['drift_max'] = g['overall']['drift_max_noncrossing']

    gm = measure_gamut(oklab, DEVICE, n_hues=36, n_L=100, n_C=80)
    _oklab_cache['cusps_srgb'] = gm['sRGB']['valid_cusps']
    _oklab_cache['cusps_p3'] = gm['P3']['valid_cusps']
    _oklab_cache['mono'] = gm['sRGB']['monotonicity_violations']
    _oklab_cache['cusp_smooth'] = gm['sRGB']['smoothness_max_jump']

    sp = measure_special_gradients(oklab, DEVICE)
    _oklab_cache['bgr'] = sp['blue_white_midpoint']['G_over_R']

    ach = measure_achromatic(oklab, DEVICE)
    _oklab_cache['gray_pure'] = ach['gray_ramp_pure']['max_chroma']

    mv = measure_munsell_value(oklab, DEVICE)
    _oklab_cache['munsell_v'] = mv['dL_cv']

    mh = measure_munsell_hue(oklab, DEVICE)
    _oklab_cache['munsell_h'] = mh['spacing_cv']

    ma = measure_macadam_isotropy(oklab, DEVICE)
    _oklab_cache['macadam'] = ma['mean_ratio']

    cp = measure_chroma_preservation(oklab, DEVICE)
    _oklab_cache['chroma_pres'] = cp['mean_preservation']
    _oklab_cache['muddy'] = cp['n_muddy']

    hr = measure_hue_reversal(oklab, DEVICE)
    _oklab_cache['hue_rev'] = hr['hues_with_reversals']

    ec = measure_extreme_chroma_stability(oklab, DEVICE)
    _oklab_cache['chroma_amp'] = ec['max_amplification']

    _oklab_cache['_pairs'] = pairs
    _oklab_cache['_labels'] = labels

    return _oklab_cache


def fast_score(checkpoint_path, verbose=False):
    """Evaluate checkpoint against OKLab. Returns (wins, losses, ties, details)."""
    ok = _get_oklab_baseline()
    pairs, labels = ok['_pairs'], ok['_labels']

    space = HelmCT(str(checkpoint_path), DEVICE)

    # Run critical metrics
    g = measure_gradients(space, pairs, labels, DEVICE)
    gm = measure_gamut(space, DEVICE, n_hues=36, n_L=100, n_C=80)  # coarse for speed
    sp = measure_special_gradients(space, DEVICE)
    ach = measure_achromatic(space, DEVICE)
    mv = measure_munsell_value(space, DEVICE)
    mh = measure_munsell_hue(space, DEVICE)
    ma = measure_macadam_isotropy(space, DEVICE)
    cp = measure_chroma_preservation(space, DEVICE)
    hr = measure_hue_reversal(space, DEVICE)
    ec = measure_extreme_chroma_stability(space, DEVICE)

    # Compare (1% tolerance)
    TOL = 0.01
    metrics = {}
    wins = losses = ties = 0

    def compare(name, ours, theirs, lower_better=True):
        nonlocal wins, losses, ties
        if lower_better:
            better = ours < theirs
            rel = abs(ours - theirs) / (abs(theirs) + 1e-30)
        else:
            better = ours > theirs
            rel = abs(ours - theirs) / (abs(max(ours, theirs)) + 1e-30)

        if ours == theirs or rel <= TOL:
            ties += 1
            result = 'TIE'
        elif better:
            wins += 1
            result = 'WIN'
        else:
            losses += 1
            result = 'LOSS'

        metrics[name] = {'ours': ours, 'oklab': theirs, 'result': result}
        if verbose:
            print(f"  {name:25s}: {ours:.4f} vs {theirs:.4f} → {result}")

    compare('grad_cv', g['overall']['cv_mean'], ok['grad_cv'], lower_better=True)
    compare('grad_p95', g['overall']['cv_p95'], ok['grad_p95'], lower_better=True)
    compare('banding', g['overall']['banding_mean'], ok['banding'], lower_better=True)
    compare('worst_cv', g['overall']['cv_max'], ok['worst_cv'], lower_better=True)
    compare('drift_max', g['overall']['drift_max_noncrossing'], ok['drift_max'], lower_better=True)
    compare('cusps_srgb', gm['sRGB']['valid_cusps'], ok['cusps_srgb'], lower_better=False)
    compare('cusps_p3', gm['P3']['valid_cusps'], ok['cusps_p3'], lower_better=False)
    compare('mono', gm['sRGB']['monotonicity_violations'], ok['mono'], lower_better=True)
    compare('cusp_smooth', gm['sRGB']['smoothness_max_jump'], ok['cusp_smooth'], lower_better=True)
    compare('bgr', sp['blue_white_midpoint']['G_over_R'], ok['bgr'], lower_better=False)
    compare('gray_pure', ach['gray_ramp_pure']['max_chroma'], ok['gray_pure'], lower_better=True)
    compare('munsell_v', mv['dL_cv'], ok['munsell_v'], lower_better=True)
    compare('munsell_h', mh['spacing_cv'], ok['munsell_h'], lower_better=True)
    compare('macadam', ma['mean_ratio'], ok['macadam'], lower_better=True)
    compare('chroma_pres', cp['mean_preservation'], ok['chroma_pres'], lower_better=False)
    compare('muddy', cp['n_muddy'], ok['muddy'], lower_better=True)
    compare('hue_rev', hr['hues_with_reversals'], ok['hue_rev'], lower_better=True)
    compare('chroma_amp', ec['max_amplification'], ok['chroma_amp'], lower_better=True)

    return wins, losses, ties, metrics


if __name__ == "__main__":
    # Test with production
    prod = Path(__file__).parent.parent.parent / "src/helmlab/data/gen_params.json"
    t0 = time.time()
    print("Loading OKLab baseline...")
    w, l, t, d = fast_score(prod, verbose=True)
    elapsed = time.time() - t0
    print(f"\nScore: {w}-{l}-{t} ({elapsed:.1f}s)")

    # Test with rational
    rat = Path(__file__).parent.parent / "checkpoints/v2_rational_a38_c5.json"
    if rat.exists():
        t0 = time.time()
        print(f"\n--- Rational ---")
        w2, l2, t2, d2 = fast_score(rat, verbose=True)
        elapsed2 = time.time() - t0
        print(f"\nScore: {w2}-{l2}-{t2} ({elapsed2:.1f}s)")
