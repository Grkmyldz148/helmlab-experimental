# Rational Transfer CMA-ES Optimization Result

**Date**: 2026-04-02
**Duration**: ~25 minutes (coarse gamut, 150 gen × 16 pop)
**Result**: 10-3-5 (18 fast_eval metrics)

## Best Parameters
- a=4.89, c=5.35, b=1.46
- M2 ab-rows: CMA-ES optimized (different from OKLab)

## Comparison (fast_eval, 18 metrics)
| Metric | Rational CMA | Production depcubic |
|--------|-------------|-------------------|
| WINs | 10 | 12 |
| LOSSes | 3 | 0 |
| TIEs | 5 | 6 |

## Key findings
- Rational transfer CAN'T beat production depcubic without PW L_corr
- Gray ramp: 0.633 (terrible — rational doesn't preserve achromatic well)
- Munsell: 16.4% and 230.7° (terrible — no L correction)
- These are the same problems as before, just worse

## Rational's strengths
- Hue reversals: 34 vs 80 (much better than OKLab!)
- MacAdam isotropy: 1.58 (best we've ever seen!)
- Cusp smoothness: 0.28 (good)
- Zero holes likely (bounded derivative)
- Chroma amp: 3.60 (good, near <3x target)

## Verdict
Rational transfer needs PW L_corr and achromatic correction to be competitive.
Without them, the perceptual metrics (Munsell, gray) are too bad.

**39-5-17 (depcubic + cp=0.98 + α=0.022) remains the best.**

## Next possible steps
1. Add PW L_corr to rational pipeline and re-optimize
2. Add achromatic correction (neutral blend or NC)
3. Or: accept 39-5-17 as the practical maximum
