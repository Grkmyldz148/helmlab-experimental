# BREAKTHROUGH: Rational Transfer Achieves All 3 Impossible Targets

**Date**: 2026-04-02

## The Discovery

f(x) = x·(a + b·x) / (1 + c·x) with a=3.8, c=5, b=2.2:

| Target | OKLab | Depcubic (prod) | **Rational** | goal.md |
|--------|-------|----------------|-------------|---------|
| Blue G/R | 1.41 | 1.52 | **1.80** | ≥1.50 ✓ |
| Chroma amp | 5.79x | 3.79x | **2.8x** | <3x ✓ |
| Gamut holes | 46 | 3 | **0** | 0 ✓ |
| Cusps (36 tested) | 36/36 | 36/36 | **36/36** | 360 |
| RT precision | 1.78e-15 | 1e-7 | **~1e-15** | <1e-14 ✓ |
| Gradient CV proxy | ~38% | ~38% | **~20%** | ≤37% ✓ |

## ALL THREE previously "impossible" targets met simultaneously:
1. **Chroma amp <3x** — council said "mathematically unachievable" with linear M2
2. **Zero gamut holes** — blue fold eliminated by bounded derivative
3. **Blue G/R ≥1.50** — without enrichment!

## Why It Works

Rational transfer has **bounded derivative at ALL points**:
- f'(0) = a = 3.8 (finite, vs cbrt's ∞)
- f'(∞) → b/c = 0.44 (finite, vs cbrt's 0)
- No point where derivative explodes → no gamut fold → no chroma amplification blow-up

The depcubic was a step in this direction (f'(0)=1/α≈45, finite) but still approached cbrt for large x.
Rational is bounded EVERYWHERE — fundamentally different behavior.

## Sweet Spot Parameters

Best candidates (all pass BGR≥1.50, Amp<3x, Holes=0):
| a | c | b | BGR | Amp | CV proxy |
|---|---|---|-----|-----|---------|
| 3.2 | 4 | 1.80 | 1.56 | 2.5x | 22% |
| 3.5 | 4 | 1.50 | 1.67 | 2.7x | 19% |
| 3.8 | 5 | 2.20 | 1.80 | 2.8x | 20% |
| 4.0 | 5 | 2.00 | 1.90 | 3.0x | 18% |

## Next Steps
1. Add rational transfer to HelmCT ColorBench class
2. Run full 61-metric ColorBench
3. If WINs > 39: optimize M1/M2 for rational transfer
4. This could be the path to 50+ WINs

## Caveat
- Proxy metrics only (5 gradient pairs, 36 cusps, no full CIEDE2000)
- Full ColorBench may show different results
- M1/M2 were designed for depcubic, not rational — re-optimization needed
- Achromatic error is 1.4e-4 (OKLab M1) — needs fixing
