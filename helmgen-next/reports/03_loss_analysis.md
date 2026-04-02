# LOSS Analysis: 6 Losses — Can Any Be Flipped?

**Date**: 2026-04-01

## 6 LOSSes Detailed

### LOSS 1: RT sRGB 16.7M — 1.00e-07 vs 1.78e-15
**Gap**: 56,000x worse
**Cause**: L-gated hue enrichment uses Halley iteration for inverse. 8 iterations converge to ~1e-7, not 1e-15.
**Fix options**:
- More iterations (20→50): diminishing returns, ~1e-8 at best
- Remove enrichment: loses Blue G/R WIN (1.517→~1.41)
- Analytical enrichment inverse: not possible for sin²(π·L) gate function
**Verdict**: **UNFIXABLE** without losing Blue G/R WIN. Accept as architectural cost.

### LOSS 2: Red-White midpoint G-B — 0.063 vs 0.062
**Gap**: 1.6%
**Cause**: M1 perturbation shifts red→white gradient midpoint slightly toward green.
**Fix options**:
- Red-sector enrichment (amp~0.01 at h≈20°): might shift G-B by 0.001
- M1 entry [1,0] tweak: affects L/M cone balance → cascade risk
- Fourier hue correction at h≈15°: proven to cancel in interpolation (history.txt)
**Verdict**: **MAYBE FIXABLE** — needs targeted experiment. Low confidence.

### LOSS 3: CVD deutan — 0.15 vs 0.16
**Gap**: 6.25%
**Cause**: M1/M2 cone overlap structure. Deutan removes M cone → our space loses slightly more gradient discriminability.
**Fix options**:
- M1 redesign for cone separation: structural change, cascade risk
- Post-M2 CVD-aware correction: new architecture
**Verdict**: **UNFIXABLE** in current architecture. M1/M2 structural.

### LOSS 4: Eased animation CV — 65.0 vs 64.1
**Gap**: 1.4%
**Cause**: PW L_corr + enrichment create small non-uniformity under cubic easing.
**Fix options**:
- PW breakpoint tuning for near-white region (L≈0.8-0.95)
- Smooth the PW transition at high L values
- Risk: Munsell Value regression (currently 0.03% WIN)
**Verdict**: **MAYBE FIXABLE** — PW fine-tuning experiment needed. Medium confidence.

### LOSS 5-6: Primary hue disc sRGB 1.65° / P3 1.37° vs 1.32° / 1.08°
**Gap**: 25% / 27%
**Cause**: M2 ab-row eccentricity. Our M2 (CMA-ES+24.5° rotation) creates more hue jumps near primary vertices.
**Fix options**:
- M2 ab-row scaling: tried in reports.md — "6/8 same, 2 worse"
- Fourier hue correction 2nd harmonic: might circularize but risks Munsell Hue
- M2 re-optimization targeting primary smoothness: loses other gamut wins
**Verdict**: **UNFIXABLE** in current architecture. M2 structural trade-off.

## Summary

| LOSS | Fixable? | Confidence | Risk |
|------|----------|------------|------|
| RT sRGB | NO | — | Would lose Blue G/R |
| Red-White G-B | MAYBE | Low | Cascade to other hue metrics |
| CVD deutan | NO | — | M1/M2 structural |
| Eased animation | MAYBE | Medium | Munsell regression |
| Primary disc sRGB | NO | — | M2 structural |
| Primary disc P3 | NO | — | M2 structural |

**Maximum LOSS flips**: 0-2 (Red-White + Eased animation, if lucky)
