# Gradient CV Is Immovable in This Architecture

**Date**: 2026-04-01

## Evidence

### Enrichment amplitude sweep (0→0.09):
- CV range: 38.06-38.11% (0.05% total variation)
- Enrichment does NOT affect gradient CV

### PW L_corr scale sweep (0→2.0):
- Best: PW×0.5 → CV=38.035% (barely beats OKLab's 38.095%)
- Worst: PW×2.0 → CV=38.814%
- PW has minimal effect on mean CV (0.08% range at useful values)
- PW DOES affect p95 significantly: 138→142 (confirms reports.md finding)

### OKLab comparison:
- OKLab: 38.095%
- Our best: 38.035% (PW×0.5)
- Difference: 0.06% — within TIE tolerance (1%)

## Root Cause

Gradient CV is determined by the M1 × transfer function interaction.
depcubic(α=0.02) with perturbed OKLab M1 produces nearly identical
gradient step distribution as OKLab's cbrt with OKLab M1.

No downstream stage (M2, enrichment, PW, radial, log-chroma) can change this.

## Implication

**Gradient CV TIE (38.08 vs 38.03) is STRUCTURAL and UNFIXABLE in any
M1→monotone_transfer→M2→post-processing architecture where M1 ≈ OKLab M1.**

To break this TIE, we need M1 that's fundamentally different from OKLab.
But M1 grid search showed: different M1 → more LOSSes than TIE flips.

## This means:
- At least 2 gradient TIEs (cv_mean, cv_p95) are PERMANENT
- Plus banding (1.8=1.8), worst CV (412.6=412.6) — all gradient-related
- That's 4 TIEs that CANNOT become WINs

Combined with 6 ceiling TIEs + 1 impossible (invisible steps) = 11 permanent TIEs.
Maximum possible: 61 - 11 = 50 WINs. But that requires flipping ALL remaining 2 TIEs
AND reducing LOSSes to 0. Both effectively impossible.

## Honest maximum: 36-38 WINs.
