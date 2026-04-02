# Rational + PW L_corr Optimization — FAILED

**Date**: 2026-04-02

## Results
- Rational bare: 10-3-5
- Rational + enrichment: 10-3-5
- Rational + production PW: 9-5-4 (PW incompatible)
- Rational + CMA-ES PW: 10-7-1 (optimizer broke everything)

## Root Cause
PW L_corr designed for depcubic is incompatible with rational transfer.
CMA-ES on 19 PW params with fast_eval scoring created degenerate solutions
(gradient CV 76%, chroma amp 157x).

## The 3 persistent LOSSes in rational:
1. Gray pure: 0.633 — rational f(x)=x(a+bx)/(1+cx) is NOT neutral-preserving
   when M1 mixes channels. f(kx) ≠ k·f(x) for different k values.
2. Munsell Value: 16.4% — no L correction at all
3. Munsell Hue: 230° — M2 ab-rows not aligned for Munsell

## Key Insight
The rational transfer's achromatic problem is FUNDAMENTAL:
- cbrt: f(kx) = k^(1/3) · f(x) — scale-equivariant → perfect neutrals
- depcubic: nearly scale-equivariant for large x → near-perfect neutrals
- rational: f(kx) ≠ k^α · f(x) for any α → broken neutrals

This means rational ALWAYS needs NC (neutral correction) post-M2.
Without NC, gray ramp will always be bad.

## Verdict
Rational transfer is not a drop-in replacement for depcubic.
It needs a fundamentally different pipeline design:
M1 → rational → NC → M2 → Lab (NC before M2, not after)
Or: M1 → rational → M2 → NC → Lab (NC after M2)

This is a larger architectural change than parameter tuning.

## Session conclusion
**39-5-17 remains the best achievable score.**
