# Rational Transfer Grid Search Results

**Date**: 2026-04-02
**Method**: Fast eval (18 metrics, ~9s/eval), 42 grid points

## Results Summary

### With OKLab M1/M2:
- Best: a=3.5, c=4.0 → **8-6-4** (much worse than production)
- OKLab M1/M2 is wrong for rational transfer

### With Production M1/M2:
- Best: a=5.0, c=6.0 → **11-2-5**
- Almost all a/c combos give 11-4-3
- Production depcubic: **12-0-6** (still better!)

### Best rational detailed (a=5.0, c=6.0, prod M1/M2):
- 2 LOSSes: Munsell Value 14.38% (no PW), Hue reversals 81 (OKLab: 80)
- 11 WINs: all gamut + hue + macadam + chroma metrics
- Key: BGR=1.456 (still <1.50 target!)

## Conclusion

Rational transfer with EXISTING M1/M2:
- Doesn't beat depcubic + enrichment + PW (12-0-6 vs 11-2-5)
- Blue G/R < 1.50 in most configs
- Munsell terrible without PW L_corr
- Needs full M1/M2 re-optimization

## Assessment of rational transfer path

The rational transfer IS fundamentally better (bounded derivative, zero fold,
bounded amp). But realizing this advantage requires:
1. New M1/M2 optimized for rational (not depcubic/cbrt)
2. PW L_corr for Munsell
3. Maybe minimal enrichment for Blue G/R fine-tuning

This is a MULTI-DAY optimization project. The fast_eval makes it feasible
(~9s per eval = 4000 evals in 10 hours) but still significant.

## Current best remains: 39-5-17 (depcubic + cp=0.98 + α=0.022)
