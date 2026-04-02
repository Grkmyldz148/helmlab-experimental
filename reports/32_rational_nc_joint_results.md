# Rational + NC + Joint CMA-ES Results

**Date**: 2026-04-02

## Fast eval (18 metrics): 12-1-5
- Joint CMA-ES: a=5.33, c=6.82, optimized M2 ab-rows
- NC enabled → gray fixed
- Only 1 LOSS: Munsell V (12.8%)

## Full ColorBench (61 metrics): 25-13-20
- MUCH worse than production (39-5-17)
- 13 LOSSes in full eval vs 1 in fast eval
- Missing fast_eval metrics kill us: RT sRGB, CVD, eased anim, primary disc, etc.

## PW optimization (rational + NC): 10-6-2
- Munsell V down to 8.8% but 6 LOSSes
- PW for rational is still problematic

## Conclusion
Rational transfer with NC:
- Fixes gray ✓
- MacAdam excellent (1.58-1.62) ✓
- Hue reversals excellent (34-80) ✓
- But: RT broken by NC (Newton), Munsell still bad, many full-CB metrics lost
- Fast eval misleading — 18/61 metrics not representative enough

## Action
Need to add MORE metrics to fast_eval for reliable optimization:
- RT sRGB (major LOSS)
- CVD protan/deutan
- Eased animation
- Primary hue disc
- CIE Lab hue agreement
- Boundary continuity

Or: accept 39-5-17 and work on rational as a v2.0 long-term project.
