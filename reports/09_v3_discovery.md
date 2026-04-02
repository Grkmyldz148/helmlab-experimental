# V3 Key Discovery: Enrichment Was Unnecessary

**Date**: 2026-04-01

## The Discovery

Removing enrichment + PW L_corr from the pipeline gives:
- RT: 1.78e-15 → 2.11e-15 (PASS, target <1e-14)
- Blue G/R: **2.01** (PASS, target ≥1.50) — enrichment was unnecessary!
- 360 cusps: maintained
- Analytical invertibility: restored (no Newton)
- Cost: ~1.5x OKLab (PASS, target ≤2x)

The production M1/M2 already gives G/R=2.01 without any hue correction.
The enrichment (amp=0.055, center=264.5°) only moved it from 2.01→1.52 — WORSE.

## What V3 needs to fix (3 remaining FAILs from F category)
1. **Achromatic 9.21e-9** → target <1e-10 (analytic M2 constraint)
2. **Chroma amp 3.44x** → target <3x (M2 re-optimization)
3. **Munsell CV 14.5%** → target <2% (RQ_L correction)

## Action: M2 CMA-ES Optimization
- Freeze M1 (production, perturbed OKLab)
- Freeze alpha=0.02
- 6 DOF: M2 ab-rows (L-row from achromatic constraint)
- Hard gates: cusps=360, achromatic<1e-14, RT<1e-14
- Soft objective: minimize chroma_amp + munsell_cv
