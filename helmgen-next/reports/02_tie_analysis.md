# TIE Analysis: 13 Non-Ceiling TIEs — How to Flip Each

**Date**: 2026-04-01
**Based on**: ColorBench metric implementation analysis

## Classification

### IMPOSSIBLE to flip (2):
| # | Metric | Value | Reason |
|---|--------|-------|--------|
| 5 | Invisible steps | 99.7% = 99.7% | Quantization ceiling (256 levels). Can't exceed 100%. |
| 12-13 | RT P3/Rec2020 | ~2e-15 vs ~1.6e-15 | IEEE 754 float64 rounding noise. Both at machine epsilon. |

### POSSIBLE but requires NEW M1 (7):
These TIEs exist because `OKLab M1 + depcubic ≈ OKLab pipeline`. Same input → same output.

| # | Metric | Value | Need | How |
|---|--------|-------|------|-----|
| 1 | Gradient CV mean | 38.08 vs 38.03 | <37.65 | Different M1 changes gradient shape |
| 2 | Gradient CV p95 | 138.16 vs 137.14 | <135.77 | Different M1 fixes worst-case pairs |
| 3 | Banding mean | 1.8 = 1.8 | <1.78 | M1/transfer changes step distribution |
| 4 | Worst CV | 412.6 = 412.6 | <408.5 | Fix the single worst pair via M1 |
| 6 | CVD protan | 0.13 = 0.13 | >0.1313 | M1 redesign for cone separation |
| 7 | Multi-stop CV | 37.7 = 37.7 | <37.33 | Different gradient shape from M1 |
| 9 | Muddy gradients | 12 = 12 | ≤11 | M1/chroma changes eliminate 1 muddy pair |

### POSSIBLE with parameter tuning (3):
| # | Metric | Value | Need | How |
|---|--------|-------|------|-----|
| 8 | Chroma pres | 0.416 vs 0.414 | >0.4181 | Increase chroma power slightly |
| 10 | OOG pairs | 9.8% = 9.8% | <9.7% | M2/enrichment reduces excursions |
| 11 | OOG max dist | 0.1101 vs 0.1103 | <0.1092 | Flatten worst excursion pair |

## Critical Insight

**The current M1 is a perturbed OKLab M1 (delta[0,2]=+0.004).** This is close enough to OKLab that 7 gradient/banding/CVD metrics produce nearly identical results. To break these TIEs, we need M1 that's **further from OKLab**.

But MEMORY.md warns: "M1 değiştirince cliff, 1000-trip RT, photo gamut yeni kayıplar" — every M1 change risks new losses.

## Strategy Options

### Option A: Keep M1, flip what we can (3-4 TIE flips)
- Chroma preservation (+chroma power)
- OOG pairs/distance (M2 tuning)
- Maybe muddy gradients
- Target: **39-40 WIN**

### Option B: New M1 grid search (7-10 TIE flips, risk LOSSes)
- Grid search M1 perturbation with ALL 61 metrics as constraints
- Risk: each M1 change may create new LOSSes
- Target: **43-46 WIN** (if lucky)

### Option C: Architecture change (10+ TIE flips)
- Per-channel transfer or geodesic interpolation
- Fundamentally different gradient behavior
- Target: **46-50 WIN** (but months of work)
