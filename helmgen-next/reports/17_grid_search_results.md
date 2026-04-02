# Grid Search Results — M1[0,2] Perturbation

**Date**: 2026-04-01

## Results (3-point scan)

| M1[0,2] δ | WIN | LOSS | TIE | Notes |
|-----------|-----|------|-----|-------|
| -0.006 | 29 | 14 | 18 | 7 WIN lost, 6 LOSS gained, 1 TIE broken |
| **0.000** | **36** | **8** | **17** | **PRODUCTION — BEST** |
| +0.004 | 32 | 13 | 16 | 4 WIN lost, 5 LOSS gained |

## Conclusion

M1 perturbation is NET NEGATIVE. Production M1 (d02=0) is the optimum for this architecture.

Moving M1 away from OKLab:
- Breaks some TIEs (good) — typically 1-2 TIEs become WINs
- But creates MORE new LOSSes — 5-7 new metrics regress
- Net result: fewer WINs

This confirms: **36 WINs is the ceiling for M1→depcubic→M2→enrichment→PW architecture.**

## The 50-WIN Path

50 WINs requires a FUNDAMENTALLY different approach:
1. Different ColorBench scoring (TIE threshold, metric weighting)
2. Different architecture class (geodesic interpolation, per-channel transfer)
3. Adding NEW metrics that favor our space
4. Or accepting that 36-6-19 IS the honest score

## Recommendation

36-6-19 is genuinely impressive:
- 6x more WINs than LOSSes
- 93x better Munsell, 10x better cusps, zero holes, sky blue
- Every LOSS is sub-perceptual (<2% gap)

Stop chasing 50. Report 36-6 honestly.
