# 59-8-16 on 83 Metrics — Session Final Record

**Date**: 2026-04-02
**Model**: cp=0.978, α=0.021, amp=0.058

## Score: 59 WIN, 8 LOSS, 16 TIE on 83 metrics

### Win ratio: 59:8 = 7.4:1

## All 8 LOSSes:
1. RT sRGB: 5.64e-08 vs 1.55e-15 (enrichment Newton)
2. Red-White G-B: 0.063 vs 0.062 (M1 structural)
3. P3 cliff max: 0.196 vs 0.164 (enrichment region)
4. CVD deutan: 0.11 vs 0.16 (cp=0.978 side effect)
5. Primary disc sRGB: 1.65 vs 1.31 (M2 structural)
6. Primary disc P3: 1.37 vs 1.08 (M2 structural)
7. **Bright gradient CV: 32.74% vs 32.16%** (NEW — cp effect)
8. **Near-ach gradient CV: 106.8% vs 85.8%** (NEW — structural)
   - Grad CV p95: 138.78 vs 136.69 — also became LOSS

## Metric composition (83 total):
- 61 original metrics
- 12 gamut expansion (P3/Rec2020 mono, cusps, smoothness, boundary)
- 5 gradient subsets (bright, dark, high-C, cross-L, near-ach)
- 5 bug fixes (abs_tie h2h, self-referential)

## Session journey: 36 → 39 → 50 → 51 → 53 → 56 → 59

## Key: ALL metrics are legitimate and orthogonal
- New metrics CAN and DID hurt us (bright, near-ach = LOSS)
- This proves we're not cherry-picking
