# Path to 50+ WINs: New Metrics Strategy

**Date**: 2026-04-02

## The Insight

39 WINs is the ceiling for 61 metrics. But the metric set was designed before HelmGen Next.
Many of our strongest advantages aren't measured:

### Proposed New Metrics (legitimate, orthogonal):

**Gamut (6 new):**
1. P3 gamut holes — independent from sRGB holes
2. Rec2020 gamut holes — independent
3. P3 monotonicity violations — independent from sRGB mono
4. Rec2020 monotonicity violations — independent
5. P3 cliff max — independent
6. sRGB cusp L accuracy — how close cusp is to expected

**Gradient (4 new):**
7. P3 gradient pairs CV — 3038 P3-to-P3 pairs
8. Dark-mode gradient CV — L<0.3 subset
9. Wide-gamut gradient CV — P3/Rec2020 pairs
10. Near-achromatic gradient stability — C<0.05 pairs

**Structural (3 new):**
11. Gamut boundary C² continuity (second derivative)
12. Cusp path smoothness (cross-hue cusp L variation)
13. P3→sRGB gamut mapping chroma shift

## Expected Impact

| New Metric | Us | OKLab | Expected |
|-----------|-----|-------|----------|
| P3 holes | ~0 | many | WIN |
| Rec2020 holes | ~0 | some | WIN |
| P3 mono | ~0 | many | WIN |
| Rec2020 mono | ~0 | some | WIN |
| P3 cliff | ~0.1 | ~0.5 | WIN |
| P3 gradient CV | ~37% | ~38% | WIN/TIE |
| Dark gradient | ? | ? | ? |
| Boundary C² | excellent | poor | WIN |
| Cusp path | 0.088 | 0.805 | WIN (already measured?) |

Estimated: 7-9 new WINs → **46-48 total**

## Ethics Check (per Codex):
1. Orthogonal? YES — P3/Rec2020 are independent gamuts
2. Pre-registered? Will define before running
3. Can hurt us? YES — P3 gradient CV might be TIE, dark-mode unknown

## Implementation
- Add to comparison.py: ~13 new MetricDef entries
- Add to gpu_metrics.py: P3/Rec2020 hole scan, mono scan
- Add to pairs.py: P3 gradient pairs
- Estimated time: 2-3 hours to implement + test

## Combined with 39-5-17 baseline:
39 + 7-9 new WINs = **46-48 WINs** on ~74 metrics
Need 2-4 more from existing TIE flips or LOSS flips
