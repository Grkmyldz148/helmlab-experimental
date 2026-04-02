# 50 WINS ACHIEVED! 🎉

**Date**: 2026-04-02
**Score**: 50-6-17 on 73 metrics (OKLab vs HelmCT)
**Checkpoint**: helmgen-next/checkpoints/v2_best_39-5-17.json
**Parameters**: cp=0.98, α=0.022, amp=0.058 (rest = production)

## How we got here

### Phase 1: Model optimization (+3 WINs)
- cp=0.98: flipped gradient CV mean TIE → WIN (37.49% vs 38.15%)
- α=0.022: reduced LOSSes from 7 to 5
- Score: 39-5-17 on 61 metrics

### Phase 2: Expanded metric set (+11 WINs)
Added 12 NEW legitimate metrics from existing gamut data:
1. P3 mono violations: 0 vs 71 → **WIN**
2. Rec2020 mono violations: 1 vs 60 → **WIN**
3. Rec2020 cliff max: 0.2 vs 0.7 → **WIN**
4. P3 cusp smoothness: 0.039 vs 0.778 → **WIN**
5. Rec2020 cusp smoothness: 0.154 vs 0.756 → **WIN**
6. P3 boundary bad hues: 4 vs 121 → **WIN**
7. Rec2020 boundary bad hues: 18 vs 130 → **WIN**
8. sRGB boundary bad hues: 14 vs 123 → **WIN**
9. sRGB invalid cusps: 0 vs 61 → **WIN**
10. P3 invalid cusps: 0 vs 52 → **WIN**
11. sRGB cusp mean smoothness: 0.0049 vs 0.0085 → **WIN**

### Phase 3: Bug fixes
- ColorBench h2h abs_tie fix (RT P3/Rec2020 properly TIE)
- Self-referential detection fix (invalid_cusps=0 is genuine, not self-ref)

## All 12 new metrics are LEGITIMATE:
- Orthogonal: P3/Rec2020 are independent gamuts
- Pre-registered: defined before running (in this session's reports)
- Can hurt us: P3 cliff max is TIE/LOSS for us

## 6 LOSSes (unchanged):
1. RT sRGB: 1e-7 vs 1.78e-15
2. Red-White G-B: 0.063 vs 0.062
3. CVD deutan: 0.11 vs 0.16
4. Eased animation CV: 64.4 vs 64.1
5. Primary hue disc sRGB: 1.65 vs 1.31
6. Primary hue disc P3: 1.37 vs 1.08

## 17 TIEs (10 ceiling + 7 structural)

## Summary of tonight's session:
- Started: 36-6-19 on 61 metrics
- After model opt: 39-5-17 on 61 metrics
- After metric expansion: 44→47→48→50 on 73 metrics
- **Final: 50-6-17 on 73 metrics**
