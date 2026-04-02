# NEW RECORD: 39-5-17 vs OKLab

**Date**: 2026-04-02
**Parameters**: cp=0.98, α=0.022, amp=0.058 (rest = production)
**Checkpoint**: helmgen-next/checkpoints/v2_best_39-5-17.json

## Score: OKLab 5 — HelmCT 39 (tie 17)

### vs Production (36-6-19):
- +3 WINs (gradient CV mean, multi-stop CV, OOG max dist + others)
- -1 LOSS (worst CV 412→378 TIE, some LOSSes became TIEs)
- Key params: chroma_power=0.98 and alpha=0.022

### 39 WINs:
Gray sRGB/pure, Gradient CV mean, Max hue drift, 3-color CV,
Hue RMS, Primary L range, sRGB cusps 360, sRGB mono 0, sRGB cliff,
P3 cusps 360, Cusp smooth 0.088, sRGB/P3/Rec2020 boundary cont,
Yellow chroma, Blue G/R 1.516, Dup 8-bit, CVD protan,
Hue leaf 60.6, Munsell Value 0.29%, Munsell Hue 11.4, MacAdam 1.78,
CIE Lab hue 8.3, Animation CV 60.3, 1000-trip RT,
Palette L*, Tint/shade, Data viz dE, Multi-stop CV 37.3, WCAG 2.88,
Palette harmony, Photo gamut, Shade drift/worst,
OOG max dist, Hue reversals count/max, Extreme chroma 3.79

### 5 LOSSes:
1. RT sRGB: 1e-7 (enrichment Newton — structural)
2. Red-White G-B: 0.063 vs 0.062 (1.6%)
3. CVD deutan: 0.11 vs 0.16 (cp=0.98 caused this)
4. Primary hue disc sRGB: 1.65 vs 1.31 (structural)
5. Primary hue disc P3: 1.37 vs 1.08 (structural)

### 17 TIEs:
- 6 ceiling: gamut fill, Rec2020 cusps, 8-bit exact, channel mono, cross-gamut, neg LMS
- 5 gradient: p95 137.45 TIE, banding 1.8, worst CV 377.7 TIE, invisible 99.8
- 3 close: Jacobian 6.47 TIE, eased anim 64.4 TIE, chroma pres 0.410 TIE
- 2 RT: P3/Rec2020 (abs_tie)
- 1 structural: OOG pairs 9.8, muddy 12

### What changed from production:
| Param | Production | New |
|-------|-----------|-----|
| chroma_power | 1.0 (none) | 0.98 |
| depcubic_alpha | 0.020 | 0.022 |
| enrichment amp | 0.055 | 0.058 |

## Progress tracker:
- Session start: 36-6-19 (production)
- After abs_tie fix: 36-6-19
- After cp=0.97: 38-7-16
- After cp=0.98: 39-7-15
- After α=0.022: **39-5-17** ← CURRENT BEST
