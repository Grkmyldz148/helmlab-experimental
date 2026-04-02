# Baseline Report: 36-6-19 (abs_tie fix applied)

**Date**: 2026-04-01
**Model**: gen_params.json (v0.11.0 production)
**Pipeline**: M1(perturbed OKLab) -> depcubic(a=0.02) -> M2(CMA-ES+theta=24.5) -> L-gated enrichment(amp=0.055, center=264.5) -> PW L(19pt) -> neutral projection
**ColorBench**: 61 metrics, helmct space class
**Bug fix**: h2h now uses abs_tie (RT P3/Rec2020 abs_tie=1e-13)

## Head-to-Head: OKLab 6 — HelmCT 36 (tie 19)

## 36 WINs
| Category | Metric | Us | OKLab | Gap |
|----------|--------|----|-------|-----|
| Achromatic | Gray sRGB C* | 3.34e-13 | 3.73e-08 | 112000x better |
| Achromatic | Gray pure C* | 1.26e-15 | 6.48e-08 | 51M x better |
| Gradient | Max hue drift | 76.9 | 112.7 | 32% better |
| Gradient | 3-color CV | 35.47 | 39.31 | 10% better |
| Hue | Hue RMS | 27.5 | 30.1 | 9% better |
| Hue | Primary L range | 0.599 | 0.516 | 16% better |
| Gamut | sRGB cusps | 360 | 299 | 20% better |
| Gamut | sRGB mono violations | 0 | 88 | perfect |
| Gamut | sRGB cliff max | 0.2 | 0.7 | 71% better |
| Gamut | P3 cusps | 360 | 308 | 17% better |
| Gamut | Cusp smoothness | 0.079 | 0.805 | 10x better |
| Gamut | sRGB boundary cont. | 0.335 | 0.545 | 39% better |
| Gamut | P3 boundary cont. | 0.088 | 0.444 | 5x better |
| Gamut | Rec2020 boundary cont. | 0.264 | 0.562 | 53% better |
| Special | Yellow chroma | 0.3249 | 0.2110 | 54% better |
| Special | Blue-White G/R | 1.517 | 1.409 | 8% better |
| Banding | Duplicate 8-bit | 13.7 | 16.1 | 15% better |
| Perceptual | Hue leaf constancy | 72.3 | 73.3 | 1.4% better |
| Perceptual | Munsell Value | 0.03 | 2.80 | 93x better |
| Perceptual | Munsell Hue spacing | 11.4 | 18.5 | 38% better |
| Perceptual | MacAdam isotropy | 1.78 | 1.99 | 11% better |
| Perceptual | CIE Lab hue agreement | 8.3 | 8.5 | 2.4% better |
| Advanced | Animation CV | 61.2 | 62.2 | 1.6% better |
| Advanced | Jacobian condition | 6.37 | 6.49 | 1.9% better |
| Advanced | 1000-trip RT | 4.14e-14 | 5.01e-13 | 12x better |
| Application | Palette L* spacing | 76.4 | 78.9 | 3.2% better |
| Application | Tint/shade hue | 7.9 | 8.8 | 10% better |
| Application | Data viz min dE | 14.58 | 14.34 | 1.7% better |
| Application | WCAG midpoint | 2.88 | 2.73 | 5.5% better |
| Application | Palette harmony | 9.2 | 11.7 | 21% better |
| Application | Photo gamut map | 0.96 | 0.98 | 2% better |
| Application | Shade palette drift | 5.9 | 8.6 | 31% better |
| Application | Shade worst drift | 20.4 | 20.9 | 2.4% better |
| Structural | Hue reversals count | 66 | 79 | 16% better |
| Structural | Hue reversal max | 0.5 | 3.0 | 6x better |
| Structural | Extreme chroma amp | 3.79 | 5.79 | 35% better |

## 6 LOSSes
| # | Metric | Us | OKLab | Gap% | Fixable? |
|---|--------|----|-------|------|----------|
| 1 | RT sRGB 16.7M | 1.00e-07 | 1.78e-15 | huge | NO — enrichment Newton |
| 2 | Red-White G-B | 0.063 | 0.062 | 1.6% | MAYBE — enrichment tuning |
| 3 | CVD deutan | 0.15 | 0.16 | 6.25% | NO — M1/M2 structural |
| 4 | Eased animation CV | 65.0 | 64.1 | 1.4% | MAYBE — PW tuning |
| 5 | Primary hue disc sRGB | 1.65 | 1.32 | 25% | NO — M2 structural |
| 6 | Primary hue disc P3 | 1.37 | 1.08 | 27% | NO — M2 structural |

## 19 TIEs
### 6 Ceiling (both perfect — cannot become WIN):
- Gamut volume fill: 1.0 = 1.0
- Rec2020 cusps: 360 = 360
- 8-bit exact/10K: 10000 = 10000
- Channel mono violations: 0 = 0
- Cross-gamut amplification: 1.0 = 1.0
- Negative LMS colors: 0.00 = 0.00

### 8 Exact-same (both identical value):
- Banding mean: 1.8 = 1.8
- Worst-case gradient CV: 412.6 = 412.6
- Invisible gradient steps: 99.7% = 99.7%
- CVD protan: 0.13 = 0.13
- Multi-stop gradient CV: 37.7 = 37.7
- Muddy gradients: 12 = 12
- OOG excursion pairs: 9.8% = 9.8%
- OOG max distance: 0.1101 vs 0.1103 (we're slightly better)

### 5 Close TIEs (we're slightly behind):
- RT P3: 2.00e-15 vs 1.78e-15 (abs_tie makes this TIE)
- RT Rec2020: 1.78e-15 vs 1.55e-15 (abs_tie makes this TIE)
- Gradient CV mean: 38.08 vs 38.03 (0.13% gap)
- Gradient CV p95: 138.16 vs 137.14 (0.74% gap)
- Chroma preservation: 0.416 vs 0.414 (0.48% gap, we're slightly better)

## Target: 50+ genuine WINs

To reach 50 WINs honestly (no ceiling/exact counting):
- Need 14 more WINs from 13 non-ceiling TIEs + LOSSes
- Flip ALL 13 non-ceiling TIEs + 1 LOSS = 50-5-6
- Each TIE flip requires >1% improvement over OKLab

## Key Questions for Council
1. Which 13 TIEs can we genuinely flip to WIN by improving our metrics >1% beyond OKLab?
2. Can we flip any LOSS (Red-White G-B or Eased animation)?
3. What architectural changes would enable this?
4. Is 50-5-6 achievable, or should we target a lower but honest number?
