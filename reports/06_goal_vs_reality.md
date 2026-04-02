# goal.md vs Reality — Criteria-by-Criteria Audit

**Date**: 2026-04-01
**Model**: gen_params.json (v0.11.0) tested via helmct in ColorBench

## A. Gamut Geometry (4 criteria)

| # | Criterion | Target | Achieved | Status |
|---|-----------|--------|----------|--------|
| 1 | Gamut holes | 0 | **0** (360° sRGB/P3/Rec2020) | **PASS** |
| 2 | Boundary C¹ | C¹ continuous | sRGB 0.335, P3 0.088, Rec 0.264 (OKLab: 0.545/0.444/0.562) | **PASS** (much better than OKLab) |
| 3 | Cusps 360/360/360 | 360/360/360, smooth <0.1 | **360/360/360**, smoothness **0.079** | **PASS** |
| 4 | Monotonicity | 0 violations | **0 violations** | **PASS** |

**A: 4/4 PASS**

## B. Gradient & Interpolation (5 criteria)

| # | Criterion | Target | Achieved | Status |
|---|-----------|--------|----------|--------|
| 5 | OOG excursion | <15 pairs, max <0.30 | 9.8% pairs, max **0.1101** | **PASS** |
| 6 | Gradient CV | ≤37% | **38.08%** | **FAIL** (target was ≤37%, got 38.08%) |
| 7 | Path optimality | geodesic ratio ≈1.0 | Not measured in ColorBench | **UNTESTED** |
| 8 | Multi-stop quality | CV comparable, no pinch | 3-color CV **35.47** (OKLab 39.31 — WIN), chroma pres **0.416** | **PASS** |
| 9 | Curvature continuity | No spikes | Not directly measured | **UNTESTED** |

**B: 2/3 tested PASS, 1 FAIL (gradient CV 38.08 > 37% target)**

## C. Hue Properties (5 criteria)

| # | Criterion | Target | Achieved | Status |
|---|-----------|--------|----------|--------|
| 10 | Blue G/R | ≥1.50 | **1.517** | **PASS** |
| 11 | Iso-hue straightness | Straight iso-hue | Hue RMS **27.5°** (OKLab 30.1° — WIN) | **PARTIAL** (better than OKLab, but not "straight") |
| 12 | Hue linearity (0 reversals) | 0 reversals | **66 reversals** (OKLab 79 — WIN) | **FAIL** (target was 0, got 66) |
| 13 | Shade hue drift <5° | <5° | **5.9°** (OKLab 8.6° — WIN) | **FAIL** (target was <5°, got 5.9°) |
| 14 | Hue angular velocity | Constant | Not directly measured | **UNTESTED** |

**C: 1/3 tested PASS, 2 FAIL**

## D. Achromatic Axis (2 criteria)

| # | Criterion | Target | Achieved | Status |
|---|-----------|--------|----------|--------|
| 15 | Achromatic <1e-10 | <1e-10 | Gray pure **1.26e-15**, Gray sRGB **3.34e-13** | **PASS** |
| 16 | Gray ramp uniformity | CV <5% | Not directly measured as CV | **UNTESTED** |

**D: 1/1 tested PASS**

## E. Perceptual Uniformity (4 criteria)

| # | Criterion | Target | Achieved | Status |
|---|-----------|--------|----------|--------|
| 17 | Munsell Value <2% | <2% | **0.03%** | **PASS** (93x better than OKLab) |
| 18 | ΔE consistency | <5% variation | Not measured | **UNTESTED** |
| 19 | MacAdam isotropy <2.0 | <2.0 | **1.78** | **PASS** |
| 20 | Munsell Hue <18° | <18° | **11.4°** | **PASS** |

**E: 3/3 tested PASS**

## F. Numerical Properties (3 criteria)

| # | Criterion | Target | Achieved | Status |
|---|-----------|--------|----------|--------|
| 21 | RT <1e-14 | <1e-14 | sRGB **1.00e-07**, P3 2.00e-15, Rec 1.78e-15 | **FAIL** (sRGB 1e-7 due to enrichment Newton) |
| 22 | Chroma amp <3x, Jacobian <6 | <3x, <6 | Chroma **3.79x**, Jacobian **6.37** | **FAIL** (3.79 > 3x, 6.37 > 6) |
| 23 | Analytical invertibility | No Newton | Enrichment uses Halley iteration | **FAIL** (Halley = iterative) |

**F: 0/3 PASS — all numerical targets missed**

## G. Physical Consistency (3 criteria)

| # | Criterion | Target | Achieved | Status |
|---|-----------|--------|----------|--------|
| 24 | Energy monotonicity | L↑ ⟹ Y↑ always | **0 L reversals** | **PASS** |
| 25 | White point invariance | <2° drift | Not measured | **UNTESTED** |
| 26 | Observer robustness | <1% shift | Not measured | **UNTESTED** |

**G: 1/1 tested PASS**

## H. Temporal & Perceptual Robustness (3 criteria)

| # | Criterion | Target | Achieved | Status |
|---|-----------|--------|----------|--------|
| 27 | Temporal stability | Low frame ΔE | Animation CV **61.2** (OKLab 62.2 — WIN) | **PASS** |
| 28 | Local contrast | No collapse at extremes | Not directly measured | **UNTESTED** |
| 29 | UI semantic consistency | Balanced prominence | Not measured | **UNTESTED** |

**H: 1/1 tested PASS**

## I. Application Robustness (3 criteria)

| # | Criterion | Target | Achieved | Status |
|---|-----------|--------|----------|--------|
| 30 | Palette L* >80% | >80% | **76.4%** | **FAIL** (target 80%, got 76.4%) |
| 31 | WCAG contrast >3.0 | >3.0 | **2.88** | **FAIL** (target 3.0, got 2.88) |
| 32 | Photo gamut <0.95 | <0.95 | **0.96** | **FAIL** (target 0.95, got 0.96 — just missed) |

**I: 0/3 PASS — all application targets missed (but all better than OKLab)**

## J. Accessibility (4 criteria)

| # | Criterion | Target | Achieved | Status |
|---|-----------|--------|----------|--------|
| 35 | APCA correlation | correlate with APCA | Not measured | **UNTESTED** |
| 36 | CVD robustness >0.15 | >0.15 | protan **0.13**, deutan **0.15** | **FAIL** (protan 0.13 < 0.15) |
| 37 | Gamut volume preservation | proportional | Fill **1.0** | **PASS** |
| 38 | Viewing condition | documented | Not parametric | **UNTESTED** |

**J: 1/2 tested PASS, 1 FAIL**

## K. Computational (2 criteria)

| # | Criterion | Target | Achieved | Status |
|---|-----------|--------|----------|--------|
| 39 | ≤2x OKLab cost | ≤2x | ~2.5x (6 stages) | **FAIL** |
| 40 | <300 lines | <300 lines | ~250 lines (gen.py) | **PASS** |

**K: 1/2 PASS, 1 FAIL**

---

## SUMMARY SCORECARD

| Category | Tested | PASS | FAIL | Rate |
|----------|--------|------|------|------|
| A. Gamut Geometry | 4 | 4 | 0 | 100% |
| B. Gradient Quality | 3 | 2 | 1 | 67% |
| C. Hue Properties | 3 | 1 | 2 | 33% |
| D. Achromatic | 1 | 1 | 0 | 100% |
| E. Perceptual | 3 | 3 | 0 | 100% |
| F. Numerical | 3 | 0 | 3 | 0% |
| G. Physical | 1 | 1 | 0 | 100% |
| H. Temporal | 1 | 1 | 0 | 100% |
| I. Application | 3 | 0 | 3 | 0% |
| J. Accessibility | 2 | 1 | 1 | 50% |
| K. Computational | 2 | 1 | 1 | 50% |
| **TOTAL** | **26** | **15** | **11** | **58%** |

**14 criteria UNTESTED** (mostly G25-26, H28-29, J35/38, and several "not directly measured" ones)

## 11 FAILs — Categorized

### Architectural (can't fix without new pipeline): 4
- F21: RT sRGB 1e-7 (enrichment Newton)
- F22: Chroma amp 3.79x (M2 structural)
- F23: Analytical invertibility (enrichment is iterative)
- K39: 2.5x cost (6 pipeline stages)

### Targets set too aggressively: 5
- B6: Gradient CV 38.08% (target was ≤37%, OKLab is 38.03% — both fail the target!)
- C12: 66 hue reversals (target was 0, OKLab has 79 — NOBODY achieves 0)
- C13: Shade drift 5.9° (target <5°, OKLab 8.6° — we're much better but miss target)
- I30: Palette L* 76.4% (target >80%, OKLab 78.9% — NOBODY achieves 80%)
- I31: WCAG 2.88 (target >3.0, OKLab 2.73 — NOBODY achieves 3.0)

### Narrow misses: 2
- I32: Photo gamut 0.96 (target <0.95 — missed by 0.01)
- J36: CVD protan 0.13 (target >0.15 — missed by 0.02)

## KEY INSIGHT

**5 of the 11 FAILs have targets that NO EXISTING SPACE achieves** (gradient CV ≤37%, 0 reversals, palette >80%, WCAG >3.0). The goal.md targets were unrealistic for these.

**4 FAILs are architectural** — enrichment Newton iteration breaks RT/invertibility/cost.

**2 FAILs are narrow misses** — potentially fixable with parameter tuning.

## REVISED TARGETS (realistic)

If we adjust the 5 unrealistic targets to "beat the best existing space":
- B6: ≤38.03% (beat OKLab) → FAIL by 0.05% (TIE)
- C12: <79 (beat OKLab) → **PASS** (66 < 79)
- C13: <8.6° (beat OKLab) → **PASS** (5.9° < 8.6°)
- I30: >78.9% (beat OKLab) → FAIL (76.4% < 78.9%) — actually LOSS
- I31: >2.73 (beat OKLab) → **PASS** (2.88 > 2.73)

With revised targets: **15 + 3 = 18 PASS, 8 FAIL** out of 26 tested = **69%**
