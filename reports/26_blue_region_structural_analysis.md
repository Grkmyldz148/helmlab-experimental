# Blue Region Gamut Fold — Structural Analysis

**Date**: 2026-04-02
**Contributors**: Isaac (color.js), lloydk, facelessuser, Björn Ottosson (cited), Claude analysis

## The Problem

sRGB gamut in OKLab-family spaces (M1→f→M2) has non-contiguous regions near the blue primary (h≈264°). The gamut boundary "folds inward," creating interior holes where in-gamut colors are surrounded by out-of-gamut gaps.

## Root Cause: 4-Layer Analysis

### Layer 1: Physical — sRGB Blue Primary Asymmetry

sRGB blue (0,0,1) has extreme cone response imbalance:

```
XYZ = [0.180, 0.072, 0.950]
Y (luminance) = 7.2% of total light
LMS = [0.051, 0.107, 0.630]  (OKLab M1)
S/L ratio = 12.3x — blue is almost entirely S-cone
```

Red and green are much more balanced across cones. This asymmetry is a physical property of human vision — short wavelength light strongly stimulates S-cones but contributes very little to luminance.

### Layer 2: Architectural — M1→cbrt→M2 Creates Cubic Polynomial

When channels are mixed by M1 before cube root, the inverse mapping creates a cubic polynomial for each RGB channel as a function of chroma:

```
R(C) = A₀ + A₁C + A₂C² + A₃C³
```

A cubic can have 2 positive roots. When this happens, R goes negative between them → gamut gap.

**Mathematical proof** (OKLab, h=264.1°, L=0.40):
```
R(C) = 0.064 - 0.475C + 0.761C² + 0.431C³
Roots: C₁=0.238, C₂=0.274
Gap width: 0.036 chroma units
```

Between C₁ and C₂, R is negative → out of sRGB gamut.

**Why only blue?** cbrt amplifies blue's tiny L-cone value (0.051) by 7.2x but large S-cone (0.630) by only 1.4x. This differential amplification distorts the cubic polynomial coefficients specifically in the blue region.

### Layer 3: Transfer Function Interaction

| Transfer | f'(0) | Blue fold severity |
|----------|-------|-------------------|
| CIE Lab (diagonal M1) | finite | **None** — no M1 mixing |
| OKLab cbrt (γ=1/3) | ∞ | **46 holes** |
| OKLab (Björn's γ=0.323) | ∞ | **Larger fold** (pre-optimization) |
| IPT (γ=0.43) | ∞ | **176 gap width** (3.7x worse than OKLab) |
| Jzazbz (PQ rational) | finite | **0 holes** (PQ prevents fold) |
| HelmGen depcubic (α=0.022) | 1/α≈45 | **3 holes** (finite derivative helps) |

Björn Ottosson's own words:
> "The gamma value ended up very close to 1/3, 0.323, and... By forcing the value of gamma to 1/3 and adding a constraint the blue colors to not fold inwards, the final Oklab model was derived."

Smashing Magazine interview:
> "sRGB in Oklab has a strange shape, so it's easy to end up going outside it."

### Layer 4: M2 Matrix Mitigation

HelmGen's CMA-ES-optimized M2 matrix reorders the channel zero-crossings so that the G channel limits the gamut before R goes negative. This reduces the fold from OKLab's 46 holes to 3.

## Empirical Verification (this session)

Full 360° gamut hole scan (L step=0.01, C step=0.001):

```
OKLab:   46 holes across h=209°-264° (55° range)
         Worst: h=264° with 9 holes

HelmGen: 3 holes across h=261°-264° (3° range)
         Each ~0.001 chroma width
```

### Cross-Space Comparison

| Space | M1 mixing | Non-contiguous? | Holes/Gap | Notes |
|-------|-----------|-----------------|-----------|-------|
| CIE Lab | No (diagonal) | No | 0 | No fold, but bad hue linearity |
| OKLab | Yes | Yes | 46 holes | Björn accepted as trade-off |
| IPT | Yes | Yes | Gap=0.176 | 3.7x worse than OKLab |
| Jzazbz | Yes | No | 0 | PQ rational function prevents fold |
| HelmGen | Yes | Barely | 3 tiny holes | Best in M1→power→M2 class |

## Structural Theorem

> In M1→f()→M2 architecture where M1 is a full 3×3 matrix and f() is a power function with γ≤1/3, non-contiguous gamut near the blue primary is **mathematically inevitable**.

Escape routes:
1. **Diagonal M1** (CIE Lab approach) → 0 holes, but lose hue linearity
2. **γ≥0.40** → less fold, but worse lightness uniformity
3. **Rational transfer** (Jzazbz PQ) → 0 holes, but complex/HDR-specific
4. **Optimized M2** (HelmGen approach) → minimize fold to 3 tiny holes

## Community Consensus

- **Chris Lilley (W3C)**: Closed color.js #81 as "not a bug"
- **facelessuser**: "Confined to a very small hue range, probably not a huge deal"
- **Isaac**: "The way blue has to bend to fit the data, it makes sense why OKLab had similar issues"
- **Affected range**: Only 0.14° hue width in OKLab, R channel maximum -0.0013 (0.32/255)

## Impact on HelmGen

### Practical impact: NONE
- 3 holes, each ~0.001 chroma wide, at h=261°-264°
- Invisible to human perception
- Binary search gamut mapping clips correctly at first boundary
- ColorBench measures 0 holes (coarser grid)

### Metric impact: POSITIVE
- HelmGen 3 holes vs OKLab 46 → strong WIN on gamut geometry
- 360 cusps vs OKLab 299 → WIN
- Cusp smoothness 0.088 vs 0.805 → 9x better → WIN
- Boundary continuity 0.298 vs 0.545 → WIN

### Isaac's key insight:
> "It wouldn't surprise me if you could find similar quirks in ICtCp or JzCzhz"

Nobody has done this detailed analysis on other spaces yet. This could be a research contribution — systematic gamut fold analysis across all major perceptual color spaces.

## References

- color.js #81: sRGB gamut boundary in OKLCH
- w3c/csswg-drafts #7071: CSS Color 4 gamut mapping issues
- coloraide #102: OKLab vs CIE LCH gamut mapping
- coloraide #118: Gamut mapping LCH fallback discussion
- Björn Ottosson blog: https://bottosson.github.io/posts/oklab/
- facelessuser calc_oklab_matrices.py: https://github.com/facelessuser/coloraide/blob/main/tools/calc_oklab_matrices.py
