# HelmGen Next — The Perfect Generation Color Space

**Goal:** Build the world's best generation color space. Every single metric must equal or exceed the best existing space for that specific property.

**Reference spaces:** OKLab, CIE Lab, CAM16-UCS, Jzazbz, IPT, ICtCp, CIECAM02, ProLab, HelmGen v0.10.1

**Principle:** "Perfect" doesn't mean "as good as OKLab." It means better than every space at what each does best.

**Reality check:** Human color perception follows Riemannian geometry, not Euclidean. Some criteria provably conflict (MacAdam isotropy vs iso-hue straightness). Where mathematical impossibility exists, we target the **provably optimal Pareto balance** — not perfection in isolation, but the best simultaneous trade-off ever achieved.

**Precision note:** All numerical targets assume float64 (f64) arithmetic unless stated otherwise. f32 implementations may relax targets by ~1e6 (e.g., 1e-14 → 1e-8).

---

## A. Gamut Geometry (4 criteria)

### 1. Gamut slice smoothness
Zero holes, zero spikes, zero concavity at ALL hues (0-360°), ALL gamuts (sRGB, P3, Rec.2020). Per-pixel scan at 2000×2000 resolution must show zero interior holes.

| Space | Holes | Target |
|-------|-------|--------|
| OKLab | 474 | — |
| HelmGen v0.10.1 | 121 | — |
| **HelmGen Next** | — | **0** |

### 2. Gamut boundary differentiability
C¹ continuous (first derivative continuous) at all gamut face transitions. Preferably C². No derivative discontinuities — this is the root cause of visible "spikes" in gamut slice viewers. Must pass visual inspection in gamut viewer at all hues.

### 3. Cusps
360/360/360 valid cusps in sRGB, P3, Rec.2020. Cusp smoothness (max jump between adjacent hues) < 0.1. No dead zones.

| Space | sRGB | P3 | Rec.2020 | Target |
|-------|------|-----|---------|--------|
| OKLab | 294 | 309 | 360 | — |
| HelmGen v0.10.1 | 360 | 360 | 360 | — |
| **HelmGen Next** | — | — | — | **360/360/360** |

### 4. Monotonicity after cusp
After cusp (max chroma), chroma must decrease monotonically toward white. Zero violations in any gamut.

| Space | Violations | Target |
|-------|-----------|--------|
| OKLab | 87 | — |
| HelmGen v0.10.1 | 1 | — |
| **HelmGen Next** | — | **0** |

---

## B. Gradient & Interpolation Quality (5 criteria)

### 5. Gradient out-of-gamut excursion
Minimum OOG during linear Lab interpolation between two in-gamut sRGB colors.

| Space | OOG pairs/1000 | Max distance | Target |
|-------|---------------|-------------|--------|
| OKLab | 18 | 0.40 | — |
| HelmGen v0.10.1 | 21 | 0.54 | — |
| **HelmGen Next** | — | — | **< 15, max < 0.30** |

### 6. Gradient uniformity (CV)
Coefficient of variation of CIEDE2000 step sizes across 3038 gradient pairs.

| Space | Mean CV | Target |
|-------|---------|--------|
| OKLab | 38.03% | — |
| HelmGen v0.10.1 | 38.31% | — |
| **HelmGen Next** | — | **≤ 37%** |

### 7. Interpolation path optimality
Interpolation path should approximate perceptual geodesic (minimum ΔE path), not just Euclidean Lab straight line. Average path ΔE vs theoretical geodesic ΔE ratio ≈ 1.0.

### 8. Multi-stop gradient quality
3-color and 4-color gradient CV comparable to 2-color. No pinch points where chroma collapses. Chroma preservation: no muddy/gray midpoints in vivid-to-vivid gradients.

### 9. Interpolation curvature continuity
Gradient path must have smooth curvature — no sudden curvature spikes. OKLab has curvature spikes at some hues. Measure: max curvature change between adjacent gradient steps.

---

## C. Hue Properties (5 criteria)

### 10. Hue preservation in gradients
- Blue→White midpoint = sky-blue (G/R ≈ 1.5), not purple
- Red→White midpoint = pink, not orange
- Yellow→White midpoint = cream, not green

| Space | Blue G/R | Target |
|-------|---------|--------|
| OKLab | 1.41 | — |
| HelmGen v0.10.1 | 1.51 | — |
| **HelmGen Next** | — | **≥ 1.50** |

### 11. Iso-hue line straightness
Constant-hue lines in Lab must be straight. If iso-hue bends, gradient hue drift is inevitable. Reference: IPT (near-straight iso-hue) — best in class for this.

### 12. Hue linearity under desaturation
dHue/dChroma ≈ 0 everywhere. Not just "no reversal" but no acceleration/deceleration of hue during chroma reduction. Zero reversals.

| Space | Reversals | Target |
|-------|----------|--------|
| OKLab | 79 | — |
| HelmGen v0.10.1 | 63 | — |
| **HelmGen Next** | — | **0** |

### 13. Shade palette hue consistency
10-shade palette for any base color: max hue drift < 5°.

| Space | Max drift | Target |
|-------|----------|--------|
| OKLab | 8.6° | — |
| HelmGen v0.10.1 | 5.8° | — |
| **HelmGen Next** | — | **< 5°** |

### 14. Hue perceptual angular velocity
Hue wheel rotation speed must be perceptually constant. Equal angular steps in Lab hue must correspond to equal perceived hue changes. No regions where hue "speeds up" or "slows down."

---

## D. Achromatic Axis (2 criteria)

### 15. Structural achromatic guarantee
a=b=0 for ALL grays — structural guarantee from shared transfer function, not numerical. Max achromatic chroma drift < 1e-10 across entire L range. Must be architecture-level guarantee.

| Space | Max chroma | Target |
|-------|-----------|--------|
| OKLab | 3.73e-8 | — |
| HelmGen v0.10.1 | 5.18e-9 | — |
| **HelmGen Next** | — | **< 1e-10** |

### 16. Gray ramp perceptual uniformity
Equal sRGB gray steps must map to perceptually equal L steps. Gray ramp step CV < 5%.

---

## E. Perceptual Uniformity (4 criteria)

### 17. Lightness uniformity (Munsell Value)
Overall: < 2%. Per-hue: < 3% at every hue (especially yellow).

| Space | Value | Target |
|-------|-------|--------|
| OKLab | 2.80% | — |
| HelmGen v0.10.1 | 2.01% | — |
| **HelmGen Next** | — | **< 2%** |

### 18. ΔE consistency across hues
ΔE error balanced across all hue regions. No blue-region bias. Variation < 5% across hue slices.

### 19. MacAdam ellipse isotropy
Local distance uniformity ratio. Note: perfect isotropy (1.0) provably conflicts with iso-hue straightness (criterion 11) due to Riemannian geometry of human color perception. Target is the best achievable Pareto balance.

| Space | Isotropy | Target |
|-------|---------|--------|
| OKLab | 1.99 | — |
| CIE Lab | ~2.5 | — |
| **HelmGen Next** | — | **< 2.0** (Pareto optimal with iso-hue) |

### 20. Munsell Hue spacing
Perceptually equal hue steps map to equal hue angles.

| Space | Spacing | Target |
|-------|---------|--------|
| OKLab | 18.5° | — |
| HelmGen v0.10.1 | 19.1° | — |
| **HelmGen Next** | — | **< 18°** |

---

## F. Numerical Properties (3 criteria)

### 21. Round-trip precision
Forward + inverse chain: < 10⁻¹⁴ max error. 1000-trip accumulation: < 10⁻¹². All 16.7M 8-bit sRGB colors: zero bit errors.

| Space | Max error | Target |
|-------|----------|--------|
| OKLab | 1.78e-15 | — |
| HelmGen v0.10.1 | 1.03e-6 | — |
| **HelmGen Next** | — | **< 1e-14** |

### 22. Numerical stability & conditioning
Extreme chroma amplification < 3x at all gamut primaries. Jacobian condition number: mean < 6, max < 40. NaN/Inf: never under any input.

| Space | Chroma amp | Jacobian mean | Target |
|-------|-----------|--------------|--------|
| OKLab | 5.79x | 6.49 | — |
| HelmGen v0.10.1 | 11.0x | 6.20 | — |
| **HelmGen Next** | — | — | **< 3x, < 6** |

### 23. Analytical invertibility
ALL stages analytically invertible. No Newton iteration required (or guaranteed convergence in < 5 iterations). Inverse must be exact, not approximate.

---

## G. Physical Consistency (3 criteria)

### 24. Energy monotonicity
L↑ ⟹ physical luminance Y↑ always. No non-monotonic L-luminance mapping anywhere. Critical for HDR, tone mapping, physically-based rendering. Reference: Jzazbz (PQ transfer) — best in class.

### 25. White point invariance
D65→D50 chromatic adaptation: max hue drift < 2° for any color. Critical for ICC profiles and print. Reference: CIECAM02 — best chromatic adaptation.

### 26. Observer model robustness
Stable under CIE 2° vs 10° standard observer change. Max Lab shift < 1% of gamut range.

---

## H. Temporal & Perceptual Robustness (3 criteria)

### 27. Temporal gradient stability
Same gradient at different sampling rates / resolutions must look identical. No color jitter in animation, video, scroll UI. Measure: max frame-to-frame ΔE at different frame rates.

### 28. Local contrast preservation
Small ΔL differences must remain perceptually visible across entire L range. Especially critical in dark mode (L < 0.2). No contrast collapse at extremes. WCAG is global — this is local.

### 29. UI semantic consistency
"Primary / success / error / warning" colors generated at equal L must have balanced perceived prominence. Saturation and hue should not create unintended hierarchy. (Application-level but space design affects it.)

---

## I. Application Robustness (3 criteria)

### 30. Palette L* spacing uniformity
Generated palettes have perceptually equal lightness steps.

| Space | Score | Target |
|-------|-------|--------|
| OKLab | 78.9% | — |
| HelmGen v0.10.1 | 75.1% | — |
| **HelmGen Next** | — | **> 80%** |

### 31. WCAG contrast accuracy
L-based contrast ratio correlates with WCAG luminance contrast.

| Space | Midpoint CR | Target |
|-------|-----------|--------|
| OKLab | 2.73 | — |
| HelmGen v0.10.1 | 2.90 | — |
| **HelmGen Next** | — | **> 3.0** |

### 32. Photo gamut mapping fidelity
Gamut mapping from P3/Rec.2020 to sRGB preserves perceived color. L shift < 1.0.

| Space | L shift | Target |
|-------|---------|--------|
| OKLab | 0.98 | — |
| HelmGen v0.10.1 | 1.03 | — |
| **HelmGen Next** | — | **< 0.95** |

---

## J. Accessibility & Wide Gamut (4 criteria)

### 35. APCA contrast correlation (WCAG 3.0)
WCAG 2.x contrast algorithm is mathematically flawed in dark mode. L-based contrast must correlate with APCA (Accessible Perceptual Contrast Algorithm), not just WCAG 2.x luminance ratio.

### 36. CVD robustness (Color Vision Deficiency)
Palettes generated in this space must maintain distinguishability under Protanopia, Deuteranopia, and Tritanopia simulation. Minimum pairwise ΔE under CVD simulation > 0.1 for any 8-step gradient.

| Space | CVD protan min ΔE | Target |
|-------|------------------|--------|
| OKLab | 0.13 | — |
| HelmGen v0.10.1 | 0.13 | — |
| **HelmGen Next** | — | **> 0.15** |

### 37. Gamut volume preservation
When mapping from Rec.2020 to P3 to sRGB, chroma compression must be proportional — no region should "explode" or "collapse" disproportionately. Especially Rec.2020 green must not oversaturate relative to other primaries.

### 38. Viewing condition adaptability
The space should define behavior under different surround luminance. At minimum: how L maps under dark surround (dark mode) vs average surround. Hunt effect (brightness increases perceived colorfulness) and Bezold-Brücke shift (brightness changes perceived hue) should be documented, even if not parametrically modeled in v1.

---

## K. Computational (2 criteria)

### 39. Performance cost
≤ 2x OKLab computational cost. Branchless implementation possible. SIMD / GPU / WebGPU friendly. No data-dependent branches in hot path.

### 40. Implementation simplicity
Entire forward+inverse fits in < 300 lines of code. No lookup tables > 256 entries. No iterative solvers in forward path.

---

## Known Trade-offs

These criteria can conflict — the optimization must balance them:

| Trade-off | Why | Mathematically proven? |
|-----------|-----|----------------------|
| MacAdam isotropy ↔ Iso-hue straightness | Human color perception is Riemannian, not Euclidean. Mapping curved space to flat coordinates forces distortion somewhere. | **Yes** — Gauss's Theorema Egregium |
| Uniformity ↔ Hue linearity | Perfect uniformity warps hue lines | Empirically observed |
| Geodesic paths ↔ Computational cost | Geodesic requires metric tensor integration | Mathematical fact |
| Gamut smoothness ↔ Hue accuracy | sRGB-aligned M1 gives smooth gamut but constrains hue | Empirically observed |
| Round-trip precision ↔ Transfer function choice | cbrt: 1e-15 but 294 cusps. softcbrt: 360 cusps but 1e-6 | Numerical fact |

**Where mathematical impossibility exists (Gauss), we target the provably optimal Pareto balance. Where it's just hard, we optimize until we beat everyone.**

---

## Architecture Candidates

1. **M1→transfer→M2** (OKLab family) — simple, proven, limited
2. **M1→transfer→M2→enrichment** (HelmGen family) — flexible, more params
3. **Dual-stage** (Jzazbz-style) — HDR capable, more complex
4. **CAM-based** (CAM16-UCS style) — adaptation aware, very complex
5. **Hybrid** (analytical base + learned correction) — best of both?

## Key Decisions Needed
- **Transfer function:** cbrt vs softcbrt vs PQ vs hybrid spline (linear near zero, cbrt above — like sRGB transfer but smoother)? Hybrid spline may be the key to C¹ continuity + high round-trip precision.
- **C¹ gamut boundary guarantee:** Must be formulated as analytical constraint, NOT just optimization objective. If left as loss function, optimizer will always leave a spike somewhere (proven by our v0.10.1 experience).
- **M1+M2 sufficient?** Almost certainly no. 34+ criteria with only 18 DOF (two 3×3 matrices) is insufficient. Enrichment stages or spatial warping needed. M1→transfer→M2→enrichment (HelmGen family) is minimum viable architecture.
- **Trade-off weights:** Multi-objective Pareto optimization with adaptive weights. Not fixed weights — use NSGA-II or similar to explore the Pareto front, then select the best point.

---

## Validation Pipeline

Every candidate model must pass:
1. **ColorBench** — all 61+ metrics, must beat OKLab h2h
2. **Gamut slice viewer** — visual inspection at ALL hues, ALL gamuts
3. **Per-pixel hole scan** — 2000×2000, zero holes
4. **Exhaustive 16.7M round-trip** — zero 8-bit errors
5. **Animation test** — 60fps gradient sweep, no jitter
6. **Dark mode test** — L < 0.2 contrast preservation, APCA correlation
7. **Multi-gamut test** — sRGB, P3, Rec.2020 all clean
8. **CVD simulation** — palette distinguishability under protanopia, deuteranopia, tritanopia
9. **f32 stability test** — all criteria re-verified at single precision

No model ships without passing ALL 9 validation stages.
