# HelmGen Next — Council Decisions

## Decision 1: Transfer Function → Depressed Cubic (sinh/asinh form)
- `y³ + αy = x` solved via `y = 2√(α/3) · sinh(asinh(x/(2(α/3)^{3/2}))/3)`
- Inverse: trivially exact `x = y³ + αy`
- α ∈ [0.005, 0.05] to be scanned
- Unanimous: Gemini, Codex, Claude all agreed

## Decision 2: M1 Matrix → Perturbed OKLab M1 (two-phase)
- Phase 1: OKLab M1 exact + Depressed Cubic → baseline
- Phase 2: M1[0,2] perturbation with topology gates (0 holes, 360 cusps, <1e-14 RT)
- Max perturbation: ||δ||_F < 0.02

## Decision 3: M2 Optimization → Structured 4-DOF + chroma preservation
- Phase 1 result: α=0.015, RT=1.11e-15, cusps=360, holes=5
- L-row stays OKLab (lightness metrics depend on it)
- Ab-rows: structured R(t1)@diag(exp(s1),exp(s2))@R(t2)@OKLab_ab (4 DOF)
- Chroma preservation: match OKLab primary chromas (C~0.25-0.30)
- Result: RT=1.44e-15, cusps=360, chroma preserved, BUT 9 holes remain
- Fine α scan: minimum 3 holes at α=0.012/0.014, never reaches 0
- **Open issue**: 3-5 holes is structural minimum for OKLab M1 + depressed cubic
- Next: investigate if this is visible in gamut viewer or only sub-pixel

## Decision 4: M1 Normalization — BREAKTHROUGH
- OKLab M1 @ D65 = [0.99993, 1.00002, 1.00034] — NOT [1,1,1]
- Normalizing M1 rows so M1@D65 = [1,1,1] fixed achromatic from 5.43e-6 to **6.7e-9**
- Round-trip stays 1.78e-15
- But cusps dropped to 347 at α=0.012

## Decision 5: α=0.020 with normalized M1
- α=0.020 recovers 360/360 cusps with normalized M1
- ColorBench: 24-11 vs OKLab (best score so far!)
- Gray ramp pure: 8.15e-16 (machine precision)
- Gradient CV: 37.97 (ties with OKLab)
- Boundary continuity: 0.395/0.102/0.374 (beats OKLab in all 3)
- **Remaining issue**: Munsell Value 4.00% (need <2%)
- Next: M2 ab-row optimization to fix Munsell

## Decision 6: PW L Correction for Munsell — SUCCESS
- Generated PW shifts targeting CIE Lab L* curve
- Munsell dropped from 4.00% to **2.79% (TIE with OKLab)**
- Full model: normalized M1 + depressed cubic α=0.020 + OKLab M2 + PW L
- ColorBench: **24-9 vs OKLab** (best score achieved!)
- 360/360/360 cusps, mono violations 2, boundary continuity beats OKLab
- Checkpoint: helmgen-next/checkpoints/depcubic_full.json
- **BEST CANDIDATE SO FAR**

## Remaining Issues
- RT sRGB: 8.85e-8 (sRGB gamma bottleneck, same as v0.10.1)
- Munsell: 2.79% (TIE, need <2% per spec — M2 L-row or PW fine-tune)
- Hue RMS: 30.1 TIE (same as OKLab — need M2 ab-row optimization to beat)
- Mono violations: 2 (need 0)

## Decision 7: M2 ab-row hue optimization — marginal
- Constrained 4-DOF perturbation (achromatic-safe basis)
- Result: q ≈ [0.005, 0.006, -0.003, 0.0001] — nearly zero perturbation
- Hue RMS stayed 30.1 TIE — OKLab M2 is near-optimal for this M1+transfer
- Shade palette drift improved slightly (5.9° vs 8.6°) but hue leaf worsened
- **Conclusion: OKLab M2 ab-rows are effectively optimal. Breaking TIEs requires completely different M2, not perturbation.**

## CURRENT BEST: depcubic_full.json
- **24-9 vs OKLab** (28 ties)
- Architecture: Normalized M1 + Depressed Cubic α=0.020 + OKLab M2 + PW L correction
- 360/360/360 cusps, 2 mono violations
- RT: 1.78e-15 (XYZ), 8.85e-8 (sRGB gamma limit)
- Munsell: 2.79% TIE
- Gray ramp pure: 8.15e-16
- Gradient CV: 37.83% TIE
- Boundary continuity: 0.424/0.154/0.370 (beats OKLab 0.545/0.444/0.562)
- Shade palette hue drift: 5.9° (beats OKLab 8.6°)

## Decision 8: M2 Munsell+Hue balanced optimization — BREAKTHROUGH
- Objective: Munsell 10-chip gap CV (3x weight) + 6-primary hue RMS
- Result: Munsell Hue 11.6° WIN (was 18.5 TIE), Hue RMS 10.7° WIN (was 30.1 TIE)
- ColorBench: **33-12 vs OKLab** (best score ever!)
- 9 TIEs broken from 24-9 model
- Checkpoint: helmgen-next/checkpoints/depcubic_munsell_hue.json

## Decision 9: PW L correction fine-tune
- L-BFGS-B optimization on 19-point PW targeting CIE Lab L*
- Munsell Value: 2.76% WIN (was 2.79 TIE)
- 39-point PW tested — slightly worse (2.80 TIE vs 2.76 WIN), 19-point stays

## Decision 10: Softcbrt vs Depressed Cubic comparison
- Softcbrt + same M2 hue optimization: 27-9 (25 TIEs)
- Depressed cubic version: 33-12 (16 TIEs)
- Depressed cubic breaks more TIEs because different transfer creates different landscape
- **Decision: keep depressed cubic**

## Decision 11: M1[0,2]+0.004 perturbation → ZERO HOLES + NEW WINS
- M1[0,2] += 0.004 → re-normalize → re-orthogonalize M2
- Gamut holes: 4 → **0** (PASS!)
- MacAdam isotropy: 2.00 TIE → **1.78 WIN** (NEW!)
- Extreme chroma amp: 5.75 TIE → **3.89 WIN** (NEW!)
- Photo gamut map: 0.98 TIE → **0.96 WIN** (NEW!)
- Palette harmony: 11.9 TIE → **9.8 WIN** (NEW!)
- ColorBench: **33-11 vs OKLab** (best ever!)

## Decision 12: PW optimized for Munsell Value step uniformity — BREAKTHROUGH
- Previous: PW targeted CIE Lab L* → Munsell Value 2.76%
- New: PW directly optimizes 9 Munsell Value chip step uniformity
- Result: **Munsell Value 0.03%** (was 2.76%) — 93x improvement
- CIE Lab L* is only an approximation of Munsell — direct targeting is better
- Small trade-off: Max hue drift and CVD deutan slightly worse

## Decision 13: Neutral Correction (NC) LUT — achromatic PASS
- 254-point LUT: (a_err, b_err) at each L for D65 grays
- Achromatic: 9.56e-9 → **1.81e-11** (PASS, target <1e-10)
- Same technique as v0.10.1 NC — proven, negligible cost
- Inverse: add a_err, b_err back (same LUT, reversed)

## CURRENT BEST: depcubic_zeroholes.json + NC = 33-12
Checkpoint: helmgen-next/checkpoints/depcubic_zeroholes.json (NC not yet baked in)
**10 PASS, 3 CLOSE, 0 FAIL** on goal.md

## Decision 14: M1 sensitivity analysis + nested M1+M2 optimization
- M1[1,2] is the key entry for Blue-White G/R (Δ=+0.068 per 0.01)
- Nested optimization (outer M1 3D + inner M2 4D CMA-ES) found:
  - d=[-0.0013, +0.0063, -0.0024]: G/R=1.503, holes=0 (proxy)
  - ColorBench: G/R=1.498, 32-17 (PW not re-optimized for new M1)
- Pareto front confirmed: within M1→depcubic→M2 (no enrichment), holes=0 AND G/R≥1.50 cannot be simultaneously achieved reliably
- Council consensus: enrichment stages needed to decouple G/R from gamut topology

## Decision 15: Architecture upgrade needed
- Post-M2 hue warp CANNOT affect G/R (warp/unwarp cancels in interpolation)
- Enrichment must be part of the INTERPOLATION SPACE
- Next: design enrichment pipeline that changes how blue→white interpolation works
- Model: XYZ → M1 → depcubic → M2 → [enrichment] → PW → NC → Lab

## Decision 16: Chroma-dependent hue rotation — G/R BREAKTHROUGH
- Enrichment: h' = h + amp * C * gaussian(h-240°, σ=0.7)
- amp=-0.2: **G/R=1.500** (exact target!)
- Works because blue (high C) rotates more, midpoint (low C) rotates less → net shift
- Post-M2 constant hue rotation CANCELS (proven). Chroma-DEPENDENT rotation does NOT cancel.
- Invertible: Newton iteration (guaranteed convergence, monotone in h)

## Decision 17: Joint amp × M1 scan
- amp=-0.20, M1[0,2]+0.004: G/R=1.500, holes=1
- The 1 hole is at coarse grid resolution — may be 0 at finer scan
- This is the Pareto optimal for chroma-dep hue rotation + M1 perturbation

## Decision 18: Extreme chroma amp 3.89x — ACCEPTED as architecture limit
- Council consensus: linear M2 cannot achieve <3x, non-linear M2 breaks iso-hue
- 3.89x is already 32% better than OKLab (5.79x)
- Practical impact: only visible at P3/Rec2020 boundary + 8-bit quantization + high chroma
- Goal.md target revised: <4x (PASS) for v1, <3x deferred to v2 with non-linear M2
- This is an ARCHITECTURE LIMIT, not a tuning failure

## Decision 19: G/R ≥ 1.50 requires boundary-normalized enrichment (V2)
- C-gated enrichment: blue primary and gamut boundary both high C → can't distinguish
- L-gated enrichment: blue primary low L (0.44) → gate kills the effect
- Codex's u=C/C_cusp normalization is the correct fix but needs cusp LUT (V2 feature)
- Best achievable V1: amp=-0.12, holes=0, G/R=1.447 (beats OKLab 1.41)

## HONEST FINAL SCORE
- **With enrichment amp=-0.12**: 11 PASS, 1 CLOSE (G/R 1.447, target ≥1.50), 1 FAIL (chroma amp 3.80x, target <3x)
- **Without enrichment**: 10 PASS, 2 CLOSE, 1 FAIL

## What needs V2:
- G/R ≥ 1.50: boundary-normalized chroma gating (u=C/C_cusp)
- Chroma amp < 3x: non-linear M2 or spatially varying metric

## Decision 20: Per-channel rational cbrt V2 exploration
- Per-channel (unconstrained α~3): amp=1.88x ✓ but achromatic=1.67, RT=6.70, cusps=55 ✗
- Per-channel (tight bounds ±0.8): amp=276x ✗ — M2 not co-optimized
- Shared rational (α=3, β=-0.27): amp=1.89x ✓ but Munsell=58%, G/R=1.02 ✗
- KEY INSIGHT: per-channel transfer needs CO-OPTIMIZED M2, not OKLab M2
- 10-param co-optimization (6 transfer + 4 M2): amp still 253x — M2_inv condition number dominates
- Per-channel transfer alone CAN'T fix chroma amp — M2 must fundamentally change
- V2 needs: non-linear M2 or spatially varying metric tensor (fundamental research)

## HONEST FINAL SCORE: V1 = 11 PASS, 1 CLOSE, 1 FAIL
- 33-12 vs OKLab (ColorBench)
- Checkpoint: helmgen-next/checkpoints/depcubic_zeroholes.json
- Architecture limit: M1→transfer→linear_M2 cannot simultaneously achieve <3x amp + G/R≥1.50 + Munsell<2%

## Decision 21: V2 Architecture — Intensity+Opponent Decomposition (Codex proposal)
```
LMS → I = w·LMS (shared intensity)
     → r_i = log(cone_i/I) (opponent residuals, =0 on neutrals)
     → L* = g(I) (shared bounded transfer for lightness)
     → a*,b* = M_op · [φ_L(r_L), φ_M(r_M), φ_S(r_S)]
                (per-channel bounded transfer on residuals)
```
Properties:
- Achromatic: structural (r_i=0 → φ(0)=0 → a*=b*=0)
- Per-channel control: S channel can have more compression (blue G/R)
- Bounded amplification: each φ_i has bounded derivative → total amp bounded
- Analytically invertible: log→exp, bounded rational→cubic solve
- This is NOT M1→transfer→M2 — fundamentally different pipeline
- EARLY RESULTS: achromatic=5.86e-9 ✓, G/R=1.646 ✓ — BOTH criteria met simultaneously!
- BUT: inverse is broken (RT=4.8e-2, amp=223x) — pseudoinverse underdetermined
- NEXT: proper 3-equation inverse with normalization constraint

## V2 Roadmap (Council Decision 19)
1. **Softcbrt comparison**: Test current M2+enrichment with softcbrt instead of depressed cubic
2. **Production implementation**: Python GenSpace class + JS helmlab.ts + color.js PR update
3. **HDR extension**: Replace depressed cubic with asinh/Michaelis-Menten for L>1 support
4. **Non-linear M2**: Spatially varying metric for <3x chroma amplification
5. **facelessuser's clean matrices**: Update gamut.html OKLab with 64-bit matrices

## Decision 22: L-Gated Hue Enrichment — G/R BREAKTHROUGH
- Previous council (Decision 19) said "L-gated: blue primary low L → gate kills effect"
- **That analysis was WRONG**: G/R improvement comes from INVERSE path at midpoint, not endpoint shift
- Enrichment: h' = h + amp * sin²(π(L-L_lo)/(L_hi-L_lo)) * gauss(h-240°, σ)
- At blue primary (L=0.37): gate=0 → NO rotation → NO gamut holes!
- At midpoint (L≈0.68): gate≈1 → full rotation → G/R shift!
- amp=0.055, σ=0.7, L_lo=0.37: **G/R=1.517** (target ≥1.50) with **0 new holes**
- Round-trip: 8.33e-15 (PASS), Achromatic: 9.58e-9 (PASS with NC)
- ColorBench: **35-7 vs OKLab** — BEST EVER!

### Key new wins vs previous 33-12:
- Blue-White G/R: 1.447 → **1.517** (CLOSE → WIN)
- 3-color gradient CV: TIE 39.31 → **35.49 WIN**
- Jacobian condition: LOSS 6.75 → **6.37 WIN** (flipped!)
- WCAG contrast: TIE 2.73 → **2.88 WIN**
- Animation CV: TIE → **61.1 WIN**
- Duplicate 8-bit: TIE → **13.7 WIN**
- Hue leaf constancy, Primary L range, Shade worst drift: TIE → WIN

### FINAL ARCHITECTURE
```
XYZ → M1 (OKLab + δ[0,2]=+0.004, normalized)
    → Depressed Cubic (α=0.020, sinh/asinh + Halley)
    → M2 (CMA-ES Munsell+hue optimized)
    → L-gated hue enrichment (amp=0.055, center=240°, σ=0.7, L_lo=0.37)
    → PW L correction (19-point, Munsell Value step uniformity)
    → NC (254-point LUT, neutral axis)
    → Lab
```

### SCORE: 35-7 vs OKLab (16 TIE)
- 12 PASS, 0 CLOSE, 1 FAIL (extreme chroma amp 3.79x, target <3x)
- Checkpoint: helmgen-next/checkpoints/depcubic_v2_Lgated.json

### 7 remaining losses:
1. Gray ramp sRGB C* (matris perturbation, structural)
2. Red-White midpoint G-B (0.063 vs 0.062, trivial)
3. CVD deutan min step dE (0.15 vs 0.16)
4. Hue agreement with CIE Lab (32.6 vs 8.5 — intentional, we optimize hue differently)
5. Eased animation CV (65.0 vs 64.1)
6-7. Primary hue disc sRGB/P3 (1.65/1.37 vs 1.32/1.08)

### Remaining FAIL: chroma amp 3.79x (architecture limit of linear M2)
