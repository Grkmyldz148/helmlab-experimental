# V3 M2 Optimization — FAILED (Chroma Collapse)

**Date**: 2026-04-01

## Result: 24-18-17 (REGRESSION from 36-6-19)

CMA-ES found amp=1.36x by collapsing the b-row to near-zero:
```
M2 b-row: [0.013, -0.002, -0.012]  ← essentially zero
```

This is the SAME failure mode as the previous session's M2 optimization.
The optimizer minimizes chroma amp by deleting the b-channel entirely.

## Metrics Destroyed
- Extreme chroma amp: 299x (was 3.79x) — the REAL amp metric measures P3/Rec2020 corners
- Munsell Hue: 188.9° (was 11.4°) — hue structure destroyed
- MacAdam: 5.44 (was 1.78) — isotropy destroyed
- sRGB cusps: 255 (was 360) — gamut broken
- Blue G/R: 1.38 (was 2.01) — blue lost

## Root Cause
The proxy metric (Jacobian at 6 primaries with ±0.001 perturbation) does NOT match
ColorBench's "Extreme chroma amplification" which tests P3/Rec2020 corner primaries
with much larger perturbations and different methodology.

## Lesson Learned (AGAIN)
1. NEVER optimize M2 without ColorBench-in-the-loop
2. chroma_amp proxy ≠ ColorBench extreme chroma metric
3. b-row collapse is the CMA-ES attractor for amp minimization
4. Need HARD constraint: det(M2_ab) > threshold, or σ_min(M2_ab) > 0.1

## Next Step
Fix optimizer with:
1. σ_min(M2[1:3,:]) > 0.3 hard gate
2. Use ColorBench extreme chroma metric directly (expensive but correct)
3. OR: use Codex's structured parameterization R(θ)·diag(exp(s1),exp(s2))·shear(k)·OKLab_ab
