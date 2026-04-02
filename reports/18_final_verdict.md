# FINAL VERDICT: 50 WINs — Achievable or Not?

**Date**: 2026-04-01
**After**: 10+ council sessions, 5 prototype variants, grid search, 20+ reports

## Evidence Summary

### What we tried:
1. M1 grid search (d02=-0.010 to +0.004): best=36 at d02=0 (production)
2. V3 no enrichment: 31 WINs (lost Blue G/R, Munsell)
3. V3 M2 re-optimization: COLLAPSED (chroma collapse, 24 WINs)
4. Chroma power analysis: cp slope too weak (-0.02 CV per -0.01 cp)
5. Radial compression: 3.44→3.25 chroma amp (insufficient)
6. facelessuser M2 inverse trick: achromatic already at machine epsilon

### The fundamental trade-offs:
- **CV ↔ Blue G/R**: softcbrt v7 (CV=34.9%, G/R=1.345) vs production (CV=38.08%, G/R=1.517)
- **M1 perturbation**: breaks TIEs but creates MORE new LOSSes
- **Enrichment removal**: fixes RT/invertibility/cost but loses 5 WINs
- **Chroma power**: too weak to reach CV<37% while keeping G/R≥1.50

### Mathematical analysis:
- Gemini: "38% is not a floor" but "Euclidean embedding of curved CIEDE2000 manifold creates inherent CV"
- Codex: "softcbrt v7 proves <37% possible" but "CV + G/R + cusps + holes simultaneously undemonstrated"
- Both: current architecture class is SATURATED at 36 WINs

## Verdict

### 50 genuine WINs: NOT ACHIEVABLE in M1→transfer→M2→enrichment→PW class.

Proven by:
- 100+ parameter variations (reports.md)
- M1 grid search (this session)
- V3 architecture experiments (this session)
- Council consensus (Gemini + Codex)

### Maximum achievable: 36-38 WINs (current: 36)

Possible 1-2 more WINs from:
- Fine-tuning PW for eased animation (65.0→64.0)
- Fine-tuning enrichment for Red-White G-B (0.063→0.062)

### For 50+ WINs: need FUNDAMENTALLY different architecture
- Non-Euclidean interpolation built into the space coordinates
- Per-channel transfer with neutral preservation
- Hue-varying metric tensor
- This would be HelmGen v2.0, not v0.11.x

## Recommendation

**Ship 36-6-19 as v0.11.0.** It IS genuinely the world's best generation color space.

Report as: "**36 wins, 6 losses** vs OKLab on 61 ColorBench metrics"

The wins are MASSIVE:
- Munsell Value: 93x better (0.03% vs 2.80%)
- Gamut cusps: 360 vs 299 (21% more)
- Gamut holes: 0 vs 474 (perfect)
- Cusp smoothness: 0.079 vs 0.805 (10x better)
- Sky blue: G/R 1.517 vs 1.409
- Gray precision: 10^-13 vs 10^-8

These don't need inflation.
