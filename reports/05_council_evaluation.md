# Council Evaluation & Final Decision

**Date**: 2026-04-01
**Lead Architect**: Claude (Opus 4.6)

## Council Summary

### Gemini (Flash 3.1):
- Proposed concrete parameter changes: ck=0.105, depcubic_d=-0.278, cp=0.815
- **Problem**: Referenced OLD pipeline (v9 with cross_term + chroma_power). Current depcubic pipeline has NONE of these params.
- **Verdict**: Proposals INVALID — wrong pipeline.

### Codex (GPT-5.4):
- "Current class ceiling is about 40-42 genuine wins"
- "reports.md already killed the obvious knobs: 100+ perturbations, cp×ck flat, 3rd harmonic worse, L_corr worsened tails"
- "No exact safe parameter from data" for most TIE flips
- **Verdict**: Realistic assessment. Agrees 36→42 is hard, 50 is impossible in this class.

## My Decision

### 1. What is the HONEST maximum WIN count?

**36-6-19 is the real score.** With targeted work:
- 3-4 TIE flips possible (chroma pres, OOG, muddy grads, maybe eased anim)
- 1-2 LOSS flips possible (Red-White, Eased anim — low confidence)
- **Realistic maximum: 39-42 WIN, 4-6 LOSS, 13-18 TIE**

### 2. What should the README say?

**HONEST OPTIONS:**

Option A (conservative): "36 wins vs OKLab on 61 metrics (6 losses, 19 ties)"
Option B (with context): "36-6 vs OKLab (19 ties including 6 where both achieve perfection)"
Option C (category focus): "Dominates OKLab in gamut (10-0), perceptual (5-0), structural (4-2)"

**RECOMMENDATION: Option B** — shows the 36 direct wins AND explains the 6 ceiling ties. Honest, impressive, defensible.

### 3. Action plan for improvement

**Phase 1 — Quick wins (1-2 days):**
- Fix ColorBench `genenriched` class to support depcubic (so `genenriched` gives same result as `helmct`)
- Tune chroma preservation via enrichment sigma fine-tuning
- Try PW breakpoint shift for eased animation CV

**Phase 2 — M1 search (1 week):**
- Grid search M1 perturbation with FULL 61-metric ColorBench as gate
- Hard constraint: no metric regression from current 36 WINs
- Soft target: flip gradient CV, banding, multi-stop, muddy, CVD protan TIEs
- Realistic gain: +3-6 WINs

**Phase 3 — Architecture (future):**
- Geodesic interpolation for gradient() function
- Per-channel transfer for CVD improvement
- Non-linear M2 for primary hue disc

### 4. The "50-6" claim

**Must be corrected everywhere:**
- README.md: 50→36 (or 42 after Phase 2)
- Landing page: same
- MEMORY.md: update score
- Blog posts: if published, need correction

This is not optional. Publishing false benchmarks damages credibility.

## Metric Scorecard: What's Actually Impressive

Even at 36-6-19, HelmGen Next is genuinely the best generation color space:
- **Munsell Value**: 93x better than OKLab (0.03% vs 2.80%)
- **Gamut**: 360 cusps vs 299, zero holes vs 474, 10x smoother cusps
- **Blue→White**: Sky blue G/R=1.517 vs purple-ish 1.409
- **Gray precision**: 10^-13 vs 10^-8 (100,000x better)
- **Round-trip**: 12x better on 1000-trip stability

These wins are MASSIVE and don't need inflated numbers to be impressive.
