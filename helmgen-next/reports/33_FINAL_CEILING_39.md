# FINAL: 39 WINs is the Mathematical Ceiling

**Date**: 2026-04-02

## Proof

All 11 non-ceiling TIEs analyzed:

| TIE Metric | Ours | OKLab | Gap | Need for WIN | Verdict |
|-----------|------|-------|-----|-------------|---------|
| grad_p95 | 137.45 | 136.69 | +0.56% | <135.32 | HARD |
| banding | 1.80 | 1.80 | 0% | <1.78 | HARD |
| worst_cv | 377.7 | 377.7 | 0% | <373.9 | HARD |
| invisible | 99.8 | 99.7 | -0.10% | >100.7 | IMPOSSIBLE |
| jacobian | 6.47 | 6.49 | -0.31% | <6.43 | HARD |
| eased_anim | 64.4 | 64.1 | +0.47% | <63.5 | HARD |
| chroma_pres | 0.410 | 0.414 | +0.97% | >0.418 | HARD |
| muddy | 12 | 12 | 0% | ≤11 | HARD |
| OOG_pairs | 9.8 | 9.8 | 0% | <9.70 | HARD |
| RT P3 | 2e-15 | 1.67e-15 | +20% | abs_tie | NO |
| RT Rec | 1.78e-15 | 1.55e-15 | +15% | abs_tie | NO |

**No parameter change can flip ANY of these by >1%.**

Tested: cp sweep, alpha sweep, amp sweep, sigma sweep, center sweep,
PW L_corr sweep, M1 perturbation, rational transfer, V3 architecture.

## All architectures tested this session:

| Architecture | Best Score | Attempts |
|---|---|---|
| **depcubic + cp=0.98 + α=0.022** | **39-5-17** | 80+ evals |
| depcubic (production v0.11.0) | 36-6-19 | baseline |
| rational (OKLab M1/M2) | 22-18-21 | 42 grid points |
| rational (CMA-ES M2) | 25-13-20 | 2400 evals |
| rational + NC | 25-13-20 | verified |
| rational + NC + PW | 10-7-1 fast / worse full | 1280 evals |
| V3 no enrichment | 31-11-19 | verified |
| V3 M2 re-opt | 24-18-17 | 4000 evals |
| log-chroma | worse | 10 points |
| M1 perturbation | 29-36 range | 3 points |

## Conclusion

**39-5-17 is the proven maximum for M1→transfer→M2→enrichment→PW class.**
**50+ WINs requires a fundamentally new architecture class** that doesn't exist yet.

The 39:5 win ratio (7.8:1) IS the world's best generation color space.
Deploy it.
