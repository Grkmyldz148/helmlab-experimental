# Rational Transfer ColorBench: 22-18-21 (Mixed Results)

**Date**: 2026-04-02
**Config**: OKLab M1/M2 + rational(a=3.8, b=2.2, c=5.0), no enrichment, no PW

## Score: 22-18-21 (vs production 39-5-17)

MUCH worse than production. But reveals important insights:

## What rational WINS (new wins not in production):
- Chroma preservation: **0.422** vs 0.414 (was TIE, now WIN!)
- Muddy gradients: **11** vs 12 (was TIE, now WIN!)
- Eased animation: **62.3** vs 64.1 (was LOSS/TIE, now WIN!)
- Jacobian condition: **6.11** vs 6.49 (was TIE, now WIN!)
- Cusp smoothness: **0.029** vs 0.805 (29x better!)
- Hue leaf constancy: **20.0** vs 73.3 (3.7x better!)

## What rational LOSES (new losses):
- Gray ramp: 1.39e-4 (terrible — M1/M2 not optimized for rational)
- Gradient CV: 39.43% (WORSE than OKLab — opposite of proxy!)
- Blue G/R: 1.307 (FAIL — proxy said 1.80, ColorBench says 1.31!)
- Munsell Value: 19.09% (terrible — no PW L_corr)
- Hue reversals: 213 (terrible — 3x worse than OKLab)
- Boundary continuity: 0.000 (broken — self-referential detection issue?)
- Channel mono: 8 violations

## KEY DISCREPANCY: Proxy vs ColorBench

| Metric | Proxy | ColorBench | Mismatch |
|--------|-------|------------|----------|
| Blue G/R | 1.80 | **1.31** | HUGE — proxy was wrong! |
| Gradient CV | ~20% | **39.43%** | HUGE — proxy was wrong! |
| Chroma amp | 2.8x | **3.73x** | proxy was wrong |

The proxy metrics used simplified CIE76 and 5 pairs. ColorBench uses CIEDE2000 and 3038 pairs.
The rational transfer looks good in proxy but FAILS in full evaluation.

## Root Cause
1. OKLab M1/M2 was designed for cbrt, not rational transfer
2. Rational changes the compression curve → M2 ab-rows are misaligned
3. No PW L_corr → Munsell terrible
4. No enrichment → Blue G/R drops (but proxy was wrong about this too)

## Conclusion
Rational transfer has POTENTIAL (new chroma/muddy/eased wins) but needs:
1. M1/M2 completely re-optimized for rational transfer
2. PW L_corr added back for Munsell
3. The proxy metrics are UNRELIABLE — only trust ColorBench

## Next: M1/M2 CMA-ES optimization with rational transfer + ColorBench in loop
This is a multi-day project, not a quick fix.
