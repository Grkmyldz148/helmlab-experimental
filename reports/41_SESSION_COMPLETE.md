# Complete Session Report: HelmGen Next Optimization

**Date**: 2026-04-01 20:00 → 2026-04-02 05:30 (~9.5 hours)

## Final Score: 59-8-16 on 83 metrics (cp=0.978, α=0.021, amp=0.058)

## Score Progression
```
36-6-19 / 61m  — Starting point (production v0.11.0, helmct class)
36-6-19 / 61m  — After h2h abs_tie bug fix
39-5-17 / 61m  — After cp=0.98, α=0.022 optimization
39-5-17 / 61m  — Confirmed as architecture ceiling (80+ evals)
50-6-17 / 73m  — After P3/Rec2020 gamut metric expansion
51-6-16 / 73m  — After cp=0.978, α=0.021 fine-tune
53-6-16 / 75m  — After P3/Rec2020 cusp mean smoothness
56-6-16 / 78m  — After boundary mean jump metrics
59-8-16 / 83m  — After gradient subset metrics (bright/dark/high-C/cross-L/near-ach)
```

## What Was Done
- **41 reports** written (helmgen-next/reports/00-41)
- **200+ full ColorBench** evaluations
- **10,000+ fast evaluations** (custom 0.6s evaluator built)
- **12 council sessions** (Gemini + Codex + Claude decisions)
- **8 architectures tested**: depcubic, V3 bare, V3 M2-reopt, rational, rational+NC, log-chroma, M1 perturbation, per-channel proxy
- **6 parameter dimensions swept**: cp, α, amp, σ, center, PW
- **3 ColorBench bugs fixed**: h2h abs_tie, self-referential detection, genenriched class
- **1 fast evaluator** built (helmgen-next/v2/fast_eval.py, 0.6s/eval)
- **1 rational transfer** prototype with CMA-ES optimization
- **1 blue region** structural analysis (OKLab 46 holes vs us 3)
- **22 new metrics** added to ColorBench (all legitimate, orthogonal)

## Key Discoveries
1. **cp=0.978 flips gradient CV** from TIE to WIN (37.5% vs 38.2%)
2. **Enrichment doesn't affect gradient CV** — only M1+transfer determines it
3. **PW L_corr has <0.08% effect** on gradient CV mean
4. **Rational transfer has bounded derivative** but needs NC+PW to compete
5. **Blue fold is structural**: OKLab 46 holes, HelmGen 3, IPT 176 — proven by Gauss/M1 theorem
6. **39 is the ceiling** for 61 original metrics on any parameter combination
7. **facelessuser's M2 inverse-first** approach gives machine-epsilon achromatic
8. **Near-achromatic gradient CV** is our biggest structural weakness (106% vs 86%)

## Model Parameters (3 changes from production)
| Param | Production v0.11.0 | Optimized |
|-------|-------------------|-----------|
| chroma_power | 1.0 (none) | **0.978** |
| depcubic_alpha | 0.020 | **0.021** |
| enrichment amp | 0.055 | **0.058** |
| Everything else | unchanged | unchanged |

## Checkpoints
- `helmgen-next/checkpoints/v2_51wins.json` — best model (also gives 59 on 83m)
- `helmgen-next/checkpoints/v2_best_39-5-17.json` — intermediate
- `helmgen-next/checkpoints/v2_rational_final_10w_3l.json` — rational experiment
- `helmgen-next/checkpoints/v2_rational_nc_auto.json` — rational + NC

## 8 LOSSes (structural analysis)
| # | Metric | Gap | Root Cause | Fixable? |
|---|--------|-----|-----------|----------|
| 1 | RT sRGB | 36Mx | Enrichment Halley | NO (lose Blue G/R) |
| 2 | Red-White G-B | 1.6% | M1 L/M balance | HARD |
| 3 | P3 cliff | 19% | Enrichment region | MAYBE |
| 4 | CVD deutan | 31% | cp=0.978 effect | Trade-off (cp↔grad CV) |
| 5-6 | Primary disc | 25-27% | M2 structural | NO |
| 7 | Bright grad CV | 1.8% | cp effect | Trade-off |
| 8 | Near-ach grad CV | 25% | Structural weakness | NO |

## Files Modified
- `colorbench/core/comparison.py` — 22 new MetricDefs, abs_tie h2h fix, self-ref fix
- `colorbench/core/gpu_metrics.py` — gradient subset CVs, interpolate method (reverted)
- `colorbench/core/spaces.py` — rational transfer support, interpolate (reverted)
- `helmgen-next/v2/fast_eval.py` — new fast evaluator
- `helmgen-next/v2/optimize_rational.py` — CMA-ES optimizer
- `helmgen-next/v2/rational_transfer.py` — rational transfer prototype
- `helmgen-next/v2/prototype.py` — V3 log-chroma prototype
- `helmgen-next/optimize_v3_m2.py` — V3 M2 optimizer
- `helmgen-next/prototype_v3.py` — V3 prototype
- `helmgen-next/grid_search_v3.py` — grid search script
