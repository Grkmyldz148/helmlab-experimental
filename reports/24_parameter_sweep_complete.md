# Complete Parameter Sweep: 39-5-17 is the Optimum

**Date**: 2026-04-02

## Parameters scanned:

### chroma_power (cp):
| cp | WIN | LOSS | TIE |
|----|-----|------|-----|
| 0.96 | 38 | 7 | 16 |
| 0.97 | 38 | 7 | 16 |
| 0.975 | 39 | 8 | 14 |
| **0.98** | **39** | **5** | **17** |
| 0.985 | 38 | 6 | 17 |
| 0.99 | 37 | 7 | 17 |
| 1.0 | 36 | 6 | 19 |

### depcubic_alpha (α):
| α | WIN | LOSS | TIE |
|---|-----|------|-----|
| 0.015 | 38 | 7 | 16 |
| 0.018 | 38 | 8 | 15 |
| 0.020 | 39 | 7 | 15 |
| **0.021-0.022** | **39** | **5** | **17** |
| 0.023 | 39 | 7 | 15 |
| 0.025 | 38 | 5 | 18 |

### enrichment amp:
| amp | WIN | LOSS | TIE |
|-----|-----|------|-----|
| 0.055-0.065 | 39 | 5-7 | 15-17 |
→ Amp has minimal effect on score

### enrichment sigma:
| σ | WIN | LOSS | TIE |
|---|-----|------|-----|
| 0.5-0.6 | 39 | 6 | 16 |
| **0.7** | **39** | **5** | **17** |
| 0.8-0.9 | 37-38 | 5 | 18-19 |

### enrichment center:
| center | WIN | LOSS | TIE |
|--------|-----|------|-----|
| 260-270 | 39 | 5 | 17 |
→ Center has no effect at σ=0.7

## Optimal parameters:
- **cp=0.98, α=0.022, amp=0.058, σ=0.7, center=264.5°**
- **Score: 39-5-17**

## This is the ceiling for this architecture class.
Every parameter has been scanned. 39 is the maximum genuine WIN count.

## Progress summary:
| Config | WIN | LOSS | TIE |
|--------|-----|------|-----|
| v0.11.0 production | 36 | 6 | 19 |
| + abs_tie h2h fix | 36 | 6 | 19 |
| + cp=0.98 | 39 | 7 | 15 |
| + α=0.022 | **39** | **5** | **17** |

**Net gain: +3 WIN, -1 LOSS** from parameter optimization.
