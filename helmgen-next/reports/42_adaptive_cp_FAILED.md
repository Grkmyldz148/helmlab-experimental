# Adaptive Chroma Power — Did Not Help

**Date**: 2026-04-02

## Results
| cp_delta | WIN | LOSS | TIE |
|----------|-----|------|-----|
| 0 (current) | 59 | 8 | 16 |
| 0.01 | 59 | 8 | 16 |
| 0.02 | 58 | 8 | 17 |
| 0.03+ | 55-56 | 9 | 18-19 |

Adaptive cp does NOT fix the 3 weak categories:
- Near-achromatic CV: 106.81% unchanged (OKLab: 85.79%)
- Bright CV: 32.84% unchanged (OKLab: 32.16%)
- CVD deutan: 0.11 unchanged (OKLab: 0.16)

## Root Cause
Near-achromatic CV is NOT caused by cp. It's structural:
- Even cp=1.0 gives near-ach CV=103.5% (still LOSS)
- The depcubic + M2 geometry is inherently less uniform near achromatic axis
- OKLab's cbrt + M2 is better calibrated for near-achromatic gradients

## These 3 LOSSes are ARCHITECTURAL LIMITS:
1. Near-ach CV: depcubic geometry near achromatic axis
2. Bright CV: M2 ab-row alignment for high-L region
3. CVD deutan: cone overlap structure in M1/M2

## 59-8-16 IS the final score for this architecture.
