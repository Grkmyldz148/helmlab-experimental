# Model Comparison — Final

**Date**: 2026-04-02

## All tested models on 83 metrics:

| Model | WIN | LOSS | TIE | Ratio | Best? |
|-------|-----|------|-----|-------|-------|
| **cp=0.978 α=0.021 amp=0.058** | **59** | **8** | **16** | **7.4:1** | **YES** |
| cp=1.0 α=0.021 amp=0.058 | 55 | 10 | 18 | 5.5:1 | |
| Production v0.11.0 (cp=1, α=0.02, amp=0.055) | 55 | 11 | 17 | 5.0:1 | |

## cp=0.978 is the BEST on ALL dimensions:
- Most WINs (59)
- Fewest LOSSes (8)
- Best ratio (7.4:1)

Removing cp does NOT reduce LOSSes — it increases them from 8 to 10-11.
The 3 "cp-caused LOSSes" (p95, bright, CVD deutan) are smaller than
the 4 WINs cp creates (gradient CV mean, dark CV, high-chroma CV, cross-L CV).

## Enrichment is REQUIRED:
Without enrichment, Blue G/R drops to 1.37-1.43 (FAIL, target ≥1.50).
No M1/M2 combination can reach 1.50 without enrichment.

## DEFINITIVE BEST MODEL: cp=0.978, α=0.021, amp=0.058
Checkpoint: helmgen-next/checkpoints/v2_51wins.json
