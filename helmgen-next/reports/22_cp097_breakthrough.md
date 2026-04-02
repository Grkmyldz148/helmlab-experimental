# BREAKTHROUGH: cp=0.97 → 38-7-16 (from 36-6-19)

**Date**: 2026-04-01

## Result
Adding chroma_power=0.97 to production pipeline: **38 WIN, 7 LOSS, 16 TIE**

## What flipped
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Gradient CV mean | 38.06% TIE | **37.26% WIN** | TIE→WIN! |
| Multi-stop CV | 37.7 TIE | **37.0 WIN** | TIE→WIN! |
| CVD protan | 0.13 TIE | **0.13 WIN** | TIE→WIN (threshold?) |
| OOG max dist | 0.1101 TIE | **0.1014 WIN** | TIE→WIN! |
| CVD deutan | 0.15 TIE | 0.09 LOSS | TIE→LOSS |
| Chroma pres | 0.416 TIE | 0.409 LOSS | TIE→LOSS |
| Worst CV | 412.6 TIE | 412.6 LOSS | TIE→LOSS (OKLab improved?) |

## Net: +4 WIN flips, -3 new LOSSes = +1 net WIN, but +2 over production

## Parameters
- chroma_power: 0.97
- enrichment amp: 0.060 (up from 0.055)
- Everything else: production

## Next: Fine-tune cp between 0.97-1.00 to find Pareto optimal
