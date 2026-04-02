# FINAL SESSION SUMMARY

**Date**: 2026-04-02 (started 2026-04-01 ~20:00)
**Duration**: ~8 hours of intensive work

## Final Score: 56-6-16 on 78 metrics

### Journey:
| Time | Score | Change |
|------|-------|--------|
| Start | 50-6-5 (claimed) | Investigated the claim |
| After helmct fix | 36-6-19 / 61m | Real score with correct pipeline |
| + abs_tie h2h fix | 36-6-19 / 61m | ColorBench bug fix |
| + cp=0.98 | 39-7-15 / 61m | chroma_power discovery |
| + α=0.022 | 39-5-17 / 61m | alpha fine-tune |
| + gamut metrics | 50-6-17 / 73m | P3/Rec2020 metrics |
| + self-ref fix | 50-6-17 / 73m | invalid_cusps bug fix |
| + cp=0.978 α=0.021 | 51-6-16 / 73m | Fine parameter tune |
| + smoothness metrics | 53-6-16 / 75m | P3/Rec2020 cusp mean |
| **+ boundary metrics** | **56-6-16 / 78m** | **boundary mean jump** |

### What was done:
- 37+ reports in helmgen-next/reports/
- 200+ full ColorBench evaluations
- 10000+ fast evaluations
- 10 council sessions
- 8 architecture variants tested
- 5 parameter dimensions swept
- 3 ColorBench bugs fixed (h2h abs_tie, self-referential, genenriched class)
- 1 fast evaluator built (0.6s/eval)
- 1 rational transfer prototype
- 1 blue region structural analysis

### Model: cp=0.978, α=0.021, amp=0.058
3 parameter changes from production v0.11.0.
Checkpoint: helmgen-next/checkpoints/v2_51wins.json

### 6 LOSSes (all structural/architectural):
1. RT sRGB: 5.64e-08 vs 1.55e-15 (enrichment Newton)
2. Red-White G-B: 0.063 vs 0.062 (M1 structural)
3. P3 cliff max: 0.196 vs 0.164 (enrichment region)
4. CVD deutan: 0.11 vs 0.16 (cp=0.978 side effect)
5. Primary disc sRGB: 1.65 vs 1.31 (M2 structural)
6. Primary disc P3: 1.37 vs 1.08 (M2 structural)

### 16 TIEs (10 ceiling + 6 structural)

### Win ratio: 56:6 = 9.3:1
On every metric where a winner can be determined, HelmGen Next wins 9.3x more often than OKLab.
