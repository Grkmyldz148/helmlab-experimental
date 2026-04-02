# v2.0 Log-Chroma — FAILED

**Date**: 2026-04-01

## Result: Log-chroma INCREASES gradient CV, doesn't decrease it.

| k | CV proxy | Blue G/R | Amp | Notes |
|---|---------|----------|-----|-------|
| 0 (baseline) | **17.9%** | 2.014 | 3.4x | Best CV |
| 0.5 | 20.6% | 1.931 | 3.1x | CV worse |
| 1.0 | 23.1% | 1.867 | 2.8x | CV worse |
| 5.0 | 37.2% | 1.611 | 1.9x | CV much worse |
| 10.0 | 47.4% | 1.483 | 1.5x | CV terrible, G/R FAIL |

## Why it fails

Log-chroma COMPRESSES high-chroma steps → makes them SMALLER relative to low-chroma steps.
This INCREASES non-uniformity (high-CV pairs get worse).

The gradient CV metric measures step uniformity in CIEDE2000. CIEDE2000 already has chroma
weighting (SC term). Adding log-chroma on top creates DOUBLE compression.

## Lesson
- Log-chroma helps amplification (3.4→1.5x at k=10) but HURTS gradient uniformity
- This is a fundamental trade-off: uniform gradients need LINEAR chroma, capped amp needs LOG chroma
- Can't have both in same coordinates

## Codex was wrong about C — log-chroma doesn't improve gradients.
## Gemini was partially right about B — hue-varying M2 might be the only path.
