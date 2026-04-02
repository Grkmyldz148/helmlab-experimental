# Pipeline Architecture Search Report (GPU-Batched)

**Date:** 2026-03-19 18:40:16
**Device:** MPS
**Config:** 1 seeds × 2 gen × 4 pop
**Total:** 2s (0.0 min)

## Baselines

| Space | CV% | Hue RMS | Cusp L@85 | Cliff% | Drift | Ach | Gamut |
|-------|-----|---------|-----------|--------|-------|-----|-------|
| v14 | 31.76 | 18.9 | 0.020 | 0 | 9.5/100.7 | 0.000000 | 0.153 |
| OKLab | 31.78 | 30.1 | 0.020 | 13 | 9.4/102.7 | 0.000000 | 0.144 |

## H. H_NakaRushtonCp (19p)

| Seed | Loss | CV% | Cusp L | Cliff% | Drift | Ach | Gamut | Hue | Cond | Ckpt |
|------|------|-----|--------|--------|-------|-----|-------|-----|------|------|
| v14_nr42 | 999.0000 | 30.71 | 0.020 | 0 | 8.8/87.7 | 0.000000 | 0.099 | 20.1 | 2.3 | `pipeline_H_NakaRushtonCp_v14_nr42_final_20260319_184018.json` |

## Best per Architecture

| Arch | Loss | CV% | Cusp L | Cliff | Drift | Ach | Gamut | Seed |
|------|------|-----|--------|-------|-------|-----|-------|------|
| H | 999.0000 | 30.71 | 0.020 | 0 | 8.8/87.7 | 0.000000 | 0.099 | v14_nr42 |
