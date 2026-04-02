# Pipeline Architecture Search Report (GPU-Batched)

**Date:** 2026-03-18 12:23:38
**Device:** CUDA (NVIDIA H100 80GB HBM3)
**Config:** 3 seeds × 300 gen × 96 pop
**Total:** 41s (0.7 min)

## Baselines

| Space | CV% | Hue RMS | Cusp L@85 | Cliff% | Drift | Ach | Gamut |
|-------|-----|---------|-----------|--------|-------|-----|-------|
| v14 | 22.73 | 18.9 | 0.966 | 0 | 14.5/100.7 | 0.000000 | 0.153 |
| OKLab | 22.90 | 30.1 | 0.831 | 4 | 14.6/102.7 | 0.000000 | 0.144 |

## A. A_SharedGamma (14p)

| Seed | Loss | CV% | Cusp L | Cliff% | Drift | Ach | Gamut | Hue | Cond | Ckpt |
|------|------|-----|--------|--------|-------|-----|-------|-----|------|------|
| v14 | 6.9904 | 27.86 | 0.822 | 9 | 11.6/45.8 | 0.000000 | 0.117 | 12.9 | 3.0 | `pipeline_A_SharedGamma_v14_final_20260318_122341.json` |
| oklab | 8.6909 | 26.40 | 0.822 | 0 | 11.7/47.9 | 0.000000 | 0.252 | 10.3 | 1.5 | `pipeline_A_SharedGamma_oklab_final_20260318_122343.json` |
| random | 999.0000 | 31.48 | 0.485 | 100 | 18.8/136.4 | 0.000000 | 0.010 | 73.0 | 2.0 | `pipeline_A_SharedGamma_random_final_20260318_122343.json` |

## B. B_NakaRushton (16p)

| Seed | Loss | CV% | Cusp L | Cliff% | Drift | Ach | Gamut | Hue | Cond | Ckpt |
|------|------|-----|--------|--------|-------|-----|-------|-----|------|------|
| oklab | 13.4103 | 35.63 | 0.806 | 0 | 13.8/45.5 | 0.000000 | 0.108 | 26.2 | 3.2 | `pipeline_B_NakaRushton_oklab_final_20260318_122349.json` |
| v14 | 15.2449 | 28.91 | 0.831 | 0 | 12.1/51.2 | 0.000000 | 0.180 | 11.7 | 1.6 | `pipeline_B_NakaRushton_v14_final_20260318_122346.json` |
| random | 999.0000 | 25.65 | 0.654 | 0 | 15.9/110.6 | 0.000000 | 0.019 | 161.2 | 2.1 | `pipeline_B_NakaRushton_random_final_20260318_122349.json` |

## C. C_DivNorm (22p)

| Seed | Loss | CV% | Cusp L | Cliff% | Drift | Ach | Gamut | Hue | Cond | Ckpt |
|------|------|-----|--------|--------|-------|-----|-------|-----|------|------|
| oklab | 6.8699 | 32.44 | 0.831 | 0 | 12.8/44.1 | 0.009626 | 0.046 | 17.1 | 2.9 | `pipeline_C_DivNorm_oklab_final_20260318_122402.json` |
| v14 | 58.5871 | 34.97 | 0.300 | 0 | 12.5/52.0 | 0.002679 | 0.323 | 10.0 | 2.1 | `pipeline_C_DivNorm_v14_final_20260318_122355.json` |
| random | 999.0000 | 49.46 | 0.300 | 0 | 24.5/127.7 | 0.006122 | 0.010 | 132.5 | 5.2 | `pipeline_C_DivNorm_random_final_20260318_122402.json` |

## D. D_LogWeighted (16p)

| Seed | Loss | CV% | Cusp L | Cliff% | Drift | Ach | Gamut | Hue | Cond | Ckpt |
|------|------|-----|--------|--------|-------|-----|-------|-----|------|------|
| v14 | 768.3266 | 37.45 | 0.763 | 9 | 14.8/95.4 | 0.000000 | 0.287 | 15.7 | 6.9 | `pipeline_D_LogWeighted_v14_final_20260318_122404.json` |
| random | 772.5646 | 35.94 | 0.418 | 4 | 11.3/90.6 | 0.000000 | 0.046 | 144.6 | 2.2 | `pipeline_D_LogWeighted_random_final_20260318_122405.json` |
| oklab | 999.0000 | 35.28 | 0.704 | 0 | 18.1/137.7 | 0.000000 | 0.180 | 32.0 | 2.1 | `pipeline_D_LogWeighted_oklab_final_20260318_122404.json` |

## E. E_PowerEnriched (17p)

| Seed | Loss | CV% | Cusp L | Cliff% | Drift | Ach | Gamut | Hue | Cond | Ckpt |
|------|------|-----|--------|--------|-------|-----|-------|-----|------|------|
| v14 | 1.4792 | 20.37 | 0.797 | 0 | 9.8/39.9 | 0.000000 | 0.350 | 6.8 | 3.2 | `pipeline_E_PowerEnriched_v14_final_20260318_122410.json` |
| oklab | 1.5572 | 21.29 | 0.890 | 0 | 9.8/39.8 | 0.000000 | 0.350 | 7.0 | 2.7 | `pipeline_E_PowerEnriched_oklab_final_20260318_122415.json` |
| random | 59.2057 | 32.01 | 0.831 | 0 | 15.9/61.1 | 0.000000 | 0.064 | 35.1 | 3.1 | `pipeline_E_PowerEnriched_random_final_20260318_122419.json` |

## Best per Architecture

| Arch | Loss | CV% | Cusp L | Cliff | Drift | Ach | Gamut | Seed |
|------|------|-----|--------|-------|-------|-----|-------|------|
| A | 6.9904 | 27.86 | 0.822 | 9 | 11.6/45.8 | 0.000000 | 0.117 | v14 |
| B | 13.4103 | 35.63 | 0.806 | 0 | 13.8/45.5 | 0.000000 | 0.108 | oklab |
| C | 6.8699 | 32.44 | 0.831 | 0 | 12.8/44.1 | 0.009626 | 0.046 | oklab |
| D | 768.3266 | 37.45 | 0.763 | 9 | 14.8/95.4 | 0.000000 | 0.287 | v14 |
| E | 1.4792 | 20.37 | 0.797 | 0 | 9.8/39.9 | 0.000000 | 0.350 | v14 |
