# Helmlab Experimental

Research experiments, optimization checkpoints, and analysis scripts for [Helmlab](https://github.com/Grkmyldz148/helmlab) GenSpace development.

This repository documents the journey from the initial OKLab-based GenSpace (23/43 benchmark wins) to the production v0.10.0 softened cube root pipeline (27/48 wins, 360/360 cusps, no lavender in Blue→White gradients).

## What's in this repo

### `checkpoints/` — 330+ model checkpoints

Every intermediate model saved during optimization. Key checkpoints:

| Checkpoint | Score vs OKLab | Pipeline | Notes |
|-----------|---------------|----------|-------|
| `helmlab_fine_best.json` | **27-7** | softcbrt + new M1 | **Production v0.10.0** |
| `helmlab_softcbrt_v7.json` | 31-9 | softcbrt + OKLab M1 | Most wins (cp=0.87) |
| `helmlab_softcbrt_perfect.json` | 23-6 | softcbrt + OKLab M1 | Fewest losses (cp=1.0) |
| `helmlab_okprime_v8.json` | 30-12 | Naka-Rushton | Best NR pipeline |
| `helmlab_v14.json` | 30-12 | cbrt + dual cross-term | Best cbrt pipeline |
| `v7b_nodelta.json` | 23-14 | cbrt (original) | v0.4.0 production |

### `checkpoints/experiment_report.md` — Full experiment log

Detailed report of all 480+ experiments including:
- Transfer function comparison (cbrt, softcbrt, Naka-Rushton, CIE Lab delta, power gamma)
- M1 optimization (OKLab perturbation, v7b blend, random search)
- M2 rotation sweep (15°-35°)
- Enrichment parameter grid search (cp, ck, PW scale, hue rotation)
- Fundamental trade-off analysis (gradient CV vs achromatic, Blue G/R vs cusps)
- Deep analysis of OKLab, CIE Lab, and v14 pipelines

### `scripts/` — 53 optimization scripts

CMA-ES and grid search scripts used throughout development:
- `optimize_gen_space.py` — Core GenSpace optimizer
- `optimize_v5_*.py` through `optimize_v31d_*.py` — Versioned experiments
- `optimize_hue_correction.py`, `optimize_transfer.py` — Component-level optimization
- `optimize_neural_space.py` — Neural architecture search attempt
- `optimize_pipeline_search.py` — Systematic pipeline architecture comparison

### `visualizations/` — Interactive comparison demos

HTML files for visual gradient comparison between models.

### `results/` — ColorBench evaluation outputs

JSON results from ColorBench evaluations of various models.

## Key Discoveries

### 1. Softened Cube Root Transfer Function

The breakthrough that enabled 360/360 cusps while preserving gradient quality:

```
f(x) = sign(x) * ((|x| + epsilon)^(1/3) - epsilon^(1/3))
```

Standard cube root `x^(1/3)` has infinite derivative at zero, causing gamut boundary singularities (OKLab: only 294/360 valid cusps in sRGB). Adding a small epsilon smooths the singularity while keeping the function nearly identical to cube root for typical color values.

**Exact analytical inverse:** `f_inv(y) = sign(y) * ((|y| + epsilon^(1/3))^3 - epsilon)`

### 2. M1 Perturbation

A tiny perturbation to OKLab's M1 matrix (M1[0,2] -= 0.006, then re-normalized for D65) converts 4 TIE metrics to WIN without creating new losses. This adjusts the L-cone response to blue light, improving Blue→White gradient quality.

### 3. Piecewise-Linear L Correction

Replaces polynomial L correction with a 19-breakpoint piecewise-linear function. Advantages:
- Exact analytical inverse (no Newton iteration)
- Monotonicity guaranteed by construction
- Better Munsell Value uniformity than OKLab

### 4. Trade-off Map

Every color space makes trade-offs. The key ones discovered:

| Trade-off | Left side | Right side |
|-----------|-----------|------------|
| Gradient CV vs Achromatic | cp < 1 improves gradient | cp = 1 preserves gray |
| Blue G/R vs Cusps | Larger epsilon fixes cusps | Smaller epsilon preserves Blue G/R |
| Hue agreement vs Munsell Hue | CIE Lab hue alignment | Munsell hue alignment |
| CVD vs Blue G/R | Cone-response M1 | Perceptual M1 |
| RT precision vs Enrichment | Fewer stages = better RT | More stages = more perceptual wins |

## Evaluation

All models were evaluated using [ColorBench](https://github.com/Grkmyldz148/colorbench) — 48 metrics, 3,038 gradient pairs, 3 gamuts (sRGB, Display P3, Rec.2020).

```bash
# Compare a checkpoint against OKLab
cd colorbench
python run.py oklab helmct --json ../checkpoints/helmlab_fine_best.json
```

## Timeline

- **v0.4.0** — Original cbrt GenSpace (23/43 vs OKLab)
- **v14** — Dual cross-term + enrichment (30-12 vs OKLab, but lavender Blue→White)
- **OKLab-Prime v8** — Naka-Rushton transfer (30-12, blue fixed but gradient CV poor)
- **Softcbrt v1-v7** — Softened cube root discovery (28-10 to 31-9)
- **PERFECT** — cp=1.0 variant (23-6, no visual metric worse than OKLab)
- **Grid best** — M1 perturbation + softcbrt (27-7, production v0.10.0)

## Repository Structure

```
helmlab-experimental/
├── README.md                           # This file
├── checkpoints/                        # 330+ model checkpoints (.json)
│   ├── experiment_report.md            # Full experiment log (480+ experiments)
│   ├── helmlab_fine_best.json          # Production v0.10.0
│   ├── helmlab_softcbrt_*.json         # Softcbrt variants
│   ├── helmlab_okprime_*.json          # Naka-Rushton variants
│   ├── helmlab_v*.json                 # cbrt pipeline variants
│   ├── pipeline_*.json                 # Architecture search results
│   └── ...                             # All other checkpoints
├── scripts/                            # 53 optimization scripts
│   ├── optimize_gen_space.py           # Core GenSpace optimizer
│   ├── optimize_v*.py                  # Versioned experiments
│   └── ...
├── visualizations/                     # Interactive HTML demos
└── results/                            # ColorBench evaluation outputs
```

## Related

- **[Helmlab](https://github.com/Grkmyldz148/helmlab)** — Production color space library
- **[ColorBench](https://github.com/Grkmyldz148/colorbench)** — Color space evaluation benchmark
- **[Paper](https://arxiv.org/abs/2602.23010)** — arXiv:2602.23010

## Author

**[Gorkem Yildiz](https://gorkemyildiz.com)**

## License

MIT
