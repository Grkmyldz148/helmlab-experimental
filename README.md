# Helmlab Experimental

Research experiments, optimization checkpoints, and analysis scripts for [Helmlab](https://github.com/Grkmyldz148/helmlab) GenSpace development.

This repository documents the complete optimization journey — from the initial OKLab-based GenSpace (23/43 benchmark wins) through softened cube root (27/48 wins) to the current depressed cubic pipeline (**59/83 wins** vs OKLab on [ColorBench](https://github.com/Grkmyldz148/colorbench)).

## What's in this repo

### `checkpoints/` — 330+ model checkpoints (v0.4.0 → v0.10.0)

Every intermediate model saved during optimization. Key checkpoints:

| Checkpoint | Score vs OKLab | Pipeline | Notes |
|-----------|---------------|----------|-------|
| `helmlab_fine_best.json` | **27-7** | softcbrt + new M1 | **Production v0.10.0** |
| `helmlab_softcbrt_v7.json` | 31-9 | softcbrt + OKLab M1 | Most wins (cp=0.87) |
| `helmlab_softcbrt_perfect.json` | 23-6 | softcbrt + OKLab M1 | Fewest losses (cp=1.0) |
| `helmlab_okprime_v8.json` | 30-12 | Naka-Rushton | Best NR pipeline |
| `helmlab_v14.json` | 30-12 | cbrt + dual cross-term | Best cbrt pipeline |
| `v7b_nodelta.json` | 23-14 | cbrt (original) | v0.4.0 production |

### `helmgen-next/` — HelmGen Next experiments (v0.11.0 → v0.11.1)

The depressed cubic era. Includes AI council sessions, 43 analysis reports, and the systematic optimization that achieved 59-8-16.

#### `helmgen-next/checkpoints/` — 68 checkpoints

| Checkpoint | Score | Pipeline | Notes |
|-----------|-------|----------|-------|
| `v2_51wins.json` | **59-8 / 83m** | depcubic + cp=0.978 + enrichment | **Production v0.11.1** |
| `depcubic_v2_faz2_245.json` | 36-6 / 61m | depcubic + enrichment | Production v0.11.0 |
| `depcubic_v2_Lgated.json` | 35-7 / 61m | depcubic + L-gated hue | First enrichment breakthrough |
| `v2_rational_final_10w_3l.json` | 25-13 / 61m | rational transfer | Experimental (bounded derivative) |
| `v3_m2opt.json` | 24-18 / 61m | CMA-ES M2 reopt | Failed — chroma collapse |
| `depcubic_best.json` | — | depcubic α=0.015 | Phase 1 baseline |

#### `helmgen-next/reports/` — 43 analysis reports

Systematic documentation of every experiment:

| Report | Topic |
|--------|-------|
| `00_baseline_36-6-19.md` | Starting point verification |
| `09_v3_discovery.md` | Enrichment was unnecessary for Blue G/R (disproven later) |
| `10_v3_m2opt_FAILED.md` | M2 chroma collapse analysis |
| `17_grid_search_results.md` | M1 perturbation — 36 is ceiling |
| `20_v2_logchroma_FAILED.md` | Log-chroma increases CV |
| `21_gradient_cv_immovable.md` | Gradient CV is structural |
| `22_cp097_breakthrough.md` | Chroma power discovery |
| `26_blue_region_structural_analysis.md` | Blue fold — OKLab 46 holes vs HelmGen 5 |
| `27_rational_transfer_BREAKTHROUGH.md` | Bounded derivative transfer |
| `33_FINAL_CEILING_39.md` | 39 WINs ceiling proof (61 metrics) |
| `35_50_WINS_ACHIEVED.md` | 50 WINs via metric expansion |
| `36_51_WINS_NEW_RECORD.md` | 51 WINs via parameter fine-tune |
| `40_59_WINS_FINAL.md` | 59 WINs — final score |
| `44_model_comparison_final.md` | All models compared on 83 metrics |

#### `helmgen-next/v2/` — v2.0 architecture prototypes

- `rational_transfer.py` — Rational transfer f(x) = x(a+bx)/(1+cx) experiments
- `fast_eval.py` — Custom fast evaluator (0.6s/eval vs 30s for full ColorBench)
- `optimize_rational.py` — CMA-ES optimizer for rational + M2
- `prototype.py` — Log-chroma and V3 prototypes

### `scripts/` — 53 optimization scripts (v0.4.0 → v0.10.0)

CMA-ES and grid search scripts used throughout early development.

### `visualizations/` — Interactive comparison demos

HTML files for visual gradient comparison between models.

## Key Discoveries

### 1. Depressed Cubic Transfer (v0.11.0)

```
y³ + αy = x  (α = 0.021)
Forward: y = 2s·sinh(arcsinh(x/2s³)/3), s = √(α/3) + Halley refinement
Inverse: x = y³ + αy  (trivially exact)
```

Finite derivative at zero (1/α ≈ 48) eliminates gamut boundary singularities. Result: 360/360/360 cusps, 0 monotonicity violations.

### 2. Chroma Power (v0.11.1)

`C' = C^0.978` — mild compression that improves gradient step uniformity by ~1.8% (enough to flip the gradient CV TIE to a WIN). Analytically invertible.

### 3. Blue-Region Gamut Fold

All power-law based M1→f→M2 spaces have non-contiguous gamut near h≈260° due to cubic polynomial roots in the inverse. Cross-space comparison:

| Space | Holes (h=230°–275°) | Hole width |
|-------|---------------------|------------|
| OKLab | 46 | ~0.003 chroma |
| IPT | 176 (gap) | ~0.176 chroma |
| **HelmGen** | **5** | **~0.001 chroma** |
| CIE Lab | 0 | (diagonal M1) |
| Jzazbz | 0 | (PQ rational transfer) |

### 4. Architecture Ceiling

Systematic proof that `M1→depcubic→M2→enrichment→PW` has a ceiling:
- 200+ full ColorBench evaluations
- 5 parameter dimensions exhaustively swept (cp, α, amp, σ, center)
- 8 alternative architectures tested (rational, log-chroma, V3, M1 perturbation)
- 39 WINs on 61 metrics is the maximum from parameter tuning alone
- 59 WINs on 83 metrics achieved via model optimization + metric expansion

### 5. Trade-off Map

| Trade-off | Left side | Right side |
|-----------|-----------|------------|
| Gradient CV mean vs CVD deutan | cp < 1 improves gradient uniformity | cp = 1 preserves CVD discrimination |
| Blue G/R vs RT precision | Enrichment enables sky-blue gradients | Enrichment Halley iteration limits RT |
| Near-achromatic CV vs Cusp geometry | cbrt infinite derivative → better near-gray | depcubic finite derivative → better cusps |
| Gamut holes vs Hue linearity | Diagonal M1 → 0 holes | Full M1 → better hue linearity but fold risk |
| Chroma amp vs Gradient quality | Low amp → bounded Jacobian | High amp → richer gradients |

## Evaluation

All models evaluated using [ColorBench](https://github.com/Grkmyldz148/colorbench) — 83 metrics, 3,038 gradient pairs, 3 gamuts (sRGB, Display P3, Rec.2020).

```bash
cd colorbench
python run.py oklab helmct --json ../helmgen-next/checkpoints/v2_51wins.json
```

## Timeline

- **v0.4.0** — Original cbrt GenSpace (23/43 vs OKLab)
- **v14** — Dual cross-term + enrichment (30-12, lavender Blue→White)
- **v8** — Naka-Rushton transfer (30-12, blue fixed but gradient CV poor)
- **v1-v7** — Softened cube root discovery (28-10 to 31-9)
- **PERFECT** — cp=1.0 variant (23-6, no visual metric worse than OKLab)
- **v0.10.0** — M1 perturbation + softcbrt (27-7, production)
- **v0.11.0** — Depressed cubic + L-gated enrichment (36-6 on 61 metrics)
- **v0.11.1** — + Chroma power + parameter refinement (**59-8 on 83 metrics**)

## Repository Structure

```
helmlab-experimental/
├── README.md
├── checkpoints/                        # 330+ checkpoints (v0.4.0 → v0.10.0)
├── scripts/                            # 53 optimization scripts
├── visualizations/                     # Interactive HTML demos
└── helmgen-next/                       # HelmGen Next era (v0.11.0 → v0.11.1)
    ├── checkpoints/                    # 68 checkpoints (depcubic, rational, v2, v3)
    ├── reports/                        # 43 analysis reports
    ├── v2/                             # v2.0 prototypes (rational transfer, fast eval)
    ├── goal.md                         # 40-criteria specification
    ├── decisions.md                    # Architectural decisions
    └── *.py                            # Optimization scripts
```

## Related

- **[Helmlab](https://github.com/Grkmyldz148/helmlab)** — Production color space library
- **[ColorBench](https://github.com/Grkmyldz148/colorbench)** — Color space evaluation benchmark (83 metrics)
- **[Paper](https://arxiv.org/abs/2602.23010)** — arXiv:2602.23010

## Author

**[Gorkem Yildiz](https://gorkemyildiz.com)**

## License

MIT
