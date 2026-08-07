# A Comparison of Fusion Techniques for Multi-Modal Human Activity Recognition on the HARMES Dataset

Code and precomputed results to reproduce the experiments and figures from the
paper. We compare seven multimodal fusion strategies for human activity
recognition on the [HARMES dataset](https://zenodo.org/records/19425719)
(dual-wrist IMU + ambient audio + humidity, 20 participants, 16 activity classes
including a Null class), under a leakage-free, participant-grouped evaluation
protocol.

## Key results

3-fold group cross-validation, macro F1:

| Model | Modality | Macro F1 |
|---|---|---|
| TinyHAR | IMU | 0.696 |
| AST | Audio | 0.734 |
| TSMixer | Humidity | 0.088 |
| **GMF** | **fusion** | **0.827** |
| Late Fusion | fusion | 0.817 |
| CMA | fusion | 0.795 |
| CLS Transformer | fusion | 0.793 |
| MBT | fusion | 0.787 |
| LMF | fusion | 0.747 |
| Decision Fusion | fusion | 0.747 |

GMF under leave-one-participant-out (LOPO): **0.819** macro F1.

### Statistical separability

Because each participant is held out in exactly one fold, every model yields one
macro F1 per participant on an identical window set, giving 20 paired
observations per comparison. Testing those pairs
(`notebooks/20_significance_tests.py`) shows the methods fall into three tiers
rather than a strict order. All 16 cross-tier comparisons are significant after
Holm correction (largest adjusted *p* = 0.0013) and none of the 5 within-tier
comparisons are (smallest adjusted *p* = 0.413).

| Tier | Methods |
|---|---|
| 1 | GMF, Late Fusion |
| 2 | CMA, CLS Transformer, MBT |
| 3 | LMF, Decision Fusion |

GMF's 0.011 lead over Late Fusion is **not** statistically significant (paired
*t*(19) = 2.074, *p* = 0.052; Wilcoxon *W* = 58, *p* = 0.083; 95% CI
[−0.000, +0.022]).

That should not be read as evidence the two methods are equal. A sensitivity
analysis shows the smallest difference this design can detect at 80% power is
0.016 macro F1, which is larger than the 0.011 gap actually observed, and that
conclusion holds across the 95% confidence interval of the difference standard
deviation. Equivalence testing (TOST) puts the same point the other way: the two
are equivalent within ±0.03 macro F1 (*p* = 0.001) but equivalence cannot be
claimed within ±0.01 (*p* = 0.586). The difference between GMF and late
concatenation therefore falls **below the resolution of a 20-participant
design**; separating them would need roughly 39 participants if the true
difference is the size measured here.

The tier separation is unaffected, since those gaps are three to eight times
larger and sit well above the detection threshold. The paper's conclusion that
the simplest fusion mechanisms outperform the attention- and transformer-based
ones holds. LMF and Decision Fusion do not significantly outperform the AST
unimodal baseline (adjusted *p* = 0.127).

## Repository layout

```
.
├── src/                 # model + data pipeline
│   ├── data/            # HARMES dataset loader, windowing, participant splits
│   ├── encoders/        # TinyHAR (IMU), AST (audio), TSMixer (humidity)
│   ├── fusion/          # 7 fusion methods (late, gmf, mbt, cma, lmf, decision, cls)
│   ├── training/        # training loop
│   └── viz/             # figure style
├── jobs/                # training + evaluation scripts (SLURM .sh + .py)
├── notebooks/           # figure-reproduction scripts (run from repo root)
└── results/             # precomputed metrics + raw predictions (ship with repo)
    ├── *_results.json   # per-model 3-fold metrics + confusion matrices
    ├── predictions/     # raw per-window predictions (gzipped CSV)
    └── lopo/            # LOPO metrics + predictions
```

Running the notebooks creates a `figures/<topic>/` folder with the generated
figures (PDF + SVG); it is not committed to the repository.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the figures (no dataset download needed)

Every figure is generated from the precomputed predictions and metrics in
`results/`, so you do **not** need the raw dataset to reproduce them. Run any
notebook from the repository root:

```bash
python notebooks/01_unimodal_overview.py      # unimodal baselines + confusion matrices
python notebooks/02_fusion_overview.py        # fusion leaderboard + GMF-vs-AST/TinyHAR diffs
python notebooks/10_lefthander_analysis.py    # handedness robustness
python notebooks/11_lopo_analysis.py          # leave-one-participant-out
python notebooks/20_significance_tests.py     # paired significance tests across participants
```

Figures are written to a `figures/<topic>/` folder (created on first run) as
PDF + SVG. The other notebooks (`03`–`21`) reproduce the per-method deep-dives
and additional analyses.

Notebooks are stored as [jupytext](https://jupytext.readthedocs.io/) percent
scripts; open them as notebooks in Jupyter or run them directly with `python`.

## Reproducing the models from scratch (requires the dataset)

1. Download the HARMES dataset from Zenodo:
   <https://zenodo.org/records/19425719> (DOI: 10.5281/zenodo.19425719).
2. Place the preprocessed participant files under `data/harmes/` (the default),
   or point the scripts elsewhere via the `HARMES_DATA` environment variable:

   ```bash
   export HARMES_DATA=/path/to/harmes
   ```

   Run training and evaluation from the repository root.
3. Train and evaluate. Each method has a per-fold script, e.g.:

   ```bash
   PYTHONPATH=. python jobs/train_gmf_10s.py --fold 0
   PYTHONPATH=. python jobs/evaluate_fusion_all_10s.py
   ```

   The `.sh` files are SLURM submission wrappers used on the cluster.

### Configuration (matches the paper)

| Setting | Value |
|---|---|
| Window | 500 samples = 10 s @ 50 Hz |
| Step | 250 (3-fold, 50% overlap) / 500 (LOPO, no overlap) |
| Embedding dim | 128 |
| Optimizer | Adam, cosine annealing, 50 epochs, batch 32 |
| Learning rate | 1e-3 (LMF: 5e-3) |
| Early stopping | patience 10 on validation macro F1 |
| AST backbone | frozen (linear head trained) |

## Dataset

HARMES: Human Activity Recognition with Motion, Environmental Sensing and Sound.
Burchard, Brückner, Bock, Van Laerhoven (2026). Zenodo.
DOI: [10.5281/zenodo.19425719](https://zenodo.org/records/19425719). License: CC BY 4.0.

## Citation

The preprint is available on arXiv:
[arXiv:2606.27886](https://arxiv.org/abs/2606.27886) \[cs.LG\].

```bibtex
@misc{mohamady_fusion_harmes_2026,
  title         = {A Comparison of Fusion Techniques for Multi-Modal Human
                   Activity Recognition on the HARMES Dataset},
  author        = {Mohamady, Ahmed and Burchard, Robin and Van Laerhoven, Kristof},
  year          = {2026},
  eprint        = {2606.27886},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2606.27886},
  url           = {https://arxiv.org/abs/2606.27886}
}
```

## License

MIT (see [LICENSE](LICENSE)). The HARMES dataset is licensed separately under CC BY 4.0.
