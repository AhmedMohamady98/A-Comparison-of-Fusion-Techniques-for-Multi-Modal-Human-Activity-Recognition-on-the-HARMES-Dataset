# A Comparison of Fusion Techniques for Multi-Modal Human Activity Recognition on the HARMES Dataset

Code and precomputed results to reproduce the experiments and figures from the
paper. We compare seven multimodal fusion strategies for human activity
recognition on the [HARMES dataset](https://zenodo.org/records/19425719)
(wrist IMU + ambient audio + humidity, 20 participants, 16 activity classes
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
| MBT | fusion | 0.788 |
| LMF | fusion | 0.747 |
| Decision Fusion | fusion | 0.747 |

GMF under leave-one-participant-out (LOPO): **0.819** macro F1.

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
python notebooks/01_unimodal_overview.py     # unimodal baselines + confusion matrices
python notebooks/02_fusion_overview.py        # fusion leaderboard + GMF-vs-AST/TinyHAR diffs
python notebooks/10_lefthander_analysis.py    # handedness robustness
python notebooks/11_lopo_analysis.py          # leave-one-participant-out
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

```bibtex
@inproceedings{mohamady_fusion_harmes,
  title     = {A Comparison of Fusion Techniques for Multi-Modal Human
               Activity Recognition on the HARMES Dataset},
  author    = {Mohamady, Ahmed and Burchard, Robin},
  booktitle = {iWOAR},
  year      = {2026}
}
```

## License

MIT (see [LICENSE](LICENSE)). The HARMES dataset is licensed separately under CC BY 4.0.
