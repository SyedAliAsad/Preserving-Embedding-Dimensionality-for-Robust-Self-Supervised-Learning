# Dimensionality-Preserving Self-Supervised Learning

**Thesis project**: Investigating whether explicitly preserving embedding dimensionality via Von Neumann entropy regularization improves robustness of self-supervised representations under distribution shift.

## Project Structure

```
dim-ssl/
├── src/                        # Main Python package
│   ├── models/
│   │   ├── encoder.py          # ResNet backbone (strips FC layer)
│   │   ├── projector.py        # MLP projection head
│   │   └── ssl_model.py        # Encoder + Projector wrapper
│   ├── losses/
│   │   ├── simclr.py           # NT-Xent contrastive loss
│   │   ├── vicreg.py           # Variance-Invariance-Covariance loss
│   │   └── dim_reg.py          # Von Neumann entropy regularizer (core contribution)
│   ├── data/
│   │   ├── augmentations.py    # SimCLR-style augmentations
│   │   └── datasets.py         # CIFAR-100, STL-10, ImageNet-100 loaders
│   ├── evaluation/
│   │   ├── linear_probe.py     # Linear probing evaluation
│   │   └── spectral.py         # Eigenvalue/effective rank diagnostics
│   └── utils/
│       ├── config.py           # YAML config loading
│       ├── schedule.py         # λ(t) warmup schedules
│       └── logging.py          # W&B logging + spectral plots
├── configs/                    # Experiment configs (YAML)
│   ├── simclr_cifar100.yaml           # Baseline
│   ├── vicreg_cifar100.yaml           # Baseline
│   ├── simclr_cifar100_dimreg.yaml    # Proposed method
│   └── vicreg_cifar100_dimreg.yaml    # Proposed method
├── scripts/
│   └── train.py                # Main training entry point
├── notebooks/
│   ├── train_colab.ipynb       # Colab launcher for training
│   └── analysis_colab.ipynb    # Post-training analysis & plots
├── setup.py
├── requirements.txt
└── README.md
```

## Quick Start (Google Colab)

1. Push this repo to your GitHub
2. Open `notebooks/train_colab.ipynb` in Colab
3. Enable GPU runtime
4. Run all cells

## Quick Start (Local / Server)

```bash
pip install -e .

# SimCLR baseline
python scripts/train.py --config configs/simclr_cifar100.yaml

# SimCLR + dimensionality regularizer
python scripts/train.py --config configs/simclr_cifar100_dimreg.yaml

# Override any config value
python scripts/train.py --config configs/simclr_cifar100_dimreg.yaml \
    --overrides dim_reg_weight=0.5 epochs=100 seed=123
```

## Key Idea

Standard SSL methods (SimCLR, VICReg) optimize for invariance between augmented views, which can suppress features that are weakly correlated with augmentations but important for robustness. We add a regularizer that maximizes the Von Neumann entropy of the embedding covariance matrix:

```
L_total = L_ssl + λ(t) · L_dim

L_dim = -H_VN(Σ) / log(D)
```

where H_VN is the Von Neumann entropy (entropy of the normalized eigenvalue spectrum). This encourages the representation to maintain high effective dimensionality, preserving weak but informative features.

## Experiment Tracking

Uses [Weights & Biases](https://wandb.ai). Set up with:
```bash
wandb login
```

## TODO

- [ ] Reproduce SimCLR/VICReg baselines on CIFAR-100
- [ ] Verify spectral diagnostics on baselines (show feature suppression)
- [ ] Run dim-reg experiments and compare
- [ ] CIFAR-100-C corruption robustness evaluation
- [ ] STL-10 experiments
- [ ] ImageNet-100 experiments
- [ ] Cross-dataset OOD evaluation
- [ ] Ablation: λ sweep, schedule comparison, projector vs encoder regularization
- [ ] Thesis figures and writeup
