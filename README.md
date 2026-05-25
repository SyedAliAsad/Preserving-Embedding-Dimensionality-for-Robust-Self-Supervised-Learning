**Preserving Embedding Dimensionality for Robust Self-Supervised Learning
Master's Thesis — University of Eastern Finland
Author: Syed Ali Asad | Supervisor: Professor Xiao-Zhi Gao | Year: 2026**

**Overview**
Self-supervised learning (SSL) methods like SimCLR and VICReg learn visual representations without labeled data by enforcing invariance between augmented views of the same image. This works well on clean benchmarks, but these representations are often surprisingly fragile under distribution shift.
This thesis investigates why. The core finding is that SSL training quietly compresses representations into a small fraction of the available embedding space — a phenomenon called spectral collapse. Standard SimCLR on CIFAR-100 uses only 23.7% of 512 available dimensions at convergence. The other 76% carry negligible variance and are effectively wasted.
To fix this, we propose L_dim: a dimensionality-preserving regularizer based on the Von Neumann entropy of the embedding covariance matrix. It encourages the representation to spread its energy more evenly across all available dimensions, and we show that this measurably improves robustness under image corruptions.

**Key Results**
ConfigurationClean AccEffective RankRel. RobustnessSimCLR baseline48.49%121.6 / 51280.46%SimCLR + L_dim (λ=0.25)45.49%302.1 / 51281.83%SimCLR + L_dim (λ=0.5)42.05%363.5 / 51282.45% ↑VICReg baseline47.20%159.1 / 51281.98%Barlow Twins baseline47.60%143.7 / 512—Barlow Twins + L_dim45.60%236.1 / 512—
Cross-dataset validation on STL-10:
ConfigurationClean AccEffective RankRank UsageSimCLR baseline84.7%141.1 / 51227.6%SimCLR + L_dim (λ=0.5)80.8%373.8 / 51273.0%

**Method**
The total training objective is:
L_total = L_ssl + λ(t) · L_dim
Where:

L_ssl is any standard SSL loss (NT-Xent for SimCLR, variance-invariance-covariance for VICReg)
L_dim = −H_VN(Σ) / log(D) is the negative normalized Von Neumann entropy of the encoder covariance matrix
λ(t) follows a linear warmup schedule over the first 20 epochs

**Key Design Choices**

Applied at the encoder output, not the projector (projector absorbs the gradient signal)
EMA covariance estimator with momentum 0.99 for stable estimates
Blended covariance (50% batch + 50% EMA) to maintain gradient flow while keeping stability


**Repository Structure**
dim-ssl/
├── src/
│   ├── losses/
│   │   ├── simclr.py          # NT-Xent contrastive loss
│   │   ├── vicreg.py          # VICReg loss
│   │   ├── barlow_twins.py    # Barlow Twins loss
│   │   └── dim_reg.py         # Von Neumann entropy regularizer (core contribution)
│   ├── models/
│   │   └── ssl_model.py       # Encoder + projection head
│   ├── data/
│   │   └── augmentations.py   # CIFAR-100 and STL-10 augmentations
│   └── utils/
│       └── config.py          # Config loading and defaults
├── scripts/
│   ├── train.py               # Main training script
│   ├── eval_robustness.py     # CIFAR-100-C corruption robustness evaluation
│   ├── eval_robustness_stl10.py  # STL-10 corruption evaluation
│   └── make_plots.py          # Generate all thesis figures
├── configs/
│   ├── simclr_cifar100.yaml
│   ├── simclr_cifar100_dimreg.yaml
│   ├── vicreg_cifar100.yaml
│   ├── vicreg_cifar100_dimreg.yaml
│   ├── barlow_cifar100.yaml
│   ├── barlow_cifar100_dimreg.yaml
│   ├── simclr_stl10.yaml
│   └── simclr_stl10_dimreg.yaml
└── README.md

**Getting Started**
**Installation**
bashgit clone https://github.com/SyedAliAsad/Preserving-Embedding-Dimensionality-for-Robust-Self-Supervised-Learning.git
cd Preserving-Embedding-Dimensionality-for-Robust-Self-Supervised-Learning
pip install -e .
Training
bash# SimCLR baseline on CIFAR-100
python scripts/train.py --config configs/simclr_cifar100.yaml

# SimCLR + L_dim at optimal lambda
python scripts/train.py --config configs/simclr_cifar100_dimreg.yaml

# VICReg baseline
python scripts/train.py --config configs/vicreg_cifar100.yaml

# Barlow Twins + L_dim
python scripts/train.py --config configs/barlow_cifar100_dimreg.yaml

# STL-10 experiments
python scripts/train.py --config configs/simclr_stl10.yaml \
    --overrides num_workers=8 batch_size=512
Override any config value with --overrides key=value key2=value2.
Robustness Evaluation
bash# CIFAR-100-C evaluation
python scripts/eval_robustness.py \
    --checkpoint ./checkpoints/epoch_399.pth \
    --data_dir ./data \
    --output results.json

# STL-10 evaluation
python scripts/eval_robustness_stl10.py \
    --checkpoint ./checkpoints/epoch_399.pth \
    --data_dir ./data \
    --output results_stl10.json
Generate Figures
bashpython scripts/make_plots.py --output_dir ./figures

**Experimental Setup**
SettingValueEncoderResNet-18 (512-d output)DatasetCIFAR-100 / STL-10Epochs400OptimizerAdamW (lr=0.001, wd=1e-4)SimCLR batch size512VICReg batch size256EMA momentum0.99Warmup epochs20GPUNVIDIA H100 80GBTraining time~2.5h per run

How L_dim Differs from Existing Methods
MethodSpectral ControlScopeHard/SoftVICReg variance termPer-dimensionLocalSoftVICReg covariance termPer-pairLocalSoftBarlow TwinsPer-pair (normalized)LocalSoftWhiteningFull spectrumGlobalHardL_dim (ours)Full spectrumGlobalSoft
L_dim is the only method that provides global spectral awareness through a soft constraint applied directly at the encoder output.

**Citation**
If you use this code or build on this work, please cite:
@mastersthesis{asad2026preserving,
  title={Preserving Embedding Dimensionality for Robust Self-Supervised Learning},
  author={Asad, Syed Ali},
  school={University of Eastern Finland},
  year={2026}
}

**Acknowledgements**
This work was completed as part of a Master's thesis at the University of Eastern Finland under the supervision of Professor Xiao-Zhi Gao.
