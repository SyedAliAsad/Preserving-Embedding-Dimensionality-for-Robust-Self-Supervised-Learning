"""Logging utilities for W&B experiment tracking."""

import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def setup_wandb(cfg: dict) -> bool:
    """Initialize W&B run. Returns True if successful."""
    if wandb is None:
        print("wandb not installed, skipping logging.")
        return False

    wandb.init(
        project=cfg.get("wandb_project", "dim-ssl"),
        entity=cfg.get("wandb_entity"),
        config=cfg,
        name=_run_name(cfg),
    )
    return True


def _run_name(cfg: dict) -> str:
    """Generate descriptive run name."""
    parts = [
        cfg["method"],
        cfg["backbone"],
        cfg["dataset"],
    ]
    if cfg.get("use_dim_reg"):
        parts.append(f"dimreg{cfg['dim_reg_weight']}")
    return "_".join(parts)


def log_spectral_plot(eigenvalues: np.ndarray, epoch: int, prefix: str = "spectral"):
    """Log eigenvalue spectrum plot to W&B.

    Args:
        eigenvalues: Array of eigenvalues in descending order
        epoch: Current epoch
        prefix: Metric prefix for W&B
    """
    if wandb is None or plt is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Linear scale
    axes[0].bar(range(len(eigenvalues)), eigenvalues, color="#4A90D9", alpha=0.8)
    axes[0].set_xlabel("Component index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title(f"Eigenvalue spectrum (epoch {epoch})")

    # Log scale
    axes[1].bar(range(len(eigenvalues)), eigenvalues, color="#D94A4A", alpha=0.8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Component index")
    axes[1].set_ylabel("Eigenvalue (log)")
    axes[1].set_title(f"Eigenvalue spectrum - log scale (epoch {epoch})")

    plt.tight_layout()
    wandb.log({f"{prefix}/spectrum_plot": wandb.Image(fig)}, step=epoch)
    plt.close(fig)
