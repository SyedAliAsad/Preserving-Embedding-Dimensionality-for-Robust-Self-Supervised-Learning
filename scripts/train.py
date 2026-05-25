"""Main SSL pretraining script.

Usage:
    python scripts/train.py --config configs/simclr_cifar100.yaml
    python scripts/train.py --config configs/vicreg_cifar100_dimreg.yaml
"""

import argparse
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from src.models import SSLModel
from src.losses import SimCLRLoss, VICRegLoss, DimensionalityRegularizer
from src.data import get_ssl_dataloader, get_eval_dataloaders
from src.evaluation import linear_probe, compute_spectral_diagnostics
from src.utils import load_config, lambda_schedule, setup_wandb, log_spectral_plot

try:
    import wandb
except ImportError:
    wandb = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, cfg):
    """Build optimizer with linear warmup + cosine decay."""
    base_lr = cfg["lr"]
    opt_name = cfg.get("optimizer", "adamw")

    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=0.9,
            weight_decay=cfg["weight_decay"],
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=cfg["weight_decay"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # Cosine annealing after warmup
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"] - cfg["warmup_epochs"],
    )
    # Linear warmup
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=cfg["warmup_epochs"],
    )
    # Combine: warmup then cosine
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg["warmup_epochs"]],
    )
    return optimizer, scheduler


def train_one_epoch(
    model, loader, ssl_criterion, dim_reg, optimizer, cfg, epoch, device,
):
    """Train for one epoch. Returns dict of averaged metrics."""
    model.train()
    total_loss = 0.0
    total_ssl = 0.0
    total_dim = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for view1, view2, _ in pbar:
        view1 = view1.to(device)
        view2 = view2.to(device)

        # Forward pass
        z1 = model(view1)
        z2 = model(view2)

        # SSL loss
        ssl_loss = ssl_criterion(z1, z2)

        # Dimensionality regularizer
        dim_loss = torch.tensor(0.0, device=device)
        if cfg["use_dim_reg"] and dim_reg is not None:
            lam = lambda_schedule(
                epoch=epoch,
                max_epochs=cfg["epochs"],
                base_weight=cfg["dim_reg_weight"],
                schedule=cfg["dim_reg_schedule"],
                warmup_epochs=cfg["dim_reg_warmup_epochs"],
            )
            # Apply to projector output or encoder output
            if cfg["dim_reg_on"] == "encoder":
                with torch.no_grad():
                    h1 = model.encode(view1)
                # Recompute with grad for dim_reg
                h1_grad = model.encode(view1)
                dim_loss = dim_reg(h1_grad)
            else:
                # Concatenate both views for better covariance estimate
                z_both = torch.cat([z1, z2], dim=0)
                dim_loss = dim_reg(z_both)
            dim_loss = lam * dim_loss

        loss = ssl_loss + dim_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_ssl += ssl_loss.item()
        total_dim += dim_loss.item()
        n_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", ssl=f"{ssl_loss.item():.4f}")

    return {
        "loss": total_loss / n_batches,
        "ssl_loss": total_ssl / n_batches,
        "dim_loss": total_dim / n_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    # Allow any config override via command line
    parser.add_argument("--overrides", nargs="*", default=[], help="key=value overrides")
    args = parser.parse_args()

    # Parse overrides
    overrides = {}
    for ov in args.overrides:
        k, v = ov.split("=", 1)
        # Try to parse as int/float/bool
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                continue
        if v == "true":
            v = True
        elif v == "false":
            v = False
        overrides[k] = v

    cfg = load_config(args.config, overrides)
    print(f"Config: {cfg}")

    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- Data ----
    train_loader = get_ssl_dataloader(
        dataset=cfg["dataset"],
        data_dir=cfg["data_dir"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    # ---- Model ----
    model = SSLModel(
        backbone=cfg["backbone"],
        proj_hidden_dim=cfg["proj_hidden_dim"],
        proj_out_dim=cfg["proj_out_dim"],
        proj_layers=cfg["proj_layers"],
    ).to(device)

    # ---- Losses ----
    if cfg["method"] == "simclr":
        ssl_criterion = SimCLRLoss(temperature=cfg["temperature"])
    elif cfg["method"] == "vicreg":
        ssl_criterion = VICRegLoss(
            sim_weight=cfg["sim_weight"],
            var_weight=cfg["var_weight"],
            cov_weight=cfg["cov_weight"],
        )
    elif cfg["method"] == "barlow_twins":
        from src.losses import BarlowTwinsLoss
        ssl_criterion = BarlowTwinsLoss(
            lambda_coeff=cfg.get("bt_lambda", 0.005),
            scale_loss=cfg.get("bt_scale", 0.025),
        )
    else:
        raise ValueError(f"Unknown method: {cfg['method']}")

    dim_reg = None
    if cfg["use_dim_reg"]:
        embed_dim = cfg["proj_out_dim"] if cfg["dim_reg_on"] == "projector" else model.repr_dim
        dim_reg = DimensionalityRegularizer(
            embed_dim=embed_dim,
            momentum=cfg["dim_reg_momentum"],
        ).to(device)

    # ---- Optimizer ----
    optimizer, scheduler = build_optimizer(model, cfg)

    # ---- Logging ----
    use_wandb = setup_wandb(cfg)

    # ---- Checkpointing ----
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # ---- Training loop ----
    print(f"\nStarting training: {cfg['method']} on {cfg['dataset']}")
    print(f"  Dim regularizer: {'ON' if cfg['use_dim_reg'] else 'OFF'}")
    print(f"  Epochs: {cfg['epochs']}, Batch size: {cfg['batch_size']}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")

    for epoch in range(cfg["epochs"]):
        t0 = time.time()

        # Train
        metrics = train_one_epoch(
            model, train_loader, ssl_criterion, dim_reg,
            optimizer, cfg, epoch, device,
        )
        scheduler.step()

        epoch_time = time.time() - t0
        metrics["epoch_time"] = epoch_time
        metrics["lr"] = scheduler.get_last_lr()[0]

        # Dim reg diagnostics
        if dim_reg is not None:
            diag = dim_reg.get_diagnostics()
            metrics.update({f"dim_reg/{k}": v for k, v in diag.items()
                          if not isinstance(v, np.ndarray)})

        # Log
        print(f"Epoch {epoch:3d} | loss {metrics['loss']:.4f} | "
              f"ssl {metrics['ssl_loss']:.4f} | dim {metrics['dim_loss']:.4f} | "
              f"lr {metrics['lr']:.5f} | {epoch_time:.1f}s")

        if use_wandb:
            wandb.log({k: v for k, v in metrics.items()
                      if not isinstance(v, np.ndarray)}, step=epoch)

        # Periodic evaluation
        if (epoch + 1) % cfg["eval_every"] == 0 or epoch == cfg["epochs"] - 1:
            print(f"\n--- Evaluation at epoch {epoch} ---")

            # Linear probe
            num_classes = {"cifar100": 100, "stl10": 10, "imagenet100": 100}[cfg["dataset"]]
            eval_train, eval_test = get_eval_dataloaders(
                dataset=cfg["dataset"], data_dir=cfg["data_dir"],
                batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
            )
            probe_results = linear_probe(
                model, eval_train, eval_test,
                num_classes=num_classes, device=device,
                epochs=cfg["probe_epochs"], lr=cfg["probe_lr"],
            )
            print(f"  Linear probe: train={probe_results['train_acc']:.3f}, "
                  f"test={probe_results['test_acc']:.3f}, "
                  f"best={probe_results['best_test_acc']:.3f}")

            # Spectral diagnostics
            spectral = compute_spectral_diagnostics(
                model, eval_test, device=device, use_projector=False,
            )
            print(f"  Effective rank (encoder): {spectral['effective_rank']:.1f} / {spectral['embed_dim']}")
            print(f"  VN entropy: {spectral['von_neumann_entropy']:.3f}")

            if use_wandb:
                wandb.log({
                    "eval/train_acc": probe_results["train_acc"],
                    "eval/test_acc": probe_results["test_acc"],
                    "eval/best_test_acc": probe_results["best_test_acc"],
                    "spectral/effective_rank": spectral["effective_rank"],
                    "spectral/vn_entropy": spectral["von_neumann_entropy"],
                    "spectral/participation_ratio": spectral["participation_ratio"],
                    "spectral/condition_number": spectral["condition_number"],
                }, step=epoch)
                log_spectral_plot(spectral["eigenvalues"], epoch)

            print()

        # Checkpoint
        if (epoch + 1) % 50 == 0 or epoch == cfg["epochs"] - 1:
            ckpt_path = os.path.join(cfg["checkpoint_dir"], f"epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "metrics": metrics,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

            # Auto-backup to Google Drive if drive_backup_dir is set
            drive_dir = cfg.get("drive_backup_dir")
            if drive_dir:
                try:
                    import shutil
                    os.makedirs(drive_dir, exist_ok=True)
                    drive_ckpt = os.path.join(drive_dir, f"epoch_{epoch}.pth")
                    shutil.copy(ckpt_path, drive_ckpt)
                    print(f"  Drive backup: {drive_ckpt}")
                except Exception as e:
                    print(f"  Drive backup failed: {e}")

    if use_wandb:
        wandb.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
