"""YAML config loading with defaults."""

import yaml
from pathlib import Path


_DEFAULTS = {
    # Model
    "backbone": "resnet18",
    "proj_hidden_dim": 2048,
    "proj_out_dim": 128,
    "proj_layers": 3,

    # SSL method
    "method": "simclr",  # "simclr", "vicreg", or "barlow_twins"
    "temperature": 0.5,  # SimCLR
    "sim_weight": 25.0,  # VICReg
    "var_weight": 25.0,
    "cov_weight": 1.0,
    "bt_lambda": 0.005,  # Barlow Twins off-diagonal weight
    "bt_scale": 0.025,   # Barlow Twins loss scale

    # Dimensionality regularizer
    "use_dim_reg": False,
    "dim_reg_weight": 0.1,
    "dim_reg_momentum": 0.99,
    "dim_reg_schedule": "linear_warmup",  # "constant", "linear_warmup", "cosine_warmup"
    "dim_reg_warmup_epochs": 10,
    "dim_reg_on": "projector",  # "projector" or "encoder"

    # Training
    "dataset": "cifar100",
    "data_dir": "./data",
    "batch_size": 256,
    "epochs": 200,
    "optimizer": "adamw",
    "lr": 0.001,
    "weight_decay": 1e-4,
    "lr_schedule": "cosine",
    "warmup_epochs": 10,
    "num_workers": 2,

    # Evaluation
    "eval_every": 20,
    "probe_epochs": 100,
    "probe_lr": 0.01,

    # Logging
    "wandb_project": "dim-ssl",
    "wandb_entity": None,
    "log_spectral_every": 20,

    # Misc
    "seed": 42,
    "device": "cuda",
    "checkpoint_dir": "./checkpoints",
    "drive_backup_dir": None,
}


def load_config(path: str = None, overrides: dict = None) -> dict:
    """Load config from YAML file, filling in defaults.

    Args:
        path: Path to YAML config file (optional)
        overrides: Dict of overrides applied last (optional)

    Returns:
        Complete config dict
    """
    cfg = dict(_DEFAULTS)

    if path is not None:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                file_cfg = yaml.safe_load(f) or {}
            cfg.update(file_cfg)
        else:
            raise FileNotFoundError(f"Config file not found: {path}")

    if overrides:
        cfg.update(overrides)

    # Coerce fields that YAML sometimes parses as strings (e.g. 1e-4)
    _FLOAT_KEYS = {
        "lr", "weight_decay", "temperature", "sim_weight", "var_weight",
        "cov_weight", "dim_reg_weight", "dim_reg_momentum", "probe_lr",
    }
    for k in _FLOAT_KEYS:
        if k in cfg and isinstance(cfg[k], str):
            cfg[k] = float(cfg[k])

    return cfg
