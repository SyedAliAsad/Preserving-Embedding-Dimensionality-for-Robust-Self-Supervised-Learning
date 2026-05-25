"""CIFAR-100-C corruption robustness evaluation.

Downloads CIFAR-100-C (if needed), loads a trained SSL model,
trains a linear probe on clean data, then evaluates on all
15 corruption types × 5 severity levels.

Usage:
    python scripts/eval_robustness.py \
        --checkpoint ./checkpoints/epoch_399.pth \
        --data_dir ./data \
        --output results_robustness.json
"""

import argparse
import json
import os
import urllib.request
import tarfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models import SSLModel
from src.data.augmentations import get_eval_transform


CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

CIFAR100C_URL = "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar.gz"


def download_cifar100c(data_dir: str):
    """Download and extract CIFAR-100-C if not present."""
    cifar100c_dir = os.path.join(data_dir, "CIFAR-100-C")
    if os.path.exists(cifar100c_dir):
        print(f"CIFAR-100-C already exists at {cifar100c_dir}")
        return cifar100c_dir

    tar_path = os.path.join(data_dir, "CIFAR-100-C.tar.gz")
    if not os.path.exists(tar_path):
        print(f"Downloading CIFAR-100-C (~680MB)...")
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(CIFAR100C_URL, tar_path)
        print("Download complete.")

    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print(f"Extracted to {cifar100c_dir}")

    # Clean up tar file to save space
    os.remove(tar_path)
    return cifar100c_dir


def load_corruption_data(cifar100c_dir: str, corruption: str, severity: int):
    """Load a specific corruption type and severity from CIFAR-100-C.

    CIFAR-100-C stores each corruption as a numpy array of shape (50000, 32, 32, 3).
    The 50000 images are 5 severities × 10000 images each, stored sequentially.

    Args:
        cifar100c_dir: Path to CIFAR-100-C directory
        corruption: Corruption name (e.g., "gaussian_noise")
        severity: Severity level 1-5

    Returns:
        images: numpy array (10000, 32, 32, 3) uint8
        labels: numpy array (10000,) int
    """
    images_path = os.path.join(cifar100c_dir, f"{corruption}.npy")
    labels_path = os.path.join(cifar100c_dir, "labels.npy")

    all_images = np.load(images_path)  # (50000, 32, 32, 3)
    all_labels = np.load(labels_path)  # (50000,)

    # Extract the 10000 images for this severity level
    start = (severity - 1) * 10000
    end = severity * 10000
    images = all_images[start:end]
    labels = all_labels[start:end]

    return images, labels


def make_corruption_loader(images, labels, transform, batch_size=256):
    """Create a DataLoader from corruption numpy arrays.

    Applies the same eval transform used for clean test data.
    """
    from PIL import Image

    class CorruptionDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, lbls, tfm):
            self.imgs = imgs
            self.lbls = lbls
            self.tfm = tfm

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            img = Image.fromarray(self.imgs[idx])
            return self.tfm(img), self.lbls[idx]

    ds = CorruptionDataset(images, labels, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=2, pin_memory=True)


@torch.no_grad()
def extract_features(model, loader, device):
    """Extract encoder features from a dataloader."""
    model.eval()
    features, labels = [], []
    for x, y in tqdm(loader, desc="Extracting", leave=False):
        x = x.to(device)
        h = model.encode(x)
        features.append(h.cpu())
        labels.append(y)
    return torch.cat(features), torch.cat(labels)


def train_linear_probe(train_features, train_labels, num_classes, device,
                       epochs=100, lr=0.3):
    """Train a linear classifier on extracted features."""
    feat_dim = train_features.shape[1]
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)

    for epoch in range(epochs):
        classifier.train()
        logits = classifier(train_features)
        loss = criterion(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return classifier


@torch.no_grad()
def evaluate_classifier(classifier, features, labels, device):
    """Evaluate a linear classifier on features."""
    classifier.eval()
    features = features.to(device)
    labels = labels.to(device)
    logits = classifier(features)
    acc = (logits.argmax(1) == labels).float().mean().item()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default="results_robustness.json")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--probe_epochs", type=int, default=100)
    parser.add_argument("--probe_lr", type=float, default=0.3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load model ----
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    model = SSLModel(
        backbone=cfg["backbone"],
        proj_hidden_dim=cfg["proj_hidden_dim"],
        proj_out_dim=cfg["proj_out_dim"],
        proj_layers=cfg["proj_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded: {cfg['method']} | backbone={cfg['backbone']}")

    # ---- Download CIFAR-100-C ----
    cifar100c_dir = download_cifar100c(args.data_dir)

    # ---- Train linear probe on clean CIFAR-100 ----
    print("\nTraining linear probe on clean CIFAR-100...")
    import torchvision.datasets as tv_datasets
    transform = get_eval_transform("cifar100")

    clean_train = tv_datasets.CIFAR100(args.data_dir, train=True, download=True,
                                        transform=transform)
    clean_test = tv_datasets.CIFAR100(args.data_dir, train=False, download=True,
                                       transform=transform)

    train_loader = DataLoader(clean_train, batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(clean_test, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    # Extract clean features
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)

    # Train linear probe
    classifier = train_linear_probe(
        train_features, train_labels,
        num_classes=100, device=device,
        epochs=args.probe_epochs, lr=args.probe_lr,
    )

    # Clean accuracy
    clean_acc = evaluate_classifier(classifier, test_features, test_labels, device)
    print(f"Clean test accuracy: {clean_acc:.4f}")

    # ---- Evaluate on corruptions ----
    print(f"\nEvaluating on {len(CORRUPTIONS)} corruptions × 5 severities...")
    results = {
        "clean_acc": clean_acc,
        "config": cfg,
        "checkpoint": args.checkpoint,
        "corruptions": {},
    }

    all_corruption_accs = []

    for corruption in CORRUPTIONS:
        print(f"\n  {corruption}:")
        severity_accs = []

        for severity in range(1, 6):
            images, labels = load_corruption_data(cifar100c_dir, corruption, severity)
            loader = make_corruption_loader(images, labels, transform, args.batch_size)

            # Extract features and evaluate
            corr_features, corr_labels = extract_features(model, loader, device)
            acc = evaluate_classifier(classifier, corr_features, corr_labels, device)
            severity_accs.append(acc)
            print(f"    severity {severity}: {acc:.4f}")

        mean_acc = np.mean(severity_accs)
        results["corruptions"][corruption] = {
            "severities": {s+1: a for s, a in enumerate(severity_accs)},
            "mean": mean_acc,
        }
        all_corruption_accs.append(mean_acc)
        print(f"    mean: {mean_acc:.4f}")

    # ---- Summary ----
    mean_corruption_acc = np.mean(all_corruption_accs)
    results["mean_corruption_acc"] = mean_corruption_acc
    results["relative_robustness"] = mean_corruption_acc / clean_acc if clean_acc > 0 else 0

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Clean accuracy:          {clean_acc:.4f}")
    print(f"Mean corruption accuracy: {mean_corruption_acc:.4f}")
    print(f"Relative robustness:     {results['relative_robustness']:.4f}")
    print(f"{'='*60}")

    # Per-category summary
    noise = np.mean([results["corruptions"][c]["mean"]
                     for c in ["gaussian_noise", "shot_noise", "impulse_noise"]])
    blur = np.mean([results["corruptions"][c]["mean"]
                    for c in ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"]])
    weather = np.mean([results["corruptions"][c]["mean"]
                       for c in ["snow", "frost", "fog", "brightness"]])
    digital = np.mean([results["corruptions"][c]["mean"]
                       for c in ["contrast", "elastic_transform", "pixelate", "jpeg_compression"]])

    results["category_means"] = {
        "noise": noise, "blur": blur, "weather": weather, "digital": digital,
    }
    print(f"\nBy category:")
    print(f"  Noise:   {noise:.4f}")
    print(f"  Blur:    {blur:.4f}")
    print(f"  Weather: {weather:.4f}")
    print(f"  Digital: {digital:.4f}")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
