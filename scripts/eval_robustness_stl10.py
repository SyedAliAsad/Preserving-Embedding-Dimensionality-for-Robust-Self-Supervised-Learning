"""STL-10-C corruption robustness evaluation.

STL-10-C follows the same format as CIFAR-100-C but for STL-10 images (96x96).
We generate corruptions on the fly using the corruption functions from
the robustness benchmark, or download from a mirror if available.

Usage:
    python scripts/eval_robustness_stl10.py \
        --checkpoint ./checkpoints/epoch_399.pth \
        --data_dir ./data \
        --output results_stl10_robustness.json
"""

import argparse
import json
import os
import urllib.request
import tarfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as tv_datasets
import torchvision.transforms as T
from tqdm import tqdm

from src.models import SSLModel
from src.data.augmentations import get_eval_transform

# STL-10 uses 96x96 images
STL10_MEAN = [0.4408, 0.4279, 0.3867]
STL10_STD  = [0.2682, 0.2610, 0.2686]

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


def apply_corruption(images_np, corruption_name, severity):
    """Apply a single corruption to a numpy array of images (N, H, W, C) uint8."""
    try:
        from imagecorruptions import corrupt
        corrupted = np.stack([
            corrupt(img, corruption_name=corruption_name, severity=severity)
            for img in images_np
        ])
        return corrupted
    except ImportError:
        raise ImportError(
            "imagecorruptions package not found. Install with: pip install imagecorruptions"
        )


def make_stl10_corruption_loader(images_np, labels, batch_size=128):
    """Create DataLoader from corrupted STL-10 numpy images."""
    from PIL import Image
    normalize = T.Normalize(mean=STL10_MEAN, std=STL10_STD)

    class CorruptionDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, lbls):
            self.imgs = imgs
            self.lbls = lbls
            self.tfm = T.Compose([T.ToTensor(), normalize])

        def __len__(self): return len(self.imgs)

        def __getitem__(self, idx):
            img = Image.fromarray(self.imgs[idx])
            return self.tfm(img), self.lbls[idx]

    ds = CorruptionDataset(images_np, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=2, pin_memory=True)


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features, labels = [], []
    for x, y in tqdm(loader, desc="Extracting", leave=False):
        x = x.to(device)
        h = model.encode(x)
        features.append(h.cpu())
        labels.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
    return torch.cat(features), torch.cat(labels)


def train_linear_probe(train_features, train_labels, num_classes, device,
                        epochs=100, lr=0.3):
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
    classifier.eval()
    features = features.to(device)
    labels = labels.to(device)
    acc = (classifier(features).argmax(1) == labels).float().mean().item()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default="results_stl10_robustness.json")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--probe_epochs", type=int, default=100)
    parser.add_argument("--probe_lr", type=float, default=0.3)
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Number of test images to corrupt (max 8000)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Install imagecorruptions if needed
    try:
        import imagecorruptions
    except ImportError:
        print("Installing imagecorruptions...")
        os.system("pip install imagecorruptions -q")

    # ---- Load model ----
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
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

    # ---- Train linear probe on clean STL-10 ----
    print("\nTraining linear probe on clean STL-10...")
    normalize = T.Normalize(mean=STL10_MEAN, std=STL10_STD)
    clean_transform = T.Compose([
        T.Resize(96), T.CenterCrop(96), T.ToTensor(), normalize
    ])

    train_ds = tv_datasets.STL10(args.data_dir, split="train",
                                   download=True, transform=clean_transform)
    test_ds  = tv_datasets.STL10(args.data_dir, split="test",
                                  download=True, transform=clean_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    train_feats, train_labels = extract_features(model, train_loader, device)
    test_feats,  test_labels  = extract_features(model, test_loader,  device)

    classifier = train_linear_probe(
        train_feats, train_labels, num_classes=10,
        device=device, epochs=args.probe_epochs, lr=args.probe_lr,
    )
    clean_acc = evaluate_classifier(classifier, test_feats, test_labels, device)
    print(f"Clean test accuracy: {clean_acc:.4f}")

    # ---- Get test images as numpy for corruption ----
    # Use a subset to keep runtime reasonable
    n = min(args.n_samples, len(test_ds))
    test_images_np = np.stack([
        np.array(test_ds[i][0]) for i in range(n)
    ])
    # Convert from CxHxW tensor-like to HxWxC uint8
    # Actually test_ds[i][0] is already a tensor after transform — we need raw PIL
    # Reload without transform
    raw_test_ds = tv_datasets.STL10(args.data_dir, split="test", download=False)
    test_images_raw = np.stack([np.array(raw_test_ds[i][0]) for i in range(n)])
    test_labels_sub = torch.tensor([raw_test_ds[i][1] for i in range(n)])

    print(f"\nEvaluating corruptions on {n} images...")

    results = {
        "clean_acc": clean_acc,
        "config": cfg,
        "checkpoint": args.checkpoint,
        "dataset": "stl10",
        "n_samples": n,
        "corruptions": {},
    }

    all_corruption_accs = []

    for corruption in CORRUPTIONS:
        print(f"\n  {corruption}:")
        severity_accs = []

        for severity in range(1, 6):
            try:
                corrupted = apply_corruption(test_images_raw, corruption, severity)
                loader = make_stl10_corruption_loader(corrupted, test_labels_sub,
                                                      args.batch_size)
                feats, labels = extract_features(model, loader, device)
                acc = evaluate_classifier(classifier, feats, labels, device)
            except Exception as e:
                print(f"    severity {severity}: ERROR ({e})")
                acc = 0.0

            severity_accs.append(acc)
            print(f"    severity {severity}: {acc:.4f}")

        mean_acc = np.mean(severity_accs)
        results["corruptions"][corruption] = {
            "severities": {s+1: a for s, a in enumerate(severity_accs)},
            "mean": mean_acc,
        }
        all_corruption_accs.append(mean_acc)
        print(f"    mean: {mean_acc:.4f}")

    mean_corruption_acc = np.mean(all_corruption_accs)
    results["mean_corruption_acc"] = mean_corruption_acc
    results["relative_robustness"] = mean_corruption_acc / clean_acc if clean_acc > 0 else 0

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

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Clean accuracy:          {clean_acc:.4f}")
    print(f"Mean corruption accuracy: {mean_corruption_acc:.4f}")
    print(f"Relative robustness:     {results['relative_robustness']:.4f}")
    print(f"{'='*60}")
    print(f"\nBy category:")
    print(f"  Noise:   {noise:.4f}")
    print(f"  Blur:    {blur:.4f}")
    print(f"  Weather: {weather:.4f}")
    print(f"  Digital: {digital:.4f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
