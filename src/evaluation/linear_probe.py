"""Linear probing evaluation for SSL representations."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def _extract_features(model, loader, device):
    """Extract backbone features from an entire dataset."""
    model.eval()
    features, labels = [], []
    for x, y in tqdm(loader, desc="Extracting features", leave=False):
        x = x.to(device)
        h = model.encode(x)
        features.append(h.cpu())
        labels.append(y)
    return torch.cat(features), torch.cat(labels)


def linear_probe(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    device: str = "cuda",
    epochs: int = 100,
    lr: float = 0.01,
) -> dict:
    """Train a linear classifier on frozen SSL representations.

    Args:
        model: SSLModel instance (uses model.encode for features)
        train_loader, test_loader: Eval dataloaders (standard transforms)
        num_classes: Number of downstream classes
        device: Compute device
        epochs: Linear probe training epochs
        lr: Learning rate

    Returns:
        dict with train_acc, test_acc
    """
    # Extract features (frozen encoder)
    train_feats, train_labels = _extract_features(model, train_loader, device)
    test_feats, test_labels = _extract_features(model, test_loader, device)

    # Simple linear classifier
    feat_dim = train_feats.shape[1]
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Move features to device
    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    test_feats = test_feats.to(device)
    test_labels = test_labels.to(device)

    # Train
    best_test_acc = 0.0
    for epoch in range(epochs):
        classifier.train()
        logits = classifier(train_feats)
        loss = criterion(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Evaluate
        classifier.eval()
        with torch.no_grad():
            train_acc = (classifier(train_feats).argmax(1) == train_labels).float().mean().item()
            test_acc = (classifier(test_feats).argmax(1) == test_labels).float().mean().item()
            best_test_acc = max(best_test_acc, test_acc)

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "best_test_acc": best_test_acc,
    }
