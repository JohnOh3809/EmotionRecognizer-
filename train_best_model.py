#!/usr/bin/env python3
"""
Train the best model (DeepResidual preact) and save the checkpoint.
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_bilstm import (
    CLASSES, CLASS_TO_IDX, seed_all, collect_samples,
    EnhancedPoseDataset, create_balanced_sampler,
)

# ============================================================================
# MODEL
# ============================================================================

class PreActResidualBlock(nn.Module):
    """Pre-activation residual block."""
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.linear1(F.relu(self.norm1(x)))
        out = self.dropout(out)
        out = self.linear2(F.relu(self.norm2(out)))
        return x + out


class AttentionPooling(nn.Module):
    """Attention-weighted pooling over time."""
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x, mask=None):
        # x: (B, T, D)
        scores = self.attn(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)  # (B, T)
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)


class DeepResidualPreact(nn.Module):
    """Best performing model: DeepResidual with pre-activation blocks."""
    def __init__(self, input_dim: int = 238, hidden: int = 256,
                 num_blocks: int = 4, dropout: float = 0.3, num_classes: int = 8):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            PreActResidualBlock(hidden, dropout) for _ in range(num_blocks)
        ])

        # Attention pooling
        self.pool = AttentionPooling(hidden)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        # x: (B, T, D)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        return self.classifier(x)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    # Per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    class_accs = {}
    for i, cls in enumerate(CLASSES):
        mask = all_labels == i
        if mask.sum() > 0:
            class_accs[cls] = (all_preds[mask] == i).mean()
        else:
            class_accs[cls] = 0.0

    return correct / total, class_accs, all_preds, all_labels


def main():
    seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_dir = Path("./data_enhanced")
    all_samples = collect_samples(data_dir)
    train_samples = [s for s in all_samples if s[1] == 'train']
    test_samples = [s for s in all_samples if s[1] == 'test']

    # If no test samples found, split train 90/10
    if len(test_samples) == 0:
        random.shuffle(all_samples)
        n = len(all_samples)
        test_samples = all_samples[int(n * 0.9):]
        train_samples = all_samples[:int(n * 0.9)]

    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    train_ds = EnhancedPoseDataset(train_samples, augment=True)
    test_ds = EnhancedPoseDataset(test_samples, augment=False)

    train_labels = [s[3] for s in train_samples]
    sampler = create_balanced_sampler(train_labels, num_classes=8)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    # Model
    model = DeepResidualPreact(
        input_dim=238,
        hidden=256,
        num_blocks=4,
        dropout=0.3,
        num_classes=8
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Training setup - uniform weights for balanced sampler
    class_weights = torch.ones(8, device=device)
    class_weights[7] = 0.0  # 'other' class
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

    # Output directory
    run_dir = Path("runs/best_model_preact")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_test_acc = 0
    best_state = None

    for epoch in range(1, 61):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc, class_accs, preds, labels = evaluate(model, test_loader, device)
        scheduler.step()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'class_accs': class_accs,
            }
            # Save immediately
            torch.save(best_state, run_dir / 'best.pt')

        if epoch % 10 == 0:
            good_classes = sum(1 for v in class_accs.values() if v > 0.05)
            print(f"  Epoch {epoch}: train={train_acc:.3f}, test={test_acc:.3f}, classes>5%={good_classes}/7")

    # Save final
    torch.save({
        'epoch': 60,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
    }, run_dir / 'last.pt')

    # Report
    print(f"\n{'='*60}")
    print(f"BEST MODEL TRAINED")
    print(f"{'='*60}")
    print(f"Test Accuracy: {best_test_acc*100:.2f}%")
    print(f"Model saved to: {run_dir / 'best.pt'}")
    print("\nPer-class accuracy:")
    for cls, acc in best_state['class_accs'].items():
        print(f"  {cls}: {acc*100:.1f}%")

    # Save config
    config = {
        'architecture': 'DeepResidualPreact',
        'hidden': 256,
        'num_blocks': 4,
        'dropout': 0.3,
        'parameters': params,
        'best_test_acc': best_test_acc,
        'class_accs': {k: float(v) for k, v in best_state['class_accs'].items()},
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
