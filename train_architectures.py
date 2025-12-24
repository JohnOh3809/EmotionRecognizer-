#!/usr/bin/env python3
"""
Train and compare different neural network architectures for emotion recognition.
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
from collections import Counter
from tqdm import tqdm

# Import from existing training script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_bilstm import (
    CLASSES, CLASS_TO_IDX, seed_all, collect_samples,
    EnhancedPoseDataset, create_balanced_sampler,
    BiLSTMEmotionClassifier, TCNBiLSTMClassifier, TransformerClassifier
)

# ============================================================================
# NEW ARCHITECTURES
# ============================================================================

class GRUClassifier(nn.Module):
    """GRU-based classifier - simpler than LSTM, often works well."""

    def __init__(self, in_dim: int, hidden: int = 128, num_layers: int = 2,
                 num_classes: int = 8, dropout: float = 0.3):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        x = self.proj(x)
        gru_out, _ = self.gru(x)
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)
        context = torch.sum(gru_out * attn_weights, dim=1)
        return self.head(context)


class Conv1DClassifier(nn.Module):
    """1D CNN for temporal feature extraction."""

    def __init__(self, in_dim: int, hidden: int = 128, num_classes: int = 8, dropout: float = 0.3):
        super().__init__()

        self.proj = nn.Linear(in_dim, hidden)

        self.convs = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        # x: (B, T, F)
        x = self.proj(x)  # (B, T, H)
        x = x.transpose(1, 2)  # (B, H, T)
        x = self.convs(x)  # (B, H*2, T)
        x = self.pool(x).squeeze(-1)  # (B, H*2)
        return self.head(x)


class MultiScaleClassifier(nn.Module):
    """Multi-scale temporal modeling with different kernel sizes."""

    def __init__(self, in_dim: int, hidden: int = 128, num_classes: int = 8, dropout: float = 0.3):
        super().__init__()

        self.proj = nn.Linear(in_dim, hidden)

        # Multiple scales
        self.conv3 = nn.Conv1d(hidden, hidden // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(hidden, hidden // 2, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(hidden, hidden // 2, kernel_size=7, padding=3)
        self.conv11 = nn.Conv1d(hidden, hidden // 2, kernel_size=11, padding=5)

        self.bn = nn.BatchNorm1d(hidden * 2)
        self.dropout = nn.Dropout(dropout)

        # Temporal attention
        self.attn = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        x = self.proj(x)  # (B, T, H)
        x = x.transpose(1, 2)  # (B, H, T)

        # Multi-scale features
        f3 = F.relu(self.conv3(x))
        f5 = F.relu(self.conv5(x))
        f7 = F.relu(self.conv7(x))
        f11 = F.relu(self.conv11(x))

        x = torch.cat([f3, f5, f7, f11], dim=1)  # (B, H*2, T)
        x = self.bn(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, T, H*2)

        # Attention pooling
        attn_weights = torch.softmax(self.attn(x), dim=1)
        context = torch.sum(x * attn_weights, dim=1)

        return self.head(context)


class ConvLSTMClassifier(nn.Module):
    """CNN feature extraction followed by LSTM temporal modeling."""

    def __init__(self, in_dim: int, hidden: int = 128, num_classes: int = 8, dropout: float = 0.3):
        super().__init__()

        # CNN feature extractor
        self.conv = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        self.conv_layers = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        # CNN features
        x = self.conv(x)  # (B, T, H)
        x = x.transpose(1, 2)  # (B, H, T)
        x = self.conv_layers(x)  # (B, H, T)
        x = x.transpose(1, 2)  # (B, T, H)

        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)

        # Use last hidden states from both directions
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        x = torch.cat([h_forward, h_backward], dim=1)
        x = self.dropout(x)

        return self.head(x)


class ResidualBlock(nn.Module):
    """Residual block for deep networks."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class DeepResidualClassifier(nn.Module):
    """Deep residual network for pose classification."""

    def __init__(self, in_dim: int, hidden: int = 256, num_blocks: int = 4,
                 num_classes: int = 8, dropout: float = 0.3):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden, dropout) for _ in range(num_blocks)
        ])

        # Temporal pooling
        self.pool_attention = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1, bias=False),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        x = self.proj(x)  # (B, T, H)
        x = self.blocks(x)  # (B, T, H)

        # Attention pooling
        attn = torch.softmax(self.pool_attention(x), dim=1)
        x = torch.sum(x * attn, dim=1)  # (B, H)

        return self.head(x)


class SkeletonGraphConv(nn.Module):
    """Graph convolution layer for skeleton data."""

    def __init__(self, in_features: int, out_features: int, num_joints: int = 17):
        super().__init__()
        self.num_joints = num_joints

        # Learnable adjacency matrix
        self.adj = nn.Parameter(torch.randn(num_joints, num_joints) * 0.01)
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        # x: (B, T, J, F) where J=num_joints, F=features per joint
        B, T, J, F = x.shape

        # Normalize adjacency
        adj = torch.softmax(self.adj, dim=1)

        # Graph convolution: aggregate features from neighbors
        x = x.view(B * T, J, F)
        x = torch.bmm(adj.unsqueeze(0).expand(B * T, -1, -1), x)
        x = self.fc(x)
        x = x.view(B * T, -1).transpose(0, 1)
        x = self.bn(x.view(-1, x.size(0))).view(x.size(1), -1).transpose(0, 1)
        x = x.view(B, T, J, -1)

        return F.relu(x)


class GraphSkeletonClassifier(nn.Module):
    """Graph-based classifier that models skeleton structure."""

    def __init__(self, in_dim: int, hidden: int = 128, num_classes: int = 8,
                 dropout: float = 0.3, num_joints: int = 17):
        super().__init__()

        # Features per joint (in_dim / (num_joints * num_people))
        # in_dim = 238 = 17 joints * 7 features * 2 people
        self.num_joints = num_joints
        self.features_per_joint = 7  # x, y, conf, vx, vy, ax, ay
        self.num_people = 2

        # Graph convolutions
        self.gc1 = nn.Linear(self.features_per_joint, hidden // 2)
        self.gc2 = nn.Linear(hidden // 2, hidden // 2)

        # Learnable adjacency (shared for all people)
        self.adj = nn.Parameter(torch.randn(num_joints, num_joints) * 0.01)

        # Temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden // 2 * num_joints * self.num_people,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        B, T, F = x.shape

        # Reshape to (B, T, num_people, num_joints, features_per_joint)
        x = x.view(B, T, self.num_people, self.num_joints, self.features_per_joint)

        # Graph convolution on each person's skeleton
        adj = torch.softmax(self.adj, dim=1)

        # Process each timestep
        x = x.view(B * T * self.num_people, self.num_joints, self.features_per_joint)
        x = F.relu(self.gc1(x))
        x = torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), x)
        x = F.relu(self.gc2(x))

        # Reshape back
        x = x.view(B, T, self.num_people * self.num_joints * (self.gc2.out_features))

        # Temporal modeling
        lstm_out, (h_n, _) = self.lstm(x)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        x = torch.cat([h_forward, h_backward], dim=1)
        x = self.dropout(x)

        return self.head(x)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, val_loader, test_loader, device,
                epochs=60, lr=5e-4, weight_decay=1e-4, label_smoothing=0.1):
    """Train a model and return results."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss with uniform weights (balanced sampler handles class imbalance)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_val_acc = 0
    best_model_state = None
    history = {"train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        correct, total = 0, 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += feats.size(0)

        train_acc = correct / total
        scheduler.step()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += feats.size(0)

        val_acc = val_correct / val_total
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            print(f"    Epoch {epoch}: train={train_acc:.3f}, val={val_acc:.3f}")

    # Load best model and evaluate on test
    model.load_state_dict(best_model_state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for feats, labels in test_loader:
            feats = feats.to(device)
            logits = model(feats)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = (all_preds == all_labels).mean()

    # Per-class accuracy
    class_accs = {}
    for i, cls in enumerate(CLASSES[:-1]):  # Exclude 'other'
        mask = all_labels == i
        if mask.sum() > 0:
            class_accs[cls] = (all_preds[mask] == i).mean()

    return {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "class_accs": class_accs,
        "history": history,
        "preds": all_preds,
        "labels": all_labels,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_dir = Path("./data_enhanced")
    all_items = collect_samples(data_dir)

    # Split data
    random.shuffle(all_items)
    n = len(all_items)
    train_items = all_items[:int(n * 0.8)]
    val_items = all_items[int(n * 0.8):int(n * 0.9)]
    test_items = all_items[int(n * 0.9):]

    train_labels = [it[3] for it in train_items]

    print(f"Data: train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")

    # Create datasets
    train_ds = EnhancedPoseDataset(train_items, seq_len=64, num_people=2, train=True, augment=True, aug_prob=0.5)
    val_ds = EnhancedPoseDataset(val_items, seq_len=64, num_people=2, train=False, augment=False)
    test_ds = EnhancedPoseDataset(test_items, seq_len=64, num_people=2, train=False, augment=False)

    in_dim = train_ds.in_dim
    print(f"Input dimension: {in_dim}")

    # Balanced sampler
    train_sampler = create_balanced_sampler(train_labels, len(CLASSES))

    train_loader = DataLoader(train_ds, batch_size=32, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Define architectures to test
    architectures = {
        # Existing architectures with different configs
        "BiLSTM_h128_l1": lambda: BiLSTMEmotionClassifier(in_dim, hidden=128, num_layers=1, num_classes=8, dropout=0.3),
        "BiLSTM_h256_l2": lambda: BiLSTMEmotionClassifier(in_dim, hidden=256, num_layers=2, num_classes=8, dropout=0.4),
        "TCN_BiLSTM": lambda: TCNBiLSTMClassifier(in_dim, hidden=128, num_layers=1, num_classes=8, dropout=0.3),
        "Transformer_h128": lambda: TransformerClassifier(in_dim, hidden=128, num_layers=2, num_heads=4, num_classes=8, dropout=0.3),

        # New architectures
        "GRU_h128_l2": lambda: GRUClassifier(in_dim, hidden=128, num_layers=2, num_classes=8, dropout=0.3),
        "Conv1D": lambda: Conv1DClassifier(in_dim, hidden=128, num_classes=8, dropout=0.3),
        "MultiScale": lambda: MultiScaleClassifier(in_dim, hidden=128, num_classes=8, dropout=0.3),
        "ConvLSTM": lambda: ConvLSTMClassifier(in_dim, hidden=128, num_classes=8, dropout=0.3),
        "DeepResidual": lambda: DeepResidualClassifier(in_dim, hidden=256, num_blocks=4, num_classes=8, dropout=0.3),
        "GraphSkeleton": lambda: GraphSkeletonClassifier(in_dim, hidden=128, num_classes=8, dropout=0.3),
    }

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./runs/architecture_comparison_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON")
    print("=" * 70)

    for name, model_fn in architectures.items():
        print(f"\n>>> Training {name}...")

        model = model_fn().to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {num_params:,}")

        try:
            result = train_model(model, train_loader, val_loader, test_loader, device, epochs=60)

            results.append({
                "name": name,
                "params": num_params,
                "best_val_acc": result["best_val_acc"],
                "test_acc": result["test_acc"],
                "class_accs": {k: float(v) for k, v in result["class_accs"].items()},
            })

            # Save predictions
            np.savez(output_dir / f"{name}_predictions.npz",
                     preds=result["preds"], labels=result["labels"])

            print(f"    Best Val: {result['best_val_acc']*100:.2f}%, Test: {result['test_acc']*100:.2f}%")

            # Classes with >5% accuracy
            good_classes = sum(1 for acc in result["class_accs"].values() if acc > 0.05)
            print(f"    Classes >5%: {good_classes}/7")

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "name": name,
                "params": num_params,
                "best_val_acc": 0,
                "test_acc": 0,
                "error": str(e),
            })

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Architecture':<20} {'Params':>10} {'Val Acc':>10} {'Test Acc':>10} {'Classes>5%':>12}")
    print("-" * 65)

    results_sorted = sorted(results, key=lambda x: x.get("test_acc", 0), reverse=True)
    for r in results_sorted:
        if "error" not in r:
            good_classes = sum(1 for acc in r.get("class_accs", {}).values() if acc > 0.05)
            print(f"{r['name']:<20} {r['params']:>10,} {r['best_val_acc']*100:>9.2f}% {r['test_acc']*100:>9.2f}% {good_classes:>10}/7")
        else:
            print(f"{r['name']:<20} {r['params']:>10,} {'ERROR':>10} {'':>10}")

    print(f"\nResults saved to: {output_dir}")

    # Generate comparison plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sort by test accuracy
    valid_results = [r for r in results_sorted if "error" not in r]
    names = [r["name"] for r in valid_results]
    val_accs = [r["best_val_acc"] * 100 for r in valid_results]
    test_accs = [r["test_acc"] * 100 for r in valid_results]
    params = [r["params"] / 1000 for r in valid_results]  # in thousands

    x = np.arange(len(names))
    width = 0.35

    # Plot 1: Accuracy comparison
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, val_accs, width, label='Val Acc', color='steelblue', edgecolor='black')
    bars2 = ax1.bar(x + width/2, test_accs, width, label='Test Acc', color='coral', edgecolor='black')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Architecture Comparison: Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=14.3, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Efficiency (accuracy vs parameters)
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_results)))
    for i, (name, test, param) in enumerate(zip(names, test_accs, params)):
        ax2.scatter(param, test, s=100, c=[colors[i]], label=name, edgecolors='black')
    ax2.set_xlabel('Parameters (thousands)')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Efficiency: Accuracy vs Model Size')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "architecture_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(Path("comparison_plots") / "architecture_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {output_dir / 'architecture_comparison.png'}")


if __name__ == "__main__":
    main()
