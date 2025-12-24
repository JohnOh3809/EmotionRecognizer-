#!/usr/bin/env python3
"""
Train variations of the DeepResidual architecture to find optimal configuration.
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

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_bilstm import (
    CLASSES, CLASS_TO_IDX, seed_all, collect_samples,
    EnhancedPoseDataset, create_balanced_sampler,
)

# ============================================================================
# DEEP RESIDUAL VARIATIONS
# ============================================================================

class ResidualBlock(nn.Module):
    """Standard residual block."""
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


class BottleneckBlock(nn.Module):
    """Bottleneck residual block (compress then expand)."""
    def __init__(self, dim: int, bottleneck_ratio: float = 0.25, dropout: float = 0.3):
        super().__init__()
        bottleneck_dim = int(dim * bottleneck_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class PreActResidualBlock(nn.Module):
    """Pre-activation residual block (BN-ReLU-Conv pattern)."""
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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, T, D)
        # Global average pooling over time
        se = x.mean(dim=1)  # (B, D)
        se = self.fc(se).unsqueeze(1)  # (B, 1, D)
        return x * se


class DeepResidualV2(nn.Module):
    """
    DeepResidual with various configuration options.
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        num_blocks: int = 4,
        num_classes: int = 8,
        dropout: float = 0.3,
        block_type: str = "standard",  # standard, bottleneck, preact
        pooling: str = "attention",    # attention, mean, max, last
        use_se: bool = False,          # Squeeze-and-Excitation
    ):
        super().__init__()

        self.pooling_type = pooling
        self.use_se = use_se

        # Input projection
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        if block_type == "standard":
            self.blocks = nn.Sequential(*[
                ResidualBlock(hidden, dropout) for _ in range(num_blocks)
            ])
        elif block_type == "bottleneck":
            self.blocks = nn.Sequential(*[
                BottleneckBlock(hidden, 0.25, dropout) for _ in range(num_blocks)
            ])
        elif block_type == "preact":
            self.blocks = nn.Sequential(*[
                PreActResidualBlock(hidden, dropout) for _ in range(num_blocks)
            ])
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        # Optional SE block
        if use_se:
            self.se = SEBlock(hidden)

        # Temporal pooling
        if pooling == "attention":
            self.pool_attention = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.Tanh(),
                nn.Linear(hidden // 2, 1, bias=False),
            )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        x = self.proj(x)  # (B, T, H)
        x = self.blocks(x)  # (B, T, H)

        if self.use_se:
            x = self.se(x)

        # Temporal pooling
        if self.pooling_type == "attention":
            attn = torch.softmax(self.pool_attention(x), dim=1)
            x = torch.sum(x * attn, dim=1)
        elif self.pooling_type == "mean":
            x = x.mean(dim=1)
        elif self.pooling_type == "max":
            x = x.max(dim=1)[0]
        elif self.pooling_type == "last":
            x = x[:, -1, :]

        return self.head(x)


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, test_loader, device,
                epochs=60, lr=5e-4, weight_decay=1e-4):
    """Train model and return results."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

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
            print(f"      Epoch {epoch}: train={train_acc:.3f}, val={val_acc:.3f}")

    # Test evaluation
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
    for i, cls in enumerate(CLASSES[:-1]):
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


def main():
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_dir = Path("./data_enhanced")
    all_items = collect_samples(data_dir)
    random.shuffle(all_items)

    n = len(all_items)
    train_items = all_items[:int(n * 0.8)]
    val_items = all_items[int(n * 0.8):int(n * 0.9)]
    test_items = all_items[int(n * 0.9):]
    train_labels = [it[3] for it in train_items]

    print(f"Data: train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")

    # Datasets
    train_ds = EnhancedPoseDataset(train_items, seq_len=64, num_people=2, train=True, augment=True, aug_prob=0.5)
    val_ds = EnhancedPoseDataset(val_items, seq_len=64, num_people=2, train=False, augment=False)
    test_ds = EnhancedPoseDataset(test_items, seq_len=64, num_people=2, train=False, augment=False)

    in_dim = train_ds.in_dim
    train_sampler = create_balanced_sampler(train_labels, len(CLASSES))

    train_loader = DataLoader(train_ds, batch_size=32, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Define variations to test
    variations = [
        # Vary hidden dimension
        {"name": "h128_b4", "hidden": 128, "num_blocks": 4, "dropout": 0.3, "block_type": "standard", "pooling": "attention", "use_se": False},
        {"name": "h256_b4", "hidden": 256, "num_blocks": 4, "dropout": 0.3, "block_type": "standard", "pooling": "attention", "use_se": False},
        {"name": "h512_b4", "hidden": 512, "num_blocks": 4, "dropout": 0.3, "block_type": "standard", "pooling": "attention", "use_se": False},

        # Vary number of blocks
        {"name": "h256_b2", "hidden": 256, "num_blocks": 2, "dropout": 0.3, "block_type": "standard", "pooling": "attention", "use_se": False},
        {"name": "h256_b6", "hidden": 256, "num_blocks": 6, "dropout": 0.3, "block_type": "standard", "pooling": "attention", "use_se": False},
        {"name": "h256_b8", "hidden": 256, "num_blocks": 8, "dropout": 0.3, "block_type": "standard", "pooling": "attention", "use_se": False},

        # Vary dropout
        {"name": "h256_b4_d02", "hidden": 256, "num_blocks": 4, "dropout": 0.2, "block_type": "standard", "pooling": "attention", "use_se": False},
        {"name": "h256_b4_d04", "hidden": 256, "num_blocks": 4, "dropout": 0.4, "block_type": "standard", "pooling": "attention", "use_se": False},
        {"name": "h256_b4_d05", "hidden": 256, "num_blocks": 4, "dropout": 0.5, "block_type": "standard", "pooling": "attention", "use_se": False},

        # Different block types
        {"name": "h256_b4_bottleneck", "hidden": 256, "num_blocks": 4, "dropout": 0.3, "block_type": "bottleneck", "pooling": "attention", "use_se": False},
        {"name": "h256_b4_preact", "hidden": 256, "num_blocks": 4, "dropout": 0.3, "block_type": "preact", "pooling": "attention", "use_se": False},

        # Different pooling
        {"name": "h256_b4_meanpool", "hidden": 256, "num_blocks": 4, "dropout": 0.3, "block_type": "standard", "pooling": "mean", "use_se": False},
        {"name": "h256_b4_maxpool", "hidden": 256, "num_blocks": 4, "dropout": 0.3, "block_type": "standard", "pooling": "max", "use_se": False},

        # With Squeeze-and-Excitation
        {"name": "h256_b4_SE", "hidden": 256, "num_blocks": 4, "dropout": 0.3, "block_type": "standard", "pooling": "attention", "use_se": True},

        # Best combinations
        {"name": "h384_b6_preact_SE", "hidden": 384, "num_blocks": 6, "dropout": 0.35, "block_type": "preact", "pooling": "attention", "use_se": True},
    ]

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./runs/deep_residual_variations_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print("\n" + "=" * 70)
    print("DEEP RESIDUAL VARIATIONS")
    print("=" * 70)

    for var in variations:
        name = var["name"]
        print(f"\n>>> Training {name}...")

        model = DeepResidualV2(
            in_dim=in_dim,
            hidden=var["hidden"],
            num_blocks=var["num_blocks"],
            num_classes=8,
            dropout=var["dropout"],
            block_type=var["block_type"],
            pooling=var["pooling"],
            use_se=var["use_se"],
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"    Config: hidden={var['hidden']}, blocks={var['num_blocks']}, "
              f"dropout={var['dropout']}, block={var['block_type']}, pool={var['pooling']}, SE={var['use_se']}")
        print(f"    Parameters: {num_params:,}")

        try:
            result = train_model(model, train_loader, val_loader, test_loader, device, epochs=60)

            good_classes = sum(1 for acc in result["class_accs"].values() if acc > 0.05)

            results.append({
                "name": name,
                "config": var,
                "params": num_params,
                "best_val_acc": result["best_val_acc"],
                "test_acc": result["test_acc"],
                "class_accs": {k: float(v) for k, v in result["class_accs"].items()},
                "good_classes": good_classes,
            })

            np.savez(output_dir / f"{name}_predictions.npz",
                     preds=result["preds"], labels=result["labels"])

            print(f"    Best Val: {result['best_val_acc']*100:.2f}%, Test: {result['test_acc']*100:.2f}%, Classes>5%: {good_classes}/7")

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"name": name, "config": var, "error": str(e)})

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - SORTED BY TEST ACCURACY")
    print("=" * 70)

    valid_results = [r for r in results if "error" not in r]
    valid_results.sort(key=lambda x: x["test_acc"], reverse=True)

    print(f"\n{'Rank':<5} {'Name':<25} {'Params':>10} {'Val':>8} {'Test':>8} {'Cls>5%':>8}")
    print("-" * 70)

    for i, r in enumerate(valid_results, 1):
        print(f"{i:<5} {r['name']:<25} {r['params']:>10,} {r['best_val_acc']*100:>7.2f}% {r['test_acc']*100:>7.2f}% {r['good_classes']:>6}/7")

    # Generate comparison plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Accuracy comparison
    ax1 = axes[0, 0]
    names = [r["name"] for r in valid_results]
    test_accs = [r["test_acc"] * 100 for r in valid_results]
    val_accs = [r["best_val_acc"] * 100 for r in valid_results]

    x = np.arange(len(names))
    width = 0.35
    ax1.bar(x - width/2, val_accs, width, label='Val Acc', color='steelblue', edgecolor='black')
    ax1.bar(x + width/2, test_accs, width, label='Test Acc', color='coral', edgecolor='black')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('DeepResidual Variations: Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=31.16, color='green', linestyle='--', alpha=0.7, label='Baseline')

    # Plot 2: Effect of hidden dimension
    ax2 = axes[0, 1]
    hidden_results = [r for r in valid_results if r["name"].startswith("h") and "_b4" in r["name"] and "d0" not in r["name"] and "pool" not in r["name"] and "SE" not in r["name"] and "bottleneck" not in r["name"] and "preact" not in r["name"]]
    if hidden_results:
        hidden_dims = [r["config"]["hidden"] for r in hidden_results]
        hidden_test = [r["test_acc"] * 100 for r in hidden_results]
        hidden_params = [r["params"] / 1000 for r in hidden_results]

        ax2_twin = ax2.twinx()
        bars = ax2.bar(range(len(hidden_dims)), hidden_test, color='coral', edgecolor='black', alpha=0.7)
        line = ax2_twin.plot(range(len(hidden_dims)), hidden_params, 'b-o', linewidth=2, markersize=8)

        ax2.set_xticks(range(len(hidden_dims)))
        ax2.set_xticklabels([str(h) for h in hidden_dims])
        ax2.set_xlabel('Hidden Dimension')
        ax2.set_ylabel('Test Accuracy (%)', color='coral')
        ax2_twin.set_ylabel('Parameters (K)', color='blue')
        ax2.set_title('Effect of Hidden Dimension')
        ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Effect of number of blocks
    ax3 = axes[1, 0]
    block_results = [r for r in valid_results if "h256_b" in r["name"] and "d0" not in r["name"] and "pool" not in r["name"] and "SE" not in r["name"] and "bottleneck" not in r["name"] and "preact" not in r["name"]]
    if block_results:
        block_results.sort(key=lambda x: x["config"]["num_blocks"])
        num_blocks = [r["config"]["num_blocks"] for r in block_results]
        block_test = [r["test_acc"] * 100 for r in block_results]

        ax3.plot(num_blocks, block_test, 'o-', linewidth=2, markersize=10, color='coral')
        ax3.set_xlabel('Number of Residual Blocks')
        ax3.set_ylabel('Test Accuracy (%)')
        ax3.set_title('Effect of Network Depth')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(num_blocks)

    # Plot 4: Per-class accuracy for top 3
    ax4 = axes[1, 1]
    top3 = valid_results[:3]
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    x = np.arange(len(classes))
    width = 0.25
    colors = ['coral', 'steelblue', 'forestgreen']

    for idx, r in enumerate(top3):
        accs = [r["class_accs"].get(cls, 0) * 100 for cls in classes]
        ax4.bar(x + idx * width, accs, width, label=r["name"], color=colors[idx], edgecolor='black')

    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Top 3 Variations: Per-Class Accuracy')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(classes, rotation=45, ha='right')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('DeepResidual Architecture Variations Comparison', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "variations_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(Path("comparison_plots") / "deep_residual_variations.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to: {output_dir}")
    print(f"Plot saved to: comparison_plots/deep_residual_variations.png")

    # Print best configuration
    best = valid_results[0]
    print(f"\n{'='*70}")
    print("BEST CONFIGURATION")
    print(f"{'='*70}")
    print(f"Name: {best['name']}")
    print(f"Test Accuracy: {best['test_acc']*100:.2f}%")
    print(f"Config: {best['config']}")


if __name__ == "__main__":
    main()
