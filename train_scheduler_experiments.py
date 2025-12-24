#!/usr/bin/env python3
"""
Experiment with different learning rate schedulers and learning rates.
Based on the best model so far (prototypical networks / simple BiLSTM).
"""

import os
import sys
import json
import math
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Schedulers
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
    ExponentialLR,
    OneCycleLR,
    ReduceLROnPlateau,
    LambdaLR,
)

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PoseDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int = 64, max_samples_per_class: int = 400):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.samples = []
        self.labels = []

        class_counts = {c: 0 for c in CLASSES}

        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = self.data_dir / cls_name
            if not cls_dir.exists():
                continue

            files = list(cls_dir.glob("*.npz"))
            random.shuffle(files)
            if max_samples_per_class > 0:
                files = files[:max_samples_per_class]

            for f in files:
                self.samples.append(str(f))
                self.labels.append(cls_idx)
                class_counts[cls_name] += 1

        print(f"Loaded {len(self.samples)} samples")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]

        try:
            with np.load(path, allow_pickle=False) as z:
                kp = z["keypoints"]  # (T, P, 17, 3)
                vel = z["vel"]       # (T, P, 17, 2)
                acc = z["acc"]       # (T, P, 17, 2)
        except Exception:
            # Return zeros on error
            return torch.zeros(self.seq_len, 238), torch.tensor(label, dtype=torch.long)

        T, P = kp.shape[:2]
        P = min(P, 2)  # Max 2 people

        # Combine features: pos(3) + vel(2) + acc(2) = 7 per joint
        features = np.concatenate([
            kp[:, :P].reshape(T, P, -1),
            vel[:, :P].reshape(T, P, -1),
            acc[:, :P].reshape(T, P, -1)
        ], axis=-1)  # (T, P, 17*7)

        # Pad people dimension if needed
        if P < 2:
            pad_p = np.zeros((T, 2 - P, features.shape[-1]), dtype=np.float32)
            features = np.concatenate([features, pad_p], axis=1)

        # Flatten people dimension
        features = features.reshape(T, -1)  # (T, 2*17*7 = 238)

        # Replace NaN/Inf values with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Pad/truncate to seq_len
        if T < self.seq_len:
            pad = np.zeros((self.seq_len - T, features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, pad], axis=0)
        else:
            # Random crop during training, center crop otherwise
            start = random.randint(0, T - self.seq_len)
            features = features[start:start + self.seq_len]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class SimpleBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1,
                 num_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


def get_warmup_scheduler(optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
    """Linear warmup followed by cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def get_warmup_linear_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup followed by linear decay to 0."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return max(0.0, 1.0 - (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
    return LambdaLR(optimizer, lr_lambda)


def get_scheduler(scheduler_name: str, optimizer, epochs: int, steps_per_epoch: int, lr: float):
    """Get scheduler by name."""
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    elif scheduler_name == "cosine_warm_restarts":
        return CosineAnnealingWarmRestarts(optimizer, T_0=epochs // 4, T_mult=2, eta_min=lr * 0.01)
    elif scheduler_name == "step":
        return StepLR(optimizer, step_size=epochs // 3, gamma=0.5)
    elif scheduler_name == "exponential":
        return ExponentialLR(optimizer, gamma=0.98)
    elif scheduler_name == "one_cycle":
        return OneCycleLR(optimizer, max_lr=lr * 10, epochs=epochs, steps_per_epoch=steps_per_epoch)
    elif scheduler_name == "warmup_cosine":
        return get_warmup_scheduler(optimizer, warmup_epochs=epochs // 10, total_epochs=epochs, base_lr=lr)
    elif scheduler_name == "warmup_linear":
        return get_warmup_linear_scheduler(optimizer, warmup_epochs=epochs // 10, total_epochs=epochs)
    elif scheduler_name == "plateau":
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    elif scheduler_name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def train_one_epoch(model, loader, optimizer, scheduler, device, criterion, is_one_cycle=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if is_one_cycle and scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def get_class_accuracies(preds, labels, classes):
    """Calculate per-class accuracy."""
    class_acc = {}
    for i, cls in enumerate(classes):
        mask = labels == i
        if mask.sum() > 0:
            class_acc[cls] = float((preds[mask] == i).sum() / mask.sum())
        else:
            class_acc[cls] = 0.0
    return class_acc


def run_experiment(
    data_dir: str,
    scheduler_name: str,
    lr: float,
    epochs: int = 100,
    hidden_dim: int = 128,
    num_layers: int = 1,
    dropout: float = 0.3,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    seed: int = 42,
    device: str = "cuda"
):
    """Run a single experiment with given scheduler and learning rate."""
    set_seed(seed)

    # Load data
    dataset = PoseDataset(data_dir, seq_len=64)

    # Split data
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    n = len(indices)
    train_idx = indices[:int(0.7 * n)]
    val_idx = indices[int(0.7 * n):int(0.85 * n)]
    test_idx = indices[int(0.85 * n):]

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)

    # Weighted sampler for balanced training
    train_labels = [dataset.labels[i] for i in train_idx]
    class_counts = np.bincount(train_labels, minlength=len(CLASSES))
    class_weights = 1.0 / (class_counts + 1)
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    input_dim = 238  # 17 joints * 7 features * 2 people
    model = SimpleBiLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=len(CLASSES),
        dropout=dropout
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    is_one_cycle = scheduler_name == "one_cycle"
    is_plateau = scheduler_name == "plateau"
    scheduler = get_scheduler(scheduler_name, optimizer, epochs, len(train_loader), lr)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Training
    best_val_acc = 0
    best_model_state = None
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "test_acc": [], "lr": []
    }

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler if is_one_cycle else None,
            device, criterion, is_one_cycle
        )

        # Evaluate
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device, criterion)
        _, test_acc, _, _ = evaluate(model, test_loader, device, criterion)

        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        history["lr"].append(current_lr)

        # Best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # Step scheduler
        if scheduler is not None and not is_one_cycle:
            if is_plateau:
                scheduler.step(val_acc)
            else:
                scheduler.step()

        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, lr={current_lr:.6f}")

    # Load best model and evaluate
    model.load_state_dict(best_model_state)
    _, final_val_acc, val_preds, val_labels = evaluate(model, val_loader, device, criterion)
    _, final_test_acc, test_preds, test_labels = evaluate(model, test_loader, device, criterion)

    # Per-class accuracy
    class_acc = get_class_accuracies(test_preds, test_labels, CLASSES)
    n_classes_above_5 = sum(1 for v in class_acc.values() if v > 0.05)

    return {
        "scheduler": scheduler_name,
        "lr": lr,
        "best_val_acc": best_val_acc,
        "final_test_acc": final_test_acc,
        "class_acc": class_acc,
        "n_classes_above_5": n_classes_above_5,
        "history": history
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data_enhanced")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Experiments to run - reduced set for faster testing
    schedulers = [
        "none",
        "cosine",
        "one_cycle",
        "warmup_cosine",
        "plateau"
    ]

    learning_rates = [1e-4, 5e-4, 1e-3]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"runs/scheduler_experiments_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Run experiments
    print("\n" + "=" * 60)
    print("SCHEDULER AND LEARNING RATE EXPERIMENTS")
    print("=" * 60)

    # First, test all schedulers with a fixed learning rate
    print("\n--- Testing Schedulers (LR=5e-4) ---\n")
    for sched in schedulers:
        print(f"\nScheduler: {sched}")
        try:
            result = run_experiment(
                data_dir=args.data_dir,
                scheduler_name=sched,
                lr=5e-4,
                epochs=args.epochs,
                seed=args.seed,
                device=device
            )
            all_results.append(result)
            print(f"  Best Val: {result['best_val_acc']*100:.2f}%, Test: {result['final_test_acc']*100:.2f}%, Classes: {result['n_classes_above_5']}")
        except Exception as e:
            print(f"  Error: {e}")

    # Then, test learning rates with the best scheduler
    best_scheduler = max(all_results, key=lambda x: x['best_val_acc'])['scheduler']
    print(f"\n--- Testing Learning Rates (Best Scheduler: {best_scheduler}) ---\n")

    for lr in learning_rates:
        print(f"\nLR: {lr}")
        try:
            result = run_experiment(
                data_dir=args.data_dir,
                scheduler_name=best_scheduler,
                lr=lr,
                epochs=args.epochs,
                seed=args.seed,
                device=device
            )
            all_results.append(result)
            print(f"  Best Val: {result['best_val_acc']*100:.2f}%, Test: {result['final_test_acc']*100:.2f}%, Classes: {result['n_classes_above_5']}")
        except Exception as e:
            print(f"  Error: {e}")

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        # Convert numpy values to Python native types
        serializable_results = []
        for r in all_results:
            sr = {k: v for k, v in r.items() if k != 'history'}
            sr['best_val_acc'] = float(sr['best_val_acc'])
            sr['final_test_acc'] = float(sr['final_test_acc'])
            sr['class_acc'] = {k: float(v) for k, v in sr['class_acc'].items()}
            serializable_results.append(sr)
        json.dump(serializable_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Scheduler':<25} {'LR':<10} {'Val Acc':<10} {'Test Acc':<10} {'Classes':<8}")
    print("-" * 63)

    sorted_results = sorted(all_results, key=lambda x: x['best_val_acc'], reverse=True)
    for r in sorted_results:
        print(f"{r['scheduler']:<25} {r['lr']:<10.0e} {r['best_val_acc']*100:<10.2f} {r['final_test_acc']*100:<10.2f} {r['n_classes_above_5']:<8}")

    # Generate comparison plot
    print("\nGenerating comparison plots...")
    generate_comparison_plots(all_results, output_dir)

    print(f"\nResults saved to: {output_dir}")


def generate_comparison_plots(results, output_dir):
    """Generate comparison plots for all experiments."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Separate scheduler and LR experiments
    scheduler_results = [r for r in results if r['lr'] == 5e-4]
    lr_results = [r for r in results if r['scheduler'] == results[0]['scheduler'] and r['lr'] != 5e-4] if results else []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Scheduler comparison
    if scheduler_results:
        names = [r['scheduler'] for r in scheduler_results]
        val_accs = [r['best_val_acc'] * 100 for r in scheduler_results]
        test_accs = [r['final_test_acc'] * 100 for r in scheduler_results]

        x = np.arange(len(names))
        width = 0.35

        axes[0, 0].bar(x - width/2, val_accs, width, label='Val', color='steelblue')
        axes[0, 0].bar(x + width/2, test_accs, width, label='Test', color='coral')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Scheduler Comparison (LR=5e-4)')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].axhline(y=14.3, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Classes detected per scheduler
    if scheduler_results:
        names = [r['scheduler'] for r in scheduler_results]
        n_classes = [r['n_classes_above_5'] for r in scheduler_results]
        colors = ['green' if n >= 6 else 'orange' if n >= 4 else 'red' for n in n_classes]

        axes[0, 1].bar(names, n_classes, color=colors, edgecolor='black')
        axes[0, 1].set_ylabel('Number of Classes')
        axes[0, 1].set_title('Classes with >5% Accuracy')
        axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        axes[0, 1].set_ylim(0, 8)
        axes[0, 1].axhline(y=7, color='green', linestyle='--', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Learning rate curves (if we have history)
    if scheduler_results and 'history' in scheduler_results[0]:
        for r in scheduler_results[:5]:  # Top 5
            if 'history' in r and 'lr' in r['history']:
                axes[1, 0].plot(r['history']['lr'], label=r['scheduler'], linewidth=1.5)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedules')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')

    # Plot 4: Val accuracy over epochs
    if scheduler_results and 'history' in scheduler_results[0]:
        for r in scheduler_results[:5]:  # Top 5
            if 'history' in r and 'val_acc' in r['history']:
                axes[1, 1].plot(r['history']['val_acc'], label=r['scheduler'], linewidth=1.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].set_title('Validation Accuracy Over Training')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "scheduler_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # LR comparison plot
    if lr_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        lrs = [r['lr'] for r in lr_results]
        val_accs = [r['best_val_acc'] * 100 for r in lr_results]
        test_accs = [r['final_test_acc'] * 100 for r in lr_results]

        ax.plot(lrs, val_accs, 'o-', label='Val', color='steelblue', linewidth=2, markersize=8)
        ax.plot(lrs, test_accs, 's-', label='Test', color='coral', linewidth=2, markersize=8)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Learning Rate Comparison (Scheduler: {lr_results[0]["scheduler"]})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "lr_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
