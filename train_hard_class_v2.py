#!/usr/bin/env python3
"""
Training script to improve angry and surprise classification.
Uses existing data loading from train_bilstm.py but with targeted improvements.

Techniques:
1. Focal Loss with higher gamma for hard classes
2. Anti-neutral regularization
3. Higher sampling weight for angry/surprise
4. Class-specific data augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import from existing train_bilstm
from train_bilstm import (
    collect_samples, EnhancedPoseDataset, CLASSES, CLASS_TO_IDX,
    seed_all
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HARD_CLASSES = [0, 6]  # angry, surprise
NEUTRAL_IDX = 4
INPUT_DIM = 238
SEQ_LEN = 64
NUM_CLASSES = 8  # Including 'other'


class FocalLoss(nn.Module):
    """Focal loss with class-specific gamma."""
    def __init__(self, gamma_easy=1.0, gamma_hard=3.0, hard_classes=[0, 6], alpha=None):
        super().__init__()
        self.gamma_easy = gamma_easy
        self.gamma_hard = gamma_hard
        self.hard_classes = hard_classes
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)

        # Different gamma for hard vs easy classes
        gamma = torch.full_like(ce_loss, self.gamma_easy)
        for hc in self.hard_classes:
            gamma[targets == hc] = self.gamma_hard

        focal_loss = ((1 - pt) ** gamma) * ce_loss
        return focal_loss.mean()


class AntiNeutralLoss(nn.Module):
    """Penalize excessive neutral predictions for non-neutral samples."""
    def __init__(self, neutral_idx=4, weight=0.5):
        super().__init__()
        self.neutral_idx = neutral_idx
        self.weight = weight

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        neutral_prob = probs[:, self.neutral_idx]
        non_neutral_mask = (targets != self.neutral_idx).float()
        penalty = (neutral_prob * non_neutral_mask).mean()
        return self.weight * penalty


class PreActResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.relu(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + residual


class ImprovedDeepResidual(nn.Module):
    """DeepResidual with temporal attention for hard class improvement."""
    def __init__(self, input_dim=238, hidden_dim=256, num_blocks=4, num_classes=8, dropout=0.3):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.blocks = nn.ModuleList([
            PreActResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])

        # Multi-head attention for temporal modeling
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)

        # Attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        # Temporal self-attention
        x_attn, _ = self.temporal_attn(x, x, x)
        x = x + x_attn

        # Attention pooling
        attn_weights = self.attn_pool(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        features = (x * attn_weights).sum(dim=1)

        return self.classifier(features)


def train_epoch(model, loader, optimizer, criterion, anti_neutral=None, device=DEVICE):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        sequences, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        logits = model(sequences)

        loss = criterion(logits, labels)
        if anti_neutral is not None:
            loss += anti_neutral(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device=DEVICE):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            sequences, labels = batch[0].to(device), batch[1]
            logits = model(sequences)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()

    class_accs = {}
    for i, c in enumerate(CLASSES):
        mask = all_labels == i
        if mask.sum() > 0:
            class_accs[c] = (all_preds[mask] == i).mean()
        else:
            class_accs[c] = 0.0

    return accuracy, class_accs, all_preds, all_labels


def create_confusion_matrix(labels, preds, class_names):
    """Create confusion matrix without sklearn."""
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def main():
    print(f"Using device: {DEVICE}")
    seed_all(42)

    # Collect data
    data_dir = Path("./data_enhanced").resolve()
    items = collect_samples(data_dir)

    train_items = [it for it in items if it[1] == "train"]
    test_items = [it for it in items if it[1] == "test"]

    if len(test_items) == 0:
        random.shuffle(train_items)
        n = len(train_items)
        test_items = train_items[int(n * 0.9):]
        train_items = train_items[:int(n * 0.9)]

    print(f"Train: {len(train_items)}, Test: {len(test_items)}")

    # Print class distribution
    train_labels = [it[3] for it in train_items]
    label_counts = Counter(train_labels)
    print("\nClass distribution:")
    for i, c in enumerate(CLASSES):
        marker = " ***" if i in HARD_CLASSES else ""
        print(f"  {c}: {label_counts.get(i, 0)}{marker}")

    # Create datasets
    train_ds = EnhancedPoseDataset(
        train_items, seq_len=SEQ_LEN, num_people=2,
        train=True, augment=True, aug_prob=0.5
    )
    test_ds = EnhancedPoseDataset(
        test_items, seq_len=SEQ_LEN, num_people=2,
        train=False, augment=False
    )

    # Create sampler with higher weight for hard classes
    sample_weights = []
    for it in train_items:
        label = it[3]
        if label in HARD_CLASSES:
            sample_weights.append(4.0 / label_counts[label])  # 4x weight
        else:
            sample_weights.append(1.0 / label_counts[label])

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    # Experiments
    experiments = [
        {
            'name': 'baseline',
            'gamma_easy': 0.0,
            'gamma_hard': 0.0,
            'anti_neutral_weight': 0.0,
            'hard_class_weight': 1.0,
        },
        {
            'name': 'focal_loss',
            'gamma_easy': 1.0,
            'gamma_hard': 2.0,
            'anti_neutral_weight': 0.0,
            'hard_class_weight': 1.0,
        },
        {
            'name': 'class_focal',
            'gamma_easy': 1.0,
            'gamma_hard': 3.0,
            'anti_neutral_weight': 0.0,
            'hard_class_weight': 1.0,
        },
        {
            'name': 'anti_neutral',
            'gamma_easy': 1.0,
            'gamma_hard': 3.0,
            'anti_neutral_weight': 0.5,
            'hard_class_weight': 1.0,
        },
        {
            'name': 'hard_weights',
            'gamma_easy': 1.0,
            'gamma_hard': 3.0,
            'anti_neutral_weight': 0.3,
            'hard_class_weight': 2.0,
        },
        {
            'name': 'aggressive',
            'gamma_easy': 0.5,
            'gamma_hard': 4.0,
            'anti_neutral_weight': 0.7,
            'hard_class_weight': 3.0,
        },
        {
            'name': 'combined_best',
            'gamma_easy': 1.0,
            'gamma_hard': 3.5,
            'anti_neutral_weight': 0.5,
            'hard_class_weight': 2.5,
        },
    ]

    results = []

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {exp['name']}")
        print(f"{'='*60}")

        # Create model
        model = ImprovedDeepResidual(
            input_dim=INPUT_DIM, hidden_dim=256, num_blocks=4, dropout=0.3
        ).to(DEVICE)

        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")

        # Class weights
        class_weights = torch.ones(NUM_CLASSES, device=DEVICE)
        class_weights[0] = exp['hard_class_weight']  # angry
        class_weights[6] = exp['hard_class_weight']  # surprise
        class_weights[7] = 0.0  # 'other' class - ignore

        # Loss function
        if exp['gamma_easy'] > 0 or exp['gamma_hard'] > 0:
            criterion = FocalLoss(
                gamma_easy=exp['gamma_easy'],
                gamma_hard=exp['gamma_hard'],
                hard_classes=HARD_CLASSES,
                alpha=class_weights
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Anti-neutral loss
        anti_neutral = None
        if exp['anti_neutral_weight'] > 0:
            anti_neutral = AntiNeutralLoss(weight=exp['anti_neutral_weight'])

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

        best_test_acc = 0
        best_class_accs = None
        best_preds = None
        best_labels = None

        for epoch in range(60):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, anti_neutral)
            test_acc, class_accs, preds, labels = evaluate(model, test_loader)
            scheduler.step()

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_class_accs = class_accs.copy()
                best_preds = preds.copy()
                best_labels = labels.copy()

            if (epoch + 1) % 20 == 0:
                angry_acc = class_accs['angry'] * 100
                surprise_acc = class_accs['surprise'] * 100
                print(f"  Epoch {epoch+1}: train={train_acc:.3f}, test={test_acc:.3f}, "
                      f"angry={angry_acc:.1f}%, surprise={surprise_acc:.1f}%")

        print(f"\nBest Test: {best_test_acc*100:.2f}%")
        print(f"  Angry: {best_class_accs['angry']*100:.1f}%")
        print(f"  Surprise: {best_class_accs['surprise']*100:.1f}%")

        results.append({
            'name': exp['name'],
            'config': exp,
            'test_acc': best_test_acc,
            'class_accs': best_class_accs,
            'preds': best_preds,
            'labels': best_labels
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - HARD CLASS IMPROVEMENT")
    print("="*70)
    print(f"\n{'Experiment':<20} {'Test':>8} {'Angry':>8} {'Surprise':>10} {'Combined':>10}")
    print("-"*70)

    for r in results:
        angry = r['class_accs']['angry'] * 100
        surprise = r['class_accs']['surprise'] * 100
        combined = angry + surprise
        print(f"{r['name']:<20} {r['test_acc']*100:>7.2f}% {angry:>7.1f}% {surprise:>9.1f}% {combined:>9.1f}%")

    # Find best
    best_angry = max(results, key=lambda x: x['class_accs']['angry'])
    best_surprise = max(results, key=lambda x: x['class_accs']['surprise'])
    best_combined = max(results, key=lambda x: x['class_accs']['angry'] + x['class_accs']['surprise'])

    print(f"\nBest Angry: {best_angry['name']} ({best_angry['class_accs']['angry']*100:.1f}%)")
    print(f"Best Surprise: {best_surprise['name']} ({best_surprise['class_accs']['surprise']*100:.1f}%)")
    print(f"Best Combined: {best_combined['name']}")

    # Compare with previous best
    print("\n" + "="*70)
    print("IMPROVEMENT OVER PREVIOUS BEST (33.97% overall)")
    print("="*70)
    print(f"Previous: Angry=1.0%, Surprise=17.7%")
    print(f"Best new: Angry={best_angry['class_accs']['angry']*100:.1f}%, Surprise={best_surprise['class_accs']['surprise']*100:.1f}%")
    print(f"Angry improvement: +{best_angry['class_accs']['angry']*100 - 1.0:.1f}%")
    print(f"Surprise improvement: +{best_surprise['class_accs']['surprise']*100 - 17.7:.1f}%")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = f'runs/hard_class_v2_{timestamp}'
    os.makedirs(run_dir, exist_ok=True)

    save_results = []
    for r in results:
        save_results.append({
            'name': r['name'],
            'test_acc': float(r['test_acc']),
            'class_accs': {k: float(v) for k, v in r['class_accs'].items()},
            'config': r['config']
        })

    with open(f'{run_dir}/results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Overall accuracy
    ax1 = axes[0, 0]
    names = [r['name'] for r in results]
    accs = [r['test_acc'] * 100 for r in results]
    colors = ['green' if r['name'] == best_combined['name'] else 'steelblue' for r in results]
    ax1.bar(range(len(names)), accs, color=colors)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Overall Test Accuracy')
    ax1.axhline(y=33.97, color='red', linestyle='--', label='Previous best (33.97%)')
    ax1.legend()

    # Plot 2: Hard class accuracy
    ax2 = axes[0, 1]
    x = np.arange(len(names))
    width = 0.35
    angry_accs = [r['class_accs']['angry'] * 100 for r in results]
    surprise_accs = [r['class_accs']['surprise'] * 100 for r in results]
    ax2.bar(x - width/2, angry_accs, width, label='Angry', color='red', alpha=0.7)
    ax2.bar(x + width/2, surprise_accs, width, label='Surprise', color='orange', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Hard Class Accuracy')
    ax2.axhline(y=1.0, color='red', linestyle=':', alpha=0.5, label='Prev Angry (1%)')
    ax2.axhline(y=17.7, color='orange', linestyle=':', alpha=0.5, label='Prev Surprise (17.7%)')
    ax2.legend()

    # Plot 3: Per-class for best combined
    ax3 = axes[1, 0]
    best_r = best_combined
    class_names = list(best_r['class_accs'].keys())
    class_vals = [best_r['class_accs'][c] * 100 for c in class_names]
    colors = ['red' if c in ['angry', 'surprise'] else 'steelblue' for c in class_names]
    ax3.bar(class_names, class_vals, color=colors)
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title(f"Per-Class: {best_r['name']}")
    ax3.axhline(y=14.29, color='gray', linestyle='--', label='Random')
    for i, v in enumerate(class_vals):
        ax3.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

    # Plot 4: Confusion matrix for best
    ax4 = axes[1, 1]
    cm = create_confusion_matrix(best_r['labels'], best_r['preds'], CLASSES)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    im = ax4.imshow(cm_norm, cmap='Blues')
    ax4.set_xticks(range(len(CLASSES)))
    ax4.set_yticks(range(len(CLASSES)))
    ax4.set_xticklabels([c[:3] for c in CLASSES])
    ax4.set_yticklabels([c[:3] for c in CLASSES])
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('True')
    ax4.set_title(f"Confusion Matrix: {best_r['name']}")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax4.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center',
                    color='white' if cm_norm[i,j] > 0.5 else 'black', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{run_dir}/hard_class_improvements.png', dpi=150, bbox_inches='tight')
    plt.savefig('comparison_plots/hard_class_improvements.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {run_dir}/hard_class_improvements.png")

    # Save best predictions
    np.savez(f'{run_dir}/best_predictions.npz',
             preds=best_combined['preds'],
             labels=best_combined['labels'])

    print(f"Results saved to: {run_dir}/")

    return results


if __name__ == '__main__':
    results = main()
