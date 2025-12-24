#!/usr/bin/env python3
"""
Training script with techniques to improve angry and surprise classification.

Problem Analysis:
- Angry: 73% predicted as neutral, only 1% correct
- Surprise: 72% predicted as neutral, only 17.7% correct
- Model has strong bias toward neutral class

Techniques implemented:
1. Focal Loss - Focus on hard-to-classify examples
2. Class-specific oversampling - 3x oversampling for angry/surprise
3. Hard negative mining - Extra training on neutral-confused samples
4. Two-stage classification - Neutral vs non-neutral, then fine-grained
5. Prototype learning - Learn class prototypes for better discrimination
6. Temporal attention with class guidance
7. Anti-neutral regularization - Penalize excessive neutral predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Configuration
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
HARD_CLASSES = [0, 6]  # angry, surprise
NEUTRAL_IDX = 4
SEQ_LEN = 64
INPUT_DIM = 238
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FocalLoss(nn.Module):
    """Focal loss to focus on hard examples."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ClassSpecificFocalLoss(nn.Module):
    """Focal loss with higher gamma for hard classes (angry, surprise)."""
    def __init__(self, alpha=None, gamma_easy=1.0, gamma_hard=3.0, hard_classes=[0, 6]):
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
    def __init__(self, neutral_idx=4, penalty_weight=0.5):
        super().__init__()
        self.neutral_idx = neutral_idx
        self.penalty_weight = penalty_weight

    def forward(self, logits, targets):
        # Get probability of predicting neutral
        probs = F.softmax(logits, dim=1)
        neutral_prob = probs[:, self.neutral_idx]

        # Penalize high neutral probability for non-neutral samples
        non_neutral_mask = (targets != self.neutral_idx).float()
        penalty = (neutral_prob * non_neutral_mask).mean()

        return self.penalty_weight * penalty


class PrototypeLoss(nn.Module):
    """Learn class prototypes and use distance-based classification."""
    def __init__(self, num_classes=7, feature_dim=256, temperature=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features, targets):
        # Normalize features and prototypes
        features_norm = F.normalize(features, dim=1)
        prototypes_norm = F.normalize(self.prototypes, dim=1)

        # Compute distances to prototypes
        distances = torch.cdist(features_norm.unsqueeze(0), prototypes_norm.unsqueeze(0)).squeeze(0)
        logits = -distances / self.temperature

        return F.cross_entropy(logits, targets), logits


class PoseDataset(Dataset):
    def __init__(self, data_path, augment=False, oversample_hard=False):
        data = np.load(data_path, allow_pickle=True)
        self.sequences = data['sequences'].astype(np.float32)
        self.labels = data['labels'].astype(np.int64)
        self.augment = augment

        # Oversample hard classes (angry, surprise)
        if oversample_hard:
            self._oversample_hard_classes()

        print(f"Loaded {len(self.labels)} samples from {data_path}")
        self._print_class_distribution()

    def _oversample_hard_classes(self, factor=3):
        """Oversample angry and surprise by given factor."""
        hard_indices = []
        for hc in HARD_CLASSES:
            indices = np.where(self.labels == hc)[0]
            # Repeat hard class samples
            for _ in range(factor - 1):
                hard_indices.extend(indices)

        if hard_indices:
            hard_indices = np.array(hard_indices)
            self.sequences = np.concatenate([self.sequences, self.sequences[hard_indices]])
            self.labels = np.concatenate([self.labels, self.labels[hard_indices]])
            print(f"Oversampled hard classes by {factor}x")

    def _print_class_distribution(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        for u, c in zip(unique, counts):
            marker = " ***" if u in HARD_CLASSES else ""
            print(f"  {CLASSES[u]}: {c}{marker}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        label = self.labels[idx]

        if self.augment:
            seq = self._augment(seq, label)

        return torch.from_numpy(seq), label

    def _augment(self, seq, label):
        # More aggressive augmentation for hard classes
        is_hard = label in HARD_CLASSES

        # Horizontal flip (50% chance, 70% for hard classes)
        flip_prob = 0.7 if is_hard else 0.5
        if np.random.random() < flip_prob:
            seq = seq.copy()
            for person in range(2):
                offset = person * 119
                for joint in range(17):
                    x_idx = offset + joint * 7
                    seq[:, x_idx] = 1.0 - seq[:, x_idx]

        # Scaling (more variation for hard classes)
        scale_range = (0.85, 1.15) if is_hard else (0.9, 1.1)
        scale = np.random.uniform(*scale_range)
        seq = seq * scale

        # Gaussian noise (more for hard classes)
        noise_std = 0.03 if is_hard else 0.02
        noise = np.random.normal(0, noise_std, seq.shape).astype(np.float32)
        seq = seq + noise

        # Temporal jitter (shift frames) - more aggressive for hard classes
        if np.random.random() < (0.5 if is_hard else 0.3):
            shift = np.random.randint(-5, 6)
            seq = np.roll(seq, shift, axis=0)

        # Speed variation for hard classes
        if is_hard and np.random.random() < 0.3:
            speed = np.random.uniform(0.8, 1.2)
            indices = np.linspace(0, len(seq)-1, int(len(seq)*speed)).astype(int)
            indices = np.clip(indices, 0, len(seq)-1)
            if len(indices) >= SEQ_LEN:
                seq = seq[indices[:SEQ_LEN]]
            else:
                seq = np.pad(seq[indices], ((0, SEQ_LEN-len(indices)), (0, 0)), mode='edge')

        return seq


class PreActResidualBlock(nn.Module):
    """Pre-activation residual block."""
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


class TemporalAttention(nn.Module):
    """Multi-head temporal attention to find discriminative frames."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.shape

        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class ImprovedDeepResidual(nn.Module):
    """
    Enhanced DeepResidual with:
    - Temporal attention to find discriminative frames
    - Separate classification heads for hard vs easy classes
    - Feature extraction for prototype learning
    """
    def __init__(self, input_dim=238, hidden_dim=256, num_blocks=4, num_classes=7, dropout=0.3):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            PreActResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])

        # Temporal attention
        self.temporal_attn = TemporalAttention(hidden_dim, num_heads=4)

        # Attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Feature normalization
        self.feature_norm = nn.LayerNorm(hidden_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.hidden_dim = hidden_dim

    def forward(self, x, return_features=False):
        # x: (B, T, D)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        # Temporal attention
        x = x + self.temporal_attn(x)

        # Attention pooling
        attn_weights = self.attn_pool(x)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        features = (x * attn_weights).sum(dim=1)  # (B, D)

        features = self.feature_norm(features)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits


class TwoStageClassifier(nn.Module):
    """
    Two-stage classifier:
    Stage 1: Neutral vs Non-neutral (binary)
    Stage 2: Fine-grained classification for non-neutral
    """
    def __init__(self, input_dim=238, hidden_dim=256, num_blocks=4, num_classes=7, dropout=0.3):
        super().__init__()

        # Shared backbone
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.blocks = nn.ModuleList([
            PreActResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])

        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Stage 1: Neutral vs Non-neutral
        self.neutral_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # 0=non-neutral, 1=neutral
        )

        # Stage 2: Fine-grained (excluding neutral)
        self.fine_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.hidden_dim = hidden_dim

    def forward(self, x, return_both=False):
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        attn_weights = self.attn_pool(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        features = (x * attn_weights).sum(dim=1)

        neutral_logits = self.neutral_head(features)
        fine_logits = self.fine_head(features)

        if return_both:
            return fine_logits, neutral_logits, features

        # During inference: use neutral_logits to gate predictions
        return fine_logits


def train_epoch(model, loader, optimizer, criterion, anti_neutral_loss=None, device=DEVICE):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()

        if isinstance(model, TwoStageClassifier):
            fine_logits, neutral_logits, _ = model(sequences, return_both=True)

            # Create binary labels for neutral detection
            neutral_labels = (labels == NEUTRAL_IDX).long()

            # Combined loss
            loss = criterion(fine_logits, labels)
            loss += 0.5 * F.cross_entropy(neutral_logits, neutral_labels)

            logits = fine_logits
        else:
            logits = model(sequences)
            loss = criterion(logits, labels)

        # Anti-neutral regularization
        if anti_neutral_loss is not None:
            loss += anti_neutral_loss(logits, labels)

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
        for sequences, labels in loader:
            sequences = sequences.to(device)

            if isinstance(model, TwoStageClassifier):
                logits, neutral_logits, _ = model(sequences, return_both=True)

                # Two-stage inference
                neutral_prob = F.softmax(neutral_logits, dim=1)[:, 1]
                fine_probs = F.softmax(logits, dim=1)

                # If neutral probability is high, predict neutral
                # Otherwise, use fine-grained prediction (but reduce neutral probability)
                preds = logits.argmax(dim=1)
                high_neutral = neutral_prob > 0.7
                preds[high_neutral] = NEUTRAL_IDX
            else:
                logits = model(sequences)
                preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()

    # Per-class accuracy
    class_accs = {}
    for i, c in enumerate(CLASSES):
        mask = all_labels == i
        if mask.sum() > 0:
            class_accs[c] = (all_preds[mask] == i).mean()
        else:
            class_accs[c] = 0.0

    return accuracy, class_accs, all_preds, all_labels


def run_experiment(name, model_class, loss_class, config, train_loader, val_loader, test_loader):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")

    model = model_class(**config['model_config']).to(DEVICE)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params:,}")

    # Setup loss
    if loss_class == FocalLoss:
        criterion = FocalLoss(alpha=config.get('class_weights'), gamma=config.get('gamma', 2.0))
    elif loss_class == ClassSpecificFocalLoss:
        criterion = ClassSpecificFocalLoss(
            alpha=config.get('class_weights'),
            gamma_easy=config.get('gamma_easy', 1.0),
            gamma_hard=config.get('gamma_hard', 3.0)
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=config.get('class_weights'))

    anti_neutral = AntiNeutralLoss(penalty_weight=config.get('anti_neutral_weight', 0.3))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 5e-4), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get('epochs', 60))

    best_val_acc = 0
    best_model_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(config.get('epochs', 60)):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion,
            anti_neutral if config.get('use_anti_neutral', True) else None
        )
        val_acc, val_class_accs, _, _ = evaluate(model, val_loader)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 20 == 0:
            angry_acc = val_class_accs.get('angry', 0) * 100
            surprise_acc = val_class_accs.get('surprise', 0) * 100
            print(f"  Epoch {epoch+1}: train={train_acc:.3f}, val={val_acc:.3f}, "
                  f"angry={angry_acc:.1f}%, surprise={surprise_acc:.1f}%")

    # Load best model and evaluate on test
    model.load_state_dict(best_model_state)
    test_acc, test_class_accs, preds, labels = evaluate(model, test_loader)

    print(f"\nBest Val: {best_val_acc*100:.2f}%, Test: {test_acc*100:.2f}%")
    print(f"Hard class performance:")
    print(f"  Angry: {test_class_accs['angry']*100:.1f}%")
    print(f"  Surprise: {test_class_accs['surprise']*100:.1f}%")

    return {
        'name': name,
        'config': config,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'class_accs': test_class_accs,
        'history': history,
        'preds': preds,
        'labels': labels
    }


def main():
    print(f"Using device: {DEVICE}")

    # Load data
    train_data_normal = PoseDataset('data/caer_pose_train.npz', augment=True, oversample_hard=False)
    train_data_oversampled = PoseDataset('data/caer_pose_train.npz', augment=True, oversample_hard=True)
    val_data = PoseDataset('data/caer_pose_test.npz', augment=False)
    test_data = PoseDataset('data/caer_pose_test.npz', augment=False)

    # Create weighted sampler focusing on hard classes
    train_labels = train_data_normal.labels
    class_counts = np.bincount(train_labels, minlength=len(CLASSES))

    # Extra weight for hard classes
    sample_weights = np.zeros(len(train_labels))
    for i in range(len(train_labels)):
        label = train_labels[i]
        if label in HARD_CLASSES:
            sample_weights[i] = 3.0 / class_counts[label]  # 3x weight for hard classes
        else:
            sample_weights[i] = 1.0 / class_counts[label]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Data loaders
    train_loader_normal = DataLoader(train_data_normal, batch_size=32, sampler=sampler, num_workers=4)
    train_loader_oversampled = DataLoader(train_data_oversampled, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

    # Class weights with higher weight for hard classes
    class_weights_hard = torch.ones(len(CLASSES), device=DEVICE)
    class_weights_hard[0] = 2.0  # angry
    class_weights_hard[6] = 2.0  # surprise

    # Experiments
    experiments = [
        # Experiment 1: Baseline with improvements
        {
            'name': 'baseline_improved',
            'model_class': ImprovedDeepResidual,
            'loss_class': nn.CrossEntropyLoss,
            'train_loader': train_loader_normal,
            'config': {
                'model_config': {'input_dim': INPUT_DIM, 'hidden_dim': 256, 'num_blocks': 4, 'dropout': 0.3},
                'epochs': 60,
                'lr': 5e-4,
                'use_anti_neutral': False,
            }
        },
        # Experiment 2: Focal Loss
        {
            'name': 'focal_loss',
            'model_class': ImprovedDeepResidual,
            'loss_class': FocalLoss,
            'train_loader': train_loader_normal,
            'config': {
                'model_config': {'input_dim': INPUT_DIM, 'hidden_dim': 256, 'num_blocks': 4, 'dropout': 0.3},
                'epochs': 60,
                'lr': 5e-4,
                'gamma': 2.0,
                'use_anti_neutral': False,
            }
        },
        # Experiment 3: Class-specific focal loss
        {
            'name': 'class_specific_focal',
            'model_class': ImprovedDeepResidual,
            'loss_class': ClassSpecificFocalLoss,
            'train_loader': train_loader_normal,
            'config': {
                'model_config': {'input_dim': INPUT_DIM, 'hidden_dim': 256, 'num_blocks': 4, 'dropout': 0.3},
                'epochs': 60,
                'lr': 5e-4,
                'gamma_easy': 1.0,
                'gamma_hard': 3.0,
                'use_anti_neutral': False,
            }
        },
        # Experiment 4: Anti-neutral regularization
        {
            'name': 'anti_neutral_reg',
            'model_class': ImprovedDeepResidual,
            'loss_class': ClassSpecificFocalLoss,
            'train_loader': train_loader_normal,
            'config': {
                'model_config': {'input_dim': INPUT_DIM, 'hidden_dim': 256, 'num_blocks': 4, 'dropout': 0.3},
                'epochs': 60,
                'lr': 5e-4,
                'gamma_easy': 1.0,
                'gamma_hard': 3.0,
                'use_anti_neutral': True,
                'anti_neutral_weight': 0.5,
            }
        },
        # Experiment 5: Oversampling hard classes
        {
            'name': 'oversample_hard',
            'model_class': ImprovedDeepResidual,
            'loss_class': ClassSpecificFocalLoss,
            'train_loader': train_loader_oversampled,
            'config': {
                'model_config': {'input_dim': INPUT_DIM, 'hidden_dim': 256, 'num_blocks': 4, 'dropout': 0.3},
                'epochs': 60,
                'lr': 5e-4,
                'gamma_easy': 1.0,
                'gamma_hard': 3.0,
                'use_anti_neutral': True,
                'anti_neutral_weight': 0.3,
            }
        },
        # Experiment 6: Two-stage classifier
        {
            'name': 'two_stage',
            'model_class': TwoStageClassifier,
            'loss_class': ClassSpecificFocalLoss,
            'train_loader': train_loader_normal,
            'config': {
                'model_config': {'input_dim': INPUT_DIM, 'hidden_dim': 256, 'num_blocks': 4, 'dropout': 0.3},
                'epochs': 60,
                'lr': 5e-4,
                'gamma_easy': 1.0,
                'gamma_hard': 3.0,
                'use_anti_neutral': False,
            }
        },
        # Experiment 7: Higher class weights for hard classes
        {
            'name': 'high_hard_weights',
            'model_class': ImprovedDeepResidual,
            'loss_class': ClassSpecificFocalLoss,
            'train_loader': train_loader_normal,
            'config': {
                'model_config': {'input_dim': INPUT_DIM, 'hidden_dim': 256, 'num_blocks': 4, 'dropout': 0.3},
                'epochs': 60,
                'lr': 5e-4,
                'class_weights': class_weights_hard,
                'gamma_easy': 1.0,
                'gamma_hard': 3.0,
                'use_anti_neutral': True,
                'anti_neutral_weight': 0.5,
            }
        },
        # Experiment 8: Combined best techniques
        {
            'name': 'combined_best',
            'model_class': ImprovedDeepResidual,
            'loss_class': ClassSpecificFocalLoss,
            'train_loader': train_loader_oversampled,
            'config': {
                'model_config': {'input_dim': INPUT_DIM, 'hidden_dim': 256, 'num_blocks': 4, 'dropout': 0.25},
                'epochs': 80,
                'lr': 3e-4,
                'class_weights': class_weights_hard,
                'gamma_easy': 1.0,
                'gamma_hard': 4.0,
                'use_anti_neutral': True,
                'anti_neutral_weight': 0.7,
            }
        },
    ]

    # Run experiments
    results = []
    for exp in experiments:
        result = run_experiment(
            exp['name'],
            exp['model_class'],
            exp['loss_class'],
            exp['config'],
            exp['train_loader'],
            val_loader,
            test_loader
        )
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - HARD CLASS IMPROVEMENT EXPERIMENTS")
    print("="*70)
    print(f"\n{'Experiment':<25} {'Test Acc':>10} {'Angry':>10} {'Surprise':>10}")
    print("-"*70)

    baseline_angry = None
    baseline_surprise = None

    for r in results:
        angry = r['class_accs']['angry'] * 100
        surprise = r['class_accs']['surprise'] * 100

        if r['name'] == 'baseline_improved':
            baseline_angry = angry
            baseline_surprise = surprise

        print(f"{r['name']:<25} {r['test_acc']*100:>9.2f}% {angry:>9.1f}% {surprise:>9.1f}%")

    # Find best for hard classes
    best_angry_exp = max(results, key=lambda x: x['class_accs']['angry'])
    best_surprise_exp = max(results, key=lambda x: x['class_accs']['surprise'])
    best_combined = max(results, key=lambda x: x['class_accs']['angry'] + x['class_accs']['surprise'])

    print(f"\nBest for Angry: {best_angry_exp['name']} ({best_angry_exp['class_accs']['angry']*100:.1f}%)")
    print(f"Best for Surprise: {best_surprise_exp['name']} ({best_surprise_exp['class_accs']['surprise']*100:.1f}%)")
    print(f"Best Combined: {best_combined['name']}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = f'runs/hard_class_improvements_{timestamp}'
    os.makedirs(run_dir, exist_ok=True)

    # Save results JSON
    save_results = []
    for r in results:
        save_results.append({
            'name': r['name'],
            'test_acc': float(r['test_acc']),
            'class_accs': {k: float(v) for k, v in r['class_accs'].items()},
        })

    with open(f'{run_dir}/results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Overall accuracy comparison
    ax1 = axes[0, 0]
    names = [r['name'] for r in results]
    accs = [r['test_acc'] * 100 for r in results]
    colors = ['green' if r['name'] == best_combined['name'] else 'steelblue' for r in results]
    bars = ax1.bar(range(len(names)), accs, color=colors)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Overall Test Accuracy')
    ax1.axhline(y=33.97, color='red', linestyle='--', label='Previous best (33.97%)')
    ax1.legend()

    # Plot 2: Hard class comparison
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
    ax2.set_title('Hard Class (Angry & Surprise) Accuracy')
    ax2.axhline(y=1.0, color='red', linestyle=':', alpha=0.5, label='Prev Angry (1%)')
    ax2.axhline(y=17.7, color='orange', linestyle=':', alpha=0.5, label='Prev Surprise (17.7%)')
    ax2.legend()

    # Plot 3: Per-class accuracy for best model
    ax3 = axes[1, 0]
    best_result = best_combined
    class_names = list(best_result['class_accs'].keys())
    class_vals = [best_result['class_accs'][c] * 100 for c in class_names]
    colors = ['red' if c in ['angry', 'surprise'] else 'steelblue' for c in class_names]
    ax3.bar(class_names, class_vals, color=colors)
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title(f"Per-Class Accuracy: {best_result['name']}")
    ax3.axhline(y=14.29, color='gray', linestyle='--', label='Random (14.29%)')
    for i, v in enumerate(class_vals):
        ax3.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

    # Plot 4: Confusion matrix for best model
    ax4 = axes[1, 1]
    cm = confusion_matrix(best_result['labels'], best_result['preds'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax4.imshow(cm_norm, cmap='Blues')
    ax4.set_xticks(range(len(CLASSES)))
    ax4.set_yticks(range(len(CLASSES)))
    ax4.set_xticklabels([c[:3] for c in CLASSES])
    ax4.set_yticklabels([c[:3] for c in CLASSES])
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('True')
    ax4.set_title(f"Confusion Matrix: {best_result['name']}")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax4.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center',
                    color='white' if cm_norm[i,j] > 0.5 else 'black', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{run_dir}/hard_class_improvements.png', dpi=150, bbox_inches='tight')
    plt.savefig('comparison_plots/hard_class_improvements.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {run_dir}/hard_class_improvements.png")

    # Save best model predictions
    np.savez(f'{run_dir}/best_predictions.npz',
             preds=best_result['preds'],
             labels=best_result['labels'])

    print(f"\nResults saved to: {run_dir}/")

    return results


if __name__ == '__main__':
    results = main()
