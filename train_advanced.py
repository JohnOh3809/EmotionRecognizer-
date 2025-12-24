#!/usr/bin/env python3
"""
train_advanced.py

Advanced training with:
- K-fold cross-validation
- Model ensembling
- Metric learning (triplet loss, contrastive loss)
- Prototype networks for few-shot learning
"""

import os
import json
import time
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Emotion classes
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
NUM_JOINTS = 17


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_label(s: str) -> str:
    s = s.strip().lower()
    mapping = {
        "anger": "angry", "angry": "angry", "disgust": "disgust",
        "fear": "fear", "happy": "happy", "neutral": "neutral",
        "sad": "sad", "surprise": "surprise",
    }
    return mapping.get(s, s)


def parse_meta(npz_path: Path) -> Dict[str, Any]:
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            if "meta" not in z:
                return {}
            a = z["meta"]
            if a.size == 0:
                return {}
            s = a[0]
            if hasattr(s, "item"):
                s = s.item()
            if isinstance(s, bytes):
                s = s.decode("utf-8", errors="ignore")
            if isinstance(s, str):
                return json.loads(s)
    except Exception:
        return {}
    return {}


def infer_label(npz_path: Path) -> str:
    meta = parse_meta(npz_path)
    lab = str(meta.get("label", meta.get("label_name", ""))).strip().lower()
    if lab and lab not in ("unknown", "none", "null"):
        return normalize_label(lab)
    return normalize_label(npz_path.parent.name)


# ==================== Dataset ====================

class PoseDataset(Dataset):
    """Dataset for pose sequences."""

    def __init__(self, items: List[Tuple], seq_len: int = 30, num_people: int = 2,
                 augment: bool = False, aug_prob: float = 0.5):
        self.items = items
        self.seq_len = seq_len
        self.num_people = num_people
        self.augment = augment
        self.aug_prob = aug_prob
        # Input: 17 joints * (3 pos + 2 vel + 2 acc) * 2 people = 238
        self.input_dim = NUM_JOINTS * 7 * num_people

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        npz_path, label_idx = self.items[idx][:2]

        try:
            with np.load(npz_path, allow_pickle=True) as z:
                kp = z["keypoints"]  # (T, P, 17, 3)
                vel = z["vel"]       # (T, P, 17, 2)
                acc = z["acc"]       # (T, P, 17, 2)
        except Exception:
            # Return zeros on error
            return torch.zeros(self.seq_len, self.input_dim), label_idx

        T = kp.shape[0]
        P = min(kp.shape[1], self.num_people)

        # Pad people dimension if needed
        if P < self.num_people:
            pad_p = self.num_people - P
            kp = np.pad(kp, ((0, 0), (0, pad_p), (0, 0), (0, 0)))
            vel = np.pad(vel, ((0, 0), (0, pad_p), (0, 0), (0, 0)))
            acc = np.pad(acc, ((0, 0), (0, pad_p), (0, 0), (0, 0)))
        else:
            kp = kp[:, :self.num_people]
            vel = vel[:, :self.num_people]
            acc = acc[:, :self.num_people]

        # Concatenate features
        feat = np.concatenate([
            kp.reshape(T, -1),
            vel.reshape(T, -1),
            acc.reshape(T, -1)
        ], axis=1)

        # Sample or pad to seq_len
        if T >= self.seq_len:
            start = random.randint(0, T - self.seq_len) if self.augment else (T - self.seq_len) // 2
            feat = feat[start:start + self.seq_len]
        else:
            pad = np.zeros((self.seq_len - T, feat.shape[1]), dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=0)

        # Augmentation
        if self.augment and random.random() < self.aug_prob:
            # Random scaling
            if random.random() < 0.5:
                feat = feat * np.random.uniform(0.9, 1.1)
            # Random noise
            if random.random() < 0.5:
                feat = feat + np.random.randn(*feat.shape).astype(np.float32) * 0.02
            # Temporal jitter
            if random.random() < 0.3:
                shift = random.randint(-2, 2)
                feat = np.roll(feat, shift, axis=0)

        # Replace NaN/Inf
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.tensor(feat, dtype=torch.float32), label_idx


# ==================== Models ====================

class SimpleBiLSTM(nn.Module):
    """Simple BiLSTM classifier."""

    def __init__(self, in_dim: int, hidden: int = 128, num_layers: int = 1,
                 num_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers, batch_first=True,
                           bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last timestep
        out = self.dropout(out)
        return self.fc(out)

    def get_embedding(self, x):
        """Get embedding before classification head."""
        out, _ = self.lstm(x)
        return out[:, -1, :]


class EmbeddingNet(nn.Module):
    """Network that outputs embeddings for metric learning."""

    def __init__(self, in_dim: int, hidden: int = 128, embed_dim: int = 64,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers, batch_first=True,
                           bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Linear(hidden * 2, embed_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.embed(out)
        return F.normalize(out, p=2, dim=1)  # L2 normalize


class PrototypicalNet(nn.Module):
    """Prototypical Network for few-shot learning."""

    def __init__(self, in_dim: int, hidden: int = 128, embed_dim: int = 64,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.encoder = EmbeddingNet(in_dim, hidden, embed_dim, num_layers, dropout)

    def forward(self, support_x, support_y, query_x, num_classes: int = 7):
        """
        support_x: (N_support, T, D) - support examples
        support_y: (N_support,) - support labels
        query_x: (N_query, T, D) - query examples
        """
        # Encode all examples
        support_emb = self.encoder(support_x)  # (N_support, embed_dim)
        query_emb = self.encoder(query_x)      # (N_query, embed_dim)

        # Compute prototypes (mean embedding per class)
        prototypes = []
        for c in range(num_classes):
            mask = support_y == c
            if mask.sum() > 0:
                proto = support_emb[mask].mean(dim=0)
            else:
                proto = torch.zeros_like(support_emb[0])
            prototypes.append(proto)
        prototypes = torch.stack(prototypes)  # (num_classes, embed_dim)

        # Compute distances (negative squared Euclidean)
        dists = torch.cdist(query_emb, prototypes)  # (N_query, num_classes)
        return -dists  # Return negative distance as logits

    def encode(self, x):
        return self.encoder(x)


# ==================== Losses ====================

class TripletLoss(nn.Module):
    """Triplet loss with hard negative mining."""

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        embeddings: (N, embed_dim)
        labels: (N,)
        """
        # Compute pairwise distances
        dist_mat = torch.cdist(embeddings, embeddings)  # (N, N)

        loss = 0.0
        count = 0

        for i in range(len(labels)):
            anchor_label = labels[i]

            # Positive: same class, furthest
            pos_mask = labels == anchor_label
            pos_mask[i] = False
            if pos_mask.sum() == 0:
                continue
            pos_dist = dist_mat[i][pos_mask].max()  # Hard positive

            # Negative: different class, closest
            neg_mask = labels != anchor_label
            if neg_mask.sum() == 0:
                continue
            neg_dist = dist_mat[i][neg_mask].min()  # Hard negative

            loss += F.relu(pos_dist - neg_dist + self.margin)
            count += 1

        return loss / max(count, 1)


class ContrastiveLoss(nn.Module):
    """Supervised contrastive loss."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        embeddings: (N, embed_dim) - L2 normalized
        labels: (N,)
        """
        # Compute similarity matrix
        sim_mat = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Mask for positive pairs (same class, excluding self)
        labels = labels.view(-1, 1)
        mask = (labels == labels.t()).float()
        mask.fill_diagonal_(0)

        # For each anchor, compute loss
        exp_sim = torch.exp(sim_mat)
        exp_sim.fill_diagonal_(0)

        # Denominator: sum of all negatives
        neg_mask = 1 - mask
        neg_mask.fill_diagonal_(0)

        log_prob = sim_mat - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of positive pairs
        pos_count = mask.sum(dim=1)
        loss = -(mask * log_prob).sum(dim=1) / (pos_count + 1e-8)

        return loss[pos_count > 0].mean()


class CenterLoss(nn.Module):
    """Center loss for intra-class compactness."""

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, embed_dim))

    def forward(self, embeddings, labels):
        centers_batch = self.centers[labels]
        return ((embeddings - centers_batch) ** 2).sum(dim=1).mean()


# ==================== Training Functions ====================

def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    total = sum(counts.values())
    active_classes = sum(1 for i in range(num_classes) if counts.get(i, 0) > 0)

    weights = []
    for i in range(num_classes):
        count = counts.get(i, 0)
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (active_classes * count))

    weights = torch.tensor(weights, dtype=torch.float32)
    non_zero_sum = weights.sum()
    if non_zero_sum > 0:
        weights = weights / non_zero_sum * active_classes
    return weights


def train_epoch(model, loader, optimizer, criterion, device, use_metric_loss=False,
                metric_criterion=None, metric_weight=0.5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if use_metric_loss and hasattr(model, 'get_embedding'):
            logits = model(x)
            embeddings = model.get_embedding(x)
            ce_loss = criterion(logits, y)
            metric_loss = metric_criterion(embeddings, y)
            loss = ce_loss + metric_weight * metric_loss
        else:
            logits = model(x)
            loss = criterion(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
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

    # Per-class accuracy
    class_acc = {}
    for i, cls in enumerate(CLASSES):
        mask = np.array(all_labels) == i
        if mask.sum() > 0:
            class_acc[cls] = float((np.array(all_preds)[mask] == i).sum() / mask.sum())
        else:
            class_acc[cls] = 0.0

    return total_loss / total, correct / total, class_acc, all_preds, all_labels


# ==================== K-Fold Cross Validation ====================

def train_kfold(args, items, device):
    """Train with K-fold cross-validation."""
    print(f"\n{'='*60}")
    print(f"K-Fold Cross Validation (K={args.n_folds})")
    print(f"{'='*60}")

    # Prepare data
    labels = [item[1] for item in items]
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_val_preds = []
    all_val_labels = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(items, labels)):
        print(f"\n--- Fold {fold + 1}/{args.n_folds} ---")

        train_items = [items[i] for i in train_idx]
        val_items = [items[i] for i in val_idx]

        train_ds = PoseDataset(train_items, seq_len=args.seq_len, augment=True, aug_prob=args.aug_prob)
        val_ds = PoseDataset(val_items, seq_len=args.seq_len, augment=False)

        train_labels = [item[1] for item in train_items]
        class_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)

        # Balanced sampler
        sample_weights = [1.0 / (Counter(train_labels)[l] + 1) for l in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Model
        model = SimpleBiLSTM(
            in_dim=train_ds.input_dim,
            hidden=args.hidden,
            num_layers=args.layers,
            num_classes=NUM_CLASSES,
            dropout=args.dropout
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        best_val_acc = 0
        best_model_state = None

        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, class_acc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

        # Evaluate best model
        model.load_state_dict(best_model_state)
        _, final_acc, class_acc, preds, labels_list = evaluate(model, val_loader, criterion, device)

        fold_results.append({
            'fold': fold + 1,
            'val_acc': final_acc,
            'class_acc': class_acc
        })
        all_val_preds.extend(preds)
        all_val_labels.extend(labels_list)

        print(f"  Best Val Acc: {final_acc:.4f}")

    # Summary
    mean_acc = np.mean([r['val_acc'] for r in fold_results])
    std_acc = np.std([r['val_acc'] for r in fold_results])

    print(f"\n{'='*60}")
    print(f"K-Fold Results: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
    print(f"{'='*60}")

    return fold_results, mean_acc, std_acc


# ==================== Ensemble Training ====================

def train_ensemble(args, train_items, val_items, test_items, device):
    """Train an ensemble of diverse models."""
    print(f"\n{'='*60}")
    print(f"Ensemble Training (N={args.n_ensemble})")
    print(f"{'='*60}")

    train_ds = PoseDataset(train_items, seq_len=args.seq_len, augment=True, aug_prob=args.aug_prob)
    val_ds = PoseDataset(val_items, seq_len=args.seq_len, augment=False)
    test_ds = PoseDataset(test_items, seq_len=args.seq_len, augment=False)

    train_labels = [item[1] for item in train_items]
    class_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)

    sample_weights = [1.0 / (Counter(train_labels)[l] + 1) for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Diverse model configurations
    configs = [
        {'hidden': 64, 'layers': 1, 'dropout': 0.3},
        {'hidden': 128, 'layers': 1, 'dropout': 0.4},
        {'hidden': 96, 'layers': 1, 'dropout': 0.35},
        {'hidden': 64, 'layers': 2, 'dropout': 0.4},
        {'hidden': 128, 'layers': 1, 'dropout': 0.5},
    ][:args.n_ensemble]

    models = []
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    for i, cfg in enumerate(configs):
        print(f"\n--- Training Model {i+1}/{len(configs)} (h={cfg['hidden']}, l={cfg['layers']}, d={cfg['dropout']}) ---")

        model = SimpleBiLSTM(
            in_dim=train_ds.input_dim,
            hidden=cfg['hidden'],
            num_layers=cfg['layers'],
            num_classes=NUM_CLASSES,
            dropout=cfg['dropout']
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        best_val_acc = 0
        best_state = None

        for epoch in range(args.epochs):
            # Different random seed per model for diversity
            seed_all(42 + i * 1000 + epoch)

            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: val_acc={val_acc:.3f}")

        model.load_state_dict(best_state)
        models.append(model)
        print(f"  Best Val Acc: {best_val_acc:.4f}")

    # Ensemble prediction
    print("\n--- Ensemble Evaluation ---")

    def ensemble_predict(loader):
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)

                # Average logits from all models
                logits_sum = None
                for model in models:
                    model.eval()
                    logits = model(x)
                    if logits_sum is None:
                        logits_sum = logits
                    else:
                        logits_sum += logits

                preds = logits_sum.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())

        return all_preds, all_labels

    val_preds, val_labels = ensemble_predict(val_loader)
    val_acc = np.mean(np.array(val_preds) == np.array(val_labels))

    test_preds, test_labels = ensemble_predict(test_loader)
    test_acc = np.mean(np.array(test_preds) == np.array(test_labels))

    # Per-class accuracy
    print(f"\nEnsemble Val Accuracy: {val_acc*100:.2f}%")
    print(f"Ensemble Test Accuracy: {test_acc*100:.2f}%")

    print("\nPer-class accuracy (test):")
    for i, cls in enumerate(CLASSES):
        mask = np.array(test_labels) == i
        if mask.sum() > 0:
            acc = (np.array(test_preds)[mask] == i).sum() / mask.sum()
            print(f"  {cls}: {acc*100:.1f}%")

    return models, val_acc, test_acc


# ==================== Metric Learning Training ====================

def train_metric_learning(args, train_items, val_items, test_items, device):
    """Train with metric learning (triplet/contrastive loss)."""
    print(f"\n{'='*60}")
    print(f"Metric Learning Training ({args.metric_type})")
    print(f"{'='*60}")

    train_ds = PoseDataset(train_items, seq_len=args.seq_len, augment=True, aug_prob=args.aug_prob)
    val_ds = PoseDataset(val_items, seq_len=args.seq_len, augment=False)
    test_ds = PoseDataset(test_items, seq_len=args.seq_len, augment=False)

    train_labels = [item[1] for item in train_items]
    class_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)

    sample_weights = [1.0 / (Counter(train_labels)[l] + 1) for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Larger batch for metric learning
    train_loader = DataLoader(train_ds, batch_size=args.batch_size * 2, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model with embedding + classifier
    class MetricModel(nn.Module):
        def __init__(self, in_dim, hidden, embed_dim, num_classes, dropout):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hidden, 1, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(dropout)
            self.embed = nn.Linear(hidden * 2, embed_dim)
            self.fc = nn.Linear(embed_dim, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.dropout(out)
            emb = self.embed(out)
            return self.fc(emb)

        def get_embedding(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.dropout(out)
            emb = self.embed(out)
            return F.normalize(emb, p=2, dim=1)

    model = MetricModel(
        in_dim=train_ds.input_dim,
        hidden=args.hidden,
        embed_dim=args.embed_dim,
        num_classes=NUM_CLASSES,
        dropout=args.dropout
    ).to(device)

    ce_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    if args.metric_type == 'triplet':
        metric_criterion = TripletLoss(margin=args.triplet_margin)
    elif args.metric_type == 'contrastive':
        metric_criterion = ContrastiveLoss(temperature=args.temperature)
    elif args.metric_type == 'center':
        metric_criterion = CenterLoss(NUM_CLASSES, args.embed_dim).to(device)
    else:
        metric_criterion = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if metric_criterion and hasattr(metric_criterion, 'parameters'):
        optimizer.add_param_group({'params': metric_criterion.parameters(), 'lr': args.lr * 0.5})

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val_acc = 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(x)
            embeddings = model.get_embedding(x)

            ce_loss = ce_criterion(logits, y)
            if metric_criterion:
                m_loss = metric_criterion(embeddings, y)
                loss = ce_loss + args.metric_weight * m_loss
            else:
                loss = ce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_acc = correct / total
        val_loss, val_acc, class_acc, _, _ = evaluate(model, val_loader, ce_criterion, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

    # Final evaluation
    model.load_state_dict(best_state)
    _, test_acc, class_acc, _, _ = evaluate(model, test_loader, ce_criterion, device)

    print(f"\nBest Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    print("\nPer-class accuracy (test):")
    for cls, acc in class_acc.items():
        print(f"  {cls}: {acc*100:.1f}%")

    return model, best_val_acc, test_acc


# ==================== Prototypical Network Training ====================

def train_prototypical(args, train_items, val_items, test_items, device):
    """Train with prototypical networks for few-shot learning."""
    print(f"\n{'='*60}")
    print(f"Prototypical Network Training")
    print(f"{'='*60}")

    train_ds = PoseDataset(train_items, seq_len=args.seq_len, augment=True, aug_prob=args.aug_prob)
    val_ds = PoseDataset(val_items, seq_len=args.seq_len, augment=False)
    test_ds = PoseDataset(test_items, seq_len=args.seq_len, augment=False)

    model = PrototypicalNet(
        in_dim=train_ds.input_dim,
        hidden=args.hidden,
        embed_dim=args.embed_dim,
        num_layers=1,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Episode-based training
    n_support = args.n_support  # Support examples per class
    n_query = args.n_query      # Query examples per class

    # Group items by class
    class_items = {c: [] for c in range(NUM_CLASSES)}
    for item in train_items:
        class_items[item[1]].append(item)

    best_val_acc = 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Sample episodes
        for _ in range(args.episodes_per_epoch):
            support_x, support_y = [], []
            query_x, query_y = [], []

            for c in range(NUM_CLASSES):
                if len(class_items[c]) < n_support + n_query:
                    continue

                selected = random.sample(class_items[c], min(n_support + n_query, len(class_items[c])))

                for i, item in enumerate(selected[:n_support]):
                    x, _ = train_ds[train_items.index(item)]
                    support_x.append(x)
                    support_y.append(c)

                for item in selected[n_support:n_support + n_query]:
                    x, _ = train_ds[train_items.index(item)]
                    query_x.append(x)
                    query_y.append(c)

            if len(support_x) == 0 or len(query_x) == 0:
                continue

            support_x = torch.stack(support_x).to(device)
            support_y = torch.tensor(support_y).to(device)
            query_x = torch.stack(query_x).to(device)
            query_y = torch.tensor(query_y).to(device)

            optimizer.zero_grad()
            logits = model(support_x, support_y, query_x, NUM_CLASSES)
            loss = F.cross_entropy(logits, query_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == query_y).sum().item()
            total += len(query_y)

        scheduler.step()

        if total > 0:
            train_acc = correct / total
        else:
            train_acc = 0

        # Validation: use all training data as support, validate on val set
        model.eval()
        with torch.no_grad():
            # Build support from training data (sample subset)
            support_x, support_y = [], []
            for c in range(NUM_CLASSES):
                samples = random.sample(class_items[c], min(20, len(class_items[c])))
                for item in samples:
                    x, _ = train_ds[train_items.index(item)]
                    support_x.append(x)
                    support_y.append(c)

            support_x = torch.stack(support_x).to(device)
            support_y = torch.tensor(support_y).to(device)

            val_correct = 0
            val_total = 0

            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
            for query_x, query_y in val_loader:
                query_x = query_x.to(device)
                logits = model(support_x, support_y, query_x, NUM_CLASSES)
                preds = logits.argmax(dim=1)
                val_correct += (preds == query_y.to(device)).sum().item()
                val_total += len(query_y)

            val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

    print(f"\nBest Val Accuracy: {best_val_acc*100:.2f}%")

    return model, best_val_acc


# ==================== Main ====================

def load_data(data_dir: Path, max_samples_per_class: int = 0):
    """Load all data items."""
    items = []

    for cls in CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            continue

        cls_items = []
        for npz_file in cls_dir.glob("*.npz"):
            label = infer_label(npz_file)
            if label in CLASS_TO_IDX:
                cls_items.append((str(npz_file), CLASS_TO_IDX[label]))

        if max_samples_per_class > 0 and len(cls_items) > max_samples_per_class:
            random.shuffle(cls_items)
            cls_items = cls_items[:max_samples_per_class]

        items.extend(cls_items)

    random.shuffle(items)
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="advanced")
    parser.add_argument("--mode", choices=["kfold", "ensemble", "metric", "prototypical", "all"], default="all")

    # Model params
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--embed_dim", type=int, default=64)

    # Training params
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--aug_prob", type=float, default=0.5)
    parser.add_argument("--max_samples_per_class", type=int, default=400)

    # K-fold params
    parser.add_argument("--n_folds", type=int, default=5)

    # Ensemble params
    parser.add_argument("--n_ensemble", type=int, default=5)

    # Metric learning params
    parser.add_argument("--metric_type", choices=["triplet", "contrastive", "center"], default="triplet")
    parser.add_argument("--metric_weight", type=float, default=0.5)
    parser.add_argument("--triplet_margin", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.1)

    # Prototypical params
    parser.add_argument("--n_support", type=int, default=5)
    parser.add_argument("--n_query", type=int, default=5)
    parser.add_argument("--episodes_per_epoch", type=int, default=100)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_dir = Path(args.data_dir).expanduser().resolve()
    items = load_data(data_dir, args.max_samples_per_class)
    print(f"Loaded {len(items)} samples")

    # Split data
    random.shuffle(items)
    n = len(items)
    train_items = items[:int(0.7 * n)]
    val_items = items[int(0.7 * n):int(0.85 * n)]
    test_items = items[int(0.85 * n):]

    print(f"Split: train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")

    # Run selected mode(s)
    results = {}

    if args.mode in ["kfold", "all"]:
        fold_results, mean_acc, std_acc = train_kfold(args, items, device)
        results['kfold'] = {'mean': mean_acc, 'std': std_acc}

    if args.mode in ["ensemble", "all"]:
        models, val_acc, test_acc = train_ensemble(args, train_items, val_items, test_items, device)
        results['ensemble'] = {'val_acc': val_acc, 'test_acc': test_acc}

    if args.mode in ["metric", "all"]:
        model, val_acc, test_acc = train_metric_learning(args, train_items, val_items, test_items, device)
        results['metric'] = {'val_acc': val_acc, 'test_acc': test_acc}

    if args.mode in ["prototypical", "all"]:
        model, val_acc = train_prototypical(args, train_items, val_items, test_items, device)
        results['prototypical'] = {'val_acc': val_acc}

    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for method, res in results.items():
        print(f"{method}: {res}")


if __name__ == "__main__":
    main()
