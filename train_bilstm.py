#!/usr/bin/env python3
"""
train_bilstm.py

Train a BiLSTM emotion classifier on the enhanced pose data from enhance_data.py.
Uses keypoints, velocity, and acceleration features.

7 emotion classes: angry, disgust, fear, happy, neutral, sad, surprise

Features:
- Class-weighted loss to handle imbalance
- Strong data augmentation for minority classes
- Timestamped run directories
"""

import os
import json
import time
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Emotion classes (7 classes)
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "other"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# COCO-17 keypoint indices
NUM_JOINTS = 17


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_label(s: str) -> str:
    s = s.strip().lower()
    mapping = {
        "anger": "angry",
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "neutral": "neutral",
        "sad": "sad",
        "surprise": "surprise",
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
    """Infer label from metadata or parent directory."""
    meta = parse_meta(npz_path)
    lab = str(meta.get("label", meta.get("label_name", ""))).strip().lower()
    if lab and lab not in ("unknown", "none", "null"):
        return normalize_label(lab)
    return normalize_label(npz_path.parent.name)


def infer_split(npz_path: Path) -> str:
    """Infer train/val/test split from path."""
    parts = [x.lower() for x in npz_path.parts]
    if "train" in parts:
        return "train"
    if "validation" in parts or "val" in parts:
        return "val"
    if "test" in parts:
        return "test"
    return "unknown"


def collect_samples(data_dir: Path) -> List[Tuple[Path, str, str, int]]:
    """
    Collect all enhanced .npz files.
    Returns: list of (path, split, label, class_idx)
    """
    items = []
    for npz_path in sorted(data_dir.rglob("*__kin.npz")):
        label = infer_label(npz_path)
        if label not in CLASS_TO_IDX:
            continue
        split = infer_split(npz_path)
        if split == "unknown":
            split = "train"
        items.append((npz_path, split, label, CLASS_TO_IDX[label]))
    return items


class PoseAugmentor:
    """
    Data augmentation for pose sequences.
    Applies various transformations to increase data diversity.
    """

    def __init__(self, aug_prob: float = 0.5):
        self.aug_prob = aug_prob

    def __call__(self, kps: np.ndarray, vel: np.ndarray, acc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply augmentations to pose data.
        kps: (T, K, 17, 3) - x, y, conf
        vel: (T, K, 17, 2) - vx, vy
        acc: (T, K, 17, 2) - ax, ay
        """
        # Random horizontal flip
        if random.random() < self.aug_prob:
            kps, vel, acc = self.horizontal_flip(kps, vel, acc)

        # Random scaling
        if random.random() < self.aug_prob:
            kps, vel, acc = self.random_scale(kps, vel, acc)

        # Random rotation
        if random.random() < self.aug_prob:
            kps, vel, acc = self.random_rotation(kps, vel, acc)

        # Add Gaussian noise
        if random.random() < self.aug_prob:
            kps, vel, acc = self.add_noise(kps, vel, acc)

        # Time warping (speed variation)
        if random.random() < self.aug_prob * 0.5:
            kps, vel, acc = self.time_warp(kps, vel, acc)

        # Random joint dropout
        if random.random() < self.aug_prob * 0.3:
            kps, vel, acc = self.joint_dropout(kps, vel, acc)

        return kps, vel, acc

    def horizontal_flip(self, kps, vel, acc):
        """Flip poses horizontally."""
        # Flip x coordinates (assuming normalized to [0, 1])
        kps = kps.copy()
        vel = vel.copy()
        kps[..., 0] = 1.0 - kps[..., 0]  # Flip x
        vel[..., 0] = -vel[..., 0]  # Flip vx
        acc[..., 0] = -acc[..., 0]  # Flip ax

        # Swap left/right keypoints (COCO format)
        # Left-right pairs: (1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)
        swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
        for l, r in swap_pairs:
            kps[:, :, [l, r]] = kps[:, :, [r, l]]
            vel[:, :, [l, r]] = vel[:, :, [r, l]]
            acc[:, :, [l, r]] = acc[:, :, [r, l]]

        return kps, vel, acc

    def random_scale(self, kps, vel, acc):
        """Random scaling of poses."""
        scale = random.uniform(0.8, 1.2)
        kps = kps.copy()
        vel = vel.copy()
        acc = acc.copy()

        # Scale x, y coordinates around center
        kps[..., :2] = (kps[..., :2] - 0.5) * scale + 0.5
        vel[..., :2] = vel[..., :2] * scale
        acc[..., :2] = acc[..., :2] * scale

        return kps, vel, acc

    def random_rotation(self, kps, vel, acc):
        """Random rotation of poses."""
        angle = random.uniform(-15, 15) * np.pi / 180  # -15 to 15 degrees
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        kps = kps.copy()
        vel = vel.copy()
        acc = acc.copy()

        # Rotate around center (0.5, 0.5)
        x = kps[..., 0] - 0.5
        y = kps[..., 1] - 0.5
        kps[..., 0] = x * cos_a - y * sin_a + 0.5
        kps[..., 1] = x * sin_a + y * cos_a + 0.5

        # Rotate velocities
        vx, vy = vel[..., 0], vel[..., 1]
        vel[..., 0] = vx * cos_a - vy * sin_a
        vel[..., 1] = vx * sin_a + vy * cos_a

        # Rotate accelerations
        ax, ay = acc[..., 0], acc[..., 1]
        acc[..., 0] = ax * cos_a - ay * sin_a
        acc[..., 1] = ax * sin_a + ay * cos_a

        return kps, vel, acc

    def add_noise(self, kps, vel, acc):
        """Add Gaussian noise."""
        kps = kps.copy()
        vel = vel.copy()
        acc = acc.copy()

        noise_std = random.uniform(0.005, 0.02)
        kps[..., :2] += np.random.randn(*kps[..., :2].shape).astype(np.float32) * noise_std
        vel += np.random.randn(*vel.shape).astype(np.float32) * noise_std * 2
        acc += np.random.randn(*acc.shape).astype(np.float32) * noise_std * 4

        return kps, vel, acc

    def time_warp(self, kps, vel, acc):
        """Time warping - speed up or slow down."""
        T = kps.shape[0]
        if T < 4:
            return kps, vel, acc

        # Random speed factor
        speed = random.uniform(0.8, 1.2)
        new_T = int(T / speed)
        new_T = max(4, min(new_T, T * 2))

        # Interpolate
        old_indices = np.linspace(0, T - 1, new_T)
        new_kps = np.zeros((new_T, *kps.shape[1:]), dtype=np.float32)
        new_vel = np.zeros((new_T, *vel.shape[1:]), dtype=np.float32)
        new_acc = np.zeros((new_T, *acc.shape[1:]), dtype=np.float32)

        for i, idx in enumerate(old_indices):
            low = int(idx)
            high = min(low + 1, T - 1)
            alpha = idx - low
            new_kps[i] = kps[low] * (1 - alpha) + kps[high] * alpha
            new_vel[i] = vel[low] * (1 - alpha) + vel[high] * alpha
            new_acc[i] = acc[low] * (1 - alpha) + acc[high] * alpha

        return new_kps, new_vel, new_acc

    def joint_dropout(self, kps, vel, acc):
        """Randomly drop some joints."""
        kps = kps.copy()
        vel = vel.copy()
        acc = acc.copy()

        # Drop 1-3 random joints
        n_drop = random.randint(1, 3)
        drop_joints = random.sample(range(NUM_JOINTS), n_drop)

        for j in drop_joints:
            kps[:, :, j, :] = 0
            vel[:, :, j, :] = 0
            acc[:, :, j, :] = 0

        return kps, vel, acc


class EnhancedPoseDataset(Dataset):
    """
    Dataset for enhanced pose data with augmentation support.
    """

    def __init__(
        self,
        items: List[Tuple[Path, str, str, int]],
        seq_len: int = 64,
        num_people: int = 2,
        train: bool = True,
        augment: bool = True,
        aug_prob: float = 0.5,
    ):
        self.items = items
        self.seq_len = seq_len
        self.num_people = num_people
        self.train = train
        self.augment = augment and train
        self.augmentor = PoseAugmentor(aug_prob=aug_prob) if self.augment else None
        self.features_per_joint = 7  # x, y, conf, vx, vy, ax, ay
        self.in_dim = NUM_JOINTS * self.features_per_joint * num_people

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        npz_path, split, label, y = self.items[idx]

        try:
            with np.load(npz_path, allow_pickle=False) as z:
                kps = z["keypoints"].astype(np.float32)  # (T, K, 17, 3)
                vel = z["vel"].astype(np.float32)        # (T, K, 17, 2)
                acc = z["acc"].astype(np.float32)        # (T, K, 17, 2)

            T = kps.shape[0]
            K = min(kps.shape[1], self.num_people)

            kps = kps[:, :K]
            vel = vel[:, :K]
            acc = acc[:, :K]

            # Apply augmentation before cropping
            if self.augment and self.augmentor is not None:
                kps, vel, acc = self.augmentor(kps, vel, acc)
                T = kps.shape[0]  # T might change after time_warp

            # Pad or truncate to seq_len
            if T < self.seq_len:
                pad_kps = np.zeros((self.seq_len, K, NUM_JOINTS, 3), dtype=np.float32)
                pad_vel = np.zeros((self.seq_len, K, NUM_JOINTS, 2), dtype=np.float32)
                pad_acc = np.zeros((self.seq_len, K, NUM_JOINTS, 2), dtype=np.float32)
                pad_kps[:T] = kps
                pad_vel[:T] = vel
                pad_acc[:T] = acc
                kps, vel, acc = pad_kps, pad_vel, pad_acc
            elif T > self.seq_len:
                if self.train:
                    start = random.randint(0, T - self.seq_len)
                else:
                    start = (T - self.seq_len) // 2
                kps = kps[start:start + self.seq_len]
                vel = vel[start:start + self.seq_len]
                acc = acc[start:start + self.seq_len]

            # Pad people dimension if needed
            if kps.shape[1] < self.num_people:
                pad_k = self.num_people - kps.shape[1]
                kps = np.pad(kps, ((0, 0), (0, pad_k), (0, 0), (0, 0)), mode='constant')
                vel = np.pad(vel, ((0, 0), (0, pad_k), (0, 0), (0, 0)), mode='constant')
                acc = np.pad(acc, ((0, 0), (0, pad_k), (0, 0), (0, 0)), mode='constant')

            # Replace NaN with 0
            kps = np.nan_to_num(kps, nan=0.0)
            vel = np.nan_to_num(vel, nan=0.0)
            acc = np.nan_to_num(acc, nan=0.0)

            # Concatenate features
            features = np.concatenate([kps, vel, acc], axis=-1)  # (T, K, 17, 7)
            features = features.reshape(self.seq_len, -1)  # (T, K*17*7)

            x = torch.from_numpy(features)
            y_tensor = torch.tensor(y, dtype=torch.long)

        except Exception as e:
            x = torch.zeros((self.seq_len, self.in_dim), dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)

        return x, y_tensor


class BiLSTMEmotionClassifier(nn.Module):
    """
    BiLSTM model for emotion classification from pose sequences.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
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
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return self.head(context)


class TemporalBlock(nn.Module):
    """Temporal convolutional block with residual connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        res = self.downsample(x)
        return torch.relu(out + res)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for sequence data."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TCNBiLSTMClassifier(nn.Module):
    """
    Temporal CNN + BiLSTM hybrid model for emotion classification.
    Combines local temporal patterns (TCN) with long-range dependencies (BiLSTM).
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 512,
        num_layers: int = 3,
        num_classes: int = 7,
        dropout: float = 0.4,
        num_heads: int = 8,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Temporal CNN blocks with increasing dilation
        self.tcn = nn.Sequential(
            TemporalBlock(hidden, hidden, kernel_size=3, dilation=1, dropout=dropout * 0.5),
            TemporalBlock(hidden, hidden, kernel_size=3, dilation=2, dropout=dropout * 0.5),
            TemporalBlock(hidden, hidden, kernel_size=3, dilation=4, dropout=dropout * 0.5),
        )

        # BiLSTM for sequential modeling
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.lstm_ln = nn.LayerNorm(hidden)

        # Multi-head attention
        self.attention = MultiHeadAttention(hidden, num_heads=num_heads, dropout=dropout)
        self.attn_ln = nn.LayerNorm(hidden)

        # Pooling attention
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.Tanh(),
            nn.Linear(hidden // 4, 1, bias=False),
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, T, in_dim)
        x = self.input_proj(x)  # (B, T, hidden)

        # TCN expects (B, C, T)
        x_tcn = x.transpose(1, 2)
        x_tcn = self.tcn(x_tcn)
        x = x_tcn.transpose(1, 2)  # back to (B, T, hidden)

        # BiLSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_ln(lstm_out)

        # Self-attention with residual
        attn_out = self.attention(lstm_out)
        attn_out = self.attn_ln(attn_out + lstm_out)

        # Attention pooling
        pool_weights = self.pool_attn(attn_out)
        pool_weights = torch.softmax(pool_weights, dim=1)
        context = torch.sum(attn_out * pool_weights, dim=1)

        # Also add max and mean pooling
        max_pool = attn_out.max(dim=1)[0]

        # Concatenate different pooling strategies
        combined = torch.cat([context, max_pool], dim=1)

        return self.head(combined)


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Compute inverse frequency class weights.

    Classes with 0 samples get weight 0 (they won't contribute to loss).
    """
    counts = Counter(labels)
    total = sum(counts.values())

    # Count how many classes actually have samples
    active_classes = sum(1 for i in range(num_classes) if counts.get(i, 0) > 0)

    weights = []
    for i in range(num_classes):
        count = counts.get(i, 0)
        if count == 0:
            # No samples for this class - set weight to 0
            weights.append(0.0)
        else:
            # Inverse frequency with smoothing
            weight = total / (active_classes * count)
            weights.append(weight)

    # Normalize so non-zero weights sum to active_classes (not num_classes)
    weights = torch.tensor(weights, dtype=torch.float32)
    non_zero_sum = weights.sum()
    if non_zero_sum > 0:
        weights = weights / non_zero_sum * active_classes
    return weights


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha,
            reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Transformer-based model for emotion classification from pose sequences.
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        num_layers: int = 4,
        num_classes: int = 7,
        dropout: float = 0.3,
        num_heads: int = 8,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.pos_encoder = PositionalEncoding(hidden, dropout=dropout * 0.5)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))

        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        B = x.size(0)
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.transformer(x)
        cls_out = x[:, 0]

        return self.head(cls_out)


def create_balanced_sampler(labels: List[int], num_classes: int) -> WeightedRandomSampler:
    """Create a sampler that oversamples minority classes."""
    counts = Counter(labels)
    max_count = max(counts.values())

    # Weight each sample inversely to its class frequency
    sample_weights = []
    for label in labels:
        weight = max_count / counts[label]
        sample_weights.append(weight)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )


@torch.no_grad()
def evaluate(model, loader, device, criterion=None, num_classes=7):
    """Evaluate model and return accuracy, loss, and per-class metrics."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        if criterion is not None:
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)

    acc = correct / max(total, 1)
    avg_loss = total_loss / max(total, 1) if criterion is not None else 0.0

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    class_acc = {}
    for i, cls in enumerate(CLASSES):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc[cls] = float((all_preds[mask] == i).sum() / mask.sum())
        else:
            class_acc[cls] = 0.0

    return acc, avg_loss, class_acc, all_preds, all_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Enhanced data directory")
    ap.add_argument("--run_name", type=str, default="", help="Optional run name suffix")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--num_people", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--aug_prob", type=float, default=0.7, help="Augmentation probability")
    ap.add_argument("--use_balanced_sampler", action="store_true", help="Use balanced sampling")
    ap.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    ap.add_argument("--model", type=str, default="tcn_bilstm", choices=["bilstm", "tcn_bilstm", "transformer"],
                    help="Model architecture: bilstm, tcn_bilstm, or transformer")
    ap.add_argument("--num_heads", type=int, default=8, help="Number of attention heads (tcn_bilstm only)")
    ap.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    ap.add_argument("--focal_loss", action="store_true", help="Use focal loss instead of CE")
    ap.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    ap.add_argument("--max_samples_per_class", type=int, default=0, help="Max samples per class (0=no limit)")
    args = ap.parse_args()

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"bilstm_{timestamp}"
    if args.run_name:
        run_name += f"_{args.run_name}"
    workdir = os.path.join("./runs", run_name)
    os.makedirs(workdir, exist_ok=True)

    # Save config
    with open(os.path.join(workdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Run directory: {workdir}")

    # Collect samples
    data_dir = Path(args.data_dir).expanduser().resolve()
    items = collect_samples(data_dir)

    # Split by inferred split
    train_items = [it for it in items if it[1] == "train"]
    val_items = [it for it in items if it[1] == "val"]
    test_items = [it for it in items if it[1] == "test"]

    # If no explicit splits, randomly split with stratification
    if len(val_items) == 0 and len(test_items) == 0:
        all_items = train_items.copy()
        random.shuffle(all_items)
        n = len(all_items)
        train_items = all_items[:int(n * 0.8)]
        val_items = all_items[int(n * 0.8):int(n * 0.9)]
        test_items = all_items[int(n * 0.9):]

    # Cap samples per class to reduce bias from majority classes
    if args.max_samples_per_class > 0:
        print(f"\nCapping training samples to max {args.max_samples_per_class} per class...")
        items_by_class = {}
        for it in train_items:
            label = it[3]
            if label not in items_by_class:
                items_by_class[label] = []
            items_by_class[label].append(it)

        capped_train_items = []
        for label, class_items in items_by_class.items():
            random.shuffle(class_items)
            capped_items = class_items[:args.max_samples_per_class]
            capped_train_items.extend(capped_items)
            if len(class_items) > args.max_samples_per_class:
                print(f"  Class {CLASSES[label]}: {len(class_items)} -> {len(capped_items)}")

        random.shuffle(capped_train_items)
        train_items = capped_train_items

    # Print class distribution
    train_labels = [it[3] for it in train_items]
    label_counts = Counter(train_labels)
    print(f"\nSamples: train={len(train_items)} val={len(val_items)} test={len(test_items)}")
    print(f"Classes: {CLASSES}")
    print("Training class distribution:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {label_counts.get(i, 0)}")

    # Compute class weights
    # When using balanced sampler, use uniform weights to avoid double-penalizing majority classes
    if args.use_balanced_sampler:
        # Uniform weights - balanced sampler handles class balance
        class_weights = torch.ones(len(CLASSES), device=device)
        # Zero weight for classes with no samples
        for i in range(len(CLASSES)):
            if label_counts.get(i, 0) == 0:
                class_weights[i] = 0.0
        print(f"\nUsing UNIFORM class weights (balanced sampler handles balance)")
    else:
        class_weights = compute_class_weights(train_labels, len(CLASSES)).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    # Create datasets with augmentation
    train_ds = EnhancedPoseDataset(
        train_items, seq_len=args.seq_len, num_people=args.num_people,
        train=True, augment=True, aug_prob=args.aug_prob
    )
    val_ds = EnhancedPoseDataset(
        val_items, seq_len=args.seq_len, num_people=args.num_people,
        train=False, augment=False
    )
    test_ds = EnhancedPoseDataset(
        test_items, seq_len=args.seq_len, num_people=args.num_people,
        train=False, augment=False
    )

    # Create data loaders
    if args.use_balanced_sampler:
        train_sampler = create_balanced_sampler(train_labels, len(CLASSES))
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.num_workers, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Model
    in_dim = train_ds.in_dim
    print(f"Input dimension: {in_dim}")

    # Model selection
    if args.model == "tcn_bilstm":
        model = TCNBiLSTMClassifier(
            in_dim=in_dim,
            hidden=args.hidden,
            num_layers=args.layers,
            num_classes=len(CLASSES),
            dropout=args.dropout,
            num_heads=args.num_heads,
        ).to(device)
        print(f"Using TCN-BiLSTM model (hidden={args.hidden}, layers={args.layers}, heads={args.num_heads})")
    elif args.model == "transformer":
        model = TransformerClassifier(
            in_dim=in_dim,
            hidden=args.hidden,
            num_layers=args.layers,
            num_classes=len(CLASSES),
            dropout=args.dropout,
            num_heads=args.num_heads,
        ).to(device)
        print(f"Using Transformer model (hidden={args.hidden}, layers={args.layers}, heads={args.num_heads})")
    else:
        model = BiLSTMEmotionClassifier(
            in_dim=in_dim,
            hidden=args.hidden,
            num_layers=args.layers,
            num_classes=len(CLASSES),
            dropout=args.dropout,
        ).to(device)
        print(f"Using BiLSTM model (hidden={args.hidden}, layers={args.layers})")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Warmup + Cosine annealing scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function
    if args.focal_loss:
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing
        )
        print(f"Using Focal Loss (gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        print("Using Cross Entropy Loss")

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_acc": [],
        "lr": [],
    }

    best_val_acc = -1.0
    metrics_path = os.path.join(workdir, "metrics.jsonl")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})

        scheduler.step()

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_acc, val_loss, val_class_acc, _, _ = evaluate(model, val_loader, device, criterion)
        test_acc, _, test_class_acc, _, _ = evaluate(model, test_loader, device, criterion)

        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        history["lr"].append(scheduler.get_last_lr()[0])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "val_class_acc": val_class_acc,
            "seconds": dt,
            "lr": scheduler.get_last_lr()[0],
        }
        print(json.dumps(row))

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        # Save checkpoints
        torch.save({"model": model.state_dict(), "args": vars(args), "history": history},
                   os.path.join(workdir, "last.pt"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "args": vars(args), "history": history},
                       os.path.join(workdir, "best.pt"))
            print(f"New best model saved (val_acc={best_val_acc:.4f})")

    # Final evaluation
    print("\n" + "=" * 50)
    print("Final Evaluation on Test Set:")
    checkpoint = torch.load(os.path.join(workdir, "best.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_acc, test_loss, test_class_acc, test_preds, test_labels = evaluate(model, test_loader, device, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("Per-class accuracy:")
    for cls, acc in test_class_acc.items():
        print(f"  {cls}: {acc:.4f}")

    # Save final results with file paths for visualization
    test_paths = [str(item[0]) for item in test_items]
    np.savez(
        os.path.join(workdir, "test_predictions.npz"),
        preds=test_preds,
        labels=test_labels,
        classes=CLASSES,
        paths=test_paths,
    )

    # Save history for plotting
    np.save(os.path.join(workdir, "history.npy"), history)

    print(f"\nTraining complete. Results saved to: {workdir}")


if __name__ == "__main__":
    main()
