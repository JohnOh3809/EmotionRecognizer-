#!/usr/bin/env python3
"""
emotion_pipeline_yolopose17_full.py

A full, **server-friendly** skeleton-emotion pipeline built around
**Ultralytics YOLO Pose (COCO-17 keypoints)**.

This replaces MediaPipe/OpenPose dependencies with a single PyTorch stack:
  - Filtering/extraction: use filtering_yolopose17_sharded.py
  - Enhancement/kinematics: use enhance_yolopose17_kinematics.py
  - Training: this script (TCN/LSTM + k-fold + ensembling + visualizations)
  - Optional: SimCLR-style pretraining + GBDT baseline on aggregated kinematic features

Dataset layout expected
-----------------------
After extraction OR enhancement, put all samples into:
  data_dir/<label>/*.npz

Where each .npz has:
  - keypoints: (T, K, 17, 3) OR (T,17,3)
Optionally (if enhanced):
  - vel: (T, K, 17, 2)
  - acc: (T, K, 17, 2)
  - time: (T,)

Commands
--------
1) Train neural model with k-fold and class-wise validation plots:
    python metric.py train \
      --data_dir ../data_kin_yolo \
      --arch tcn \
      --feat_mode pos_vel_acc_score \
      --seq_len 64 \
      --keep_people 2 \
      --k_folds 5 \
      --test_frac 0.15 \
      --epochs 60 \
      --batch_size 128 \
      --lr 3e-4 \
      --weight_decay 1e-3 \
      --scheduler cosine \
      --augment 1 \
      --run_dir ../runs_yolo

2) SimCLR-style pretrain + linear probe (tiny):
    python emotion_pipeline_yolopose17_full.py pretrain_simclr \
      --data_dir ../data_kin_yolo \
      --seq_len 64 \
      --keep_people 2 \
      --epochs 200 \
      --batch_size 256 \
      --run_dir ../runs_yolo

3) GBDT baseline on aggregated features:
    python emotion_pipeline_yolopose17_full.py train_gbdt \
      --data_dir ../data_kin_yolo \
      --keep_people 2 \
      --run_dir ../runs_yolo

Notes
-----
- This script is intentionally small-model friendly to reduce overfitting.
- Uses k-fold on the TRAIN split; holds out a fixed TEST split.
- Saves best checkpoints with date/time stamps (per fold).
- Writes per-class confusion matrix + per-class accuracy plots.

Install deps
------------
pip install -U numpy scikit-learn matplotlib joblib
(Plus torch, which you already have in your venv.)
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Repro helpers
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Data loading
# -----------------------------
J = 17

COCO_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

def list_samples(data_dir: Path) -> Tuple[List[Path], List[str]]:
    files: List[Path] = []
    labels: List[str] = []
    for lab_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        lab = lab_dir.name
        for f in sorted(lab_dir.glob("*.npz")):
            files.append(f)
            labels.append(lab)
    if not files:
        raise RuntimeError(f"No samples found under {data_dir}/<label>/*.npz")
    return files, labels

def build_label_map(labels: List[str]) -> Tuple[Dict[str, int], List[str]]:
    uniq = sorted(set(labels))
    return {l: i for i, l in enumerate(uniq)}, uniq

def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        kps = z["keypoints"].astype(np.float32)
        vel = z["vel"].astype(np.float32) if "vel" in z else None
        acc = z["acc"].astype(np.float32) if "acc" in z else None
        time = z["time"].astype(np.float32) if "time" in z else None
    # normalize kps shape
    if kps.ndim == 3 and kps.shape[1] == J and kps.shape[2] == 3:
        kps = kps[:, None, :, :]  # (T,1,17,3)
    if kps.ndim != 4 or kps.shape[2] != J or kps.shape[3] != 3:
        raise ValueError(f"Bad keypoints shape {kps.shape} in {path}")
    if vel is not None and vel.ndim == 3:
        vel = vel[:, None, :, :]
    if acc is not None and acc.ndim == 3:
        acc = acc[:, None, :, :]
    return kps, vel, acc, time

def center_scale_rotate_xy(xy: np.ndarray, score: np.ndarray, min_score: float = 0.1) -> np.ndarray:
    """
    Lightweight normalization for training-time augment / fallback:
      - center by mid-hip if possible else mean of valid joints
      - scale by torso length (mid-shoulder to mid-hip) with RMS fallback
      - rotate shoulders horizontal

    xy: (T,K,17,2), score: (T,K,17)
    returns xy_norm same shape
    """
    # indices
    L_SH, R_SH = 5, 6
    L_HIP, R_HIP = 11, 12

    m = np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1]) & np.isfinite(score) & (score >= min_score)
    T, K, _, _ = xy.shape

    # mean center fallback
    w = m.astype(np.float32)[..., None]  # (T,K,J,1)
    mean_center = (xy * w).sum(axis=2) / np.maximum(w.sum(axis=2), 1e-6)  # (T,K,2)

    hip_ok = m[:, :, L_HIP] & m[:, :, R_HIP]
    center = mean_center.copy()
    center[hip_ok] = 0.5 * (xy[hip_ok, L_HIP] + xy[hip_ok, R_HIP])

    xy_c = xy - center[:, :, None, :]

    sh_ok = m[:, :, L_SH] & m[:, :, R_SH]
    mid_sh = mean_center.copy()
    mid_sh[sh_ok] = 0.5 * (xy[sh_ok, L_SH] + xy[sh_ok, R_SH])

    torso = np.linalg.norm(mid_sh - center, axis=2)  # (T,K)
    rad2 = xy_c[..., 0] ** 2 + xy_c[..., 1] ** 2
    rad2 = np.where(m, rad2, np.nan)
    rms = np.sqrt(np.nanmean(rad2, axis=2))
    rms = np.where(np.isfinite(rms), rms, 1.0)
    torso = np.where(np.isfinite(torso) & (torso > 1e-6), torso, rms)
    torso = np.maximum(torso, 1.0)
    xy_c = xy_c / torso[:, :, None, None]

    # rotate
    vec = xy_c[:, :, L_SH] - xy_c[:, :, R_SH]  # (T,K,2)
    ang = np.arctan2(vec[..., 1], vec[..., 0])  # (T,K)
    ca = np.cos(-ang).astype(np.float32)
    sa = np.sin(-ang).astype(np.float32)

    x = xy_c[..., 0]
    y = xy_c[..., 1]
    xr = x * ca[:, :, None] - y * sa[:, :, None]
    yr = x * sa[:, :, None] + y * ca[:, :, None]
    xy_r = np.stack([xr, yr], axis=3).astype(np.float32)
    xy_c = np.where(sh_ok[:, :, None, None], xy_r, xy_c)

    xy_c = np.where(m[:, :, :, None], xy_c, 0.0).astype(np.float32)
    return xy_c


@dataclass
class AugCfg:
    rot_deg: float = 8.0
    scale_jitter: float = 0.08
    noise_std: float = 0.01
    drop_joint_prob: float = 0.02
    time_mask_prob: float = 0.05
    blur_prob: float = 0.25
    blur_kernel: int = 5  # must be odd

def apply_aug(features: torch.Tensor, aug: AugCfg) -> torch.Tensor:
    """
    features: (B,T,D)
    Augment in feature space (already normalized coordinates assumed).
    Includes:
      - gaussian noise
      - random joint dropout (approx via feature dropout)
      - temporal blur (1D conv smoothing)
      - time masking
    """
    B, T, D = features.shape
    x = features

    # gaussian noise
    if aug.noise_std > 0:
        x = x + torch.randn_like(x) * aug.noise_std

    # feature dropout (proxy for joint dropout)
    if aug.drop_joint_prob > 0:
        drop = torch.rand((B, 1, D), device=x.device) < aug.drop_joint_prob
        x = torch.where(drop, torch.zeros_like(x), x)

    # temporal blur (gaussian-like smoothing kernel)
    if aug.blur_prob > 0 and aug.blur_kernel >= 3 and (aug.blur_kernel % 2 == 1):
        if torch.rand(()) < aug.blur_prob:
            k = aug.blur_kernel
            # simple box blur kernel normalized
            kernel = torch.ones((1, 1, k), device=x.device) / float(k)
            # apply per feature dim with groups=D (depthwise conv1d)
            x_t = x.transpose(1, 2)  # (B,D,T)
            x_t = x_t.reshape(B * D, 1, T)
            x_t = F.pad(x_t, (k // 2, k // 2), mode="replicate")
            x_t = F.conv1d(x_t, kernel)
            x = x_t.reshape(B, D, T).transpose(1, 2)

    # time masking
    if aug.time_mask_prob > 0:
        mask = torch.rand((B, T, 1), device=x.device) < aug.time_mask_prob
        x = torch.where(mask, torch.zeros_like(x), x)

    return x


class PoseSeqDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        files: List[Path],
        y: np.ndarray,
        label_names: List[str],
        seq_len: int,
        keep_people: int,
        feat_mode: str,
        train: bool,
        augment: bool,
        seed: int,
    ):
        self.files = files
        self.y = y.astype(np.int64)
        self.label_names = label_names
        self.seq_len = int(seq_len)
        self.keep_people = int(keep_people)
        self.feat_mode = str(feat_mode)
        self.train = bool(train)
        self.augment = bool(augment)
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.files)

    def _to_features(self, kps: np.ndarray, vel: Optional[np.ndarray], acc: Optional[np.ndarray]) -> np.ndarray:
        """
        Build per-frame features (T, D).
        kps: (T,K,17,3) normalized xy recommended but not assumed.
        vel/acc optional.
        Supported feat_mode:
          pos
          pos_score
          pos_vel
          pos_vel_score
          pos_vel_acc
          pos_vel_acc_score
        """
        T, K, Jj, C = kps.shape
        K = min(K, self.keep_people)
        kps = kps[:, :K]
        score = kps[..., 2]  # (T,K,J)
        xy = kps[..., 0:2]   # (T,K,J,2)

        # training-time lightweight normalization to reduce identity / camera drift
        xy = center_scale_rotate_xy(xy, score, min_score=0.1)  # (T,K,J,2)

        parts: List[np.ndarray] = []

        if "pos" in self.feat_mode:
            parts.append(xy.reshape(T, -1))  # (T, K*J*2)

        if "score" in self.feat_mode:
            parts.append(np.nan_to_num(score, nan=0.0).reshape(T, -1))  # (T, K*J)

        if "vel" in self.feat_mode:
            if vel is None:
                # derive simple finite difference on xy (dt=1) as fallback
                v = np.zeros_like(xy)
                v[1:] = xy[1:] - xy[:-1]
            else:
                v = vel[:, :K]  # (T,K,J,2)
            parts.append(np.nan_to_num(v, nan=0.0).reshape(T, -1))

        if "acc" in self.feat_mode:
            if acc is None:
                a = np.zeros_like(xy)
                a[2:] = (xy[2:] - xy[1:-1]) - (xy[1:-1] - xy[:-2])
            else:
                a = acc[:, :K]
            parts.append(np.nan_to_num(a, nan=0.0).reshape(T, -1))

        feat = np.concatenate(parts, axis=1).astype(np.float32)  # (T,D)
        return feat

    def __getitem__(self, idx: int):
        path = self.files[idx]
        kps, vel, acc, _time = load_npz(path)
        feat = self._to_features(kps, vel, acc)  # (T,D)
        T = feat.shape[0]

        L = self.seq_len
        if T >= L:
            if self.train:
                s = int(self.rng.randint(0, T - L + 1))
            else:
                s = int((T - L) // 2)
            x = feat[s:s+L]
        else:
            # pad with last frame
            pad = np.repeat(feat[-1:], L - T, axis=0)
            x = np.concatenate([feat, pad], axis=0)

        return torch.from_numpy(x), int(self.y[idx])


# -----------------------------
# Models
# -----------------------------
def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class TinyTCN(nn.Module):
    """
    Small TCN (causal-ish) with residual blocks.
    Input: (B,T,D)
    """
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 64, layers: int = 3, k: int = 3, dropout: float = 0.4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        blocks = []
        for i in range(layers):
            dilation = 2 ** i
            blocks.append(TCNBlock(hidden, hidden, k=k, dilation=dilation, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        h = self.in_proj(x)
        for b in self.blocks:
            h = b(h)
        h = self.norm(h)
        # global average pool over time
        h = h.mean(dim=1)
        return self.head(h)


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int, dropout: float):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C) -> (B,C,T)
        xt = x.transpose(1, 2)
        h = self.conv1(xt)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.drop(h)
        res = xt if self.proj is None else self.proj(xt)
        y = h + res
        return y.transpose(1, 2)


class TinyLSTM(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128, layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=(dropout if layers > 1 else 0.0),
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        h = self.norm(h)
        h = h.mean(dim=1)
        return self.head(h)


# -----------------------------
# Training utilities
# -----------------------------
@dataclass
class TrainCfg:
    arch: str
    feat_mode: str
    seq_len: int
    keep_people: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    label_smoothing: float
    scheduler: str
    warmup_epochs: int
    patience: int
    augment: bool
    seed: int
    device: str

    # regularization
    dropout: float
    hidden: int
    layers: int
    kernel: int


def make_model(cfg: TrainCfg, in_dim: int, n_classes: int) -> nn.Module:
    if cfg.arch == "tcn":
        return TinyTCN(in_dim=in_dim, n_classes=n_classes, hidden=cfg.hidden, layers=cfg.layers, k=cfg.kernel, dropout=cfg.dropout)
    if cfg.arch == "lstm":
        return TinyLSTM(in_dim=in_dim, n_classes=n_classes, hidden=cfg.hidden, layers=max(1, cfg.layers // 2), dropout=cfg.dropout * 0.5)
    raise ValueError(f"Unknown arch {cfg.arch}")

def make_scheduler(cfg: TrainCfg, optim: torch.optim.Optimizer):
    if cfg.scheduler == "none":
        return None
    if cfg.scheduler == "cosine":
        # cosine with warmup: implement manually in loop via LambdaLR
        def lr_lambda(epoch: int):
            if epoch < cfg.warmup_epochs:
                return float(epoch + 1) / float(max(1, cfg.warmup_epochs))
            t = (epoch - cfg.warmup_epochs) / float(max(1, cfg.epochs - cfg.warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * t))
        import math
        return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optim, step_size=max(1, cfg.epochs // 3), gamma=0.3)
    raise ValueError(f"Unknown scheduler {cfg.scheduler}")

@torch.no_grad()
def evaluate(model: nn.Module, dl, device: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    ys = []
    ps = []
    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        losses.append(float(loss.item()))
        pred = logits.argmax(dim=1)
        ys.append(yb.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.zeros((0,), np.int64)
    y_pred = np.concatenate(ps) if ps else np.zeros((0,), np.int64)
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    return float(np.mean(losses) if losses else 0.0), acc, y_true, y_pred

def plot_confusion(cm: np.ndarray, class_names: List[str], out_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def plot_per_class_acc(cm: np.ndarray, class_names: List[str], out_path: Path, title: str) -> None:
    per = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(111)
    ax.bar(range(len(class_names)), per)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def train_one_fold(
    fold_dir: Path,
    cfg: TrainCfg,
    train_files: List[Path],
    train_y: np.ndarray,
    val_files: List[Path],
    val_y: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    set_seed(cfg.seed)

    # dataset to infer in_dim
    tmp_ds = PoseSeqDataset(train_files[:1], train_y[:1], class_names, cfg.seq_len, cfg.keep_people, cfg.feat_mode,
                            train=True, augment=False, seed=cfg.seed)
    x0, _ = tmp_ds[0]
    in_dim = int(x0.shape[1])
    n_classes = len(class_names)

    ds_tr = PoseSeqDataset(train_files, train_y, class_names, cfg.seq_len, cfg.keep_people, cfg.feat_mode,
                           train=True, augment=cfg.augment, seed=cfg.seed)
    ds_va = PoseSeqDataset(val_files, val_y, class_names, cfg.seq_len, cfg.keep_people, cfg.feat_mode,
                           train=False, augment=False, seed=cfg.seed)

    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = make_model(cfg, in_dim=in_dim, n_classes=n_classes).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = make_scheduler(cfg, optim)

    aug_cfg = AugCfg()
    best = {"val_acc": -1.0, "val_loss": 1e9, "epoch": -1}
    best_path = None
    bad = 0

    # training log
    log_path = fold_dir / "log.jsonl"
    with open(log_path, "w", encoding="utf-8") as lf:
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            losses = []
            correct = 0
            total = 0

            for xb, yb in dl_tr:
                xb = xb.to(cfg.device, non_blocking=True)
                yb = yb.to(cfg.device, non_blocking=True)

                if cfg.augment:
                    xb = apply_aug(xb, aug_cfg)

                logits = model(xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=cfg.label_smoothing)

                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

                losses.append(float(loss.item()))
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())

            if sched is not None:
                sched.step()

            tr_loss = float(np.mean(losses)) if losses else 0.0
            tr_acc = float(correct / max(1, total))

            va_loss, va_acc, y_true, y_pred = evaluate(model, dl_va, cfg.device)

            # best tracking on val acc (primary), then loss
            improved = (va_acc > best["val_acc"] + 1e-6) or (va_acc >= best["val_acc"] - 1e-6 and va_loss < best["val_loss"] - 1e-4)
            if improved:
                best = {"val_acc": float(va_acc), "val_loss": float(va_loss), "epoch": int(epoch)}
                bad = 0
                # save checkpoint with timestamp
                ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                best_path = fold_dir / f"best_{ts}_epoch{epoch:03d}_acc{va_acc:.3f}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "cfg": dataclasses.asdict(cfg),
                    "in_dim": in_dim,
                    "class_names": class_names,
                }, best_path)
            else:
                bad += 1

            # plots every epoch? too expensive; do every 5
            if epoch == 1 or epoch % 5 == 0 or improved:
                cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
                plot_confusion(cm, class_names, fold_dir / f"conf_epoch{epoch:03d}.png", f"Fold val confusion epoch {epoch}")
                plot_per_class_acc(cm, class_names, fold_dir / f"perclass_epoch{epoch:03d}.png", f"Fold val per-class acc epoch {epoch}")

            rec = {
                "epoch": epoch,
                "tr_loss": tr_loss,
                "tr_acc": tr_acc,
                "va_loss": float(va_loss),
                "va_acc": float(va_acc),
                "lr": float(optim.param_groups[0]["lr"]),
                "best_val_acc": float(best["val_acc"]),
                "best_epoch": int(best["epoch"]),
            }
            lf.write(json.dumps(rec) + "\n")
            lf.flush()

            print(f"epoch {epoch:03d} | tr {tr_loss:.4f} acc {tr_acc:.3f} | va {va_loss:.4f} acc {va_acc:.3f} | lr {optim.param_groups[0]['lr']:.2e}")

            if bad >= cfg.patience:
                print("[INFO] early stopping")
                break

    # final fold summary
    summary = {
        "best_val_acc": float(best["val_acc"]),
        "best_val_loss": float(best["val_loss"]),
        "best_epoch": int(best["epoch"]),
        "best_path": str(best_path) if best_path else "",
        "params": int(count_params(model)),
        "in_dim": int(in_dim),
    }
    with open(fold_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def train_kfold(
    data_dir: Path,
    run_dir: Path,
    cfg: TrainCfg,
    k_folds: int,
    test_frac: float,
) -> None:
    files, labels = list_samples(data_dir)
    lab2id, class_names = build_label_map(labels)
    y = np.array([lab2id[l] for l in labels], dtype=np.int64)

    # fixed test holdout
    idx = np.arange(len(files))
    tr_idx, te_idx = train_test_split(idx, test_size=float(test_frac), random_state=cfg.seed, stratify=y)
    files_tr = [files[i] for i in tr_idx]
    y_tr = y[tr_idx]
    files_te = [files[i] for i in te_idx]
    y_te = y[te_idx]

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run = run_dir / f"train_{cfg.arch}_{ts}"
    run.mkdir(parents=True, exist_ok=True)

    with open(run / "run_cfg.json", "w", encoding="utf-8") as f:
        json.dump({"cfg": dataclasses.asdict(cfg), "k_folds": int(k_folds), "test_frac": float(test_frac), "classes": class_names}, f, indent=2)

    print(f"[INFO] classes={len(class_names)} | train={len(files_tr)} | test={len(files_te)} | run={run}")
    print(f"[INFO] feat_mode={cfg.feat_mode} seq_len={cfg.seq_len} keep_people={cfg.keep_people} arch={cfg.arch}")

    skf = StratifiedKFold(n_splits=int(k_folds), shuffle=True, random_state=cfg.seed)
    fold_summaries = []

    for fold, (a, b) in enumerate(skf.split(np.zeros_like(y_tr), y_tr)):
        fold_dir = run / f"fold{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        tr_files = [files_tr[i] for i in a]
        tr_y = y_tr[a]
        va_files = [files_tr[i] for i in b]
        va_y = y_tr[b]

        print(f"\n=== Fold {fold:02d} | train={len(tr_files)} val={len(va_files)} ===")
        summ = train_one_fold(fold_dir, cfg, tr_files, tr_y, va_files, va_y, class_names)
        summ["fold"] = int(fold)
        fold_summaries.append(summ)

    # save fold summaries
    with open(run / "fold_summaries.json", "w", encoding="utf-8") as f:
        json.dump(fold_summaries, f, indent=2)

    # simple ensemble eval on test using best checkpoint of each fold
    print("\n=== Ensemble eval on TEST ===")
    preds_all = []
    y_true_all = []

    # build test loader once
    ds_te = PoseSeqDataset(files_te, y_te, class_names, cfg.seq_len, cfg.keep_people, cfg.feat_mode, train=False, augment=False, seed=cfg.seed)
    dl_te = torch.utils.data.DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    for summ in fold_summaries:
        ckpt_path = summ.get("best_path", "")
        if not ckpt_path or not Path(ckpt_path).exists():
            continue
        ck = torch.load(ckpt_path, map_location=cfg.device)
        in_dim = int(ck["in_dim"])
        model = make_model(cfg, in_dim=in_dim, n_classes=len(class_names)).to(cfg.device)
        model.load_state_dict(ck["model"], strict=True)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for xb, yb in dl_te:
                xb = xb.to(cfg.device)
                logits = model(xb)
                fold_preds.append(F.softmax(logits, dim=1).detach().cpu().numpy())
        preds_all.append(np.concatenate(fold_preds, axis=0))

    if not preds_all:
        print("[WARN] no fold checkpoints found for ensemble")
        return

    prob = np.mean(np.stack(preds_all, axis=0), axis=0)
    y_pred = prob.argmax(axis=1)
    y_true = y_te
    acc = float((y_true == y_pred).mean())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plot_confusion(cm, class_names, run / "test_confusion.png", f"Ensemble TEST confusion | acc={acc:.3f}")
    plot_per_class_acc(cm, class_names, run / "test_perclass.png", f"Ensemble TEST per-class acc | acc={acc:.3f}")

    with open(run / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump({"test_acc": acc, "classes": class_names}, f, indent=2)

    print(f"[OK] TEST acc={acc:.3f} | plots: {run/'test_confusion.png'}")


# -----------------------------
# SimCLR-lite (optional)
# -----------------------------
class SimCLRProj(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B,D)
    sim = (z @ z.t()) / tau
    sim = sim - torch.eye(2*B, device=z.device) * 1e9
    # positives: i<->i+B
    pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)
    denom = torch.logsumexp(sim, dim=1)
    loss = (-pos + denom).mean()
    return loss

def pretrain_simclr(
    data_dir: Path,
    run_dir: Path,
    seq_len: int,
    keep_people: int,
    feat_mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
) -> None:
    files, labels = list_samples(data_dir)
    lab2id, class_names = build_label_map(labels)
    y = np.array([lab2id[l] for l in labels], dtype=np.int64)

    ds = PoseSeqDataset(files, y, class_names, seq_len, keep_people, feat_mode, train=True, augment=True, seed=seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    # infer in_dim
    x0, _ = ds[0]
    in_dim = int(x0.shape[1])

    # encoder: tiny TCN backbone returning pooled feature
    backbone = TinyTCN(in_dim=in_dim, n_classes=64, hidden=64, layers=3, k=3, dropout=0.4).to(device)
    # replace head with identity for embeddings
    backbone.head = nn.Identity()
    proj = SimCLRProj(in_dim=64, emb_dim=128).to(device)

    optim = torch.optim.AdamW(list(backbone.parameters()) + list(proj.parameters()), lr=lr, weight_decay=1e-4)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run = run_dir / f"simclr_{ts}"
    run.mkdir(parents=True, exist_ok=True)

    aug_cfg = AugCfg()

    for ep in range(1, epochs + 1):
        backbone.train()
        proj.train()
        losses = []
        for xb, _yb in dl:
            xb = xb.to(device)
            x1 = apply_aug(xb, aug_cfg)
            x2 = apply_aug(xb, aug_cfg)

            h1 = backbone(x1)  # (B,64)
            h2 = backbone(x2)

            z1 = proj(h1)
            z2 = proj(h2)

            loss = nt_xent(z1, z2, tau=0.2)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(proj.parameters()), 1.0)
            optim.step()

            losses.append(float(loss.item()))

        if ep == 1 or ep % 10 == 0:
            print(f"simclr epoch {ep:03d} | loss {np.mean(losses):.4f}")
            torch.save({"backbone": backbone.state_dict(), "proj": proj.state_dict(), "in_dim": in_dim, "feat_mode": feat_mode, "seq_len": seq_len, "keep_people": keep_people},
                       run / f"simclr_epoch{ep:03d}.pt")

    print(f"[OK] simclr run: {run}")


# -----------------------------
# GBDT baseline (aggregated features)
# -----------------------------
def aggregate_features(path: Path, keep_people: int, min_score: float = 0.1) -> np.ndarray:
    kps, vel, acc, _t = load_npz(path)
    T, K, _, _ = kps.shape
    K = min(K, keep_people)
    xy = kps[:, :K, :, 0:2]
    sc = kps[:, :K, :, 2]

    xy = center_scale_rotate_xy(xy, sc, min_score=min_score)

    feats = []
    # positions stats
    feats.append(np.nanmean(xy, axis=(0, 1)).reshape(-1))
    feats.append(np.nanstd(xy, axis=(0, 1)).reshape(-1))

    if vel is not None:
        v = vel[:, :K]
        feats.append(np.nanmean(v, axis=(0, 1)).reshape(-1))
        feats.append(np.nanstd(v, axis=(0, 1)).reshape(-1))
    if acc is not None:
        a = acc[:, :K]
        feats.append(np.nanmean(a, axis=(0, 1)).reshape(-1))
        feats.append(np.nanstd(a, axis=(0, 1)).reshape(-1))

    # joint confidence stats
    feats.append(np.nanmean(sc, axis=(0, 1)).reshape(-1))
    feats.append(np.nanstd(sc, axis=(0, 1)).reshape(-1))

    return np.concatenate(feats, axis=0).astype(np.float32)


def train_gbdt(data_dir: Path, run_dir: Path, keep_people: int, seed: int) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    files, labels = list_samples(data_dir)
    lab2id, class_names = build_label_map(labels)
    y = np.array([lab2id[l] for l in labels], dtype=np.int64)

    X = np.stack([aggregate_features(p, keep_people=keep_people) for p in tqdm(files, desc="GBDT feats")], axis=0)
    idx = np.arange(len(files))
    tr_idx, te_idx = train_test_split(idx, test_size=0.15, random_state=seed, stratify=y)

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("gbdt", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, random_state=seed)),
    ])
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    acc = float((yte == ypred).mean())

    cm = confusion_matrix(yte, ypred, labels=list(range(len(class_names))))

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run = run_dir / f"gbdt_{ts}"
    run.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, class_names, run / "confusion.png", f"GBDT TEST confusion | acc={acc:.3f}")
    plot_per_class_acc(cm, class_names, run / "perclass.png", f"GBDT TEST per-class acc | acc={acc:.3f}")

    joblib.dump({"model": clf, "class_names": class_names}, run / "gbdt.joblib")
    with open(run / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"test_acc": acc, "classes": class_names, "X_dim": int(X.shape[1])}, f, indent=2)

    print(f"[OK] GBDT test acc={acc:.3f} | run={run}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_tr = sub.add_parser("train")
    ap_tr.add_argument("--data_dir", type=str, required=True)
    ap_tr.add_argument("--run_dir", type=str, default="./runs_yolo")
    ap_tr.add_argument("--arch", type=str, default="tcn", choices=["tcn", "lstm"])
    ap_tr.add_argument("--feat_mode", type=str, default="pos_vel_acc_score",
                       choices=["pos", "pos_score", "pos_vel", "pos_vel_score", "pos_vel_acc", "pos_vel_acc_score"])
    ap_tr.add_argument("--seq_len", type=int, default=64)
    ap_tr.add_argument("--keep_people", type=int, default=2)
    ap_tr.add_argument("--k_folds", type=int, default=5)
    ap_tr.add_argument("--test_frac", type=float, default=0.15)
    ap_tr.add_argument("--epochs", type=int, default=60)
    ap_tr.add_argument("--batch_size", type=int, default=128)
    ap_tr.add_argument("--lr", type=float, default=3e-4)
    ap_tr.add_argument("--weight_decay", type=float, default=1e-3)
    ap_tr.add_argument("--label_smoothing", type=float, default=0.08)
    ap_tr.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    ap_tr.add_argument("--warmup_epochs", type=int, default=3)
    ap_tr.add_argument("--patience", type=int, default=10)
    ap_tr.add_argument("--augment", type=int, default=1)
    ap_tr.add_argument("--seed", type=int, default=1337)
    ap_tr.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    # model size knobs
    ap_tr.add_argument("--dropout", type=float, default=0.45)
    ap_tr.add_argument("--hidden", type=int, default=64)
    ap_tr.add_argument("--layers", type=int, default=3)
    ap_tr.add_argument("--kernel", type=int, default=3)

    ap_sc = sub.add_parser("pretrain_simclr")
    ap_sc.add_argument("--data_dir", type=str, required=True)
    ap_sc.add_argument("--run_dir", type=str, default="./runs_yolo")
    ap_sc.add_argument("--seq_len", type=int, default=64)
    ap_sc.add_argument("--keep_people", type=int, default=2)
    ap_sc.add_argument("--feat_mode", type=str, default="pos_vel_acc_score")
    ap_sc.add_argument("--epochs", type=int, default=200)
    ap_sc.add_argument("--batch_size", type=int, default=256)
    ap_sc.add_argument("--lr", type=float, default=3e-4)
    ap_sc.add_argument("--seed", type=int, default=1337)
    ap_sc.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    ap_gb = sub.add_parser("train_gbdt")
    ap_gb.add_argument("--data_dir", type=str, required=True)
    ap_gb.add_argument("--run_dir", type=str, default="./runs_yolo")
    ap_gb.add_argument("--keep_people", type=int, default=2)
    ap_gb.add_argument("--seed", type=int, default=1337)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cmd = args.cmd

    if cmd == "train":
        cfg = TrainCfg(
            arch=args.arch,
            feat_mode=args.feat_mode,
            seq_len=int(args.seq_len),
            keep_people=int(args.keep_people),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            label_smoothing=float(args.label_smoothing),
            scheduler=str(args.scheduler),
            warmup_epochs=int(args.warmup_epochs),
            patience=int(args.patience),
            augment=bool(int(args.augment)),
            seed=int(args.seed),
            device=str(args.device),
            dropout=float(args.dropout),
            hidden=int(args.hidden),
            layers=int(args.layers),
            kernel=int(args.kernel),
        )
        train_kfold(Path(args.data_dir).expanduser().resolve(),
                    Path(args.run_dir).expanduser().resolve(),
                    cfg, k_folds=int(args.k_folds), test_frac=float(args.test_frac))
        return

    if cmd == "pretrain_simclr":
        pretrain_simclr(Path(args.data_dir).expanduser().resolve(),
                        Path(args.run_dir).expanduser().resolve(),
                        seq_len=int(args.seq_len),
                        keep_people=int(args.keep_people),
                        feat_mode=str(args.feat_mode),
                        epochs=int(args.epochs),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        device=str(args.device),
                        seed=int(args.seed))
        return

    if cmd == "train_gbdt":
        train_gbdt(Path(args.data_dir).expanduser().resolve(),
                   Path(args.run_dir).expanduser().resolve(),
                   keep_people=int(args.keep_people),
                   seed=int(args.seed))
        return

    raise RuntimeError(f"Unknown cmd {cmd}")


if __name__ == "__main__":
    main()