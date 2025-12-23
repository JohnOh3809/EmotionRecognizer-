import os
import platform
import argparse
from collections import Counter

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from caer_dataset import collect_videos, CLASSES
from pose_extract import load_or_extract, precache_all
from model import BiLSTMSkeletonEmotion
from utils import ensure_dir

class CAERSkeletonDataset(Dataset):
    def __init__(self, items, split: str, cache_dir: str, seq_len: int, target_fps: int):
        self.rows = [(vp, y) for (vp, sp, _, y) in items if sp == split]
        self.split = split
        self.cache_dir = cache_dir
        self.seq_len = seq_len
        self.target_fps = target_fps

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        video_path, y = self.rows[idx]
        seq = load_or_extract(
            cache_dir=self.cache_dir,
            video_path=video_path,
            seq_len=self.seq_len,
            target_fps=self.target_fps,
            train=(self.split == "train")
        )  # (T,33,3)

        x = seq.reshape(seq.shape[0], -1)  # (T, 33*3)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    all_preds, all_y = [], []
    total_loss = 0.0

    for x, y in tqdm(loader, leave=False):
        # prefer non-blocking transfers when pin_memory=True on the DataLoader
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        # During evaluation we don't need gradients — wrap to avoid autograd overhead
        if not train:
            with torch.no_grad():
                logits = model(x)
                loss = criterion(logits, y)
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_y.extend(y.detach().cpu().tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = accuracy_score(all_y, all_preds) if len(all_y) else 0.0
    return avg_loss, acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=False, default=None, help="Path to extracted CAER dataset root. If omitted, the script will try to auto-detect common locations (./CAER, ./data/CAER or env var CAER_ROOT).")
    ap.add_argument("--workdir", type=str, default="./runs/caer_skeleton", help="Where to save caches and checkpoints.")
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--target_fps", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--num_workers", type=int, default=-1, help="-1 => auto safe default by OS")
    ap.add_argument("--logfile", type=str, default=None, help="Optional path to write training logs to")
    ap.add_argument("--precache_only", action="store_true", help="Only extract/cache pose then exit.")
    args = ap.parse_args()

    workdir = ensure_dir(args.workdir)
    cache_dir = ensure_dir(os.path.join(workdir, "pose_cache"))
    ckpt_path = os.path.join(workdir, "best_model.pt")

    # Logging: console + optional file
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if args.logfile:
        fh = logging.FileHandler(args.logfile, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Safer DataLoader defaults on Windows (multiprocessing spawn issues)
    if args.num_workers == -1:
        num_workers = 0 if platform.system().lower().startswith("win") else 2
    else:
        num_workers = args.num_workers

    # Auto-detect data_root if not provided
    if args.data_root is None:
        candidates = []
        env_root = os.environ.get("CAER_ROOT")
        if env_root:
            candidates.append(env_root)
        candidates.extend(["./CAER", "./data/CAER", "./caer/CAER", "./dataset/CAER"])
        found = None
        for c in candidates:
            if c and os.path.isdir(c):
                # quick check for split dirs
                if any(os.path.isdir(os.path.join(c, x)) for x in ("train", "val", "validation", "test")):
                    found = c
                    break
        if found is None:
            raise RuntimeError("--data_root not provided and no dataset found in common locations. Set --data_root or CAER_ROOT env var.")
        args.data_root = found
    items = collect_videos(args.data_root)
    if not items:
        raise RuntimeError(
            "No labeled videos found. Check your CAER folder layout and --data_root path. "
            "Expected directories containing train/validation(or val)/test and emotion labels."
        )

    logger.info("Total labeled videos: %d", len(items))
    logger.info("Split counts: %s", Counter([s for _, s, _, _ in items]))
    logger.info("Label counts: %s", Counter([lab for _, _, lab, _ in items]))

    if args.precache_only:
        precache_all(items, cache_dir, args.seq_len, args.target_fps)
        logger.info("Pre-cache complete: %s", cache_dir)
        return

    train_ds = CAERSkeletonDataset(items, "train", cache_dir, args.seq_len, args.target_fps)
    val_ds   = CAERSkeletonDataset(items, "val", cache_dir, args.seq_len, args.target_fps)
    test_ds  = CAERSkeletonDataset(items, "test", cache_dir, args.seq_len, args.target_fps)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = BiLSTMSkeletonEmotion(
        in_dim=33 * 3,
        hidden=args.hidden,
        num_layers=args.num_layers,
        num_classes=len(CLASSES),
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        logger.info("Epoch %02d | train loss %.4f acc %.4f | val loss %.4f acc %.4f", epoch, tr_loss, tr_acc, va_loss, va_acc)

        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model": model.state_dict(), "classes": CLASSES}, ckpt_path)
            logger.info("Saved best checkpoint -> %s", ckpt_path)

    # Test
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te_loss, te_acc = run_epoch(model, test_loader, optimizer, criterion, device, train=False)
    logger.info("TEST | loss %.4f acc %.4f", te_loss, te_acc)

if __name__ == "__main__":
    main()
