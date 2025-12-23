import os
import platform
import argparse
import logging
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from caer_dataset import collect_videos, CLASSES
from pose_extract import load_or_extract, precache_all
from model import BiLSTMSkeletonEmotion
from train import CAERSkeletonDataset, run_epoch
from utils import ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--workdir", type=str, default="./runs/caer_skeleton_test100")
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--target_fps", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_items", type=int, default=100, help="Total labeled videos to use (quick test)")
    args = ap.parse_args()

    workdir = ensure_dir(args.workdir)
    cache_dir = ensure_dir(os.path.join(workdir, "pose_cache"))
    ckpt_path = os.path.join(workdir, "best_model.pt")

    logger = logging.getLogger("quick_train")
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    logger.addHandler(sh)

    # gather and limit items
    items = collect_videos(args.data_root)
    if not items:
        raise RuntimeError("No labeled videos found in data_root")
    items = items[: args.num_items]
    logger.info("Using %d items for quick test", len(items))
    logger.info("Split counts: %s", Counter([s for _, s, _, _ in items]))

    # If dataset has no 'train' items (common when CAER labels all as 'test'),
    # create a simple split: 70% train, 15% val, 15% test for the quick run.
    splits = [s for _, s, _, _ in items]
    if not any(s == "train" for s in splits):
        n = len(items)
        n_train = max(1, int(n * 0.7))
        n_val = max(1, int(n * 0.15))
        new_items = []
        for i, (vp, _, lab, idx) in enumerate(items):
            if i < n_train:
                new_items.append((vp, "train", lab, idx))
            elif i < n_train + n_val:
                new_items.append((vp, "val", lab, idx))
            else:
                new_items.append((vp, "test", lab, idx))
        items = new_items
        logger.info("Assigned quick splits: train=%d val=%d test=%d", n_train, n_val, n - n_train - n_val)

    # precache (will reuse existing caches if present)
    precache_all(items, cache_dir, args.seq_len, args.target_fps)

    # DataLoader worker choice (safe on Windows)
    num_workers = 0 if platform.system().lower().startswith("win") else 2

    train_ds = CAERSkeletonDataset(items, "train", cache_dir, args.seq_len, args.target_fps)
    val_ds   = CAERSkeletonDataset(items, "val", cache_dir, args.seq_len, args.target_fps)
    test_ds  = CAERSkeletonDataset(items, "test", cache_dir, args.seq_len, args.target_fps)

    # Device and performance knobs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Build DataLoader kwargs tuned for GPU/CPU
    pin_mem = True if device.type == "cuda" else False
    def make_dl(dataset, shuffle):
        kwargs = dict(batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_mem)
        if num_workers > 0:
            kwargs.update(persistent_workers=True, prefetch_factor=2)
        return DataLoader(dataset, **kwargs)

    train_loader = make_dl(train_ds, shuffle=True)
    val_loader   = make_dl(val_ds, shuffle=False)
    test_loader  = make_dl(test_ds, shuffle=False)

    model = BiLSTMSkeletonEmotion(
        in_dim=33 * 3,
        hidden=256,
        num_layers=2,
        num_classes=len(CLASSES),
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
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

    # final test (if checkpoint exists)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"]) 
        te_loss, te_acc = run_epoch(model, test_loader, optimizer, criterion, device, train=False)
        logger.info("TEST | loss %.4f acc %.4f", te_loss, te_acc)
    else:
        logger.info("No checkpoint saved during quick run.")


if __name__ == "__main__":
    main()
