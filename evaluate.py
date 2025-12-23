import os
import argparse
import platform
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from caer_dataset import collect_videos, CLASSES
from pose_extract import load_or_extract
from model import BiLSTMSkeletonEmotion

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
            train=False
        )  # (T,33,3)
        x = seq.reshape(seq.shape[0], -1)  # (T, 99)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_y.extend(y.cpu().tolist())
    return np.array(all_y), np.array(all_preds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--workdir", type=str, default="./runs/caer_skeleton")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--target_fps", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=-1)
    args = ap.parse_args()

    cache_dir = os.path.join(args.workdir, "pose_cache")
    ckpt_path = os.path.join(args.workdir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first.")

    if args.num_workers == -1:
        num_workers = 0 if platform.system().lower().startswith("win") else 2
    else:
        num_workers = args.num_workers

    items = collect_videos(args.data_root)
    ds = CAERSkeletonDataset(items, args.split, cache_dir, args.seq_len, args.target_fps)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = BiLSTMSkeletonEmotion(in_dim=33*3, num_classes=len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model"])

    y_true, y_pred = evaluate(model, loader, device)

    acc = accuracy_score(y_true, y_pred)
    print(f"Split: {args.split} | Accuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

if __name__ == "__main__":
    main()
