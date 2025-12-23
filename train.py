
import os, json, time, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from caer_dataset import collect_videos, CLASSES
from pose_extract import load_or_extract, precache_all
from model import BiLSTMSkeletonEmotion

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CAERSkeletonDataset(Dataset):
    def __init__(self, items, cache_dir, seq_len, target_fps, train):
        self.items = items
        self.cache_dir = cache_dir
        self.seq_len = seq_len
        self.target_fps = target_fps
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        vp, split, label_str, y = self.items[idx]
        try:
            seq = load_or_extract(self.cache_dir, vp, self.seq_len, self.target_fps, train=self.train)
            # seq: (T,33,3) -> flatten joints to D=99
            x = seq.reshape(self.seq_len, -1).astype(np.float32)
        except Exception as e:
            # Robust fallback: return zeros but keep label (so training continues)
            x = np.zeros((self.seq_len, 33*3), dtype=np.float32)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--workdir", default="./runs/caer_skeleton")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--target_fps", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--precache_only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Optional: limit total videos for quick test")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    cache_dir = os.path.join(args.workdir, "pose_cache")
    os.makedirs(cache_dir, exist_ok=True)

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = collect_videos(args.data_root)
    if args.limit and args.limit > 0:
        items = items[:args.limit]

    train_items = [it for it in items if it[1] == "train"]
    val_items   = [it for it in items if it[1] == "val"]
    test_items  = [it for it in items if it[1] == "test"]

    print(f"Found videos: train={len(train_items)} val={len(val_items)} test={len(test_items)}")
    print(f"Classes: {CLASSES}")

    if args.precache_only:
        print("Pre-caching pose sequences (this can take a while)...")
        precache_all(items, cache_dir=cache_dir, seq_len=args.seq_len, target_fps=args.target_fps)
        print("Pre-cache complete.")
        return

    train_ds = CAERSkeletonDataset(train_items, cache_dir, args.seq_len, args.target_fps, train=True)
    val_ds   = CAERSkeletonDataset(val_items,   cache_dir, args.seq_len, args.target_fps, train=False)
    test_ds  = CAERSkeletonDataset(test_items,  cache_dir, args.seq_len, args.target_fps, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    in_dim = 33 * 3
    model = BiLSTMSkeletonEmotion(in_dim=in_dim, hidden=args.hidden, num_layers=args.layers,
                                 num_classes=len(CLASSES), dropout=args.dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_val = -1.0
    metrics_path = os.path.join(args.workdir, "metrics.jsonl")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            optim.step()

            total_loss += loss.item() * y.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_acc = run_eval(model, val_loader, device)
        test_acc = run_eval(model, test_loader, device)

        dt = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "seconds": dt,
            "device": str(device),
        }
        print(json.dumps(row))
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        # save checkpoints
        last_path = os.path.join(args.workdir, "last.pt")
        torch.save({"model": model.state_dict(), "args": vars(args)}, last_path)

        if val_acc > best_val:
            best_val = val_acc
            best_path = os.path.join(args.workdir, "best.pt")
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
            print(f"Saved new best: {best_path} (val_acc={best_val:.4f})")

    print("Done. Check workdir for best.pt, last.pt, metrics.jsonl")

if __name__ == "__main__":
    main()
