
import os, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from caer_dataset import collect_videos, CLASSES
from model import BiLSTMSkeletonEmotion
from pose_extract import load_or_extract

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, items, cache_dir, seq_len, target_fps):
        self.items = items
        self.cache_dir = cache_dir
        self.seq_len = seq_len
        self.target_fps = target_fps

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        vp, split, label_str, y = self.items[idx]
        seq = load_or_extract(self.cache_dir, vp, self.seq_len, self.target_fps, train=False)
        x = seq.reshape(self.seq_len, -1).astype(np.float32)
        return torch.from_numpy(x), int(y)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--workdir", default="./runs/caer_skeleton")
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--ckpt", default="", help="Path to checkpoint (default: workdir/best.pt)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--target_fps", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    cache_dir = os.path.join(args.workdir, "pose_cache")
    ckpt_path = args.ckpt or os.path.join(args.workdir, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    in_dim = 33 * 3
    model = BiLSTMSkeletonEmotion(in_dim=in_dim, num_classes=len(CLASSES))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    items = collect_videos(args.data_root)
    split_items = [it for it in items if it[1] == args.split]
    ds = EvalDataset(split_items, cache_dir, args.seq_len, args.target_fps)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    y_true, y_pred = [], []
    for x, y in dl:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(y)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    rep = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)

    out = {
        "split": args.split,
        "n": len(y_true),
        "acc": float((np.array(y_true) == np.array(y_pred)).mean()),
    }

    os.makedirs(args.workdir, exist_ok=True)
    with open(os.path.join(args.workdir, f"eval_{args.split}.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savetxt(os.path.join(args.workdir, f"confusion_{args.split}.csv"), cm, fmt="%d", delimiter=",")

    print(json.dumps(out, indent=2))
    print(rep)
    print(f"Wrote: eval_{args.split}.json and confusion_{args.split}.csv in {args.workdir}")

if __name__ == "__main__":
    main()
