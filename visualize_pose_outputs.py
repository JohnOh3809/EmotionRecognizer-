#!/usr/bin/env python3
"""
visualize_yolopose17_frames.py

Walk a directory of YOLO-Pose extracted samples (*.npz) and export **annotated frames**
(one image per frame) by drawing COCO-17 skeletons on the original video frames.

This matches your request:
- "goes through all data in a directory and visualizes all samples"
- "draw skeleton upon the frame"
- "label frame number and label"
- outputs each frame as an image, not a single mp4.

Expected .npz format (from filtering_yolopose17_sharded.py):
  keypoints: (T,K,17,3)
  frame_indices: (T,)
  meta[0] JSON containing "video_path" + "label"

If a video file cannot be opened, it will draw skeletons on a blank canvas.

Example:
python visualize_pose_outputs.py \
  --data_dir ../data \
  --out_dir  ../data/viz \
  --score_thr 0.20 \
  --max_people 3 \
  --max_samples 200 \
  --max_frames_per_sample 120
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

J = 17
EDGES = [
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

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def list_npz(data_dir: Path) -> List[Path]:
    return sorted(data_dir.rglob("*.npz"))

def read_meta(npz_path: Path) -> Dict:
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

def draw_skeleton(frame: np.ndarray, kps: np.ndarray, score_thr: float) -> None:
    """
    frame: (H,W,3) BGR
    kps: (K,17,3)
    """
    H, W = frame.shape[:2]
    K = kps.shape[0]
    for pi in range(K):
        p = kps[pi]
        xy = p[:, :2]
        sc = p[:, 2]
        ok = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1]) & np.isfinite(sc) & (sc >= score_thr)

        # lines
        for a, b in EDGES:
            if ok[a] and ok[b]:
                x1, y1 = int(xy[a, 0]), int(xy[a, 1])
                x2, y2 = int(xy[b, 0]), int(xy[b, 1])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

        # joints
        for j in range(J):
            if ok[j]:
                x, y = int(xy[j, 0]), int(xy[j, 1])
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA)

        cv2.putText(frame, f"p{pi}", (10, 30 + 18 * pi), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

def get_frame(cap: cv2.VideoCapture, idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--score_thr", type=float, default=0.20)
    ap.add_argument("--max_people", type=int, default=3)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--max_frames_per_sample", type=int, default=0)
    ap.add_argument("--every_n", type=int, default=1, help="Only export every N-th frame in the sample.")
    ap.add_argument("--jpeg_quality", type=int, default=92)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    files = list_npz(data_dir)
    if args.max_samples and args.max_samples > 0:
        files = files[: int(args.max_samples)]
    if not files:
        raise RuntimeError(f"No .npz found under {data_dir}")

    for p in tqdm(files, desc="Samples"):
        meta = read_meta(p)
        label = str(meta.get("label", p.parent.name))
        video_path = meta.get("video_path", "")

        with np.load(p, allow_pickle=False) as z:
            kps = z["keypoints"].astype(np.float32)  # (T,K,17,3)
            frame_idx = z["frame_indices"].astype(np.int32) if "frame_indices" in z else np.arange(kps.shape[0], dtype=np.int32)

        if kps.ndim != 4 or kps.shape[2] != J or kps.shape[3] != 3:
            continue

        T = kps.shape[0]
        K = min(int(args.max_people), kps.shape[1])
        kps = kps[:, :K]

        # output subdir
        sample_dir = out_dir / label / p.stem
        ensure_dir(sample_dir)

        cap = None
        if video_path and Path(video_path).exists():
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                cap.release()
                cap = None

        # determine canvas
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)

        max_frames = int(args.max_frames_per_sample) if args.max_frames_per_sample and args.max_frames_per_sample > 0 else T

        for ti in range(min(T, max_frames)):
            if args.every_n > 1 and (ti % int(args.every_n) != 0):
                continue
            fi = int(frame_idx[ti]) if ti < len(frame_idx) else int(ti)

            if cap is not None:
                frame = get_frame(cap, fi)
                if frame is None:
                    frame = blank.copy()
            else:
                frame = blank.copy()

            draw_skeleton(frame, kps[ti], score_thr=float(args.score_thr))

            cv2.putText(frame, f"label: {label}", (10, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"frame: {fi}", (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            out_path = sample_dir / f"frame_{ti:05d}_src{fi:07d}.jpg"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])

        if cap is not None:
            cap.release()

    print(f"[OK] wrote annotated frames to: {out_dir}")

if __name__ == "__main__":
    main()