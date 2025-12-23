import os
import random
import numpy as np
import cv2
from tqdm import tqdm

import mediapipe as mp

# Robust import across MediaPipe package layouts
try:
    mp_pose = mp.solutions.pose  # older/typical API
except AttributeError:
    # Newer builds sometimes don't expose `solutions` at top-level
    try:
        import mediapipe.solutions as mp_solutions
        mp_pose = mp_solutions.pose
    except Exception:
        # Fallback for some installs where solutions live under mediapipe.python
        from mediapipe.python.solutions import pose as mp_pose


from utils import md5_of_path, ensure_dir


def make_cache_path(cache_dir: str, video_path: str, seq_len: int) -> str:
    h = md5_of_path(video_path)
    return os.path.join(cache_dir, f"{h}_T{seq_len}.npy")

def sample_frame_indices(total_frames: int, fps: float, seq_len: int, target_fps: int, train: bool):
    if total_frames <= 0:
        return np.zeros((seq_len,), dtype=int)

    stride = max(int(round(fps / max(target_fps, 1))), 1)
    needed = seq_len * stride

    if train and total_frames >= needed:
        start = random.randint(0, max(total_frames - needed, 0))
        idx = start + np.arange(seq_len) * stride
    else:
        idx = np.linspace(0, total_frames - 1, seq_len).astype(int)

    idx = np.clip(idx, 0, max(total_frames - 1, 0))
    return idx.astype(int)

def normalize_skeleton(seq: np.ndarray) -> np.ndarray:
    """
    Normalize to reduce camera effects:
    - center at mid-hip
    - scale by shoulder-hip distance
    seq: (T,33,3) where last dim is (x, y, visibility)
    """
    seq = seq.copy()
    hip = (seq[:, 23, :2] + seq[:, 24, :2]) / 2.0
    sh  = (seq[:, 11, :2] + seq[:, 12, :2]) / 2.0
    scale = np.linalg.norm(sh - hip, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-3)

    seq[:, :, 0] = (seq[:, :, 0] - hip[:, 0:1]) / scale
    seq[:, :, 1] = (seq[:, :, 1] - hip[:, 1:2]) / scale
    return seq

def extract_pose_sequence(
    video_path: str,
    seq_len: int = 32,
    target_fps: int = 10,
    train: bool = False,
    model_complexity: int = 1
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = sample_frame_indices(total_frames, fps, seq_len, target_fps, train=train)

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    seq = np.zeros((seq_len, 33, 3), dtype=np.float32)

    for t, fi in enumerate(frame_idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks is None:
            continue
        lms = res.pose_landmarks.landmark
        for k in range(33):
            seq[t, k, 0] = lms[k].x
            seq[t, k, 1] = lms[k].y
            seq[t, k, 2] = lms[k].visibility

    cap.release()
    pose.close()

    return normalize_skeleton(seq)

def load_or_extract(
    cache_dir: str,
    video_path: str,
    seq_len: int,
    target_fps: int,
    train: bool
) -> np.ndarray:
    ensure_dir(cache_dir)
    cp = make_cache_path(cache_dir, video_path, seq_len)
    if os.path.exists(cp):
        return np.load(cp)
    seq = extract_pose_sequence(video_path, seq_len=seq_len, target_fps=target_fps, train=train)
    np.save(cp, seq)
    return seq

def precache_all(items, cache_dir: str, seq_len: int, target_fps: int):
    ensure_dir(cache_dir)
    for (vp, split, _, _) in tqdm(items, desc="Pre-caching pose"):
        train = (split == "train")
        _ = load_or_extract(cache_dir, vp, seq_len, target_fps, train=train)
