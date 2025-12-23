
import os
import random
import atexit
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp

from utils import md5_of_path, ensure_dir

_BaseOptions = mp.tasks.BaseOptions
_PoseLandmarker = mp.tasks.vision.PoseLandmarker
_PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
_RunningMode = mp.tasks.vision.RunningMode

_LANDMARKER = None

def _default_model_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get("MP_POSE_TASK_MODEL", os.path.join(here, "models", "pose_landmarker_full.task"))

def get_landmarker():
    global _LANDMARKER
    if _LANDMARKER is not None:
        return _LANDMARKER

    model_path = _default_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"PoseLandmarker model not found at: {model_path}\n"
            f"Expected: ./models/pose_landmarker_full.task\n"
            f"Set MP_POSE_TASK_MODEL to override."
        )

    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=model_path),
        running_mode=_RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _LANDMARKER = _PoseLandmarker.create_from_options(options)
    atexit.register(lambda: _LANDMARKER.close() if _LANDMARKER is not None else None)
    return _LANDMARKER

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
    return np.sort(idx.astype(int))

def normalize_skeleton(seq: np.ndarray) -> np.ndarray:
    seq = seq.copy()
    hip = (seq[:, 23, :2] + seq[:, 24, :2]) / 2.0
    sh  = (seq[:, 11, :2] + seq[:, 12, :2]) / 2.0
    scale = np.linalg.norm(sh - hip, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-3)

    seq[:, :, 0] = (seq[:, :, 0] - hip[:, 0:1]) / scale
    seq[:, :, 1] = (seq[:, :, 1] - hip[:, 1:2]) / scale
    return seq

def extract_pose_sequence(video_path: str, seq_len: int = 32, target_fps: int = 10, train: bool = False) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = sample_frame_indices(total_frames, fps, seq_len, target_fps, train=train)

    landmarker = get_landmarker()
    seq = np.zeros((seq_len, 33, 3), dtype=np.float32)

    for t, fi in enumerate(frame_idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        ts_ms = int((fi / fps) * 1000.0)
        res = landmarker.detect_for_video(mp_image, ts_ms)
        if not res.pose_landmarks:
            continue

        lms = res.pose_landmarks[0]
        for k in range(min(33, len(lms))):
            seq[t, k, 0] = float(lms[k].x)
            seq[t, k, 1] = float(lms[k].y)
            v = getattr(lms[k], "visibility", None)
            if v is None:
                v = getattr(lms[k], "presence", 0.0)
            seq[t, k, 2] = float(v)

    cap.release()
    return normalize_skeleton(seq)

def load_or_extract(cache_dir: str, video_path: str, seq_len: int, target_fps: int, train: bool) -> np.ndarray:
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
