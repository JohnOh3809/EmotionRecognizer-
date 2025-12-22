import os
import glob
from pathlib import Path
from typing import List, Tuple
from utils import VIDEO_EXTS, is_video_file

# Standard 7 emotion categories used in CAER-style setups
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

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

def infer_split_from_path(p: str) -> str:
    parts = [x.lower() for x in Path(p).parts]
    if "train" in parts:
        return "train"
    if "validation" in parts or "val" in parts:
        return "val"
    if "test" in parts:
        return "test"
    return "unknown"

def infer_label_from_path(p: str) -> str:
    # Finds the nearest directory part matching a class label.
    parts = [normalize_label(x) for x in Path(p).parts]
    for x in reversed(parts):
        if x in CLASS_TO_IDX:
            return x
    return "unknown"

def collect_videos(data_root: str) -> List[Tuple[str, str, str, int]]:
    """
    Returns list of (video_path, split, label_str, label_idx)
    """
    all_files = []
    for ext in VIDEO_EXTS:
        all_files.extend(glob.glob(os.path.join(data_root, "**", f"*{ext}"), recursive=True))

    items = []
    for f in all_files:
        if not is_video_file(f):
            continue
        split = infer_split_from_path(f)
        label = infer_label_from_path(f)
        if split == "unknown" or label == "unknown":
            continue
        items.append((f, split, label, CLASS_TO_IDX[label]))
    return items
