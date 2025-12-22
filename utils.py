import hashlib
import os
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

def md5_of_path(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def is_video_file(p: str) -> bool:
    return Path(p).suffix.lower() in VIDEO_EXTS
