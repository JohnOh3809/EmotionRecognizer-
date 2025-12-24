# config.py
from pathlib import Path

# ======================
# ROOT PATHS (EDIT HERE)
# ======================

# CAER dataset root (contains train/validation/test folders)
CAER_ROOT     = Path("/home/az2/Documents/EmotionRecognizer-/CAER/CAER")
OPENPOSE_ROOT = Path("/home/az2/Documents/EmotionRecognizer-/openpose")
OUTPUT_ROOT   = Path("/home/az2/Documents/EmotionRecognizer-/data")
# YOLO weights
YOLO_WEIGHTS = Path("./yolov8n-pose.pt")

# ======================
# VALIDATION (fail early)
# ======================

def _check_dir(p: Path, name: str):
    if not p.exists():
        raise RuntimeError(f"{name} does not exist: {p}")
    return p.resolve()


