# config.py
from pathlib import Path

# ======================
# ROOT PATHS (EDIT HERE)
# ======================

# CAER dataset root (contains train/validation/test folders)
CAER_ROOT     = Path("/home/ubuntu/CAER")
OPENPOSE_ROOT = Path("/home/ubuntu/openpose")
OUTPUT_ROOT   = Path("/home/ubuntu/temp/caer_openpose")
# YOLO weights
YOLO_WEIGHTS = Path("./models/yolov8n.pt")

# ======================
# VALIDATION (fail early)
# ======================

def _check_dir(p: Path, name: str):
    if not p.exists():
        raise RuntimeError(f"{name} does not exist: {p}")
    return p.resolve()


