#!/usr/bin/env bash
# Convenience script: create venv, install deps, and run full training with logfile
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$REPO_ROOT/.venv"
PY="$VENV/bin/python"
LOGDIR="$REPO_ROOT/runs/caer_skeleton"
LOGFILE="$LOGDIR/train.log"
mkdir -p "$LOGDIR"
if [ ! -d "$VENV" ]; then
  echo "Creating venv..."
  python3 -m venv "$VENV"
  "$PY" -m pip install --upgrade pip setuptools wheel
  echo "Installing requirements... (this may take a while)"
  "$PY" -m pip install -r "$REPO_ROOT/requirements.txt"
fi
# Auto-detect data root: prefer env CAER_ROOT or ./CAER
DATA_ROOT="${CAER_ROOT:-$REPO_ROOT/CAER}"
if [ ! -d "$DATA_ROOT" ]; then
  echo "Data root not found: $DATA_ROOT"
  echo "Please set CAER_ROOT or place dataset at $REPO_ROOT/CAER"
  exit 1
fi
# Run training in background with nohup
nohup "$PY" "$REPO_ROOT/train.py" --data_root "$DATA_ROOT" --epochs 10 --batch_size 32 --num_workers 4 --logfile "$LOGFILE" > "$LOGDIR/train.out" 2>&1 &
echo "Training started. Tail logs with: tail -F $LOGFILE"