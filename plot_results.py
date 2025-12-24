#!/usr/bin/env python3
"""
plot_results.py

Plot training results and sample predictions for the BiLSTM emotion classifier.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "other"]


def plot_training_curves(history: Dict[str, List], save_path: str):
    """Plot training loss, validation loss, accuracy curves, and learning rate."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves (train + val)
    axes[0, 0].plot(history["train_loss"], label="Train Loss", color="blue", linewidth=2)
    if "val_loss" in history:
        axes[0, 0].plot(history["val_loss"], label="Val Loss", color="orange", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curves")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[0, 1].plot(history["train_acc"], label="Train", color="blue", linewidth=2)
    axes[0, 1].plot(history["val_acc"], label="Validation", color="orange", linewidth=2)
    axes[0, 1].plot(history["test_acc"], label="Test", color="green", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Accuracy Curves")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate
    if "lr" in history:
        axes[1, 0].plot(history["lr"], color="red", linewidth=2)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].set_title("Learning Rate Schedule")
        axes[1, 0].grid(True, alpha=0.3)

    # Train vs Val loss gap (overfitting indicator)
    if "val_loss" in history:
        gap = [v - t for t, v in zip(history["train_loss"], history["val_loss"])]
        axes[1, 1].plot(gap, color="purple", linewidth=2)
        axes[1, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Val Loss - Train Loss")
        axes[1, 1].set_title("Generalization Gap (positive = overfitting)")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to: {save_path}")


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, classes: List[str], save_path: str):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes,
                yticklabels=classes, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion Matrix (Counts)")

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes,
                yticklabels=classes, ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix (Normalized)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to: {save_path}")


def plot_per_class_accuracy(labels: np.ndarray, preds: np.ndarray, classes: List[str], save_path: str):
    """Plot per-class accuracy bar chart."""
    class_acc = {}
    for i, cls in enumerate(classes):
        mask = labels == i
        if mask.sum() > 0:
            class_acc[cls] = float((preds[mask] == i).sum() / mask.sum())
        else:
            class_acc[cls] = 0.0

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    bars = ax.bar(classes, list(class_acc.values()), color=colors, edgecolor="black")

    ax.set_xlabel("Emotion Class")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.set_ylim(0, 1)
    ax.axhline(y=np.mean(list(class_acc.values())), color="red", linestyle="--",
               label=f"Mean: {np.mean(list(class_acc.values())):.3f}")
    ax.legend()

    # Add value labels on bars
    for bar, val in zip(bars, class_acc.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per-class accuracy to: {save_path}")


def extract_frame_from_video(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Extract a single frame from a video file."""
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def get_frame_from_npz(npz_path: str) -> Optional[np.ndarray]:
    """Load metadata from npz and extract a representative frame from the source video."""
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            if "meta" not in z:
                return None
            meta_bytes = z["meta"][0]
            if isinstance(meta_bytes, bytes):
                meta = json.loads(meta_bytes.decode("utf-8"))
            else:
                meta = json.loads(str(meta_bytes))

        # Get video path and frame indices from metadata
        meta_in = meta.get("meta_in", {})
        video_path = meta_in.get("video_path", "")
        frame_indices = meta_in.get("frame_indices", [])
        start_frame = meta_in.get("start_frame", 0)

        if not video_path or not os.path.exists(video_path):
            return None

        # Use middle frame from the sequence
        if frame_indices:
            mid_idx = len(frame_indices) // 2
            frame_num = start_frame + frame_indices[mid_idx]
        else:
            end_frame = meta_in.get("end_frame", start_frame + 10)
            frame_num = (start_frame + end_frame) // 2

        return extract_frame_from_video(video_path, frame_num)
    except Exception as e:
        print(f"Warning: Could not extract frame from {npz_path}: {e}")
        return None


def plot_sample_predictions(
    preds: np.ndarray,
    labels: np.ndarray,
    classes: List[str],
    paths: List[str],
    save_path: str,
    n_samples: int = 12,
):
    """Plot sample predictions with actual video frames."""
    # Find correct and incorrect predictions
    correct_mask = preds == labels
    incorrect_mask = ~correct_mask

    # Get indices for correct and incorrect samples
    correct_idx = np.where(correct_mask)[0]
    incorrect_idx = np.where(incorrect_mask)[0]

    # Sample some correct and incorrect predictions
    n_correct = min(n_samples // 2, len(correct_idx))
    n_incorrect = min(n_samples // 2, len(incorrect_idx))

    sample_correct = np.random.choice(correct_idx, n_correct, replace=False) if n_correct > 0 else []
    sample_incorrect = np.random.choice(incorrect_idx, n_incorrect, replace=False) if n_incorrect > 0 else []

    all_samples = list(sample_correct) + list(sample_incorrect)

    if len(all_samples) == 0:
        print("No samples to plot")
        return

    # Create plot
    n_cols = 4
    n_rows = (len(all_samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = np.atleast_2d(axes)

    for i, idx in enumerate(all_samples):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        true_label = classes[labels[idx]]
        pred_label = classes[preds[idx]]
        is_correct = labels[idx] == preds[idx]

        # Try to extract frame from video
        frame = None
        if idx < len(paths):
            frame = get_frame_from_npz(paths[idx])

        if frame is not None:
            ax.imshow(frame)
        else:
            # Fallback: show colored background with text
            ax.set_facecolor("lightgray")

        # Add label overlay
        color = "green" if is_correct else "red"
        title = f"True: {true_label} | Pred: {pred_label}"
        ax.set_title(title, fontsize=10, color=color, fontweight="bold")
        ax.axis("off")

    # Hide unused axes
    for i in range(len(all_samples), n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis("off")

    plt.suptitle("Sample Predictions (Green=Correct, Red=Incorrect)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved sample predictions to: {save_path}")


def plot_prediction_distribution(preds: np.ndarray, labels: np.ndarray, classes: List[str], save_path: str):
    """Plot distribution of predictions vs true labels."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # True label distribution
    unique, counts = np.unique(labels, return_counts=True)
    axes[0].bar([classes[i] for i in unique], counts, color="steelblue", edgecolor="black")
    axes[0].set_xlabel("Emotion Class")
    axes[0].set_ylabel("Count")
    axes[0].set_title("True Label Distribution")
    axes[0].tick_params(axis="x", rotation=45)

    # Predicted distribution
    unique, counts = np.unique(preds, return_counts=True)
    axes[1].bar([classes[i] for i in unique], counts, color="coral", edgecolor="black")
    axes[1].set_xlabel("Emotion Class")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Predicted Label Distribution")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction distribution to: {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=str, required=True, help="Training workdir with results")
    ap.add_argument("--data_dir", type=str, default="", help="Enhanced data directory (optional)")
    args = ap.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    plots_dir = workdir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load history
    history_path = workdir / "history.npy"
    if history_path.exists():
        history = np.load(history_path, allow_pickle=True).item()
        plot_training_curves(history, str(plots_dir / "training_curves.png"))
    else:
        print(f"Warning: history.npy not found in {workdir}")

    # Load predictions
    preds_path = workdir / "test_predictions.npz"
    if preds_path.exists():
        data = np.load(preds_path, allow_pickle=True)
        preds = data["preds"]
        labels = data["labels"]
        classes = list(data["classes"])

        # Plot confusion matrix
        plot_confusion_matrix(labels, preds, classes, str(plots_dir / "confusion_matrix.png"))

        # Plot per-class accuracy
        plot_per_class_accuracy(labels, preds, classes, str(plots_dir / "per_class_accuracy.png"))

        # Plot prediction distribution
        plot_prediction_distribution(preds, labels, classes, str(plots_dir / "prediction_distribution.png"))

        # Plot sample predictions with video frames
        paths = list(data["paths"]) if "paths" in data else []
        plot_sample_predictions(
            preds, labels, classes, paths,
            str(plots_dir / "sample_predictions.png"),
        )

        # Print classification report
        print("\n" + "=" * 50)
        print("Classification Report:")
        print(classification_report(labels, preds, target_names=classes))

    else:
        print(f"Warning: test_predictions.npz not found in {workdir}")

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
