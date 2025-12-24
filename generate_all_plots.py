#!/usr/bin/env python3
"""
Generate plots for all training runs and create comparison charts.
"""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

RUNS_DIR = Path("/home/az2/Documents/EmotionRecognizer-/runs")
OUTPUT_DIR = Path("/home/az2/Documents/EmotionRecognizer-/comparison_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def load_run_data(run_dir):
    """Load history and predictions from a run directory."""
    history_path = run_dir / "history.npy"
    preds_path = run_dir / "test_predictions.npz"

    data = {"name": run_dir.name}

    if history_path.exists():
        try:
            history = np.load(history_path, allow_pickle=True).item()
            data["history"] = history
        except Exception as e:
            print(f"  Warning: Could not load history from {run_dir.name}: {e}")

    if preds_path.exists():
        try:
            preds_data = np.load(preds_path, allow_pickle=True)
            data["preds"] = preds_data["preds"]
            data["labels"] = preds_data["labels"]
            if "classes" in preds_data:
                data["classes"] = list(preds_data["classes"])
        except Exception as e:
            print(f"  Warning: Could not load predictions from {run_dir.name}: {e}")

    return data


def plot_single_run(data, output_dir):
    """Generate plots for a single run."""
    run_name = data["name"]
    plots_dir = output_dir / run_name
    plots_dir.mkdir(exist_ok=True)

    # Training curves
    if "history" in data:
        history = data["history"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss curves
        if "train_loss" in history:
            axes[0, 0].plot(history["train_loss"], label="Train", color="blue", linewidth=2)
        if "val_loss" in history:
            axes[0, 0].plot(history["val_loss"], label="Val", color="orange", linewidth=2)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Loss Curves")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy curves
        if "train_acc" in history:
            axes[0, 1].plot(history["train_acc"], label="Train", color="blue", linewidth=2)
        if "val_acc" in history:
            axes[0, 1].plot(history["val_acc"], label="Val", color="orange", linewidth=2)
        if "test_acc" in history:
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
        else:
            axes[1, 0].axis("off")

        # Best metrics text
        best_val = max(history.get("val_acc", [0]))
        best_test = max(history.get("test_acc", [0]))
        final_train = history.get("train_acc", [0])[-1] if history.get("train_acc") else 0

        axes[1, 1].axis("off")
        text = f"Best Val Acc: {best_val*100:.2f}%\n"
        text += f"Best Test Acc: {best_test*100:.2f}%\n"
        text += f"Final Train Acc: {final_train*100:.2f}%\n"
        text += f"Overfit Gap: {(final_train - best_val)*100:.2f}%"
        axes[1, 1].text(0.5, 0.5, text, transform=axes[1, 1].transAxes,
                       fontsize=14, verticalalignment='center', horizontalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(run_name, fontsize=12)
        plt.tight_layout()
        plt.savefig(plots_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Confusion matrix and per-class accuracy
    if "preds" in data and "labels" in data:
        preds = data["preds"]
        labels = data["labels"]
        classes = data.get("classes", CLASSES)

        # Per-class accuracy
        fig, ax = plt.subplots(figsize=(10, 5))
        class_acc = {}
        for i, cls in enumerate(classes):
            mask = labels == i
            if mask.sum() > 0:
                class_acc[cls] = float((preds[mask] == i).sum() / mask.sum())
            else:
                class_acc[cls] = 0.0

        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        bars = ax.bar(list(class_acc.keys()), list(class_acc.values()), color=colors, edgecolor="black")
        ax.set_xlabel("Emotion Class")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Per-Class Accuracy - {run_name}")
        ax.set_ylim(0, 1)
        mean_acc = np.mean(list(class_acc.values()))
        ax.axhline(y=mean_acc, color="red", linestyle="--", label=f"Mean: {mean_acc:.3f}")
        ax.legend()

        for bar, val in zip(bars, class_acc.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                   f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(plots_dir / "per_class_accuracy.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, preds, labels=range(len(classes)))
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        im1 = axes[0].imshow(cm, cmap='Blues')
        axes[0].set_xticks(range(len(classes)))
        axes[0].set_yticks(range(len(classes)))
        axes[0].set_xticklabels(classes, rotation=45, ha='right')
        axes[0].set_yticklabels(classes)
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        axes[0].set_title("Confusion Matrix (Counts)")
        plt.colorbar(im1, ax=axes[0])

        # Add text annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8)

        im2 = axes[1].imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        axes[1].set_xticks(range(len(classes)))
        axes[1].set_yticks(range(len(classes)))
        axes[1].set_xticklabels(classes, rotation=45, ha='right')
        axes[1].set_yticklabels(classes)
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")
        axes[1].set_title("Confusion Matrix (Normalized)")
        plt.colorbar(im2, ax=axes[1])

        for i in range(len(classes)):
            for j in range(len(classes)):
                axes[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha='center', va='center', fontsize=8)

        plt.suptitle(run_name, fontsize=12)
        plt.tight_layout()
        plt.savefig(plots_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Generated plots for {run_name}")


def plot_comparison(all_data):
    """Generate comparison plots across all runs."""
    # Filter runs with history
    runs_with_history = [d for d in all_data if "history" in d]

    if not runs_with_history:
        print("No runs with history found")
        return

    # Extract best metrics
    results = []
    for data in runs_with_history:
        history = data["history"]
        best_val = max(history.get("val_acc", [0]))
        best_test = max(history.get("test_acc", [0]))
        final_train = history.get("train_acc", [0])[-1] if history.get("train_acc") else 0

        # Count classes with >5% accuracy
        n_classes = 0
        if "preds" in data and "labels" in data:
            preds = data["preds"]
            labels = data["labels"]
            for i in range(7):
                mask = labels == i
                if mask.sum() > 0:
                    acc = (preds[mask] == i).sum() / mask.sum()
                    if acc > 0.05:
                        n_classes += 1

        # Shorten name for display
        short_name = data["name"].replace("bilstm_20251223_", "").replace("small_", "s_").replace("tiny_", "t_")

        results.append({
            "name": short_name,
            "full_name": data["name"],
            "val_acc": best_val,
            "test_acc": best_test,
            "train_acc": final_train,
            "n_classes": n_classes,
            "overfit": final_train - best_val
        })

    # Sort by val accuracy
    results.sort(key=lambda x: x["val_acc"], reverse=True)

    # Plot comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Val vs Test accuracy
    names = [r["name"][:25] for r in results[:15]]  # Top 15
    val_accs = [r["val_acc"] * 100 for r in results[:15]]
    test_accs = [r["test_acc"] * 100 for r in results[:15]]

    x = np.arange(len(names))
    width = 0.35

    axes[0, 0].bar(x - width/2, val_accs, width, label='Val', color='steelblue')
    axes[0, 0].bar(x + width/2, test_accs, width, label='Test', color='coral')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Validation vs Test Accuracy')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[0, 0].legend()
    axes[0, 0].axhline(y=14.3, color='gray', linestyle='--', alpha=0.5, label='Random')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Number of classes with >5% accuracy
    n_classes = [r["n_classes"] for r in results[:15]]
    colors = ['green' if n >= 6 else 'orange' if n >= 4 else 'red' for n in n_classes]
    axes[0, 1].bar(names, n_classes, color=colors, edgecolor='black')
    axes[0, 1].set_ylabel('Number of Classes')
    axes[0, 1].set_title('Classes with >5% Accuracy (green=6+, orange=4-5, red=<4)')
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[0, 1].axhline(y=7, color='green', linestyle='--', alpha=0.5)
    axes[0, 1].set_ylim(0, 8)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Overfitting (train - val gap)
    overfit = [r["overfit"] * 100 for r in results[:15]]
    colors = ['red' if o > 20 else 'orange' if o > 10 else 'green' for o in overfit]
    axes[1, 0].bar(names, overfit, color=colors, edgecolor='black')
    axes[1, 0].set_ylabel('Train - Val Accuracy (%)')
    axes[1, 0].set_title('Overfitting Gap (green=<10%, orange=10-20%, red=>20%)')
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[1, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Summary table
    axes[1, 1].axis('off')
    table_data = []
    for r in results[:10]:
        table_data.append([
            r["name"][:20],
            f"{r['val_acc']*100:.1f}%",
            f"{r['test_acc']*100:.1f}%",
            str(r["n_classes"]),
            f"{r['overfit']*100:.1f}%"
        ])

    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=['Run', 'Val', 'Test', '#Cls', 'Overfit'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Top 10 Runs Summary', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_all_runs.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {OUTPUT_DIR / 'comparison_all_runs.png'}")

    # Training curves comparison (top 5)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    top_5 = [d for d in all_data if d["name"] in [r["full_name"] for r in results[:5]]]
    colors = plt.cm.tab10(np.linspace(0, 1, 5))

    for i, data in enumerate(top_5):
        if "history" in data:
            history = data["history"]
            short_name = data["name"].replace("bilstm_20251223_", "")[:20]
            if "val_acc" in history:
                axes[0].plot(history["val_acc"], label=short_name, color=colors[i], linewidth=2)
            if "train_acc" in history:
                axes[1].plot(history["train_acc"], label=short_name, color=colors[i], linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Accuracy")
    axes[0].set_title("Val Accuracy Over Training (Top 5)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Training Accuracy")
    axes[1].set_title("Train Accuracy Over Training (Top 5)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_top5_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved top 5 curves to {OUTPUT_DIR / 'comparison_top5_curves.png'}")


def main():
    print("Loading all runs...")
    all_data = []

    for run_dir in sorted(RUNS_DIR.iterdir()):
        if run_dir.is_dir() and run_dir.name.startswith("bilstm_"):
            print(f"  Loading {run_dir.name}...")
            data = load_run_data(run_dir)
            if "history" in data or "preds" in data:
                all_data.append(data)

    print(f"\nLoaded {len(all_data)} runs with data")

    # Generate individual plots
    print("\nGenerating individual run plots...")
    for data in all_data:
        plot_single_run(data, OUTPUT_DIR)

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(all_data)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
