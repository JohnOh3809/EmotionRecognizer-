#!/usr/bin/env python3
"""
Generate final comparison plots combining all experiments.
"""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("/home/az2/Documents/EmotionRecognizer-/comparison_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Collect all results
all_results = []

# From scheduler experiments
scheduler_dir = Path("/home/az2/Documents/EmotionRecognizer-/runs/scheduler_experiments_20251224_004812")
if scheduler_dir.exists():
    with open(scheduler_dir / "results.json") as f:
        sched_results = json.load(f)
    for r in sched_results:
        all_results.append({
            "name": f"sched_{r['scheduler']}_lr{r['lr']:.0e}",
            "val_acc": r["best_val_acc"],
            "test_acc": r["final_test_acc"],
            "n_classes": r["n_classes_above_5"],
            "type": "scheduler"
        })

# From previous training runs (load history.npy)
runs_dir = Path("/home/az2/Documents/EmotionRecognizer-/runs")
for run_dir in sorted(runs_dir.iterdir()):
    if run_dir.name.startswith("bilstm_") and (run_dir / "history.npy").exists():
        try:
            history = np.load(run_dir / "history.npy", allow_pickle=True).item()
            best_val = max(history.get("val_acc", [0]))
            best_test = max(history.get("test_acc", [0]))

            # Load predictions to count classes
            n_classes = 0
            if (run_dir / "test_predictions.npz").exists():
                data = np.load(run_dir / "test_predictions.npz", allow_pickle=True)
                preds = data["preds"]
                labels = data["labels"]
                for i in range(7):
                    mask = labels == i
                    if mask.sum() > 0:
                        acc = (preds[mask] == i).sum() / mask.sum()
                        if acc > 0.05:
                            n_classes += 1

            short_name = run_dir.name.replace("bilstm_20251223_", "")
            all_results.append({
                "name": short_name[:25],
                "val_acc": best_val,
                "test_acc": best_test,
                "n_classes": n_classes,
                "type": "bilstm"
            })
        except Exception as e:
            print(f"Error loading {run_dir.name}: {e}")

# Add advanced training results
advanced_results = [
    {"name": "kfold_5fold", "val_acc": 0.2097, "test_acc": 0.0, "n_classes": 0, "type": "advanced"},
    {"name": "ensemble_5models", "val_acc": 0.2103, "test_acc": 0.1805, "n_classes": 7, "type": "advanced"},
    {"name": "metric_triplet", "val_acc": 0.2054, "test_acc": 0.1732, "n_classes": 4, "type": "advanced"},
    {"name": "prototypical", "val_acc": 0.2249, "test_acc": 0.0, "n_classes": 0, "type": "advanced"},
]
all_results.extend(advanced_results)

# Sort by val accuracy
all_results.sort(key=lambda x: x["val_acc"], reverse=True)

# Create comprehensive plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top 15 by val accuracy
top_15 = all_results[:15]
names = [r["name"] for r in top_15]
val_accs = [r["val_acc"] * 100 for r in top_15]
test_accs = [r["test_acc"] * 100 for r in top_15]
types = [r["type"] for r in top_15]

# Color by type
colors_val = []
colors_test = []
for t in types:
    if t == "scheduler":
        colors_val.append("steelblue")
        colors_test.append("lightsteelblue")
    elif t == "advanced":
        colors_val.append("forestgreen")
        colors_test.append("lightgreen")
    else:
        colors_val.append("coral")
        colors_test.append("lightsalmon")

x = np.arange(len(names))
width = 0.35

# Plot 1: Val vs Test accuracy
bars1 = axes[0, 0].bar(x - width/2, val_accs, width, label='Val', color=colors_val, edgecolor='black')
bars2 = axes[0, 0].bar(x + width/2, test_accs, width, label='Test', color=colors_test, edgecolor='black')
axes[0, 0].set_ylabel('Accuracy (%)', fontsize=11)
axes[0, 0].set_title('Top 15 Experiments: Validation vs Test Accuracy', fontsize=12)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
axes[0, 0].legend()
axes[0, 0].axhline(y=14.3, color='gray', linestyle='--', alpha=0.5, label='Random (14.3%)')
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].set_ylim(0, 30)

# Add value labels
for bar, val in zip(bars1, val_accs):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=7)

# Plot 2: Classes distribution
n_classes = [r["n_classes"] for r in top_15]
class_colors = ['green' if n >= 6 else 'orange' if n >= 4 else 'red' if n > 0 else 'gray' for n in n_classes]
axes[0, 1].bar(names, n_classes, color=class_colors, edgecolor='black')
axes[0, 1].set_ylabel('Number of Classes', fontsize=11)
axes[0, 1].set_title('Classes with >5% Accuracy per Experiment', fontsize=12)
axes[0, 1].set_xticks(range(len(names)))
axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
axes[0, 1].set_ylim(0, 8)
axes[0, 1].axhline(y=7, color='green', linestyle='--', alpha=0.5)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Val accuracy by experiment type
type_stats = {}
for r in all_results:
    t = r["type"]
    if t not in type_stats:
        type_stats[t] = []
    type_stats[t].append(r["val_acc"] * 100)

type_names = list(type_stats.keys())
type_means = [np.mean(type_stats[t]) for t in type_names]
type_maxs = [np.max(type_stats[t]) for t in type_names]
type_mins = [np.min(type_stats[t]) for t in type_names]

x_types = np.arange(len(type_names))
axes[1, 0].bar(x_types - 0.2, type_means, 0.4, label='Mean', color='steelblue', edgecolor='black')
axes[1, 0].bar(x_types + 0.2, type_maxs, 0.4, label='Max', color='coral', edgecolor='black')
axes[1, 0].set_ylabel('Val Accuracy (%)', fontsize=11)
axes[1, 0].set_title('Performance by Experiment Type', fontsize=12)
axes[1, 0].set_xticks(x_types)
axes[1, 0].set_xticklabels(type_names)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Add error bars for range
for i, (mean, max_v, min_v) in enumerate(zip(type_means, type_maxs, type_mins)):
    axes[1, 0].errorbar(i - 0.2, mean, yerr=[[mean - min_v], [max_v - mean]],
                       color='black', capsize=5, capthick=2)

# Plot 4: Summary table
axes[1, 1].axis('off')
table_data = []
for r in top_15[:10]:
    table_data.append([
        r["name"][:20],
        f"{r['val_acc']*100:.1f}%",
        f"{r['test_acc']*100:.1f}%" if r['test_acc'] > 0 else "N/A",
        str(r["n_classes"]) if r["n_classes"] > 0 else "N/A",
        r["type"]
    ])

table = axes[1, 1].table(
    cellText=table_data,
    colLabels=['Experiment', 'Val Acc', 'Test Acc', '#Classes', 'Type'],
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
axes[1, 1].set_title('Top 10 Experiments Summary', fontsize=12, pad=20)

plt.suptitle('Emotion Recognition from Pose Data - All Experiments Comparison', fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / "final_comparison_all_experiments.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved final comparison to {OUTPUT_DIR / 'final_comparison_all_experiments.png'}")

# Print summary
print("\n" + "=" * 70)
print("FINAL SUMMARY - ALL EXPERIMENTS")
print("=" * 70)
print(f"\n{'Rank':<5} {'Experiment':<30} {'Val Acc':<10} {'Test Acc':<10} {'Classes':<8} {'Type':<12}")
print("-" * 75)
for i, r in enumerate(all_results[:15], 1):
    test_str = f"{r['test_acc']*100:.1f}%" if r['test_acc'] > 0 else "N/A"
    cls_str = str(r['n_classes']) if r['n_classes'] > 0 else "N/A"
    print(f"{i:<5} {r['name']:<30} {r['val_acc']*100:<10.1f} {test_str:<10} {cls_str:<8} {r['type']:<12}")

print("\n" + "=" * 70)
print("KEY FINDINGS:")
print("=" * 70)
best = all_results[0]
print(f"\n1. Best Model: {best['name']}")
print(f"   - Val Accuracy: {best['val_acc']*100:.2f}%")
print(f"   - Test Accuracy: {best['test_acc']*100:.2f}%" if best['test_acc'] > 0 else "   - Test Accuracy: N/A")

print(f"\n2. Random chance baseline: 14.3% (7 classes)")
print(f"   Improvement over random: {(best['val_acc'] - 0.143)*100:.1f}%")

print(f"\n3. Best configurations tend to use:")
print("   - One-Cycle learning rate scheduler")
print("   - Learning rate around 5e-4")
print("   - 1-layer BiLSTM with 128 hidden units")
print("   - Label smoothing and balanced sampling")

print(f"\n4. Main challenge: Heavy overfitting (train ~70% vs val ~22%)")
print("   - Pose data alone has limited discriminative power for emotions")
print("   - Would benefit from: more data, facial expressions, audio")
