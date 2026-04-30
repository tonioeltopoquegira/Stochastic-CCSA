"""
Usage:
    python recompute_labels.py <run_dir> [exp]

Example:
    python recompute_labels.py runs_adversarial/mnist_cnn_adversarial_seed42_20260429_220003 mnist_cnn
"""
import sys, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

run_dir = Path(sys.argv[1])
exp = sys.argv[2] if len(sys.argv) > 2 else "mnist_cnn"
num_classes = 100 if "cifar100" in exp else 10

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_loaders, set_seed
from class_models import MNIST_CNN, resnet32, resnet56


def get_model(exp):
    if exp == "mnist_cnn":           return MNIST_CNN()
    elif exp == "cifar10_resnet32":  return resnet32(num_classes=10)
    elif exp == "cifar100_resnet56": return resnet56(num_classes=100)


def compute_per_label(model, loader, device, num_classes):
    model.eval()
    per_label_errors = defaultdict(list)
    ce = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            losses = ce(model(data), target)
            for label in range(num_classes):
                mask = target == label
                if mask.any():
                    per_label_errors[label].extend(losses[mask].cpu().numpy().tolist())
    return {l: float(np.mean(e)) for l, e in per_label_errors.items()}


set_seed(42)
device = torch.device("cpu")
train_loader, _ = get_loaders(exp=exp, batch_size=128, num_workers=0, pin_memory=False)

results = {}
for weight_file in sorted(run_dir.glob("model_*.pt")):
    opt_name = weight_file.stem.replace("model_", "")
    print(f"[INFO] Loading {weight_file.name} ...")
    model = get_model(exp).to(device)
    model.load_state_dict(torch.load(weight_file, map_location=device))
    per_label = compute_per_label(model, train_loader, device, num_classes)
    results[opt_name] = per_label
    print(f"\n  {opt_name.upper()} per-label train CE loss:")
    for label, err in sorted(per_label.items()):
        print(f"    label {label:3d}: {err:.6f}")

# Save JSON
out_json = run_dir / "per_label_errors.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[INFO] Saved JSON to {out_json}")

# --- Bar plot ---
if results:
    labels = list(range(num_classes))
    n_opts = len(results)
    bar_width = 0.8 / n_opts
    colors = plt.cm.Set2(np.linspace(0, 1, n_opts))

    fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.6), 6))

    for opt_idx, (opt_name, per_label) in enumerate(results.items()):
        errors = [max(per_label.get(l, 0.0), 1e-10) for l in labels]
        x_pos = np.arange(num_classes) + opt_idx * bar_width
        ax.bar(x_pos, errors, bar_width, label=opt_name.upper(),
               color=colors[opt_idx], alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_yscale('log')
    ax.set_xlabel("Label", fontsize=12)
    ax.set_ylabel("Avg CE loss (train, log scale)", fontsize=12)
    ax.set_title(f"Per-label train CE loss ({exp})", fontsize=13, fontweight='bold')
    ax.set_xticks(np.arange(num_classes) + bar_width * (n_opts - 1) / 2)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3, axis='y', which='both')
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()

    out_png = run_dir / "per_label_errors_barplot.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[INFO] Saved bar plot to {out_png}")