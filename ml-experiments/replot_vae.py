"""
Standalone re-plotter for mnist_vae runs.

Usage
-----
python replot_vae.py --rundir ./runs_vae/mnist_vae_seed42_20240523_142100

# show only specific runs
python replot_vae.py --rundir ./runs_vae/mnist_vae_seed42_... \\
    --show adam_additive_b0.1 adam_additive_b1 adam_additive_b10 \\
           ccsa_kl_constrained_kl5 ccsa_kl_constrained_kl10

# save plots to a different directory
python replot_vae.py --rundir ./runs_vae/mnist_vae_seed42_... --outdir ./figs
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── colour / style helpers ──────────────────────────────────────────────────
_CMAP = plt.get_cmap("tab10")

def _style(label: str, idx: int):
    """Return (color, linestyle, marker) for a run label."""
    color = _CMAP(idx % 10)
    if label.startswith("adam"):
        ls, mk = "-", "o"
    elif "kl_constrained" in label:
        ls, mk = "--", "s"
    elif "recon_constrained" in label:
        ls, mk = "-.", "^"
    else:           # ccsa_additive
        ls, mk = ":", "D"
    return color, ls, mk


# ── loaders ─────────────────────────────────────────────────────────────────
def load_run(rundir: Path, show: list[str] | None):
    """
    Returns
    -------
    batch : dict  label → {losses, recons, kls, evals}  (from results.npz)
    epoch : dict  label → logs dict                      (from logs.json)
    labels : list[str]  ordered list of labels to plot
    """
    npz_path  = rundir / "results.npz"
    logs_path = rundir / "logs.json"

    if not npz_path.exists():
        sys.exit(f"[ERROR] results.npz not found in {rundir}")
    if not logs_path.exists():
        sys.exit(f"[ERROR] logs.json not found in {rundir}")

    data = np.load(npz_path)
    with open(logs_path) as f:
        epoch_logs = json.load(f)

    # discover all labels present in the npz
    all_labels = sorted({k.rsplit("_", 1)[0] for k in data.files
                         if k.endswith("_losses")})

    if show:
        missing = [l for l in show if l not in all_labels]
        if missing:
            print(f"[WARN] labels not found in run and will be skipped: {missing}")
            print(f"       available: {all_labels}")
        labels = [l for l in show if l in all_labels]
        if not labels:
            sys.exit("[ERROR] None of the requested labels exist in this run.")
    else:
        labels = all_labels

    batch, epoch = {}, {}
    for lbl in labels:
        batch[lbl] = {
            "losses": data[f"{lbl}_losses"],
            "recons": data[f"{lbl}_recons"],
            "kls":    data[f"{lbl}_kls"],
            "evals":  data[f"{lbl}_evals"],
        }
        epoch[lbl] = epoch_logs.get(lbl, {})

    return batch, epoch, labels


# ── individual plot functions ────────────────────────────────────────────────
def plot_batch_curves(batch, labels, outdir, log_x=False, tradeoff_filter=None):
    """Three-panel: objective / recon / KL vs cumulative evals."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["objective (train batch)", "reconstruction BCE (train batch)", "KL (train batch)"]
    keys   = ["losses", "recons", "kls"]

    # Apply filter if provided
    if tradeoff_filter is not None:
        def should_include(label):
            label_mode = None
            for mode in tradeoff_filter.keys():
                if mode in label:
                    label_mode = mode
                    break
            if label_mode is None:
                return False  # Exclude modes not in filter
            config = tradeoff_filter[label_mode]
            if isinstance(config, list):
                for val in config:
                    if label.endswith(f"_b{val}") or label.endswith(f"_kl{val}"):
                        return True
                return False
            elif isinstance(config, dict) and "exclude_values" in config:
                for val in config["exclude_values"]:
                    if label.endswith(f"_b{val}") or label.endswith(f"_kl{val}"):
                        return False
                return True
            return True
        filtered_labels = [lbl for lbl in labels if should_include(lbl)]
    else:
        filtered_labels = labels

    for idx, lbl in enumerate(filtered_labels):
        color, ls, _ = _style(lbl, idx)
        x = batch[lbl]["evals"]
        # Adam uses forward + backward, so multiply by 2 for fair comparison
        if lbl.startswith("adam"):
            x = x * 2.0
        for ax, key in zip(axes, keys):
            y = batch[lbl][key]
            ax.plot(x, y, label=lbl, alpha=0.6, color=color, linestyle=ls, linewidth=1.2)

    for ax, title in zip(axes, titles):
        ax.set_xlabel("evals")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        ax.legend(fontsize=7, ncol=max(1, len(labels)//6))
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out = outdir / "vae_curves_vs_evals.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_epoch_curves(epoch, labels, outdir, tradeoff_filter=None):
    """Two-panel: val recon / val KL per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Apply filter if provided
    if tradeoff_filter is not None:
        def should_include(label):
            label_mode = None
            for mode in tradeoff_filter.keys():
                if mode in label:
                    label_mode = mode
                    break
            if label_mode is None:
                return False  # Exclude modes not in filter
            config = tradeoff_filter[label_mode]
            if isinstance(config, list):
                for val in config:
                    if label.endswith(f"_b{val}") or label.endswith(f"_kl{val}"):
                        return True
                return False
            elif isinstance(config, dict) and "exclude_values" in config:
                for val in config["exclude_values"]:
                    if label.endswith(f"_b{val}") or label.endswith(f"_kl{val}"):
                        return False
                return True
            return True
        filtered_labels = [lbl for lbl in labels if should_include(lbl)]
    else:
        filtered_labels = labels

    for idx, lbl in enumerate(filtered_labels):
        logs = epoch.get(lbl, {})
        ep = logs.get("epoch")
        if not ep:
            continue
        color, ls, mk = _style(lbl, idx)
        ep = np.array(ep, dtype=np.float32)
        axes[0].plot(ep, np.array(logs["val_recon"], dtype=np.float32),
                     label=lbl, color=color, linestyle=ls, marker=mk,
                     markersize=4, linewidth=1.4)
        axes[1].plot(ep, np.array(logs["val_kl"], dtype=np.float32),
                     label=lbl, color=color, linestyle=ls, marker=mk,
                     markersize=4, linewidth=1.4)

    for ax, ylabel, title in zip(
        axes,
        ["val reconstruction BCE", "val KL"],
        ["Val reconstruction per epoch", "Val KL per epoch"],
    ):
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=max(1, len(labels)//6))
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out = outdir / "vae_val_per_epoch.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


def plot_kl_recon_tradeoff(epoch, labels, outdir, run_params=None, tradeoff_filter=None):
    """
    KL vs recon scatter: final epoch only as a point.
    Each mode (adam_additive, ccsa_additive, ccsa_kl_constrained, ccsa_recon_constrained) 
    has a different marker. Same beta levels share the same color.
    
    Parameters
    ----------
    tradeoff_filter : dict or None
        Filter which runs to include. Example:
        {
            "ccsa_additive": ["1", "10"],
            "ccsa_kl_constrained": ["10"],
            "adam_additive": exclude_values=["0.1"]  # exclude specific values
        }
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Define markers by mode
    mode_markers = {
        "adam_additive": "o",
        "ccsa_additive": "D",
        "ccsa_kl_constrained": "*",
        "ccsa_recon_constrained": "^",
    }
    
    # Build filter logic
    def should_include(label):
        if tradeoff_filter is None:
            return True
        # Check which mode this label belongs to
        label_mode = None
        for mode in tradeoff_filter.keys():
            if mode in label:
                label_mode = mode
                break
        if label_mode is None:
            # Mode not in filter dict, exclude it
            return False
        
        config = tradeoff_filter[label_mode]
        if isinstance(config, list):
            # Include only these values (exact match after _b or _kl)
            for val in config:
                if label.endswith(f"_b{val}") or label.endswith(f"_kl{val}"):
                    return True
            return False
        elif isinstance(config, dict) and "exclude_values" in config:
            # Exclude specific values (exact match after _b or _kl)
            for val in config["exclude_values"]:
                if label.endswith(f"_b{val}") or label.endswith(f"_kl{val}"):
                    return False
            return True
        return True
    
    filtered_labels = [lbl for lbl in labels if should_include(lbl)]
    
    # Extract unique beta/kl values for coloring
    beta_values = set()
    for lbl in filtered_labels:
        # Extract beta or kl value
        if "_b" in lbl:
            val = lbl.split("_b")[-1]
            beta_values.add(val)
        elif "_kl" in lbl:
            val = lbl.split("_kl")[-1]
            beta_values.add(val)
    
    beta_values = sorted(beta_values, key=lambda x: float(x))
    
    # Define colors for different beta/kl values
    colors_palette = plt.cm.tab20(np.linspace(0, 1, max(20, len(beta_values))))
    beta_to_color = {val: colors_palette[idx] for idx, val in enumerate(beta_values)}
    
    for lbl in filtered_labels:
        logs = epoch.get(lbl, {})
        recons = logs.get("val_recon")
        kls    = logs.get("val_kl")
        if not recons or not kls:
            continue
        
        # Determine mode from label
        mode = None
        for m in mode_markers.keys():
            if m in lbl:
                mode = m
                break
        if mode is None:
            mode = "ccsa_additive"  # default
        
        # Extract beta/kl value for color
        beta_val = None
        if "_b" in lbl:
            beta_val = lbl.split("_b")[-1]
        elif "_kl" in lbl:
            beta_val = lbl.split("_kl")[-1]
        
        marker = mode_markers[mode]
        # Use black for ccsa_kl_constrained, otherwise use beta-based coloring
        if mode == "ccsa_kl_constrained":
            color = "black"
        else:
            color = beta_to_color.get(beta_val, "black")
        
        # Plot only the final epoch
        r_final = recons[-1]
        k_final = kls[-1]
        # Increase marker size for stars
        size = 300 if marker == "*" else 150
        ax.scatter(r_final, k_final, marker=marker, s=size, color=color, 
                   zorder=5, label=lbl, edgecolors='black', linewidth=0.5)

    ax.set_xlabel("val reconstruction BCE", fontsize=12)
    ax.set_ylabel("val KL divergence", fontsize=12)
    ax.set_title("KL–reconstruction tradeoff (final epoch)", fontsize=13, fontweight='bold')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc='best', frameon=True)
    plt.tight_layout()
    out = outdir / "vae_kl_recon_tradeoff.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")


def print_summary(epoch, labels):
    """Print a summary table from logged epoch metrics (no model needed)."""
    col = max(30, max(len(l) for l in labels) + 2)
    sep = "-" * (col + 48)
    print()
    print("=" * (col + 48))
    print(f"  Summary from last logged epoch")
    print("=" * (col + 48))
    print(f"{'run':<{col}}  {'val recon':>10}  {'val KL':>8}  {'train recon':>12}  {'train KL':>10}")
    print(sep)
    for lbl in labels:
        logs = epoch.get(lbl, {})
        vr = logs.get("val_recon",   [float("nan")])
        vk = logs.get("val_kl",      [float("nan")])
        tr = logs.get("train_recon", [float("nan")])
        tk = logs.get("train_kl",    [float("nan")])
        print(f"{lbl:<{col}}  {vr[-1]:>10.3f}  {vk[-1]:>8.4f}  {tr[-1]:>12.3f}  {tk[-1]:>10.4f}")
    print(sep)
    print()


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Re-plot a saved mnist_vae run.")
    p.add_argument("--rundir", required=True,
                   help="path to the run directory (contains results.npz, logs.json)")
    p.add_argument("--show", nargs="*", default=None,
                   help="labels to include (default: all). e.g. --show adam_additive_b1 "
                        "ccsa_kl_constrained_kl5")
    p.add_argument("--outdir", default=None,
                   help="where to save plots (default: same as --rundir)")
    p.add_argument("--log-x", action="store_true",
                   help="also log-scale the x-axis on batch-curve plots")
    p.add_argument("--tradeoff-filter", default=None,
                   help="JSON string to filter tradeoff plot. e.g. '{\"ccsa_additive\": [\"1\", \"10\"], "
                        "\"ccsa_kl_constrained\": [\"10\"], \"adam_additive\": {\"exclude_values\": [\"0.1\"]}}'")
    args = p.parse_args()

    rundir = Path(args.rundir)
    outdir = Path(args.outdir) if args.outdir else rundir
    outdir.mkdir(parents=True, exist_ok=True)

    batch, epoch, labels = load_run(rundir, args.show)
    print(f"[replot] {len(labels)} runs: {labels}")

    # load run_params for threshold lines
    rp_path = rundir / "run_params.json"
    run_params = None
    if rp_path.exists():
        with open(rp_path) as f:
            run_params = json.load(f)

    # Parse tradeoff filter if provided
    tradeoff_filter = None
    if args.tradeoff_filter:
        try:
            tradeoff_filter = json.loads(args.tradeoff_filter)
        except json.JSONDecodeError as e:
            print(f"[WARNING] Failed to parse --tradeoff-filter: {e}")

    plot_batch_curves(batch, labels, outdir, log_x=args.log_x, tradeoff_filter=tradeoff_filter)
    plot_epoch_curves(epoch, labels, outdir, tradeoff_filter=tradeoff_filter)
    plot_kl_recon_tradeoff(epoch, labels, outdir, run_params=run_params, tradeoff_filter=tradeoff_filter)
    print_summary(epoch, labels)
    print(f"[replot] done → {outdir}")


if __name__ == "__main__":
    main()
