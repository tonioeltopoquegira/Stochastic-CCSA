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
def plot_batch_curves(batch, labels, outdir, log_x=False):
    """Three-panel: objective / recon / KL vs cumulative evals."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["objective (train batch)", "reconstruction BCE (train batch)", "KL (train batch)"]
    keys   = ["losses", "recons", "kls"]

    for idx, lbl in enumerate(labels):
        color, ls, _ = _style(lbl, idx)
        x = batch[lbl]["evals"]
        for ax, key in zip(axes, keys):
            y = batch[lbl][key]
            ax.plot(x, y, label=lbl, alpha=0.6, color=color, linestyle=ls, linewidth=1.2)

    for ax, title in zip(axes, titles):
        ax.set_xlabel("cumulative weighted evals")
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


def plot_epoch_curves(epoch, labels, outdir):
    """Two-panel: val recon / val KL per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, lbl in enumerate(labels):
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


def plot_kl_recon_tradeoff(epoch, labels, outdir, run_params=None):
    """
    KL vs recon scatter: each epoch is one point, coloured by run.
    Square marker = first epoch, star = last epoch, line connects epochs in order.
    Optional threshold lines if run_params is provided.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    for idx, lbl in enumerate(labels):
        logs = epoch.get(lbl, {})
        recons = logs.get("val_recon")
        kls    = logs.get("val_kl")
        if not recons or not kls:
            continue
        color, ls, _ = _style(lbl, idx)
        r = np.array(recons, dtype=np.float32)
        k = np.array(kls,    dtype=np.float32)
        ax.plot(r, k, linestyle="-", linewidth=1.0, alpha=0.5, color=color)
        ax.scatter(r[0],  k[0],  marker="s", s=60,  color=color, zorder=4)   # start
        ax.scatter(r[-1], k[-1], marker="*", s=120, color=color, zorder=5,   # end
                   label=lbl)

    if run_params:
        kl_thr    = run_params.get("kl_threshold")
        recon_thr = run_params.get("recon_threshold")
        if isinstance(kl_thr, list):
            for v in kl_thr:
                ax.axhline(v, color="gray", linestyle="--", linewidth=0.9,
                           label=f"KL ≤ {v}")
        elif kl_thr is not None:
            ax.axhline(kl_thr, color="gray", linestyle="--", linewidth=0.9,
                       label=f"KL ≤ {kl_thr}")
        if isinstance(recon_thr, list):
            for v in recon_thr:
                ax.axvline(v, color="silver", linestyle="-.", linewidth=0.9,
                           label=f"recon ≤ {v}")
        elif recon_thr is not None:
            ax.axvline(recon_thr, color="silver", linestyle="-.", linewidth=0.9,
                       label=f"recon ≤ {recon_thr}")

    ax.set_xlabel("val reconstruction BCE")
    ax.set_ylabel("val KL divergence")
    ax.set_title("KL–reconstruction tradeoff  (■=start  ★=end)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=max(1, len(labels)//8))
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    out = outdir / "vae_kl_recon_tradeoff.png"
    plt.savefig(out, dpi=150); plt.close()
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

    plot_batch_curves(batch, labels, outdir, log_x=args.log_x)
    plot_epoch_curves(epoch, labels, outdir)
    plot_kl_recon_tradeoff(epoch, labels, outdir, run_params=run_params)
    print_summary(epoch, labels)
    print(f"[replot] done → {outdir}")


if __name__ == "__main__":
    main()
