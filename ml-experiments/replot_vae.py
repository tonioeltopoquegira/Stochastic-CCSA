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
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def _make_beta_cmap(labels):
    
    import re
    beta_strs = set()
    for lbl in labels:
        ref = _adam_seed_base(lbl) or lbl
        if ("adam_additive" in ref or "ccsa_additive" in ref) and "_b" in ref:
            m = re.search(r"_b([0-9.]+)(?:_|$)", ref)
            if m:
                beta_strs.add(m.group(1))
    if not beta_strs:
        return {}
    betas = sorted(beta_strs, key=float)
    log_vals = np.array([np.log(float(b)) for b in betas], dtype=float)
    lo, hi = log_vals.min(), log_vals.max()
    cmap = plt.cm.Blues
    out = {}
    for b_str, lv in zip(betas, log_vals):
        # map [lo, hi] → [0.35, 0.90]: light (small beta) → dark (large beta)
        t = 0.35 + 0.55 * ((lv - lo) / (hi - lo) if hi > lo else 0.5)
        out[b_str] = cmap(t)
    return out


def _additive_beta_color(lbl, beta_cmap):
    """Return the beta-based color if lbl is an additive run, else None."""
    import re
    ref = _adam_seed_base(lbl) or lbl
    if ("adam_additive" in ref or "ccsa_additive" in ref) and "_b" in ref:
        m = re.search(r"_b([0-9.]+)(?:_|$)", ref)
        if m:
            b_str = m.group(1)
            return beta_cmap.get(b_str)
    return None


# ── loaders ─────────────────────────────────────────────────────────────────
def load_run(rundir: Path, show: list[str] | None):
    """
    Returns
    -------
    batch : dict  label → {losses, recons, kls, evals}  (from results.npz)
    epoch : dict  label → logs dict                      (from logs.json)
    best  : dict  label → {val_recon, val_kl}           (best checkpoint metrics from logs)
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

    batch, epoch, best = {}, {}, {}
    for lbl in labels:
        batch[lbl] = {
            "losses": data[f"{lbl}_losses"],
            "recons": data[f"{lbl}_recons"],
            "kls":    data[f"{lbl}_kls"],
            "evals":  data[f"{lbl}_evals"],
        }
        logs = epoch_logs.get(lbl, {})
        epoch[lbl] = logs

        # use the last logged epoch metrics.
        val_recons = logs.get("val_recon", [])
        val_kls = logs.get("val_kl", [])
        best[lbl] = {
            "val_recon": float(val_recons[-1]) if val_recons else float("nan"),
            "val_kl": float(val_kls[-1]) if val_kls else float("nan"),
            "source": "logs",
        }

    return batch, epoch, best, labels



def evaluate_checkpoints(rundir: Path, run_params: dict | None,
                         force: bool = False, batch_size: int = 256,
                         num_workers: int = 0, device_str: str | None = None):
   
    ckpts = (sorted(rundir.glob("best_*.pt"))
             + sorted(rundir.glob("last_*.pt"))
             + sorted(rundir.glob("min_violation_*.pt")))
    if not ckpts:
        return {}

    label_type_to_path = {}
    for p in ckpts:
        stem = p.stem
        if stem.startswith("best_feasible_"):
            key = stem[len("best_feasible_"):]
            ckpt_type = "best_feasible"
        elif stem.startswith("best_"):
            key = stem[len("best_"):]
            ckpt_type = "best"
        elif stem.startswith("last_"):
            key = stem[len("last_"):]
            ckpt_type = "last"
        elif stem.startswith("min_violation_"):
            key = stem[len("min_violation_"):]
            ckpt_type = "min_violation"
        else:
            key = stem
            ckpt_type = "best"
        label_type_to_path[(key, ckpt_type)] = p

    import torch
    from class_models import MNIST_VAE
    from utils import get_loaders, evaluate_vae

    latent_dim = int((run_params or {}).get("latent_dim", 20))
    hidden_dim = int((run_params or {}).get("hidden_dim", 400))

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    pin = (device.type == "cuda")
    train_loader, test_loader = get_loaders(
        exp="mnist_vae", batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin,
    )

    results: dict = {}
    print(f"[replot] evaluating {len(label_type_to_path)} checkpoints (device={device}) …")
    for (lbl, ckpt_type), path in label_type_to_path.items():
        cache_key = f"{ckpt_type}_{lbl}"
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            state = torch.load(path, map_location=device)
        model = MNIST_VAE(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
        model.load_state_dict(state)
        _, va_r, va_k = evaluate_vae(model, test_loader, device)
        _, tr_r, tr_k = evaluate_vae(model, train_loader, device)
        results[cache_key] = {
            "val_recon":   float(va_r),
            "val_kl":      float(va_k),
            "train_recon": float(tr_r),
            "train_kl":    float(tr_k),
            "ckpt_type":   ckpt_type,
            "label":       lbl,
        }
        print(f"    {lbl:<40s}  ({ckpt_type})  val recon={va_r:8.3f}  kl={va_k:8.4f}")

    return results


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
                    if label.endswith(f"_b{val}") or label.endswith(f"_kl{val}") or label.endswith(f"_r{val}"):
                        return True
                return False
            elif isinstance(config, dict) and "exclude_values" in config:
                for val in config["exclude_values"]:
                    if label.endswith(f"_b{val}") or label.endswith(f"_kl{val}") or label.endswith(f"_r{val}"):
                        return False
                return True
            return True
        filtered_labels = [lbl for lbl in labels if should_include(lbl)]
    else:
        filtered_labels = labels

    beta_cmap = _make_beta_cmap(filtered_labels)
    for idx, lbl in enumerate(filtered_labels):
        _, ls, _ = _style(lbl, idx)
        color = _additive_beta_color(lbl, beta_cmap) or _CMAP(idx % 10)
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
                    if label.endswith(f"_b{val}") or label.endswith(f"_kl{val}") or label.endswith(f"_r{val}"):
                        return True
                return False
            elif isinstance(config, dict) and "exclude_values" in config:
                for val in config["exclude_values"]:
                    if label.endswith(f"_b{val}") or label.endswith(f"_kl{val}") or label.endswith(f"_r{val}"):
                        return False
                return True
            return True
        filtered_labels = [lbl for lbl in labels if should_include(lbl)]
    else:
        filtered_labels = labels

    beta_cmap = _make_beta_cmap(filtered_labels)
    for idx, lbl in enumerate(filtered_labels):
        logs = epoch.get(lbl, {})
        ep = logs.get("epoch")
        if not ep:
            continue
        _, ls, mk = _style(lbl, idx)
        color = _additive_beta_color(lbl, beta_cmap) or _CMAP(idx % 10)
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


def _adam_seed_base(label: str) -> str | None:
    """If label is an adam seed run (e.g. adam_additive_s42), return the base label
    (e.g. adam_additive). Returns None if not a seed run."""
    import re
    m = re.match(r"^(adam_.+)_s\d+$", label)
    return m.group(1) if m else None


def _checkpoint_base_label(label: str) -> str:
    """Strip checkpoint-only prefixes so plots use the original run label."""
    for prefix in ("feasible_", "best_feasible_", "best_", "last_",
                   "min_violation_", "model_"):
        if label.startswith(prefix):
            return label[len(prefix):]
    return label


def _prefer_feasible_checkpoints(checkpoint_metrics: dict) -> dict:
    """Collapse multiple checkpoint variants for each run.

    Priority for a given run label:
      1) best_feasible_* (best objective among feasible iterates)
      2) min_violation_* (closest-to-feasible when nothing was feasible)
      3) best_*          (best objective overall, may be far from feasible)
      4) model_*         (restored-best copy saved at end of training;
                          also picks up manually dropped-in single-file runs)
      5) last_*          (final weights)
    """
    selected = {}
    for cache_key, metrics in checkpoint_metrics.items():
        ckpt_type = metrics.get("ckpt_type", "best")
        label = metrics.get("label")
        if label is None:
            label = cache_key

        rank = 0
        if ckpt_type == "best_feasible":
            rank = 5
        elif ckpt_type == "min_violation":
            rank = 4
        elif ckpt_type == "best":
            rank = 3
        elif ckpt_type == "last":
            rank = 2

        prev = selected.get(label)
        if prev is None or rank > prev[0]:
            selected[label] = (rank, metrics)

    return {lbl: metrics for lbl, (_rank, metrics) in selected.items()}


def plot_kl_recon_tradeoff(epoch, best, labels, outdir, run_params=None, tradeoff_filter=None):
    """
    KL vs recon scatter: best checkpoint metrics as points.
    Each mode (adam_additive, ccsa_additive, ccsa_kl_constrained, ccsa_recon_constrained)
    has a different marker. Same beta levels share the same color.
    Adam seed runs (label pattern adam_*_s{N}) are shown as dots of the same
    color and shape as their base group, with a single shared legend entry.

    Parameters
    ----------
    best : dict
        Best-checkpoint metrics: label → {val_recon, val_kl}
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
        # For adam seed runs, apply the filter against the base label
        base = _adam_seed_base(label)
        check_label = base if base else label
        label_mode = None
        for mode in tradeoff_filter.keys():
            if mode in check_label:
                label_mode = mode
                break
        if label_mode is None:
            return False
        config = tradeoff_filter[label_mode]
        if isinstance(config, list):
            for val in config:
                if check_label.endswith(f"_b{val}") or check_label.endswith(f"_kl{val}") or check_label.endswith(f"_r{val}"):
                    return True
            return False
        elif isinstance(config, dict) and "exclude_values" in config:
            for val in config["exclude_values"]:
                if check_label.endswith(f"_b{val}") or check_label.endswith(f"_kl{val}") or check_label.endswith(f"_r{val}"):
                    return False
            return True
        return True

    filtered_labels = [lbl for lbl in labels if should_include(lbl)]

    beta_cmap = _make_beta_cmap(filtered_labels)

    # Track which legend labels have already been added
    legend_seen = set()

    for lbl in filtered_labels:
        best_metrics = best.get(lbl, {})
        r_final = best_metrics.get("val_recon")
        k_final = best_metrics.get("val_kl")
        if r_final is None or k_final is None:
            continue

        # For seed runs, use the base label for styling and legend
        base = _adam_seed_base(lbl)
        ref = base if base else lbl
        legend_label = ref if base else lbl

        # Determine mode
        mode = None
        for m in mode_markers.keys():
            if m in ref:
                mode = m
                break
        if mode is None:
            mode = "ccsa_additive"

        marker = mode_markers[mode]
        # Additive modes: log-scale beta coloring (Blues, darker=larger)
        # Constrained modes: black / fixed fallback
        color = _additive_beta_color(lbl, beta_cmap)
        if color is None:
            if mode == "ccsa_kl_constrained":
                color = "red"
            elif mode == "ccsa_recon_constrained":
                color = "green"
            else:
                color = "black"

        size = 300 if marker == "*" else 150

        # Only add a legend entry the first time this group label appears
        plot_label = legend_label if legend_label not in legend_seen else "_nolegend_"
        legend_seen.add(legend_label)

        ax.scatter(r_final, k_final, marker=marker, s=size, color=color,
                   zorder=5, label=plot_label, edgecolors='black', linewidth=0.5)

    # Constraint threshold lines
    kl_thresholds, recon_thresholds = set(), set()
    for lbl in filtered_labels:
        if "kl_constrained_kl" in lbl:
            try: kl_thresholds.add(float(lbl.split("_kl")[-1]))
            except ValueError: pass
        elif "recon_constrained_r" in lbl:
            try: recon_thresholds.add(float(lbl.split("_r")[-1]))
            except ValueError: pass
    for thr in sorted(kl_thresholds):
        ax.axhline(thr, color='red', linestyle='--', linewidth=1.0, alpha=0.6,
                   zorder=2, label=f"KL ≤ {thr:g}")
    for thr in sorted(recon_thresholds):
        ax.axvline(thr, color='green', linestyle='--', linewidth=1.0, alpha=0.6,
                   zorder=2, label=f"recon ≤ {thr:g}")

    ax.set_xlabel("val reconstruction BCE", fontsize=12)
    ax.set_ylabel("val KL divergence", fontsize=12)
    ax.set_title("KL–reconstruction tradeoff", fontsize=13, fontweight='bold')
    ax.set_xscale("log")
    ax.set_yscale("log")
    #ax.legend(fontsize=8, loc='best', frameon=True)

    # Add a colorbar showing the beta scale (log) if there are additive runs
    if beta_cmap:
        import matplotlib.colors as mcolors
        beta_floats = sorted(float(b) for b in beta_cmap)
        norm = mcolors.LogNorm(vmin=beta_floats[0], vmax=beta_floats[-1])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("beta", fontsize=10)

    # Simple horizontal legend above the plot, beside the title
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='^', color='green', linestyle='None', markersize=10),
        Line2D([0], [0], marker='*', color='red', linestyle='None', markersize=12),
        Line2D([0], [0], marker='o', color='blue', linestyle='None', markersize=8),
    ]
    legend_labels = ["reconstruction-constrained", "kl-constrained", "fixed-beta adam"]

    # White background, no legend frame. Place legend above axes and to the right of title.
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
   
    ax.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(0.55, 1.18),
              frameon=False, ncol=3, fontsize=10)

    plt.tight_layout()
    out = outdir / "vae_kl_recon_tradeoff.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out}")


def print_summary(best, labels):
    """Print a summary table from best checkpoint metrics."""
    col = max(30, max(len(l) for l in labels) + 2)
    sep = "-" * (col + 30)
    print()
    print("=" * (col + 30))
    print(f"  Summary from best checkpoint")
    print("=" * (col + 30))
    print(f"{'run':<{col}}  {'val recon':>10}  {'val KL':>8}")
    print(sep)
    for lbl in labels:
        best_metrics = best.get(lbl, {})
        vr = best_metrics.get("val_recon", float("nan"))
        vk = best_metrics.get("val_kl", float("nan"))
        print(f"{lbl:<{col}}  {vr:>10.3f}  {vk:>8.4f}")
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
    p.add_argument("--max-beta", type=float, default=None,
                   help="exclude additive runs whose beta is greater than this value")
    p.add_argument("--recompute", action="store_true",
                   help="force re-evaluation of all best_*.pt checkpoints, ignoring "
                        "the tradeoff_metrics.json cache")
    p.add_argument("--eval-device", default=None, dest="eval_device",
                   help="device for checkpoint evaluation (e.g. cpu, cuda). "
                        "Default: cuda if available.")
    args = p.parse_args()

    rundir = Path(args.rundir)
    outdir = Path(args.outdir) if args.outdir else rundir
    outdir.mkdir(parents=True, exist_ok=True)

    batch, epoch, best, labels = load_run(rundir, args.show)
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

    # Tradeoff plot
    ckpt_metrics = evaluate_checkpoints(
        rundir, run_params, force=args.recompute,
        device_str=args.eval_device,
    )
    if ckpt_metrics:
        preferred_ckpts = _prefer_feasible_checkpoints(ckpt_metrics)
        best_ckpt = {lbl: {"val_recon": m["val_recon"], "val_kl": m["val_kl"]}
                     for lbl, m in preferred_ckpts.items()}

        if args.show:
            tradeoff_labels = [l for l in args.show if l in preferred_ckpts]
            missing = [l for l in args.show if l not in preferred_ckpts]
            if missing:
                print(f"[WARN] --show labels not found among checkpoints, "
                      f"tradeoff will skip: {missing}")
        else:
            tradeoff_labels = sorted(best_ckpt.keys())

        # Apply numeric beta filter if requested (exclude additive runs with beta > max_beta)
        if args.max_beta is not None:
            import re
            def _extract_beta(lbl):
                ref = _adam_seed_base(lbl) or lbl
                m = re.search(r"_b([0-9.]+)(?:_|$)", ref)
                return float(m.group(1)) if m else None

            before = len(tradeoff_labels)
            tradeoff_labels = [l for l in tradeoff_labels if (_extract_beta(l) is None or _extract_beta(l) <= args.max_beta)]
            after = len(tradeoff_labels)
            print(f"[replot] filtered tradeoff labels by max_beta={args.max_beta}: {before} -> {after}")

        # Drop ccsa_additive
        def _keep(lbl):
            base = _adam_seed_base(lbl) or lbl
            return not (base == "ccsa_additive" or base.startswith("ccsa_additive_"))

        dropped = [l for l in tradeoff_labels if not _keep(l)]
        tradeoff_labels = [l for l in tradeoff_labels if _keep(l)]
        if dropped:
            print(f"[replot] dropped from tradeoff (ccsa_additive): {dropped}")
        print(f"[replot] tradeoff over {len(tradeoff_labels)} checkpoints")
        plot_kl_recon_tradeoff(epoch, best_ckpt, tradeoff_labels, outdir,
                               run_params=run_params, tradeoff_filter=tradeoff_filter)
        print_summary(best_ckpt, tradeoff_labels)
    else:
        print("[replot] no best_*.pt checkpoints found — "
              "falling back to logs.json for tradeoff plot")
        plot_kl_recon_tradeoff(epoch, best, labels, outdir,
                               run_params=run_params, tradeoff_filter=tradeoff_filter)
        print_summary(best, labels)

    print(f"[replot] done → {outdir}")


if __name__ == "__main__":
    main()
