# data_utils.py
import time, json
import random
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Experiments data loaders
def get_loaders(exp, batch_size=128, num_workers=4, pin_memory=False, dataset_for_vit="cifar10"):
    if exp == "mnist_cnn":
        tr = transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        te = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train = datasets.MNIST("./data", train=True, download=True, transform=tr)
        test  = datasets.MNIST("./data", train=False, transform=te)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader  = DataLoader(test,  batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader
    elif exp == "cifar10_resnet32":
        mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
        tr = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
        te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train = datasets.CIFAR10("./data", train=True, download=True, transform=tr)
        test  = datasets.CIFAR10("./data", train=False, transform=te)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader  = DataLoader(test,  batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader
    elif exp == "cifar100_resnet56":
        mean = (0.5071, 0.4867, 0.4408); std = (0.2675, 0.2565, 0.2761)
        tr = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
        te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train = datasets.CIFAR100("./data", train=True, download=True, transform=tr)
        test  = datasets.CIFAR100("./data", train=False, transform=te)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader  = DataLoader(test,  batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader
    
    if exp == "mnist_vae":
        # no normalization -> pixels in [0,1]
        tr = transforms.Compose([transforms.ToTensor()])
        te = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST("./data", train=True, download=True, transform=tr)
        test  = datasets.MNIST("./data", train=False, transform=te)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader  = DataLoader(test,  batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader

    elif exp == "cifar10_vae":
        # Filter CIFAR10 to use only images of a single class (class_idx)
        tr = transforms.Compose([transforms.ToTensor()])
        te = transforms.Compose([transforms.ToTensor()])
        full_train = datasets.CIFAR10("./data", train=True, download=True, transform=tr)
        full_test  = datasets.CIFAR10("./data", train=False, transform=te)

        # create subsets where target == class_idx
        train_idx = [i for i, (_, t) in enumerate(full_train) if t == int(class_idx)]
        test_idx = [i for i, (_, t) in enumerate(full_test) if t == int(class_idx)]

        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"No examples found for class_idx={class_idx} in CIFAR10")

        train = Subset(full_train, train_idx)
        test = Subset(full_test, test_idx)

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader  = DataLoader(test,  batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader
    else:
        raise ValueError("Unknown experiment key")


def train_epoch(model, loader, optimizer, criterion, device, show_progress=False, desc=""):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    batch_losses, batch_evals = [], []

    iterator = tqdm(loader, desc=desc) if show_progress else loader

    for data, target in iterator:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss_tensor = criterion(outputs, target)
        loss_tensor.backward()
        optimizer.step()
        batch_loss = float(loss_tensor.item())
        preds = outputs.argmax(dim=1)
        weighted_evals = 1.0  # 1 eval per batch w/ our convention

        batch_size = data.size(0)
        total_loss += batch_loss * batch_size
        correct += preds.eq(target).sum().item()
        n += batch_size
        batch_losses.append(batch_loss)
        batch_evals.append(weighted_evals)

    return total_loss / n, correct / n, batch_losses, batch_evals


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        loss = criterion(outputs, target)
        total_loss += loss.item() * data.size(0)
        correct += outputs.argmax(dim=1).eq(target).sum().item()
        n += data.size(0)
    return total_loss / n, correct / n




import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any


def postprocess_losses(
    runs: Dict[str, str],
    plot_eval_limit: Optional[float] = None,
    x_limit: Optional[float] = None,
    plot_ylim: Optional[float] = None,
    save_fig: bool = True,
    show_fig: bool = False,
    out_suffix: Optional[str] = None,
    color_map: Optional[Dict[str, str]] = None,
    figsize: tuple = (35, 4.8),
    markers: bool = False,
    log_x: bool = False,
    log_y: bool = False,
    legend_loc: str = "best",
    plot_every: int = 1,  
    optimizers: Optional[list] = None,
    special_both: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Load and plot optimizer losses for multiple run directories.

    Args:
        runs: dict {label: run_dir_path}, e.g. {"Adam": "runs/adam", "SGD": "runs/sgd"}
    """

    def load_results_from_folder(folder: Path) -> Dict[str, Any]:
        """Helper to load results from results.npz, logs.json, or all_results.pkl"""
        results_npz = folder / "results.npz"
        logs_json = folder / "logs.json"
        all_results_pkl = folder / "all_results.pkl"

        loaded = {}
        if results_npz.exists():
            try:
                npz = np.load(results_npz, allow_pickle=True)
                for key in npz.keys():
                    loaded[key] = npz[key]
            except Exception as e:
                print(f"[WARN] Failed to load {results_npz}: {e}")

        if logs_json.exists():
            try:
                with open(logs_json, "r") as f:
                    loaded["logs_json"] = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load {logs_json}: {e}")

        all_results = {}
        if all_results_pkl.exists():
            try:
                with open(all_results_pkl, "rb") as f:
                    all_results = pickle.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load {all_results_pkl}: {e}")
                all_results = {}

        # Reconstruct if pickle missing
        if not all_results:
            losses_keys = [k for k in loaded.keys() if isinstance(k, str) and k.endswith("_losses")]
            for lk in losses_keys:
                opt_name = lk[: -len("_losses")]
                try:
                    losses = np.array(loaded[lk]).astype(np.float32)
                except Exception:
                    losses = np.array(loaded.get(lk, []), dtype=np.float32)
                evals_key = f"{opt_name}_evals"
                evals = np.array(loaded.get(evals_key, []), dtype=np.float32)
                logs = {}
                if "logs_json" in loaded and opt_name in loaded["logs_json"]:
                    logs = loaded["logs_json"][opt_name]
                all_results[opt_name] = (losses.tolist(), evals.tolist(), logs)

        # Normalize arrays
        for k, v in list(all_results.items()):
            losses, evals, logs = v
            losses = np.array(losses, dtype=np.float32)
            evals = np.array(evals, dtype=np.float32)
            if evals.size == 0 and losses.size > 0:
                evals = np.arange(1, losses.size + 1, dtype=np.float32)
            all_results[k] = (losses, evals, logs)

        return all_results

   
    # --- Load all runs ---
    all_results = {}
    for run_name, run_path in runs.items():
        run_path = Path(run_path)
        if not run_path.exists():
            print(f"[WARN] Run dir {run_path} not found, skipping.")
            continue
        run_results = load_results_from_folder(run_path)

        if optimizers is None:
            # Default rules with special-both exceptions
            opt_keys = list(run_results.keys())
            has_ccsa = any("ccsa" in k.lower() for k in opt_keys)
            has_adam = any("adam" in k.lower() for k in opt_keys)

            if has_ccsa and has_adam:
                if special_both and run_name in special_both:
                    # Keep both optimizers
                    for opt_name, v in run_results.items():
                        if "ccsa" in opt_name.lower() or "adam" in opt_name.lower():
                            all_results[f"{run_name}_{opt_name}"] = v
                else:
                    # Default: prefer only CCSA
                    for opt_name, v in run_results.items():
                        if "ccsa" in opt_name.lower():
                            all_results[f"{run_name}_{opt_name}"] = v
            else:
                # If only one optimizer exists, keep it
                for opt_name, v in run_results.items():
                    all_results[f"{run_name}_{opt_name}"] = v
        else:
            # Explicit filter from user
            allowed_opts = {o.lower() for o in optimizers}
            for opt_name, v in run_results.items():
                if any(o in opt_name.lower() for o in allowed_opts):
                    all_results[f"{run_name}_{opt_name}"] = v

    

    # --- Shared budget ---
    max_evals_per_opt = {name: float(evals.max()) if evals.size > 0 else 0.0
                         for name, (_, evals, _) in all_results.items()}
    if x_limit is not None:
        shared_budget = float(x_limit)
    elif plot_eval_limit is not None:
        shared_budget = float(plot_eval_limit)
    else:
        shared_budget = float(min(max_evals_per_opt.values()))

    # --- Color map ---
    opt_names = sorted(all_results.keys())
    if color_map is None:
        okabe_ito = [
            "#0072B2", "#E69F00", "#009E73", "#D55E00",
            "#CC79A7", "#F0E442", "#56B4E9", "#000000"
        ]
        extra = [("#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255)))
                 for r, g, b, *_ in plt.cm.tab10.colors]
        palette = okabe_ito + extra
        color_map = {name: palette[i % len(palette)] for i, name in enumerate(opt_names)}

    def style_for_name(name: str):
        lname = name.lower()
        color = color_map.get(name, "#000000")
        if "ensemble" in lname:
            ls = "-"
            lw = 2.4
        elif "+ al" in lname or "al" in lname or "active" in lname:
            ls = ":"
            lw = 2.0
        else:
            ls = "--" if "mlp" in lname else "-"
            lw = 1.4
        return color, ls, lw

    # --- Collect y-limits ---
    collected_losses = []
    for _, (losses, evals, _) in all_results.items():
        mask = evals <= shared_budget
        selected = losses if mask.sum() == 0 else losses[mask]
        if selected.size > 0:
            collected_losses.append(selected)
    if collected_losses:
        all_losses_concat = np.concatenate(collected_losses)
        y_min = float(np.percentile(all_losses_concat, 1))
        y_max = float(np.percentile(all_losses_concat, 95))
        if y_min < 0:
            y_min = 0.0
        y_span = max(y_max - y_min, 1e-6)
        y_min = max(0.0, y_min - 0.05 * y_span)
        y_max = y_max + 0.05 * y_span
    else:
        y_min, y_max = 0.0, 1.0

    if plot_ylim is not None:
        y_max = plot_ylim

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    summary = {"shared_budget": shared_budget, "per_optimizer": {}}

    for opt_name in opt_names:
        losses, evals, logs = all_results[opt_name]
        if losses.size == 0:
            continue
        color, ls, lw = style_for_name(opt_name)

        mask = evals <= shared_budget
        if mask.sum() == 0:
            x = evals
            y = losses
        else:
            x = evals[mask]
            y = losses[mask]

        # Extend to shared_budget
        if x.size > 1 and x[-1] < shared_budget:
            y_at_budget = float(np.interp(shared_budget, evals, losses))
            x = np.concatenate([x, [shared_budget]])
            y = np.concatenate([y, [y_at_budget]])
        elif x.size == 1:
            y_at_budget = float(np.interp(shared_budget, evals, losses))
            x = np.array([x[0], shared_budget], dtype=float)
            y = np.array([y[0], y_at_budget], dtype=float)

        # ↓↓↓ NEW: subsample for plotting
        if plot_every > 1 and x.size > plot_every:
            x = x[::plot_every]
            y = y[::plot_every]

        marker_symbol = None if not markers else "o"
        ax.plot(x, y, label=opt_name.upper(), color=color, linestyle=ls,
                linewidth=lw, marker=marker_symbol, markersize=5)

        # Metrics (always computed on full data, not subsampled)
        full_evals, full_losses = evals, losses
        if full_losses.size > 0:
            idx_best = int(np.nanargmin(full_losses))
            best_loss = float(full_losses[idx_best])
            eval_at_best = float(full_evals[idx_best])
            final_loss_at_budget = float(np.interp(shared_budget, full_evals, full_losses))
            auc = float(np.trapz(full_losses, full_evals)) if full_evals.size > 1 else 0.0
            auc_norm = auc / shared_budget if shared_budget > 0 else float("nan")

            summary["per_optimizer"][opt_name] = {
                "best_loss": best_loss,
                "eval_at_best": eval_at_best,
                "final_loss_at_budget": final_loss_at_budget,
                "auc": auc,
                "auc_norm": auc_norm,
                "max_eval_seen": float(full_evals.max()) if full_evals.size > 0 else 0.0,
                "n_points": int(full_losses.size),
            }

    if log_x:
        all_x = np.concatenate([all_results[n][1] for n in opt_names if all_results[n][1].size > 0])
        if np.any(all_x <= 0):
            raise ValueError("log_x=True but some evals <= 0; cannot log-scale x.")
        ax.set_xscale("log")
    if log_y:
        all_y = np.concatenate([all_results[n][0] for n in opt_names if all_results[n][0].size > 0])
        if np.any(all_y <= 0):
            raise ValueError("log_y=True but some losses <= 0; cannot log-scale y.")
        ax.set_yscale("log")

    ax.set_xlabel("Cumulative weighted evals", fontsize=22)
    ax.set_ylabel("Batch loss", fontsize=22)
    ax.set_title("Batch loss vs evals", fontsize=24)
    ax.set_ylim(bottom=0.0 if not log_y else None, top=y_max if not log_y else None)
    ax.set_xlim(left=10.0 if log_x else 0.0, right=shared_budget)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc=legend_loc)

    fig_name = "loss_vs_evals"
    if out_suffix:
        fig_name += f"_{out_suffix}"
    fig_name += ".png"
    fig_path = Path.cwd() / fig_name
    if save_fig:
        fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    if show_fig:
        plt.show()
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for opt_name in opt_names:
        _, _, logs = all_results[opt_name]
        if not logs or "epoch" not in logs:
            continue

        epochs = np.array(logs.get("epoch", []))
        train_eval_loss = np.array(logs.get("train_eval_loss", []), dtype=np.float32)
        val_eval_loss = np.array(logs.get("val_eval_loss", []), dtype=np.float32)

        color, ls, lw = style_for_name(opt_name)

        if train_eval_loss.size > 0:
            ax2.plot(
                epochs, train_eval_loss, label=f"{opt_name.upper()} (train)",
                color=color, linestyle=ls, linewidth=lw
            )
        if val_eval_loss.size > 0 and not np.all(np.isnan(val_eval_loss)):
            ax2.plot(
                epochs, val_eval_loss, label=f"{opt_name.upper()} (val)",
                color=color, linestyle=":", linewidth=lw
            )

    ax2.set_xlabel("Epoch", fontsize=20)
    ax2.set_ylabel("Eval loss", fontsize=20)
    ax2.set_title("Loss per epoch", fontsize=22)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend()

    fig2_name = "loss_per_epoch"
    if out_suffix:
        fig2_name += f"_{out_suffix}"
    fig2_name += ".png"
    fig2_path = Path.cwd() / fig2_name
    if save_fig:
        fig2.savefig(fig2_path, dpi=180, bbox_inches="tight")
    if show_fig:
        plt.show()
    plt.close(fig2)

    summary["files"]["epoch_figure"] = str(fig2_path) if save_fig else None


    print(f"\nPostprocess summary (budget={shared_budget:.2f}, y_max={y_max:.4f}):")
    hdr = f"{'OPT':20s} | {'best_loss':>10s} | {'eval@best':>10s} | {'final@budget':>12s} | {'auc_norm':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for opt_name in opt_names:
        metrics = summary["per_optimizer"].get(opt_name)
        if metrics:
            print(
                f"{opt_name:20s} | {metrics['best_loss']:10.4f} | {metrics['eval_at_best']:10.1f} | "
                f"{metrics['final_loss_at_budget']:12.4f} | {metrics['auc_norm']:9.6f}"
            )

    summary["files"] = {"figure": str(fig_path) if save_fig else None}
    return summary






if __name__ == "__main__":
    import argparse
    import json as _json
    from pathlib import Path as _Path

    p = argparse.ArgumentParser(description="Postprocess and plot optimizer losses for multiple runs")

    # runs
    p.add_argument("--runs", type=str, nargs="+", required=True,
                   help="Run directories to include, format: name=path. "
                        "Example: --runs adam=./runs/adam_seed0 sgd=./runs/sgd_run")

    # x / y limits and budget
    p.add_argument("--plot-eval-limit", type=float, default=None, help="Legacy eval limit (kept for compatibility)")
    p.add_argument("--x-limit", type=float, default=None, help="Explicit x-axis cumulative-eval limit (overrides plot-eval-limit)")
    p.add_argument("--plot-ylim", type=float, default=None, help="Optional y-axis top limit")

    # figure / output options
    p.add_argument("--no-save-fig", action="store_true", help="Do not save the produced figure")
    p.add_argument("--show-fig", action="store_true", help="Show the figure (useful in interactive sessions)")
    p.add_argument("--out-suffix", type=str, default=None, help="Suffix appended to the saved figure filename")

    # plotting style
    p.add_argument("--markers", action="store_true", help="Show markers on plotted lines")
    p.add_argument("--log-x", action="store_true", help="Plot x axis in log scale (requires positive x values)")
    p.add_argument("--log-y", action="store_true", help="Plot y axis in log scale (requires positive y values)")
    p.add_argument("--legend-loc", type=str, default="best", help="Legend location (matplotlib loc string)")

    # colors and sizing
    p.add_argument("--color-map", type=str, default=None,
                   help="JSON string or path to JSON file mapping optimizer_name->hex color. "
                        "Example JSON string: '{\"adam\":\"#ff0000\",\"ccsa\":\"#00ff00\"}'")
    p.add_argument("--figsize", type=str, default=None,
                   help="Figure size as two comma-separated floats, e.g. 9,4.8. If omitted uses default in function.")
    p.add_argument("--plot-every", type=int, default=1,
               help="Subsample factor: plot every Nth point (default=1 → plot all points)")
    p.add_argument(
    "--optimizers",
    type=str,
    default=None,
    help="Comma-separated list of optimizers to include. "
         "Default: keep only CCSA if available, otherwise keep all.")
    
    p.add_argument(
        "--special-both",
        type=str,
        default=None,
        help="Comma-separated run names where both Adam and CCSA should be plotted if present "
            "(e.g., --special-both 128,256,512)."
    )




    args = p.parse_args()

    # parse figsize if provided
    if args.figsize:
        try:
            w, h = args.figsize.split(",")
            figsize = (float(w.strip()), float(h.strip()))
        except Exception as e:
            raise ValueError(f"Invalid --figsize value '{args.figsize}'. Use format WIDTH,HEIGHT (e.g. 9,4.8). Error: {e}")
    else:
        figsize = (18, 8)

    # parse color_map: accept either a JSON string or path to JSON file
    parsed_color_map = None
    if args.color_map:
        s = args.color_map.strip()
        path_candidate = _Path(s)
        if path_candidate.exists():
            try:
                with open(path_candidate, "r") as f:
                    parsed_color_map = _json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load color map JSON from file {s}: {e}")
        else:
            try:
                parsed_color_map = _json.loads(s)
            except Exception as e:
                raise ValueError(
                    "Failed to parse --color-map. Provide a path to a JSON file or a valid JSON string. "
                    f"Error: {e}"
                )
    if args.optimizers:
        optimizers = [o.strip().lower() for o in args.optimizers.split(",")]
    else:
        optimizers = None


    # parse runs
    runs = {}
    for item in args.runs:
        if "=" not in item:
            raise ValueError(f"Invalid --runs entry '{item}', must be in name=path format")
        name, path = item.split("=", 1)
        runs[name.strip()] = path.strip()

    if args.special_both:
        special_both = [s.strip() for s in args.special_both.split(",")]
    else:
        special_both = None


    # call function with CLI args
    summary = postprocess_losses(
        runs=runs,
        plot_eval_limit=args.plot_eval_limit,
        x_limit=args.x_limit,
        plot_ylim=args.plot_ylim,
        save_fig=not args.no_save_fig,
        show_fig=args.show_fig,
        out_suffix=args.out_suffix,
        color_map=parsed_color_map,
        figsize=figsize,
        markers=args.markers,
        log_x=args.log_x,
        log_y=args.log_y,
        legend_loc=args.legend_loc,
        plot_every=args.plot_every,
        optimizers=optimizers,
        special_both=special_both,   # <-- NEW
    )




    if summary and "files" in summary and summary["files"].get("figure"):
        print(f"\nSaved figure: {summary['files']['figure']}")
