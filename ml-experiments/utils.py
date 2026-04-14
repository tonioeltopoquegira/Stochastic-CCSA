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


import matplotlib.pyplot as plt
from typing import Optional, Dict, Any


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Experiments data loaders and standard transformations
def get_loaders(exp, batch_size=128, num_workers=4, pin_memory=False):
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
        tr = transforms.Compose([transforms.ToTensor()])
        te = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST("./data", train=True, download=True, transform=tr)
        test  = datasets.MNIST("./data", train=False, transform=te)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader  = DataLoader(test,  batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader

    elif exp == "cifar10_vae":
        # Filter CIFAR10 to use only images of the 0 class 
        tr = transforms.Compose([transforms.ToTensor()])
        te = transforms.Compose([transforms.ToTensor()])
        full_train = datasets.CIFAR10("./data", train=True, download=True, transform=tr)
        full_test  = datasets.CIFAR10("./data", train=False, transform=te)

        train_idx = [i for i, (_, t) in enumerate(full_train) if t == 0]
        test_idx = [i for i, (_, t) in enumerate(full_test) if t == 0]

        train = Subset(full_train, train_idx)
        test = Subset(full_test, test_idx)

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader  = DataLoader(test,  batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader
    else:
        raise ValueError("Unknown experiment key")

# Training epoch for classification models
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
        #print(f"Model parameters (first 5): {list(model.parameters())[0].data.flatten()[:5]}")
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

# Evaluation for classification models
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

# Train epoch for baseline for VAE
def train_epoch_vae(model, loader, optimizer, device, beta=1.0, show_progress=False, desc=""):
    model.train()
    batch_losses = []
    batch_recons = []
    batch_kls = []
    batch_evals = []

    iterator = loader
    for batch in iterator:
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)

        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        recon_loss, kl_loss = model.loss_components(recon, x, mu, logvar, reduction='mean')
        total_loss = recon_loss + beta * kl_loss
        total_loss.backward()
        optimizer.step()

        batch_losses.append(float(total_loss.item()))
        batch_recons.append(float(recon_loss.item()))
        batch_kls.append(float(kl_loss.item()))
        batch_evals.append(1.0)

    epoch_total = float(np.mean(batch_losses)) if batch_losses else None
    epoch_recon = float(np.mean(batch_recons)) if batch_recons else None
    epoch_kl = float(np.mean(batch_kls)) if batch_kls else None

    return epoch_total, epoch_recon, epoch_kl, batch_losses, batch_recons, batch_kls, batch_evals

# Evaluate baseline for VAE
@torch.no_grad()
def evaluate_vae(model, loader, device, beta=1.0):
    model.eval()
    totals, recons, kls = [], [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)
        recon, mu, logvar = model(x)
        recon_loss, kl_loss = model.loss_components(recon, x, mu, logvar, reduction='mean')
        totals.append(float((recon_loss + beta * kl_loss).item()))
        recons.append(float(recon_loss.item()))
        kls.append(float(kl_loss.item()))
    if not totals:
        return None, None, None
    return float(np.mean(totals)), float(np.mean(recons)), float(np.mean(kls))




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
    no_shared_budget: bool = False,
) -> Dict[str, Any]:
    """
    Postprocess training results and plot losses vs cumulative evals.

    Robust: normalizes lists->np.arrays, defends against missing/empty runs,
    and respects the `no_shared_budget` flag to show full curves when requested.

    ADDED: also attempts to create an epoch-only loss plot (saved as
    loss_vs_epochs[_<out_suffix>].png) when epoch summaries exist in `logs`.
    """
    import json
    import pickle
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    def load_results_from_folder(folder: Path) -> Dict[str, Any]:
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
                # If pickle can't be loaded, fall back to NPZ + logs_json below
                print(f"[WARN] Failed to load {all_results_pkl}: {e}")
                all_results = {}

        # Build results dict if pickle missing or empty
        if not all_results:
            losses_keys = [k for k in loaded.keys() if isinstance(k, str) and k.endswith("_losses")]
            for lk in losses_keys:
                opt_name = lk[: -len("_losses")]
                losses = loaded.get(lk, [])
                evals_key = f"{opt_name}_evals"
                evals = loaded.get(evals_key, [])
                logs = {}
                if "logs_json" in loaded and opt_name in loaded["logs_json"]:
                    logs = loaded["logs_json"][opt_name]
                all_results[opt_name] = (losses, evals, logs)

        return all_results

    # === Load runs ===
    all_results: Dict[str, Tuple[Any, Any, Any]] = {}
    for run_name, run_path in runs.items():
        run_path = Path(run_path)
        if not run_path.exists():
            print(f"[WARN] Run dir {run_path} not found, skipping.")
            continue
        run_results = load_results_from_folder(run_path)

        if optimizers is None:
            for opt_name, v in run_results.items():
                all_results[f"{run_name}_{opt_name}"] = v
        else:
            allowed_opts = {o.lower() for o in optimizers}
            for opt_name, v in run_results.items():
                if any(o in opt_name.lower() for o in allowed_opts):
                    all_results[f"{run_name}_{opt_name}"] = v

    # === Normalize results: convert losses/evals to 1D numpy arrays ===
    normalized_results: Dict[str, Tuple[np.ndarray, np.ndarray, Any]] = {}
    for name, trio in all_results.items():
        # Expect (losses, evals, logs)
        if not isinstance(trio, (list, tuple)) or len(trio) < 2:
            losses = np.array([], dtype=np.float32)
            evals = np.array([], dtype=np.float32)
            logs = {}
        else:
            losses_raw, evals_raw = trio[0], trio[1]
            logs = trio[2] if len(trio) > 2 else {}

            try:
                losses = np.asarray(losses_raw, dtype=np.float32)
            except Exception:
                losses = np.array([], dtype=np.float32)
            try:
                evals = np.asarray(evals_raw, dtype=np.float32)
            except Exception:
                evals = np.array([], dtype=np.float32)

            # Force to 1-D
            if losses.ndim > 1:
                losses = losses.reshape(-1)
            if evals.ndim > 1:
                evals = evals.reshape(-1)

            # If logs is None, use {}
            if logs is None:
                logs = {}
            elif not isinstance(logs, dict):
                try:
                    logs = dict(logs)
                except Exception:
                    pass

        normalized_results[name] = (losses, evals, logs)

    all_results = normalized_results
    opt_names = sorted(all_results.keys())

    # === Compute per-opt maximum evals (robustly) ===
    max_evals_per_opt: Dict[str, float] = {}
    for name, (_, evals, _) in all_results.items():
        max_val = 0.0
        try:
            if isinstance(evals, np.ndarray):
                max_val = float(evals.max()) if evals.size > 0 else 0.0
            else:
                # fallback for weird objects
                arr = np.asarray(evals, dtype=np.float32)
                max_val = float(arr.max()) if arr.size > 0 else 0.0
        except Exception:
            max_val = 0.0
        max_evals_per_opt[name] = max_val

    # === Shared budget logic ===
    if not max_evals_per_opt:
        shared_budget = 0.0
    else:
        if no_shared_budget:
            shared_budget = max(max_evals_per_opt.values())
        elif x_limit is not None:
            shared_budget = float(x_limit)
        elif plot_eval_limit is not None:
            shared_budget = float(plot_eval_limit)
        else:
            # Trim to shortest complete curve by default so all curves overlap
            shared_budget = float(min(max_evals_per_opt.values()))

    # === Color palette ===
    if color_map is None:
        okabe_ito = [
            "#0072B2", "#E69F00", "#009E73", "#D55E00",
            "#CC79A7", "#F0E442", "#56B4E9", "#000000"
        ]
        try:
            extra = [("#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255)))
                     for r, g, b, *_ in plt.cm.tab10.colors]
        except Exception:
            extra = []
        palette = okabe_ito + extra
        color_map = {name: palette[i % len(palette)] for i, name in enumerate(opt_names)}

    def style_for_name(name: str):
        lname = name.lower()
        color = color_map.get(name, "#000000")
        if "ensemble" in lname:
            return color, "-", 2.4
        elif "al" in lname or "active" in lname:
            return color, ":", 2.0
        else:
            return color, "-", 1.4

    # === Plotting (original batch-loss figure) ===
    fig, ax = plt.subplots(figsize=figsize)
    summary: Dict[str, Any] = {"shared_budget": shared_budget, "per_optimizer": {}}

    for opt_name in opt_names:
        losses, evals, logs = all_results[opt_name]

        # Require numpy arrays for plotting
        if not isinstance(losses, np.ndarray) or not isinstance(evals, np.ndarray):
            print(f"[WARN] Skipping {opt_name}: losses/evals not numpy arrays")
            continue

        if losses.size == 0 or evals.size == 0:
            # nothing to plot
            continue

        # Keep x,y as provided (we assume evals are already cumulative as you saved them)
        # === Convert to cumulative weighted evals ===
        if evals.size > 0:
            increments = np.diff(np.insert(evals, 0, 0.0))
            x = np.cumsum(increments)
        else:
            x = np.array([], dtype=np.float32)

        y = losses.copy()

        # If plot_every > 1, downsample AFTER trimming to shared budget so the sampled
        # points are within the displayed range.
        # Optionally trim to shared budget (only if we are not showing full curves)
        if not no_shared_budget:
            try:
                mask = x <= shared_budget
                if np.any(mask):
                    x = x[mask]
                    y = y[mask]
                else:
                    # No points within shared_budget: skip this curve to avoid plotting empty line
                    continue
            except Exception:
                # If comparison fails for some reason, skip trimming
                pass

        # apply explicit plot_eval_limit / x_limit if provided (they have priority)
        if plot_eval_limit is not None:
            try:
                mask = x <= float(plot_eval_limit)
                if np.any(mask):
                    x = x[mask]
                    y = y[mask]
                else:
                    continue
            except Exception:
                pass
        if x_limit is not None:
            try:
                mask = x <= float(x_limit)
                if np.any(mask):
                    x = x[mask]
                    y = y[mask]
                else:
                    continue
            except Exception:
                pass

        # downsample for plotting if requested
        if plot_every > 1 and x.size > plot_every:
            x = x[::plot_every]
            y = y[::plot_every]

        color, ls, lw = style_for_name(opt_name)
        ax.plot(x, y, label=opt_name.upper(),
                color=color, linestyle=ls, linewidth=lw,
                marker="o" if markers else None, markersize=5)

        # populate per-optimizer summary information
        summary["per_optimizer"][opt_name] = {
            "num_points": int(x.size),
            "max_eval_saved": float(max_evals_per_opt.get(opt_name, 0.0)),
            "final_eval_shown": float(x.max()) if x.size > 0 else 0.0,
        }

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("Cumulative weighted evals", fontsize=22)
    ax.set_ylabel("Batch loss", fontsize=22)
    ax.set_title("Batch loss vs evals", fontsize=24)
    ax.legend(loc=legend_loc)

    if plot_ylim is not None:
        ax.set_ylim(top=plot_ylim)

    # Save & show (original figure)
    fig_name = "loss_vs_evals"
    if out_suffix:
        fig_name += f"_{out_suffix}"
    fig_name += ".png"
    fig_path = Path.cwd() / fig_name
    if save_fig:
        try:
            fig.savefig(fig_path, dpi=180, bbox_inches="tight")
        except Exception as e:
            print(f"[WARN] Failed to save figure {fig_path}: {e}")
    if show_fig:
        plt.show()
    plt.close(fig)

    # === NEW: Epoch-only plot (added without changing original functionality) ===
    # This block will only create a plot if epoch summaries exist in the `logs` dict
    try:
        epoch_found = False
        fig_e, ax_e = plt.subplots(figsize=(18, 8))
        for opt_name in opt_names:
            losses, evals, logs = all_results[opt_name]
            # logs is expected to be a dict with 'train_eval_loss' and optionally 'epoch'
            if isinstance(logs, dict) and logs.get("train_eval_loss"):
                eloss = np.asarray(logs.get("train_eval_loss", []), dtype=np.float32)
                if eloss.size == 0:
                    continue
                eevals = np.asarray(logs.get("epoch", np.arange(1, len(eloss) + 1)), dtype=np.float32)
                # downsample epoch points if requested
                if plot_every > 1 and eevals.size > plot_every:
                    eevals = eevals[::plot_every]
                    eloss = eloss[::plot_every]
                color, ls, lw = style_for_name(opt_name)
                ax_e.plot(eevals, eloss, label=opt_name.upper(), color=color, linestyle=ls,
                          linewidth=lw, marker="o" if markers else None, markersize=6)
                epoch_found = True

        if epoch_found:
            ax_e.set_xlabel("Epoch", fontsize=16)
            ax_e.set_ylabel("Epoch loss", fontsize=16)
            ax_e.set_title("Epoch loss curves", fontsize=18)
            if log_y:
                ax_e.set_yscale("log")
            ax_e.legend(loc=legend_loc)
            if plot_ylim is not None:
                ax_e.set_ylim(top=plot_ylim)

            epoch_fig_name = "loss_vs_epochs"
            if out_suffix:
                epoch_fig_name += f"_{out_suffix}"
            epoch_fig_name += ".png"
            epoch_fig_path = Path.cwd() / epoch_fig_name
            if save_fig:
                try:
                    fig_e.savefig(epoch_fig_path, dpi=180, bbox_inches="tight")
                    print(f"[INFO] Saved epoch figure: {epoch_fig_path}")
                except Exception as e:
                    print(f"[WARN] Failed to save epoch figure {epoch_fig_path}: {e}")
            if show_fig:
                plt.show()
        else:
            # no epoch info available in logs for any run; close the axis and continue silently
            plt.close(fig_e)
            print("[INFO] No epoch-level logs found in any run; epoch plot skipped.")
    except Exception as _e:
        # Do not break the original behavior if epoch plotting fails for any reason
        print(f"[WARN] Epoch-plotting encountered an error: {_e}")

    return {"files": {"figure": str(fig_path) if save_fig else None},
            "shared_budget": shared_budget,
            "per_optimizer": summary["per_optimizer"]}












if __name__ == "__main__":
    import argparse
    import json as _json
    from pathlib import Path as _Path

    p = argparse.ArgumentParser(description="Postprocess and plot optimizer losses for multiple runs")

    # runs
    p.add_argument("--runs", type=str, nargs="+", required=True)

    # x / y limits and budget
    p.add_argument("--plot-eval-limit", type=float, default=None)
    p.add_argument("--x-limit", type=float, default=None)
    p.add_argument("--plot-ylim", type=float, default=None)

    # figure / output options
    p.add_argument("--no-save-fig", action="store_true")
    p.add_argument("--show-fig", action="store_true")
    p.add_argument("--out-suffix", type=str, default=None)

    # plotting style
    p.add_argument("--markers", action="store_true")
    p.add_argument("--log-x", action="store_true")
    p.add_argument("--log-y", action="store_true")
    p.add_argument("--legend-loc", type=str, default="best")

    # colors and sizing
    p.add_argument("--color-map", type=str, default=None)
    p.add_argument("--figsize", type=str, default=None)
    p.add_argument("--plot-every", type=int, default=1)
    p.add_argument("--optimizers", type=str, default=None)
    
    p.add_argument("--special-both",type=str,default=None)

    p.add_argument("--no-shared-budget", action="store_true")


    


    args = p.parse_args()

    
    if args.figsize:
        try:
            w, h = args.figsize.split(",")
            figsize = (float(w.strip()), float(h.strip()))
        except Exception as e:
            raise ValueError(f"Invalid --figsize value '{args.figsize}'. Use format WIDTH,HEIGHT (e.g. 9,4.8). Error: {e}")
    else:
        figsize = (18, 8)

    
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


    # run from the directory containing your run folder
    import numpy as np, pickle
    from pathlib import Path

    run_dir = Path("runs/mnist_cnn_seed0_20250910_001851")  # <- change to your run path




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
        special_both=special_both,
        no_shared_budget=args.no_shared_budget,  
    )




    if summary and "files" in summary and summary["files"].get("figure"):
        print(f"\nSaved figure: {summary['files']['figure']}")
