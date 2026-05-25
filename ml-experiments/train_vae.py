"""
Train MNIST VAE under four objective formulations:

  adam_additive          Adam, minimize recon + beta * KL
  ccsa_additive          CCSA, minimize recon + beta * KL  (no constraints)
  ccsa_kl_constrained    CCSA, minimize recon  s.t. KL <= kl_threshold
  ccsa_recon_constrained CCSA, minimize KL     s.t. recon <= recon_threshold

The CCSA modes wrap ccsa.optimizer.CCSAOptimizer directly so we can pass
constraint callables (g, dg). Inside each outer step the same minibatch is
shared between objective and constraint evaluations.
"""

import argparse
import copy
import dataclasses
import itertools
import json
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from class_models import MNIST_VAE
from utils import (set_seed, get_loaders, train_epoch_vae, evaluate_vae)
from ccsa.optimizer import CCSAOptimizer as CCSAOptimizer_Core
from ccsa.params import MMA_RhoParams, MMA_SigmaParams


# ---------------------------------------------------------------------------
# Parameter packing helpers
# ---------------------------------------------------------------------------
def pack_params(params: List[torch.Tensor]):
    shapes = [tuple(p.shape) for p in params]
    sizes = [p.numel() for p in params]
    x0 = np.concatenate(
        [p.detach().cpu().numpy().ravel().astype(np.float64) for p in params]
    ) if params else np.zeros(0, dtype=np.float64)
    return x0, shapes, sizes


def unpack_to_params(x: np.ndarray, params: List[torch.Tensor], shapes, sizes):
    offset = 0
    for p, shape, size in zip(params, shapes, sizes):
        chunk = x[offset:offset + size].reshape(shape)
        with torch.no_grad():
            p.copy_(torch.tensor(chunk, dtype=p.dtype, device=p.device))
        offset += size


def flatten_grad(g_tensors, params):
    parts = []
    for gt, p in zip(g_tensors, params):
        if gt is None:
            parts.append(np.zeros(p.numel(), dtype=np.float64))
        else:
            parts.append(gt.detach().cpu().numpy().ravel().astype(np.float64))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


# ---------------------------------------------------------------------------
# Adam baseline
# ---------------------------------------------------------------------------
def run_adam(model, train_loader, test_loader, device, args, outdir: Path,
             label: str = "adam_additive"):
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    batch_losses, batch_recons, batch_kls, all_evals = [], [], [], []
    cumulative_eval = 0.0
    logs = {"epoch": [], "train_total": [], "train_recon": [], "train_kl": [],
            "val_total": [], "val_recon": [], "val_kl": [], "time": [],
            "best_epoch": [], "best_val_loss": []}
    t0 = time.time()
    best_val_loss = float('inf')
    best_state_dict = None
    ckpt_path = outdir / f"best_{label}.pt"

    for epoch in range(1, args.epochs + 1):
        ep_total, ep_recon, ep_kl, bl, br, bk, be = train_epoch_vae(
            model, train_loader, optimizer, device, beta=args.beta,
            show_progress=True, desc=f"{label} ep{epoch}"
        )
        batch_losses.extend(bl); batch_recons.extend(br); batch_kls.extend(bk)
        for e in be:
            cumulative_eval += e
            all_evals.append(cumulative_eval)

        val_total, val_recon, val_kl = evaluate_vae(model, test_loader, device, beta=args.beta)
        val_combined = val_recon + args.beta * val_kl   # objective for additive mode
        logs["epoch"].append(epoch)
        logs["train_total"].append(ep_total); logs["train_recon"].append(ep_recon); logs["train_kl"].append(ep_kl)
        logs["val_total"].append(val_total); logs["val_recon"].append(val_recon); logs["val_kl"].append(val_kl)
        logs["time"].append(time.time() - t0)

        improved = val_combined < best_val_loss
        marker = " ✓ best" if improved else ""
        print(f"[{label}] ep {epoch}: train recon={ep_recon:.2f} kl={ep_kl:.3f} | "
              f"val recon={val_recon:.2f} kl={val_kl:.3f}{marker}")
        if improved:
            best_val_loss = val_combined
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state_dict, ckpt_path)
            logs["best_epoch"].append(epoch)
            logs["best_val_loss"].append(best_val_loss)

    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
        print(f"[{label}] Restored best checkpoint (epoch {logs['best_epoch'][-1]}, "
              f"val combined={best_val_loss:.2f})")

    return batch_losses, batch_recons, batch_kls, all_evals, logs


# ---------------------------------------------------------------------------
# CCSA driver (covers all three CCSA modes)
# ---------------------------------------------------------------------------
class TrainingComplete(Exception):
    pass


def run_ccsa(model, train_loader, test_loader, device, args, mode: str, outdir: Path,
             label: str = None):
    """
    mode in {"additive", "kl_constrained", "recon_constrained"}.
    label is the unique run identifier used for filenames and logs;
    defaults to ccsa_{mode} if not provided.
    Evaluates validation loss every --checkpoint-every outer steps and saves
    best_<label>.pt whenever it improves. Restores best weights at the end.
    """
    tag = label if label is not None else f"ccsa_{mode}"
    params = [p for p in model.parameters() if p.requires_grad]
    x0, shapes, sizes = pack_params(params)
    batches_per_epoch = len(train_loader)
    total_outer = int(args.epochs * batches_per_epoch)

    train_iter = itertools.cycle(train_loader)

    # State shared by callables for this run
    state = {
        "batch": None,
        "cache_x": None,
        "recon": None,
        "kl": None,
        "recon_grad": None,
        "kl_grad": None,
        "outer_calls": 0,
        "inner_calls": 0,
        "cumulative_eval": 0.0,
    }

    # Best-checkpoint tracking (metric: val_recon + val_kl, beta=1, same for all modes)
    best_val_loss = float('inf')
    best_state_dict = None
    ckpt_path = outdir / f"best_{tag}.pt"

    batch_losses, batch_recons, batch_kls = [], [], []
    cumulative_evals_trace = []
    logs = {"epoch": [], "train_total": [], "train_recon": [], "train_kl": [],
            "val_total": [], "val_recon": [], "val_kl": [], "time": [],
            "best_step": [], "best_val_loss": []}
    t0 = time.time()

    def _fetch_new_batch():
        data, _ = next(train_iter)
        state["batch"] = data.to(device)

    def _forward_with_grad(x: np.ndarray):
        unpack_to_params(x, params, shapes, sizes)
        model.train()
        data = state["batch"]
        recon, mu, logvar = model(data)
        bce, kl = model.loss_components(recon, data, mu, logvar, reduction='mean')
        g_bce = torch.autograd.grad(bce, params, retain_graph=True, allow_unused=True)
        g_kl  = torch.autograd.grad(kl,  params, retain_graph=False, allow_unused=True)
        state["recon"] = float(bce.item())
        state["kl"]    = float(kl.item())
        # clip each gradient to unit norm to prevent surrogate-step explosions
        _clip = lambda g: g / max(1.0, float(np.linalg.norm(g)))
        state["recon_grad"] = _clip(flatten_grad(g_bce, params))
        state["kl_grad"]    = _clip(flatten_grad(g_kl,  params))
        state["cache_x"] = id(x)

    def _forward_only(x: np.ndarray):
        unpack_to_params(x, params, shapes, sizes)
        model.eval()
        data = state["batch"]
        with torch.no_grad():
            recon, mu, logvar = model(data)
            bce, kl = model.loss_components(recon, data, mu, logvar, reduction='mean')
        state["recon"] = float(bce.item())
        state["kl"]    = float(kl.item())
        state["recon_grad"] = None
        state["kl_grad"]    = None
        state["cache_x"] = id(x)

    def _maybe_checkpoint():
        nonlocal best_val_loss, best_state_dict
        ckpt_every = getattr(args, 'checkpoint_every', 50)
        ckpt_warmup = getattr(args, 'checkpoint_warmup', 200)
        if state["outer_calls"] % ckpt_every != 0:
            return
        if state["outer_calls"] < ckpt_warmup:
            return
        val_total, val_recon, val_kl = evaluate_vae(model, test_loader, device, beta=args.beta)
        # Track the actual objective for this mode
        if mode == "additive":
            tracked = val_recon + args.beta * val_kl
            tracked_label = f"recon+{args.beta}*kl={tracked:.2f}"
        elif mode == "kl_constrained":
            tracked = val_recon
            tracked_label = f"recon={tracked:.2f}"
        elif mode == "recon_constrained":
            tracked = val_kl
            tracked_label = f"kl={tracked:.4f}"
        improved = tracked < best_val_loss
        if improved:
            best_val_loss = tracked
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state_dict, ckpt_path)
            logs["best_step"].append(state["outer_calls"])
            logs["best_val_loss"].append(best_val_loss)
            print(f"  [{tag}] step {state['outer_calls']:5d} ✓ new best  "
                  f"val recon={val_recon:.2f}  kl={val_kl:.4f}  "
                  f"{tracked_label}  → saved {ckpt_path.name}")
        else:
            print(f"  [{tag}] step {state['outer_calls']:5d}   "
                  f"val recon={val_recon:.2f}  kl={val_kl:.4f}  "
                  f"{tracked_label}  (best={best_val_loss:.2f})")

    def _maybe_log_epoch():
        if batches_per_epoch <= 0 or state["outer_calls"] % batches_per_epoch != 0:
            return
        epoch_idx = state["outer_calls"] // batches_per_epoch
        val_total, val_recon, val_kl = evaluate_vae(model, test_loader, device, beta=args.beta)
        train_total, train_recon, train_kl = evaluate_vae(model, train_loader, device, beta=args.beta)
        logs["epoch"].append(epoch_idx)
        logs["train_total"].append(train_total); logs["train_recon"].append(train_recon); logs["train_kl"].append(train_kl)
        logs["val_total"].append(val_total); logs["val_recon"].append(val_recon); logs["val_kl"].append(val_kl)
        logs["time"].append(time.time() - t0)
        print(f"[{tag}] ep {epoch_idx}: train recon={train_recon:.2f} kl={train_kl:.3f} | "
              f"val recon={val_recon:.2f} kl={val_kl:.3f}")

    # -------- objective callable --------
    def objective(x, grad=None):
        if state["outer_calls"] >= total_outer:
            raise TrainingComplete()

        if grad is True:
            _fetch_new_batch()
            _forward_with_grad(x)
            state["outer_calls"] += 1
            state["inner_calls"] = 0
            state["cumulative_eval"] += 1.0
        else:
            _forward_only(x)
            state["inner_calls"] += 1
            state["cumulative_eval"] += float(args.inner_eval_weight)

        if mode == "additive":
            f_val  = state["recon"] + args.beta * state["kl"]
            f_grad = (state["recon_grad"] + args.beta * state["kl_grad"]) if grad is True else None
        elif mode == "kl_constrained":
            f_val  = state["recon"]
            f_grad = state["recon_grad"] if grad is True else None
        elif mode == "recon_constrained":
            f_val  = state["kl"]
            f_grad = state["kl_grad"] if grad is True else None
        else:
            raise ValueError(f"unknown mode {mode}")

        batch_losses.append(f_val)
        batch_recons.append(state["recon"])
        batch_kls.append(state["kl"])
        cumulative_evals_trace.append(state["cumulative_eval"])

        if grad is True:
            _maybe_checkpoint()
            _maybe_log_epoch()
            return (f_val, f_grad)
        return f_val

    # -------- constraint callables (only for constrained modes) --------
    if mode == "kl_constrained":
        def g_fun(x):
            if state["cache_x"] != id(x):
                _forward_only(x)
            return np.array([state["kl"] - args.kl_threshold], dtype=np.float64)

        def dg_fun(x):
            if state["cache_x"] != id(x) or state["kl_grad"] is None:
                _forward_with_grad(x)
            return state["kl_grad"].reshape(1, -1)

    elif mode == "recon_constrained":
        def g_fun(x):
            if state["cache_x"] != id(x):
                _forward_only(x)
            return np.array([state["recon"] - args.recon_threshold], dtype=np.float64)

        def dg_fun(x):
            if state["cache_x"] != id(x) or state["recon_grad"] is None:
                _forward_with_grad(x)
            return state["recon_grad"].reshape(1, -1)

    else:
        g_fun = None
        dg_fun = None

    # -------- CCSA setup --------
    rho_params   = MMA_RhoParams()
    sigma_params = MMA_SigmaParams(sigma_min=1e-2)

    ccsa_config = {
        "use_quadratic_surrogates": True,
        "conservative": False,
        "max_inner": 1,
        "rho_init": 1.0,
        "update_rule": "multiplier",
        "store_history": False,
        "gradient_clip_norm": 1.0,
        "rho_params": dataclasses.asdict(rho_params),
        "sigma_params": dataclasses.asdict(sigma_params),
    }
    with open(outdir / f"ccsa_config_{tag}.json", "w") as _f:
        json.dump(ccsa_config, _f, indent=2)

    ccsa_opt = CCSAOptimizer_Core(
        params=x0,
        fun=objective,
        g=g_fun,
        dg=dg_fun,
        bounds=None,
        use_quadratic_surrogates=True,
        conservative=False,
        max_inner=1,
        rho_init=1.0,
        sigma_min=1e-2,
        update_rule='multiplier',
        update_rule_kwargs={'lr': 5e-1, 'beta1': 0.05, 'beta2': 0.2, 'eps': 1e-8,
                            'min_curv': 1e-4, 'max_curv': 10000.0},
        store_history=False,
    )

    pbar = tqdm(total=total_outer, desc=f"CCSA[{mode}]", unit="step")
    try:
        for _ in range(total_outer):
            ccsa_opt.step()
            pbar.update(1)
            if state["outer_calls"] >= total_outer:
                break
    except TrainingComplete:
        pass
    finally:
        pbar.close()

    unpack_to_params(ccsa_opt.x_k, params, shapes, sizes)

    # Restore best checkpoint into model
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
        print(f"[{tag}] Restored best checkpoint (val combined={best_val_loss:.2f}) from {ckpt_path.name}")
    else:
        print(f"[{tag}] No checkpoint saved — using final weights.")

    return batch_losses, batch_recons, batch_kls, cumulative_evals_trace, logs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(all_results, outdir: Path, exp: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for mode_name, (losses, recons, kls, evals, _logs) in all_results.items():
        x = np.array(evals, dtype=np.float32)
        axes[0].plot(x, np.array(losses, dtype=np.float32), label=mode_name, alpha=0.7)
        axes[1].plot(x, np.array(recons, dtype=np.float32), label=mode_name, alpha=0.7)
        axes[2].plot(x, np.array(kls, dtype=np.float32), label=mode_name, alpha=0.7)
    for ax, title in zip(axes, ["objective", "reconstruction (BCE)", "KL"]):
        ax.set_xlabel("cumulative weighted evals")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs evals ({exp})")
        ax.legend(); ax.grid(True, alpha=0.3)
    axes[0].set_yscale('log'); axes[1].set_yscale('log'); axes[2].set_yscale('log')
    plt.tight_layout()
    plt.savefig(outdir / "vae_curves_vs_evals.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for mode_name, (_l, _r, _k, _e, logs) in all_results.items():
        if not logs["epoch"]:
            continue
        ep = np.array(logs["epoch"], dtype=np.float32)
        axes[0].plot(ep, np.array(logs["val_recon"], dtype=np.float32), marker='o', label=mode_name)
        axes[1].plot(ep, np.array(logs["val_kl"], dtype=np.float32), marker='o', label=mode_name)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("val reconstruction BCE")
    axes[0].set_title("Val recon per epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val KL")
    axes[1].set_title("Val KL per epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[0].set_yscale('log'); axes[1].set_yscale('log')
    plt.tight_layout()
    plt.savefig(outdir / "vae_val_per_epoch.png", dpi=150)
    plt.close()


def save_reconstructions(model, test_loader, device, outdir: Path, tag: str, n: int = 8):
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:n].to(device)
        recon, _, _ = model(data)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.3, 3))
    for i in range(n):
        axes[0, i].imshow(data[i, 0].cpu().numpy(), cmap='gray'); axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].view(28, 28).cpu().numpy(), cmap='gray'); axes[1, i].axis('off')
    axes[0, 0].set_title("orig", loc='left'); axes[1, 0].set_title("recon", loc='left')
    plt.tight_layout()
    plt.savefig(outdir / f"reconstructions_{tag}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("[INFO] Device:", device)

    num_workers = 0 if device.type == "cpu" else args.num_workers
    train_loader, test_loader = get_loaders(
        exp="mnist_vae", batch_size=args.batch_size,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    if args.mode == "all":
        base_modes = ["adam_additive", "ccsa_additive", "ccsa_kl_constrained", "ccsa_recon_constrained"]
    else:
        base_modes = [args.mode]

    # Expand each base mode over its sweep parameters to get a flat run list.
    # Each entry: (label, base_mode, run_args) where run_args has the
    # specific beta / kl_threshold / recon_threshold for that run.
    def _label(base, suffix):
        return f"{base}_{suffix}" if suffix else base

    runs = []   # (label, base_mode, run_args)
    for base_mode in base_modes:
        if base_mode in ("adam_additive", "ccsa_additive"):
            multi = len(args.beta) > 1
            for beta in args.beta:
                ra = copy.copy(args)
                ra.beta = beta
                lbl = _label(base_mode, f"b{beta:g}" if multi else "")
                runs.append((lbl, base_mode, ra))
        elif base_mode == "ccsa_kl_constrained":
            multi = len(args.kl_threshold) > 1
            for thr in args.kl_threshold:
                ra = copy.copy(args)
                ra.kl_threshold = thr
                ra.beta = float(args.beta[0])   # scalar; beta not used as objective here
                lbl = _label(base_mode, f"kl{thr:g}" if multi else "")
                runs.append((lbl, base_mode, ra))
        elif base_mode == "ccsa_recon_constrained":
            multi = len(args.recon_threshold) > 1
            for thr in args.recon_threshold:
                ra = copy.copy(args)
                ra.recon_threshold = thr
                ra.beta = float(args.beta[0])   # scalar; beta not used as objective here
                lbl = _label(base_mode, f"r{thr:g}" if multi else "")
                runs.append((lbl, base_mode, ra))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"mnist_vae_seed{args.seed}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    run_params = dict(vars(args))
    run_params["ccsa_configs"] = {}
    with open(outdir / "run_params.json", "w") as f:
        json.dump(run_params, f, indent=2)

    all_results = {}
    trained_models = {}
    run_args_map = {}   # label → run_args (for beta used in that run)
    for label, base_mode, ra in runs:
        print(f"\n[INFO] === Running: {label} ===")
        model = MNIST_VAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
        if base_mode == "adam_additive":
            res = run_adam(model, train_loader, test_loader, device, ra,
                           outdir=outdir, label=label)
        elif base_mode == "ccsa_additive":
            res = run_ccsa(model, train_loader, test_loader, device, ra,
                           mode="additive", outdir=outdir, label=label)
        elif base_mode == "ccsa_kl_constrained":
            res = run_ccsa(model, train_loader, test_loader, device, ra,
                           mode="kl_constrained", outdir=outdir, label=label)
        elif base_mode == "ccsa_recon_constrained":
            res = run_ccsa(model, train_loader, test_loader, device, ra,
                           mode="recon_constrained", outdir=outdir, label=label)
        else:
            raise ValueError(f"unknown base_mode {base_mode}")
        all_results[label] = res
        trained_models[label] = model
        run_args_map[label] = ra

    # Merge per-run CCSA configs into run_params.json
    for label, base_mode, _ in runs:
        if base_mode.startswith("ccsa_"):
            cfg_path = outdir / f"ccsa_config_{label}.json"
            if cfg_path.exists():
                with open(cfg_path) as _f:
                    run_params["ccsa_configs"][label] = json.load(_f)
    with open(outdir / "run_params.json", "w") as f:
        json.dump(run_params, f, indent=2)

    # Save results
    save_dict = {}
    serializable_logs = {}
    for label, (losses, recons, kls, evals, logs) in all_results.items():
        save_dict[f"{label}_losses"] = np.array(losses, dtype=np.float32)
        save_dict[f"{label}_recons"] = np.array(recons, dtype=np.float32)
        save_dict[f"{label}_kls"]    = np.array(kls,    dtype=np.float32)
        save_dict[f"{label}_evals"]  = np.array(evals,  dtype=np.float32)
        serial = {}
        for k, v in logs.items():
            serial[k] = list(v) if isinstance(v, (list, tuple, np.ndarray)) else v
        serializable_logs[label] = serial
    np.savez_compressed(outdir / "results.npz", **save_dict)
    with open(outdir / "logs.json", "w") as f:
        json.dump(serializable_logs, f, indent=2)
    with open(outdir / "all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    for label, m in trained_models.items():
        torch.save(m.state_dict(), outdir / f"model_{label}.pt")
        save_reconstructions(m, test_loader, device, outdir, tag=label)

    make_plots(all_results, outdir, exp="mnist_vae")

    # ------------------------------------------------------------------
    # Final summary .txt — re-evaluate best-restored model for each run
    # ------------------------------------------------------------------
    col = max(30, max(len(lbl) for lbl, *_ in runs) + 2)
    sep = "-" * (col + 48)
    lines = []
    lines.append("=" * (col + 48))
    lines.append(f"  VAE experiment summary   [{timestamp}]")
    lines.append(f"  epochs={args.epochs}  betas={args.beta}"
                 f"  kl_thrs={args.kl_threshold}  recon_thrs={args.recon_threshold}")
    lines.append("=" * (col + 48))
    lines.append(f"{'run':<{col}}  {'TRAIN recon':>12}  {'TRAIN kl':>10}  {'VAL recon':>10}  {'VAL kl':>8}")
    lines.append(sep)
    for label, m in trained_models.items():
        ra = run_args_map[label]
        _, tr_recon, tr_kl = evaluate_vae(m, train_loader, device, beta=ra.beta)
        _, va_recon, va_kl = evaluate_vae(m, test_loader,  device, beta=ra.beta)
        lines.append(
            f"{label:<{col}}  {tr_recon:>12.3f}  {tr_kl:>10.4f}  {va_recon:>10.3f}  {va_kl:>8.4f}"
        )
    lines.append(sep)
    summary_txt = "\n".join(lines) + "\n"
    summary_path = outdir / "summary.txt"
    summary_path.write_text(summary_txt)
    print(summary_txt)
    print(f"[INFO] Saved results to {outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["adam_additive", "ccsa_additive",
                                      "ccsa_kl_constrained", "ccsa_recon_constrained", "all"],
                   default="all")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, dest="batch_size", default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--latent-dim", type=int, default=20)
    p.add_argument("--hidden-dim", type=int, default=400)
    p.add_argument("--beta", type=float, nargs='+', default=[1.0],
                   help="beta(s) on KL in additive modes; multiple values run a sweep "
                        "(e.g. --beta 0.1 1 10)")
    p.add_argument("--kl-threshold", type=float, nargs='+', default=[5.0],
                   dest="kl_threshold",
                   help="KL upper bound(s) for ccsa_kl_constrained; multiple values run a sweep "
                        "(e.g. --kl-threshold 3 5 10)")
    p.add_argument("--recon-threshold", type=float, nargs='+', default=[100.0],
                   dest="recon_threshold",
                   help="recon upper bound(s) for ccsa_recon_constrained; multiple values run a sweep "
                        "(e.g. --recon-threshold 80 100 120)")
    p.add_argument("--inner-eval-weight", type=float, default=0.5)
    p.add_argument("--print-every", type=int, default=50, dest="print_every",
                   help="print batch-level recon/KL every N outer steps (0=off)")
    p.add_argument("--checkpoint-every", type=int, default=50, dest="checkpoint_every",
                   help="evaluate val and checkpoint best model every N outer steps (CCSA only)")
    p.add_argument("--checkpoint-warmup", type=int, default=200, dest="checkpoint_warmup",
                   help="skip checkpointing for the first N outer steps (avoids saving on uninformative init)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--outdir", type=str, default="./runs_vae")
    args = p.parse_args()
    run(args)
