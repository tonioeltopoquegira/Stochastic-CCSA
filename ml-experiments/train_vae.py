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
def run_adam(model, train_loader, test_loader, device, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    batch_losses, batch_recons, batch_kls, all_evals = [], [], [], []
    cumulative_eval = 0.0
    logs = {"epoch": [], "train_total": [], "train_recon": [], "train_kl": [],
            "val_total": [], "val_recon": [], "val_kl": [], "time": []}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_total, ep_recon, ep_kl, bl, br, bk, be = train_epoch_vae(
            model, train_loader, optimizer, device, beta=args.beta,
            show_progress=True, desc=f"Adam ep{epoch}"
        )
        batch_losses.extend(bl); batch_recons.extend(br); batch_kls.extend(bk)
        for e in be:
            cumulative_eval += e
            all_evals.append(cumulative_eval)

        val_total, val_recon, val_kl = evaluate_vae(model, test_loader, device, beta=args.beta)
        logs["epoch"].append(epoch)
        logs["train_total"].append(ep_total); logs["train_recon"].append(ep_recon); logs["train_kl"].append(ep_kl)
        logs["val_total"].append(val_total); logs["val_recon"].append(val_recon); logs["val_kl"].append(val_kl)
        logs["time"].append(time.time() - t0)
        print(f"[adam_additive] ep {epoch}: train recon={ep_recon:.2f} kl={ep_kl:.3f} | "
              f"val recon={val_recon:.2f} kl={val_kl:.3f}")
    return batch_losses, batch_recons, batch_kls, all_evals, logs


# ---------------------------------------------------------------------------
# CCSA driver (covers all three CCSA modes)
# ---------------------------------------------------------------------------
class TrainingComplete(Exception):
    pass


def run_ccsa(model, train_loader, test_loader, device, args, mode: str):
    """
    mode in {"additive", "kl_constrained", "recon_constrained"}.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    x0, shapes, sizes = pack_params(params)
    batches_per_epoch = len(train_loader)
    total_outer = int(args.epochs * batches_per_epoch)

    train_iter = itertools.cycle(train_loader)

    # State shared by callables for this run
    state = {
        "batch": None,        # currently-locked minibatch
        "cache_x": None,      # id of x for which cache below is valid
        "recon": None,
        "kl": None,
        "recon_grad": None,
        "kl_grad": None,
        "outer_calls": 0,
        "inner_calls": 0,
        "cumulative_eval": 0.0,
    }

    batch_losses, batch_recons, batch_kls = [], [], []
    cumulative_evals_trace = []
    logs = {"epoch": [], "train_total": [], "train_recon": [], "train_kl": [],
            "val_total": [], "val_recon": [], "val_kl": [], "time": []}
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
        # Two separate backwards on the same forward graph
        g_bce = torch.autograd.grad(bce, params, retain_graph=True, allow_unused=True)
        g_kl = torch.autograd.grad(kl, params, retain_graph=False, allow_unused=True)
        state["recon"] = float(bce.item())
        state["kl"] = float(kl.item())
        state["recon_grad"] = flatten_grad(g_bce, params)
        state["kl_grad"] = flatten_grad(g_kl, params)
        state["cache_x"] = id(x)

    def _forward_only(x: np.ndarray):
        unpack_to_params(x, params, shapes, sizes)
        model.eval()
        data = state["batch"]
        with torch.no_grad():
            recon, mu, logvar = model(data)
            bce, kl = model.loss_components(recon, data, mu, logvar, reduction='mean')
        state["recon"] = float(bce.item())
        state["kl"] = float(kl.item())
        state["recon_grad"] = None
        state["kl_grad"] = None
        state["cache_x"] = id(x)

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
        tag = f"ccsa_{mode}"
        print(f"[{tag}] ep {epoch_idx}: train recon={train_recon:.2f} kl={train_kl:.3f} | "
              f"val recon={val_recon:.2f} kl={val_kl:.3f}")

    # -------- objective callable --------
    def objective(x, grad=None):
        if state["outer_calls"] >= total_outer:
            raise TrainingComplete()

        if grad is True:
            # Outer step: load a new batch and recompute everything
            _fetch_new_batch()
            _forward_with_grad(x)
            state["outer_calls"] += 1
            state["inner_calls"] = 0
            state["cumulative_eval"] += 1.0
        else:
            # Inner step on the same locked batch
            _forward_only(x)
            state["inner_calls"] += 1
            state["cumulative_eval"] += float(args.inner_eval_weight)

        if mode == "additive":
            f_val = state["recon"] + args.beta * state["kl"]
            f_grad = (state["recon_grad"] + args.beta * state["kl_grad"]) if grad is True else None
        elif mode == "kl_constrained":
            f_val = state["recon"]
            f_grad = state["recon_grad"] if grad is True else None
        elif mode == "recon_constrained":
            f_val = state["kl"]
            f_grad = state["kl_grad"] if grad is True else None
        else:
            raise ValueError(f"unknown mode {mode}")

        batch_losses.append(f_val)
        batch_recons.append(state["recon"])
        batch_kls.append(state["kl"])
        cumulative_evals_trace.append(state["cumulative_eval"])

        if grad is True:
            pe = getattr(args, 'print_every', 50)
            if pe > 0 and state["outer_calls"] % pe == 0:
                constr_str = ""
                if mode == "recon_constrained":
                    constr_str = f"  recon={state['recon']:.1f}(≤{args.recon_threshold})"
                elif mode == "kl_constrained":
                    constr_str = f"  kl={state['kl']:.3f}(≤{args.kl_threshold})"
                print(f"  step {state['outer_calls']:5d}/{total_outer} | "
                      f"recon={state['recon']:.2f}  kl={state['kl']:.4f}{constr_str}")
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
    # conservative=True for constrained modes: rejected steps stay at x_k rather than
    # triggering the feasibility solver (feasibility_minimization.py solves an (n+1)-dim
    # SLSQP that is completely infeasible at neural-network scale).
    is_constrained = g_fun is not None
    ccsa_opt = CCSAOptimizer_Core(
        params=x0,
        fun=objective,
        g=g_fun,
        dg=dg_fun,
        bounds=None,
        use_quadratic_surrogates=True,
        conservative=is_constrained,
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
        mode_list = ["adam_additive", "ccsa_additive", "ccsa_kl_constrained", "ccsa_recon_constrained"]
    else:
        mode_list = [args.mode]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"mnist_vae_seed{args.seed}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "run_params.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    all_results = {}
    trained_models = {}
    for mode_name in mode_list:
        print(f"\n[INFO] === Running mode: {mode_name} ===")
        model = MNIST_VAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
        if mode_name == "adam_additive":
            res = run_adam(model, train_loader, test_loader, device, args)
        elif mode_name == "ccsa_additive":
            res = run_ccsa(model, train_loader, test_loader, device, args, mode="additive")
        elif mode_name == "ccsa_kl_constrained":
            res = run_ccsa(model, train_loader, test_loader, device, args, mode="kl_constrained")
        elif mode_name == "ccsa_recon_constrained":
            res = run_ccsa(model, train_loader, test_loader, device, args, mode="recon_constrained")
        else:
            raise ValueError(f"unknown mode {mode_name}")
        all_results[mode_name] = res
        trained_models[mode_name] = model

    # Save results
    save_dict = {}
    serializable_logs = {}
    for mode_name, (losses, recons, kls, evals, logs) in all_results.items():
        save_dict[f"{mode_name}_losses"] = np.array(losses, dtype=np.float32)
        save_dict[f"{mode_name}_recons"] = np.array(recons, dtype=np.float32)
        save_dict[f"{mode_name}_kls"] = np.array(kls, dtype=np.float32)
        save_dict[f"{mode_name}_evals"] = np.array(evals, dtype=np.float32)
        serial = {}
        for k, v in logs.items():
            serial[k] = list(v) if isinstance(v, (list, tuple, np.ndarray)) else v
        serializable_logs[mode_name] = serial
    np.savez_compressed(outdir / "results.npz", **save_dict)
    with open(outdir / "logs.json", "w") as f:
        json.dump(serializable_logs, f, indent=2)
    with open(outdir / "all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    for mode_name, m in trained_models.items():
        torch.save(m.state_dict(), outdir / f"model_{mode_name}.pt")
        save_reconstructions(m, test_loader, device, outdir, tag=mode_name)

    make_plots(all_results, outdir, exp="mnist_vae")
    print(f"\n[INFO] Saved results to {outdir}")


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
    p.add_argument("--beta", type=float, default=1.0,
                   help="beta coefficient on KL in additive modes")
    p.add_argument("--kl-threshold", type=float, default=5.0,
                   help="upper bound on per-sample KL for ccsa_kl_constrained")
    p.add_argument("--recon-threshold", type=float, default=100.0,
                   help="upper bound on per-sample BCE for ccsa_recon_constrained")
    p.add_argument("--inner-eval-weight", type=float, default=0.5)
    p.add_argument("--print-every", type=int, default=50, dest="print_every",
                   help="print batch-level recon/KL every N outer steps (0=off)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--outdir", type=str, default="./runs_vae")
    args = p.parse_args()
    run(args)
