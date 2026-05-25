"""
Min-Max Training with CCSA Optimizer - Warm Start Version.

This script reuses the existing train_ml.py infrastructure, just adding adversarial
loss functions and per-label error tracking.

This version implements warm-start training:
1. First trains AdamW for the full number of epochs
2. Saves the trained AdamW model
3. Uses the AdamW-trained weights as initialization for CCSA-EPI training
4. Runs CCSA-EPI training for the full number of epochs starting from these warm-started weights
"""

import argparse
import time
from pathlib import Path
import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from class_models import MNIST_CNN, resnet32, resnet56
from utils import set_seed, get_loaders, train_epoch, evaluate
from optim_torch import CCSATorchOptimizer

from datetime import datetime


def get_model_for_exp(exp):
    if exp == "mnist_cnn":
        return MNIST_CNN()
    elif exp == "cifar10_resnet32":
        return resnet32(num_classes=10)
    elif exp == "cifar100_resnet56":
        return resnet56(num_classes=100)
    else:
        raise ValueError(f"Unknown experiment: {exp}")


def get_num_classes(exp):
    if exp == "mnist_cnn":
        return 10
    elif exp == "cifar10_resnet32":
        return 10
    elif exp == "cifar100_resnet56":
        return 100
    else:
        raise ValueError(f"Unknown experiment: {exp}")


def compute_per_label_avg_errors(model, test_loader, device, num_classes):
    """
    Compute average classification error per label on test set.
    
    Returns:
        dict: label -> average error (loss) for samples with that label
    """
    model.eval()
    per_label_errors = defaultdict(list)
    ce_unreduced = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            losses = ce_unreduced(outputs, target)
            
            for label in range(num_classes):
                mask = target == label
                if mask.any():
                    per_label_errors[label].extend(losses[mask].cpu().numpy().tolist())
    
    # Return average loss per label
    return {label: float(np.mean(errs)) if errs else 0.0 
            for label, errs in per_label_errors.items()}


def run(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("[INFO] Device:", device)

    # Use 0 workers for CPU (multiprocessing overhead + semaphore leaks)
    num_workers = 0 if device.type == "cpu" else args.num_workers
    print(f"[INFO] Using {num_workers} workers for data loading")
    
    train_loader, test_loader = get_loaders(exp=args.exp, batch_size=args.batch_size,
                                            num_workers=num_workers,
                                            pin_memory=(device.type == "cuda"))

    model_factory = lambda: get_model_for_exp(args.exp)
    criterion = nn.CrossEntropyLoss()
    num_classes = get_num_classes(args.exp)
    
    # Debug mode for constraint analysis
    debug_constraints = getattr(args, 'debug_constraints', False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"{args.exp}_adversarial_warmstart_seed{args.seed}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "run_params.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    optim_list = [args.opt] if args.opt else ["adamw", "ccsa_epi"]
    all_results = {}

    trained_models = {}
    adamw_model_state = None  # Store AdamW model state for warm-start

    # If warm-start weights are provided, load them immediately
    if args.warmstart_weights is not None:
        print(f"[INFO] Loading pre-trained AdamW weights from: {args.warmstart_weights}")
        adamw_model_state = torch.load(args.warmstart_weights, map_location=device)
        # Skip AdamW training if weights are provided
        if "adamw" in optim_list:
            optim_list.remove("adamw")
            print(f"[INFO] Skipping AdamW training (using provided weights)")

    for opt_name in optim_list:
        model = model_factory().to(device)

        # If this is ccsa_epi and we have a warm-start model, load it
        if opt_name == "ccsa_epi" and adamw_model_state is not None:
            print(f"[INFO] Loading warm-start weights from AdamW")
            model.load_state_dict(adamw_model_state)

        if opt_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            use_ccsa = False
        elif opt_name == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            use_ccsa = False
        elif opt_name == "ccsa_epi":
            optimizer = CCSATorchOptimizer(
                model.parameters(),
                lr=args.lr,
                use_quadratic_surrogates=True,
                conservative=False,
                max_inner=1,
                rho_init=1.0,
                sigma_min=1e-2,
                update_rule='multiplier',  
                verbose=args.verbose,
                inner_eval_weight=0.5,
                loss_fn_type='adversarial_minmax',
                update_rule_kwargs={'lr': 5e-1, 'beta1': 0.05, 'beta2': 0.2, 'eps': 1e-8,
                'min_curv': 1e-4, 'max_curv': 10000.0}
            )
            use_ccsa = True

        elif opt_name == "ccsa_standard":
            optimizer = CCSATorchOptimizer(
                model.parameters(),
                lr=args.lr,
                use_quadratic_surrogates=True,
                conservative=False,
                max_inner=1,
                rho_init=1.0,
                sigma_min=1e-2,
                update_rule='multiplier',
                verbose=args.verbose,
                inner_eval_weight=0.5,
            )
            all_batch_losses, all_evals, logs = optimizer.optimize_training(
                train_loader, model, criterion, device, args.epochs, test_loader=test_loader
            )
            # Compute per-label errors on trained model
            per_label_error_history = {}
            for epoch in logs.get("epoch", []):
                per_label_error_history[epoch] = compute_per_label_avg_errors(
                    model, train_loader, device, num_classes
                )
            trained_models[opt_name] = model                                         
            all_results[opt_name] = (all_batch_losses, all_evals, logs, per_label_error_history) 
            continue                              

        else:
            raise ValueError(f"Unknown optimizer {opt_name}")

        print(f"[INFO] Running optimizer: {opt_name.upper()}")

        if not use_ccsa:
            # ===== Standard AdamW training =====
            cumulative_eval = 0.0
            all_batch_losses, all_evals = [], []
            logs = {
                "epoch": [],
                "train_eval_loss": [],
                "train_eval_acc": [],
                "val_eval_loss": [],
                "val_eval_acc": [],
                "time": [],
            }
            per_label_error_history = {}
            t0 = time.time()
            
            for epoch in range(1, args.epochs + 1):
                tr_loss, tr_acc, batch_losses, batch_evals = train_epoch(
                    model, train_loader, optimizer, criterion, device,
                    show_progress=True, desc=f"Epoch {epoch}"
                )
                all_batch_losses.extend(batch_losses)
                for e in batch_evals:
                    cumulative_eval += e
                    all_evals.append(cumulative_eval)

                train_eval_loss, train_eval_acc = evaluate(model, train_loader, criterion, device)
                val_eval_loss, val_eval_acc = evaluate(model, test_loader, criterion, device)

                logs["epoch"].append(epoch)
                logs["train_eval_loss"].append(train_eval_loss)
                logs["train_eval_acc"].append(train_eval_acc)
                logs["val_eval_loss"].append(val_eval_loss)
                logs["val_eval_acc"].append(val_eval_acc)
                logs["time"].append(time.time() - t0)

                # Compute per-label errors
                #per_label_errors = compute_per_label_avg_errors(model, test_loader, device, num_classes)
               
                per_label_errors = compute_per_label_avg_errors(model, train_loader, device, num_classes)
                per_label_error_history[epoch] = per_label_errors

                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"train_eval_loss={train_eval_loss:.4f} train_eval_acc={train_eval_acc:.4f} | "
                    f"val_eval_loss={val_eval_loss:.4f} val_eval_acc={val_eval_acc:.4f}"
                )

            # If this is AdamW, save the model state for warm-start
            if opt_name == "adamw":
                adamw_model_state = model.state_dict().copy()
                print(f"[INFO] Saved AdamW model state for warm-start")

        else:
            # ===== CCSA training with constrained adversarial formulation =====
            all_batch_losses, all_evals, logs = optimizer.optimize_training_constrained_adversarial(
                train_loader, model, criterion, device, args.epochs, num_classes, test_loader=test_loader
            )
            
            # Compute per-label errors at each epoch
            per_label_error_history = {}
            if "epoch" in logs and len(logs["epoch"]) > 0:
                for epoch in logs["epoch"]:
                    per_label_errors = compute_per_label_avg_errors(model, train_loader, device, num_classes)
                    per_label_error_history[epoch] = per_label_errors

        all_results[opt_name] = (all_batch_losses, all_evals, logs, per_label_error_history)

        trained_models[opt_name] = model

    # Save the results
    save_dict = {}
    serializable_logs = {}
    for opt_name, (losses, evals, logs, per_label_hist) in all_results.items():
        save_dict[f"{opt_name}_losses"] = np.array(losses, dtype=np.float32)
        save_dict[f"{opt_name}_evals"] = np.array(evals, dtype=np.float32)

        serial_logs = {}
        for k, v in logs.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                serial_logs[k] = np.array(v).tolist()
            else:
                serial_logs[k] = v
        serializable_logs[opt_name] = serial_logs

    np.savez_compressed(outdir / "results.npz", **save_dict)

    with open(outdir / "logs.json", "w") as f:
        json.dump(serializable_logs, f, indent=2)

    with open(outdir / "all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    # Save model weights for each optimizer
    for opt_name, m in trained_models.items():
        torch.save(m.state_dict(), outdir / f"model_{opt_name}.pt")
    print(f"[INFO] Saved model weights: {[f'model_{k}.pt' for k in trained_models]}")

    print(f"[INFO] Saved results to {outdir}")

    # ===== PLOTTING =====
    
   
    plt.figure(figsize=(8, 6))
    for opt_name, (losses, evals, _, _) in all_results.items():
        x = np.array(evals, dtype=np.float32)
        y = np.array(losses, dtype=np.float32)
        valid_idx = (x > 0) & (y > 0)
        if valid_idx.any():
            plt.plot(x[valid_idx], y[valid_idx], label=opt_name.upper(), alpha=0.7)
    plt.xlabel("Cumulative weighted evals")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Batch loss")
    plt.title(f"Batch loss vs evals ({args.exp})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "plot_01_batch_loss_vs_evals.png", dpi=150)
    plt.close()
    
    # Plot 2: Epoch-level test loss
    plt.figure(figsize=(8, 6))
    for opt_name, (_, _, logs, _) in all_results.items():
        if "val_eval_loss" in logs and len(logs["val_eval_loss"]) > 0:
            epochs = np.array(logs["epoch"], dtype=np.float32) if "epoch" in logs else np.arange(1, len(logs["val_eval_loss"]) + 1)
            val_loss = np.array(logs["val_eval_loss"], dtype=np.float32)
            plt.plot(epochs, val_loss, marker='o', label=opt_name.upper(), linewidth=2, markersize=6)
    plt.xlabel("Epoch")
    plt.ylabel("Test loss")
    plt.yscale("log")
    plt.title(f"Test loss per epoch ({args.exp})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "plot_02_epoch_test_loss.png", dpi=150)
    plt.close()
    
    # Plot 3: Test accuracy per epoch
    plt.figure(figsize=(8, 6))
    for opt_name, (_, _, logs, _) in all_results.items():
        if "val_eval_acc" in logs and len(logs["val_eval_acc"]) > 0:
            epochs = np.array(logs["epoch"], dtype=np.float32) if "epoch" in logs else np.arange(1, len(logs["val_eval_acc"]) + 1)
            val_acc = np.array(logs["val_eval_acc"], dtype=np.float32)
            plt.plot(epochs, val_acc, marker='o', label=opt_name.upper(), linewidth=2, markersize=6)
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title(f"Test accuracy per epoch ({args.exp})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(outdir / "plot_03_test_accuracy.png", dpi=150)
    plt.close()
    
    # Plot 4: Per-label error distribution (all optimizers together with different colors)
    # Get unique epochs across all optimizers
    all_epochs_set = set()
    for opt_name, (_, _, logs, _) in all_results.items():
        if "epoch" in logs:
            all_epochs_set.update(logs["epoch"])
    
    if all_epochs_set:
        epochs_list = sorted(list(all_epochs_set))
        num_epochs = len(epochs_list)
        
        # Color map for optimizers
        colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))
        opt_colors = {opt_name: colors[i] for i, opt_name in enumerate(all_results.keys())}
        
        fig, axes = plt.subplots(1, num_epochs, figsize=(6 * num_epochs, 5))
        if num_epochs == 1:
            axes = [axes]
        
        for epoch_idx, epoch in enumerate(epochs_list):
            ax = axes[epoch_idx]
            
            # Collect errors from all optimizers for this epoch
            x_offset = 0
            bar_width = 0.8 / len(all_results)  # Divide bar width by number of optimizers
            
            for opt_idx, (opt_name, (_, _, logs, per_label_hist)) in enumerate(all_results.items()):
                if epoch in per_label_hist:
                    label_errors = per_label_hist[epoch]
                    labels_list = sorted(label_errors.keys())
                    errors = [label_errors[l] for l in labels_list]
                else:
                    labels_list = list(range(num_classes))
                    errors = [0.0] * num_classes
                
                # Plot bars for this optimizer, offset by optimizer index
                x_pos = np.arange(len(labels_list)) + opt_idx * bar_width
                ax.bar(x_pos, errors, bar_width, label=opt_name.upper(), 
                       color=opt_colors[opt_name], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_yscale('log')
            ax.set_xlabel("Label", fontsize=11)
            ax.set_ylabel("Average Error (log scale)", fontsize=11)
            ax.set_title(f"Epoch {epoch}", fontsize=12, fontweight='bold')
            ax.set_xticks(np.arange(num_classes) + bar_width * (len(all_results) - 1) / 2)
            ax.set_xticklabels(range(num_classes))
            ax.grid(True, alpha=0.3, axis='y', which='both')
            if epoch_idx == 0:
                ax.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(outdir / "plot_04_per_label_error_dist_all_optimizers.png", dpi=150)
        plt.close()

    print("[INFO] Plots saved:")
    print("       - plot_01_batch_loss_vs_evals.png")
    print("       - plot_02_epoch_test_loss.png")
    print("       - plot_03_test_accuracy.png")
    print("       - plot_04_per_label_error_dist_*.png (one per optimizer)")
    print("[INFO] Warm-start training complete:")
    print("       1. AdamW trained for full epochs")
    print("       2. CCSA-EPI initialized with AdamW weights")
    print("       3. CCSA-EPI trained for full epochs from warm-start")




if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Adversarial Min-Max Training with CCSA (Warm-Start Version)")
    p.add_argument("--exp", choices=["mnist_cnn", "cifar10_resnet32", "cifar100_resnet56"], required=True)
    p.add_argument("--opt", choices=["ccsa_epi", "ccsa_standard", "adamw"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, dest="batch_size", default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="./runs_adversarial")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--warmstart-weights", type=str, default=None,
                   help="Path to pre-trained AdamW model weights (.pt file). If provided, skips AdamW training.")

    args = p.parse_args()
    run(args)
