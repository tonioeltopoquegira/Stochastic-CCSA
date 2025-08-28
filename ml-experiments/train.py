import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from models import MNIST_CNN, resnet32, resnet56, TinyViT
from utils import set_seed, get_loaders, train_epoch, evaluate
from optimizers import CCSAOptimizer


def get_model_for_exp(exp):
    if exp == "mnist_cnn":
        return MNIST_CNN()
    elif exp == "cifar10_resnet32":
        return resnet32(num_classes=10)
    elif exp == "cifar100_resnet56":
        return resnet56(num_classes=100)
    else:
        raise ValueError(f"Unknown experiment: {exp}")


def run(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("[INFO] Device:", device)

    # Training and Testing data loaders
    train_loader, test_loader = get_loaders(exp=args.exp, batch_size=args.batch_size,
                                            pin_memory=(device.type == "cuda"))
    
    model_factory = lambda: get_model_for_exp(args.exp)

    # Loss function 
    criterion = nn.CrossEntropyLoss()

    outdir = Path(args.outdir) / f"{args.exp}_seed{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Determine which optimizers to run
    optim_list = [args.opt] if args.opt else ["adamw", "ccsa"]
    all_results = {}

    for opt_name in optim_list:
        model = model_factory().to(device)

        if opt_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            use_ccsa = False
        elif opt_name == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            use_ccsa = False
        elif opt_name == "ccsa":
            optimizer = CCSAOptimizer(model.parameters(),
                                    lr=args.lr,
                                    inner_gradients=args.inner_gradients,
                                    always_improve=args.always_improve,
                                    sigma_min=args.sigma_min,
                                    maxeval=args.maxeval,
                                    max_inner_eval=getattr(args, "max_inner_eval", 1e6),
                                    verbose=args.verbose)
            use_ccsa = True
        else:
            raise ValueError(f"Unknown optimizer {opt_name}")


        print(f"[INFO] Running optimizer: {opt_name.upper()}")

        # training
        if not use_ccsa:
            cumulative_eval = 0.0
            all_batch_losses, all_evals = [], []
            logs = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "time": []}
            t0 = time.time()
            for epoch in range(1, args.epochs + 1):

                # One epoch
                tr_loss, tr_acc, batch_losses, batch_evals = train_epoch(
                    model, train_loader, optimizer, criterion, device,
                    show_progress=True, desc=f"Epoch {epoch}"
                )
                all_batch_losses.extend(batch_losses)

                for e in batch_evals:
                    cumulative_eval += e
                    all_evals.append(cumulative_eval)
                
                # Test 
                val_loss, val_acc = evaluate(model, test_loader, criterion, device)

                elapsed = time.time() - t0
                logs["epoch"].append(epoch)
                logs["train_loss"].append(tr_loss)
                logs["train_acc"].append(tr_acc)
                logs["val_loss"].append(val_loss)
                logs["val_acc"].append(val_acc)
                logs["time"].append(elapsed)
                print(f"Epoch {epoch}/{args.epochs} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        else:
            all_batch_losses, all_evals = optimizer.optimize_training(
                train_loader, model, criterion, device, args.epochs, test_loader=test_loader
            )
            logs = {"epoch": list(range(1, args.epochs+1)), "train_loss": all_batch_losses,
                    "train_acc": [None]*len(all_batch_losses),
                    "val_loss": [None]*len(all_batch_losses),
                    "val_acc": [None]*len(all_batch_losses),
                    "time": [None]*len(all_batch_losses)}

        all_results[opt_name] = (all_batch_losses, all_evals, logs)

    # Combined plotting
   
    plt.figure(figsize=(8, 4))
    for opt_name, (losses, evals, _) in all_results.items():
        x = np.array(evals, dtype=np.float32)
        y = np.array(losses, dtype=np.float32)
        if args.plot_eval_limit:
            mask = x <= args.plot_eval_limit
            x, y = x[mask], y[mask]
        plt.plot(x, y, label=opt_name.upper())
    plt.xlabel("Cumulative weighted evals")
    plt.ylabel("Batch loss")
    plt.title(f"Batch loss vs evals ({args.exp})")
    plt.legend()
    if args.plot_ylim:
        plt.ylim(top=args.plot_ylim)   
    plt.tight_layout()
    plt.savefig(outdir / "loss_vs_evals_combined.png", dpi=150)
    plt.close()

    print(f"[INFO] Finished. Results in {outdir}")



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", choices=["mnist_cnn", "cifar10_resnet32", "cifar100_resnet56"], required=True)
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "mnist"], default="cifar10")
    p.add_argument("--opt", choices=["adam", "adamw", "ccsa"])
    p.add_argument("--plot-eval-limit", type=float, default=None)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, dest="batch_size", default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--inner-gradients", type=int, default=0)
    p.add_argument("--always-improve", type=int, default=0)
    p.add_argument("--sigma-min", type=float, default=0.0)
    p.add_argument("--maxeval", type=int, default=5)
    p.add_argument("--max-inner-eval", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="./runs")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--plot-ylim", type=float, default=None)

    args = p.parse_args()
    run(args)
