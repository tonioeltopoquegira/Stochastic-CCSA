import argparse
import time
from pathlib import Path
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from class_models import MNIST_CNN, resnet32, resnet56
from utils import set_seed, get_loaders, train_epoch, evaluate
from optimizers import CCSAOptimizer

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


def run(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("[INFO] Device:", device)

    train_loader, test_loader = get_loaders(exp=args.exp, batch_size=args.batch_size,
                                            pin_memory=(device.type == "cuda"))

    model_factory = lambda: get_model_for_exp(args.exp)
    criterion = nn.CrossEntropyLoss()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"{args.exp}_seed{args.seed}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "run_params.json", "w") as f:
        json.dump(vars(args), f, indent=2)

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
                                      max_inner_eval=args.max_inner_eval,
                                      verbose=args.verbose)
            use_ccsa = True
        else:
            raise ValueError(f"Unknown optimizer {opt_name}")

        print(f"[INFO] Running optimizer: {opt_name.upper()}")

        if not use_ccsa:
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

                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"train_eval_loss={train_eval_loss:.4f} train_eval_acc={train_eval_acc:.4f} | "
                    f"val_eval_loss={val_eval_loss:.4f} val_eval_acc={val_eval_acc:.4f}"
                )
        else:

            all_batch_losses, all_evals, logs = optimizer.optimize_training(
                train_loader, model, criterion, device, args.epochs, test_loader=test_loader
            )

        all_results[opt_name] = (all_batch_losses, all_evals, logs)

    # Save the results
    save_dict = {}
    serializable_logs = {}
    for opt_name, (losses, evals, logs) in all_results.items():
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

    print(f"[INFO] Saved results to {outdir}")

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
    p.add_argument("--maxeval", type=int, default=int(1e6))
    p.add_argument("--max-inner-eval", type=int, default=200)   
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="./runs")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--plot-ylim", type=float, default=None)

    args = p.parse_args()
    run(args)
