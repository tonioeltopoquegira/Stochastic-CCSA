import torch
import nlopt
import numpy as np
import itertools
import time
from typing import Tuple, List
from tqdm import tqdm
from utils import evaluate  

# Exception to signal training completion (vs possible convergence or maxeval in NLOPT)
class TrainingComplete(Exception):
    pass

class CCSAOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, inner_gradients=0, always_improve=0,
                 sigma_min=0.0, maxeval=5, max_inner_eval=200, verbose=False,
                 beta=1.0, inner_eval_weight=0.5):
        
        defaults = dict(lr=lr, inner_gradients=inner_gradients,
                        always_improve=always_improve,
                        sigma_min=sigma_min,
                        maxeval=maxeval,
                        max_inner_eval=max_inner_eval,
                        verbose=verbose,
                        beta=beta,
                        inner_eval_weight=inner_eval_weight)
        super().__init__(params, defaults)


    def _pack_params(self, params):
        shapes = [tuple(p.shape) for p in params]
        sizes = [p.numel() for p in params]
        x0_parts = [p.detach().cpu().numpy().ravel().astype(np.float64) for p in params]
        x0 = np.concatenate(x0_parts) if x0_parts else np.array([], dtype=np.float64)
        return x0, shapes, sizes

    def _unpack_to_params(self, x, params, shapes, sizes):
        offset = 0
        for p, shape, size in zip(params, shapes, sizes):
            chunk = x[offset:offset+size].reshape(shape)
            arr_t = torch.tensor(chunk, dtype=p.dtype, device=p.device)
            with torch.no_grad():
                p.copy_(arr_t)
            offset += size

    def optimize_training(self,
                        train_loader,
                        model: torch.nn.Module,
                        criterion,
                        device,
                        epochs: int,
                        test_loader=None,
                        version:str = 'mma') -> Tuple[List[float], List[float], dict]:
       
        group = self.param_groups[0]
        params = [p for p in group["params"] if p.requires_grad]

        x0, shapes, sizes = self._pack_params(params)
        nvars = x0.size
        batches_per_epoch = len(train_loader)
        total_outer = int(epochs * batches_per_epoch)

        # Preparing for logging
        outer_calls = 0
        inner_calls = 0
        cumulative_eval_counter = 0.0  # monotonic cumulative counter
        batch_losses = []              # record all eval losses (outer & inner)
        cumulative_weighted_evals = []

        logs = {"epoch": [], "train_eval_loss": [], "train_eval_acc": [],
                "val_eval_loss": [], "val_eval_acc": [], "time": []}
        t0 = time.time()

        train_iter = itertools.cycle(train_loader)
        current_batch = None

        # Create optimizer once
        if version == 'mma':
            opt = nlopt.opt(nlopt.LD_MMA, nvars)
        elif version == 'ccsaq':
            opt = nlopt.opt(nlopt.LD_CCSAQ, nvars)

        # Setting parameters
        opt.set_param("inner_gradients", int(group.get("inner_gradients", 0)))
        opt.set_param("always_improve", int(group.get("always_improve", 0)))
        opt.set_param("sigma_min", float(group.get("sigma_min", 1e-4)))
        opt.set_param("inner_maxeval", 5)
        opt.set_maxeval(int(group.get("maxeval", 1e6)))

        pbar = tqdm(total=total_outer, desc="CCSA outer steps", unit="step", leave=True)
        verbose = group.get("verbose", False)
        inner_weight = float(group.get("inner_eval_weight", 0.5))  # optional param

        def loss(x, grad):
            nonlocal outer_calls, inner_calls, current_batch, cumulative_eval_counter

            if outer_calls >= total_outer:
                raise TrainingComplete("Reached requested outer steps")

            self._unpack_to_params(x, params, shapes, sizes)

            if grad.size > 0:
                # OUTER EVAL: compute gradients and return value for optimizer step
                data, target = next(train_iter)
                current_batch = (data.to(device), target.to(device))

                model.train()
                loss_tensor = criterion(model(current_batch[0]), current_batch[1])

                # compute outer gradients
                g_tensors = torch.autograd.grad(loss_tensor, params, retain_graph=False,
                                                create_graph=False, allow_unused=True)
                g_vec = np.concatenate([
                    (gt.detach().cpu().numpy().ravel().astype(np.float64) if gt is not None else np.zeros(p.numel(), dtype=np.float64))
                    for gt, p in zip(g_tensors, params)
                ]) if params else np.array([], dtype=np.float64)
                grad[:] = g_vec

                # bookkeeping: mark an outer evaluation
                outer_calls += 1
                cumulative_eval_counter += 1.0
                batch_losses.append(float(loss_tensor.item()))
                cumulative_weighted_evals.append(cumulative_eval_counter)

                # epoch-level logging 
                if batches_per_epoch > 0 and outer_calls % batches_per_epoch == 0:
                    epoch_idx = outer_calls // batches_per_epoch
                    tr_loss, tr_acc = evaluate(model, train_loader, criterion, device)
                    if test_loader is not None:
                        va_loss, va_acc = evaluate(model, test_loader, criterion, device)
                    else:
                        va_loss, va_acc = None, None

                    elapsed = time.time() - t0
                    logs["epoch"].append(epoch_idx)
                    logs["train_eval_loss"].append(tr_loss)
                    logs["train_eval_acc"].append(tr_acc)
                    logs["val_eval_loss"].append(va_loss)
                    logs["val_eval_acc"].append(va_acc)
                    logs["time"].append(elapsed)

                    if verbose:
                        msg = (f"[CCSA][epoch {epoch_idx}] train_eval_loss={tr_loss:.4f} train_eval_acc={tr_acc:.4f}")
                        if va_loss is not None and va_acc is not None:
                            msg += f" | val_eval_loss={va_loss:.4f} val_eval_acc={va_acc:.4f}"
                        print(msg)

                if verbose:
                    print(f"Outer eval #{outer_calls}, Loss = {loss_tensor.item():.4f}, cumulative_eval={cumulative_eval_counter:.1f}")

                # reset inner_calls AFTER using it
                inner_calls = 0

                pbar.update(1)
                return float(loss_tensor.item())

            else:
                # INNER EVAL: compute objective value (no grad)
                if current_batch is None:
                    data, target = next(train_iter)
                    current_batch = (data.to(device), target.to(device))
                with torch.no_grad():
                    model.eval()
                    loss_tensor = criterion(model(current_batch[0]), current_batch[1])

                inner_calls += 1
                cumulative_eval_counter += inner_weight
                batch_losses.append(float(loss_tensor.item()))
                cumulative_weighted_evals.append(cumulative_eval_counter)

                if verbose:
                    print(f"Inner eval #{inner_calls}, Loss = {loss_tensor.item():.4f}, cumulative_eval={cumulative_eval_counter:.1f}")

                return float(loss_tensor.item())

        opt.set_min_objective(loss)

        try:
            xopt = opt.optimize(x0)
        except TrainingComplete:
            x_parts = [p.detach().cpu().numpy().ravel().astype(np.float64) for p in params]
            xopt = np.concatenate(x_parts) if x_parts else np.array([], dtype=np.float64)
        finally:
            try:
                pbar.close()
            except Exception:
                pass

        logs["final_cumulative_eval"] = cumulative_eval_counter

        self._unpack_to_params(xopt, params, shapes, sizes)
        return batch_losses, cumulative_weighted_evals, logs




    