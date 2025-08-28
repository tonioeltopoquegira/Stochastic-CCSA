import torch
import nlopt
import numpy as np
import itertools
from typing import Optional, Tuple, List
from tqdm import tqdm

class TrainingComplete(Exception):
    pass

class CCSAOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, inner_gradients=0, always_improve=0,
                 sigma_min=0.0, maxeval=5, max_inner_eval=None, verbose=False):
        defaults = dict(lr=lr, inner_gradients=inner_gradients,
                        always_improve=always_improve,
                        sigma_min=sigma_min,
                        maxeval=maxeval,
                        max_inner_eval=max_inner_eval,
                        verbose=verbose)
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
                          test_loader=None) -> Tuple[List[float], List[float], List[dict]]:

        group = self.param_groups[0]

        params = [p for p in group["params"] if p.requires_grad]

        x0, shapes, sizes = self._pack_params(params)
        nvars = x0.size

        batches_per_epoch = len(train_loader)

        total_outer = int(epochs * batches_per_epoch)

        outer_calls = 0
        inner_calls = 0
        outer_batch_losses = []
        cumulative_weighted_evals = []

        train_iter = itertools.cycle(train_loader)
        current_batch = None

        opt = nlopt.opt(nlopt.LD_CCSAQ, nvars)
        try: opt.set_param("inner_gradients", int(group.get("inner_gradients", 0)))
        except Exception: pass
        try: opt.set_param("always_improve", int(group.get("always_improve", 0)))
        except Exception: pass
        try: opt.set_param("sigma_min", float(group.get("sigma_min", 0.0)))
        except Exception: pass
        try: opt.set_maxeval(group.get("maxeval", 1e6), 1e6)
        except Exception: pass

        try: opt.set_inner_maxeval(group.get("max_inner_eval", 1e6), 1e6)
        except Exception: pass


        pbar = tqdm(total=total_outer, desc="CCSA outer steps", unit="step", leave=True)

        verbose = group.get("verbose", False)  # fetch verbose flag from optimizer

        def loss(x, grad):

            nonlocal outer_calls, inner_calls, current_batch

            if outer_calls >= total_outer:
                raise TrainingComplete("Reached requested outer steps")

            self._unpack_to_params(x, params, shapes, sizes)

            if grad.size > 0:

                # outer step
                data, target = next(train_iter)
                current_batch = (data.to(device), target.to(device))

                model.train()
                loss_tensor = criterion(model(current_batch[0]), current_batch[1])

                g_tensors = torch.autograd.grad(loss_tensor, params, retain_graph=False,
                                                create_graph=False, allow_unused=True)
                g_vec = np.concatenate([
                    (gt.detach().cpu().numpy().ravel().astype(np.float64) if gt is not None else np.zeros(p.numel(), dtype=np.float64))
                    for gt, p in zip(g_tensors, params)
                ]) if params else np.array([], dtype=np.float64)


                grad[:] = g_vec

                outer_calls += 1

                # log the loss on outer step
                outer_batch_losses.append(float(loss_tensor.item()))

                # log the cumulative weighted evaluations as outer_calls + 0.5 * inner_calls
                cumulative_weighted_evals.append(outer_calls + 0.5 * inner_calls)

                if verbose:
                    print(f"Outer eval #{outer_calls}, Inner eval #{inner_calls}, Loss = {loss_tensor.item():.4f}")

                pbar.update(1)
                return float(loss_tensor.item())
            else:
                # inner step
                if current_batch is None:
                    data, target = next(train_iter)
                    current_batch = (data.to(device), target.to(device))
                with torch.no_grad():
                    model.eval()
                    loss_tensor = criterion(model(current_batch[0]), current_batch[1])
                inner_calls += 1

                if verbose:
                    print(f"Inner eval #{inner_calls}, Loss = {loss_tensor.item():.4f}")

                return float(loss_tensor.item())


        opt.set_min_objective(loss)

        try:
            xopt = opt.optimize(x0)
        except TrainingComplete:
            x_parts = [p.detach().cpu().numpy().ravel().astype(np.float64) for p in params]
            xopt = np.concatenate(x_parts) if x_parts else np.array([], dtype=np.float64)
        finally:
            try: pbar.close()
            except Exception: pass

        self._unpack_to_params(xopt, params, shapes, sizes)

        return outer_batch_losses, cumulative_weighted_evals
