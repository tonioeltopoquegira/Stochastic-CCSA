"""
PyTorch wrapper for the CCSA optimizer (Consecutive Convex Separable Approximation).

This module provides a torch.optim.Optimizer interface to the standalone CCSAOptimizer
from the ccsa module, enabling constrained optimization for neural network training.

Default behavior:
  - use_quadratic_surrogates=True  (quadratic convex separable approximations)
  - conservative=False             (uses feasibility minimization when applicable)
  - update_rule='multiplier'       (robust curvature parameter updates on rejection)
"""

import torch
import numpy as np
import itertools
import time
from typing import Tuple, List, Optional, Callable
from tqdm import tqdm
import sys
from pathlib import Path

# Import the standalone CCSA optimizer
sys.path.insert(0, str(Path(__file__).parent.parent))
from ccsa.optimizer import CCSAOptimizer as CCSAOptimizer_Core


# Exception to signal training completion
class TrainingComplete(Exception):
    pass


class CCSATorchOptimizer(torch.optim.Optimizer):
    """
    PyTorch wrapper for the CCSA optimizer.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (for compatibility; not directly used by CCSA)
        
        === CCSA Algorithm Configuration ===
        use_quadratic_surrogates: If True, uses quadratic convex separable approximations
            (CCSA style). If False, uses MMA-style moving asymptotes. Default: True
        conservative: If True, strictly enforces descent (conservative). If False,
            uses feasibility minimization to recover feasible iterates. Default: False
        max_inner: Maximum inner subproblem solves per outer iteration. Default: 5
        rho_init: Initial curvature parameter for objective. Default: 1.0
        sigma_min: Minimum asymptote spacing. Default: 1e-3
        update_rule: Curvature update strategy. Options:
            - 'multiplier': Multiplier method (increases rho on rejection). Default.
            - 'adam_violation': Adam-based adaptation using surrogate violations.
            - 'adam_secant': Adam-based adaptation using secant estimates.
            Default: 'multiplier'
        
        === Training Control ===
        verbose: Print detailed optimization progress. Default: False
        inner_eval_weight: Weight for inner evaluations in cumulative counter.
            Values < 1.0 discount inner evals. Default: 0.5
    """
    
    # Default CCSA configuration 
    CCSA_DEFAULTS = {
        'use_quadratic_surrogates': True,   # Quadratic surrogates (CCSA style)
        'conservative': False,               # Use feasibility minimization
        'max_inner': 1,
        'rho_init': 1.0,
        'sigma_min': 1e-6,
        'update_rule': None,  # Default to Adam-based update for better performance
        'update_rule_kwargs': None,
        'store_history': False,  # Disable history by default to save memory
    }

    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 # CCSA algorithm configuration
                 use_quadratic_surrogates: bool = True,
                 conservative: bool = False,
                 max_inner: int = 1,
                 rho_init: float = 1.0,
                 sigma_min: float = 1e-6,
                 update_rule: str = 'multiplier',
                 update_rule_kwargs: Optional[dict] = None,
                 # Training control
                 verbose: bool = False,
                 inner_eval_weight: float = 1.0):
        
        # Store CCSA-specific configs
        self.ccsa_config = {
            'use_quadratic_surrogates': use_quadratic_surrogates,
            'conservative': conservative,
            'max_inner': max_inner,
            'rho_init': rho_init,
            'sigma_min': sigma_min,
            'update_rule': update_rule,
            'update_rule_kwargs': update_rule_kwargs,
            'store_history': False,  
        }
        
        defaults = dict(
            lr=lr,
            verbose=verbose,
            inner_eval_weight=inner_eval_weight,
            **self.ccsa_config
        )
        super().__init__(params, defaults)

    def _pack_params(self, params: List[torch.Tensor]) -> Tuple[np.ndarray, List[tuple], List[int]]:
        """Pack PyTorch parameters into flat numpy array."""
        shapes = [tuple(p.shape) for p in params]
        sizes = [p.numel() for p in params]
        x0_parts = [p.detach().cpu().numpy().ravel().astype(np.float64) for p in params]
        x0 = np.concatenate(x0_parts) if x0_parts else np.array([], dtype=np.float64)
        return x0, shapes, sizes

    def _unpack_to_params(self, 
                         x: np.ndarray, 
                         params: List[torch.Tensor],
                         shapes: List[tuple],
                         sizes: List[int]) -> None:
        """Unpack flat numpy array back into PyTorch parameters."""
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
                         criterion: torch.nn.Module,
                         device: torch.device,
                         epochs: int,
                         test_loader: Optional[object] = None) -> Tuple[List[float], List[float], dict]:
        """
        Run CCSA optimization over the training dataset.
        
        Args:
            train_loader: DataLoader for training data
            model: PyTorch model to optimize
            criterion: Loss function
            device: Device to run on (cpu or cuda)
            epochs: Number of training epochs
            test_loader: Optional DataLoader for validation
            
        Returns:
            tuple: (batch_losses, cumulative_weighted_evals, logs)
                - batch_losses: Loss value at each evaluation (outer + inner)
                - cumulative_weighted_evals: Cumulative weighted evaluation count
                - logs: Dictionary with epoch-level metrics (loss, accuracy, time)
        """
        
        group = self.param_groups[0]
        params = [p for p in group["params"] if p.requires_grad]

        x0, shapes, sizes = self._pack_params(params)
        nvars = x0.size
        batches_per_epoch = len(train_loader)
        total_outer = int(epochs * batches_per_epoch)

        # Evaluation tracking
        outer_calls = 0
        inner_calls = 0
        cumulative_eval_counter = 0.0
        batch_losses = []
        cumulative_weighted_evals = []

        logs = {
            "epoch": [],
            "train_eval_loss": [],
            "train_eval_acc": [],
            "val_eval_loss": [],
            "val_eval_acc": [],
            "time": []
        }
        t0 = time.time()

        train_iter = itertools.cycle(train_loader)
        current_batch = None

        # === Create CCSA optimizer ===
        ccsa_config = group.copy()
        ccsa_config.pop('params', None)
        ccsa_config.pop('lr', None)
        ccsa_config.pop('verbose', None)
        ccsa_config.pop('inner_eval_weight', None)
        
        # Extract CCSA-specific parameters
        ccsa_kwargs = {k: ccsa_config.pop(k, self.CCSA_DEFAULTS[k]) 
                      for k in self.CCSA_DEFAULTS.keys() 
                      if k in ccsa_config or k in self.CCSA_DEFAULTS}
        
        # Build objective and gradient functions
        def objective_and_grad(x: np.ndarray, grad=None):
            """
            Objective function for CCSA optimizer.
            - With grad=True: compute gradient (outer evaluation)
            - Without grad: compute loss only (inner evaluation)
            """
            nonlocal outer_calls, inner_calls, current_batch, cumulative_eval_counter

            #print(f"x[:5] = {x[:5]}")

            if outer_calls >= total_outer:
                raise TrainingComplete("Reached requested number of epochs")

            self._unpack_to_params(x, params, shapes, sizes)

            # ===== OUTER EVALUATION: Compute gradients =====
            if grad is True:
                if current_batch is None:
                    data, target = next(train_iter)
                    current_batch = (data.to(device), target.to(device))

                model.train()
                loss_tensor = criterion(model(current_batch[0]), current_batch[1])

                # Compute gradients
                g_tensors = torch.autograd.grad(
                    loss_tensor, params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True
                )
                g_vec = np.concatenate([
                    (gt.detach().cpu().numpy().ravel().astype(np.float64)
                     if gt is not None else np.zeros(p.numel(), dtype=np.float64))
                    for gt, p in zip(g_tensors, params)
                ]) if params else np.array([], dtype=np.float64)
                
                if g_vec.size != x.size:
                    raise ValueError(f"Gradient size {g_vec.size} != x size {x.size}")
                
                outer_calls += 1
                cumulative_eval_counter += 1.0
                batch_losses.append(float(loss_tensor.item()))
                cumulative_weighted_evals.append(cumulative_eval_counter)

                if batches_per_epoch > 0 and outer_calls % batches_per_epoch == 0:
                    epoch_idx = outer_calls // batches_per_epoch
                    tr_loss, tr_acc = _evaluate(model, train_loader, criterion, device)
                    
                    if test_loader is not None:
                        va_loss, va_acc = _evaluate(model, test_loader, criterion, device)
                    else:
                        va_loss, va_acc = None, None

                    elapsed = time.time() - t0
                    logs["epoch"].append(epoch_idx)
                    logs["train_eval_loss"].append(tr_loss)
                    logs["train_eval_acc"].append(tr_acc)
                    logs["val_eval_loss"].append(va_loss)
                    logs["val_eval_acc"].append(va_acc)
                    logs["time"].append(elapsed)

                    if group.get("verbose", False):
                        msg = f"[CCSA][epoch {epoch_idx}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}"
                        if va_loss is not None and va_acc is not None:
                            msg += f" | val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
                        print(msg)

                if group.get("verbose", False):
                    print(f"  Outer eval #{outer_calls}: loss={loss_tensor.item():.4f}, "
                          f"cumulative_eval={cumulative_eval_counter:.1f}")

                inner_calls = 0
                loss_val = float(loss_tensor.item())
                
                # Cleanup after we've extracted the loss value
                del loss_tensor, g_tensors
                
                return (loss_val, g_vec)

            else:
                # ===== INNER EVALUATION: Compute loss only =====
                if current_batch is None:
                    data, target = next(train_iter)
                    current_batch = (data.to(device), target.to(device))

                with torch.no_grad():
                    model.eval()
                    loss_tensor = criterion(model(current_batch[0]), current_batch[1])

                inner_calls += 1
                inner_weight = float(group.get("inner_eval_weight", 0.5))
                cumulative_eval_counter += inner_weight
                loss_val = float(loss_tensor.item())
                batch_losses.append(loss_val)
                cumulative_weighted_evals.append(cumulative_eval_counter)
                
                # Cleanup
                del loss_tensor
                
                current_batch = None

                return loss_val

        # Create CCSA optimizer instance
        ccsa_opt = CCSAOptimizer_Core(
            params=x0,
            fun=objective_and_grad,
            g=None,  # No explicit constraints for unconstrained NN training
            bounds=None,
            **ccsa_kwargs,
        )

        # Run optimization
        pbar = tqdm(total=total_outer, desc="CCSA outer steps", unit="step", leave=True)
        
        try:
            xopt = x0.copy()
            for outer_step in range(total_outer):
                ccsa_opt.step()
                pbar.update(1)
                
                # Clear memory every 10 iterations 
                if (outer_step + 1) % 10 == 0:
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                
                if outer_calls >= total_outer:
                    break

        except TrainingComplete:
            pass
        finally:
            pbar.close()
            # Final cleanup
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        logs["final_cumulative_eval"] = cumulative_eval_counter

        # Unpack optimized parameters
        self._unpack_to_params(ccsa_opt.x_k, params, shapes, sizes)

        return batch_losses, cumulative_weighted_evals, logs


# Utility function for evaluation (same as in utils.py but included here for completeness)
def _evaluate(model: torch.nn.Module,
             data_loader,
             criterion: torch.nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            
            # Accuracy
            pred = output.argmax(dim=1)
            total_correct += (pred == target).sum().item()
            total_samples += data.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, accuracy
