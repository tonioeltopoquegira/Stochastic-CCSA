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
from utils import evaluate


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
                 inner_eval_weight: float = 1.0,
                 loss_fn_type: str = 'standard'): 
        
        # Store CCSA-specific configs
        self.ccsa_config = {
            'use_quadratic_surrogates': use_quadratic_surrogates,
            'conservative': False,
            'max_inner': max_inner,
            'rho_init': rho_init,
            'sigma_min': sigma_min,
            'update_rule': update_rule,
            'update_rule_kwargs': update_rule_kwargs,
            'store_history': False,  
        }
        self.loss_fn_type = loss_fn_type
        
        defaults = dict(
            lr=lr,
            verbose=verbose,
            inner_eval_weight=inner_eval_weight,
            loss_fn_type=loss_fn_type,
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

    def _compute_per_label_max_loss(self,
                                   model: torch.nn.Module,
                                   batch: Tuple[torch.Tensor, torch.Tensor],
                                   criterion: torch.nn.Module,
                                   device: torch.device,
                                   num_classes: int) -> np.ndarray:
        """Compute max loss per label on a batch (for adversarial min-max constraints)."""
        data, target = batch
        model.eval()
        per_label_max_loss = np.full(num_classes, -np.inf, dtype=np.float64)
        
        with torch.no_grad():
            outputs = model(data)
            ce_unreduced = torch.nn.CrossEntropyLoss(reduction='none')
            losses = ce_unreduced(outputs, target)
            
            for label in range(num_classes):
                mask = target == label
                if mask.any():
                    max_loss = losses[mask].max().item()
                    per_label_max_loss[label] = float(max_loss)
        
        return per_label_max_loss

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

    def optimize_training_constrained_adversarial(
        self,
        train_loader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        device: torch.device,
        epochs: int,
        num_classes: int,
        test_loader=None,
        print_every: int = 10,       # print t + all constraints every N outer steps
    ) -> Tuple[List[float], List[float], dict]:
        """
        Epigraph minimax training via CCSA.
    
        Problem:
            minimize_{theta, t}   t
            subject to   g_k(theta) - t <= 0   for k = 0 ... num_classes-1
    
        where  g_k(theta) = E[ CE_loss(f_theta(x), k) | y = k ]
                        = mean loss over all samples of class k in the current batch.
    
        Each constraint has its OWN gradient row in the Jacobian:
            d/d_theta  g_k(theta)   (via autograd on mean class loss)
            d/dt       g_k - t  =  -1
    
        The network weights theta are updated ONLY through the constraint Jacobian,
        weighted by the dual variables lambda_k that CCSA maintains internally.
        This is the correct epigraph formulation.
    
        Args:
            print_every: Print t, per-class mean losses, constraint values, and
                        violation count every this many outer steps. Set to 0
                        to suppress step-level printing entirely.
        """
    
        from ccsa.optimizer import CCSAOptimizer as CCSAOptimizer_Core
    
        group = self.param_groups[0]
        params = [p for p in group["params"] if p.requires_grad]
    
        # ------------------------------------------------------------------ #
        #  Pack initial theta into flat numpy vector                           #
        # ------------------------------------------------------------------ #
        def pack(params):
            shapes = [tuple(p.shape) for p in params]
            sizes  = [p.numel()      for p in params]
            x0     = np.concatenate([p.detach().cpu().numpy().ravel().astype(np.float64)
                                    for p in params])
            return x0, shapes, sizes
    
        def unpack(x, params, shapes, sizes):
            offset = 0
            for p, sh, sz in zip(params, shapes, sizes):
                chunk = x[offset:offset + sz].reshape(sh)
                with torch.no_grad():
                    p.copy_(torch.tensor(chunk, dtype=p.dtype, device=p.device))
                offset += sz
    
        x0, shapes, sizes = pack(params)
        n_theta = x0.size
    
        batches_per_epoch = len(train_loader)
        total_outer       = int(epochs * batches_per_epoch)
    
        # ------------------------------------------------------------------ #
        #  Tracking                                                            #
        # ------------------------------------------------------------------ #
        outer_calls            = 0
        cumulative_eval        = 0.0
        batch_losses: List[float] = []
        cumulative_evals: List[float] = []
        logs = {k: [] for k in ["epoch", "train_eval_loss", "train_eval_acc",
                                "val_eval_loss", "val_eval_acc", "time"]}
        t0 = time.time()
    
        train_iter    = itertools.cycle(train_loader)
        current_batch = None          # set once per outer call, shared by obj + constraints
    
        # ------------------------------------------------------------------ #
        #  Diagnostic printer                                                  #
        # ------------------------------------------------------------------ #
        def _print_step_diagnostics(step: int, t_val: float):
            """
            Print a compact table of:
            - current step and t (epigraph variable)
            - per-class mean loss  g_k(theta)
            - constraint value     g_k(theta) - t
            - whether each constraint is violated (> 0)
            """
            if current_batch is None:
                return
    
            data, target = current_batch
            model.eval()
            ce = torch.nn.CrossEntropyLoss(reduction="none")
            with torch.no_grad():
                losses = ce(model(data), target)
    
            mean_losses = {}
            for k in range(num_classes):
                mask = target == k
                if mask.any():
                    mean_losses[k] = losses[mask].mean().item()
    
            violations = {k: v - t_val for k, v in mean_losses.items() if v - t_val > 0}
            max_viol   = max(violations.values()) if violations else 0.0
    
            lines = [
                f"\n{'─'*62}",
                f"  step {step:>5d} | t = {t_val:.6f} | "
                f"violated = {len(violations)}/{len(mean_losses)}  "
                f"max_viol = {max_viol:+.6f}",
                f"  {'cls':>4}  {'mean_loss':>10}  {'gk - t':>10}  {'status'}",
                f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*8}",
            ]
            for k in sorted(mean_losses.keys()):
                gk  = mean_losses[k]
                val = gk - t_val
                flag = "VIOLATED" if val > 0 else "ok"
                lines.append(f"  {k:>4}  {gk:>10.6f}  {val:>+10.6f}  {flag}")
            lines.append(f"{'─'*62}")
            print("\n".join(lines))
    
        # ------------------------------------------------------------------ #
        #  Estimate a good t_init: max mean class loss on first batch          #
        # ------------------------------------------------------------------ #
        def estimate_t_init():
            data, target = next(iter(train_loader))
            data, target = data.to(device), target.to(device)
            model.eval()
            ce = torch.nn.CrossEntropyLoss(reduction="none")
            with torch.no_grad():
                losses = ce(model(data), target)
            means = []
            for k in range(num_classes):
                mask = target == k
                if mask.any():
                    means.append(losses[mask].mean().item())
            return float(max(means)) * 1.2 if means else 5.0   # 20% headroom
    
        t_init = 3.365
        print(f"[EPIGRAPH] t_init = {t_init:.4f}")
    
        # Augmented variable: [theta (n_theta,), t (1,)]
        x0_aug = np.concatenate([x0, np.array([t_init], dtype=np.float64)])
        n_aug  = x0_aug.size   # n_theta + 1
    
        # ------------------------------------------------------------------ #
        #  CCSA kwargs                                                         #
        # ------------------------------------------------------------------ #
        ccsa_defaults = {
            "use_quadratic_surrogates": True,
            "conservative": False,
            "max_inner": 1,
            "rho_init": 1.0,
            "sigma_min": 1e-6,
            "update_rule": "multiplier",
            "update_rule_kwargs": None,
            "store_history": False,
        }
        ccsa_kwargs = {}
        for k, default in ccsa_defaults.items():
            val = group.get(k, default)
            ccsa_kwargs[k] = val if val is not None else default
        # Propagate user-supplied update_rule_kwargs
        if group.get("update_rule_kwargs") is not None:
            ccsa_kwargs["update_rule_kwargs"] = group["update_rule_kwargs"]
    
        # ------------------------------------------------------------------ #
        #  Objective:  f(theta, t) = t                                         #
        #  grad_theta = 0,  grad_t = 1                                         #
        # ------------------------------------------------------------------ #
        def objective_and_grad(x_aug: np.ndarray, grad=None):
            nonlocal outer_calls, current_batch, cumulative_eval
    
            if outer_calls >= total_outer:
                raise TrainingComplete
    
            t_val = float(x_aug[-1])
    
            if grad is True:
                # Draw a fresh batch for this outer step
                data, target = next(train_iter)
                current_batch = (data.to(device), target.to(device))
    
                # Unpack theta into model
                unpack(x_aug[:n_theta], params, shapes, sizes)
    
                # Gradient: only w.r.t. t  (theta gradient = 0 in epigraph form)
                g_vec        = np.zeros(n_aug, dtype=np.float64)
                g_vec[-1]    = 1.0          # d(t)/d(t) = 1
    
                outer_calls     += 1
                cumulative_eval += 1.0
    
                # For plotting: report standard mean CE loss (not t)
                model.eval()
                with torch.no_grad():
                    report_loss = float(criterion(model(current_batch[0]),
                                                current_batch[1]).item())
                batch_losses.append(report_loss)
                cumulative_evals.append(cumulative_eval)
    
                # Step-level diagnostic: t + all constraints
                if print_every > 0 and outer_calls % print_every == 0:
                    _print_step_diagnostics(step=outer_calls, t_val=t_val)
    
                # Epoch-level evaluation
                if outer_calls % batches_per_epoch == 0:
                    epoch_idx = outer_calls // batches_per_epoch
                    unpack(x_aug[:n_theta], params, shapes, sizes)
                    tr_loss, tr_acc = evaluate(model, train_loader, criterion, device)
                    va_loss, va_acc = (evaluate(model, test_loader, criterion, device)
                                    if test_loader else (None, None))
    
                    logs["epoch"].append(epoch_idx)
                    logs["train_eval_loss"].append(tr_loss)
                    logs["train_eval_acc"].append(tr_acc)
                    logs["val_eval_loss"].append(va_loss)
                    logs["val_eval_acc"].append(va_acc)
                    logs["time"].append(time.time() - t0)
    
                    print(
                        f"\n[CCSA-EPI] epoch {epoch_idx}/{epochs} | "
                        f"t={t_val:.5f} | "
                        f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
                        f"va_loss={va_loss:.4f} va_acc={va_acc:.4f}"
                    )
                    # Full constraint table at every epoch regardless of print_every
                    _print_step_diagnostics(step=outer_calls, t_val=t_val)
    
                return (t_val, g_vec)
    
            else:
                # Inner eval: objective value only
                return t_val
    
        # ------------------------------------------------------------------ #
        #  Constraints:  g_k(theta, t) = mean_loss_k(theta) - t  <= 0         #
        #                                                                      #
        #  Returns shape (num_classes,)                                        #
        #  Classes absent from the current batch get value 0 (always sat.)    #
        # ------------------------------------------------------------------ #
        def constraint_func(x_aug: np.ndarray) -> np.ndarray:
            if current_batch is None:
                return np.zeros(num_classes, dtype=np.float64)
    
            t_val = float(x_aug[-1])
            unpack(x_aug[:n_theta], params, shapes, sizes)
    
            data, target = current_batch
            model.eval()
            ce = torch.nn.CrossEntropyLoss(reduction="none")
            with torch.no_grad():
                losses = ce(model(data), target)
    
            g_vals = np.zeros(num_classes, dtype=np.float64)
            for k in range(num_classes):
                mask = target == k
                if mask.any():
                    g_vals[k] = losses[mask].mean().item() - t_val
                # else: 0  (trivially satisfied, no contribution)
    
            return g_vals
    
        # ------------------------------------------------------------------ #
        #  Constraint Jacobian:  shape (num_classes, n_aug)                   #
        #                                                                      #
        #  Row k:  [ d/d_theta  mean_loss_k(theta),   -1  ]                   #
        #                                                                      #
        #  Each row is computed independently via autograd on that class's     #
        #  mean loss — this is what gives each constraint its OWN gradient.   #
        # ------------------------------------------------------------------ #
        def constraint_jacobian(x_aug: np.ndarray) -> np.ndarray:
            if current_batch is None:
                return np.zeros((num_classes, n_aug), dtype=np.float64)
    
            unpack(x_aug[:n_theta], params, shapes, sizes)
    
            data, target = current_batch
            model.train()   # enable grad tracking
    
            # Enable grads on params
            for p in params:
                p.requires_grad_(True)
    
            ce = torch.nn.CrossEntropyLoss(reduction="none")
            jac = np.zeros((num_classes, n_aug), dtype=np.float64)
    
            # Forward pass once; retain graph so we can backward per-class
            outputs = model(data)
            losses  = ce(outputs, target)   # (batch,)
    
            for k in range(num_classes):
                mask = target == k
                if not mask.any():
                    # No samples of class k: trivial constraint, zero gradient row
                    jac[k, -1] = -1.0   # d(0 - t)/dt = -1 (still need the t-part)
                    continue
    
                mean_loss_k = losses[mask].mean()
    
                # Zero all param grads before backward
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()
    
                # Backprop through mean_loss_k only
                mean_loss_k.backward(retain_graph=True)
    
                # Harvest gradients into row k of the Jacobian
                grad_parts = []
                for p in params:
                    if p.grad is not None:
                        grad_parts.append(
                            p.grad.detach().cpu().numpy().ravel().astype(np.float64)
                        )
                    else:
                        grad_parts.append(np.zeros(p.numel(), dtype=np.float64))
    
                jac[k, :n_theta] = np.concatenate(grad_parts)
                jac[k, -1]       = -1.0    # d(g_k - t)/dt = -1
    
            # Clean up
            model.zero_grad()
            for p in params:
                p.requires_grad_(False)
            model.eval()
    
            # Safety: replace any NaN/Inf
            jac = np.nan_to_num(jac, nan=0.0, posinf=1e6, neginf=-1e6)
            return jac
    
        # ------------------------------------------------------------------ #
        #  Build and run CCSA                                                  #
        # ------------------------------------------------------------------ #
        ccsa_opt = CCSAOptimizer_Core(
            params=x0_aug,
            fun=objective_and_grad,
            g=constraint_func,
            dg=constraint_jacobian,
            bounds=None,
            **ccsa_kwargs,
        )
    
        pbar = tqdm(total=total_outer, desc="CCSA-EPI", unit="step", leave=True)
        try:
            for step in range(total_outer):
                ccsa_opt.step()
                pbar.update(1)
                if outer_calls >= total_outer:
                    break
        except TrainingComplete:
            pass
        finally:
            pbar.close()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    
        logs["final_cumulative_eval"] = cumulative_eval
    
        # Write final theta back into the model
        unpack(ccsa_opt.x_k[:n_theta], params, shapes, sizes)
    
        return batch_losses, cumulative_evals, logs


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
