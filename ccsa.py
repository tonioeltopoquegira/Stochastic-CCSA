from typing import Callable, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize
import torch
from torch.optim.optimizer import Optimizer



# Approximator 
class Approximator:
    """
    Build a convex, separable surrogate for (linearized) objective and constraints.
    MMA surrogate:
      surrogate(x) = linearization_of_f + sum_j ( p_j / (U_j - x_j + eps) + q_j / (x_j - L_j + eps) )
    and constraints are linearized and added as penalties with penalty parameter rho.

    The classic MMA surrogate is slightly different
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)

    def build(self,
              f0: float, # current objective value at x0 (scalar)
              df: np.ndarray, # gradient of objective at x0 (shape (n,))
              g0: np.ndarray, # current constraints at x0 (shape (m,)) (g_i(x)<=0 desired)
              dg: np.ndarray, # jacobian of constraints at x0 (shape (m,n))
              x0: np.ndarray, # current point (shape (n,))
              L: np.ndarray, # asymptotes lower (shape (n,))
              U: np.ndarray, # asymptotes upper (shape (n,))
              rho: float = 1.0 # penalty parameter for constraints
              ) -> Tuple[Callable[[np.ndarray], float], list]:
        """
        Returns
        -------
        surrogate_fn : function x -> surrogate_value (float)
        bounds : list of (lb, ub) tuples for scipy.optimize
        """
        n = x0.size
        m = 0 if g0 is None else int(np.atleast_1d(g0).size)
        dg = np.atleast_2d(dg) if m > 0 else np.zeros((0, n), dtype=float)
        g0 = np.atleast_1d(g0) if m > 0 else np.zeros(0, dtype=float)

        # per-coordinate curvature-like weights 
        # choose p_j and q_j based on gradient magnitude and distance to asymptotes
        p = np.maximum(np.abs(df), 1e-6) * 0.5
        q = np.maximum(np.abs(df), 1e-6) * 0.5

        # Add contribution from constraints: accumulate absolute jacobian weights
        if m > 0:
            abs_dg = np.sum(np.abs(dg), axis=0)  # shape (n,) 
            p += 0.5 * rho * abs_dg
            q += 0.5 * rho * abs_dg

        eps = self.eps

        def surrogate(x: np.ndarray) -> float:
            # x: candidate point (n,)
            # linearization term
            linear = float(f0 + np.dot(df, (x - x0)))
            # penalty for linearized constraints (quadratic penalty)
            cons_term = 0.0
            if m > 0:
                # linearized constraints g0 + dg @ (x - x0)
                lin_g = g0 + dg.dot(x - x0)
                # quadratic penalty for positive (violating) linearized constraints
                cons_term = float(0.5 * rho * np.sum(np.maximum(lin_g, 0.0) ** 2))

            
            sep = 0.0
            # avoid division by zero by adding eps
            denom1 = (U - x + eps)
            denom2 = (x - L + eps)
            sep = np.sum(p / denom1 + q / denom2)

            return linear + cons_term + float(sep)

        
        eps = 1e-12
        bounds = []
        for j in range(n):
            lb = L[j] + eps
            ub = U[j] - eps
           
            if not np.isfinite(lb):
                lb = -1e20
            if not np.isfinite(ub):
                ub = 1e20
            bounds.append((lb, ub))

        return surrogate, bounds



# Asymptote updater 
class AsymptoteUpdater:
    """
    Svanberg-style moving asymptotes.
    This follows the classic MMA asymptote updates.
    """

    def __init__(self,
                 lower_bound: Optional[float] = None,
                 upper_bound: Optional[float] = None,
                 expand: float = 1.2,
                 contract: float = 0.7,
                 sigma_min: float = 1e-6,
                 sigma_max: float = 1e20):
        
        self.expand = float(expand)
        self.contract = float(contract)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # storage for sigma and history
        self.sigma = None
        self._prev_x = None
        self._prev_prev_x = None

    def init_asymptotes(self, x0: np.ndarray):
        n = x0.size
        # Initialize sigma_j (half-widths) similar to original MMA heuristics:
        # keep within [sigma_min, sigma_max]
        sigma = np.maximum(0.5 * np.abs(x0) + 0.1, self.sigma_min)
        sigma = np.minimum(sigma, self.sigma_max)
        self.sigma = sigma.copy()

        L = x0 - self.sigma
        U = x0 + self.sigma

        # Not sure if this is needed: apply optional global bounds
        if self.lower_bound is not None:
            L = np.maximum(L, self.lower_bound)
        if self.upper_bound is not None:
            U = np.minimum(U, self.upper_bound)

        # initialize history
        self._prev_prev_x = x0.copy()
        self._prev_x = x0.copy()

        return L, U

    def update(self, x_old: np.ndarray, x_new: np.ndarray, L: np.ndarray, U: np.ndarray):
        """
        Update sigma based on three-point sign test and recompute L,U:
            x_{k-2} = self._prev_prev_x
            x_{k-1} = x_old
            x_k     = x_new
        """
        n = x_old.size

        x_k = x_new
        x_km1 = x_old
        x_km2 = self._prev_prev_x if self._prev_prev_x is not None else x_km1

        # update sigma 
        for j in range(n):
            diff1 = x_k[j] - x_km1[j]
            diff2 = x_km1[j] - x_km2[j]
            prod = diff1 * diff2
            if prod > 0.0:
                # same direction twice -> expand
                self.sigma[j] = min(self.sigma[j] * self.expand, self.sigma_max)
            elif prod < 0.0:
                # direction reversal -> contract
                self.sigma[j] = max(self.sigma[j] * self.contract, self.sigma_min)
            else:
                # no change (prod == 0)
                pass

        # recompute L and U around current x_k 
        L_new = x_k - self.sigma
        U_new = x_k + self.sigma

        # Not sure if it is needed: apply optional global bounds
        if self.lower_bound is not None:
            L_new = np.maximum(L_new, self.lower_bound)
        if self.upper_bound is not None:
            U_new = np.minimum(U_new, self.upper_bound)

        # enforce minimal width 
        widths = U_new - L_new
        min_width = self.sigma_min
        for j in range(n):
            if widths[j] < min_width:
                center = 0.5 * (U_new[j] + L_new[j])
                L_new[j] = center - 0.5 * min_width
                U_new[j] = center + 0.5 * min_width
                
                self.sigma[j] = 0.5 * min_width

       
        self._prev_prev_x = x_old.copy()
        self._prev_x = x_new.copy()

        return L_new, U_new



# Subproblem solver 

class SubproblemSolver:
    """
    Solve surrogate(x) with box bounds. Default uses scipy.optimize.minimize with L-BFGS-B.
    """

    def __init__(self, method="L-BFGS-B", options: Optional[dict] = None):
        self.method = method
        self.options = options or {"maxiter": 200}

    def solve(self, surrogate_fn: Callable[[np.ndarray], float], x0: np.ndarray, bounds: Sequence[Tuple[float, float]]):
        
        res = minimize(lambda z: float(surrogate_fn(z)), x0.copy(), method=self.method, bounds=bounds, options=self.options)
        if not res.success:
            # NOT SURE IF THIS IS A GOOD IDEA
            x_proj = np.array([np.clip(x0[j], bounds[j][0], bounds[j][1]) for j in range(x0.size)])
            return x_proj, res
        return res.x, res



# PyTorch Optimizer wrapper

class MMAOptimizerTorch(Optimizer):
    """
    PyTorch optimizer wrapper that performs outer MMA iterations.
    fun and cons closures must accept numpy arrays and return numpy results (not torch).
    """

    def __init__(self, params, fun: Callable, cons: Callable,
                 bounds: Optional[Sequence[Tuple[float, float]]] = None,
                 rho_init: float = 1.0, max_inner: int = 5,
                 sigma_min: float = 1e-6, sigma_max: float = 1e20,
                 expand: float = 1.2, contract: float = 0.7):
        defaults = dict()
        super().__init__(params, defaults)
        self.fun = fun
        self.cons = cons
        self.bounds = bounds
        self.rho = float(rho_init)
        self.max_inner = int(max_inner)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.expand = float(expand)
        self.contract = float(contract)

        # internal modular components
        self.approximator = Approximator(eps=1e-10)
        # create updater with sigma_min / sigma_max
        self.asym_updater = AsymptoteUpdater(expand=self.expand, contract=self.contract,
                                             sigma_min=self.sigma_min, sigma_max=self.sigma_max)
        self.sub_solver = SubproblemSolver()

        # metrics tracking
        n = sum(p.numel() for group in self.param_groups for p in group['params'])
        self.metrics = {
            "weighted_evals": 0,
            "sigma_adjustments": 0,
            "bound_hits": 0,
            "subproblem_iterations": [],  # Tracks iterations in subproblem solver as a list
            "sigma_min_hits": np.zeros(n),  # Tracks sigma min limit hits per parameter
            "sigma_changes": np.zeros((n, 2))  # Tracks enlargements (0) and restrictions (1) per parameter
        }

        self.state["initialized"] = False

    def _init_state_from_params(self):
        params = []
        for group in self.param_groups:
            for p in group['params']:
                params.append(p)
        flat = torch.cat([p.data.view(-1).cpu() for p in params])
        x0 = flat.clone().detach().numpy().astype(float)
        n = x0.size

        # bounds
        if self.bounds:
            lb = np.array([b[0] for b in self.bounds], dtype=float)
            ub = np.array([b[1] for b in self.bounds], dtype=float)
            if lb.size != n or ub.size != n:
                raise ValueError("bounds length must match parameter dimensionality")
        else:
            lb = -np.inf * np.ones(n, dtype=float)
            ub = np.inf * np.ones(n, dtype=float)

        # initial asymptotes 
        L, U = self.asym_updater.init_asymptotes(x0)

        self.state.update({
            "x": x0,
            "L": L,
            "U": U,
            "lb": lb,
            "ub": ub,
            "x_prev": x0.copy(),
            "rho": float(self.rho)
        })
        self.state["initialized"] = True

    def step(self):
        """
        Perform one outer MMA iteration (with inner subproblem solves).
        Returns (f_best, g_best)
        """
        if not self.state.get("initialized", False):
            self._init_state_from_params()

        x = self.state["x"].copy()
        L = self.state["L"].copy()
        U = self.state["U"].copy()
        lb = self.state["lb"]
        ub = self.state["ub"]
        rho = float(self.state["rho"])

        # Evaluate objective and constraints at current x
        fval, df = self.fun(x, grad=True)
        gval, dg = self.cons(x, grad=True)
        gval = np.atleast_1d(gval)
        dg = np.atleast_2d(dg)

        f_best = float(fval)
        x_best = x.copy()
        g_best = gval.copy()

        # inner loop: try solving subproblems (increase rho if infeasible)
        for inner in range(self.max_inner):
            surrogate_fn, bounds = self.approximator.build(f0=fval, df=df, g0=gval, dg=dg, x0=x, L=L, U=U, rho=rho)
            # intersect bounds with global box bounds:
            final_bounds = []
            for j, (lb_j, ub_j) in enumerate(bounds):
                final_lb = max(lb_j, lb[j]) if np.isfinite(lb[j]) else lb_j
                final_ub = min(ub_j, ub[j]) if np.isfinite(ub[j]) else ub_j
                final_bounds.append((final_lb, final_ub))

            x_candidate, res = self.sub_solver.solve(surrogate_fn, x, final_bounds)

            # Track subproblem iterations and add gradient iteration
            self.metrics["subproblem_iterations"].append(res.nit + 1)

            f_cand, df_cand = self.fun(x_candidate, grad=True)
            g_cand, dg_cand = self.cons(x_candidate, grad=True)
            g_cand = np.atleast_1d(g_cand)
            improved = (f_cand < f_best - 1e-12)
            feasible_cand = np.all(g_cand <= 0.0)

            # metrics tracking
            self.metrics["weighted_evals"] += 1
            if not improved:
                # increase penalty and try again
                rho *= 2.0
                self.metrics["sigma_adjustments"] += 1
            else:
                f_best, x_best, g_best = float(f_cand), x_candidate.copy(), g_cand.copy()
                # accept this inner solution
                break

        # update asymptotes based on three-point rule 
        L_new, U_new = self.asym_updater.update(x_old=x, x_new=x_best, L=L.copy(), U=U.copy())

        # Track sigma changes and min hits
        for j in range(len(self.asym_updater.sigma)):
            if self.asym_updater.sigma[j] == self.sigma_min:
                self.metrics["sigma_min_hits"][j] += 1
            if self.asym_updater.sigma[j] > self.state["L"][j]:
                self.metrics["sigma_changes"][j, 0] += 1  # Enlargement
            elif self.asym_updater.sigma[j] < self.state["L"][j]:
                self.metrics["sigma_changes"][j, 1] += 1  # Restriction

        # write back state
        self.state["x"] = x_best.copy()
        self.state["L"] = L_new
        self.state["U"] = U_new
        self.state["x_prev"] = x.copy()
        self.state["rho"] = float(rho)

        # write back to torch params (preserve dtype/device)
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.data.numel()
                new_vals = torch.from_numpy(x_best[idx:idx + numel].astype(p.data.cpu().numpy().dtype))
                new_vals = new_vals.to(device=p.data.device, dtype=p.data.dtype)
                p.data.copy_(new_vals.view_as(p))
                idx += numel

        return f_best, g_best, self.metrics

