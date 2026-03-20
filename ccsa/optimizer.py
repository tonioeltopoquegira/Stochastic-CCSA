from typing import Callable, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from ccsa.params import (MMA_RhoParams, MMA_SigmaParams, update_rho,
                         AdamCurvParams, init_adam_curv_state,
                         adam_curv_update, adam_secant_update)
from ccsa.asymptote import AsymptoteUpdater
from ccsa.dual import DualSubproblemBuilder
from ccsa.feasibility_minimization import feasibility_solver

# -------------------------
# Modular MMA optimizer (flat parameter vector)
# -------------------------
class CCSAOptimizer: 
    MMA_RHOMIN = 1e-5

    def __init__(self,
                 params: np.ndarray,
                 fun: Callable,
                 g: Optional[Callable] = None,
                 bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 rho_init: float = 1.0,
                 max_inner: int = 5,
                 expand: float = 1.2,
                 contract: float = 0.7,
                 rho_params: Optional[MMA_RhoParams] = None,
                 sigma_params: Optional[MMA_SigmaParams] = None,
                 sigma_min: float = 1e-3,
                 use_quadratic_surrogates: bool = False,
                 df: Optional[Callable] = None,
                 dg: Optional[Callable] = None,
                 x0: Optional[np.ndarray] = None,
                 conservative = True,
                 update_rule: str = 'multiplier',   # 'multiplier' | 'adam_violation' | 'adam_secant'
                 update_rule_kwargs: Optional[dict] = None,
                 per_coord_rho: bool = False):      # Per-coordinate curvature for quadratic surrogates
        """
        params: initial flat parameter array (will be copied). If x0 provided, it overrides params.
        fun: callable (x, grad=True) -> (f, df) if grad requested, otherwise fun(x) -> float
        g: callable(x) -> array of constraints (<=0)
        df, dg: optional gradient/jacobian providers
        bounds: sequence of (lb, ub) arrays or None
        """
        self.fun = fun
        self.g = g
        self.df = df
        self.dg = dg
        self.max_inner = int(max_inner)
        self.per_coord_rho = bool(per_coord_rho)  # Per-coordinate curvature flag (quadratic mode only)
        self.conservative = conservative
        
        # Validate: per_coord_rho only works with quadratic surrogates and multiplier/adam_secant
        if self.per_coord_rho:
            if not use_quadratic_surrogates:
                raise ValueError("per_coord_rho requires use_quadratic_surrogates=True")
            if update_rule == 'adam_violation':
                raise ValueError("per_coord_rho incompatible with adam_violation (violation is global). Use multiplier or adam_secant.")
        
        # Initialize rho placeholder
        self.rho = float(rho_init)

        self.rho_params = rho_params if rho_params is not None else MMA_RhoParams()
        # if no sigma_params provided, create from legacy small set
        if sigma_params is None:
            sigma_params = MMA_SigmaParams(expand=expand, contract=contract, sigma_min=sigma_min)
        self.sigma_params = sigma_params

        if not self.conservative and self.max_inner > 1:
            import warnings
            warnings.warn(
                "max_inner > 1 has no effect when conservative=False: "
                "the inner loop always exits after one iteration.",
                UserWarning, stacklevel=2
            )

        # initialize x_k (current major iterate)
        # This can be changed directly to the first one I guess
        if x0 is not None:
            x0_arr = np.asarray(x0, dtype=float).ravel()
        else:
            x0_arr = np.asarray(params, dtype=float).ravel()
        self.x_k = x0_arr.copy()
        n = self.x_k.size
        self.grad_f_k = None   # stores gradient at x_k across outer steps, used by adam_secant
        self.grad_g_k = None   # stores dg/dx rows at x_k across outer steps, for adam_secant
        
        # NOW convert rho to per-coordinate vector if needed (now we know n)
        if self.per_coord_rho:
            self.rho = np.full(n, float(rho_init), dtype=float)  # (n,) vector

        # bounds handling: either None or (lb_array, ub_array)
        if bounds is not None:
            lb_arr = np.asarray(bounds[0], dtype=float).ravel()
            ub_arr = np.asarray(bounds[1], dtype=float).ravel()
            if lb_arr.size != n or ub_arr.size != n:
                raise ValueError("bounds arrays must match parameter dimensionality")
        else:
            lb_arr = -np.inf * np.ones(n, dtype=float)
            ub_arr = np.inf * np.ones(n, dtype=float)

        self.lb = lb_arr
        self.ub = ub_arr

        # asymptotes manager
        self.asym = AsymptoteUpdater(sigma_params=self.sigma_params,
                                     lower_bound=self.lb, upper_bound=self.ub)
        
        L, U = self.asym.init_asymptotes(self.x_k)
        self.L = L
        self.U = U

        # per-constraint curvature parameters rho_c (initially None -> set at first evaluate)
        self.rho_c = None

        # flag: use quadratic separable surrogates (CCSA-style) instead of MMA moving-asymptotes
        self.use_quadratic_surrogates = bool(use_quadratic_surrogates)

        # bookkeeping (some keys renamed to match math)
        self.metrics = {
            "weighted_evals": 0,
            "sigma_adjustments": 0,
            "bound_hits": 0,
            "cumulative_wval": 0.0,
            "acceptance_stats": {
                "feasible_accept": 0,
                "infeasible_accept": 0,
                "improve_only": 0,
                "reject": 0,
                "feasibility_min": 0
            },
            "curvature_updates": {
                "multiplier_on_rejection": 0,
                "adam_violation_updates": 0,
                "adam_secant_updates": 0,
                "total_updates": 0
            },
            "rho_history": [],
            "rho_c_history": [],
            "rho_vec_history": [],                  # Per-coordinate rho history (when per_coord_rho=True)
            "objective_violation_history": [],     # Track OBJECTIVE violations only
            "constraint_violation_history": [],    # Track CONSTRAINT violations only
            "violation_history": []                 # Legacy: combined
        }
        self.history = {
            "x": [x0_arr.copy()],
            "weighted_evals": [1],
            "sigma": [],    # store sigma evolution
            "rho": [],      # store scalar rho
            "rho_c": []     # store per-constraint rho_c vector
        }
        self.acceptance_trace = []

        # adaptive curvature update settings
        self.update_rule = str(update_rule) if update_rule is not None else 'fixed'
        self.update_rule_kwargs = dict(update_rule_kwargs) if update_rule_kwargs is not None else {}
        if self.update_rule in ('adam_violation', 'adam_secant'):
            kw = {'lr': 1e-2, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8,
                'min_curv': 1e-6, 'max_curv': 1e3}
            kw.update(self.update_rule_kwargs)
            self._adam_curv_params = AdamCurvParams(**kw)
            # SEPARATE Adam states: one for rho (objective), one list for rho_c (constraints)
            if self.per_coord_rho:
                # Per-coordinate: list of Adam states, one per coordinate
                self._curv_state_obj_list = [init_adam_curv_state() for _ in range(n)]
            else:
                # Scalar: single Adam state for rho
                self._curv_state_obj = init_adam_curv_state()      # For rho (objective violations)
            self._curv_state_c_list = []                        # For rho_c (constraint violations)
        else:
            self._adam_curv_params = None
            if self.per_coord_rho:
                self._curv_state_obj_list = []
            else:
                self._curv_state_obj = None
            self._curv_state_c_list = []

    def reset(self, x0: Optional[np.ndarray] = None):
        """
        Reset optimizer state to initial conditions.
        Useful for running multiple optimizations from the same starting point
        without creating a new optimizer instance.
        
        Args:
            x0: Optional new starting point. If None, uses the original x0.
        """
        if x0 is None:
            x0_arr = self.x_k.copy()
        else:
            x0_arr = np.atleast_1d(np.asarray(x0, dtype=float))
        
        # Reset position
        self.x_k = x0_arr.copy()
        n = self.x_k.size
        
        # Reset gradients
        self.grad_f_k = None
        self.grad_g_k = None
        
        # Reset curvature parameters
        self.rho = float(self.rho_params.rho_init if hasattr(self.rho_params, 'rho_init') else 1.0)
        if self.rho_c is not None:
            self.rho_c = np.ones(self.rho_c.size) * 1.0
        # Reset separate Adam states for objective and constraints
        if self.update_rule in ('adam_violation', 'adam_secant'):
            self._curv_state_obj = init_adam_curv_state()
            self._curv_state_c_list = [init_adam_curv_state() for _ in range(self.rho_c.size)] if self.rho_c is not None else []
        
        # Reset asymptotes
        L, U = self.asym.init_asymptotes(self.x_k)
        self.L = L
        self.U = U
        
        # Reset Adam curvature states
        if self.update_rule in ('adam_violation', 'adam_secant'):
            self._curv_state = init_adam_curv_state()
            if hasattr(self, '_curv_state_c') and self._curv_state_c is not None:
                self._curv_state_c = None
        
        # Reset metrics
        self.metrics = {
            "weighted_evals": 0,
            "sigma_adjustments": 0,
            "bound_hits": 0,
            "cumulative_wval": 0.0,
            "acceptance_stats": {
                "feasible_accept": 0,
                "infeasible_accept": 0,
                "improve_only": 0,
                "reject": 0,
                "feasibility_min": 0
            },
            "curvature_updates": {
                "multiplier_on_rejection": 0,
                "adam_violation_updates": 0,
                "adam_secant_updates": 0,
                "total_updates": 0
            },
            "rho_history": [],
            "rho_vec_history": [],
            "violation_history": []
        }
        
        # Reset history
        self.history = {
            "x": [x0_arr.copy()],
            "weighted_evals": [0],
            "sigma": [],
            "rho": [],
            "rho_c": []
        }
        
        # Reset acceptance trace
        self.acceptance_trace = []





    def step(self):
        x_k = self.x_k.copy()
        L = self.L.copy()
        U = self.U.copy()
        rho = float(self.rho)

        # Evaluate f and gradient at x_k
        if self.df is None:
            f_k, grad_f_k = self.fun(x_k, grad=True)
        else:
            f_k = float(self.fun(x_k))
            grad_f_k = np.asarray(self.df(x_k), dtype=float).ravel()

        # Evaluate constraints at x_k
        if self.g is not None:
            g_k = np.atleast_1d(self.g(x_k)).astype(float)
            m = g_k.size
            if self.dg is not None:
                grad_g_k = np.atleast_2d(self.dg(x_k)).astype(float)
            else:
                grad_g_k = np.zeros((m, x_k.size), dtype=float)
            if self.rho_c is None:
                self.rho_c = np.full(m, rho)
            # initialize per-constraint Adam state if using adaptive updates
            if self.update_rule == 'adam_curv' and getattr(self, '_curv_state_c', None) is None:
                # create list of per-constraint states
                self._curv_state_c = [init_adam_curv_state() for _ in range(m)]
        else:
            g_k = np.zeros(0, dtype=float)
            grad_g_k = np.zeros((0, x_k.size), dtype=float)
            m = 0
            if self.rho_c is None:
                self.rho_c = np.zeros(0, dtype=float)
            if self.update_rule == 'adam_curv' and getattr(self, '_curv_state_c', None) is None:
                self._curv_state_c = []

        f_best = float(f_k)
        x_best = x_k.copy()
        g_best = g_k.copy()
        wval_used = 0.0
        accept_type = "reject"
        grad_f_best = grad_f_k.copy()
        violation_f = 0.0
        violation_gc = np.zeros(m, dtype=float)

        # Initialize metric fields if first run
        if "acceptance_stats" not in self.metrics:
            self.metrics["acceptance_stats"] = {
                "feasible_accept": 0,
                "infeasible_accept": 0,
                "improve_only": 0,
                "reject": 0,
                "feasibility_min": 0
            }
        if "sigma_history" not in self.metrics:
            self.metrics["sigma_history"] = []
        if "rho_history" not in self.metrics:
            self.metrics["rho_history"] = []
        if "rho_c_history" not in self.metrics:
            self.metrics["rho_c_history"] = []
        if "x_history" not in self.metrics:
            self.metrics["x_history"] = [self.x_k.copy()]
        if "cumulative_weighted_evals_history" not in self.metrics:
            self.metrics["cumulative_weighted_evals_history"] = [1]

        # Inner loop: solve subproblem(s)
        for inner in range(self.max_inner):
            sigma_vec = self.asym.sigma.copy()

            builder = DualSubproblemBuilder(
                f_k=f_k, grad_f_k=grad_f_k, x_k=x_k, g_k=g_k, grad_g_k=grad_g_k,
                lb=self.lb, ub=self.ub, sigma=sigma_vec, rho=rho, rho_c=self.rho_c,
                quadratic=self.use_quadratic_surrogates
            )

            # Determine x_candidate
            y_opt = np.zeros(m, dtype=float)
            if m > 0:
                obj_only, obj_with_grad = builder.build_dual_objective()
                y0 = np.zeros(m, dtype=float)
                dual_bounds = [(0.0, np.inf) for _ in range(m)]
                res = minimize(lambda yy: obj_with_grad(yy),
                            y0,
                            method='L-BFGS-B',
                            jac=True,
                            bounds=dual_bounds,
                            options={'maxiter': 10, 'ftol': 1e-30})
                y_opt = res.x

            x_candidate, tilde_f, tilde_gc, w_val, val_extra = builder.reconstruct_xcandidate_from_y(y_opt)

            # Evaluate true f and g
            if self.df is None:
                f_cur, grad_f_cur = self.fun(x_candidate, grad=True)
            else:
                f_cur = float(self.fun(x_candidate))
                grad_f_cur = np.asarray(self.df(x_candidate), dtype=float).ravel()

            gcur = np.atleast_1d(self.g(x_candidate)).astype(float) if m > 0 else np.zeros(0, dtype=float)

            # inside the inner loop
            violation_f = float(f_cur - tilde_f)
            violation_gc = (gcur - tilde_gc) if m > 0 else np.zeros(0, dtype=float)

            self.metrics["weighted_evals"] += 0.5

            # Acceptance criteria
            improved = f_cur < f_best
            feasible_cur = np.all(gcur <= 0.0) if m > 0 else True
            inner_done = tilde_f >= f_cur and (np.all(tilde_gc >= gcur) if m > 0 else True)
            accept = False

            if improved and (inner_done or feasible_cur or m == 0):
                accept = True
                accept_type = "feasible_accept" if feasible_cur and m > 0 else "improve_only"
            elif m > 0 and not feasible_cur and np.max(gcur) < np.max(np.maximum(g_k, 0.0)):
                accept = True
                accept_type = "infeasible_accept"
            else:
                if not self.conservative:
                    # run feasibility minimization using current curvature
                    x_bar, success = feasibility_solver(self.L, self.U, x_k, g_k, grad_g_k, (self.lb, self.ub))
                    
                    # evaluate true f and g at x_bar
                    if self.df is None:
                        f_bar, grad_f_bar = self.fun(x_bar, grad=True)
                    else:
                        f_bar = float(self.fun(x_bar))
                        grad_f_bar = np.asarray(self.df(x_bar), dtype=float).ravel()
                    
                    g_bar = np.atleast_1d(self.g(x_bar)).astype(float) if m > 0 else np.zeros(0, dtype=float)
                    
                    # accept the feasibility-minimized point
                    accept = True
                    accept_type = 'feasibility_min'
                    f_best = float(f_bar)
                    x_best = x_bar.copy()
                    g_best = g_bar.copy() if m > 0 else np.zeros(0, dtype=float)
                    grad_f_k = grad_f_bar.copy()
                    wval_used = 0.0  # feasibility minimization doesn't use w_val in the same way
                    grad_f_best = grad_f_bar.copy()  
                else:
                    accept_type = "reject"
                    

            if accept:
                f_best = float(f_cur)
                x_best = x_candidate.copy()
                g_best = gcur.copy() if m > 0 else np.zeros(0, dtype=float)
                grad_f_cur_accepted = grad_f_cur.copy()  # used by adam_secant at bottom
                wval_used = w_val
                break
            else:
                if self.update_rule == 'multiplier':
                    if violation_f > 0.0:
                        # When per_coord_rho: rho is (n,), w_val is (n,), use element-wise
                        # When not: rho is scalar, w_val is scalar, use scalar
                        rho = update_rho(rho, violation_f, w_val, self.rho_params)
                        self.metrics["curvature_updates"]["multiplier_on_rejection"] += 1
                        self.metrics["curvature_updates"]["total_updates"] += 1
                    if m > 0 and self.rho_c is not None and self.rho_c.size > 0:
                        # Ensure sizes match before boolean indexing
                        if self.rho_c.size == m:
                            # Convert to 1D array BEFORE creating mask to handle scalar case
                            violation_gc_1d = np.atleast_1d(np.asarray(violation_gc).ravel())
                            mask = violation_gc_1d > 0.0
                            if np.any(mask):
                                # For constraints: rho_c is per-constraint (m,), w_val is per-coordinate (n,)
                                # Use mean of w_val for constraint updates
                                w_val_for_c = np.mean(w_val) if isinstance(w_val, np.ndarray) else w_val
                                self.rho_c[mask] = update_rho(
                                    self.rho_c[mask], violation_gc_1d[mask], w_val_for_c, self.rho_params
                                )
                wval_used = w_val
                


        # Update of asymptotes (NOT used in quadratic surrogate mode)
        L_new, U_new = self.asym.update(x_km1=x_k, x_kp1=x_best, L=self.L.copy(), U=self.U.copy())

        # decay not applied for adam_violation/adam_secant)
        if self.update_rule == 'multiplier':
            rho = self.rho_params.decay * rho
            if m > 0 and self.rho_c is not None and self.rho_c.size > 0:
                self.rho_c = self.rho_params.decay * self.rho_c

        # Minimum enforced for everyone
        rho = max(rho, self.MMA_RHOMIN)
        if m > 0 and self.rho_c is not None and self.rho_c.size > 0:
            self.rho_c = np.maximum(self.rho_c, self.MMA_RHOMIN)


        # Update of asymptotes (NOT used in quadratic surrogate mode)
        self.x_k = x_best.copy()
        self.L = L_new
        self.U = U_new
        self.rho = float(rho)

        # ===== OBJECTIVE CURVATURE ADAPTIVE UPDATE =====
        # Update rho based ONLY on objective violation (surrogate underestimation)
        if self.update_rule in ('adam_violation', 'adam_secant') and self._curv_state_obj is not None:
            if self.update_rule == 'adam_violation':
                # Signal: objective violation only
                new_rho, self._curv_state_obj = adam_curv_update(
                    self._curv_state_obj, violation_f, self.rho, self._adam_curv_params
                )
                self.metrics["curvature_updates"]["adam_violation_updates"] += 1
                self.metrics["curvature_updates"]["total_updates"] += 1
                # Track objective violation for diagnostics
                if "objective_violation_history" not in self.metrics:
                    self.metrics["objective_violation_history"] = []
                self.metrics["objective_violation_history"].append(float(violation_f))
            else:  # adam_secant
                # Signal: gradient-based curvature estimate for objective
                if self.grad_f_k is not None:
                    new_rho, self._curv_state_obj = adam_secant_update(
                        self._curv_state_obj,
                        grad_new=grad_f_cur,      # gradient at accepted x_best
                        grad_old=self.grad_f_k,   # gradient at x_k (previous outer iterate)
                        x_new=x_best,
                        x_old=x_k,
                        curv=self.rho,
                        params=self._adam_curv_params
                    )
                    self.metrics["curvature_updates"]["adam_secant_updates"] += 1
                    self.metrics["curvature_updates"]["total_updates"] += 1
                else:
                    new_rho = self.rho   # first step, no previous gradient yet
            self.rho = float(new_rho)

        # ===== CONSTRAINT CURVATURE ADAPTIVE UPDATE =====
        # Update rho_c based ONLY on constraint violations (per-constraint)
        if self.update_rule in ('adam_violation', 'adam_secant') and m > 0:
            try:
                # Initialize Adam states for constraints if needed
                if len(self._curv_state_c_list) != self.rho_c.size:
                    self._curv_state_c_list = [init_adam_curv_state() for _ in range(self.rho_c.size)]
                
                # Ensure violation_gc is 1D for safe indexing (do this once)
                violation_gc_1d = np.atleast_1d(np.asarray(violation_gc).ravel())
                
                # Track constraint violations for diagnostics
                if "constraint_violation_history" not in self.metrics:
                    self.metrics["constraint_violation_history"] = []
                self.metrics["constraint_violation_history"].append(violation_gc_1d.copy())
                
                for i in range(self.rho_c.size):
                    if self.update_rule == 'adam_violation':
                        # Signal: constraint violation for constraint i ONLY
                        signal = float(violation_gc_1d[i]) if i < violation_gc_1d.size else 0.0
                        new_rc, self._curv_state_c_list[i] = adam_curv_update(
                            self._curv_state_c_list[i], signal, float(self.rho_c[i]), self._adam_curv_params
                        )
                    else:  # adam_secant — use per-constraint gradient rows
                        if self.grad_g_k is not None and self.dg is not None:
                            new_rc, self._curv_state_c_list[i] = adam_secant_update(
                                self._curv_state_c_list[i],
                                grad_new=grad_g_k[i],         # current dg/dx row i
                                grad_old=self.grad_g_k[i],    # previous outer step row i
                                x_new=x_best,
                                x_old=x_k,
                                curv=float(self.rho_c[i]),
                                params=self._adam_curv_params
                            )
                        else:
                            new_rc = self.rho_c[i]
                    self.rho_c[i] = float(new_rc)
            except Exception:
                pass
        
        self.grad_f_k = grad_f_cur.copy() if accept_type != 'reject' else self.grad_f_k
        if m > 0 and accept_type != 'reject':
            self.grad_g_k = grad_g_k.copy()

        # ---- Record histories and metrics ----
        self.metrics["cumulative_wval"] += float(wval_used)
        self.metrics["acceptance_stats"][accept_type] += 1
        self.metrics["weighted_evals"] += 1.0

        # keep both metrics and history in sync for robust plotting/analysis
        self.metrics["x_history"].append(x_best.copy())
        self.metrics["sigma_history"].append(self.asym.sigma.copy())
        self.metrics["rho_history"].append(self.rho)
        self.metrics["rho_c_history"].append(self.rho_c.copy() if self.rho_c is not None else np.zeros(0))
        # cumulative weighted evals history - keep float for precise diffs
        self.metrics["cumulative_weighted_evals_history"].append(float(self.metrics["weighted_evals"]))

        # also maintain the 'history' dict so summary functions can rely on a single source
        if "x" in self.history:
            self.history["x"].append(x_best.copy())
        else:
            self.history["x"] = [x_best.copy()]
        self.history.setdefault("sigma", []).append(self.asym.sigma.copy())
        self.history.setdefault("rho", []).append(self.rho)
        self.history.setdefault("rho_c", []).append(self.rho_c.copy() if self.rho_c is not None else np.zeros(0))
        self.history.setdefault("weighted_evals", []).append(float(self.metrics["weighted_evals"]))

        # record acceptance trace per outer iteration (create if needed)
        if not hasattr(self, "acceptance_trace"):
            self.acceptance_trace = []
        self.acceptance_trace.append(accept_type)
        
        # Track violations (use maximum constraint violation or 0 if feasible)
        constraint_violation = float(np.max(g_best)) if g_best.size > 0 else 0.0
        self.metrics["violation_history"].append(constraint_violation)

        return f_best, g_best, dict(self.metrics)




    

    def summarize_diagnostics(self, show: bool = True):
        """
        Improved diagnostics and plotting. Features:
          - Sigma heatmap (log color scale) and mean ± std on log-y
          - rho (objective) and rho_c (per-constraint) evolution (thin lines, log-y)
          - Acceptance overlay on loss (y-axis = f(x), log scale) using recorded acceptance_trace
          - Inner solves per outer step (clean line)
        Returns a dict with extracted arrays for programmatic usage.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        import matplotlib.colors as mcolors
        from collections import Counter

        # --- Extract histories robustly (support both self.history and self.metrics) ---
        sigma_hist_list = self.history.get("sigma", None)
        if sigma_hist_list is None or len(sigma_hist_list) == 0:
            sigma_hist_list = self.metrics.get("sigma_history", [])

        # Make 2D array shape (n_outer, n_vars). If ragged, pad with nan.
        if len(sigma_hist_list) > 0:
            try:
                sigma_arr = np.vstack([np.asarray(s) for s in sigma_hist_list])
            except Exception:
                # ragged: pad with nan
                tmp = [np.asarray(s, dtype=float) for s in sigma_hist_list]
                maxn = max(x.size for x in tmp)
                sigma_arr = np.full((len(tmp), maxn), np.nan, dtype=float)
                for i, s in enumerate(tmp):
                    sigma_arr[i, : s.size] = s
        else:
            sigma_arr = np.zeros((0, 0), dtype=float)

        rho_hist = np.array(self.history.get("rho", []) if len(self.history.get("rho", [])) > 0 else self.metrics.get("rho_history", []), dtype=float)
        if rho_hist.size == 0:
            rho_hist = np.array([], dtype=float)

        rho_c_hist_list = self.history.get("rho_c", None)
        if rho_c_hist_list is None or len(rho_c_hist_list) == 0:
            rho_c_hist_list = self.metrics.get("rho_c_history", [])
        if len(rho_c_hist_list) > 0:
            try:
                rho_c_arr = np.vstack([np.asarray(r) for r in rho_c_hist_list])
            except Exception:
                tmp = [np.asarray(r, dtype=float) for r in rho_c_hist_list]
                maxm = max(x.size for x in tmp) if len(tmp) > 0 else 0
                rho_c_arr = np.full((len(tmp), maxm), np.nan, dtype=float)
                for i, r in enumerate(tmp):
                    rho_c_arr[i, : r.size] = r
        else:
            rho_c_arr = np.zeros((0, 0), dtype=float)

        cum_evals = np.array(self.metrics.get("cumulative_weighted_evals_history", self.history.get("weighted_evals", [1.0])), dtype=float)
        if cum_evals.size == 0:
            cum_evals = np.array([1.0], dtype=float)

        # Build f(x) history
        x_hist_list = self.history.get("x", None)
        if x_hist_list is None or len(x_hist_list) == 0:
            x_hist_list = self.metrics.get("x_history", [])
        f_hist = []
        for xx in x_hist_list:
            try:
                f_hist.append(float(self.fun(xx)))
            except Exception:
                try:
                    f_val, _ = self.fun(xx, grad=True)
                    f_hist.append(float(f_val))
                except Exception:
                    f_hist.append(np.nan)
        f_hist = np.array(f_hist, dtype=float)
        n_outer = f_hist.size if f_hist.size > 0 else max(1, sigma_arr.shape[0], rho_hist.size, rho_c_arr.shape[0], cum_evals.size)

        # Acceptance trace (prefer explicit trace)
        if hasattr(self, "acceptance_trace") and len(self.acceptance_trace) > 0:
            accept_labels = list(self.acceptance_trace)
        else:
            # fallback: mark unknown for each recorded outer iteration
            accept_labels = ["unknown"] * n_outer

        # Color/marker mapping
        accept_color_map = {
            "feasible_accept": "#2ca02c",
            "infeasible_accept": "#ffcc00",
            "improve_only": "#1f77b4",
            "reject": "#d62728",
            "unknown": "#7f7f7f"
        }
        accept_marker_map = {
            "feasible_accept": "o",
            "infeasible_accept": "s",
            "improve_only": "^",
            "reject": "x",
            "unknown": "d"
        }
        accept_colors = [accept_color_map.get(a, "#7f7f7f") for a in accept_labels]
        accept_markers = [accept_marker_map.get(a, 'o') for a in accept_labels]

        # inner counts per outer iteration from cum_evals
        inner_counts = np.diff(cum_evals, prepend=cum_evals[0]).astype(float)

        # sigma statistics
        sigma_mean = np.nanmean(sigma_arr, axis=1) if sigma_arr.size != 0 else np.array([])
        sigma_std = np.nanstd(sigma_arr, axis=1) if sigma_arr.size != 0 else np.array([])

        # ----- Start plotting -----
        fig = plt.figure(figsize=(15, 10))

        # (1) Sigma heatmap (log color scale)
        ax1 = fig.add_subplot(2, 2, 1)
        if sigma_arr.size != 0:
            # for LogNorm we need positive values; replace zeros/nans with a small positive floor
            floor = max(1e-30, np.nanmin(sigma_arr[np.isfinite(sigma_arr) & (sigma_arr > 0)]) if np.any((sigma_arr>0) & np.isfinite(sigma_arr)) else 1e-12)
            sigma_plot = np.where(np.isfinite(sigma_arr) & (sigma_arr > 0), sigma_arr, floor)
            # transpose => rows = variables, cols = outer iters
            im = ax1.imshow(sigma_plot.T, aspect='auto', origin='lower', cmap='viridis',
                            norm=LogNorm(vmin=sigma_plot.min(), vmax=sigma_plot.max()))
            cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
            cbar.set_label(r'$\sigma_j$ (log color)')
            ax1.set_xlabel("outer iteration")
            ax1.set_ylabel("variable index $j$")
            ax1.set_title("σ evolution (heatmap, log color)")
        else:
            ax1.text(0.5, 0.5, "no σ history", ha='center', va='center')
            ax1.set_title("σ evolution")

        # (2) Sigma mean ± std on log-y
        ax1b = fig.add_subplot(2, 2, 2)
        if sigma_mean.size > 0:
            ax1b.plot(range(sigma_mean.size), sigma_mean, marker='o', linewidth=1.4, label='mean σ')
            ax1b.fill_between(range(sigma_mean.size),
                              np.maximum(sigma_mean - sigma_std, 1e-30),
                              sigma_mean + sigma_std,
                              alpha=0.25)
            ax1b.set_yscale('log')
            ax1b.set_xlabel("outer iteration")
            ax1b.set_ylabel("σ (log scale)")
            ax1b.set_title("mean σ ± std (log y)")
            ax1b.grid(True, which='both', linestyle='--', linewidth=0.4)
            ax1b.legend()
        else:
            ax1b.text(0.5, 0.5, "no σ stats", ha='center', va='center')
            ax1b.set_title("mean σ ± std")

        # (3) Rho (objective) and rho_c (constraints) evolution (log-y, thin lines)
        ax2 = fig.add_subplot(2, 2, 3)
        if rho_hist.size > 0:
            ax2.plot(range(rho_hist.size), rho_hist, linewidth=1.6, color='black', label=r'$\rho$ (objective)')
        if rho_c_arr.size != 0:
            n_constraints = rho_c_arr.shape[1]
            # choose colormap for constraint lines
            cmap = plt.cm.tab20
            for i in range(n_constraints):
                color = cmap(i % 20)
                ax2.plot(range(rho_c_arr.shape[0]), rho_c_arr[:, i], linewidth=0.8, color=color, alpha=0.7)
            # plot mean
            rho_c_mean = np.nanmean(rho_c_arr, axis=1)
            ax2.plot(range(rho_c_mean.size), rho_c_mean, linewidth=1.2, linestyle='--', color='purple', label='mean(ρ_c)')
        ax2.set_yscale('log')
        ax2.set_xlabel("outer iteration")
        ax2.set_ylabel("ρ (log scale)")
        ax2.set_title("ρ (objective) and ρ_c (constraints) evolution")
        ax2.grid(True, which='both', linestyle='--', linewidth=0.4)
        ax2.legend(loc='upper left', fontsize='small')

        # (4) Acceptance overlay on loss (bottom-right)
        ax3 = fig.add_subplot(2, 2, 4)
        if f_hist.size > 0:
            xs = np.arange(f_hist.size)
            # plot loss as line
            ax3.plot(xs, f_hist, linewidth=1.4, label='f(x)', color='black', alpha=0.6)
            ax3.set_yscale('log')
            # overlay acceptance colored markers at the exact f(x) value
            plotted = set()
            npts = min(len(accept_labels), f_hist.size)
            for i in range(npts):
                lbl = accept_labels[i]
                color = accept_color_map.get(lbl, "#7f7f7f")
                marker = accept_marker_map.get(lbl, "o")
                ax3.scatter(i, f_hist[i], color=color, edgecolor='k', s=70, marker=marker, zorder=10)
                plotted.add(lbl)
            # legend for acceptance
            handles = []
            for lbl in ["feasible_accept", "infeasible_accept", "improve_only", "reject", "unknown"]:
                if lbl in plotted:
                    handles.append(plt.Line2D([0], [0], marker=accept_marker_map[lbl], color='w',
                                              markerfacecolor=accept_color_map[lbl], markeredgecolor='k',
                                              markersize=8, linestyle='None', label=lbl))
            if handles:
                ax3.legend(handles=handles, title="acceptance", loc='upper right', fontsize='small')
            ax3.set_xlabel("outer iteration")
            ax3.set_ylabel("f(x) (log scale)")
            ax3.set_title("Loss evolution with acceptance markers (y = loss)")
            ax3.grid(True, which='both', linestyle='--', linewidth=0.4)
        else:
            ax3.text(0.5, 0.5, "no f(x) history", ha='center', va='center')
            ax3.set_title("Loss evolution")

        # Supplementary small figure for inner solves only (clean line)
        plt.figure(figsize=(8, 3.5))
        xi = np.arange(inner_counts.size)
        plt.plot(xi, inner_counts, marker='o', linewidth=1.4, color='#1f77b4', label='inner solves per outer')
        plt.xlabel("outer iteration")
        plt.ylabel("inner solves")
        plt.title("Inner solves per outer iteration")
        plt.grid(True, linestyle='--', linewidth=0.4)
        plt.legend()
        if show:
            plt.show()

        # textual summary
        print("\n=== MMA Diagnostic Summary ===")
        print(f"Recorded outer iterations (f history): {f_hist.size}")
        if sigma_mean.size > 0:
            print(f"Sigma: final mean = {sigma_mean[-1]:.3e}, final std = {sigma_std[-1]:.3e}")
        if rho_hist.size > 0:
            print(f"Rho (objective): final = {rho_hist[-1]:.3e}")
        if rho_c_arr.size != 0:
            print(f"Rho_c (constraints): shape = {rho_c_arr.shape}, final mean = {np.nanmean(rho_c_arr[-1]):.3e}")
        print("\nAcceptance counts:")
        print(Counter(accept_labels))
        print(f"\nAverage inner solves per outer step (observed): {np.mean(inner_counts):.2f}")

        # return arrays for programmatic inspection
        return {
            "sigma_arr": sigma_arr,
            "sigma_mean": sigma_mean,
            "sigma_std": sigma_std,
            "rho_hist": rho_hist,
            "rho_c_arr": rho_c_arr,
            "f_hist": f_hist,
            "accept_labels": accept_labels,
            "inner_counts": inner_counts
        }

    def plot_curvature_diagnostics(self, figsize=(14, 10), show=True):
        """
        Intelligent diagnostics for curvature adaptation showing:
        1. Curvature (rho) evolution - scalar or smart per-coordinate
        2. Update rule statistics (multiplier, adam_violation, adam_secant)
        3. Violation history and acceptance statistics
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract data from metrics
        rho_hist = np.array(self.metrics["rho_history"])
        violation_hist = np.array(self.metrics["violation_history"])
        
        curvature_updates = self.metrics["curvature_updates"]
        acceptance_stats = self.metrics.get("acceptance_stats", {})
        
        # Check if per_coord_rho
        is_per_coord = False  # This version doesn't support per_coord_rho
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
        
        # --- Subplot 1: Curvature Evolution ---
        iterations = np.arange(len(violation_hist))
        
        if not is_per_coord:
            # Scalar rho - simple plot
            if rho_hist.ndim == 1:
                ax1.semilogy(iterations, rho_hist, marker='o', linewidth=1.5, color='#1f77b4', label='rho (objective)')
            else:
                ax1.semilogy(iterations, rho_hist[:, 0], marker='o', linewidth=1.5, color='#1f77b4', label='rho (objective)')
            ax1.set_ylabel("rho value (log scale)", fontsize=10)
            ax1.set_title("Curvature (rho) Evolution - Scalar Mode", fontsize=12, fontweight='bold')
            ax1.grid(True, linestyle='--', linewidth=0.4)
            ax1.legend()
        else:
            # Per-coordinate rho - smart plotting (median + 2 most changed)
            diffs = np.abs(np.diff(rho_hist, axis=0))
            if diffs.shape[0] > 0:
                total_change = np.sum(diffs, axis=0)
                top_2_indices = np.argsort(-total_change)[:min(2, len(total_change))]
                
                median_rho = np.median(rho_hist, axis=1)
                ax1.semilogy(iterations, median_rho, marker='o', linewidth=2, color='black', label='median rho_vec', zorder=10)
                
                colors = ['#ff7f0e', '#2ca02c']
                for idx, coord_idx in enumerate(top_2_indices):
                    coord_rho = rho_hist[:, coord_idx]
                    ax1.semilogy(iterations, coord_rho, marker='s', linewidth=1.2, alpha=0.7,
                                color=colors[idx % len(colors)], label=f'rho_vec[{coord_idx}]')
                
                ax1.set_ylabel("rho value (log scale)", fontsize=10)
                ax1.set_title(f"Curvature Evolution - Per-Coordinate Mode (n={rho_hist.shape[1]}, showing median + 2 most-changed)", 
                             fontsize=12, fontweight='bold')
                ax1.grid(True, linestyle='--', linewidth=0.4)
                ax1.legend()
        
        # --- Subplot 2: Update Statistics ---
        update_labels = ['multiplier\non rejection', 'adam_violation', 'adam_secant']
        update_counts = [
            curvature_updates.get("multiplier_on_rejection", 0),
            curvature_updates.get("adam_violation_updates", 0),
            curvature_updates.get("adam_secant_updates", 0)
        ]
        total_updates = curvature_updates.get("total_updates", sum(update_counts))
        
        colors_bar = ['#d62728', '#9467bd', '#8c564b']
        bars = ax2.bar(update_labels, update_counts, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, count in zip(bars, update_counts):
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2.set_ylabel("Count", fontsize=10)
        ax2.set_title(f"Curvature Update Counts (Total: {total_updates})", fontsize=12, fontweight='bold')
        ax2.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.5)
        
        # --- Subplot 3: Violation History ---
        ax3.plot(iterations, violation_hist, marker='o', linewidth=1.5, color='#e74c3c', label='Constraint Violation')
        ax3.set_xlabel("Outer Iteration", fontsize=10)
        ax3.set_ylabel("Violation", fontsize=10, color='#e74c3c')
        ax3.tick_params(axis='y', labelcolor='#e74c3c')
        ax3.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
        ax3.set_title("Constraint Violation History", fontsize=12, fontweight='bold')
        ax3.legend()
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        # Return diagnostic summary
        summary = {
            "total_outer_iterations": len(violation_hist),
            "update_counts": {
                "multiplier_on_rejection": update_counts[0],
                "adam_violation": update_counts[1],
                "adam_secant": update_counts[2],
                "total": total_updates
            },
            "rho_stats": {
                "is_per_coordinate": is_per_coord,
                "final_rho": float(self.rho),
                "rho_min": float(np.min(rho_hist)) if rho_hist.size > 0 else None,
                "rho_max": float(np.max(rho_hist)) if rho_hist.size > 0 else None,
                "rho_mean": float(np.mean(rho_hist)) if rho_hist.size > 0 else None
            },
            "violation_stats": {
                "final_violation": float(violation_hist[-1]) if len(violation_hist) > 0 else None,
                "max_violation": float(np.max(violation_hist)) if len(violation_hist) > 0 else None,
                "avg_violation": float(np.mean(violation_hist)) if len(violation_hist) > 0 else None
            },
            "acceptance_stats": acceptance_stats
        }
        
        return fig, summary

