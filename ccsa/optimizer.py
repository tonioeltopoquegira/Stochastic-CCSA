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
                 update_rule_kwargs: Optional[dict] = None):
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
        self.rho = float(rho_init)      # scalar rho (objective curvature)
        self.conservative = conservative

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
            "rho_vec_history": [],
            "violation_history": []
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
            self._curv_state = init_adam_curv_state()
        else:
            self._adam_curv_params = None
            self._curv_state = None

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

        # ---- Inner loop: solve subproblem(s) ----
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

            # ---- Acceptance criteria ----
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
                        rho = update_rho(rho, violation_f, w_val, self.rho_params)
                        self.metrics["curvature_updates"]["multiplier_on_rejection"] += 1
                        self.metrics["curvature_updates"]["total_updates"] += 1
                    if m > 0:
                        mask = violation_gc > 0.0
                        if np.any(mask):
                            self.rho_c[mask] = update_rho(
                                self.rho_c[mask], violation_gc[mask], w_val, self.rho_params
                            )
                wval_used = w_val


        # ---- Outer updates (asymptotes) ----
        L_new, U_new = self.asym.update(x_km1=x_k, x_kp1=x_best, L=self.L.copy(), U=self.U.copy())

        # enforce minimum rho - ONLY for multiplier rule (decay not applied for adam_violation/adam_secant)
        if self.update_rule == 'multiplier':
            rho = max(self.rho_params.decay * rho, self.MMA_RHOMIN)
            if m > 0:
                self.rho_c = np.maximum(self.rho_params.decay * self.rho_c, self.MMA_RHOMIN)


        # ---- Write back state ----
        self.x_k = x_best.copy()
        self.L = L_new
        self.U = U_new
        self.rho = float(rho)

        # objective curvature adaptive update
        if self.update_rule in ('adam_violation', 'adam_secant') and self._curv_state is not None:
            try:
                if self.update_rule == 'adam_violation':
                    new_rho, self._curv_state = adam_curv_update(
                        self._curv_state, violation_f, self.rho, self._adam_curv_params
                    )
                    self.metrics["curvature_updates"]["adam_violation_updates"] += 1
                    self.metrics["curvature_updates"]["total_updates"] += 1
                else:  # adam_secant
                    if self.grad_f_k is not None:
                        new_rho, self._curv_state = adam_secant_update(
                            self._curv_state,
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
            except Exception:
                pass

        # per-constraint curvature adaptive update
        if self.update_rule in ('adam_violation', 'adam_secant') and m > 0 and getattr(self, '_curv_state_c', None) is not None:
            try:
                if len(self._curv_state_c) != self.rho_c.size:
                    self._curv_state_c = [init_adam_curv_state() for _ in range(self.rho_c.size)]
                for i in range(self.rho_c.size):
                    if self.update_rule == 'adam_violation':
                        signal = float(violation_gc[i]) if i < violation_gc.size else 0.0
                        new_rc, self._curv_state_c[i] = adam_curv_update(
                            self._curv_state_c[i], signal, float(self.rho_c[i]), self._adam_curv_params
                        )
                    else:  # adam_secant — use per-constraint gradient rows
                        if self.grad_g_k is not None and self.dg is not None:
                            new_rc, self._curv_state_c[i] = adam_secant_update(
                                self._curv_state_c[i],
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




    