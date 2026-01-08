# cssca/core.py
import numpy as np
from typing import Callable, Optional, Tuple, List
from .surrogates import RecursiveSurrogates, RecursiveSurrogateConfig
from .solvers import solve_convex_subproblem_quad

class CSSCAOptimizer:
    """
    CSSCA optimizer using recursive surrogate (linearization + quadratic regularizer).
    API mirrors your MMAOptimizer style:
      - fun(x, grad=True) -> (f, df) if grad requested, else fun(x) -> float
      - g(x) -> array of constraints (<= 0)
      - dg(x) -> jacobian (m x n)
    Args:
      params: initial parameter vector (overridden by x0 if provided)
      fun, df: objective and gradient (optional)
      g, dg: constraints and jacobian (optional)
      bounds: (lb, ub) arrays or None (box X)
      rho_t_schedule, gamma_t_schedule: callables t -> rho_t, gamma_t or scalars
      tau_obj, tau_cons: regularization strengths for surrogate sample
      samples_per_iter: number of xi samples used per iteration (set 1 for online)
      surrogate_cfg: instance of RecursiveSurrogateConfig or None
    """
    def __init__(self,
                 params: np.ndarray,
                 fun: Callable,
                 g: Optional[Callable] = None,
                 bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 df: Optional[Callable] = None,
                 dg: Optional[Callable] = None,
                 x0: Optional[np.ndarray] = None,
                 rho_t_schedule = 1.0,   # can be scalar or callable(t)
                 gamma_t_schedule = 1.0, # scalar or callable(t)
                 tau_obj: float = 1.0,
                 tau_cons: float = 1.0,
                 samples_per_iter: int = 1,
                 surrogate_cfg: Optional[RecursiveSurrogateConfig] = None):
        if x0 is not None:
            self.x_k = np.asarray(x0, dtype=float).ravel().copy()
        else:
            self.x_k = np.asarray(params, dtype=float).ravel().copy()
        self.n = self.x_k.size
        self.fun = fun
        self.df = df
        self.g = g
        self.dg = dg
        self.bounds = bounds
        self.samples_per_iter = int(samples_per_iter)
        self.tau_obj = float(tau_obj)
        self.tau_cons = float(tau_cons)
        self.t = 0

        # constraints size
        if g is None:
            self.m = 0
        else:
            g0 = self.g(self.x_k)
            self.m = int(np.atleast_1d(g0).size)

        # surrogate manager
        cfg = surrogate_cfg if surrogate_cfg is not None else RecursiveSurrogateConfig(tau_obj=tau_obj, tau_cons=tau_cons)
        self.surrogates = RecursiveSurrogates(n=self.n, m=self.m, cfg=cfg)

        # step schedules
        self.rho_t_schedule = rho_t_schedule
        self.gamma_t_schedule = gamma_t_schedule

        # history
        self.history = {'x': [self.x_k.copy()], 'f': [], 'cons': []}

    def _call_fun(self, x, grad=False):
        if grad:
            if self.df is None:
                return self.fun(x, grad=True)
            else:
                return float(self.fun(x)), np.asarray(self.df(x), dtype=float).ravel()
        else:
            return float(self.fun(x))

    def _call_g(self, x, jac=False):
        if self.g is None:
            if jac:
                return np.zeros((0, x.size), dtype=float)
            return np.zeros(0, dtype=float)
        if jac:
            if self.dg is None:
                # user didn't provide jacobian -> finite diff
                eps = 1e-8
                gx = np.atleast_1d(self.g(x)).astype(float)
                m = gx.size
                jac_mat = np.zeros((m, x.size), dtype=float)
                for j in range(x.size):
                    ej = np.zeros_like(x); ej[j] = eps
                    gj = np.atleast_1d(self.g(x + ej)).astype(float)
                    jac_mat[:, j] = (gj - gx) / eps
                return jac_mat
            else:
                return np.atleast_2d(self.dg(x)).astype(float)
        else:
            return np.atleast_1d(self.g(x)).astype(float)

    def _rho_t(self):
        return self.rho_t_schedule(self.t) if callable(self.rho_t_schedule) else float(self.rho_t_schedule)

    def _gamma_t(self):
        return self.gamma_t_schedule(self.t) if callable(self.gamma_t_schedule) else float(self.gamma_t_schedule)

    def step(self, sample_drawer: Callable[[], object] = lambda: None, inner_solver_opts: dict = None):
        """
        Perform one CSSCA outer iteration:
          - draw sample(s) xi_t
          - update surrogate(s) fbar^t via recursive mixture
          - try to solve convex subproblem (objective update); if infeasible, solve feasibility problem (minimize alpha)
          - update x_{t+1} = (1 - gamma_t) x_t + gamma_t * x_bar
        sample_drawer returns a sample xi (can be anything passed to g/g functions)
        """
        if inner_solver_opts is None:
            inner_solver_opts = {}

        # 1) build/update surrogates using one or multiple samples
        rho_t = self._rho_t()
        # for multiple samples we average the sample surrogates by updating sequentially
        for s in range(self.samples_per_iter):
            xi = sample_drawer()
            # objective sample function wrapper (fun returns f(x, grad?) but surrogate expects sample g0(x, xi) -> scalar)
            def g0_fun(x, xi_local):
                # user fun should accept grad flag; for sample surrogate we only need scalar
                return float(self.fun(x))
            def dg0_fun(x, xi_local):
                if self.df is not None:
                    return np.asarray(self.df(x), dtype=float).ravel()
                else:
                    # fallback numeric approx via _call_fun
                    _, grad = self._call_fun(x, grad=True)
                    return grad
            g_cons_funs = []
            dg_cons_funs = []
            for i in range(self.m):
                def make_gi(i_local):
                    return lambda x, xi_l: float(np.atleast_1d(self.g(x))[i_local])
                def make_dgi(i_local):
                    if self.dg is None:
                        return None
                    else:
                        return lambda x, xi_l: np.atleast_2d(self.dg(x))[i_local, :].ravel()
                g_cons_funs.append(make_gi(i))
                dg_cons_funs.append(make_dgi(i))

            # now update surrogates (note: we pass xi but surrogate creation ignores it unless user wants)
            self.surrogates.update_from_sample(self.x_k, xi,
                                              g_obj_fun=lambda x, xi_l: float(self.fun(x)),
                                              dg_obj_fun=(lambda x, xi_l: (np.asarray(self.df(x), dtype=float).ravel())) if self.df is not None else None,
                                              g_cons_funs=g_cons_funs,
                                              dg_cons_funs=dg_cons_funs,
                                              rho_t=rho_t,
                                              tau_obj=self.tau_obj,
                                              tau_cons=self.tau_cons)

        # 2) Solve convex subproblem (minimize surrogate objective subject to surrogate constraints)
        x0 = self.x_k.copy()
        x_bar, feasible, info = solve_convex_subproblem_quad(self.surrogates, x0, bounds=self.bounds, **inner_solver_opts)

        # if infeasible, solve feasibility subproblem:
        if not feasible:
            # minimize alpha s.t. fbar_i(x) <= alpha for i=1..m
            # we simply minimize the maximum surrogate constraint using SLSQP by introducing scalar alpha via variable stacking
            # implement by optimizing over z = [x; alpha] with bounds for alpha large
            def feasibility_solver():
                n = self.n
                def obj_z(z):
                    x = z[:n]
                    alpha = z[n]
                    return alpha
                def grad_obj_z(z):
                    g = np.zeros_like(z); g[n] = 1.0; return g
                # constraints: fbar_i(x) - alpha <= 0  -> as SLSQP 'ineq' we do -(fbar_i - alpha) >= 0
                def make_cons_i(i):
                    def cfun(z, idx=i):
                        x = z[:n]
                        alpha = z[n]
                        vals, _ = self.surrogates.eval_constraints_surrogates(x)
                        return -(vals[idx] - alpha)
                    def cjac(z, idx=i):
                        x = z[:n]
                        alpha = z[n]
                        vals, grads = self.surrogates.eval_constraints_surrogates(x)
                        jac = np.zeros(n + 1, dtype=float)
                        jac[:n] = -grads[idx, :]
                        jac[n] = 1.0
                        return jac
                    return {'type': 'ineq', 'fun': cfun, 'jac': cjac}
                cons = [make_cons_i(i) for i in range(self.m)]
                # bounds: keep x within self.bounds, alpha unbounded
                if self.bounds is not None:
                    lb, ub = self.bounds
                    bnds = []
                    for i in range(n):
                        bnds.append((None if np.isneginf(lb[i]) else lb[i],
                                     None if np.isposinf(ub[i]) else ub[i]))
                else:
                    bnds = [(None, None)] * n
                bnds.append((None, None))  # alpha
                z0 = np.concatenate([self.x_k, np.array([1.0], dtype=float)])
                from scipy.optimize import minimize
                res = minimize(lambda z: obj_z(z),
                               z0,
                               jac=lambda z: grad_obj_z(z),
                               method='SLSQP',
                               bounds=bnds,
                               constraints=cons,
                               options={'maxiter': 200, 'ftol': 1e-9})
                return res.x[:n], res.success
            x_bar, feasible_flag = feasibility_solver()
            # we won't raise if feasibility solver fails; we keep x_bar anyway

        # 3) step update xt+1 = (1 - gamma_t) xt + gamma_t * x_bar
        gamma_t = self._gamma_t()
        self.x_k = (1.0 - gamma_t) * self.x_k + gamma_t * x_bar

        # record
        fval, _ = self._call_fun(self.x_k, grad=True)
        gvals = self._call_g(self.x_k, jac=False) if self.m > 0 else np.zeros(0)
        self.history['x'].append(self.x_k.copy())
        self.history['f'].append(fval)
        self.history['cons'].append(gvals.copy())
        self.t += 1

        return self.x_k.copy(), float(fval), gvals.copy()
