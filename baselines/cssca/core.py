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
        self.samples_per_iter = int(samples_per_iter)
        self.tau_obj = float(tau_obj)
        self.tau_cons = float(tau_cons)
        self.t = 0

        self.count = 0
        self.count_infeas = 0

        # constraints size
        if g is None:
            self.m = 0
        else:
            g0 = self.g(self.x_k)
            self.m = int(np.atleast_1d(g0).size)
        
        if bounds is not None:
            self.bounds = bounds
        else:
            B = 400.0
            lb = -B * np.ones(self.n, dtype=float)
            ub =  B * np.ones(self.n, dtype=float)
            self.bounds = (lb, ub)


        # surrogate manager
        cfg = surrogate_cfg if surrogate_cfg is not None else RecursiveSurrogateConfig(tau_obj=tau_obj, tau_cons=tau_cons)
        self.surrogates = RecursiveSurrogates(n=self.n, m=self.m, cfg=cfg)

        # step schedules
        self.rho_t_schedule = rho_t_schedule
        # gamma schedule (paper-consistent)
        if gamma_t_schedule is None:
            # γ_t = 1 / (t + 10)
            self.gamma_t_schedule = lambda t: 1.0 / (t + 10.0)
        elif callable(gamma_t_schedule):
            self.gamma_t_schedule = gamma_t_schedule
        else:
            const_val = float(gamma_t_schedule)
            self.gamma_t_schedule = lambda t, v=const_val: v


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
        
        self.count += 1

        # 1) build/update surrogates using one or multiple samples
        rho_t = self._rho_t()
        # for multiple samples we average the sample surrogates by updating sequentially
        for _ in range(self.samples_per_iter):

            # draw ONE sample
            xi = sample_drawer()

            # --- OBJECTIVE WRAPPERS (use same xi everywhere) ---

            def g0_fun(x, xi_local):
                try:
                    return float(self.fun(x, xi_local))
                except TypeError:
                    try:
                        return float(self.fun(x, xi=xi_local))
                    except TypeError:
                        return float(self.fun(x))

            def dg0_fun(x, xi_local):
                if self.df is not None:
                    try:
                        return np.asarray(self.df(x, xi_local), dtype=float).ravel()
                    except TypeError:
                        return np.asarray(self.df(x), dtype=float).ravel()

                # try oracle returning (val, grad)
                try:
                    val, grad = self.fun(x, xi_local, True)
                    return np.asarray(grad, dtype=float).ravel()
                except TypeError:
                    try:
                        val, grad = self.fun(x, xi_local, grad=True)
                        return np.asarray(grad, dtype=float).ravel()
                    except TypeError:
                        _, grad = self._call_fun(x, grad=True)
                        return np.asarray(grad, dtype=float).ravel()

            # --- CONSTRAINT WRAPPERS ---

            g_cons_funs = []
            dg_cons_funs = []

            for i in range(self.m):

                def make_gi(i_local):
                    def gi(x, xi_l):
                        try:
                            return float(np.atleast_1d(self.g(x, xi_l))[i_local])
                        except TypeError:
                            return float(np.atleast_1d(self.g(x))[i_local])
                    return gi

                def make_dgi(i_local):
                    if self.dg is None:
                        return None
                    else:
                        def dgi(x, xi_l):
                            try:
                                return np.atleast_2d(self.dg(x, xi_l))[i_local, :].ravel()
                            except TypeError:
                                return np.atleast_2d(self.dg(x))[i_local, :].ravel()
                        return dgi

                g_cons_funs.append(make_gi(i))
                dg_cons_funs.append(make_dgi(i))

            # --- Update surrogate using SAME xi everywhere ---
            self.surrogates.update_from_sample(
                self.x_k,
                xi,
                g_obj_fun=g0_fun,
                dg_obj_fun=dg0_fun,
                g_cons_funs=g_cons_funs,
                dg_cons_funs=dg_cons_funs,
                rho_t=rho_t,
                tau_obj=self.tau_obj,
                tau_cons=self.tau_cons
            )


            # PRINT ACTUAL TAU'S   
            #print("taus objective:", self.surrogates.taus[0])
            #print("taus constraints:", self.surrogates.taus[1:]

        # 2) Solve convex subproblem (minimize surrogate objective subject to surrogate constraints)
        x0 = self.x_k.copy()
        x_bar, feasible, info = solve_convex_subproblem_quad(self.surrogates, x0, bounds=self.bounds, **inner_solver_opts)
    

        ##### DEBUG
        # --- SURROGATE vs TRUE at x_bar ---
        fbar_xbar, _ = self.surrogates.eval_surrogate(x_bar)
        gbar_xbar, _ = self.surrogates.eval_constraints_surrogates(x_bar)

        f_true_bar = self._call_fun(x_bar)
        g_true_bar = self._call_g(x_bar)

        #print("---- x_bar diagnostics ----")
        #print("||x_bar - x_k||:", np.linalg.norm(x_bar - self.x_k))
        #print("x_bar norm:", np.linalg.norm(x_bar))
        #print("max |x_bar|:", np.max(np.abs(x_bar)))

        #print("SURROGATE: fbar =", fbar_xbar,
        #    "max gbar =", np.max(gbar_xbar))

        #print("TRUE:      f =", f_true_bar,
        #    "max g =", np.max(g_true_bar))
        #print("----------------------------")


        # if infeasible, solve feasibility subproblem:
        if not feasible:

            self.count_infeas += 1
            #print(self.count_infeas * 100.0 / self.count, "Percentage of")

            # minimize alpha s.t. fbar_i(x) <= alpha for i=1..m
            def feasibility_solver():
                n = self.n

                # objective: minimize alpha
                def obj_z(z):
                    return float(z[n])

                def grad_obj_z(z):
                    grad = np.zeros_like(z, dtype=float)
                    grad[n] = 1.0
                    return grad

                # constraints: alpha - fbar_i(x) >= 0
                def make_cons_i(i):
                    def cfun(z, idx=i):
                        x = z[:n]
                        alpha = z[n]
                        vals, _ = self.surrogates.eval_constraints_surrogates(x)
                        return float(alpha - vals[idx])

                    def cjac(z, idx=i):
                        x = z[:n]
                        vals, grads = self.surrogates.eval_constraints_surrogates(x)
                        jac = np.zeros(n + 1, dtype=float)
                        jac[:n] = -grads[idx, :]
                        jac[n] = 1.0
                        return jac

                    return {'type': 'ineq', 'fun': cfun, 'jac': cjac}

                cons = [make_cons_i(i) for i in range(self.m)]

                # box bounds on x
                lb, ub = self.bounds
                bnds = []
                for i in range(n):
                    bnds.append((float(lb[i]), float(ub[i])))

                # bound alpha (large but finite)
                bnds.append((-1e6, 1e6))

                # initial guess
                z0 = np.concatenate([self.x_k.copy(), np.array([0.0], dtype=float)])
                try:
                    vals_k, _ = self.surrogates.eval_constraints_surrogates(self.x_k)
                    if vals_k.size > 0:
                        z0[n] = float(np.max(vals_k))
                except Exception:
                    z0[n] = 0.0

                from scipy.optimize import minimize
                res = minimize(
                    obj_z,
                    z0,
                    jac=grad_obj_z,
                    method='SLSQP',
                    bounds=bnds,
                    constraints=cons,
                    options={'maxiter': 500, 'ftol': 1e-9}
                )

                x_sol = res.x[:n].copy()
                success = bool(res.success)
                return x_sol, success

            x_bar, feasible_flag = feasibility_solver()
            #print("Feasibility solver success:", feasible_flag)

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
