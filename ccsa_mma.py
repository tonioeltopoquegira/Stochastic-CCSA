from typing import Callable, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

# -------------------------
# Asymptote updater (3-point Svanberg rule)
# -------------------------
class AsymptoteUpdater:
    def __init__(self,
                 expand: float = 1.2,
                 contract: float = 0.7,
                 sigma_min: float = 1e-6,
                 sigma_max: float = 1e20,
                 lower_bound: Optional[float] = None,
                 upper_bound: Optional[float] = None):
        
        self.expand = float(expand)
        self.contract = float(contract)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        if lower_bound is None:
            self.lower_bound = None
        else:
            self.lower_bound = np.asarray(lower_bound, dtype=float)

        if upper_bound is None:
            self.upper_bound = None
        else:
            self.upper_bound = np.asarray(upper_bound, dtype=float)


        self.sigma = None
        self._prev_x = None
        self._prev_prev_x = None

    def init_asymptotes(self, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = x0.size
        

        if self.lower_bound is not None and self.upper_bound is not None:
            sigma = 0.5 * (self.upper_bound - self.lower_bound)
        else:
            sigma = np.maximum(np.ones_like(x0), self.sigma_min)

        sigma = np.minimum(sigma, self.sigma_max)
        self.sigma = sigma.copy()
        L = x0 - self.sigma
        U = x0 + self.sigma
        if self.lower_bound is not None:
            L = np.maximum(L, self.lower_bound)
        if self.upper_bound is not None:
            U = np.minimum(U, self.upper_bound)
        self._prev_prev_x = x0.copy()
        self._prev_x = x0.copy()
        return L, U

    def update(self, x_old: np.ndarray, x_new: np.ndarray, L: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = x_old.size
        x_k = x_new
        x_km1 = x_old
        x_km2 = self._prev_prev_x if self._prev_prev_x is not None else x_km1

        for j in range(n):
            diff1 = x_k[j] - x_km1[j]
            diff2 = x_km1[j] - x_km2[j]
            prod = diff1 * diff2
            if prod > 0.0:
                self.sigma[j] = min(self.sigma[j] * self.expand, self.sigma_max)
            elif prod < 0.0:
                self.sigma[j] = max(self.sigma[j] * self.contract, self.sigma_min)

        L_new = x_k - self.sigma
        U_new = x_k + self.sigma

        if self.lower_bound is not None:
            L_new = np.maximum(L_new, self.lower_bound)
        if self.upper_bound is not None:
            U_new = np.minimum(U_new, self.upper_bound)

        # In principle this is already ensured but for numeric safety:
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


# Dual subproblem builder
class DualSubproblemBuilder:
    """
    Build closures that:
      - compute the dual objective and gradient for given y
      - reconstruct primal xcur, dd_gval, dd_gcval, dd_wval from y
    This mirrors the dual_func / reconstruction in the C MMA code.
    """

    def __init__(self,
                 fval: float,
                 df: np.ndarray,
                 x: np.ndarray,
                 gval: np.ndarray,
                 dg: np.ndarray,
                 lb: np.ndarray,
                 ub: np.ndarray,
                 sigma: np.ndarray,
                 rho: float,
                 rhoc: np.ndarray):
        """
        All inputs are numpy arrays of appropriate shapes.
        - fval: scalar current f(x)
        - df: shape (n,)
        - x: shape (n,)
        - gval: shape (m,) or empty
        - dg: shape (m, n) or zeros
        - lb, ub: global bounds arrays shape (n,)
        - sigma: per-coordinate sigma (n,)
        - rho: scalar
        - rhoc: per-constraint rhoc (m,)
        """
        self.fval = float(fval)
        self.df = np.asarray(df, dtype=float).ravel()
        self.x = np.asarray(x, dtype=float).ravel()
        self.gval = np.asarray(gval, dtype=float).ravel() if gval is not None else np.zeros(0, dtype=float)
        self.dg = np.atleast_2d(dg) if dg is not None else np.zeros((0, self.x.size), dtype=float)
        self.lb = np.asarray(lb, dtype=float).ravel()
        self.ub = np.asarray(ub, dtype=float).ravel()
        self.sigma = np.asarray(sigma, dtype=float).ravel()
        self.rho = float(rho)
        self.rhoc = np.asarray(rhoc, dtype=float).ravel() if rhoc is not None else np.zeros(self.gval.size, dtype=float)

        self.n = self.x.size
        self.m = self.gval.size

    def reconstruct_xcur_from_y(self, y: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, float, float]:
        """
        Given y (m,), reconstruct xcur and compute:
          - dd_gval  (d->gval in C)
          - dd_gcval (gcval approximants, shape m)
          - dd_wval  (wval)
          - val_extra (sum_j (u*dx + v*dx^2)/(sigma^2 - dx^2))
        Return: xcur, dd_gval, dd_gcval, dd_wval, val_extra
        """
        y = np.asarray(y, dtype=float).ravel() if self.m > 0 else np.zeros(0, dtype=float)
        xcur = np.empty(self.n, dtype=float)
        dd_gval = float(self.fval)
        if self.m > 0:
            # initialize with current gval, but use 0 where gval is NaN (same behaviour as C)
            dd_gcval = np.where(np.isnan(self.gval), 0.0, self.gval).astype(float).copy()
        else:
            dd_gcval = np.zeros(0, dtype=float)
        dd_wval = 0.0
        val_extra = 0.0   # accumulate per-variable contribution to the dual objective

        dfcdx = self.dg if self.m > 0 else np.zeros((0, self.n), dtype=float)
        mask = ~np.isnan(self.gval) if self.m > 0 else np.zeros(0, dtype=bool)

        for j in range(self.n):
            sj = self.sigma[j]
            if sj == 0.0:
                xcur[j] = self.x[j]
                continue

            # Compute u, v (u is df/dx_j, plus constraint contributions)
            u = self.df[j]
            v = abs(self.df[j]) * sj + 0.5 * self.rho
            if self.m > 0 and mask.any():
                u += np.dot(dfcdx[mask, j], y[mask])
                v += np.dot((np.abs(dfcdx[mask, j]) * sj + 0.5 * self.rhoc[mask]), y[mask])

            sigma2_j = sj * sj

            # Follow the C code stable formula exactly:
            # C multiplies u by sigma^2 first (u_scaled), then uses the stable root form.
            u_scaled = u * sigma2_j
            if v == 0.0 or sj == 0.0:
                dx = 0.0
            else:
                # denom_term = v * sj (nonzero here)
                inner = 1.0 - (u_scaled / (v * sj)) ** 2  # equals 1 - (u * sigma / v)^2 in algebraic terms
                if inner < 0.0:
                    inner = 0.0
                sqrt_inner = np.sqrt(inner)
                denom_stable = -1.0 - sqrt_inner
                if denom_stable == 0.0:
                    dx = 0.0
                else:
                    dx = (u_scaled / v) / denom_stable

            xj = self.x[j] + dx

            # apply global bounds and asymptote clipping (±0.9*sigma)
            if xj > self.ub[j]:
                xj = self.ub[j]
            elif xj < self.lb[j]:
                xj = self.lb[j]

            high = self.x[j] + 0.9 * sj
            low = self.x[j] - 0.9 * sj
            if xj > high:
                xj = high
            elif xj < low:
                xj = low

            xcur[j] = xj

            # Variables for the dual solve: update dd_gval, dd_gcval and dd_wval
            dxj = xcur[j] - self.x[j]
            dx2 = dxj * dxj
            denomv = sigma2_j - dx2
            if denomv <= 1e-30:
                denomv = 1e-30
            denominv = 1.0 / denomv

            # c = sigma^2 * dx
            c = sigma2_j * dxj

            # dd_gval: corresponds to d->gval accumulation in C
            dd_gval += (self.df[j] * c + (abs(self.df[j]) * sj + 0.5 * self.rho) * dx2) * denominv

            if self.m > 0:
                # apply the same mask when updating dd_gcval
                if mask.any():
                    dd_gcval[mask] += (dfcdx[mask, j] * c + (np.abs(dfcdx[mask, j]) * sj + 0.5 * self.rhoc[mask]) * dx2) * denominv

            # dd_wval: wval accumulation
            dd_wval += 0.5 * dx2 * denominv

            # val_extra: the per-variable part of the dual objective (u_scaled used here)
            val_extra += (u_scaled * dx + v * dx2) * denominv

        dd_wval = float(max(dd_wval, 1e-12))
        return xcur, dd_gval, dd_gcval, dd_wval, val_extra


    def build_dual_objective(self):
        """
        Returns (obj_only, obj_with_grad) closures suitable for scipy minimize with jac=True.
        obj_with_grad(y) -> (value, grad)
        """

        def obj_and_grad(y):
            # compute xcur and approximant values using reconstruction routine
            xcur, dd_gval, dd_gcval, dd_wval, val_extra = self.reconstruct_xcur_from_y(y)
            # dual objective value (C's 'val') = dd_gval + sum_i y_i*fc_i + val_extra,
            # because dd_gval is d->gval and val_extra holds the per-variable (u*dx+v*dx2)/denom contributions.
            val = dd_gval + val_extra
            if self.m > 0:
                val += float(np.dot(y, self.gval))
                grad = -dd_gcval
            else:
                grad = np.zeros(0, dtype=float)

            # return objective to *minimize* : negative of dual 
            return -float(val), grad

        def obj_only(y):
            v, _ = obj_and_grad(y)
            return v

        def obj_with_grad(y):
            return obj_and_grad(y)

        return obj_only, obj_with_grad


# -------------------------
# Modular MMA optimizer (flat parameter vector)
# -------------------------
class MMAOptimizer:
    MMA_RHOMIN = 1e-5

    def __init__(self,
                 params: np.ndarray,
                 fun: Callable,            # fun(x, grad=True) -> (f, df) or fun(x) -> f
                 g: Optional[Callable] = None,
                 bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 rho_init: float = 1.0,
                 max_inner: int = 5,
                 sigma_min: float = 1e-6,
                 sigma_max: float = 1e20,
                 expand: float = 1.2,
                 contract: float = 0.7,
                 df: Optional[Callable] = None,
                 dg: Optional[Callable] = None,
                 x0: Optional[np.ndarray] = None):
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
        self.rho = float(rho_init)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.expand = float(expand)
        self.contract = float(contract)

        # initialize x
        if x0 is not None:
            x0_arr = np.asarray(x0, dtype=float).ravel()
        else:
            x0_arr = np.asarray(params, dtype=float).ravel()
        self.x = x0_arr.copy()
        n = self.x.size

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

        # asymptotes
        self.asym = AsymptoteUpdater(expand=self.expand, contract=self.contract,
                                     sigma_min=self.sigma_min, sigma_max=self.sigma_max)
        L, U = self.asym.init_asymptotes(self.x)
        self.L = L
        self.U = U

        # rhoc
        self.rhoc = None

        # bookkeeping
        self.metrics = {
            "weighted_evals": 0,
            "sigma_adjustments": 0,
            "bound_hits": 0,
            "cumulative_wval": 0.0,
            "acceptance_stats": {
                "feasible_accept": 0,
                "infeasible_accept": 0,
                "improve_only": 0,
                "reject": 0
            }
        }
        self.history = {
            "x": [x0.copy()],
            "weighted_evals": [1],
            "sigma": [],    # new: store sigma evolution
            "rho": [],      # new: store scalar rho
            "rhoc": []      # new: store per-constraint rhoc vector
        }


    def step(self):
        x = self.x.copy()
        L = self.L.copy()
        U = self.U.copy()
        rho = float(self.rho)

        # Evaluate f and gradient
        if self.df is None:
            fval, df = self.fun(x, grad=True)
        else:
            fval = float(self.fun(x))
            df = np.asarray(self.df(x), dtype=float).ravel()

        # Evaluate constraints
        if self.g is not None:
            gval = np.atleast_1d(self.g(x)).astype(float)
            m = gval.size
            if self.dg is not None:
                dg = np.atleast_2d(self.dg(x)).astype(float)
            else:
                dg = np.zeros((m, x.size), dtype=float)
            if self.rhoc is None:
                self.rhoc = np.full(m, rho)
        else:
            gval = np.zeros(0, dtype=float)
            dg = np.zeros((0, x.size), dtype=float)
            m = 0
            if self.rhoc is None:
                self.rhoc = np.zeros(0, dtype=float)

        f_best = float(fval)
        x_best = x.copy()
        g_best = gval.copy()
        wval_used = 0.0
        accept_type = "reject"

        # Initialize metric fields if first run
        if "acceptance_stats" not in self.metrics:
            self.metrics["acceptance_stats"] = {
                "feasible_accept": 0,
                "infeasible_accept": 0,
                "improve_only": 0,
                "reject": 0,
            }
        if "sigma_history" not in self.metrics:
            self.metrics["sigma_history"] = []
        if "rho_history" not in self.metrics:
            self.metrics["rho_history"] = []
        if "rhoc_history" not in self.metrics:
            self.metrics["rhoc_history"] = []
        if "x_history" not in self.metrics:
            self.metrics["x_history"] = [self.x.copy()]
        if "cumulative_weighted_evals_history" not in self.metrics:
            self.metrics["cumulative_weighted_evals_history"] = [1]

        # ---- inner loop: solve subproblem(s) ----
        for inner in range(self.max_inner):
            
            sigma_vec = self.asym.sigma.copy()

            builder = DualSubproblemBuilder(
                fval=fval, df=df, x=x, gval=gval, dg=dg,
                lb=self.lb, ub=self.ub, sigma=sigma_vec, rho=rho, rhoc=self.rhoc
            )

            if m > 0:
                obj_only, obj_with_grad = builder.build_dual_objective()
                y0 = np.zeros(m, dtype=float)

                feasible_start = np.all((gval <= 0) | np.isnan(gval))
                dual_bounds = [(0.0, None) for _ in range(m)]
                if not feasible_start:
                    dual_bounds = [(0.0, 1e40) for _ in range(m)]

                res = minimize(lambda yy: obj_with_grad(yy),
                               y0,
                               method='L-BFGS-B',
                               jac=True,
                               bounds=dual_bounds,
                               options={'maxiter': 200, 'ftol': 1e-9})

                y_opt = res.x
                #self.metrics["subproblem_iterations"].append(getattr(res, "nit", 0) + 1)
                xcur, dd_gval, dd_gcval, dd_wval, val_extra = builder.reconstruct_xcur_from_y(y_opt)

                '''# --- DUAL DIAGNOSTICS ---
                print("\n--- DUAL DIAGNOSTICS ---")
                print(f"inner = {inner}")
                print(f"rho = {rho:.3e}")
                print(f"dd_gval = {dd_gval:.4e}, dd_wval = {dd_wval:.4e}")
                print(f"y_opt norm = {np.linalg.norm(y_opt):.4e}, res.success = {res.success}, nit = {res.nit}")
                print(f"fval = {fval:.4e}")
                print(f"gval (first 5) = {gval[:5]}")
                print(f"dd_gcval (first 5) = {dd_gcval[:5]}")
                if self.g is not None:
                    fcval_cur = np.atleast_1d(self.g(xcur)).astype(float)
                    print(f"fcval_cur (first 5) = {fcval_cur[:5]}")
                    print(f"max|fcval_cur - dd_gcval| = {np.max(np.abs(fcval_cur - dd_gcval)):.4e}")
                print(f"xcur[:5] = {xcur[:5]}")
                print(f"sigma[:5] = {self.asym.sigma[:5]}")
                print("--------------------------\n")'''


                if self.df is None:
                    fcur, dfcur = self.fun(xcur, grad=True)
                else:
                    fcur = float(self.fun(xcur))
                    dfcur = np.asarray(self.df(xcur), dtype=float).ravel()

                if self.g is not None:
                    fcval_cur = np.atleast_1d(self.g(xcur)).astype(float)
                else:
                    fcval_cur = np.zeros(0, dtype=float)
                

                self.metrics["weighted_evals"] += 0.5

                inner_done = (dd_gval >= fcur)
                feasible_cur = True
                infeas_cur = 0.0
                for i in range(m):
                    if np.isnan(fcval_cur[i]):
                        continue
                    feasible_cur = feasible_cur and (fcval_cur[i] <= 0.0)
                    inner_done = inner_done and (dd_gcval[i] >= fcval_cur[i])
                    infeas_cur = max(infeas_cur, fcval_cur[i])

                # acceptance logic
                accept = False
                if (fcur < f_best) and (inner_done or feasible_cur or (not feasible_start)):
                    accept = True
                    if feasible_cur:
                        accept_type = "feasible_accept"
                    elif infeas_cur < np.max(np.maximum(gval, 0.0)):
                        accept_type = "infeasible_accept"
                    else:
                        accept_type = "improve_only"
                elif (not accept) and (not feasible_start and infeas_cur < np.max(np.maximum(gval, 0.0))):
                    accept = True
                    accept_type = "infeasible_accept"
                else:
                    accept_type = "reject"

                if accept:
                    f_best = float(fcur)
                    x_best = xcur.copy()
                    g_best = fcval_cur.copy()
                    df = dfcur.copy()
                    wval_used = dd_wval
                    break
                else:
                    # update rho and rhoc
                    if fcur > dd_gval:
                        rho_new = min(10.0 * rho, 1.1 * (rho + (fcur - dd_gval) / dd_wval))
                        if rho_new > rho:
                            rho = rho_new
                            self.metrics["sigma_adjustments"] += 1
                    for i in range(m):
                        if fcval_cur[i] > dd_gcval[i]:
                            rhoc_new = min(10.0 * self.rhoc[i], 1.1 * (self.rhoc[i] + (fcval_cur[i] - dd_gcval[i]) / dd_wval))
                            if rhoc_new > self.rhoc[i]:
                                self.rhoc[i] = rhoc_new
                                self.metrics["sigma_adjustments"] += 1
                    wval_used = dd_wval

            else:
                # Unconstrained case
                xcur, dd_gval, dd_gcval, dd_wval, val_extra = builder.reconstruct_xcur_from_y(np.zeros(0, dtype=float))
                if self.df is None:
                    fcur, dfcur = self.fun(xcur, grad=True)
                else:
                    fcur = float(self.fun(xcur))
                    dfcur = np.asarray(self.df(xcur), dtype=float).ravel()
                
                self.metrics["weighted_evals"] += 0.5

                if fcur < f_best - 1e-12:
                    f_best = float(fcur)
                    x_best = xcur.copy()
                    g_best = np.zeros(0, dtype=float)
                    df = dfcur.copy()
                    wval_used = dd_wval
                    accept_type = "improve_only"

                    break
                else:
                    if fcur > dd_gval:
                        rho_new = min(10.0 * rho, 1.1 * (rho + (fcur - dd_gval) / dd_wval))
                        if rho_new > rho:
                            rho = rho_new
                            self.metrics["sigma_adjustments"] += 1
                    wval_used = dd_wval
                    accept_type = "reject"

        # ---- Outer updates ----
        L_new, U_new = self.asym.update(x_old=x, x_new=x_best, L=L.copy(), U=U.copy())
        rho = max(0.1 * rho, self.MMA_RHOMIN)
        if m > 0:
            self.rhoc = np.maximum(0.1 * self.rhoc, self.MMA_RHOMIN)

        # ---- Write back state ----
        self.x = x_best.copy()
        self.L = L_new
        self.U = U_new
        self.rho = float(rho)

        # ---- Record full histories ----
        self.metrics["cumulative_wval"] += float(wval_used)
        self.metrics["acceptance_stats"][accept_type] += 1

        self.metrics["weighted_evals"] += 1.0

        self.metrics["x_history"].append(x_best.copy())
        self.metrics["sigma_history"].append(self.asym.sigma.copy())
        self.metrics["rho_history"].append(self.rho)
        self.metrics["rhoc_history"].append(self.rhoc.copy() if self.rhoc is not None else np.zeros(0))
        self.metrics["cumulative_weighted_evals_history"].append(int(self.metrics["weighted_evals"]))

        return f_best, g_best, dict(self.metrics)




