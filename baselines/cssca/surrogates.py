# cssca/surrogates.py
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List

@dataclass
class RecursiveSurrogateConfig:
    tau_obj: float = 1.0      # quadratic regularization for objective surrogate
    tau_cons: float = 1.0     # quadratic regularization for constraints (can be scalar or per-constraint)
    rho: float = 1.0           # sample mixing weight ρ_t (can be reduced externally)
    use_avg_gradients: bool = False  # whether to maintain averaged gradients (structured surrogate variant)

class RecursiveSurrogates:
    """
    Manage recursive surrogate functions fbar_i^t for objective (index 0) and constraints (1..m)
    Each surrogate is represented as a quadratic of the form:
        s(x) = const + linear^T x + tau * ||x - y||^2
    We store for each surrogate:
        - constant term
        - linear term (vector)
        - center y (last linearization point)
        - tau (strong convexity)
    The recursive update is:
        fbar^t = (1 - rho_t) * fbar^{t-1} + rho_t * ghat(x; x_t, xi_t)
    where ghat is linearization + tau ||x - x_t||^2
    """
    def __init__(self, n: int, m: int, cfg: Optional[RecursiveSurrogateConfig] = None):
        self.n = int(n)
        self.m = int(m)  # number of constraints
        self.cfg = cfg if cfg is not None else RecursiveSurrogateConfig()

        # store surrogate parameters: objective first, then constraints
        # consts: length m+1, linears: (m+1, n), centers: (m+1, n), taus: length m+1
        self.consts = np.zeros(self.m + 1, dtype=float)
        self.linears = np.zeros((self.m + 1, self.n), dtype=float)
        self.centers = np.zeros((self.m + 1, self.n), dtype=float)
        self.taus = np.full(self.m + 1, self.cfg.tau_cons, dtype=float)
        # objective tau uses tau_obj
        self.taus[0] = self.cfg.tau_obj

        # initialize centers to zero (will be set on first update)
        self.initialized = False

    def _make_sample_surrogate(self,
                               x_t: np.ndarray,
                               xi: object,
                               gi_fun: Callable[[np.ndarray, object], float],
                               dgi_fun: Optional[Callable[[np.ndarray, object], np.ndarray]],
                               tau: float) -> Tuple[float, np.ndarray, np.ndarray, float]:
        """
        Return sample surrogate coefficients for a single scalar function gi:
           ghat(x) = c0 + linear^T x + tau * ||x - x_t||^2

        returns (const_term, linear_vec, center, tau)
        """
        x_t = np.asarray(x_t, dtype=float).ravel()
        val = float(gi_fun(x_t, xi)) if gi_fun is not None else 0.0
        if dgi_fun is not None:
            grad = np.asarray(dgi_fun(x_t, xi), dtype=float).ravel()
        else:
            # finite-difference fallback (small eps)
            eps = 1e-8
            grad = np.zeros_like(x_t)
            for j in range(len(x_t)):
                ej = np.zeros_like(x_t); ej[j] = eps
                grad[j] = (gi_fun(x_t + ej, xi) - val) / eps

        # We want the sample surrogate
        #   ghat(x) = val + grad^T (x - x_t) + tau * ||x - x_t||^2
        # Represented as: const + linear^T x + tau * ||x - center||^2 with center = x_t
        # Matching terms gives const = val - grad^T x_t, linear = grad, center = x_t
        const_term = val - float(np.dot(grad, x_t))
        linear = grad.copy()
        center = x_t.copy()
        return const_term, linear, center, float(tau)

    def update_from_sample(self,
                           x_t: np.ndarray,
                           xi: object,
                           g_obj_fun: Callable[[np.ndarray, object], float],
                           dg_obj_fun: Optional[Callable[[np.ndarray, object], np.ndarray]],
                           g_cons_funs: List[Callable[[np.ndarray, object], float]],
                           dg_cons_funs: Optional[List[Optional[Callable[[np.ndarray, object], np.ndarray]]]],
                           rho_t: float,
                           tau_obj: Optional[float] = None,
                           tau_cons: Optional[float] = None):
        """
        Update all surrogates with a new sample xi observed at x_t.
        g_obj_fun: g0(x, xi) sample objective
        g_cons_funs: list of m functions gi(x, xi)
        dg_*: optional gradient providers with same calling style as provided by user
        """
        x_t = np.asarray(x_t, dtype=float).ravel()
        if tau_obj is None: tau_obj = self.cfg.tau_obj
        if tau_cons is None: tau_cons = self.cfg.tau_cons

        # objective (index 0)
        const_s, linear_s, center_s, tau_s = self._make_sample_surrogate(
            x_t, xi, g_obj_fun, dg_obj_fun, tau_obj
        )
        # recursive average:
        if not self.initialized:
            self.consts[0] = const_s
            self.linears[0, :] = linear_s
            self.centers[0, :] = center_s
            self.taus[0] = tau_s
        else:
            self.consts[0] = (1.0 - rho_t) * self.consts[0] + rho_t * const_s
            self.linears[0, :] = (1.0 - rho_t) * self.linears[0, :] + rho_t * linear_s
            # center may be kept as latest linearization point (x_t); we store last for gradient evals
            self.centers[0, :] = center_s
            self.taus[0] = (1.0 - rho_t) * self.taus[0] + rho_t * tau_s

        # constraints (1..m)
        for i in range(self.m):
            gi_fun = g_cons_funs[i]
            dgi_fun = dg_cons_funs[i] if dg_cons_funs is not None else None
            const_s, linear_s, center_s, tau_s = self._make_sample_surrogate(x_t, xi, gi_fun, dgi_fun, tau_cons)
            idx = 1 + i
            if not self.initialized:
                self.consts[idx] = const_s
                self.linears[idx, :] = linear_s
                self.centers[idx, :] = center_s
                self.taus[idx] = tau_s
            else:
                self.consts[idx] = (1.0 - rho_t) * self.consts[idx] + rho_t * const_s
                self.linears[idx, :] = (1.0 - rho_t) * self.linears[idx, :] + rho_t * linear_s
                self.centers[idx, :] = center_s
                self.taus[idx] = (1.0 - rho_t) * self.taus[idx] + rho_t * tau_s

        self.initialized = True

    def eval_surrogate(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Evaluate objective surrogate value and gradient at x.
        Returns (fbar0(x), grad_fbar0(x)).
        For constraints, call eval_constraints_surrogates.
        """
        x = np.asarray(x, dtype=float).ravel()
        # objective index 0
        c = self.consts[0]
        lin = self.linears[0, :]
        tau = self.taus[0]
        center = self.centers[0, :]
        # s(x) = const + lin^T x + tau * ||x - center||^2
        dx = x - center
        val = float(c + np.dot(lin, x) + tau * np.dot(dx, dx))
        grad = lin + 2.0 * tau * dx
        return val, grad

    def eval_constraints_surrogates(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate constraint surrogate values and gradients.
        Returns (gbar(x) as vector length m, jac matrix m x n)
        """
        x = np.asarray(x, dtype=float).ravel()
        vals = np.empty(self.m, dtype=float)
        grads = np.empty((self.m, self.n), dtype=float)
        for i in range(self.m):
            idx = 1 + i
            c = self.consts[idx]
            lin = self.linears[idx, :]
            tau = self.taus[idx]
            center = self.centers[idx, :]
            dx = x - center
            vals[i] = float(c + np.dot(lin, x) + tau * np.dot(dx, dx))
            grads[i, :] = lin + 2.0 * tau * dx
        return vals, grads
