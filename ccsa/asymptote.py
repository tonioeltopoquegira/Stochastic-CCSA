from typing import Optional, Tuple
import numpy as np
from ccsa.params import MMA_SigmaParams


# -------------------------
# Asymptote updater (3-point Svanberg rule)
# -------------------------
class AsymptoteUpdater:
    def __init__(self,
                 sigma_params: Optional[MMA_SigmaParams] = None,
                 lower_bound: Optional[float] = None,
                 upper_bound: Optional[float] = None):
        self.sigma_params = sigma_params if sigma_params is not None else MMA_SigmaParams()
        self.expand   = float(self.sigma_params.expand)
        self.contract = float(self.sigma_params.contract)
        self.sigma_min = float(self.sigma_params.sigma_min)
        self.rel_min   = float(self.sigma_params.rel_min)
        self.rel_max   = float(self.sigma_params.rel_max)

        self.lower_bound = None if lower_bound is None else np.asarray(lower_bound, dtype=float)
        self.upper_bound = None if upper_bound is None else np.asarray(upper_bound, dtype=float)

        self.sigma = None
        self.x_km1 = None
        self.x_km2 = None

    def _lb_ub(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        lb = self.lower_bound if self.lower_bound is not None else np.full(n, -np.inf)
        ub = self.upper_bound if self.upper_bound is not None else np.full(n,  np.inf)
        return np.asarray(lb, dtype=float).ravel(), np.asarray(ub, dtype=float).ravel()

    # ------------------------------------------------------------------

    def init_asymptotes(self, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(x0.size)
        lb_arr, ub_arr = self._lb_ub(n)

        # s0 = 0.5*(ub-lb) where both finite, else 1.0; then max with sigma_min
        finite = np.isfinite(lb_arr) & np.isfinite(ub_arr)
        s0 = np.where(finite, 0.5 * (ub_arr - lb_arr), 0.01)
        self.sigma = np.maximum(s0, self.sigma_min)

        L = x0 - self.sigma
        U = x0 + self.sigma
        if self.lower_bound is not None:
            L = np.maximum(L, lb_arr)
        if self.upper_bound is not None:
            U = np.minimum(U, ub_arr)

        self.x_km2 = x0.copy()
        self.x_km1 = x0.copy()
        return L, U

    def update(self, x_km1: np.ndarray, x_kp1: np.ndarray,
               L: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = x_km1.size
        x_k          = x_kp1
        x_km1_local  = x_km1
        x_km2_local  = self.x_km2 if self.x_km2 is not None else x_km1_local

        self.x_km2 = x_km1.copy()
        self.x_km1 = x_kp1.copy()

        lb_arr, ub_arr = self._lb_ub(n)

        # DISABLED: Keep sigma fixed at initialization to test hypothesis
        # Original Svanberg expand / contract logic commented out
        #diff1 = x_k       - x_km1_local    # (n,)
        #diff2 = x_km1_local - x_km2_local  # (n,)
        #prod  = diff1 * diff2               # (n,)
        #
        #sigma = self.sigma.copy()
        #sigma = np.where(prod > 0.0, sigma * self.expand,
        #        np.where(prod < 0.0, np.maximum(sigma * self.contract, self.sigma_min),
        #                 sigma))
        #
        ## --- relative bounds where box is fully finite ---
        #finite = np.isfinite(lb_arr) & np.isfinite(ub_arr)
        #width  = ub_arr - lb_arr                             # (n,)
        #sigma  = np.where(finite, np.clip(sigma, self.rel_min * width, self.rel_max * width), sigma)
        #
        ## --- global sigma_min floor ---
        #sigma = np.maximum(sigma, self.sigma_min)
        #
        #self.sigma = sigma

        # Keep sigma at current value (no updates)
        sigma = self.sigma.copy()

        # --- new asymptotes ---
        L_new = x_k - sigma
        U_new = x_k + sigma
        if self.lower_bound is not None:
            L_new = np.maximum(L_new, lb_arr)
        if self.upper_bound is not None:
            U_new = np.minimum(U_new, ub_arr)

        # --- minimal width safety ---
        widths = U_new - L_new
        narrow = widths < self.sigma_min
        if narrow.any():
            center      = 0.5 * (U_new + L_new)
            half        = 0.5 * self.sigma_min
            L_new       = np.where(narrow, center - half, L_new)
            U_new       = np.where(narrow, center + half, U_new)
            self.sigma  = np.where(narrow, half, self.sigma)

        return L_new, U_new