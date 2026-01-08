import numpy as np
from typing import Optional, Tuple
from .params import MMA_SigmaParams

class AsymptoteUpdater:
    def __init__(self,
                 sigma_params: Optional[MMA_SigmaParams] = None,
                 lower_bound: Optional[float] = None,
                 upper_bound: Optional[float] = None):

        self.sigma_params = sigma_params if sigma_params is not None else MMA_SigmaParams()
        self.expand = float(self.sigma_params.expand)
        self.contract = float(self.sigma_params.contract)
        self.sigma_min = float(self.sigma_params.sigma_min)
        self.rel_min = float(self.sigma_params.rel_min)
        self.rel_max = float(self.sigma_params.rel_max)

        # bounds
        if lower_bound is None:
            self.lower_bound = None
        else:
            self.lower_bound = np.asarray(lower_bound, dtype=float)

        if upper_bound is None:
            self.upper_bound = None
        else:
            self.upper_bound = np.asarray(upper_bound, dtype=float)

        self.sigma = None
        self.x_km1 = None
        self.x_km2 = None

    def init_asymptotes(self, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(x0.size)
        lb_arr = self.lower_bound if self.lower_bound is not None else -np.inf * np.ones(n, dtype=float)
        ub_arr = self.upper_bound if self.upper_bound is not None else np.inf * np.ones(n, dtype=float)
        lb_arr = np.asarray(lb_arr, dtype=float).ravel()
        ub_arr = np.asarray(ub_arr, dtype=float).ravel()

        self.sigma = np.empty(n, dtype=float)
        for j in range(n):
            if np.isinf(lb_arr[j]) or np.isinf(ub_arr[j]):
                s0 = 1.0
            else:
                s0 = 0.5 * (ub_arr[j] - lb_arr[j])
            s0 = max(s0, self.sigma_min)
            self.sigma[j] = s0

        L = x0 - self.sigma
        U = x0 + self.sigma
        if self.lower_bound is not None:
            L = np.maximum(L, lb_arr)
        if self.upper_bound is not None:
            U = np.minimum(U, ub_arr)

        self.x_km2 = x0.copy()
        self.x_km1 = x0.copy()
        return L, U

    def update(self, x_km1: np.ndarray, x_kp1: np.ndarray, L: np.ndarray, U: np.ndarray):
        n = x_km1.size
        x_k = x_kp1
        x_km1_local = x_km1
        x_km2_local = self.x_km2 if self.x_km2 is not None else x_km1_local

        self.x_km2 = x_km1.copy()
        self.x_km1 = x_kp1.copy()

        for j in range(n):
            diff1 = x_k[j] - x_km1_local[j]
            diff2 = x_km1_local[j] - x_km2_local[j]
            prod = diff1 * diff2
            if prod > 0.0:
                self.sigma[j] = self.sigma[j] * self.expand
            elif prod < 0.0:
                self.sigma[j] = max(self.sigma[j] * self.contract, self.sigma_min)

            lbj = self.lower_bound[j] if self.lower_bound is not None else -np.inf
            ubj = self.upper_bound[j] if self.upper_bound is not None else np.inf
            if not np.isinf(lbj) and not np.isinf(ubj):
                width = ubj - lbj
                self.sigma[j] = min(self.sigma[j], self.rel_max * width)
                self.sigma[j] = max(self.sigma[j], self.rel_min * width)

            self.sigma[j] = max(self.sigma[j], self.sigma_min)

        L_new = x_k - self.sigma
        U_new = x_k + self.sigma
        if self.lower_bound is not None:
            L_new = np.maximum(L_new, self.lower_bound)
        if self.upper_bound is not None:
            U_new = np.minimum(U_new, self.upper_bound)

        widths = U_new - L_new
        min_width = self.sigma_min
        for j in range(n):
            if widths[j] < min_width:
                center = 0.5 * (U_new[j] + L_new[j])
                L_new[j] = center - 0.5 * min_width
                U_new[j] = center + 0.5 * min_width
                self.sigma[j] = 0.5 * min_width

        return L_new, U_new
