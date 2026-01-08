import numpy as np
from typing import Tuple
from .params import update_rho   # only if needed externally, otherwise not used here

class DualSubproblemBuilder:
    """
    Build closures that:
      - compute the dual objective and gradient for given y
      - reconstruct primal x_candidate, tilde_f, tilde_gc, w_val from y
    """

    def __init__(self,
                 f_k: float,
                 grad_f_k: np.ndarray,
                 x_k: np.ndarray,
                 g_k: np.ndarray,
                 grad_g_k: np.ndarray,
                 lb: np.ndarray,
                 ub: np.ndarray,
                 sigma: np.ndarray,
                 rho: float,
                 rho_c: np.ndarray):

        self.f_k = float(f_k)
        self.grad_f_k = np.asarray(grad_f_k, dtype=float).ravel()
        self.x_k = np.asarray(x_k, dtype=float).ravel()
        self.g_k = np.asarray(g_k, dtype=float).ravel() if g_k is not None else np.zeros(0, dtype=float)
        self.grad_g_k = np.atleast_2d(grad_g_k) if grad_g_k is not None else np.zeros((0, self.x_k.size), dtype=float)
        self.lb = np.asarray(lb, dtype=float).ravel()
        self.ub = np.asarray(ub, dtype=float).ravel()
        self.sigma = np.asarray(sigma, dtype=float).ravel()
        self.rho = float(rho)
        self.rho_c = np.asarray(rho_c, dtype=float).ravel() if rho_c is not None else np.zeros(self.g_k.size, dtype=float)

        self.n = self.x_k.size
        self.m = self.g_k.size


    def reconstruct_xcandidate_from_y(self, y: np.ndarray):
        y = np.asarray(y, dtype=float).ravel() if self.m > 0 else np.zeros(0, dtype=float)

        x_candidate = np.empty(self.n, dtype=float)
        tilde_f = float(self.f_k)

        if self.m > 0:
            tilde_gc = np.where(np.isnan(self.g_k), 0.0, self.g_k).astype(float).copy()
        else:
            tilde_gc = np.zeros(0, dtype=float)

        w_val = 0.0
        val_extra = 0.0

        grad_g = self.grad_g_k if self.m > 0 else np.zeros((0, self.n))
        mask = ~np.isnan(self.g_k) if self.m > 0 else np.zeros(0, dtype=bool)

        for j in range(self.n):
            sj = self.sigma[j]
            if sj == 0.0:
                x_candidate[j] = self.x_k[j]
                continue

            u_j = self.grad_f_k[j]
            v_j = abs(self.grad_f_k[j]) * sj + 0.5 * self.rho

            if self.m > 0 and mask.any():
                u_j += np.dot(grad_g[mask, j], y[mask])
                v_j += np.dot((np.abs(grad_g[mask, j]) * sj + 0.5 * self.rho_c[mask]), y[mask])

            sigma2_j = sj * sj

            u_scaled = u_j * sigma2_j
            if v_j == 0.0 or sj == 0.0:
                dx = 0.0
            else:
                inner = 1.0 - (u_scaled / (v_j * sj)) ** 2
                inner = max(inner, 0.0)
                sqrt_inner = np.sqrt(inner)
                denom_stable = -1.0 - sqrt_inner
                dx = 0.0 if denom_stable == 0.0 else (u_scaled / v_j) / denom_stable

            xj = self.x_k[j] + dx

            if xj > self.ub[j]: xj = self.ub[j]
            elif xj < self.lb[j]: xj = self.lb[j]

            high = self.x_k[j] + 0.9 * sj
            low = self.x_k[j] - 0.9 * sj

            if xj > high: xj = high
            elif xj < low: xj = low

            x_candidate[j] = xj

            dxj = xj - self.x_k[j]
            dx2 = dxj * dxj

            denomv = sigma2_j - dx2
            denom_floor = max(1e-30, sigma2_j * 1e-12)
            if denomv <= denom_floor:
                denomv = denom_floor
            denominv = 1.0 / denomv

            c = sigma2_j * dxj

            tilde_f += (self.grad_f_k[j] * c + (abs(self.grad_f_k[j]) * sj + 0.5 * self.rho) * dx2) * denominv

            if self.m > 0 and mask.any():
                tilde_gc[mask] += (grad_g[mask, j] * c +
                                   (np.abs(grad_g[mask, j]) * sj + 0.5 * self.rho_c[mask]) * dx2) * denominv

            w_val += 0.5 * dx2 * denominv

            val_extra += (u_scaled * dx + v_j * dx2) * denominv

        w_floor = max(1e-12, np.mean(self.sigma)**2 * 1e-14)
        w_val = float(max(w_val, w_floor))

        return x_candidate, tilde_f, tilde_gc, w_val, val_extra


    def build_dual_objective(self):
        def obj_and_grad(y):
            x_candidate, tilde_f, tilde_gc, w_val, val_extra = self.reconstruct_xcandidate_from_y(y)
            val = tilde_f + val_extra
            if self.m > 0:
                val += float(np.dot(y, self.g_k))
                grad = -tilde_gc
            else:
                grad = np.zeros(0, dtype=float)

            return -float(val), grad

        def obj_only(y):
            v, _ = obj_and_grad(y)
            return v

        return obj_only, obj_and_grad
