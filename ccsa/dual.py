import numpy as np
from typing import Tuple
from scipy.linalg import solve as _slv

# Dual subproblem builder
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
                 rho_c: np.ndarray,
                 quadratic: bool = False):
        """
        Inputs (math-friendly names):
        - f_k: scalar current f(x^k)
        - grad_f_k: shape (n,)
        - x_k: shape (n,) (current major iterate)
        - g_k: shape (m,) or empty
        - grad_g_k: shape (m, n) or zeros
        - lb, ub: global bounds arrays shape (n,)
        - sigma: per-coordinate asymptotes sigma_j (n,)
        - rho: scalar curvature parameter (objective)
        - rho_c: per-constraint curvature parameters (m,)
        """
        self.f_k = float(f_k)
        self.grad_f_k = np.asarray(grad_f_k, dtype=float).ravel()
        self.x_k = np.asarray(x_k, dtype=float).ravel()
        self.g_k = np.asarray(g_k, dtype=float).ravel() if g_k is not None else np.zeros(0, dtype=float)
        self.grad_g_k = np.atleast_2d(grad_g_k) if grad_g_k is not None else np.zeros((0, self.x_k.size), dtype=float)
        self.lb = np.asarray(lb, dtype=float).ravel()
        self.ub = np.asarray(ub, dtype=float).ravel()
        self.sigma = np.asarray(sigma, dtype=float).ravel()
        self.n = self.x_k.size
        self.m = self.g_k.size

        rho_arr = np.asarray(rho, dtype=float).ravel()
        if rho_arr.size == 1:
            self.rho = np.full(self.n, float(rho_arr[0]))
        else:
            if rho_arr.size != self.n:
                raise ValueError(f"rho must be scalar or shape (n,), got size {rho_arr.size}")
            self.rho = rho_arr.copy()

        self.rho_c = np.asarray(rho_c, dtype=float).ravel() if rho_c is not None else np.zeros(self.g_k.size, dtype=float)

        # flag: use quadratic CCSA-style surrogates instead of MMA moving-asymptotes
        self.quadratic = bool(quadratic)

        # Precompute mask and filtered arrays once (avoid recomputing per call)
        if self.m > 0:
            self._mask = ~np.isnan(self.g_k)           # (m,)
            self._g_k_clean = np.where(self._mask, self.g_k, 0.0)
            # grad_g rows restricted to active constraints: shape (m_active, n)
            self._grad_g_masked = self.grad_g_k[self._mask]   # (m_act, n)
            self._rho_c_masked  = self.rho_c[self._mask]      # (m_act,)
            self._any_mask      = bool(self._mask.any())
        else:
            self._mask          = np.zeros(0, dtype=bool)
            self._g_k_clean     = np.zeros(0, dtype=float)
            self._grad_g_masked = np.zeros((0, self.n), dtype=float)
            self._rho_c_masked  = np.zeros(0, dtype=float)
            self._any_mask      = False

    def _reconstruct_mma(self, y: np.ndarray):
        """
        Fully-vectorised MMA (moving-asymptotes) reconstruction.
        Returns x_candidate, tilde_f, tilde_gc, w_val (n,), val_extra (scalar).
        """
        sigma  = self.sigma          # (n,)
        sigma2 = sigma * sigma       # (n,)

        # --- build u_j, v_j vectors ---
        # u_j = grad_f[j] + sum_{i in mask} grad_g[i,j] * y[i]
        # v_j = |grad_f[j]|*s + 0.5*rho[j] + sum_{i in mask} (|grad_g[i,j]|*s + 0.5*rho_c[i])*y[i]
        u = self.grad_f_k.copy()                               # (n,)
        v = np.abs(self.grad_f_k) * sigma + 0.5 * self.rho    # (n,)

        if self._any_mask:
            y_act = y[self._mask]                              # (m_act,)
            # u += grad_g_masked.T @ y_act   shape (n,)
            u += self._grad_g_masked.T @ y_act
            # v += (|grad_g_masked|*sigma + 0.5*rho_c_masked).T @ y_act
            v += (np.abs(self._grad_g_masked) * sigma[np.newaxis, :] +
                  0.5 * self._rho_c_masked[:, np.newaxis]).T @ y_act

        # --- compute dx for each coordinate ---
        u_scaled = u * sigma2          # u_j * sigma_j^2

        # inner = 1 - (u_scaled / (v * sigma))^2,  clipped to [0, 1]
        # Guard zero v or zero sigma
        safe_vs = np.where((v == 0.0) | (sigma == 0.0), 1.0, v * sigma)
        ratio   = np.where((v == 0.0) | (sigma == 0.0), 0.0, u_scaled / safe_vs)  # (n,)
        inner   = np.clip(1.0 - ratio**2, 0.0, None)
        sqrt_in = np.sqrt(inner)
        denom   = -1.0 - sqrt_in      # always <= -1, never zero

        # dx = (u_scaled / v) / denom  when v != 0 and sigma != 0
        safe_v = np.where(v == 0.0, 1.0, v)
        dx     = np.where((v == 0.0) | (sigma == 0.0),
                          0.0,
                          (u_scaled / safe_v) / denom)         # (n,)

        x_candidate = self.x_k + dx

        # apply global bounds
        x_candidate = np.clip(x_candidate, self.lb, self.ub)

        # asymptote clipping
        x_candidate = np.clip(x_candidate,
                              self.x_k - 0.9 * sigma,
                              self.x_k + 0.9 * sigma)

        # zero out coordinates where sigma==0
        x_candidate = np.where(sigma == 0.0, self.x_k, x_candidate)

        # --- compute surrogate values ---
        dxj  = x_candidate - self.x_k     # (n,)
        dx2  = dxj * dxj                  # (n,)

        denomv = sigma2 - dx2
        denom_floor = np.maximum(1e-30, sigma2 * 1e-12)
        denomv  = np.maximum(denomv, denom_floor)
        denominv = 1.0 / denomv            # (n,)

        c = sigma2 * dxj                  # (n,)

        # tilde_f contributions per j
        tf_contrib = (self.grad_f_k * c +
                      (np.abs(self.grad_f_k) * sigma + 0.5 * self.rho) * dx2) * denominv
        tilde_f = self.f_k + float(tf_contrib.sum())

        # tilde_gc contributions: shape (m_act, n) x (n,) -> (m_act,)
        tilde_gc = self._g_k_clean.copy()
        if self._any_mask:
            # (m_act, n) broadcast: each row i gets its own rho_c[i]
            tc_contrib = (self._grad_g_masked * c[np.newaxis, :] +
                          (np.abs(self._grad_g_masked) * sigma[np.newaxis, :] +
                           0.5 * self._rho_c_masked[:, np.newaxis]) * dx2[np.newaxis, :]) * denominv[np.newaxis, :]
            tilde_gc[self._mask] += tc_contrib.sum(axis=1)

        w_val_sum = float((0.5 * dx2 * denominv).sum())
        w_val     = np.full(self.n, w_val_sum)
        val_extra = float(((u_scaled * dxj + v * dx2) * denominv).sum())

        return x_candidate, tilde_f, tilde_gc, w_val, val_extra

    def _reconstruct_quadratic(self, y: np.ndarray):
        """
        Fully-vectorised quadratic (CCSA) reconstruction.
        Returns x_candidate, tilde_f, tilde_gc, w_val (n,), val_extra (scalar).
        """
        import os
        debug_mode = os.environ.get('CCSA_DEBUG', '0') == '1'
        
        if debug_mode:
            print(f"        [QUAD_RECON] START")
        
        sigma  = self.sigma
        sigma2 = sigma * sigma

        # u_j = rho[j] + sum_{i in mask} rho_c[i] * y[i]
        # v_j = grad_f[j] + sum_{i in mask} grad_g[i,j] * y[i]
        u = self.rho.copy()             # (n,)
        v = self.grad_f_k.copy()        # (n,)

        if self._any_mask:
            y_act = y[self._mask]                            # (m_act,)
            rho_c_contrib = float(self._rho_c_masked @ y_act)
            u += rho_c_contrib                               # broadcast: scalar added to all j
            
            if debug_mode:
                print(f"        [QUAD_RECON] Computing grad_g_contrib...")
            
            # Compute grad_g_contrib with numerical safeguards
            # Use np.errstate to suppress warnings during overflow
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                grad_g_contrib = self._grad_g_masked.T @ y_act   # (n,)
            
            # Handle numerical issues AGGRESSIVELY: clip to prevent overflow
            # Max gradient magnitude should be reasonable relative to problem scale
            max_grad_mag = 1e6  # Conservative limit
            grad_g_contrib = np.nan_to_num(grad_g_contrib, nan=0.0, posinf=max_grad_mag, neginf=-max_grad_mag)
            grad_g_contrib = np.clip(grad_g_contrib, -max_grad_mag, max_grad_mag)
            
            v += grad_g_contrib                              # (n,)
            
            # Also clip v after accumulation
            v = np.clip(v, -max_grad_mag, max_grad_mag)
            
            if debug_mode:
                print(f"        [QUAD_RECON] grad_g_contrib computed, norm={np.linalg.norm(grad_g_contrib):.6e}")
                has_inf_u = np.isinf(u).any()
                has_nan_u = np.isnan(u).any()
                has_inf_v = np.isinf(v).any()
                has_nan_v = np.isnan(v).any()
                print(f"        [QUAD_RECON] u: has_nan={has_nan_u}, has_inf={has_inf_u}")
                print(f"        [QUAD_RECON] v: has_nan={has_nan_v}, has_inf={has_inf_v}")

        # dx = -sigma2 * v / u,  clipped to [-sigma, sigma]
        if debug_mode:
            print(f"        [QUAD_RECON] Computing dx...")
        
        # Use safer computation to avoid overflow
        safe_u = np.where(u == 0.0, 1.0, u)
        
        # Clip v to reasonable range before division to avoid overflow
        v_clipped = np.clip(v, -1e8, 1e8)
        
        # Compute ratio with overflow protection
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            ratio = -sigma2 * v_clipped / safe_u
        
        # Handle overflow/underflow
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=sigma, neginf=-sigma)
        ratio = np.clip(ratio, -1e10, 1e10)  # Clip before further clipping
        dx = np.clip(ratio, -sigma, sigma)
        
        if debug_mode:
            print(f"        [QUAD_RECON] dx computed: norm={np.linalg.norm(dx):.6e}, min={dx.min():.6e}, max={dx.max():.6e}")

        x_candidate = self.x_k + dx
        x_candidate = np.clip(x_candidate, self.lb, self.ub)
        x_candidate = np.where(sigma == 0.0, self.x_k, x_candidate)
        
        if debug_mode:
            print(f"        [QUAD_RECON] x_candidate computed: norm={np.linalg.norm(x_candidate):.6e}")

        dxj  = x_candidate - self.x_k
        dx2  = dxj * dxj
        safe_sigma2 = np.maximum(1e-30, sigma2)
        dx2sig = 0.5 * dx2 / safe_sigma2                    # (n,)

        if debug_mode:
            print(f"        [QUAD_RECON] Computing tilde_f...")
        
        tilde_f  = self.f_k + float((self.grad_f_k * dxj + self.rho * dx2sig).sum())
        
        if debug_mode:
            print(f"        [QUAD_RECON] tilde_f={tilde_f:.6e}")
            print(f"        [QUAD_RECON] Computing tilde_gc...")

        tilde_gc = self._g_k_clean.copy()
        if self._any_mask:
            tc_contrib = (self._grad_g_masked * dxj[np.newaxis, :] +
                          self._rho_c_masked[:, np.newaxis] * dx2sig[np.newaxis, :])
            tilde_gc[self._mask] += tc_contrib.sum(axis=1)
        
        if debug_mode:
            print(f"        [QUAD_RECON] tilde_gc computed: min={tilde_gc.min():.6e}, max={tilde_gc.max():.6e}")
            print(f"        [QUAD_RECON] Computing w_val and val_extra...")

        w_val     = dx2sig                                    # (n,)
        val_extra = float((v * dxj + 0.5 * u * dx2 / safe_sigma2).sum())
        
        if debug_mode:
            print(f"        [QUAD_RECON] w_val_mean={float(np.mean(w_val)):.6e}, val_extra={val_extra:.6e}")
            print(f"        [QUAD_RECON] DONE")

        return x_candidate, tilde_f, tilde_gc, w_val, val_extra

    def reconstruct_xcandidate_from_y(self, y: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, float, float]:
        """
        Given y (m,), reconstruct x_candidate and compute:
        - tilde_f   : approximated objective at x_candidate
        - tilde_gc  : approximated constraints vector at x_candidate
        - w_val     : scalar weight (mean across coordinates) for rho updates
        - val_extra : per-variable contribution to dual objective (summed)
        Return: x_candidate, tilde_f, tilde_gc, w_val_scalar, val_extra
        """
        import os
        debug_mode = os.environ.get('CCSA_DEBUG', '0') == '1'
        
        if debug_mode:
            print(f"      [RECONSTRUCT ENTRY] y={y}, self.m={self.m}, self.quadratic={self.quadratic}")
        
        y = np.asarray(y, dtype=float).ravel() if self.m > 0 else np.zeros(0, dtype=float)

        try:
            if not self.quadratic:
                if debug_mode:
                    print(f"      [RECONSTRUCT] Using MMA reconstruction...")
                x_candidate, tilde_f, tilde_gc, w_val, val_extra = self._reconstruct_mma(y)
            else:
                if debug_mode:
                    print(f"      [RECONSTRUCT] Using QUADRATIC reconstruction...")
                x_candidate, tilde_f, tilde_gc, w_val, val_extra = self._reconstruct_quadratic(y)
            
            if debug_mode:
                print(f"      [RECONSTRUCT MID] x_candidate shape={x_candidate.shape}, tilde_f={tilde_f:.6e}")

            w_floor = np.maximum(1e-12, np.mean(self.sigma)**2 * 1e-14)
            w_val   = np.maximum(w_val, w_floor)
            w_val_scalar = float(np.mean(w_val))
            
            if debug_mode:
                print(f"      [RECONSTRUCT OK] w_val_scalar={w_val_scalar:.6e}, val_extra={val_extra:.6e}")

            return x_candidate, tilde_f, tilde_gc, w_val_scalar, val_extra
        except Exception as e:
            print(f"      [RECONSTRUCT ERROR] Exception: {e}")
            import traceback
            traceback.print_exc()
            raise

    def build_dual_objective(self):
        """
        Returns (obj_only, obj_with_grad) closures suitable for scipy minimize with jac=True.
        obj_with_grad(y) -> (value, grad)
        """
        
        import os
        debug_mode = os.environ.get('CCSA_DEBUG', '0') == '1'

        def obj_and_grad(y):
            if debug_mode:
                print(f"    [DUAL_OBJ] y={y}")
            try:
                x_candidate, tilde_f, tilde_gc, w_val, val_extra = self.reconstruct_xcandidate_from_y(y)
                if debug_mode:
                    print(f"      [RECONSTRUCT OK] tilde_f={tilde_f:.6e}, tilde_gc: min={tilde_gc.min():.6e}, max={tilde_gc.max():.6e}")
                    print(f"      [RECONSTRUCT OK] w_val={w_val:.6e}, val_extra={val_extra:.6e}")
                
                val = tilde_f + val_extra
                if self.m > 0:
                    g_dot_y = float(np.dot(y, self.g_k))
                    val += g_dot_y
                    if debug_mode:
                        print(f"      [DUAL_COMP] tilde_f={tilde_f:.6e}, val_extra={val_extra:.6e}, g_dot_y={g_dot_y:.6e}, total_val={val:.6e}")
                    grad = -tilde_gc
                else:
                    grad = np.zeros(0, dtype=float)
                
                if debug_mode:
                    print(f"      [DUAL_OBJ_RESULT] dual_obj=-val={-float(val):.6e}, grad_norm={np.linalg.norm(grad):.6e}")
                return -float(val), grad
            except Exception as e:
                print(f"    [ERROR] Exception in obj_and_grad: {e}")
                import traceback
                traceback.print_exc()
                raise

        def obj_only(y):
            v, _ = obj_and_grad(y)
            return v

        def obj_with_grad(y):
            return obj_and_grad(y)

        return obj_only, obj_with_grad
    


def solve_dual_projected_gradient(obj_with_grad, m, lr=0.1, max_iter=20, tol=1e-6):
    """
    max_{y >= 0}  D(y)  via projected gradient ascent.
    
    Each iteration:
      1. Evaluate gradient of D at current y  — ONE obj_with_grad call
      2. Gradient ascent step
      3. Project to y >= 0
    
    Total calls: max_iter + 1  (typically converges in 5-10 steps)
    """
    y = np.zeros(m, dtype=np.float64)

    for _ in range(max_iter):
        neg_D, neg_grad = obj_with_grad(y)
        grad = -neg_grad          # ∇D(y) = -grad(neg_D)

        y_new = np.maximum(y + lr * grad, 0.0)   # step + project

        if np.linalg.norm(y_new - y) < tol:
            y = y_new
            break
        y = y_new

    return y