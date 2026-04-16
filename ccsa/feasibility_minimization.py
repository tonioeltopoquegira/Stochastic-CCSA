import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve
import time

def eval_mma_constraint_surrogates(L, U, x_k, g_k, grad_g_k, x):
    """
    Evaluate MMA constraint surrogates and their gradients at point x.
    
    Parameters:
    - L, U: asymptote arrays (shape n,)
    - x_k: current iterate (shape n,)
    - g_k: constraint values at x_k (shape m,)
    - grad_g_k: constraint gradients at x_k (shape m, n)
    - x: evaluation point (shape n,)
    
    Returns:
    - surrogates: surrogate values (shape m,)
    - grads: surrogate gradients (shape m, n)
    """
    m = g_k.size
    n = x.size
    surrogates = np.zeros(m)
    grads = np.zeros((m, n))
    
    for i in range(m):
        s = g_k[i]
        grad_s = np.zeros(n)
        for j in range(n):
            p = (U[j] - x_k[j])**2 * max(0, grad_g_k[i, j])
            q = (x_k[j] - L[j])**2 * max(0, -grad_g_k[i, j])
            
            if U[j] > x_k[j]:
                term = p / (U[j] - x[j]) - p / (U[j] - x_k[j])
                s += term
                grad_s[j] += p / (U[j] - x[j])**2
            
            if x_k[j] > L[j]:
                term = q / (x[j] - L[j]) - q / (x_k[j] - L[j])
                s += term
                grad_s[j] += -q / (x[j] - L[j])**2
        
        surrogates[i] = s
        grads[i, :] = grad_s
    
    return surrogates, grads

def eval_quadratic_surrogates(x_k, g_k, grad_g_k, rho_c, sigma, x):
    """
    Quadratic CCSA surrogate for each constraint.
 
    surrogate_i(x) = g_k[i] + grad_g_k[i]^T dx
                     + 0.5 * rho_c[i] * sum_j dx_j^2 / sigma_j^2
 
    Returns:
        surrogates: (m,)
        grads:      (m, n)  d surrogate_i / d x_j
    """
    dx     = x - x_k                                        # (n,)
    sigma2 = np.maximum(sigma * sigma, 1e-30)               # (n,)
 
    # Linear part
    linear = g_k + grad_g_k @ dx                            # (m,)
 
    # Quadratic part: same scaled norm for all i, scaled by rho_c[i]
    dx2_over_sigma2 = (dx * dx) / sigma2                    # (n,)
    quad_scalar     = 0.5 * np.sum(dx2_over_sigma2)         # scalar
    quad_per_constr = rho_c * quad_scalar                   # (m,)
 
    surrogates = linear + quad_per_constr                   # (m,)
 
    # Gradient: d/dx_j surrogate_i = grad_g_k[i,j] + rho_c[i] * dx_j / sigma_j^2
    dx_over_sigma2 = dx / sigma2                            # (n,)
    grads = grad_g_k + rho_c[:, np.newaxis] * dx_over_sigma2[np.newaxis, :]  # (m,n)
 
    return surrogates, grads



def feasibility_solver(L, U, x_k, g_k, grad_g_k, bounds, rho_c=None, sigma=None, method='closed-form'):
    """
    Solve the feasibility subproblem.
    
    Minimize: penalty = sum_i max(surrogate_i(x), 0)^2
    s.t. L <= x <= U
    
    Uses the SAME quadratic surrogate parameters (rho_c, sigma) as the main optimizer.
    
    Parameters:
    - L, U: asymptote arrays
    - x_k: current iterate
    - g_k: constraint values at x_k (shape m,)
    - grad_g_k: constraint gradients at x_k (shape m, n)
    - bounds: (lb, ub) for x
    - rho_c: per-constraint curvature vector (shape m, default: 1.0 * ones)
    - sigma: per-coordinate step sizes (default: 0.01 * ones)
    - method: 'closed-form' (analytical active-set, default), 'lbfgs', 'cg' (numerical optimization)
    
    Returns:
    - x_bar: feasible point (or best attempt)
    """
    n = x_k.size
    m = g_k.size
    
    # Use provided rho_c and sigma, or defaults (matching main optimizer defaults)
    if rho_c is None:
        rho_c = np.ones(m) * 1.0
    if sigma is None:
        sigma = np.ones(n) * 0.01

    #print(f'Inside feasibility solver (method={method}, rho_c={rho_c})')
    t = time.time()
    
    if bounds is not None:
        lb, ub = bounds
    else:
        lb, ub = -np.inf * np.ones(n), np.inf * np.ones(n)
    
    # Evaluate initial constraint
    g_init = np.max(g_k)
    ##print(f'  Initial max constraint: {g_init:.6e}')
    
    if g_init <= 1e-7:
        #print(f'  Already feasible!')
        return x_k
    
    # Objective: minimize penalty = sum of max(g_i, 0)^2
    def obj_and_grad(x):
        """Compute penalty objective and gradient using rho_c per constraint."""
        dx = x - x_k
        
        # Compute surrogates (vectorized, no nested loops!)
        # Linear part: shape (m,)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            linear_terms = g_k + grad_g_k @ dx
        
        linear_terms = np.nan_to_num(linear_terms, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Quadratic term per constraint: 0.5 * rho_c[i] * sum((dx[j]^2) / sigma[j]^2)
        # Uses rho_c[i] from main optimizer (passed in as parameter)
        dx2_over_sigma2 = (dx * dx) / (sigma * sigma)
        dx2_over_sigma2 = np.nan_to_num(dx2_over_sigma2, nan=0.0, posinf=1e6, neginf=0.0)
        quad_terms = 0.5 * rho_c[:, np.newaxis] * dx2_over_sigma2[np.newaxis, :]
        quad_term_per_constr = np.sum(quad_terms, axis=1)  # shape (m,)
        
        surrogates = linear_terms + quad_term_per_constr  # shape (m,)
        
        # Penalty: sum of max(g_i, 0)^2
        penalty_terms = np.maximum(surrogates, 0.0)**2
        penalty_obj = np.sum(penalty_terms)
        
        # Gradient: sum over active constraints
        grad = np.zeros(n)
        for i in range(m):
            if surrogates[i] > 0:
                # d/dx [max(g_i, 0)^2] = 2 * max(g_i, 0) * (grad_g_i + rho_c[i] * dx / sigma^2)
                grad += 2.0 * surrogates[i] * grad_g_k[i]
                grad += 2.0 * surrogates[i] * rho_c[i] * dx / (sigma * sigma)
        
        grad = np.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return float(penalty_obj), grad
    
    # Bounds for L-BFGS-B only
    if method == 'lbfgs':
        bnds = [(lb[i], ub[i]) for i in range(n)]
        res = minimize(obj_and_grad, x_k, method='L-BFGS-B', jac=True, bounds=bnds,
                       options={'maxiter': 5, 'ftol': 1e-6, 'gtol': 1e-4})
    elif method == 'cg':
        # CG: simpler, no bounds support, but faster for unconstrained
        # CG does NOT support bounds - project manually instead
        res = minimize(obj_and_grad, x_k, method='CG', jac=True,
                       options={'maxiter': 10, 'gtol': 1e-2})

    # here prob needs the epigraph formulation
        
    else:
        raise ValueError(f"Unknown method: {method}")
  
    # Project to bounds
    res.x = np.clip(res.x, lb, ub)
   
    return res.x



import numpy as np
from scipy.optimize import minimize
import time
 
 

 
 
def feasibility_solver(x_k, g_k, grad_g_k, bounds,
                       rho_c=None, sigma=None,
                       L=None, U=None,
                       maxiter=50, ftol=1e-6, verbose=False):
    """
    Epigraph feasibility solver with quadratic surrogates.
 
    Solves:
        min_{x, alpha}   alpha
        s.t.  surrogate_i(x) - alpha <= 0,  i = 0,...,m-1
              lb <= x <= ub
 
    Args:
        x_k:       current iterate (n,)
        g_k:       constraint values at x_k (m,)
        grad_g_k:  constraint Jacobian (m, n)
        bounds:    (lb, ub) arrays or None
        rho_c:     per-constraint curvature (m,). Default: ones
        sigma:     per-coordinate step size (n,). Default: 0.01*ones
        L, U:      kept for API compatibility (unused for quadratic)
        maxiter:   SLSQP max iterations
        ftol:      SLSQP tolerance
        verbose:   print diagnostics
 
    Returns:
        x_bar: (n,) feasibility-minimizing point
    """
    n = x_k.size
    m = g_k.size
 
    if rho_c is None:
        rho_c = np.ones(m, dtype=np.float64)
    if sigma is None:
        sigma = np.ones(n, dtype=np.float64) * 0.01
 
    rho_c = np.asarray(rho_c, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
 
    g_max = float(np.max(g_k))
    if g_max <= 1e-7:
        return x_k.copy()
 
    t0 = time.time()
    if verbose:
        print(f'[FEAS-EPI] n={n}, m={m}, max_g={g_max:.4e}')
 
    # ------------------------------------------------------------------ #
    #  Augmented variable z = [x(n), alpha(1)]                            #
    # ------------------------------------------------------------------ #
 
    def obj(z):
        return float(z[n])
 
    def obj_jac(z):
        g      = np.zeros(n + 1)
        g[n]   = 1.0
        return g
 
    # SLSQP 'ineq': fun(z) >= 0
    # We write: alpha - surrogate_i(x) >= 0
    def make_con(i):
        def cfun(z):
            x_    = z[:n]
            alpha = z[n]
            surr, _ = eval_quadratic_surrogates(
                x_k, g_k, grad_g_k, rho_c, sigma, x_)
            return float(alpha - surr[i])
 
        def cjac(z):
            x_    = z[:n]
            surr, grads = eval_quadratic_surrogates(
                x_k, g_k, grad_g_k, rho_c, sigma, x_)
            jac       = np.zeros(n + 1)
            jac[:n]   = -grads[i]    # d/dx (alpha - surr_i) = -grad_surr_i
            jac[n]    = 1.0          # d/dalpha = 1
            return jac
 
        return {'type': 'ineq', 'fun': cfun, 'jac': cjac}
 
    constraints = [make_con(i) for i in range(m)]
 
    # Bounds on x; alpha unconstrained
    if bounds is not None:
        lb_arr, ub_arr = bounds
        x_bnds = [
            (None if np.isneginf(lb_arr[j]) else float(lb_arr[j]),
             None if np.isposinf(ub_arr[j]) else float(ub_arr[j]))
            for j in range(n)
        ]
    else:
        x_bnds = [(None, None)] * n
    all_bnds = x_bnds + [(None, None)]   # alpha free
 
    # Warm start
    alpha0 = g_max * 1.1 + 1e-4
    z0     = np.concatenate([x_k.copy(), [alpha0]])
 
    res = minimize(obj, z0, jac=obj_jac,
                   method='SLSQP',
                   bounds=all_bnds,
                   constraints=constraints,
                   options={'maxiter': maxiter, 'ftol': ftol})
 
    x_bar = res.x[:n]
    if bounds is not None:
        x_bar = np.clip(x_bar, lb_arr, ub_arr)
 
    if verbose:
        alpha_f = float(res.x[n])
        print(f'[FEAS-EPI] {time.time()-t0:.2f}s | '
              f'nit={res.nit} | success={res.success} | '
              f'alpha*={alpha_f:.4e}')
 
    return x_bar
 
import numpy as np
from scipy.linalg import solve as _slv
import time
 
 
def _solve_lambda_qp(M, b, m):
    """
    Solve  min_{lambda>=0, sum(lambda)=1}  0.5*lambda^T M lambda - b^T lambda
    via active-set on the KKT conditions.
    M is (m x m) SPD, b is (m,).
    Returns lambda (m,) >= 0 with sum = 1.
    """
    # Convert sum=1 constraint via substitution or solve with equality
    # Use active-set: start with all free, fix negatives
    lam  = np.zeros(m)
    free = np.ones(m, dtype=bool)
 
    # Augment for equality: solve [M, 1; 1^T, 0][lam; nu] = [b; 1]
    for _ in range(m + 2):
        nf  = np.sum(free)
        if nf == 0:
            break
        idx = np.where(free)[0]
        Mf  = M[np.ix_(idx, idx)]
        bf  = b[idx]
 
        # Add equality sum(lam_free)=1 via KKT
        A = np.zeros((nf + 1, nf + 1))
        A[:nf, :nf] = Mf
        A[:nf, nf]  = 1.0
        A[nf, :nf]  = 1.0
        rhs = np.zeros(nf + 1)
        rhs[:nf] = bf
        rhs[nf]  = 1.0
 
        try:
            sol = _slv(A, rhs, check_finite=False)
        except Exception:
            sol, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
 
        lf = sol[:nf]
        ln = np.zeros(m)
        ln[idx] = lf
 
        neg = free & (ln < -1e-12)
        if not np.any(neg):
            lam = np.maximum(ln, 0.0)
            break
        free[np.argmin(ln)] = False
 
    lam = np.maximum(lam, 0.0)
    s   = lam.sum()
    if s > 1e-15:
        lam /= s
    else:
        lam = np.ones(m) / m
    return lam
 
 
def feasibility_solver(L, U, x_k, g_k, grad_g_k, bounds,
                       rho_c=None, sigma=None, method=None,
                       verbose=False):
    """
    Closed-form epigraph feasibility for quadratic CCSA surrogates.
 
    Args:
        L, U:      asymptote arrays (API compat only, unused)
        x_k:       current iterate (n,)
        g_k:       constraint values (m,)
        grad_g_k:  Jacobian (m, n)
        bounds:    (lb, ub) or None
        rho_c:     per-constraint curvature (m,)
        sigma:     per-coord step size (n,)
        method:    ignored (API compat)
        verbose:   print diagnostics
 
    Returns:
        x_bar: (n,) solution
    """
    n = x_k.size
    m = g_k.size
 
    if rho_c is None:
        rho_c = np.ones(m, dtype=np.float64)
    if sigma is None:
        sigma = np.ones(n, dtype=np.float64) * 0.01
 
    rho_c = np.asarray(rho_c, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
 
    if bounds is not None:
        lb, ub = np.asarray(bounds[0]), np.asarray(bounds[1])
    else:
        lb = np.full(n, -np.inf)
        ub = np.full(n,  np.inf)
 
    t0    = time.time()
    g_max = float(np.max(g_k))
 
    if g_max <= 1e-7:
        return x_k.copy()
 
    sigma2 = np.maximum(sigma * sigma, 1e-30)   # (n,)
 
    # ------------------------------------------------------------------ #
    #  Build the (m x m) dual QP                                          #
    #                                                                      #
    #  Optimal dx for dual weights lambda:                                 #
    #    dx_j = -sigma_j^2 * (G^T lam)_j / (rho_c^T lam)                #
    #                                                                      #
    #  Substituting into surrogate_i = g_k[i] + G[i]^T dx + rho_c[i]*Q  #
    #  where Q = 0.5*||dx/sigma||^2, gives:                               #
    #                                                                      #
    #    surrogate_i(lam) = g_k[i]                                        #
    #      - (rho_c^T lam)^{-1} * sum_j G[i,j]*G^T[j]*lam * sigma_j^2   #
    #      + 0.5*rho_c[i]/(rho_c^T lam)^2 * ||G^T lam * sigma||^2        #
    #                                                                      #
    #  This is a rational function of lam. We linearise by solving a      #
    #  sequence of (m x m) QPs updated with the current rho_lam estimate. #
    # ------------------------------------------------------------------ #
 
    # Sigma-scaled gradient: W[i,j] = G[i,j]*sigma[j],  shape (m,n)
    W = grad_g_k * sigma[np.newaxis, :]
 
    # C = W @ W^T,  shape (m,m): C[i,j] = sum_k G[i,k]*G[j,k]*sigma_k^2
    C = W @ W.T
 
    lam = np.ones(m, dtype=np.float64) / m   # warm start
 
    x_bar = x_k.copy()
    best_alpha = np.inf
 
    for iteration in range(10):
        rho_lam = float(rho_c @ lam)
        if rho_lam < 1e-15:
            rho_lam = 1e-15
 
        # Dual QP matrices (from surrogate equalisation condition)
        # M[i,j] = C[i,j] / rho_lam^2 * (rho_c[i]*rho_c[j] is absorbed below)
        # Simpler: build M directly from the linearised surrogate
        #   surrogate_i ≈ g_k[i] - C[i,:] @ lam / rho_lam
        #                 + 0.5*rho_c[i] * (lam^T C lam) / rho_lam^2
        # Minimise max_i surrogate_i(lam) <=> solve dual QP:
        #   min_{lam>=0, sum=1}  -g_k^T lam + 0.5 * lam^T M_eff lam
        # where M_eff[i,j] = C[i,j]/rho_lam  (linearised)
        M_eff = C / rho_lam
        b_vec = g_k.copy()   # dual linear term = g_k
 
        lam_new = _solve_lambda_qp(M_eff, b_vec, m)
 
        # Compute dx from lam_new
        rho_lam_new  = float(rho_c @ lam_new)
        if rho_lam_new < 1e-15:
            rho_lam_new = 1e-15
 
        g_lam = grad_g_k.T @ lam_new          # (n,)
        dx    = -sigma2 * g_lam / rho_lam_new  # (n,)
 
        # Clip to sigma and bounds
        dx    = np.clip(dx, -sigma, sigma)
        x_try = np.clip(x_k + dx, lb, ub)
        dx    = x_try - x_k
 
        # Evaluate surrogates
        dx2_sigma2 = (dx * dx) / sigma2
        quad       = 0.5 * np.sum(dx2_sigma2)
        surr       = g_k + grad_g_k @ dx + rho_c * quad
 
        alpha = float(np.max(surr))
        if alpha < best_alpha:
            best_alpha = alpha
            x_bar = x_try.copy()
 
        # Convergence: check if lambda changed
        if np.linalg.norm(lam_new - lam) < 1e-8:
            lam = lam_new
            break
        lam = lam_new
 
    if verbose:
        dx_f   = x_bar - x_k
        surr_f = g_k + grad_g_k @ dx_f + rho_c * 0.5 * np.sum((dx_f/sigma)**2)
        print(f'[FEAS] {time.time()-t0:.4f}s | '
              f'iters={iteration+1} | '
              f'alpha*={float(np.max(surr_f)):.4e} | '
              f'spread={float(np.max(surr_f)-np.min(surr_f[surr_f>-1e10])):.4e} | '
              f'init_max_g={g_max:.4e}')
 
    return x_bar