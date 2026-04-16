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


def solve_feasibility_quadratic_closed_form(x_k, g_k, grad_g_k, rho_c, sigma, lb, ub, max_iter=10):
    """
    Solve the feasibility problem with CLOSED-FORM solution for quadratic surrogates.
    
    Minimize: sum_i max(g_i(x), 0)^2
    where g_i(x) = g_k[i] + grad_g_k[i]·(x-x_k) + 0.5·rho_c[i]·||x-x_k||²_σ
    
    Since the surrogates are separable quadratic, we can derive a closed-form solution
    by solving the first-order optimality conditions for active constraints.
    
    The key insight: for the penalty objective, the gradient is:
        d/dx [sum_i max(g_i, 0)^2] = sum_i 2·max(g_i,0)·(grad_g_k[i] + rho_c[i]·(x-x_k)/sigma²)
    
    Setting to 0 and solving:
        (sum_i active_i · rho_c[i] / sigma²) · (x - x_k) = -sum_i active_i · grad_g_k[i]
    
    This is an m×n system that we solve iteratively (active set method):
    - Start with all constraints potentially active
    - Solve linear system for (x - x_k)
    - Clip to bounds
    - Check which constraints are actually satisfied
    - Iterate until convergence
    
    Parameters:
    - x_k, g_k, grad_g_k: current point and constraint info
    - rho_c: per-constraint curvature (shape m,)
    - sigma: per-coordinate step sizes (shape n,)
    - lb, ub: bounds
    - max_iter: max active set iterations
    
    Returns:
    - x_opt: solution point
    - n_iter: number of iterations performed
    """
    n = x_k.size
    m = g_k.size
    
    # Compute sigma squared (per-coordinate)
    sigma2 = sigma * sigma
    sigma2_inv = 1.0 / (sigma2 + 1e-30)  # avoid division by zero
    
    # Active set method: start assuming all constraints are potentially active
    active = np.ones(m, dtype=bool)
    x = x_k.copy()
    
    for iteration in range(max_iter):
        dx = x - x_k
        
        # Evaluate linear surrogates at current x
        linear_terms = g_k + grad_g_k @ dx
        
        # Update active set: constraint i is active if surrogate > -1e-6
        # (we include slightly violated constraints to avoid cycling)
        active_new = linear_terms > -1e-6
        
        if np.array_equal(active, active_new) and iteration > 0:
            # Active set converged
            break
        
        active = active_new
        n_active = np.sum(active)
        
        if n_active == 0:
            # All constraints satisfied, return x_k
            x = x_k.copy()
            break
        
        # Build and solve the linear system:
        # (sum_i active_i · rho_c[i] / sigma²) · dx = -sum_i active_i · grad_g_k[i]
        #
        # LHS matrix (n×n): Diagonal(sum_i active_i · rho_c[i] / sigma²)
        # Since surrogates are separable, the Hessian is diagonal!
        hess_diag = np.zeros(n)
        for i in range(m):
            if active[i]:
                hess_diag += rho_c[i] * sigma2_inv
        
        # Add small regularization to avoid ill-conditioning
        hess_diag += 1e-8
        
        # RHS vector (n,): -sum_i active_i · grad_g_k[i]
        rhs = np.zeros(n)
        for i in range(m):
            if active[i]:
                rhs -= grad_g_k[i]
        
        # Solve diagonal system: dx = rhs / hess_diag
        dx_new = rhs / hess_diag
        
        # Update x with step
        x_new = x_k + dx_new
        
        # Clip to bounds
        x_new = np.clip(x_new, lb, ub)
        
        # Check convergence
        if np.linalg.norm(x_new - x) < 1e-9:
            x = x_new
            break
        
        x = x_new
    
    return x, iteration + 1


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
        
    else:
        raise ValueError(f"Unknown method: {method}")
  
    # Project to bounds
    res.x = np.clip(res.x, lb, ub)
   
    return res.x

