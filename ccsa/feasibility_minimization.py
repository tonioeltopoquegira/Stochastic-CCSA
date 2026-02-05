import numpy as np
from scipy.optimize import minimize  # <-- add this line

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

def feasibility_solver(L, U, x_k, g_k, grad_g_k, bounds):
    """
    Solve the feasibility subproblem: minimize alpha s.t. surrogate_i(x) <= alpha for all i.
    
    Parameters:
    - L, U: asymptote arrays
    - x_k: current iterate
    - g_k: constraint values at x_k
    - grad_g_k: constraint gradients at x_k
    - bounds: (lb, ub) for x
    
    Returns:
    - x_bar: feasible point (or best attempt)
    - success: bool indicating if solver succeeded
    """
    n = x_k.size
    m = g_k.size
    
    def obj_z(z):
        x = z[:n]
        alpha = z[n]
        return alpha
    
    def grad_obj_z(z):
        g = np.zeros_like(z)
        g[n] = 1.0
        return g
    
    def make_cons_i(i):
        def cfun(z, idx=i):
            x = z[:n]
            alpha = z[n]
            vals, _ = eval_mma_constraint_surrogates(L, U, x_k, g_k, grad_g_k, x)
            return -(vals[idx] - alpha)
        
        def cjac(z, idx=i):
            x = z[:n]
            alpha = z[n]
            vals, grads = eval_mma_constraint_surrogates(L, U, x_k, g_k, grad_g_k, x)
            jac = np.zeros(n + 1, dtype=float)
            jac[:n] = -grads[idx, :]
            jac[n] = 1.0
            return jac
        
        return {'type': 'ineq', 'fun': cfun, 'jac': cjac}
    
    cons = [make_cons_i(i) for i in range(m)]
    
    if bounds is not None:
        lb, ub = bounds
        bnds = []
        for i in range(n):
            bnds.append((None if np.isneginf(lb[i]) else lb[i],
                         None if np.isposinf(ub[i]) else ub[i]))
    else:
        bnds = [(None, None)] * n
    bnds.append((None, None))  # alpha
    
    z0 = np.concatenate([x_k, np.array([1.0], dtype=float)])
    
    res = minimize(obj_z, z0, jac=grad_obj_z, method='SLSQP', bounds=bnds, constraints=cons,
                   options={'maxiter': 200, 'ftol': 1e-9})
    
    return res.x[:n], res.success
