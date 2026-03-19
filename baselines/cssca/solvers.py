# cssca/solvers.py
import numpy as np
from typing import Tuple, Optional
from scipy.optimize import minimize

def solve_convex_subproblem_quad(surrogates, x0: np.ndarray,
                                 bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                 feasible_tol: float = 1e-8,
                                 maxiter: int = 200) -> Tuple[np.ndarray, bool, dict]:
    """
    Solve:
       minimize fbar0(x)
       s.t. fbar_i(x) <= 0, i=1..m
             lb <= x <= ub (optional)
    using SLSQP (it supports bounds and nonlinear constraints) with analytic gradients from surrogates.

    Returns (x_opt, feasible_flag, result_info)
    """
    n = x0.size
    lb = None; ub = None
    if bounds is not None:
        lb = np.asarray(bounds[0], dtype=float).ravel()
        ub = np.asarray(bounds[1], dtype=float).ravel()
        # convert to scipy bounds list
        bnds = [(None, None)] * n
        for i in range(n):
            bnds[i] = (None if np.isneginf(lb[i]) else lb[i],
                       None if np.isposinf(ub[i]) else ub[i])
    else:
        bnds = [(None, None)] * n

    def obj(x):
        v, g = surrogates.eval_surrogate(x)
        return float(v), g

    # constraints functions for SLSQP need dicts with 'fun' and 'jac'
    def make_cons(i):
        def cons_fun(x):
            vals, grads = surrogates.eval_constraints_surrogates(x)
            return float(vals[i])
        def cons_jac(x):
            vals, grads = surrogates.eval_constraints_surrogates(x)
            return grads[i, :].astype(float)
        return {'type': 'ineq', 'fun': lambda x, idx=i: -cons_fun(x),  # want fbar_i(x) <= 0 -> -fbar_i >= 0
                'jac': lambda x, idx=i: -cons_jac(x)}

    cons = [make_cons(i) for i in range(surrogates.m)]

    x0 = np.asarray(x0, dtype=float).ravel()

    res = minimize(lambda x: obj(x)[0],
                   x0,
                   method='SLSQP',
                   jac=lambda x: obj(x)[1],
                   bounds=bnds,
                   constraints=cons,
                   options={'maxiter': maxiter, 'ftol': 1e-9})
    
    #print("SLSQP success:", res.success)
    #print("SLSQP message:", res.message)
    #print("SLSQP iterations:", res.nit)
    #print("SLSQP objective value:", res.fun)


    x_opt = res.x
    # check feasibility: all fbar_i(x_opt) <= feasible_tol
    vals, _ = surrogates.eval_constraints_surrogates(x_opt)
    feasible = np.all(vals <= feasible_tol)
    return x_opt, bool(feasible), {'success': res.success, 'message': res.message, 'fun': res.fun, 'nit': res.nit}
