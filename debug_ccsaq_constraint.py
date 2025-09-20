import nlopt
import numpy as np
import time

'''
def make_problem(n=2, d=0.1, seed=0):
    rng = np.random.RandomState(seed)

    # objective: unbounded below -> pushes x magnitude up
    # f(x) = -0.5 * x^T x   (minimizer -> -inf), grad = -x
    state = {"obj_calls": 0, "con_calls": 0}

    def objective(x, grad):
        state["obj_calls"] += 1
        if grad.size > 0:
            # grad = d/dx (-0.5 x^T x) = -x
            grad[:] = -np.asarray(x, dtype=float)
        # return scalar
        return -0.5 * float(np.dot(x, x))

    # linear constraint: sum(x) - d <= 0
    def constraint(x, grad):
        state["con_calls"] += 1
        if grad.size > 0:
            grad[:] = np.ones(len(x), dtype=float)
        return float(np.sum(x) - d)

    return objective, constraint, state

def run_single_init(x0, n=2, d=0.1, params=None):
    objective, constraint, state = make_problem(n=n, d=d, seed=0)
    opt = nlopt.opt(nlopt.LD_CCSAQ, n)

    # tune some CCSA parameters (match defaults in your run)
    opt.set_param("inner_gradients", 0)
    opt.set_param("always_improve", 0)
    opt.set_param("sigma_min", 1e-4)
    opt.set_param("inner_maxeval", 15)
    opt.set_maxeval(200000)

    opt.set_min_objective(objective)
    # add constraint with small tolerance (like your VAE)
    opt.add_inequality_constraint(constraint, 1e-6)

    # tolerances
    #opt.set_xtol_rel(1e-8)
    #opt.set_ftol_rel(1e-12)

    print("=== RUN INIT:", x0)
    t0 = time.time()
    try:
        xopt = opt.optimize(np.array(x0, dtype=float))
        status = opt.last_optimize_result()
        fval = opt.last_optimum_value()
        print("  -> finished: status=", status, "fval=", fval)
        print("  -> xopt:", xopt)
    except RuntimeError as e:
        status = opt.last_optimize_result()
        print("  -> RuntimeError:", e, "status:", status)
        xopt = None
    dt = time.time() - t0
    print(f"  time {dt:.3f}s, obj_calls={state['obj_calls']}, con_calls={state['con_calls']}")
    print()
    return xopt, state, status

def try_inits():
    n = 2
    d = 0.1
    # inits to try (feasible, boundary, slightly infeasible, interior)
    inits = [
        np.array([0.05, 0.0]),   # feasible, near boundary (0.05 - 0 - 0.1 = -0.05)
        np.array([0.0, 0.0]),    # interior
        np.array([0.2, 0.0]),    # infeasible (0.2 - 0 - 0.1 = 0.1)
        np.array([0.11, 0.0]),   # slightly infeasible
        np.array([-0.1, -0.1]),  # safe interior
        np.array([1e-4, 1e-4])   # very small interior
    ]
    results = []
    for x0 in inits:
        res = run_single_init(x0, n=n, d=d)
        results.append((x0, res))
    return results

if __name__ == "__main__":
    print("NLopt version check")
    try:
        import nlopt
        print("nlopt version OK")
    except Exception as e:
        print("nlopt import error:", e)
    try_inits()'''



def make_train_iter(stochastic=True, rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    i = {"count": 0}
    def next_batch():
        i["count"] += 1
        if stochastic:
            return rng.randn(4, 4)  
        else:
            return np.ones((4,4))
    return next_batch, i


LOG = {"obj_calls": 0, "con_calls": 0, "calls": []}


def make_wrappers(train_iter_fn):
    
    current_batch = {"batch": None}
    def set_batch():
        current_batch["batch"] = train_iter_fn()
    def clear_batch():
        current_batch["batch"] = None

    def objective(x, grad):
        LOG["obj_calls"] += 1
        idx = LOG["obj_calls"]
        batch_used = current_batch["batch"] is not None
        LOG["calls"].append(("OBJ", idx, np.array(x, dtype=float), grad.size, batch_used, time.time()))
        print(f"[OBJ #{idx}] grad_len={grad.size} batch_cached={batch_used} x={np.array(x)}")
        # simple quadratic objective, deterministic in x
        if grad.size > 0:
            grad[:] = -np.asarray(x, dtype=float)
        return -0.5 * float(np.dot(x, x))

    def constraint(x, grad):
        LOG["con_calls"] += 1
        idx = LOG["con_calls"]
        batch_used = current_batch["batch"] is not None
        LOG["calls"].append(("CON", idx, np.array(x, dtype=float), grad.size, batch_used, time.time()))
        print(f"[CON #{idx}] grad_len={grad.size} batch_cached={batch_used} x={np.array(x)}")
        
        if grad.size > 0:
            grad[:] = np.ones(len(x), dtype=float)
        return float(np.sum(x) - 0.1)

    return set_batch, clear_batch, objective, constraint

def run_one(init_x):
    next_batch, batch_state = make_train_iter(stochastic=True)
    set_batch, clear_batch, objective, constraint = make_wrappers(next_batch)

    n = len(init_x)
    opt = nlopt.opt(nlopt.LD_CCSAQ, n)
    
    opt.set_param("inner_gradients", 0)
    opt.set_param("always_improve", 0)
    opt.set_param("sigma_min", 1e-4)
    opt.set_param("inner_maxeval", 50)
    opt.set_maxeval(10000000)
    opt.set_min_objective(objective)
    opt.add_inequality_constraint(constraint, 1e-6)
    #opt.set_xtol_rel(1e-8)
    #opt.set_ftol_rel(1e-12)

    print("\n== RUN with init:", init_x, "stochastic:", True)
    
    #set_batch()    
    t0 = time.time()
    try:
        xopt = opt.optimize(np.array(init_x, dtype=float))
        status = opt.last_optimize_result()
        fval = opt.last_optimum_value()
        print("done status:", status, "fval:", fval, "xopt:", xopt)
    except Exception as e:
        print("optimize raised:", repr(e), "last status:", opt.last_optimize_result())
        xopt = None
    dt = time.time() - t0
    print("time:", dt, "obj_calls:", LOG["obj_calls"], "con_calls:", LOG["con_calls"], "batch_count:", batch_state["count"])
    clear_batch()
    return xopt

if __name__ == "__main__":
   
    run_one([0.05, 0.0])


