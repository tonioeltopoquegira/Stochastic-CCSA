import numpy as np
import matplotlib.pyplot as plt


from utils import randcond, make_red_colors
from baselines.adam import adam
from baselines.adam_al import adam_augmented_lagrangian
from baselines.cssca.core import CSSCAOptimizer

import matplotlib.patches as mpatches


# Modified run_stochastic_exp to include the new CCSA implementation (with rng & single-sample oracle)
def stoch_convex_unc_exp(optimizer, optimizer_nlopt, noise, cond, sigma_mins=None, outer=5000, rng=None, n = 100, x_init=None):

    A = np.eye(n) if cond == 1 else randcond(n, cond)

    # sample generator (single source)
    def sample_xi():
        return rng.randn(n) * noise

    # Returns the analytical noisy gradient and function using the same sample per call
    def make_noisy_f_and_grad(A, sample_xi):
        def f_and_grad(x, grad=None):
            xi = sample_xi()           # single random sample
            p = A.dot(x + xi)
            val = float(np.linalg.norm(p)**2)
            grad_val = 2.0 * (A.T.dot(p))

            # Support both NLopt-style and MMAOptimizer-style calls
            if grad is True:
                return val, grad_val
            if isinstance(grad, np.ndarray):
                grad[:] = grad_val
                return val
            return val
        return f_and_grad


    oracle = make_noisy_f_and_grad(A, sample_xi)

    # For Adam: adaptors so adam receives gradient and function (but both use same sample in each call)
    def f_for_adam(x):
        # return f(x) (no gradient requested) -> uses its own sample
        return oracle(x, grad=None)

    def fgrad_for_adam(x):
        # return gradient array (adam expects this interface in your code)
        grad = np.zeros_like(x)
        oracle(x, grad=grad)
        return grad

    # Computes the error with respect to the minimizer
    def ferr(x):
        return np.linalg.norm(x)

    # Adam run (now consistent sampling because we call oracle inside these adaptors)
    x, vals, errs = adam(xinit=x_init, fgrad=fgrad_for_adam, ferr=ferr, f=f_for_adam, maxiters=2*10**5)

    # Plot setup
    plt.figure(figsize=(6, 4))
    plt.plot(vals, "-", label="adam", color="black", linewidth=1.25)
    if sigma_mins is not None:
        colors = make_red_colors(len(sigma_mins))

    # The expected optimum depends on the noise level and the condition number
    f_star = noise**2 * np.trace(A.T @ A)
    plt.axhline(y=f_star, color="k", linestyle="--", label="expected optimum")

    if optimizer_nlopt is not None:
            
            # Noisy function and gradient evaluations with counter using our convention.
            # This version draws a single xi per call and returns consistent value+grad.
            def f_and_grad_nlopt(x, grad):
                nonlocal evals, f_evals, x_errs
                xi = sample_xi()   # NLopt block uses same RNG
                p = A.dot(x + xi)
                val = float(np.linalg.norm(p)**2)
                if grad.size > 0:
                    grad[:] = 2 * (A.T @ p)
                    evals += 1
                    f_evals.append(evals)
                    x_errs.append(np.linalg.norm(x))
                    f_vals.append(val)
                else:
                    evals += 0.5
                return val
            

            if sigma_mins is not None:
                for sigma_min, color in zip(sigma_mins, colors):

                    evals = 0
                    f_evals, x_errs, f_vals = [], [], []

                    optimizer_nlopt.set_min_objective(f_and_grad_nlopt)
                    optimizer_nlopt.set_param("inner_gradients", 0)
                    optimizer_nlopt.set_param("always_improve", 0)
                    optimizer_nlopt.set_param("sigma_min", sigma_min)


                    xopt = optimizer_nlopt.optimize(x_init)
                    # NLopt: use a high-contrast blue if available from the palette
                    plt.loglog(np.asarray(f_evals), f_vals, label=f"nlopt σ={sigma_min}", color='tab:blue')
            else:
                evals = 0
                f_evals, x_errs, f_vals = [], [], []
                
                optimizer_nlopt.set_min_objective(f_and_grad_nlopt)
                optimizer_nlopt.set_param("inner_gradients", 0)
                optimizer_nlopt.set_param("always_improve", 0)
                optimizer_nlopt.set_maxeval(outer)


                xopt = optimizer_nlopt.optimize(x_init)
                plt.loglog(np.asarray(f_evals), f_vals, label=f"nlopt", color='tab:blue')
                    
    # Adding the new CCSA implementation (use same oracle defined above)
    if optimizer is not None:
        optimizer.fun = oracle
        if sigma_mins is not None:
            for sigma_min, color in zip(sigma_mins, colors):
                # Reset the optimizer with the new sigma_min
                optimizer.reset()
        
                # Set the sigma_min parameter for the optimizer
                optimizer.sigma_params.sigma_min = sigma_min
                
                all_x = []
                all_f = []
            
                for out in range(outer):
                    f_b, g_b, metrics = optimizer.step()
                    all_x.append(metrics["x_history"][-1])
                    all_f.append(f_b)


                # metrics now contains the x_history and cumulative_weighted_evals_history
                x_hist = metrics["x_history"]                                  # shape (k, n)
                cum_we_hist = metrics["cumulative_weighted_evals_history"]    # shape (k,)

                all_f = [oracle(xi, grad=None) for xi in all_x]

                plt.loglog(cum_we_hist, all_f, linestyle="--", color=color,
                    label=f"custom σ={sigma_min}")
        
        else:
            all_x = []
            all_f = [oracle(x_init, grad=None)]
        
            for out in range(outer):
                f_b, g_b, metrics = optimizer.step()
                all_x.append(metrics["x_history"][-1])
                all_f.append(f_b)


            # metrics now contains the x_history and cumulative_weighted_evals_history
            x_hist = metrics["x_history"]                                  # shape (k, n)
            cum_we_hist = metrics["cumulative_weighted_evals_history"]    # shape (k,)

            
            plt.loglog(cum_we_hist, all_f, linestyle="--", color="blue",
                    label=f"custom")
        

        plt.legend()
        plt.xlabel("function+gradient evaluations")
        plt.ylabel("loss")
        plt.title(f"noise={noise}, cond#={cond}")
        plt.xscale("log")
        plt.yscale("log")
        plt.show()


    optimizer.summarize_diagnostics()


def stoch_convex_con_exp(
    optimizer,
    optimizer_quad,
    optimizer_nlopt,
    noise: float = 0.2,
    cond: int = 1,
    c: np.ndarray = None,
    b: float = 0.0,
    x0: np.ndarray = None,
    sigma_mins: list = [0.0, 0.01, 0.1],
    seed: int = 0,
    maxiters_adam: int = 20000,
    mma_maxeval: int = 2000,
    active_constraint = True,
    init_feasible: bool = False,
    inner_kwargs = None,
    rho: float = 1.0,
    lambda0: float = 0.0,
    tol_outer: float = 1e-6,
    rho_multiplier: float = 2.0,
    rho_update_max: int = 6,
    max_outer = 8,
    verbose: bool = True,
    ccsa_plot_expected: bool = True,
    ccsa_n_outer: int = 100
    , cssca_tau_obj: float = 1.0, cssca_tau_cons: float = 1.0, cssca_samples_per_iter=None
):

    # deterministic RNG for this experiment (single source)
    rng = np.random.RandomState(seed)

    n = 100
    A = np.eye(n) if cond == 1 else randcond(n, cond)
    A2 = A.T @ A
    traceA2 = float(np.trace(A2))

    

    print(f"Random constraint: ||c||={np.linalg.norm(c):.6g}, b={b:.6g}")

    # analytic expected optimum
    def analytic_solution_expectation(A, c, b, noise):
        A2 = A.T @ A
        s = np.linalg.solve(A2, c)
        denom = float(c.dot(s))
        if b >= 0:
            x_star = np.zeros_like(c)
            lambda_star = 0.0
            active_flag = False
        else:
            x_star = (b / denom) * s
            lambda_star = -2.0 * b / denom
            active_flag = True
        const_term = (noise**2) * np.trace(A2)
        val = float(x_star.dot(A2.dot(x_star))) + const_term
        return x_star, lambda_star, active_flag, val, const_term

    x_star, lambda_star, active_flag, val_star, const_term = analytic_solution_expectation(A, c, b, noise)
    print("Analytic expected optimum (on expectation):")
    print(f"  active? {active_flag}")
    print(f"  ||x*||={np.linalg.norm(x_star):.6g}, λ*={lambda_star:.6g}")
    print(f"  E[f(x*)]={val_star:.6g} (noise const={const_term:.6g})")

    val_uncon = const_term
    g_uncon = -b
    print(f"Unconstrained baseline: E[f(x_uncon)]= {val_uncon:.6g}, g(x_uncon)={g_uncon:.6g}")

    # --- deterministic sampling closure used by all function/grad calls in this run ---
    def sample_xi():
        return rng.randn(n) * noise

    # single-call noisy oracle factory: returns a function f_and_grad(x, grad=None)
    def make_noisy_f_and_grad(A, sample_xi):
        def f_and_grad(x, grad=None):
            xi = sample_xi()           # single random sample
            p = A.dot(x + xi)
            val = float(np.linalg.norm(p)**2)
            grad_val = 2.0 * (A.T.dot(p))

            # If grad is True (MMAOptimizer-style request that wants value+grad)
            if grad is True:
                return val, grad_val

            # If grad is an array (NLopt-style)
            if isinstance(grad, np.ndarray):
                grad[:] = grad_val
                return val

            # Otherwise only value requested
            return val
        return f_and_grad
    
    # stochastic helpers that use sample_xi (keeps value/grad consistent if caller uses same xi)
    def fgrad_stoch(x):
        # returns gradient only (draws a sample)
        xi = sample_xi()
        return 2.0 * (A.T @ A @ (x + xi))
        #return 2.0 * (A.T @ A @ (x))

    def f_stoch_estimate(x):
        xi = sample_xi()
        return float(np.linalg.norm(A.dot(x + xi))**2)
        #return float(np.linalg.norm(A.dot(x))**2)


    def constraint_val(x):
        xi = sample_xi()
        return float(np.dot(c, x + xi) - b)

    # Also provide expected (closed-form) objective for plotting
    def expected_f(x):
        return float(np.linalg.norm(A.dot(x))**2 + (noise**2) * traceA2)


    print(f"Initial constraint g(x0)={constraint_val(x0):.6g} (init_feasible={init_feasible})")
    print()

    # --- Use oracle adaptors for AL-Adam so Adam's function+grad calls are consistent ---
    oracle_for_adam = make_noisy_f_and_grad(A, sample_xi)

    def f_for_adam(x):
        # return f(x) (no gradient requested) -> uses a sample
        return oracle_for_adam(x, grad=None)

    def fgrad_for_adam(x):
        # return gradient array (adam expects this interface)
        grad = np.zeros_like(x)
        oracle_for_adam(x, grad=grad)
        return grad

    # Run AL-Adam now with adaptor functions (keeps same RNG sequence)
    x_al, lam_al, hist_al = adam_augmented_lagrangian(
        fgrad=fgrad_for_adam,                 # stochastic gradient of the objective (adaptor)
        x0=x0.copy(),                         # initial point
        g=constraint_val,                     # stochastic constraint evaluator g(x)
        dg=lambda xx: c,                      # analytic gradient of g w.r.t x (constant)
        rho=rho,                              # initial AL penalty
        lambda0=lambda0,                      # initial Lagrange multiplier
        f_stoch_estimate=f_for_adam,          # optional stochastic function estimate for logging
        alpha=1e-3,
        maxiters=maxiters_adam,
        tol=tol_outer,
        rho_multiplier=rho_multiplier,
        rho_update_max=rho_update_max,
        max_outer=max_outer,
        verbose=verbose)

    print("AL-Adam finished:")
    last_f_est = hist_al['f_est'][-1] if len(hist_al['f_est'])>0 else np.nan
    last_g = hist_al['g'][-1] if len(hist_al['g'])>0 else constraint_val(x_al)
    print(f"  last f_est={last_f_est:.6g}, g(x)={last_g:.6g}, λ={lam_al:.6g}")
    print()

    # NLopt-MMA 
    mma_results = []
    colors_mma = plt.cm.viridis(np.linspace(0.2, 0.8, len(sigma_mins)))
    for sigma_min, color in zip(sigma_mins, colors_mma):
        
        evals = 0.0
        mma_f_evals, mma_g_vals, mma_f_vals = [], [], []
        def f_and_grad_mma(x, grad):
            nonlocal evals, mma_f_evals, mma_g_vals, mma_f_vals
            xi = sample_xi()       
            p = A.dot(x + xi)
            val = float(np.linalg.norm(p)**2)
            if grad.size > 0:
                grad[:] = 2.0 * (A.T.dot(p))
                evals += 1
                mma_f_evals.append(evals)
                mma_f_vals.append(val)
                mma_g_vals.append(constraint_val(x))
            else:
                evals += 0.5
            return val
        def cons_nl(x, grad):
            xi = sample_xi()  
            if grad.size > 0: grad[:] = c
            return float(c.dot(x + xi) - b)
        optimizer_nlopt.add_inequality_constraint(cons_nl, 0.0)
        optimizer_nlopt.set_min_objective(f_and_grad_mma)
        optimizer_nlopt.set_maxeval(mma_maxeval)
        optimizer_nlopt.set_param("rho_init", 1.0)
        optimizer_nlopt.set_param("sigma_min", float(sigma_min))

        x_mma = optimizer_nlopt.optimize(x0.copy())
        res_code = optimizer_nlopt.last_optimize_result()
        print(f"nlopt σ={sigma_min}: res={res_code}, ||x||={np.linalg.norm(x_mma):.3g}, g(x)={constraint_val(x_mma):.3g}")
        mma_results.append((sigma_min, color, mma_f_evals, mma_g_vals, mma_f_vals))

    colors_ccsa = plt.cm.inferno(np.linspace(0.2, 0.8, len(sigma_mins)))
    ccsa_results = []


    oracle = make_noisy_f_and_grad(A, sample_xi)
    optimizer.fun = oracle
    optimizer.g = constraint_val
    optimizer.dg = lambda xx: np.atleast_2d(c)
    optimizer.x0 = x0.copy()
    optimizer.params = x0.copy(),

    for sigma_min, color in zip(sigma_mins, colors_ccsa):
        
        #optimizer.reset()
        metrics = None
        f_b = None
        all_x = []
        all_f = [oracle(x0, grad=None)]

        for out in range(mma_maxeval):
            f_b, g_b, metrics = optimizer.step()
            all_x.append(metrics["x_history"][-1])

        optimizer.summarize_diagnostics()
        all_f = [oracle(xi, grad=None) for xi in all_x]

        
        x_hist = np.asarray(metrics.get("x_history", []))
        cum_we_hist = np.asarray(metrics.get("cumulative_weighted_evals_history", []), dtype=float)
        if x_hist.ndim == 1 and x_hist.size>0:
            x_hist = x_hist.reshape((1, -1))
        if x_hist.shape[0] > 0:
            f_expected_at_xhist = [expected_f(xi) for xi in x_hist]
            f_stoch_at_xhist = [oracle(xi, grad=None) for xi in x_hist]
            g_at_xhist = [constraint_val(xi) for xi in x_hist]
        else:
            f_expected_at_xhist = []
            f_stoch_at_xhist = []
            g_at_xhist = []
        if cum_we_hist.size == 0:
            cum_we_hist = np.arange(1, len(f_expected_at_xhist)+1, dtype=float)

        ccsa_results.append({
            "sigma_min": sigma_min,
            "color": color,
            "metrics": metrics,
            "x_hist": x_hist,
            "cum_we_hist": cum_we_hist,
            "f_expected_at_xhist": np.array(f_expected_at_xhist),
            "f_stoch_at_xhist": np.array(f_stoch_at_xhist),
            "g_at_xhist": np.array(g_at_xhist)
        })
        print(f"CCSA σ={sigma_min}: cumulative_wval={metrics.get('cumulative_wval', np.nan):.6g}, weighted_evals={metrics.get('weighted_evals', np.nan)}, x_history_len={len(x_hist)}")

    
    colors_ccsa = plt.cm.inferno(np.linspace(0.2, 0.8, len(sigma_mins)))
    ccsa_quad_results = []

    # --- Run baseline CSSCA optimizer (single constant rho schedule) ---
    # Allow cssca_tau_obj and cssca_tau_cons to be scalars or lists; run one config per pair
    cssca_tau_objs_list = cssca_tau_obj if isinstance(cssca_tau_obj, (list, tuple, np.ndarray)) else [cssca_tau_obj]
    cssca_tau_cons_list = cssca_tau_cons if isinstance(cssca_tau_cons, (list, tuple, np.ndarray)) else [cssca_tau_cons]
    # samples per iter
    if cssca_samples_per_iter is None:
        cssca_samples_list = [1]
    else:
        cssca_samples_list = cssca_samples_per_iter if isinstance(cssca_samples_per_iter, (list, tuple, np.ndarray)) else [cssca_samples_per_iter]

    # broadcast scalars to common length
    L = max(len(cssca_tau_objs_list), len(cssca_tau_cons_list), len(cssca_samples_list))
    def _broadcast(lst, L):
        if len(lst) == L:
            return list(lst)
        if len(lst) == 1:
            return list(lst) * L
        raise ValueError("CSSCA parameter lists must be length 1 or equal to each other")

    cssca_tau_objs_list = _broadcast(list(cssca_tau_objs_list), L)
    cssca_tau_cons_list = _broadcast(list(cssca_tau_cons_list), L)
    cssca_samples_list = _broadcast(list(cssca_samples_list), L)

    cssca_runs = []
    for tau_o, tau_c, samples_p in zip(cssca_tau_objs_list, cssca_tau_cons_list, cssca_samples_list):
        try:
            cssca_opt = CSSCAOptimizer(params=x0.copy(),
                                       fun=make_noisy_f_and_grad(A, lambda: np.zeros(n)),
                                       g=lambda xx: float(np.dot(c, xx) - b),
                                       dg=lambda xx: np.atleast_2d(c),
                                       x0=x0.copy(), rho_t_schedule=float(rho), gamma_t_schedule=1.0,
                                       tau_obj=float(tau_o), tau_cons=float(tau_c), samples_per_iter=1.0)

            cssca_f_hist = []
            cssca_cons_hist = []
            for t in range(mma_maxeval):
                x_cssca, f_cssca, cons_cssca = cssca_opt.step()
                cssca_f_hist.append(f_cssca)
                cssca_cons_hist.append(cons_cssca.copy() if hasattr(cons_cssca, 'copy') else np.atleast_1d(cons_cssca))

            cssca_cons_arr = np.vstack(cssca_cons_hist) if len(cssca_cons_hist) > 0 and np.asarray(cssca_cons_hist).ndim == 2 else np.asarray(cssca_cons_hist)
            print(f"CSSCA (tau_obj={tau_o}, tau_cons={tau_c}, rho={rho}): last f={cssca_f_hist[-1]:.6g}, last g[0]={cssca_cons_arr[-1,0] if cssca_cons_arr.size>0 else np.nan:.6g}")
            cssca_runs.append({"tau_obj": tau_o, "tau_cons": tau_c, "f_hist": cssca_f_hist, "cons_arr": cssca_cons_arr})
        except Exception as e:
            print(f"Failed to run CSSCA optimizer (tau_obj={tau_o}, tau_cons={tau_c}): {e}")
            cssca_runs.append({"tau_obj": tau_o, "tau_cons": tau_c, "f_hist": [], "cons_arr": np.array([])})

    oracle = make_noisy_f_and_grad(A, sample_xi)
    optimizer_quad.fun = oracle
    optimizer_quad.g = constraint_val
    optimizer_quad.dg = lambda xx: np.atleast_2d(c)
    optimizer_quad.x0 = x0.copy()
    optimizer_quad.params = x0.copy(),
    for sigma_min, color in zip(sigma_mins, colors_ccsa):
        
        #optimizer.reset()
        metrics = None
        f_b = None
        all_x = []
        all_f = [oracle(x0, grad=None)]

        for out in range(mma_maxeval):
            f_b, g_b, metrics = optimizer_quad.step()
            all_x.append(metrics["x_history"][-1])

        optimizer_quad.summarize_diagnostics()
        all_f = [oracle(xi, grad=None) for xi in all_x]

        
        x_hist = np.asarray(metrics.get("x_history", []))
        cum_we_hist = np.asarray(metrics.get("cumulative_weighted_evals_history", []), dtype=float)
        if x_hist.ndim == 1 and x_hist.size>0:
            x_hist = x_hist.reshape((1, -1))
        if x_hist.shape[0] > 0:
            f_expected_at_xhist = [expected_f(xi) for xi in x_hist]
            f_stoch_at_xhist = [oracle(xi, grad=None) for xi in x_hist]
            g_at_xhist = [constraint_val(xi) for xi in x_hist]
        else:
            f_expected_at_xhist = []
            f_stoch_at_xhist = []
            g_at_xhist = []
        if cum_we_hist.size == 0:
            cum_we_hist = np.arange(1, len(f_expected_at_xhist)+1, dtype=float)

        ccsa_quad_results.append({
            "sigma_min": sigma_min,
            "color": color,
            "metrics": metrics,
            "x_hist": x_hist,
            "cum_we_hist": cum_we_hist,
            "f_expected_at_xhist": np.array(f_expected_at_xhist),
            "f_stoch_at_xhist": np.array(f_stoch_at_xhist),
            "g_at_xhist": np.array(g_at_xhist)
        })
        print(f"CCSA σ={sigma_min}: cumulative_wval={metrics.get('cumulative_wval', np.nan):.6g}, weighted_evals={metrics.get('weighted_evals', np.nan)}, x_history_len={len(x_hist)}")


    # --- Ensure first plotted value reflects x0 for all result traces ---
    init_f_expected = expected_f(x0)
    init_f_stoch = f_stoch_estimate(x0)
    init_g_nom = float(np.dot(c, x0) - b)

    def _prepend_ccsa_entry(entry):
        # entry contains 'cum_we_hist', 'f_expected_at_xhist','f_stoch_at_xhist','g_at_xhist'
        cum = np.asarray(entry.get('cum_we_hist', []), dtype=float)
        fexp = np.asarray(entry.get('f_expected_at_xhist', []), dtype=float)
        fst = np.asarray(entry.get('f_stoch_at_xhist', []), dtype=float)
        g = np.asarray(entry.get('g_at_xhist', []), dtype=float)

        if cum.size == 0:
            new_cum = np.array([1.0])
        else:
            # shift existing x-axis by +1 so we can insert x0 at position 1
            new_cum = np.concatenate(([1.0], cum + 1.0))

        new_fexp = np.concatenate(([init_f_expected], fexp)) if fexp.size > 0 else np.array([init_f_expected])
        new_fst = np.concatenate(([init_f_stoch], fst)) if fst.size > 0 else np.array([init_f_stoch])
        new_g = np.concatenate(([init_g_nom], g)) if g.size > 0 else np.array([init_g_nom])

        entry['cum_we_hist'] = new_cum
        entry['f_expected_at_xhist'] = new_fexp
        entry['f_stoch_at_xhist'] = new_fst
        entry['g_at_xhist'] = new_g

    def _prepend_cssca_run(run):
        # run contains 'f_hist' (1D) and 'cons_arr' (maybe 1D or 2D)
        fh = np.asarray(run.get('f_hist', []), dtype=float)
        if fh.size == 0:
            run['f_hist'] = np.array([init_f_expected])
        else:
            run['f_hist'] = np.concatenate(([init_f_expected], fh))

        carr = run.get('cons_arr', np.array([]))
        carr = np.asarray(carr)
        if carr.size == 0:
            run['cons_arr'] = np.array([init_g_nom])
        else:
            if carr.ndim == 1:
                run['cons_arr'] = np.concatenate(([init_g_nom], carr.astype(float)))
            elif carr.ndim == 2:
                # prepend a row with init_g_nom repeated for each constraint column
                mcols = carr.shape[1]
                first_row = np.full((1, mcols), float(init_g_nom), dtype=float)
                run['cons_arr'] = np.vstack((first_row, carr.astype(float)))
            else:
                # unexpected shape: coerce to 1D and prepend
                run['cons_arr'] = np.concatenate(([init_g_nom], carr.ravel().astype(float)))

    # apply prepend to CCSA results
    for cr in ccsa_results:
        try:
            _prepend_ccsa_entry(cr)
        except Exception:
            pass
    for cr in ccsa_quad_results:
        try:
            _prepend_ccsa_entry(cr)
        except Exception:
            pass

    # apply prepend to cssca runs
    if 'cssca_runs' in locals():
        for run in cssca_runs:
            try:
                _prepend_cssca_run(run)
            except Exception:
                pass

    # Plotting: objective + constraint panels with patches
    plt.figure(figsize=(12,5))

    # define patch half-height based on noise const (visualization choice)
    patch_half = max(1e-12, 0.5 * (noise**2) * traceA2)

    # objective panel
    ax1 = plt.subplot(1,2,1)
    ax1.plot(hist_al['iter'], hist_al['f_est'], '-', color='black', label='AL-Adam')

    #for sigma_min, color, f_evals, g_vals, f_vals in mma_results:
    #    ax1.plot(f_evals, f_vals, '.-', color=color, alpha=0.9, label=f'MMA σ={sigma_min}')

    # CCSA: use their assigned color palette (from earlier 'color' entries) for visibility
    for cr in ccsa_results:
        col = cr.get('color', 'tab:orange')
        if ccsa_plot_expected:
            ax1.plot(cr['cum_we_hist'], cr['f_expected_at_xhist'], linestyle='-', linewidth=2.25,
                     color=col, label=f'custom σ={cr["sigma_min"]}')
        else:
            ax1.plot(cr['cum_we_hist'], cr['f_stoch_at_xhist'], linestyle='-', linewidth=1.5,
                     marker='o', markersize=4, color=col, label=f'custom σ={cr["sigma_min"]}')
    
    for cr in ccsa_quad_results:
        col = cr.get('color', 'tab:green')
        if ccsa_plot_expected:
            ax1.plot(cr['cum_we_hist'], cr['f_expected_at_xhist'], linestyle='-', linewidth=2.25,
                     color=col, label=f'custom quad σ={cr["sigma_min"]}')
        else:
            ax1.plot(cr['cum_we_hist'], cr['f_stoch_at_xhist'], linestyle='-', linewidth=1.5,
                     marker='s', markersize=4, color=col, label=f'custom quad σ={cr["sigma_min"]}')

    # plot CSSCA runs if available
    try:
        if 'cssca_runs' in locals() and len(cssca_runs) > 0:
            # choose tab10 colors for CSSCA runs (high contrast); cycle if more than 10 runs
            colors_css = plt.cm.tab10(np.arange(len(cssca_runs)) % 10)
            for idx, run in enumerate(cssca_runs):
                fh = run.get('f_hist', [])
                if len(fh) > 0:
                    ax1.plot(np.arange(1, len(fh)+1), fh, linestyle='-', linewidth=2.0,
                             color=colors_css[idx], label=f"CSSCA τo={run['tau_obj']}, τc={run['tau_cons']}")
    except NameError:
        pass
    


    # shaded patches for stochastic constrained and unconstrained (horizontal bands)
    # constrained: center at val_star
    ax1.axhspan(val_star - patch_half, val_star + patch_half, alpha=0.18, facecolor='tab:orange')
    # unconstrained baseline: center at val_uncon
    #ax1.axhspan(val_uncon - patch_half, val_uncon + patch_half, alpha=0.12, facecolor='tab:gray')

    # add legend patches for these bands and deduplicate legend entries (keep order)
    constrained_patch = mpatches.Patch(facecolor='tab:orange', alpha=0.18, label='stochastic constrained (± noise const)')
    handles, labels = ax1.get_legend_handles_labels()
    # deduplicate while preserving order
    from collections import OrderedDict
    unique = OrderedDict()
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    unique[constrained_patch.get_label()] = constrained_patch
    ax1.legend(list(unique.values()), list(unique.keys()), loc='best', fontsize='small')

    ax1.axhline(val_star, color='k', linestyle='--', linewidth=1.0)
    ax1.axhline(val_uncon, color='gray', linestyle=':', linewidth=1.0)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.grid(True); ax1.set_xlabel('iter/evals'); ax1.set_ylabel('stochastic f(x)'); ax1.set_title('Objective')

    # constraint panel
    ax2 = plt.subplot(1,2,2)
    ax2.plot(hist_al['iter'], hist_al['g'], '-', color='black', label='AL-Adam g(x)')
    #for sigma_min, color, f_evals, g_vals, f_vals in mma_results:
    #    ax2.plot(f_evals, g_vals, '.-', color=color, alpha=0.9, label=f'NLOPT σ={sigma_min}')
    for cr in ccsa_results:
        col = cr.get('color', 'tab:orange')
        ax2.plot(cr['cum_we_hist'], cr['g_at_xhist'], linestyle='-', color=col, linewidth=1.8, label=f'CCSA σ={cr["sigma_min"]}')
    
    for cr in ccsa_quad_results:
        col = cr.get('color', 'tab:green')
        ax2.plot(cr['cum_we_hist'], cr['g_at_xhist'], linestyle='-', color=col, linewidth=1.8, label=f'CCSA QUAD σ={cr["sigma_min"]}')

    # CSSCA constraint traces (for each run)
    try:
        if 'cssca_runs' in locals() and len(cssca_runs) > 0:
            # Use the same high-contrast tab10 colors as the objective panel for consistency
            colors_css = plt.cm.tab10(np.arange(len(cssca_runs)) % 10)
            for idx, run in enumerate(cssca_runs):
                carr = run.get('cons_arr', np.array([]))
                if carr.size == 0:
                    continue
                if carr.ndim == 2:
                    ax2.plot(np.arange(1, carr.shape[0]+1), carr[:, 0], linestyle='-', color=colors_css[idx], linewidth=2.0,
                             label=f"CSSCA g[0] τo={run['tau_obj']}, τc={run['tau_cons']}")
                else:
                    ax2.plot(np.arange(1, len(carr)+1), carr, linestyle='-', color=colors_css[idx], linewidth=2.0,
                             label=f"CSSCA g τo={run['tau_obj']}, τc={run['tau_cons']}")
    except NameError:
        pass

    ax2.axhline(0, color='k', linestyle='--', label='feasibility')
    ax2.axhline(g_uncon, color='gray', linestyle=':', label='unconstrained g')
    ax2.axhline(constraint_val(x_star), color='k', linestyle='--', linewidth=1.0, label='constrained g')
    ax2.set_xscale('log'); ax2.grid(True); ax2.set_xlabel('iter/evals'); ax2.set_ylabel('g(x)'); ax2.set_title('Constraint violation')
    ax2.legend(loc='best', fontsize='small')

    plt.tight_layout()
    plt.savefig("convex_stoch_det_constraint_experiment.png", dpi=300)
    plt.show()

    # --- Plot σ evolution for first 3 parameters (if available) ---
    last_metrics = ccsa_results[-1]['metrics'] if len(ccsa_results) > 0 else {}
    if last_metrics:
        sig_hist = np.array(last_metrics.get("sigma_history", []))  # shape (iters, n)
        rho_hist = np.array(last_metrics.get("rho_history", []))    # shape (iters,)
    else:
        sig_hist = np.array([])
        rho_hist = np.array([])

    return {
        "c": c, "b": b,
        "x_star": x_star, "val_star": val_star,
        "x_al": x_al, "lam_al": lam_al,
        "mma_results": mma_results,
        "ccsa_results": ccsa_results,
        "val_uncon": val_uncon, "g_uncon": g_uncon,
        "hist_al": hist_al
    }


# experiments_expquad.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict



def stoch_expquad_diag_exp(
    optimizer,
    optimizer_quad,
    optimizer_nlopt,
    noise: float = 0.2,
    cond: int = 1,
    c: np.ndarray = None,
    b: float = 0.0,
    x0: np.ndarray = None,
    sigma_mins: list = [0.0, 0.01, 0.1],
    seed: int = 0,
    maxiters_adam: int = 20000,
    mma_maxeval: int = 2000,
    active_constraint = True,
    init_feasible: bool = False,
    inner_kwargs = None,
    rho: float = 1.0,
    lambda0: float = 0.0,
    tol_outer: float = 1e-6,
    rho_multiplier: float = 2.0,
    rho_update_max: int = 6,
    max_outer = 8,
    verbose: bool = True,
    ccsa_plot_expected: bool = False,
    ccsa_n_outer: int = 100,
    cssca_tau_obj: float = 1.0, cssca_tau_cons: float = 1.0, cssca_samples_per_iter=None,
    curv_change: str = 'large'
):
    """
    Stochastic constrained experiment using diagonal exponential-quadratic objective:
        f(x+xi) = exp( (x+xi)^T M (x+xi) ), M diagonal
    Diagonal noise xi ~ N(0, noise^2 I) so we can compute exact expectation.

    Keeps structure and plotting of your original stoch_convex_con_exp and
    includes CSSCA baseline runs (tau broadcasting etc.).
    """

    # deterministic RNG for this experiment (single source)
    rng = np.random.RandomState(seed)

    # Problem size and diagonal curvature
    n = 100

    if curv_change == 'small':
        # choose diagonal curvatures m_i (small values to keep 1 - 2 sigma^2 m_i > 0)
        m_min = 0.005
        m_max = 0.05 * max(1, cond)
        m = np.linspace(m_min, m_max, n)   # diagonal values of M

    if curv_change == 'large':
        m_min = 0.001           # very small -> very flat directions
        m_max = 1.0 * cond       # very large -> very steep directions
        m = np.logspace(np.log10(m_min), np.log10(m_max), n)  # logarithmic spread


    # A such that quadratic = ||A (x+xi)||^2 and M = A.T @ A
    A = np.diag(np.sqrt(m))
    M = np.diag(m)

    # noise: scalar -> isotropic diagonal with variance noise^2
    Sigma_diag = (noise**2) * np.ones(n)

    # stability: require d_i = 1 - 2 sigma^2 m_i > 0 for exact expectation formula
    d = 1.0 - 2.0 * (noise**2) * m
    if np.any(d <= 0.0):
        raise ValueError("Require 1 - 2 * noise^2 * m_i > 0 for all i. Reduce noise or curvatures m.")

    # logging
    print(f"Using diagonal exponential-quadratic objective n={n}, noise={noise}, m in [{m_min:.4g},{m_max:.4g}]")

    # --- Linear constraint setup (use provided c,b,x0 if available) ---
    if c is None:
        c = np.ones(n)
    if x0 is None:
        # choose x0: feasible if requested
        x0 = np.zeros(n) if init_feasible else (c / np.linalg.norm(c))
    if b is None:
        # set b negative so constraint is active by default
        b = -0.5

    print(f"Linear constraint: ||c||={np.linalg.norm(c):.6g}, b={b:.6g}")

    # --- Exact expectation constants (diagonal case) ---
    # log determinant term for expectation prefactor
    log_det_term = -0.5 * np.sum(np.log(d))     # log(1 / sqrt(prod_i d_i))
    # m_tilde diag = m_i / d_i used to form quadratic in exponent for expected minimization
    m_tilde = m / d

    # Expected objective (exact, diagonal case)
    def expected_f(x):
        expo = np.sum(m * (x**2) / d)   # sum_i m_i x_i^2 / d_i
        return float(np.exp(log_det_term + expo))

    # Expected constraint (exact)
    def expected_g(x):
        return float(np.dot(c, x) - b)

    # Analytic expected minimizer under linear constraint for quadratic exponent:
    # minimize x^T diag(m_tilde) x subject to c^T x <= b
    def analytic_solution_expectation_diag(mtilde_diag, c_vec, b_scalar):
        D = np.diag(mtilde_diag)
        s = np.linalg.solve(D, c_vec)     # D^{-1} c
        denom = float(c_vec.dot(s))
        if b_scalar >= 0:
            x_star = np.zeros_like(c_vec)
            lambda_star = 0.0
            active_flag = False
        else:
            x_star = (b_scalar / denom) * s
            lambda_star = -2.0 * b_scalar / denom
            active_flag = True
        val = expected_f(x_star)
        return x_star, lambda_star, active_flag, val

    x_star, lambda_star, active_flag, val_star = analytic_solution_expectation_diag(m_tilde, c, b)
    print("Analytic expected optimum (on expectation):")
    print(f"  active? {active_flag}")
    print(f"  ||x*||={np.linalg.norm(x_star):.6g}, λ*={lambda_star:.6g}")
    print(f"  E[f(x*)]={val_star:.6g}")

    val_uncon = expected_f(np.zeros(n))
    g_uncon = -b
    print(f"Unconstrained baseline: E[f(0)]= {val_uncon:.6g}, g(0)={g_uncon:.6g}")

    # --- deterministic sampling closure used by all function/grad calls in this run ---
    def sample_xi():
        return rng.randn(n) * noise

    # single-call noisy oracle factory: returns a function f_and_grad(x, grad=None)
    def make_noisy_f_and_grad(A_mat, sample_xi_fn):
        # A_mat is diag(sqrt(m)) so quadratic = ||A (x+xi)||^2 = (x+xi)^T M (x+xi)
        def f_and_grad(x, grad=None):
            xi = sample_xi_fn()
            z = x + xi
            p = A_mat.dot(z)
            quad = float(np.linalg.norm(p)**2)   # (x+xi)^T M (x+xi)
            val = float(np.exp(quad))
            # gradient: 2 * M (x+xi) * val = 2 * (A.T @ p) * val
            grad_val = 2.0 * (A_mat.T.dot(p)) * val

            if grad is True:
                return val, grad_val
            if isinstance(grad, np.ndarray):
                grad[:] = grad_val
                return val
            return val
        return f_and_grad

    oracle = make_noisy_f_and_grad(A, sample_xi)

    # Adaptor functions for Adam / AL-Adam
    def f_for_adam(x):
        return oracle(x, grad=None)

    def fgrad_for_adam(x):
        grad = np.zeros_like(x)
        oracle(x, grad=grad)
        return grad

    def constraint_val(x):
        xi = sample_xi()
        return float(np.dot(c, x + xi) - b)

    print(f"Initial constraint g(x0)={constraint_val(x0):.6g} (init_feasible={init_feasible})")
    print()

    # Run AL-Adam
    x_al, lam_al, hist_al = adam_augmented_lagrangian(
        fgrad=fgrad_for_adam,
        x0=x0.copy(),
        g=constraint_val,
        dg=lambda xx: c,
        rho=rho,
        lambda0=lambda0,
        f_stoch_estimate=f_for_adam,
        alpha=1e-3,
        maxiters=maxiters_adam,
        tol=tol_outer,
        rho_multiplier=rho_multiplier,
        rho_update_max=rho_update_max,
        max_outer=max_outer,
        verbose=verbose
    )

    print("AL-Adam finished:")
    last_f_est = hist_al['f_est'][-1] if len(hist_al['f_est'])>0 else np.nan
    last_g = hist_al['g'][-1] if len(hist_al['g'])>0 else constraint_val(x_al)
    print(f"  last f_est={last_f_est:.6g}, g(x)={last_g:.6g}, λ={lam_al:.6g}")
    print()

    # NLopt-MMA baseline
    mma_results = []
    colors_mma = plt.cm.viridis(np.linspace(0.2, 0.8, len(sigma_mins)))
    for sigma_min, color in zip(sigma_mins, colors_mma):
        evals = 0.0
        mma_f_evals, mma_g_vals, mma_f_vals = [], [], []
        def f_and_grad_mma(x, grad):
            nonlocal evals, mma_f_evals, mma_g_vals, mma_f_vals
            xi = sample_xi()
            p = A.dot(x + xi)
            quad = float(np.linalg.norm(p)**2)
            val = float(np.exp(quad))
            if grad.size > 0:
                grad[:] = 2.0 * (A.T.dot(p)) * val
                evals += 1
                mma_f_evals.append(evals)
                mma_f_vals.append(val)
                mma_g_vals.append(constraint_val(x))
            else:
                evals += 0.5
            return val

        def cons_nl(x, grad):
            xi = sample_xi()
            if grad.size > 0: grad[:] = c
            return float(c.dot(x + xi) - b)

        optimizer_nlopt.add_inequality_constraint(cons_nl, 0.0)
        optimizer_nlopt.set_min_objective(f_and_grad_mma)
        optimizer_nlopt.set_maxeval(mma_maxeval)
        optimizer_nlopt.set_param("rho_init", 1.0)
        optimizer_nlopt.set_param("sigma_min", float(sigma_min))

        x_mma = optimizer_nlopt.optimize(x0.copy())
        res_code = optimizer_nlopt.last_optimize_result()
        print(f"nlopt σ={sigma_min}: res={res_code}, ||x||={np.linalg.norm(x_mma):.3g}, g(x)={constraint_val(x_mma):.3g}")
        mma_results.append((sigma_min, color, mma_f_evals, mma_g_vals, mma_f_vals))

    # --- Run custom CCSA (if provided) ---
    colors_ccsa = plt.cm.inferno(np.linspace(0.2, 0.8, len(sigma_mins)))
    ccsa_results = []

    oracle = make_noisy_f_and_grad(A, sample_xi)
    if optimizer is not None:
        optimizer.fun = oracle
        optimizer.g = constraint_val
        optimizer.dg = lambda xx: np.atleast_2d(c)
        optimizer.x0 = x0.copy()
        optimizer.params = x0.copy(),

        for sigma_min, color in zip(sigma_mins, colors_ccsa):
            try:
                optimizer.reset()
            except Exception:
                pass
            try:
                optimizer.sigma_params.sigma_min = sigma_min
            except Exception:
                pass

            metrics = None
            all_x = []
            for out in range(min(mma_maxeval, 1000)):
                f_b, g_b, metrics = optimizer.step()
                all_x.append(metrics["x_history"][-1])

            try:
                optimizer.summarize_diagnostics()
            except Exception:
                pass

            all_f = [oracle(xi, grad=None) for xi in all_x]

            x_hist = np.asarray(metrics.get("x_history", []))
            cum_we_hist = np.asarray(metrics.get("cumulative_weighted_evals_history", []), dtype=float)
            if x_hist.ndim == 1 and x_hist.size>0:
                x_hist = x_hist.reshape((1, -1))
            if x_hist.shape[0] > 0:
                f_expected_at_xhist = [expected_f(xi) for xi in x_hist]
                f_stoch_at_xhist = [oracle(xi, grad=None) for xi in x_hist]
                g_at_xhist = [constraint_val(xi) for xi in x_hist]
            else:
                f_expected_at_xhist = []
                f_stoch_at_xhist = []
                g_at_xhist = []
            if cum_we_hist.size == 0:
                cum_we_hist = np.arange(1, len(f_expected_at_xhist)+1, dtype=float)

            ccsa_results.append({
                "sigma_min": sigma_min,
                "color": color,
                "metrics": metrics,
                "x_hist": x_hist,
                "cum_we_hist": cum_we_hist,
                "f_expected_at_xhist": np.array(f_expected_at_xhist),
                "f_stoch_at_xhist": np.array(f_stoch_at_xhist),
                "g_at_xhist": np.array(g_at_xhist)
            })
            print(f"CCSA σ={sigma_min}: cumulative_wval={metrics.get('cumulative_wval', np.nan):.6g}, weighted_evals={metrics.get('weighted_evals', np.nan)}, x_history_len={len(x_hist)}")

    # --- Run baseline CSSCA optimizer (single constant rho schedule) ---
    cssca_tau_objs_list = cssca_tau_obj if isinstance(cssca_tau_obj, (list, tuple, np.ndarray)) else [cssca_tau_obj]
    cssca_tau_cons_list = cssca_tau_cons if isinstance(cssca_tau_cons, (list, tuple, np.ndarray)) else [cssca_tau_cons]
    if cssca_samples_per_iter is None:
        cssca_samples_list = [1]
    else:
        cssca_samples_list = cssca_samples_per_iter if isinstance(cssca_samples_per_iter, (list, tuple, np.ndarray)) else [cssca_samples_per_iter]

    L = max(len(cssca_tau_objs_list), len(cssca_tau_cons_list), len(cssca_samples_list))
    def _broadcast(lst, L):
        if len(lst) == L:
            return list(lst)
        if len(lst) == 1:
            return list(lst) * L
        raise ValueError("CSSCA parameter lists must be length 1 or equal to each other")

    cssca_tau_objs_list = _broadcast(list(cssca_tau_objs_list), L)
    cssca_tau_cons_list = _broadcast(list(cssca_tau_cons_list), L)
    cssca_samples_list = _broadcast(list(cssca_samples_list), L)

    cssca_runs = []
    for tau_o, tau_c, samples_p in zip(cssca_tau_objs_list, cssca_tau_cons_list, cssca_samples_list):
        
        cssca_opt = CSSCAOptimizer(params=x0.copy(),
                                    fun=oracle,
                                    g=lambda xx: float(np.dot(c, xx) - b),
                                    dg=lambda xx: np.atleast_2d(c),
                                    x0=x0.copy(), rho_t_schedule=float(rho), gamma_t_schedule=1.0,
                                    tau_obj=float(tau_o), tau_cons=float(tau_c), samples_per_iter=1.0)

        cssca_f_hist = []
        cssca_cons_hist = []
        for t in range(mma_maxeval):
            x_cssca, f_cssca, cons_cssca = cssca_opt.step()
            cssca_f_hist.append(f_cssca)
            cssca_cons_hist.append(cons_cssca.copy() if hasattr(cons_cssca, 'copy') else np.atleast_1d(cons_cssca))

        cssca_cons_arr = np.vstack(cssca_cons_hist) if len(cssca_cons_hist) > 0 and np.asarray(cssca_cons_hist).ndim == 2 else np.asarray(cssca_cons_hist)
        print(f"CSSCA (tau_obj={tau_o}, tau_cons={tau_c}, rho={rho}): last f={cssca_f_hist[-1]:.6g}, last g[0]={cssca_cons_arr[-1,0] if cssca_cons_arr.size>0 else np.nan:.6g}")
        cssca_runs.append({"tau_obj": tau_o, "tau_cons": tau_c, "f_hist": cssca_f_hist, "cons_arr": cssca_cons_arr})
        
    # --- Run CCSA-quad baseline (if provided) ---
    ccsa_quad_results = []
    oracle = make_noisy_f_and_grad(A, sample_xi)
    if optimizer_quad is not None:
        optimizer_quad.fun = oracle
        optimizer_quad.g = constraint_val
        optimizer_quad.dg = lambda xx: np.atleast_2d(c)
        optimizer_quad.x0 = x0.copy()
        optimizer_quad.params = x0.copy(),
        for sigma_min, color in zip(sigma_mins, colors_ccsa):
            try:
                optimizer_quad.reset()
            except Exception:
                pass

            metrics = None
            all_x = []
            for out in range(min(mma_maxeval, 1000)):
                f_b, g_b, metrics = optimizer_quad.step()
                all_x.append(metrics["x_history"][-1])

            
            all_f = [oracle(xi, grad=None) for xi in all_x]

            x_hist = np.asarray(metrics.get("x_history", []))
            cum_we_hist = np.asarray(metrics.get("cumulative_weighted_evals_history", []), dtype=float)
            if x_hist.ndim == 1 and x_hist.size>0:
                x_hist = x_hist.reshape((1, -1))
            if x_hist.shape[0] > 0:
                f_expected_at_xhist = [expected_f(xi) for xi in x_hist]
                f_stoch_at_xhist = [oracle(xi, grad=None) for xi in x_hist]
                g_at_xhist = [constraint_val(xi) for xi in x_hist]
            else:
                f_expected_at_xhist = []
                f_stoch_at_xhist = []
                g_at_xhist = []
            if cum_we_hist.size == 0:
                cum_we_hist = np.arange(1, len(f_expected_at_xhist)+1, dtype=float)

            ccsa_quad_results.append({
                "sigma_min": sigma_min,
                "color": color,
                "metrics": metrics,
                "x_hist": x_hist,
                "cum_we_hist": cum_we_hist,
                "f_expected_at_xhist": np.array(f_expected_at_xhist),
                "f_stoch_at_xhist": np.array(f_stoch_at_xhist),
                "g_at_xhist": np.array(g_at_xhist)
            })
            print(f"CCSA σ={sigma_min}: cumulative_wval={metrics.get('cumulative_wval', np.nan):.6g}, weighted_evals={metrics.get('weighted_evals', np.nan)}, x_history_len={len(x_hist)}")

    # --- ensure first plotted value reflects x0 for all traces (same helpers as original) ---
    init_f_expected = expected_f(x0)
    init_f_stoch = f_stoch_estimate(x0) if 'f_stoch_estimate' in locals() else expected_f(x0)
    init_g_nom = float(np.dot(c, x0) - b)

    def _prepend_ccsa_entry(entry):
        cum = np.asarray(entry.get('cum_we_hist', []), dtype=float)
        fexp = np.asarray(entry.get('f_expected_at_xhist', []), dtype=float)
        fst = np.asarray(entry.get('f_stoch_at_xhist', []), dtype=float)
        g = np.asarray(entry.get('g_at_xhist', []), dtype=float)

        if cum.size == 0:
            new_cum = np.array([1.0])
        else:
            new_cum = np.concatenate(([1.0], cum + 1.0))

        new_fexp = np.concatenate(([init_f_expected], fexp)) if fexp.size > 0 else np.array([init_f_expected])
        new_fst = np.concatenate(([init_f_stoch], fst)) if fst.size > 0 else np.array([init_f_stoch])
        new_g = np.concatenate(([init_g_nom], g)) if g.size > 0 else np.array([init_g_nom])

        entry['cum_we_hist'] = new_cum
        entry['f_expected_at_xhist'] = new_fexp
        entry['f_stoch_at_xhist'] = new_fst
        entry['g_at_xhist'] = new_g

    def _prepend_cssca_run(run):
        fh = np.asarray(run.get('f_hist', []), dtype=float)
        if fh.size == 0:
            run['f_hist'] = np.array([init_f_expected])
        else:
            run['f_hist'] = np.concatenate(([init_f_expected], fh))

        carr = run.get('cons_arr', np.array([]))
        carr = np.asarray(carr)
        if carr.size == 0:
            run['cons_arr'] = np.array([init_g_nom])
        else:
            if carr.ndim == 1:
                run['cons_arr'] = np.concatenate(([init_g_nom], carr.astype(float)))
            elif carr.ndim == 2:
                mcols = carr.shape[1]
                first_row = np.full((1, mcols), float(init_g_nom), dtype=float)
                run['cons_arr'] = np.vstack((first_row, carr.astype(float)))
            else:
                run['cons_arr'] = np.concatenate(([init_g_nom], carr.ravel().astype(float)))

    for cr in ccsa_results:
        try:
            _prepend_ccsa_entry(cr)
        except Exception:
            pass
    for cr in ccsa_quad_results:
        try:
            _prepend_ccsa_entry(cr)
        except Exception:
            pass
    if 'cssca_runs' in locals():
        for run in cssca_runs:
            try:
                _prepend_cssca_run(run)
            except Exception:
                pass

    # --- plotting (similar to original) ---
    plt.figure(figsize=(12,5))
    patch_half = max(1e-12, 0.5 * (noise**2) * np.sum(m))

    ax1 = plt.subplot(1,2,1)
    ax1.plot(hist_al['iter'], hist_al['f_est'], '-', color='black', label='AL-Adam')

    for cr in ccsa_results:
        col = cr.get('color', 'tab:orange')
        if ccsa_plot_expected:
            ax1.plot(cr['cum_we_hist'], cr['f_expected_at_xhist'], linestyle='-', linewidth=2.25,
                     color=col, label=f'custom non-cons σ={cr["sigma_min"]}')
        else:
            ax1.plot(cr['cum_we_hist'], cr['f_stoch_at_xhist'], linestyle='-', linewidth=1.5,
                     marker='o', markersize=4, color=col, label=f'custom non-cons σ={cr["sigma_min"]}')
    for cr in ccsa_quad_results:
        col = cr.get('color', 'tab:green')
        if ccsa_plot_expected:
            ax1.plot(cr['cum_we_hist'], cr['f_expected_at_xhist'], linestyle='-', linewidth=2.25,
                     color=col, label=f'custom cons σ={cr["sigma_min"]}')
        else:
            ax1.plot(cr['cum_we_hist'], cr['f_stoch_at_xhist'], linestyle='-', linewidth=1.5,
                     marker='s', markersize=4, color=col, label=f'custom cons σ={cr["sigma_min"]}')

    try:
        if 'cssca_runs' in locals() and len(cssca_runs) > 0:
            colors_css = plt.cm.tab10(np.arange(len(cssca_runs)) % 10)
            for idx, run in enumerate(cssca_runs):
                fh = run.get('f_hist', [])
                if len(fh) > 0:
                    ax1.plot(np.arange(1, len(fh)+1), fh, linestyle='-', linewidth=2.0,
                             color=colors_css[idx], label=f"CSSCA τo={run['tau_obj']}, τc={run['tau_cons']}")
    except NameError:
        print("Error!!!!")
        pass

    ax1.axhspan(val_star - patch_half, val_star + patch_half, alpha=0.18, facecolor='tab:orange')
    constrained_patch = mpatches.Patch(facecolor='tab:orange', alpha=0.18, label='stochastic constrained (± noise const)')
    handles, labels = ax1.get_legend_handles_labels()
    unique = OrderedDict()
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    unique[constrained_patch.get_label()] = constrained_patch
    #ax1.legend(list(unique.values()), list(unique.keys()), loc='best', fontsize='small')

    ax1.axhline(val_star, color='k', linestyle='--', linewidth=1.0)
    ax1.axhline(val_uncon, color='gray', linestyle=':', linewidth=1.0)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.grid(True); ax1.set_xlabel('iter/evals'); ax1.set_ylabel('stochastic f(x)'); ax1.set_title('Objective')

    ax2 = plt.subplot(1,2,2)
    ax2.plot(hist_al['iter'], hist_al['g'], '-', color='black', label='AL-Adam g(x)')
    for cr in ccsa_results:
        col = cr.get('color', 'tab:orange')
        ax2.plot(cr['cum_we_hist'], cr['g_at_xhist'], linestyle='-', color=col, linewidth=1.8, label=f'CCSA σ={cr["sigma_min"]}')
    for cr in ccsa_quad_results:
        col = cr.get('color', 'tab:green')
        ax2.plot(cr['cum_we_hist'], cr['g_at_xhist'], linestyle='-', color=col, linewidth=1.8, label=f'CCSA QUAD σ={cr["sigma_min"]}')

    # CSSCA constraint traces (for each run)
    try:
        if 'cssca_runs' in locals() and len(cssca_runs) > 0:
            colors_css = plt.cm.tab10(np.arange(len(cssca_runs)) % 10)
            for idx, run in enumerate(cssca_runs):
                carr = run.get('cons_arr', np.array([]))
                if carr.size == 0:
                    continue
                if carr.ndim == 2:
                    ax2.plot(np.arange(1, carr.shape[0]+1), carr[:, 0], linestyle='-', color=colors_css[idx], linewidth=2.0,
                             label=f"CSSCA g[0] τo={run['tau_obj']}, τc={run['tau_cons']}")
                else:
                    ax2.plot(np.arange(1, len(carr)+1), carr, linestyle='-', color=colors_css[idx], linewidth=2.0,
                             label=f"CSSCA g τo={run['tau_obj']}, τc={run['tau_cons']}")
    except NameError:
        pass

    ax2.axhline(0, color='k', linestyle='--', label='feasibility')
    ax2.axhline(g_uncon, color='gray', linestyle=':', label='unconstrained g')
    ax2.axhline(expected_g(x_star), color='k', linestyle='--', linewidth=1.0, label='constrained g')
    ax2.set_xscale('log'); ax2.grid(True); ax2.set_xlabel('iter/evals'); ax2.set_ylabel('g(x)'); ax2.set_title('Constraint violation')
    ax2.legend(loc='best', fontsize='small')

    plt.tight_layout()
    plt.savefig("expquad_diag_experiment.png", dpi=300)
    plt.show()

    # return structure similar to your original function
    return {
        "c": c, "b": b,
        "x_star": x_star, "val_star": val_star,
        "x_al": x_al, "lam_al": lam_al,
        "mma_results": mma_results,
        "ccsa_results": ccsa_results,
        "val_uncon": val_uncon, "g_uncon": g_uncon,
        "hist_al": hist_al
    }


# experiments_radial.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict

# Assumed available in your environment (same names as in your repo)
# from baselines.adam_al import adam_augmented_lagrangian
# from ccsa.optimizer import CCSAOptimizer, CSSCAOptimizer
# from utils import randcond, make_red_colors

def stoch_radial_full_run(
    optimizer,
    optimizer_quad,
    optimizer_nlopt,
    noise: float = 0.2,
    cond: int = 1,
    c: np.ndarray = None,
    b: float = 0.0,
    x0: np.ndarray = None,
    sigma_mins: list = [0.0, 0.01, 0.1],
    seed: int = 0,
    maxiters_adam: int = 20000,
    mma_maxeval: int = 2000,
    active_constraint = True,
    init_feasible: bool = False,
    rho: float = 1.0,
    lambda0: float = 0.0,
    tol_outer: float = 1e-6,
    rho_multiplier: float = 2.0,
    rho_update_max: int = 6,
    max_outer = 8,
    verbose: bool = True,
    ccsa_plot_expected: bool = True,
    cssca_tau_obj: float = 1.0, cssca_tau_cons: float = 1.0, cssca_samples_per_iter=None,
    radial_curvature: bool = True,
    m0: float = 0.01,
    alpha: float = 0.02
):
    """
    Runs a full experiment with:
      - AL-Adam (augmented Lagrangian using adam)
      - custom CCSA optimizer (stochastic)
      - CCSA quadratic-surrogate optimizer (optimizer_quad)
      - CSSCA baseline(s) (tau broadcasting, as in your original code)
    Objective: radial quadratic f(y) = m(r) * r^2 where r = ||y|| (y = x + xi)
    If radial_curvature=False, uses fixed diagonal quadratic with exact expectation.
    """
    rng = np.random.RandomState(seed)

    # sanity
    if x0 is None:
        raise ValueError("x0 must be provided and have length n")

    n = len(x0)

    # --- curvature definitions ---
    if radial_curvature:
        # m(r) = m0 + alpha * r  (you can invert sign of alpha to have other shapes)
        def m_of_r(r):
            return m0 + alpha * r
        # no per-axis diagonal M
        m_diag = None
        print("Using RADIAL curvature: m(r) = m0 + alpha * r")
    else:
        # fixed diagonal curvatures (safe range)
        m_min = 0.005
        m_max = 0.05 * max(1, cond)
        m_diag = np.linspace(m_min, m_max, n)
        m_of_r = None
        print("Using FIXED DIAGONAL curvature (exact expectation)")

    # --- expected_f: exact for fixed diag, second-order for radial ---
    def expected_f(x):
        sigma2 = noise**2
        if radial_curvature:
            r = np.linalg.norm(x)
            # f(x) = m(r) * r^2 = m0 r^2 + alpha r^3
            f_val = (m0 * r**2) + (alpha * r**3)
            # Hessian trace:
            # for m0 r^2 -> trace = 2*m0 * n
            # for alpha r^3 -> trace = 3*alpha * r * (n+1)  (derived in analysis)
            trace_H = 2.0 * m0 * n + 3.0 * alpha * r * (n + 1.0)
            return float(f_val + 0.5 * sigma2 * trace_H)
        else:
            # exact: E[ x^T M x + xi^T M xi ] = x^T M x + sigma2 * trace(M)
            q = np.sum(m_diag * (x**2))
            return float(q + sigma2 * np.sum(m_diag))

    # --- stochastic constraint (single-sample) ---
    def sample_xi():
        return rng.randn(n) * noise

    def constraint_val(x):
        xi = sample_xi()
        return float(np.dot(c, x + xi) - b)

    # --- stochastic oracle that returns consistent value+grad when requested ---
    def make_noisy_f_and_grad():
        def f_and_grad(x, grad=None):
            xi = sample_xi()
            y = x + xi
            r = np.linalg.norm(y)
            if radial_curvature:
                # f(y) = m0 * r^2 + alpha * r^3
                val = float(m0 * (r**2) + alpha * (r**3))
                if grad is True:
                    # return value and grad
                    grad_val = (2.0 * m0 + 3.0 * alpha * r) * y
                    return val, grad_val
                if isinstance(grad, np.ndarray):
                    grad[:] = (2.0 * m0 + 3.0 * alpha * r) * y
                    return val
                return val
            else:
                # fixed diagonal quadratic
                val_quad = float(np.sum(m_diag * (y**2)))
                if grad is True:
                    grad_val = 2.0 * m_diag * y
                    return val_quad, grad_val
                if isinstance(grad, np.ndarray):
                    grad[:] = 2.0 * m_diag * y
                    return val_quad
                return val_quad
        return f_and_grad

    oracle = make_noisy_f_and_grad()

    # --- ADAPTORS for AL-Adam (keeps RNG consistent) ---
    oracle_for_adam = make_noisy_f_and_grad()
    def f_for_adam(x):
        return oracle_for_adam(x, grad=None)
    def fgrad_for_adam(x):
        g = np.zeros_like(x)
        oracle_for_adam(x, grad=g)
        return g

    # --- Print initial constraint ---
    print(f"Initial constraint g(x0)={constraint_val(x0):.6g} (init_feasible={init_feasible})")

    # --- Run AL-Adam (requires your adam_augmented_lagrangian implementation) ---
    # Note: this will use the stochastic oracle adaptors and keep RNG sequence consistent.
    x_al, lam_al, hist_al = adam_augmented_lagrangian(
        fgrad=fgrad_for_adam,
        x0=x0.copy(),
        g=constraint_val,
        dg=lambda xx: c,
        rho=rho,
        lambda0=lambda0,
        f_stoch_estimate=f_for_adam,
        alpha=1e-3,
        maxiters=maxiters_adam,
        tol=tol_outer,
        rho_multiplier=rho_multiplier,
        rho_update_max=rho_update_max,
        max_outer=max_outer,
        verbose=verbose
    )

    print("AL-Adam finished:")
    last_f_est = hist_al['f_est'][-1] if len(hist_al['f_est'])>0 else np.nan
    last_g = hist_al['g'][-1] if len(hist_al['g'])>0 else constraint_val(x_al)
    print(f"  last f_est={last_f_est:.6g}, g(x)={last_g:.6g}, λ={lam_al:.6g}")

    # --- NLopt/MMA not mandatory but keep support (user passed optimizer_nlopt) ---
    # We'll not rely on NLopt in the core runs unless provided; if provided, keep original pattern.
    mma_results = []

    if optimizer_nlopt is not None:
        colors_mma = plt.cm.viridis(np.linspace(0.2, 0.8, len(sigma_mins)))
        for sigma_min, color in zip(sigma_mins, colors_mma):
            evals = 0.0
            mma_f_evals, mma_g_vals, mma_f_vals = [], [], []
            def f_and_grad_mma(x, grad):
                nonlocal evals, mma_f_evals, mma_g_vals, mma_f_vals
                xi = sample_xi()
                y = x + xi
                # evaluate value and gradient consistent with oracle
                if radial_curvature:
                    r = np.linalg.norm(y)
                    val = float(m0 * r*r + alpha * r**3)
                    if grad.size > 0:
                        grad[:] = (2.0 * m0 + 3.0 * alpha * r) * y
                        evals += 1
                        mma_f_evals.append(evals); mma_f_vals.append(val); mma_g_vals.append(constraint_val(x))
                    else:
                        evals += 0.5
                    return val
                else:
                    val = float(np.sum(m_diag * (y**2)))
                    if grad.size > 0:
                        grad[:] = 2.0 * m_diag * y
                        evals += 1
                        mma_f_evals.append(evals); mma_f_vals.append(val); mma_g_vals.append(constraint_val(x))
                    else:
                        evals += 0.5
                    return val

            def cons_nl(x, grad):
                xi = sample_xi()
                if grad.size > 0: grad[:] = c
                return float(c.dot(x + xi) - b)

            try:
                optimizer_nlopt.add_inequality_constraint(cons_nl, 0.0)
            except Exception:
                pass
            try:
                optimizer_nlopt.set_min_objective(f_and_grad_mma)
                optimizer_nlopt.set_maxeval(mma_maxeval)
                optimizer_nlopt.set_param("rho_init", 1.0)
                optimizer_nlopt.set_param("sigma_min", float(sigma_min))
                x_mma = optimizer_nlopt.optimize(x0.copy())
                res_code = optimizer_nlopt.last_optimize_result()
                print(f"NLopt-MMA run sigma_min={sigma_min}: res={res_code}, ||x||={np.linalg.norm(x_mma):.3g}, g(x)={constraint_val(x_mma):.3g}")
                mma_results.append((sigma_min, None, mma_f_evals, mma_g_vals, mma_f_vals))
            except Exception as e:
                print(f"NLopt failed: {e}")
                mma_results.append((sigma_min, None, [], [], []))

    # --- Run custom CCSA (stochastic) and CCSA-quad (quadratic surrogate) ---
    colors_ccsa = plt.cm.inferno(np.linspace(0.2, 0.8, len(sigma_mins)))
    ccsa_results = []
    ccsa_quad_results = []

    # set oracle and constraint on optimizers
    if optimizer is not None:
        optimizer.fun = oracle
        optimizer.g = constraint_val
        optimizer.dg = lambda xx: np.atleast_2d(c)
        optimizer.x0 = x0.copy()
        optimizer.params = x0.copy(),

        for sigma_min, color in zip(sigma_mins, colors_ccsa):
            # try to set sigma_min param if available
            try:
                optimizer.reset()
            except Exception:
                pass
            try:
                optimizer.sigma_params.sigma_min = sigma_min
            except Exception:
                pass

            all_x = []
            metrics = None
            # run inner loop (use mma_maxeval or smaller to keep runtime reasonable)
            for out in range(min(mma_maxeval, 1000)):
                f_b, g_b, metrics = optimizer.step()
                # metrics expected to have "x_history"
                if metrics is None or "x_history" not in metrics:
                    break
                all_x.append(metrics["x_history"][-1])

            all_f_expected = [expected_f(xi) for xi in all_x]
            g_at_xhist = [constraint_val(xi) for xi in all_x] if len(all_x)>0 else []

            x_hist = np.asarray(metrics.get("x_history", [])) if metrics is not None else np.array([])
            cum_we_hist = np.asarray(metrics.get("cumulative_weighted_evals_history", []), dtype=float) if metrics is not None else np.array([])

            if x_hist.ndim == 1 and x_hist.size>0:
                x_hist = x_hist.reshape((1, -1))

            if cum_we_hist.size == 0 and len(all_f_expected)>0:
                cum_we_hist = np.arange(1, len(all_f_expected)+1, dtype=float)

            ccsa_results.append({
                "sigma_min": sigma_min,
                "color": color,
                "metrics": metrics,
                "x_hist": x_hist,
                "cum_we_hist": cum_we_hist,
                "f_expected_at_xhist": np.array(all_f_expected),
                "f_stoch_at_xhist": np.array([oracle(xi, grad=None) for xi in all_x]) if len(all_x)>0 else np.array([]),
                "g_at_xhist": np.array(g_at_xhist)
            })
            print(f"CCSA (stochastic) σ={sigma_min}: x_history_len={len(all_x)}")

    if optimizer_quad is not None:
        optimizer_quad.fun = oracle
        optimizer_quad.g = constraint_val
        optimizer_quad.dg = lambda xx: np.atleast_2d(c)
        optimizer_quad.x0 = x0.copy()
        optimizer_quad.params = x0.copy(),

        for sigma_min, color in zip(sigma_mins, colors_ccsa):
            try:
                optimizer_quad.reset()
            except Exception:
                pass
            try:
                optimizer_quad.sigma_params.sigma_min = sigma_min
            except Exception:
                pass

            all_x = []
            metrics = None
            for out in range(min(mma_maxeval, 1000)):
                f_b, g_b, metrics = optimizer_quad.step()
                if metrics is None or "x_history" not in metrics:
                    break
                all_x.append(metrics["x_history"][-1])

            all_f_expected = [expected_f(xi) for xi in all_x]
            g_at_xhist = [constraint_val(xi) for xi in all_x] if len(all_x)>0 else []

            x_hist = np.asarray(metrics.get("x_history", [])) if metrics is not None else np.array([])
            cum_we_hist = np.asarray(metrics.get("cumulative_weighted_evals_history", []), dtype=float) if metrics is not None else np.array([])

            if x_hist.ndim == 1 and x_hist.size>0:
                x_hist = x_hist.reshape((1, -1))

            if cum_we_hist.size == 0 and len(all_f_expected)>0:
                cum_we_hist = np.arange(1, len(all_f_expected)+1, dtype=float)

            ccsa_quad_results.append({
                "sigma_min": sigma_min,
                "color": color,
                "metrics": metrics,
                "x_hist": x_hist,
                "cum_we_hist": cum_we_hist,
                "f_expected_at_xhist": np.array(all_f_expected),
                "f_stoch_at_xhist": np.array([oracle(xi, grad=None) for xi in all_x]) if len(all_x)>0 else np.array([]),
                "g_at_xhist": np.array(g_at_xhist)
            })
            print(f"CCSA-QUAD σ={sigma_min}: x_history_len={len(all_x)}")

    # --- Run CSSCA baseline (broadcasting tau lists exactly like original) ---
    cssca_tau_objs_list = cssca_tau_obj if isinstance(cssca_tau_obj, (list, tuple, np.ndarray)) else [cssca_tau_obj]
    cssca_tau_cons_list = cssca_tau_cons if isinstance(cssca_tau_cons, (list, tuple, np.ndarray)) else [cssca_tau_cons]
    if cssca_samples_per_iter is None:
        cssca_samples_list = [1]
    else:
        cssca_samples_list = cssca_samples_per_iter if isinstance(cssca_samples_per_iter, (list, tuple, np.ndarray)) else [cssca_samples_per_iter]

    L = max(len(cssca_tau_objs_list), len(cssca_tau_cons_list), len(cssca_samples_list))
    def _broadcast(lst, L):
        if len(lst) == L:
            return list(lst)
        if len(lst) == 1:
            return list(lst) * L
        raise ValueError("CSSCA parameter lists must be length 1 or equal to each other")

    cssca_tau_objs_list = _broadcast(list(cssca_tau_objs_list), L)
    cssca_tau_cons_list = _broadcast(list(cssca_tau_cons_list), L)
    cssca_samples_list = _broadcast(list(cssca_samples_list), L)

    cssca_runs = []
    for tau_o, tau_c, samples_p in zip(cssca_tau_objs_list, cssca_tau_cons_list, cssca_samples_list):
        try:
            
            cssca_opt = CSSCAOptimizer(
                params=x0.copy(),
                fun=oracle,
                g=lambda xx: float(np.dot(c, xx) - b),
                dg=lambda xx: np.atleast_2d(c),
                x0=x0.copy(),
                rho_t_schedule=float(rho),
                gamma_t_schedule=1.0,
                tau_obj=float(tau_o),
                tau_cons=float(tau_c),
                samples_per_iter=samples_p
            )

            cssca_f_hist = []
            cssca_cons_hist = []
            for t in range(mma_maxeval):
                x_cssca, f_cssca, cons_cssca = cssca_opt.step()
                cssca_f_hist.append(f_cssca)
                cssca_cons_hist.append(cons_cssca.copy() if hasattr(cons_cssca, 'copy') else np.atleast_1d(cons_cssca))

            cssca_cons_arr = np.vstack(cssca_cons_hist) if len(cssca_cons_hist) > 0 and np.asarray(cssca_cons_hist).ndim == 2 else np.asarray(cssca_cons_hist)
            print(f"CSSCA (tau_obj={tau_o}, tau_cons={tau_c}, rho={rho}): last f={cssca_f_hist[-1]:.6g}, last g[0]={cssca_cons_arr[-1,0] if cssca_cons_arr.size>0 else np.nan:.6g}")
            cssca_runs.append({"tau_obj": tau_o, "tau_cons": tau_c, "f_hist": cssca_f_hist, "cons_arr": cssca_cons_arr})
        except Exception as e:
            print(f"Failed to run CSSCA optimizer (tau_obj={tau_o}, tau_cons={tau_c}): {e}")
            cssca_runs.append({"tau_obj": tau_o, "tau_cons": tau_c, "f_hist": [], "cons_arr": np.array([])})

    # --- Prepend x0 values like original so plots begin with initial point ---
    init_f_expected = expected_f(x0)
    init_f_stoch = oracle(x0, grad=None) if radial_curvature else expected_f(x0)
    init_g_nom = float(np.dot(c, x0) - b)

    def _prepend_ccsa_entry(entry):
        cum = np.asarray(entry.get('cum_we_hist', []), dtype=float)
        fexp = np.asarray(entry.get('f_expected_at_xhist', []), dtype=float)
        fst = np.asarray(entry.get('f_stoch_at_xhist', []), dtype=float)
        g = np.asarray(entry.get('g_at_xhist', []), dtype=float)

        if cum.size == 0:
            new_cum = np.array([1.0])
        else:
            new_cum = np.concatenate(([1.0], cum + 1.0))

        new_fexp = np.concatenate(([init_f_expected], fexp)) if fexp.size > 0 else np.array([init_f_expected])
        new_fst = np.concatenate(([init_f_stoch], fst)) if fst.size > 0 else np.array([init_f_stoch])
        new_g = np.concatenate(([init_g_nom], g)) if g.size > 0 else np.array([init_g_nom])

        entry['cum_we_hist'] = new_cum
        entry['f_expected_at_xhist'] = new_fexp
        entry['f_stoch_at_xhist'] = new_fst
        entry['g_at_xhist'] = new_g

    def _prepend_cssca_run(run):
        fh = np.asarray(run.get('f_hist', []), dtype=float)
        if fh.size == 0:
            run['f_hist'] = np.array([init_f_expected])
        else:
            run['f_hist'] = np.concatenate(([init_f_expected], fh))

        carr = run.get('cons_arr', np.array([]))
        carr = np.asarray(carr)
        if carr.size == 0:
            run['cons_arr'] = np.array([init_g_nom])
        else:
            if carr.ndim == 1:
                run['cons_arr'] = np.concatenate(([init_g_nom], carr.astype(float)))
            elif carr.ndim == 2:
                mcols = carr.shape[1]
                first_row = np.full((1, mcols), float(init_g_nom), dtype=float)
                run['cons_arr'] = np.vstack((first_row, carr.astype(float)))
            else:
                run['cons_arr'] = np.concatenate(([init_g_nom], carr.ravel().astype(float)))

    for cr in ccsa_results:
        try:
            _prepend_ccsa_entry(cr)
        except Exception:
            pass
    for cr in ccsa_quad_results:
        try:
            _prepend_ccsa_entry(cr)
        except Exception:
            pass
    if 'cssca_runs' in locals():
        for run in cssca_runs:
            try:
                _prepend_cssca_run(run)
            except Exception:
                pass
    
    # --- fix any length mismatches for plotting ---
    def _fix_length_mismatch(entry):
        fexp = entry.get('f_expected_at_xhist', [])
        fst = entry.get('f_stoch_at_xhist', [])
        g = entry.get('g_at_xhist', [])
        cum = entry.get('cum_we_hist', [])

        min_len = min(len(fexp), len(fst), len(g), len(cum))
        entry['f_expected_at_xhist'] = np.asarray(fexp[:min_len])
        entry['f_stoch_at_xhist'] = np.asarray(fst[:min_len])
        entry['g_at_xhist'] = np.asarray(g[:min_len])
        entry['cum_we_hist'] = np.asarray(cum[:min_len])

    for cr in ccsa_results:
        try: _fix_length_mismatch(cr)
        except Exception: pass

    for cr in ccsa_quad_results:
        try: _fix_length_mismatch(cr)
        except Exception: pass

    if 'cssca_runs' in locals():
        for run in cssca_runs:
            try:
                # CSSCA: f_hist and cons_arr
                min_len = min(len(run.get('f_hist', [])), run.get('cons_arr', np.array([])).shape[0] if run.get('cons_arr', None) is not None else 0)
                run['f_hist'] = np.asarray(run.get('f_hist', [])[:min_len])
                if run.get('cons_arr', None) is not None and min_len>0:
                    run['cons_arr'] = np.asarray(run.get('cons_arr', [])[:min_len])
            except Exception:
                pass
    val_uncon = expected_f(np.zeros_like(x0))
    g_uncon = float(np.dot(c, np.zeros_like(x0)) - b)


    # --- Plot results (objective + constraint) similar to your original function ---
    plt.figure(figsize=(12,5))
    patch_half = max(1e-12, 0.5 * (noise**2) * (np.sum(m_diag) if not radial_curvature else (n * m0 + alpha * np.linalg.norm(x0))))

    ax1 = plt.subplot(1,2,1)
    try:
        ax1.plot(hist_al['iter'], hist_al['f_est'], '-', color='black', label='AL-Adam')
    except Exception:
        pass

    for cr in ccsa_results:
        col = cr.get('color', 'tab:orange')
        if ccsa_plot_expected:
            ax1.plot(cr['cum_we_hist'], cr['f_expected_at_xhist'], linestyle='-', linewidth=2.25, color=col, label=f'CCSA σ={cr["sigma_min"]}')
        else:
            ax1.plot(cr['cum_we_hist'], cr['f_stoch_at_xhist'], linestyle='-', marker='o', markersize=4, color=col, label=f'CCSA σ={cr["sigma_min"]}')
    for cr in ccsa_quad_results:
        col = cr.get('color', 'tab:green')
        if ccsa_plot_expected:
            ax1.plot(cr['cum_we_hist'], cr['f_expected_at_xhist'], linestyle='-', linewidth=2.25, color=col, label=f'CCSA-QUAD σ={cr["sigma_min"]}')
        else:
            ax1.plot(cr['cum_we_hist'], cr['f_stoch_at_xhist'], linestyle='-', marker='s', markersize=4, color=col, label=f'CCSA-QUAD σ={cr["sigma_min"]}')

    # CSSCA runs plotting (if any)
   
    if 'cssca_runs' in locals() and len(cssca_runs) > 0:
        colors_css = plt.cm.tab10(np.arange(len(cssca_runs)) % 10)
        for idx, run in enumerate(cssca_runs):
            fh = run.get('f_hist', [])
            if len(fh) > 0:
                ax1.plot(np.arange(1, len(fh)+1), fh, linestyle='-', linewidth=2.0,
                            color=colors_css[idx], label=f"CSSCA τo={run['tau_obj']}, τc={run['tau_cons']}")
   
    # compute val_star safely (numeric) for plotting
    if 'x_star' in locals():
        val_star = expected_f(x_star)
        expected_g_val = float(np.dot(c, x_star) - b)
    else:
        val_star = init_f_expected
        expected_g_val = g_uncon

    val_uncon = expected_f(np.zeros(n))

    # objective patch region and lines
    ax1.axhspan(val_star - patch_half, val_star + patch_half, alpha=0.18, facecolor='tab:orange')
    ax1.axhline(val_star, color='k', linestyle='--', linewidth=1.0)
    ax1.axhline(val_uncon, color='gray', linestyle=':', linewidth=1.0)

    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.grid(True); ax1.set_xlabel('iter/evals'); ax1.set_ylabel('stochastic f(x)'); ax1.set_title('Objective')
    ax1.set_ylim(bottom=1e-12, top=1e12)

    ax2 = plt.subplot(1,2,2)
    try:
        ax2.plot(hist_al['iter'], hist_al['g'], '-', color='black', label='AL-Adam g(x)')
    except Exception:
        pass

    for cr in ccsa_results:
        col = cr.get('color', 'tab:orange')
        ax2.plot(cr['cum_we_hist'], cr['g_at_xhist'], linestyle='-', color=col, linewidth=1.8, label=f'CCSA σ={cr["sigma_min"]}')
    for cr in ccsa_quad_results:
        col = cr.get('color', 'tab:green')
        ax2.plot(cr['cum_we_hist'], cr['g_at_xhist'], linestyle='-', color=col, linewidth=1.8, label=f'CCSA-QUAD σ={cr["sigma_min"]}')

    try:
        if 'cssca_runs' in locals() and len(cssca_runs) > 0:
            colors_css = plt.cm.tab10(np.arange(len(cssca_runs)) % 10)
            for idx, run in enumerate(cssca_runs):
                carr = run.get('cons_arr', np.array([]))
                if carr.size == 0:
                    continue
                if carr.ndim == 2:
                    ax2.plot(np.arange(1, carr.shape[0]+1), carr[:, 0], linestyle='-', color=colors_css[idx], linewidth=2.0,
                             label=f"CSSCA g[0] τo={run['tau_obj']}, τc={run['tau_cons']}")
                else:
                    ax2.plot(np.arange(1, len(carr)+1), carr, linestyle='-', color=colors_css[idx], linewidth=2.0,
                             label=f"CSSCA g τo={run['tau_obj']}, τc={run['tau_cons']}")
    except Exception:
        pass

   

    # constraint horizontal lines: pass numeric values, not a function
    ax2.axhline(0, color='k', linestyle='--', label='feasibility')
    ax2.axhline(g_uncon, color='gray', linestyle=':', label='unconstrained g')
    ax2.axhline(expected_g_val, color='k', linestyle='--', linewidth=1.0, label='constrained g')

    ax2.set_xscale('log'); ax2.grid(True); ax2.set_xlabel('iter/evals'); ax2.set_ylabel('g(x)'); ax2.set_title('Constraint violation')
    ax2.legend(loc='best', fontsize='small')

    plt.tight_layout()
    plt.savefig("stoch_radial_full_run.png", dpi=300)
    plt.show()

    # --- return results like your original function ---
    return {
        "c": c, "b": b,
        "x_star": x_star if 'x_star' in locals() else np.zeros(n),
        "val_star": val_star if 'val_star' in locals() else expected_f(np.zeros(n)),
        "x_al": x_al, "lam_al": lam_al,
        "mma_results": mma_results,
        "ccsa_results": ccsa_results,
        "ccsa_quad_results": ccsa_quad_results,
        "cssca_runs": cssca_runs,
        "val_uncon": expected_f(np.zeros(n)),
        "hist_al": hist_al
    }
