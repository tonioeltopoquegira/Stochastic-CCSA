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
    plt.plot(vals, "g-", label="adam", color="green") # loglog to be done
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
                    plt.loglog(np.asarray(f_evals), f_vals, label=f"nlopt σ={sigma_min}", color=color)
            else:
                evals = 0
                f_evals, x_errs, f_vals = [], [], []
                
                optimizer_nlopt.set_min_objective(f_and_grad_nlopt)
                optimizer_nlopt.set_param("inner_gradients", 0)
                optimizer_nlopt.set_param("always_improve", 0)
                optimizer_nlopt.set_maxeval(outer)


                xopt = optimizer_nlopt.optimize(x_init)
                plt.loglog(np.asarray(f_evals), f_vals, label=f"nlopt", color='red')
                    
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
    try:
        cssca_results = None
        # Use deterministic (no-sample) oracle and deterministic constraint for CSSCA
        # to match the debug harness (no xi drawn inside surrogate updates).
        cssca_opt = CSSCAOptimizer(params=x0.copy(),
                                   fun=make_noisy_f_and_grad(A, lambda: np.zeros(n)),
                                   g=lambda xx: float(np.dot(c, xx) - b),
                                   dg=lambda xx: np.atleast_2d(c),
                                   x0=x0.copy(), rho_t_schedule=float(rho), gamma_t_schedule=1.0,
                                   tau_obj=1.0, tau_cons=1.0, samples_per_iter=1)

        cssca_f_hist = []
        cssca_cons_hist = []
        # run a fixed number of outer iterations similar to mma_maxeval
        for t in range(mma_maxeval):
            # Use deterministic step (no sample drawer) so CSSCA solves the same expected problem
            x_cssca, f_cssca, cons_cssca = cssca_opt.step()
            cssca_f_hist.append(f_cssca)
            cssca_cons_hist.append(cons_cssca.copy() if hasattr(cons_cssca, 'copy') else np.atleast_1d(cons_cssca))

        cssca_cons_arr = np.vstack(cssca_cons_hist) if len(cssca_cons_hist) > 0 and np.asarray(cssca_cons_hist).ndim == 2 else np.asarray(cssca_cons_hist)
        print(f"CSSCA (rho={rho}): last f={cssca_f_hist[-1]:.6g}, last g[0]={cssca_cons_arr[-1,0] if cssca_cons_arr.size>0 else np.nan:.6g}")
    except Exception as e:
        cssca_f_hist = []
        cssca_cons_arr = np.array([])
        print(f"Failed to run CSSCA optimizer: {e}")

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


    # Plotting: objective + constraint panels with patches
    plt.figure(figsize=(12,5))

    # define patch half-height based on noise const (visualization choice)
    patch_half = max(1e-12, 0.5 * (noise**2) * traceA2)

    # objective panel
    ax1 = plt.subplot(1,2,1)
    ax1.plot(hist_al['iter'], hist_al['f_est'], '-', color='black', label='AL-Adam')

    #for sigma_min, color, f_evals, g_vals, f_vals in mma_results:
    #    ax1.plot(f_evals, f_vals, '.-', color=color, alpha=0.9, label=f'MMA σ={sigma_min}')

    # CCSA: solid lines, different color palette (not dashed)
    for cr in ccsa_results:
        if ccsa_plot_expected:
            ax1.plot(cr['cum_we_hist'], cr['f_expected_at_xhist'], linestyle='-', linewidth=2.0,
                     color='red', label=f'custom σ={cr["sigma_min"]}')
        else:
            ax1.plot(cr['cum_we_hist'], cr['f_stoch_at_xhist'], linestyle='-', linewidth=1.5,
                     marker='.', color='red', label=f'custom σ={cr["sigma_min"]}')
    
    for cr in ccsa_quad_results:
        if ccsa_plot_expected:
            ax1.plot(cr['cum_we_hist'], cr['f_expected_at_xhist'], linestyle='-', linewidth=2.0,
                     color=cr['color'], label=f'custom quad σ={cr["sigma_min"]}')
        else:
            ax1.plot(cr['cum_we_hist'], cr['f_stoch_at_xhist'], linestyle='-', linewidth=1.5,
                     marker='.', color=cr['color'], label=f'custom quad σ={cr["sigma_min"]}')

    # plot CSSCA if available
    try:
        if len(cssca_f_hist) > 0:
            ax1.plot(np.arange(1, len(cssca_f_hist)+1), cssca_f_hist, linestyle='-', color='purple', label=f'CSSCA ρ={rho}')
    except NameError:
        pass
    


    # shaded patches for stochastic constrained and unconstrained (horizontal bands)
    # constrained: center at val_star
    ax1.axhspan(val_star - patch_half, val_star + patch_half, alpha=0.18, facecolor='tab:orange')
    # unconstrained baseline: center at val_uncon
    ax1.axhspan(val_uncon - patch_half, val_uncon + patch_half, alpha=0.12, facecolor='tab:gray')

    # add legend patches for these bands
    constrained_patch = mpatches.Patch(facecolor='tab:orange', alpha=0.18, label='stochastic constrained (± noise const)')
    unconstrained_patch = mpatches.Patch(facecolor='tab:gray', alpha=0.12, label='stochastic unconstrained (± noise const)')
    # build legend (include patches)
    handles, labels = ax1.get_legend_handles_labels()
    handles = handles + [constrained_patch, unconstrained_patch]
    labels = labels + [constrained_patch.get_label(), unconstrained_patch.get_label()]
    ax1.legend(handles, labels, loc='best', fontsize='small')

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
        ax2.plot(cr['cum_we_hist'], cr['g_at_xhist'], linestyle='-', color='red', label=f'CCSA σ={cr["sigma_min"]}')
    
    for cr in ccsa_quad_results:
        ax2.plot(cr['cum_we_hist'], cr['g_at_xhist'], linestyle='-', color=cr['color'], label=f'CCSA QUAD σ={cr["sigma_min"]}')

    # CSSCA constraint trace
    try:
        if cssca_cons_arr.size != 0:
            # plot first constraint component if present
            if cssca_cons_arr.ndim == 2:
                ax2.plot(np.arange(1, cssca_cons_arr.shape[0]+1), cssca_cons_arr[:, 0], linestyle='-', color='purple', label=f'CSSCA g[0] ρ={rho}')
            else:
                ax2.plot(np.arange(1, len(cssca_cons_arr)+1), cssca_cons_arr, linestyle='-', color='purple', label=f'CSSCA g ρ={rho}')
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
