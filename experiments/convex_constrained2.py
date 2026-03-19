import numpy as np
import matplotlib.pyplot as plt


from utils import randcond, make_red_colors
from baselines.adam import adam
from baselines.adam_al import adam_augmented_lagrangian
from baselines.cssca.core import CSSCAOptimizer

import matplotlib.patches as mpatches

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
    curv_change: str = 'large',
    ccsa_names=None
):
    """
    Stochastic constrained experiment using diagonal exponential-quadratic objective:
        f(x+xi) = R exp( (x+xi)^T M (x+xi) ), M diagonal, R rotation
    Diagonal noise xi ~ N(0, noise^2 I) so we can compute exact expectation.

    Keeps structure and plotting of your original stoch_convex_con_exp and
    includes CSSCA baseline runs (tau broadcasting etc.).
    """

    # deterministic RNG for this experiment (single source)
    rng = np.random.RandomState(seed)

    # Problem size and diagonal curvature
    n = 100

    if curv_change == 'small':
        # choose diagonal curvatures m_i 
        m_min = 0.005
        m_max = 0.05 * max(1, cond)
        m = np.linspace(m_min, m_max, n)   # diagonal values of M

    if curv_change == 'large':
        m_min = 0.001           # very small -> very flat directions
        m_max = 1.0 * cond       # very large -> very steep directions
        m = np.logspace(np.log10(m_min), np.log10(m_max), n)  # logarithmic spread


    # A such that quadratic = ||A (x+xi)||^2 and M = A.T @ A
    A = np.diag(np.sqrt(m))

    # noise: scalar -> isotropic diagonal with variance noise^2
    Sigma_diag = (noise**2) * np.ones(n)

    # stability: require d_i = 1 - 2 sigma^2 m_i > 0 for exact expectation formula
    d = 1.0 - 2.0 * (noise**2) * m
    
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

    # Introduce a random rotation
    
    i_rot, j_rot = rng.choice(n, size=2, replace=False)
    theta = rng.uniform(-0.3, 0.3)  # rotation angle
    cth, sth = np.cos(theta), np.sin(theta)
    R = np.eye(n)
    R_block = np.array([[cth, -sth], [sth, cth]])
    R[np.ix_([i_rot, j_rot], [i_rot, j_rot])] = R_block


    A_rot = A.dot(R)

    log_det_term = -0.5 * np.sum(np.log(d))     # log(1 / sqrt(prod_i d_i))
    # diagonal m_tilde = m_i / d_i, then lift to full matrix via rotation:
    m_tilde_diag = m / d
    M_tilde_full = R.T @ np.diag(m_tilde_diag) @ R

    # Expected objective (exact, with rotation)
    def expected_f(x):
        expo = float(x.T @ (M_tilde_full @ x))
        return float(np.exp(log_det_term + expo))

    # Expected constraint (exact, unchanged by rotation of objective)
    def expected_g(x):
        return float(np.dot(c, x) - b)

    # Analytic expected minimizer under linear constraint for quadratic exponent:
    def analytic_solution_expectation(Mtilde_full, c_vec, b_scalar):
        s = np.linalg.solve(Mtilde_full, c_vec)     # Mtilde_full^{-1} c
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

    x_star, lambda_star, active_flag, val_star = analytic_solution_expectation(M_tilde_full, c, b)
    print("Analytic expected optimum (on expectation):")
    print(f"  active? {active_flag}")
    print(f"  ||x*||={np.linalg.norm(x_star):.6g}, λ*={lambda_star:.6g}")
    print(f"  E[f(x*)]={val_star:.6g}")

    val_uncon = expected_f(np.zeros(n))
    g_uncon = -b
    print(f"Unconstrained baseline: E[f(0)]= {val_uncon:.6g}, g(0)={g_uncon:.6g}")

    # Sampling closure used by all function/grad calls in this run ---
    def sample_xi():
        return rng.randn(n) * noise

    # Noisy oracle: returns a function f_and_grad(x, grad=None)
    def make_noisy_f_and_grad(A_mat, sample_xi_fn):
        # A_mat is A_rot so quadratic = ||A_rot (x+xi)||^2 = (x+xi)^T R^T M R (x+xi)
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

    # Run custom CCSA 
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
            # Reset optimizer state before each sigma_min run to prevent state carryover
            optimizer.reset(x0=x0.copy())
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

    # Run baseline CSSCA 
    cssca_tau_objs_list = cssca_tau_obj if isinstance(cssca_tau_obj, (list, tuple, np.ndarray)) else [cssca_tau_obj]
    cssca_tau_cons_list = cssca_tau_cons if isinstance(cssca_tau_cons, (list, tuple, np.ndarray)) else [cssca_tau_cons]
    
    if cssca_samples_per_iter is None:
        cssca_samples_list = [1]
    else:
        cssca_samples_list = cssca_samples_per_iter if isinstance(cssca_samples_per_iter, (list, tuple, np.ndarray)) else [cssca_samples_per_iter]

    # Not strictly needed
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
            # Provide CSSCA with a stochastic constraint evaluator that accepts xi
            cssca_opt = CSSCAOptimizer(params=x0.copy(),
                           #fun=make_noisy_f_and_grad(A, lambda: np.zeros(n)),
                           fun=make_noisy_f_and_grad(A, sample_xi),
                           g=lambda xx, xi=None: float(np.dot(c, xx + (xi if xi is not None else sample_xi())) - b),
                           dg=lambda xx: np.atleast_2d(c),
                           x0=x0.copy(), rho_t_schedule=float(rho), gamma_t_schedule=1.0,
                           tau_obj=float(tau_o), tau_cons=float(tau_c), samples_per_iter=1.0)

            cssca_f_hist = []
            cssca_cons_hist = []
            for t in range(1000):
                x_cssca, f_cssca, cons_cssca = cssca_opt.step()
                cssca_f_hist.append(f_cssca)
                cssca_cons_hist.append(cons_cssca.copy() if hasattr(cons_cssca, 'copy') else np.atleast_1d(cons_cssca))

            cssca_cons_arr = np.vstack(cssca_cons_hist) if len(cssca_cons_hist) > 0 and np.asarray(cssca_cons_hist).ndim == 2 else np.asarray(cssca_cons_hist)
            print(f"CSSCA (tau_obj={tau_o}, tau_cons={tau_c}, rho={rho}): last f={cssca_f_hist[-1]:.6g}, last g[0]={cssca_cons_arr[-1,0] if cssca_cons_arr.size>0 else np.nan:.6g}")
            cssca_runs.append({"tau_obj": tau_o, "tau_cons": tau_c, "f_hist": cssca_f_hist, "cons_arr": cssca_cons_arr})
        except Exception as e:
            print(f"Failed to run CSSCA optimizer (tau_obj={tau_o}, tau_cons={tau_c}): {e}")
            cssca_runs.append({"tau_obj": tau_o, "tau_cons": tau_c, "f_hist": [], "cons_arr": np.array([])})
        
    # Run CCSA-quad 
    ccsa_quad_results = []
    oracle = make_noisy_f_and_grad(A, sample_xi)
    if optimizer_quad is not None:
        optimizer_quad.fun = oracle
        optimizer_quad.g = constraint_val
        optimizer_quad.dg = lambda xx: np.atleast_2d(c)
        optimizer_quad.x0 = x0.copy()
        optimizer_quad.params = x0.copy(),
        for sigma_min, color in zip(sigma_mins, colors_ccsa):
            # Reset optimizer state before each sigma_min run to prevent state carryover
            optimizer_quad.reset(x0=x0.copy())

            metrics = None
            all_x = []
            for out in range(mma_maxeval):
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

    # PLOTTING & HELPERS
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

    plt.figure(figsize=(12,5))
    patch_half = max(1e-12, 0.5 * (noise**2) * np.sum(m))

    ax1 = plt.subplot(1,2,1)
    ax1.plot(hist_al['iter'], hist_al['f_est'], '-', color='black', label='AL-Adam')

    # allow caller to provide custom names for CCSA variants
    if ccsa_names is None:
        ccsa_names = ["custom non-cons", "custom cons"]

    for cr in ccsa_results:
        col = cr.get('color', 'tab:orange')
        label0 = f"{ccsa_names[0]} σ={cr['sigma_min']}" if len(ccsa_names) > 0 else f"custom non-cons σ={cr['sigma_min']}"
        ax1.plot(cr['cum_we_hist'], cr['f_stoch_at_xhist'], linestyle='-', linewidth=1.5,
                     marker='o', markersize=4, color=col, label=label0)
    for cr in ccsa_quad_results:
        col = cr.get('color', 'tab:green')
        label1 = f"{ccsa_names[1]} σ={cr['sigma_min']}" if len(ccsa_names) > 1 else f"custom cons σ={cr['sigma_min']}"
        ax1.plot(cr['cum_we_hist'], cr['f_stoch_at_xhist'], linestyle='-', linewidth=1.5,
                    marker='s', markersize=4, color='gray', label=label1)

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
        label0 = f"{ccsa_names[0]} σ={cr['sigma_min']}" if len(ccsa_names) > 0 else f"CCSA non-conservative σ={cr['sigma_min']}"
        ax2.plot(cr['cum_we_hist'], cr['g_at_xhist'], linestyle='-', color='gray', linewidth=1.8, label=label0)
    for cr in ccsa_quad_results:
        col = cr.get('color', 'tab:green')
        label1 = f"{ccsa_names[1]} σ={cr['sigma_min']}" if len(ccsa_names) > 1 else f"CCSA conservative σ={cr['sigma_min']}"
        ax2.plot(cr['cum_we_hist'], cr['g_at_xhist'], linestyle='-', color=col, linewidth=1.8, label=label1)

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
                             label=f"CSSCA τ={run['tau_obj']}")
                else:
                    ax2.plot(np.arange(1, len(carr)+1), carr, linestyle='-', color=colors_css[idx], linewidth=2.0,
                             label=f"CSSCA τ={run['tau_obj']}")
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
