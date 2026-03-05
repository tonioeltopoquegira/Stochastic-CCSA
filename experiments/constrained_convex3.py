import numpy as np
import matplotlib.pyplot as plt
from baselines.adam_al import adam_augmented_lagrangian
from baselines.cssca.core import CSSCAOptimizer
from plotting_3D import plot_rotated_2d_objective


def rotated_exp_stoch_constrained_exp(
    optimizer,
    optimizer_quad,
    tau: float = 0.5,
    a: float = 5.0,
    noise_std: float = 0.05,        # objective noise
    noise_std_g: float = 0.05,      # constraint noise (NEW)
    seed: int = 0,
    x0: np.ndarray = None,
    c: np.ndarray = None,
    b: float = 0.0,
    sigma_mins: list = [0.1],  # kept for interface/labels only; no NLopt
    maxiters_adam: int = 10000,
    mma_maxeval: int = 2000,
    rho: float = 1.0,
    lambda0: float = 0.0,
    tol_outer: float = 1e-6,
    rho_multiplier: float = 2.0,
    rho_update_max: int = 6,
    max_outer: int = 8,
    verbose: bool = True,
    cssca_tau_obj=[0.3, 10.0],
    cssca_tau_cons=[0.3, 10.0],
    ccsa_names=None
):

    rng = np.random.RandomState(seed)

    if x0 is None:
        x0 = np.array([1.5, -1.0])
    if c is None:
        c = np.array([1.0, -1.0])

    
    # Rotation
    theta = rng.uniform(0, 2*np.pi)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    print(f"Rotation angle θ = {theta:.4f}")
    print(f"Constraint ||c||={np.linalg.norm(c):.4f}, b={b:.4f}")
    print(f"Objective noise σ_f={noise_std}, Constraint noise σ_g={noise_std_g}")

    # Deterministic objective
    def f_det(z):
        z_rot = R @ z
        x_r, y_r = z_rot
        denom = tau**2 + a*y_r**2
        return -np.exp(-(y_r**2) - (x_r**2)/denom)

    def grad_det(z):
        z_rot = R @ z
        x_r, y_r = z_rot
        denom = tau**2 + a*y_r**2

        exp_term = np.exp(-(y_r**2) - (x_r**2)/denom)

        df_dx_r = exp_term * (2*x_r/denom)
        df_dy_r = exp_term * (
            2*y_r - (2*a*y_r*x_r**2)/(denom**2)
        )

        grad_rot = np.array([df_dx_r, df_dy_r])
        return R.T @ grad_rot

    
    # Stochastic oracle (additive noise)
    def oracle(x, grad=None):
        val = f_det(x) + rng.randn() * noise_std
        gval = grad_det(x)

        if grad is True:
            return val, gval
        if isinstance(grad, np.ndarray):
            grad[:] = gval
            return val
        return val

    def fgrad(x):
        return grad_det(x)

    def f_stoch(x):
        return f_det(x) + rng.randn() * noise_std

    
    # Stochastic constraint (additive noise)
    def constraint_val(x, xi=None):
        return float(np.dot(c, x) - b + rng.randn() * noise_std_g)

    def constraint_grad(x):
        return c   # gradient unchanged (additive noise)

    
    # AL-Adam
    x_al, lam_al, hist_al = adam_augmented_lagrangian(
        fgrad=fgrad,
        x0=x0.copy(),
        g=constraint_val,
        dg=lambda xx: constraint_grad(xx),
        rho=rho,
        lambda0=lambda0,
        f_stoch_estimate=f_stoch,
        alpha=1e-3,
        maxiters=maxiters_adam,
        tol=tol_outer,
        rho_multiplier=rho_multiplier,
        rho_update_max=rho_update_max,
        max_outer=max_outer,
        verbose=verbose
    )

    # Expected (noise-free) objective for plotting
    def expected_f(x):
        return f_det(x)

   
    # CCSA: non-conservative (optimizer) and quadratic (optimizer_quad)
    colors_ccsa = plt.cm.inferno(np.linspace(0.2, 0.8, len(sigma_mins)))
    ccsa_results = []

    # configure non-conservative CCSA optimizer 
    optimizer.fun = oracle
    optimizer.g = constraint_val
    optimizer.dg = lambda xx: np.atleast_2d(constraint_grad(xx))
    optimizer.x0 = x0.copy()
    optimizer.params = x0.copy(),

    for sigma_min, color in zip(sigma_mins, colors_ccsa):
        metrics = None
        all_x = []

        for _ in range(mma_maxeval):
            f_b, g_b, metrics = optimizer.step()
            all_x.append(metrics["x_history"][-1])

        x_hist = np.asarray(metrics.get("x_history", []))
        cum_we_hist = np.asarray(metrics.get("cumulative_weighted_evals_history", []), dtype=float)
        if x_hist.ndim == 1 and x_hist.size > 0:
            x_hist = x_hist.reshape((1, -1))
        if x_hist.shape[0] > 0:
            f_expected_at_xhist = [expected_f(xi) for xi in x_hist]
            f_stoch_at_xhist = [oracle(xi) for xi in x_hist]
            g_at_xhist = [constraint_val(xi) for xi in x_hist]
        else:
            f_expected_at_xhist = []
            f_stoch_at_xhist = []
            g_at_xhist = []
        if cum_we_hist.size == 0:
            cum_we_hist = np.arange(1, len(f_expected_at_xhist) + 1, dtype=float)

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

    # Quadratic-surrogate CCSA (conservative)
    colors_ccsa_quad = plt.cm.plasma(np.linspace(0.2, 0.8, len(sigma_mins)))
    ccsa_quad_results = []

    optimizer_quad.fun = oracle
    optimizer_quad.g = constraint_val
    optimizer_quad.dg = lambda xx: np.atleast_2d(constraint_grad(xx))
    optimizer_quad.x0 = x0.copy()
    optimizer_quad.params = x0.copy(),

    for sigma_min, color in zip(sigma_mins, colors_ccsa_quad):
        metrics = None
        all_x = []

        for _ in range(mma_maxeval):
            f_b, g_b, metrics = optimizer_quad.step()
            all_x.append(metrics["x_history"][-1])

        x_hist = np.asarray(metrics.get("x_history", []))
        cum_we_hist = np.asarray(metrics.get("cumulative_weighted_evals_history", []), dtype=float)
        if x_hist.ndim == 1 and x_hist.size > 0:
            x_hist = x_hist.reshape((1, -1))
        if x_hist.shape[0] > 0:
            f_expected_at_xhist = [expected_f(xi) for xi in x_hist]
            f_stoch_at_xhist = [oracle(xi) for xi in x_hist]
            g_at_xhist = [constraint_val(xi) for xi in x_hist]
        else:
            f_expected_at_xhist = []
            f_stoch_at_xhist = []
            g_at_xhist = []
        if cum_we_hist.size == 0:
            cum_we_hist = np.arange(1, len(f_expected_at_xhist) + 1, dtype=float)

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

    
    # CSSCA Tau Sweep
    cssca_tau_objs = cssca_tau_obj if isinstance(cssca_tau_obj, (list, tuple)) else [cssca_tau_obj]
    cssca_tau_cons = cssca_tau_cons if isinstance(cssca_tau_cons, (list, tuple)) else [cssca_tau_cons]

    L = max(len(cssca_tau_objs), len(cssca_tau_cons))
    if len(cssca_tau_objs) == 1:
        cssca_tau_objs *= L
    if len(cssca_tau_cons) == 1:
        cssca_tau_cons *= L

    cssca_runs = []
    colors_css = plt.cm.tab10(np.arange(L) % 10)

    for idx, (tau_o, tau_c) in enumerate(zip(cssca_tau_objs, cssca_tau_cons)):

        cssca_opt = CSSCAOptimizer(
            params=x0.copy(),
            fun=oracle,
            g=lambda xx: float(np.dot(c, xx) - b + rng.randn() * noise_std_g),
            dg=lambda xx: np.atleast_2d(c),
            x0=x0.copy(),
            rho_t_schedule=rho,
            gamma_t_schedule=1.0,
            tau_obj=float(tau_o),
            tau_cons=float(tau_c),
            samples_per_iter=1
        )

        f_hist, g_hist = [], []

        for _ in range(1000):
            x_cssca, f_cssca, cons_cssca = cssca_opt.step()
            f_hist.append(f_cssca)
            g_hist.append(cons_cssca[0] if hasattr(cons_cssca, '__len__') else cons_cssca)

        cssca_runs.append({
            "tau_obj": tau_o,
            "tau_cons": tau_c,
            "f_hist": np.array(f_hist),
            "g_hist": np.array(g_hist),
            "color": colors_css[idx]
        })

        print(f"CSSCA τo={tau_o}, τc={tau_c}, final f={f_hist[-1]:.4f}")

    
    # Plot
    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    # Prefer plotting observed (noisy) objective values for AL-Adam if recorded;
    # fall back to internal estimate if necessary.
    al_x_list = None
    for k in ('x', 'x_hist', 'x_history'):
        if k in hist_al and len(hist_al[k]) > 0:
            al_x_list = hist_al[k]
            break
    if al_x_list is not None and len(al_x_list) > 0:
        try:
            f_obs_al = np.array([oracle(np.asarray(xx)) for xx in al_x_list], dtype=float)
        except Exception:
            f_obs_al = np.asarray(hist_al.get('f_est', []), dtype=float)
    else:
        f_obs_al = np.asarray(hist_al.get('f_est', []), dtype=float)

    # If lengths mismatch, align to x-axis length
    x_axis_al = np.asarray(hist_al.get('iter', np.arange(1, f_obs_al.size + 1)), dtype=float)
    if x_axis_al.size != f_obs_al.size:
        # try to trim or pad f_obs_al to match axis
        n = min(x_axis_al.size, f_obs_al.size)
        x_axis_al = x_axis_al[:n]
        f_obs_al = f_obs_al[:n]

    ax1.plot(x_axis_al, f_obs_al, 'k-', label="AL-Adam (obs)")

    # CCSA (non-conservative)
    # prepare CCSA labels; support custom names passed by caller
    if ccsa_names is None:
        ccsa_names = ["non-conservative CCSA", "conservative CCSA quad"]

    for cr in ccsa_results:
        col = cr.get('color', 'tab:orange')
        # plot observed (noisy) objective values for consistency
        label0 = f"{ccsa_names[0]} (obs) σ={cr['sigma_min']}" if len(ccsa_names) > 0 else f"CCSA (obs) σ={cr['sigma_min']}"
        ax1.plot(cr['cum_we_hist'], cr['f_stoch_at_xhist'], linestyle='-', linewidth=2.0,
                 color='gray', label=label0)

    # CCSA quadratic (conservative)
    for cr in ccsa_quad_results:
        col = cr.get('color', 'tab:green')
        # plot observed (noisy) objective values for consistency
        label1 = f"{ccsa_names[1]} (obs) σ={cr['sigma_min']}" if len(ccsa_names) > 1 else f"CCSA quad (obs) σ={cr['sigma_min']}"
        ax1.plot(cr['cum_we_hist'], cr['f_stoch_at_xhist'], linestyle='-', linewidth=2.0,
                 color=col, label=label1)

    # CSSCA runs
    for run in cssca_runs:
        ax1.plot(
            np.arange(1, len(run["f_hist"]) + 1),
            run["f_hist"],
            color=run["color"],
            linewidth=2,
            label=f"CSSCA {run['tau_obj']}"
        )

    ax1.axhline(-1.0, color='k', linestyle='--')
    ax1.set_title("Objective")
    ax1.set_xscale('log')
    ax1.grid(True)

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(hist_al['iter'], hist_al['g'], 'k-', label="AL-Adam")

    for cr in ccsa_results:
        label0 = f"{ccsa_names[0]} σ={cr['sigma_min']}" if len(ccsa_names) > 0 else f"CCSA σ={cr['sigma_min']}"
        ax2.plot(cr['cum_we_hist'], cr['g_at_xhist'], linestyle='-', color='gray', linewidth=1.8,
                 label=label0)

    for cr in ccsa_quad_results:
        col = cr.get('color', 'tab:green')
        label1 = f"{ccsa_names[1]} σ={cr['sigma_min']}" if len(ccsa_names) > 1 else f"CCSA quad σ={cr['sigma_min']}"
        ax2.plot(cr['cum_we_hist'], cr['g_at_xhist'], linestyle='-', color=col, linewidth=1.8,
                 label=label1)

    for run in cssca_runs:
        ax2.plot(
            np.arange(1, len(run["g_hist"]) + 1),
            run["g_hist"],
            color=run["color"],
            linewidth=2,
            label=f"CSSCA {run['tau_obj']}"
        )

    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_title("Constraint")
    ax2.set_xscale('log')
    ax2.grid(True)
    ax2.legend(fontsize="small")

    plt.tight_layout()
    plt.show()

    plot_rotated_2d_objective(
        R=R,
        tau=tau,
        a=a,
        c=c,
        b=b,
        solver_paths=None,
        xlim=(-2, 2),
        ylim=(-2, 2)
    )


    return {
        "x_al": x_al,
        "hist_al": hist_al,
        "ccsa_results": ccsa_results,
        "ccsa_quad_results": ccsa_quad_results,
        "cssca_runs": cssca_runs
    }
