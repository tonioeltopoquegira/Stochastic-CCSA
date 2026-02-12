import numpy as np
import matplotlib.pyplot as plt
from baselines.adam_al import adam_augmented_lagrangian
from baselines.cssca.core import CSSCAOptimizer


def rotated_exp_stoch_constrained_exp(
    optimizer,
    optimizer_quad,
    optimizer_nlopt,
    tau: float = 0.5,
    a: float = 5.0,
    noise_std: float = 0.05,
    seed: int = 0,
    x0: np.ndarray = None,
    c: np.ndarray = None,
    b: float = 0.0,
    sigma_mins: list = [0.1],
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
):
    rng = np.random.RandomState(seed)
    n = 2

    if x0 is None:
        x0 = np.array([1.5, -1.0])
    if c is None:
        c = np.array([1.0, -1.0])

    # ---------------------------------------------------------
    # Rotation
    # ---------------------------------------------------------
    theta = rng.uniform(0, 2*np.pi)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    print(f"Rotation angle θ = {theta:.4f}")
    print(f"Constraint ||c||={np.linalg.norm(c):.4f}, b={b:.4f}")

    # ---------------------------------------------------------
    # Objective
    # ---------------------------------------------------------
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

    def oracle(x, grad=None):
        val = f_det(x) + rng.randn()*noise_std
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
        return f_det(x) + rng.randn()*noise_std

    def constraint_val(x, xi=None):
        return float(np.dot(c, x) - b)

    # ---------------------------------------------------------
    # AL-Adam
    # ---------------------------------------------------------
    x_al, lam_al, hist_al = adam_augmented_lagrangian(
        fgrad=fgrad,
        x0=x0.copy(),
        g=constraint_val,
        dg=lambda xx: c,
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

    # ---------------------------------------------------------
    # NLopt MMA
    # ---------------------------------------------------------
    mma_results = []
    colors_mma = plt.cm.viridis(np.linspace(0.2, 0.8, len(sigma_mins)))

    for sigma_min, color in zip(sigma_mins, colors_mma):

        evals = 0
        f_hist, g_hist = [], []

        def f_and_grad_mma(x, grad):
            nonlocal evals
            val = f_det(x) + rng.randn()*noise_std
            if grad.size > 0:
                grad[:] = grad_det(x)
            evals += 1
            f_hist.append(val)
            g_hist.append(constraint_val(x))
            return val

        def cons(x, grad):
            if grad.size > 0:
                grad[:] = c
            return constraint_val(x)

        optimizer_nlopt.add_inequality_constraint(cons, 0.0)
        optimizer_nlopt.set_min_objective(f_and_grad_mma)
        optimizer_nlopt.set_maxeval(mma_maxeval)
        optimizer_nlopt.set_param("sigma_min", float(sigma_min))

        x_mma = optimizer_nlopt.optimize(x0.copy())
        mma_results.append((sigma_min, color, f_hist, g_hist))

    # ---------------------------------------------------------
    # CSSCA Tau Sweep
    # ---------------------------------------------------------
    cssca_tau_objs = cssca_tau_obj if isinstance(cssca_tau_obj, (list,tuple)) else [cssca_tau_obj]
    cssca_tau_cons = cssca_tau_cons if isinstance(cssca_tau_cons, (list,tuple)) else [cssca_tau_cons]

    L = max(len(cssca_tau_objs), len(cssca_tau_cons))
    if len(cssca_tau_objs)==1: cssca_tau_objs *= L
    if len(cssca_tau_cons)==1: cssca_tau_cons *= L

    cssca_runs = []
    colors_css = plt.cm.tab10(np.arange(L)%10)

    for idx,(tau_o,tau_c) in enumerate(zip(cssca_tau_objs, cssca_tau_cons)):

        cssca_opt = CSSCAOptimizer(
            params=x0.copy(),
            fun=oracle,
            g=lambda xx: float(np.dot(c,xx)-b),
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
            g_hist.append(cons_cssca[0] if hasattr(cons_cssca,'__len__') else cons_cssca)

        cssca_runs.append({
            "tau_obj": tau_o,
            "tau_cons": tau_c,
            "f_hist": np.array(f_hist),
            "g_hist": np.array(g_hist),
            "color": colors_css[idx]
        })

        print(f"CSSCA τo={tau_o}, τc={tau_c}, final f={f_hist[-1]:.4f}")

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(12,5))

    ax1 = plt.subplot(1,2,1)
    ax1.plot(hist_al['iter'], hist_al['f_est'], 'k-', label="AL-Adam")

    for sigma_min,color,f_hist,g_hist in mma_results:
        ax1.plot(np.arange(len(f_hist)), f_hist, color=color,
                 label=f"MMA σ={sigma_min}")

    for run in cssca_runs:
        ax1.plot(np.arange(len(run["f_hist"])),
                 run["f_hist"],
                 color=run["color"],
                 linewidth=2,
                 label=f"CSSCA τo={run['tau_obj']}, τc={run['tau_cons']}")

    ax1.axhline(-1.0, color='k', linestyle='--')
    ax1.set_title("Objective")
    ax1.grid(True)
    ax1.set_xscale('log')
    #ax1.set_yscale('log')

    ax2 = plt.subplot(1,2,2)
    ax2.plot(hist_al['iter'], hist_al['g'], 'k-', label="AL-Adam")

    for sigma_min,color,f_hist,g_hist in mma_results:
        ax2.plot(np.arange(len(g_hist)), g_hist, color=color)

    for run in cssca_runs:
        ax2.plot(np.arange(len(run["g_hist"])),
                 run["g_hist"],
                 color=run["color"],
                 linewidth=2)

    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_title("Constraint")
    ax2.grid(True)
    ax2.set_xscale('log')
    #ax2.set_yscale('log')

    ax1.legend(fontsize="small")
    plt.tight_layout()
    plt.show()

    return {
        "x_al": x_al,
        "hist_al": hist_al,
        "mma_results": mma_results,
        "cssca_runs": cssca_runs
    }
