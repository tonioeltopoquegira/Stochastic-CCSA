import numpy as np
import nlopt
import matplotlib.pyplot as plt
from ccsa_mma import MMAOptimizer


def make_spd_matrix(n: int, cond: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
    if cond <= 1.0:
        s = np.ones(n)
    else:
        s = np.geomspace(1.0, cond, num=n)  # geometric spectrum
    Q = (U * s) @ U.T
    Q = 0.5 * (Q + Q.T)  # symmetrize
    return Q


def get_current_x(ccsa):
    """Try to extract the current iterate from an MMAOptimizer instance."""
    for attr in ["x", "x_k", "xk", "x_curr", "xval", "x_current"]:
        if hasattr(ccsa, attr):
            return getattr(ccsa, attr)
    raise AttributeError(
        "MMAOptimizer instance has no attribute 'x'. "
        "Tried x, x_k, xk, x_curr, xval, x_current."
    )


def run_deterministic_debug(
    n: int = 20,
    cond: float = 10.0,
    sigma_min: float = 1e-6,
    sigma_max: float = 1e2,
    rho_init: float = 1.0,
    max_inner: int = 5,
    max_outer: int = 200,
    nlopt_algo: str = "CCSAQ",
    seed: int = 0,
):
    np.random.seed(seed)

    # Problem data
    Q = make_spd_matrix(n, cond=cond, seed=seed)
    p = np.linspace(1.0, 2.0, n)

    def f_val(x: np.ndarray) -> float:
        return 0.5 * float(x @ (Q @ x)) + float(p @ x)

    def f_grad(x: np.ndarray) -> np.ndarray:
        return (Q @ x) + p

    x0 = np.full(n, 0.5)

    # --- NLopt setup ---
    nlopt_traj, nlopt_xs = [], []

    def nlopt_obj(x, grad):
        if grad.size > 0:
            grad[:] = f_grad(x)
        v = f_val(x)
        nlopt_traj.append(v)
        nlopt_xs.append(x.copy())
        return v

    if nlopt_algo.upper() == "CCSAQ":
        opt = nlopt.opt(nlopt.LD_CCSAQ, n)
    else:
        opt = nlopt.opt(nlopt.LD_MMA, n)

    opt.set_min_objective(nlopt_obj)
    opt.set_param("sigma_min", float(sigma_min))
    opt.set_param("inner_gradients", 0)
    opt.set_param("always_improve", 0)
    opt.set_maxeval(max_outer)

    x_nl = opt.optimize(x0.copy())
    f_nl = opt.last_optimum_value()

    # --- Custom CCSA (MMAOptimizer) ---
    def fun_only(x):
        return f_val(x)

    ccsa = MMAOptimizer(
        params=x0,
        fun=fun_only,
        g=None,
        bounds=None,
        rho_init=rho_init,
        max_inner=max_inner,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        df=f_grad,
        dg=None,
        x0=x0,
    )

    ccsa_xs = [x0.copy()]
    ccsa_fs = [f_val(x0)]

    for k in range(max_outer):
        f_best, g_best, metrics = ccsa.step()
        x_curr = get_current_x(ccsa)
        ccsa_xs.append(x_curr.copy())
        ccsa_fs.append(float(f_best))

    # --- Diagnostics ---
    m = min(15, len(nlopt_traj), len(ccsa_fs))
    print(f"\nStep-by-step comparison (first {m}):")
    for i in range(m):
        x_n = nlopt_xs[i] if i < len(nlopt_xs) else None
        x_c = ccsa_xs[i] if i < len(ccsa_xs) else None
        fn = nlopt_traj[i] if i < len(nlopt_traj) else np.nan
        fc = ccsa_fs[i] if i < len(ccsa_fs) else np.nan
        dx = np.linalg.norm(x_n - x_c) if (x_n is not None and x_c is not None) else np.nan
        print(f"it={i:03d}  f_nlopt={fn: .6e}  f_ccsa={fc: .6e}  |x_nl-x_ccsa|={dx: .3e}")

    # --- Trajectory plots ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(nlopt_traj, label=f"nlopt {nlopt_algo}", marker="o", alpha=0.7)
    plt.plot(ccsa_fs, label="custom CCSA", marker="x", alpha=0.7)
    plt.xlabel("iteration/evaluation index")
    plt.ylabel("objective f(x)")
    plt.grid(True)
    plt.legend()
    plt.title("Objective trajectories (deterministic)")

    L = min(len(nlopt_traj), len(ccsa_fs))
    diff_curve = np.abs(np.array(nlopt_traj[:L]) - np.array(ccsa_fs[:L]))

    plt.subplot(1, 2, 2)
    plt.plot(diff_curve, color="red", marker="s", label="|nlopt - custom|")
    plt.xlabel("index")
    plt.ylabel("|Δf|")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.title("Per-iteration absolute difference")

    plt.tight_layout()
    plt.show()

    # --- Final summary ---
    x_final = get_current_x(ccsa)
    print("\nFinal summary:")
    print(f"nlopt:    f* = {f_nl:.6e}, ||x*|| = {np.linalg.norm(x_nl):.3e}")
    print(f"custom:   f* = {f_val(x_final):.6e}, ||x*|| = {np.linalg.norm(x_final):.3e}")
    print(f"|x_nl - x_ccsa| = {np.linalg.norm(x_nl - x_final):.6e}")


if __name__ == "__main__":
    run_deterministic_debug(
        n=100,
        cond=1.0,
        sigma_min=1e-6,
        sigma_max=1e2,
        rho_init=1.0,
        max_inner=5,
        max_outer=200,
        nlopt_algo="MMA",
        seed=0,
    )
