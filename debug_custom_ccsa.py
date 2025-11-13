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
        # geometric spectrum from 1 to cond
        s = np.geomspace(1.0, cond, num=n)
    Q = (U * s) @ U.T
    # symmetrize guard
    Q = 0.5 * (Q + Q.T)
    return Q


def run_deterministic_debug(
    n: int = 20,
    cond: float = 10.0,
    sigma_min: float = 1e-6,
    sigma_max: float = 1e2,
    rho_init: float = 1.0,
    max_inner: int = 5,
    max_outer: int = 200,
    nlopt_algo: str = "CCSAQ",  # "CCSAQ" or "MMA"
    seed: int = 0,
):
    np.random.seed(seed)

    # Problem data (deterministic)
    Q = make_spd_matrix(n, cond=cond, seed=seed)
    p = np.linspace(1.0, 2.0, n)  # deterministic linear term

    def f_val(x: np.ndarray) -> float:
        return 0.5 * float(x @ (Q @ x)) + float(p @ x)

    def f_grad(x: np.ndarray) -> np.ndarray:
        return (Q @ x) + p

    # Common initial point
    x0 = np.full(n, 0.5)

    # --- NLopt setup ---
    nlopt_traj = []  # objective values per callback
    nlopt_xs = []    # iterate snapshots per callback

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
    # keep behavior close to our implementation
    opt.set_param("inner_gradients", 0)
    opt.set_param("always_improve", 0)
    opt.set_maxeval(max_outer)  # limit total evals for fair comparison

    x_nl = opt.optimize(x0.copy())
    f_nl = opt.last_optimum_value()

    # --- Custom CCSA (MMAOptimizer) setup ---
    # Provide fun (value-only) and df for gradients to match deterministic setup
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

    # Run a fixed number of outer iterations and collect trajectory
    ccsa_xs = [x0.copy()]
    ccsa_fs = [f_val(x0)]
    for k in range(max_outer):
        f_best, g_best, metrics = ccsa.step()
        ccsa_xs.append(ccsa.x.copy())
        ccsa_fs.append(float(f_best))

    # --- Step-by-step diagnostics (first 15 steps or available length) ---
    m = min(15, len(nlopt_traj), len(ccsa_fs))
    print("\nStep-by-step comparison (first {}):".format(m))
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
    plt.plot(nlopt_traj, label=f"nlopt {nlopt_algo}", marker='o', alpha=0.7)
    plt.plot(ccsa_fs, label="custom CCSA", marker='x', alpha=0.7)
    plt.xlabel("iteration/evaluation index")
    plt.ylabel("objective f(x)")
    #plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.title("Objective trajectories (deterministic)")

    # Absolute difference per index (truncate to common length)
    L = min(len(nlopt_traj), len(ccsa_fs))
    diff_curve = np.abs(np.array(nlopt_traj[:L]) - np.array(ccsa_fs[:L]))

    plt.subplot(1, 2, 2)
    plt.plot(diff_curve, color='red', marker='s', label='|nlopt - custom|')
    plt.xlabel("index")
    plt.ylabel("|Δf|")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.title("Per-iteration absolute difference")

    plt.tight_layout()
    plt.show()

    # Final summary
    print("\nFinal summary:")
    print("nlopt:    f* = {:.6e}, ||x*|| = {:.3e}".format(f_nl, np.linalg.norm(x_nl)))
    print("custom:   f* = {:.6e}, ||x*|| = {:.3e}".format(f_val(ccsa.x), np.linalg.norm(ccsa.x)))
    print("|x_nl - x_ccsa| = {:.6e}".format(np.linalg.norm(x_nl - ccsa.x)))


if __name__ == "__main__":
    # Example run
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
