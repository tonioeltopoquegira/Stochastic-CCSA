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


    #optimizer.summarize_diagnostics()