import numpy as np

def adam_augmented_lagrangian(fgrad, x0, g, dg=None, rho=1.0, lambda0=0.0,
                              alpha=1e-3, maxiters=500000, tol=1e-8,
                              beta1=0.9, beta2=0.999, eps=1e-8,
                              update_lambda_every=100, log_every=1,
                              f_stoch_estimate=None, rho_multiplier=2.0,
                              rho_update_max=10, inner_kwargs=None, max_outer=None,
                              verbose=None):
    
    x = x0.copy()
    lam = float(lambda0)
    mom = np.zeros_like(x)
    var = np.zeros_like(x)
    hist = {'iter': [], 'f_est': [], 'g': [], 'norm_x': [], 'lambda': []}
    total_grad_evals = 0.0              

    for k in range(1, maxiters + 1):
        g_f = fgrad(x)
        total_grad_evals += 1.0

        g_val = float(g(x))
        # Use provided constraint gradient
        if dg is None:
            raise ValueError("dg (constraint gradient function) must be provided")
        g_grad = dg(x)  # vector of same size as x

        gL = g_f + max(0.0, lam + rho * g_val) * g_grad  # proper AL gradient

        # Adam moments
        mom = beta1 * mom + (1.0 - beta1) * gL
        var = beta2 * var + (1.0 - beta2) * (gL * gL)

        mhat = mom / (1.0 - beta1**k)
        vhat = var / (1.0 - beta2**k)

        x = x - alpha * (mhat / (np.sqrt(vhat) + eps))

        if (k % update_lambda_every) == 0:
            g_val = float(g(x))
            lam = max(0.0, lam + rho * g_val)

        f_est = f_stoch_estimate(x) if f_stoch_estimate else np.nan
        total_grad_evals += 1.0
        hist['iter'].append(k)
        hist['f_est'].append(f_est)
        hist['g'].append(g_val)
        hist['norm_x'].append(np.linalg.norm(x))
        hist['lambda'].append(lam)

        step_norm = np.linalg.norm(mhat / (np.sqrt(vhat) + eps))
        if abs(g_val) <= tol and step_norm < tol:
            break

    hist['total_grad_evals'] = total_grad_evals
    return x, lam, hist
