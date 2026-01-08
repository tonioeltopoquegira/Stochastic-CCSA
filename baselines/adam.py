import numpy as np

def adam(fgrad, xinit, ferr=None, f=None, tol=1e-8, maxiters=1000,
         alpha=0.001, eps=1e-8, beta1=0.9, beta2=0.999):
    
    
    x = xinit

    # initial residual and norms
    if ferr:
        errs = [ferr(x)]

    if f:
        vals = [f(x)]
        print(vals)

    # initial Adam state
    mom = np.zeros_like(x)  # first moment (m)
    var = np.zeros_like(x)  # second moment (v)

    for i in range(1, maxiters + 1):
        grad = fgrad(x)
        
        # Adam-style moments on the residual
        mom = beta1 * mom + (1.0 - beta1) * grad
        var = beta2 * var + (1.0 - beta2) * (grad * grad)

        # bias corrections
        mhat = mom / (1.0 - beta1**i)
        vhat = var / (1.0 - beta2**i)

        # parameter update
        x = x - alpha * (mhat / (np.sqrt(vhat) + eps))

        if f:
            vals.append(f(x))
        if ferr:
            err = ferr(x)
            errs.append(err)
            if err <= tol:
                break

    if ferr:
        if f:
            return x, vals, errs
        else:
            return x, errs
    else:
        if f:
            return x, vals
        else:
            return x
