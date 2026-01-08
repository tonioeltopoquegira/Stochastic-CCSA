import numpy as np
import matplotlib.pyplot as plt

# Useful plotting utils 
def make_red_colors(k):
    cmap = plt.cm.plasma
    if k > 1:
        return [cmap(0.3 + 0.6 * i/(k-1)) for i in range(k)]
    else:
        return [cmap(0.3 + 0.6 * 0)]
    

# random nxn matrix with condition number κ and log-spaced singular values
def randcond(n, κ):
    Q, R = np.linalg.qr(np.random.randn(n, n))
    Q = Q @ np.diag(np.sign(np.diag(R))) # randomize signs
    σ = np.logspace(np.log10(1), np.log10(1/κ), num=n)
    return Q @ np.diag(σ) @ Q.T