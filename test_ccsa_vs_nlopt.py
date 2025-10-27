import numpy as np
import nlopt
from ccsa_mma import MMAOptimizer
import matplotlib.pyplot as plt

# Deterministic quadratic problem: min_x 0.5 * x^T A x + b^T x
n = 10
A = np.eye(n)
b = np.arange(1, n+1)

# Objective function for nlopt (returns value and gradient)
nlopt_traj = []
def nlopt_obj(x, grad):
    if grad.size > 0:
        grad[:] = A @ x + b
    val = 0.5 * x @ A @ x + b @ x
    nlopt_traj.append(val)
    return val

# Objective function for MMAOptimizer (returns value and gradient)
ccsa_traj = []
def mma_fun(x, grad=True):
    val = 0.5 * x @ A @ x + b @ x
    ccsa_traj.append(val)
    if grad:
        return val, A @ x + b
    else:
        return val

# Initial point
x0 = np.ones(n)


opt = nlopt.opt(nlopt.LD_MMA, n)
opt.set_min_objective(nlopt_obj)
opt.set_maxeval(100)
x_nlopt = opt.optimize(x0.copy())
val_nlopt = opt.last_optimum_value()
print("nlopt MMA result:")
print("x* =", x_nlopt)
print("f(x*) =", val_nlopt)


optimizer = MMAOptimizer(
    params=x0,
    fun=mma_fun,
    g=None,
    bounds=None,
    rho_init=1.0,
    max_inner=5,
    sigma_min=1e-6,
    sigma_max=1e20,
    df=None,
    dg=None,
    x0=x0
)
for i in range(100):
    f_best, g_best, metrics = optimizer.step()
x_ccsa = optimizer.x
val_ccsa = mma_fun(x_ccsa, grad=False)
print("Custom MMAOptimizer result:")
print("x* =", x_ccsa)
print("f(x*) =", val_ccsa)


print("Difference in x*:", np.linalg.norm(x_nlopt - x_ccsa))
print("Difference in f(x*):", abs(val_nlopt - val_ccsa))


plt.figure(figsize=(8,5))
plt.plot(nlopt_traj, label='nlopt MMA', marker='o')
plt.plot(ccsa_traj, label='Custom MMAOptimizer', marker='x')
plt.xlabel('Iteration')
plt.ylabel('Objective value')
plt.xscale('log')
plt.title('Optimization Trajectories')
plt.legend()
plt.grid(True)
plt.show()


diff_curve = np.abs(np.array(nlopt_traj[:len(ccsa_traj)]) - np.array(ccsa_traj))
plt.figure(figsize=(8,5))
plt.plot(diff_curve, label='|nlopt - custom|', color='red', marker='s')
plt.xlabel('Iteration')
plt.ylabel('Absolute difference in objective')
plt.xscale('log')
plt.title('Difference in Objective Value per Iteration')
plt.legend()
plt.grid(True)
plt.show()
