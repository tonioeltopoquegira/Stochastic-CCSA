import numpy as np
from dataclasses import dataclass

@dataclass
class MMA_RhoParams:
    softening: float = 0.5       # additive softening factor
    min_growth: float = 1.01     # multiplicative bump per iteration
    max_multiplier: float = 5.0  # max factor rho can grow per inner step
    decay: float = 0.99          # multiplicative decay per outer iteration for rho (objective) # best 0.4
    decay_c: float = 0.3    # multiplicative decay per outer iteration for rho_c (constraints)
                                 # If None, uses same value as decay

@dataclass
class MMA_SigmaParams:
    expand: float = 1.2          # expansion multiplier on consistent movement
    contract: float = 0.7        # contraction multiplier on oscillation
    sigma_min: float = 1e-6      # absolute minimal sigma (numerical floor)
    rel_min: float = 0.01        # sigma >= rel_min * (ub - lb) when finite bounds exist
    rel_max: float = 10.0        # sigma <= rel_max * (ub - lb) when finite bounds exist
   
    

def multiplier_update_rho(rho, gap, w_val, rho_params: MMA_RhoParams):
    """
    Vectorized and scalar-safe rho update.
    """
    incr = gap / max(w_val, 1e-12)
    rho_new = rho + rho_params.softening * incr

    # upper growth cap: rho_new ≤ max_multiplier * rho
    rho_cap_max = rho_params.max_multiplier * rho

    # elementwise min
    rho_new = np.minimum(rho_cap_max, rho_new)

    rho_cap_min = rho * (1/rho_params.max_multiplier)

    rho_new = np.maximum(rho_cap_min, rho_new)

    #print(rho_new)

    rho_new = max(rho_new, 1e-4)

    return rho_new


# Adam-like curvature updater helpers
@dataclass
class AdamCurvParams:
    lr: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    min_curv: float = 1e-6
    max_curv: float = 1e3


def init_adam_curv_state():
    """Initialize Adam-like state for curvature adaptation."""
    return {'m': 0.0, 'v': 0.0, 't': 0, 'last_grad': None}


def adam_curv_update(state: dict, grad: np.ndarray, curv: float, params: AdamCurvParams):
    """
    Perform one Adam-like update for a scalar curvature parameter.

    Args:
      state: dict returned by `init_adam_curv_state` (will be updated and returned)
      grad: gradient vector at current iterate (numpy array)
      curv: current scalar curvature value
      params: AdamCurvParams controlling lr, betas, eps and clipping

    Returns:
      (new_curv, new_state)
    """
    import numpy as _np

    if grad is None:
        return float(curv), state

    grad = _np.asarray(grad, dtype=float).ravel()

    if state.get('last_grad') is None:
        # store and return unchanged curvature
        new_state = dict(state)
        new_state['last_grad'] = grad.copy()
        return float(curv), new_state

    delta = grad - _np.asarray(state['last_grad'], dtype=float).ravel()
    g_cur = float(_np.dot(delta, delta))

    m = float(state.get('m', 0.0))
    v = float(state.get('v', 0.0))
    t = int(state.get('t', 0))

    t += 1
    m = params.beta1 * m + (1.0 - params.beta1) * g_cur
    v = params.beta2 * v + (1.0 - params.beta2) * (g_cur ** 2)

    # bias correction
    m_hat = m / (1.0 - (params.beta1 ** t)) if t > 0 else m
    v_hat = v / (1.0 - (params.beta2 ** t)) if t > 0 else v

    step = params.lr * m_hat / (_np.sqrt(v_hat) + params.eps)

    new_curv = float(curv + step)
    # clip
    new_curv = max(params.min_curv, min(params.max_curv, new_curv))

    new_state = {'m': m, 'v': v, 't': t, 'last_grad': grad.copy()}
    return new_curv, new_state


def init_adam_curv_state():
    """Initialize Adam-like state for curvature adaptation (violation or secant)."""
    return {'m': 0.0, 'v': 0.0, 't': 0}


def adam_secant_update(state: dict, grad_new: np.ndarray, grad_old: np.ndarray,
                       x_new: np.ndarray, x_old: np.ndarray,
                       curv: float, params: AdamCurvParams):
    """
    Adam-smoothed diagonal secant curvature estimate.

    Computes per-component secant: s_i = (grad_new_i - grad_old_i) / (x_new_i - x_old_i)
    then aggregates to a scalar via max over active components.
    Signed: positive = locally convex -> increase rho
            negative = locally concave -> decrease rho

    Args:
        state:    dict with keys 'm', 'v', 't'
        grad_new: gradient at x_new (numpy array, shape (n,))
        grad_old: gradient at x_old (numpy array, shape (n,))
        x_new:    new iterate (numpy array, shape (n,))
        x_old:    previous iterate (numpy array, shape (n,))
        curv:     current scalar curvature (rho)
        params:   AdamCurvParams

    Returns:
        (new_curv, new_state)
    """
    dx = np.asarray(x_new, dtype=float) - np.asarray(x_old, dtype=float)
    dg = np.asarray(grad_new, dtype=float) - np.asarray(grad_old, dtype=float)

    active = np.abs(dx) > 1e-12
    if not np.any(active):
        return float(curv), state   # no movement, skip update

    secant = np.where(active, dg / np.where(active, dx, 1.0), 0.0)
    g_cur = float(np.max(secant[active]))   # signed scalar: worst-case component

    t = state['t'] + 1
    m = params.beta1 * state['m'] + (1.0 - params.beta1) * g_cur
    v = params.beta2 * state['v'] + (1.0 - params.beta2) * (g_cur ** 2)

    m_hat = m / (1.0 - params.beta1 ** t)
    v_hat = v / (1.0 - params.beta2 ** t)

    step = params.lr * m_hat / (np.sqrt(v_hat) + params.eps)
    new_curv = float(np.clip(curv + step, params.min_curv, params.max_curv))

    return new_curv, {'m': m, 'v': v, 't': t}

def adam_secant_update_percoord(states: list, grad_new: np.ndarray, grad_old: np.ndarray,
                                x_new: np.ndarray, x_old: np.ndarray,
                                curv_vec: np.ndarray, params: AdamCurvParams):
    """
    Per-coordinate Adam-secant update. No aggregation — each coordinate updated independently.
    Inactive coordinates (|dx_j| <= 1e-12) are skipped and keep their current rho_j.

    Args:
        states:   list of n dicts, one Adam state per coordinate
        grad_new: gradient at x_new, shape (n,)
        grad_old: gradient at x_old, shape (n,)
        x_new:    new iterate, shape (n,)
        x_old:    previous iterate, shape (n,)
        curv_vec: current per-coordinate curvature, shape (n,)
        params:   AdamCurvParams

    Returns:
        (new_curv_vec, new_states)
    """
    dx = np.asarray(x_new, dtype=float) - np.asarray(x_old, dtype=float)
    dg = np.asarray(grad_new, dtype=float) - np.asarray(grad_old, dtype=float)
    active = np.abs(dx) > 1e-12

    new_curv = curv_vec.copy()
    new_states = list(states)

    for j in range(len(curv_vec)):
        if not active[j]:
            continue
        s_j = float(dg[j] / dx[j])   # signed scalar secant for coordinate j
        new_curv[j], new_states[j] = adam_curv_update(
            states[j], s_j, float(curv_vec[j]), params
        )

    return new_curv, new_states