import numpy as np
from dataclasses import dataclass

@dataclass
class MMA_RhoParams:
    softening: float = 0.5       # additive softening factor
    min_growth: float = 1.01     # multiplicative bump per iteration
    max_multiplier: float = 10.0  # max factor rho can grow per inner step
    decay: float = 0.1          # multiplicative decay per outer iteration

@dataclass
class MMA_SigmaParams:
    expand: float = 1.2          # expansion multiplier on consistent movement
    contract: float = 0.7        # contraction multiplier on oscillation
    sigma_min: float = 1e-6      # absolute minimal sigma (numerical floor)
    rel_min: float = 0.01        # sigma >= rel_min * (ub - lb) when finite bounds exist
    rel_max: float = 10.0        # sigma <= rel_max * (ub - lb) when finite bounds exist
   
    

def update_rho(rho, gap, w_val, rho_params: MMA_RhoParams):
    """
    Vectorized and scalar-safe rho update.
    """
    incr = gap / max(w_val, 1e-12)
    rho_candidate = rho + rho_params.softening * incr

    # upper growth cap: rho_new ≤ max_multiplier * rho
    rho_cap = rho_params.max_multiplier * rho

    # elementwise min
    rho_new = np.minimum(rho_cap, rho_candidate)

    # lower floor: rho_new ≥ rho
    rho_new = np.maximum(rho, rho_new)

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


def adam_curv_update_scalar(state: dict, signal: float, curv: float, params: AdamCurvParams):
        """
        Adam-like update for scalar signal (e.g., per-constraint violation magnitude).

        Args:
            state: dict with keys 'm','v','t','last_signal' (will be updated)
            signal: scalar signal (float)
            curv: current scalar curvature
            params: AdamCurvParams

        Returns:
            (new_curv, new_state)
        """
        import math as _math

        if signal is None:
                return float(curv), state

        # initialize storage
        last = state.get('last_signal', None)
        if last is None:
                new_state = dict(state)
                new_state['last_signal'] = float(signal)
                return float(curv), new_state

        delta = float(signal) - float(last)
        g_cur = float(delta * delta)

        m = float(state.get('m', 0.0))
        v = float(state.get('v', 0.0))
        t = int(state.get('t', 0))

        t += 1
        m = params.beta1 * m + (1.0 - params.beta1) * g_cur
        v = params.beta2 * v + (1.0 - params.beta2) * (g_cur ** 2)

        # bias correction
        m_hat = m / (1.0 - (params.beta1 ** t)) if t > 0 else m
        v_hat = v / (1.0 - (params.beta2 ** t)) if t > 0 else v

        step = params.lr * m_hat / (_math.sqrt(v_hat) + params.eps)

        new_curv = float(curv + step)
        new_curv = max(params.min_curv, min(params.max_curv, new_curv))

        new_state = {'m': m, 'v': v, 't': t, 'last_signal': float(signal)}
        return new_curv, new_state

