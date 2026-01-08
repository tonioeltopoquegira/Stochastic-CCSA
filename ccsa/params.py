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

