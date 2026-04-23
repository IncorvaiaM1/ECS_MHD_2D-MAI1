"""
Tangent predictor and adaptive step-size control for arc-length continuation.

The "tangent" here is a unit vector in the extended space (input_dict, b0_y):

    tangent = [dz_flat / ds,  d(b0_y) / ds]

where ``z_flat = flatten(input_dict)`` and ``b0_y`` is the continuation
parameter (mean magnetic field in y).

Step-size is adapted based on the number of Newton iterations required at
the previous step: fewer iters → grow ds, more → shrink.
"""

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple


def flatten_state(input_dict: Dict[str, Any], b0_y: float) -> np.ndarray:
    """
    Pack (input_dict, b0_y) into a single 1-D numpy array.

    The b0_y scalar is appended as the last element.
    """
    flat, _ = jax.flatten_util.ravel_pytree(input_dict)
    return np.append(np.array(flat), float(b0_y))


def unflatten_state(z: np.ndarray, unravel_fn) -> Tuple[Dict, float]:
    """
    Unpack a flat vector produced by ``flatten_state``.

    Returns (input_dict, b0_y).
    """
    import jax.numpy as jnp
    input_dict = unravel_fn(jnp.array(z[:-1]))
    b0_y = float(z[-1])
    return input_dict, b0_y


def compute_unravel_fn(input_dict: Dict[str, Any]):
    """Return the unravel function for input_dict (captures structure only)."""
    _, unravel_fn = jax.flatten_util.ravel_pytree(input_dict)
    return unravel_fn


def compute_tangent(z_flat: np.ndarray,
                    z_prev_flat: np.ndarray,
                    tangent_prev: np.ndarray) -> np.ndarray:
    """
    Estimate the new tangent by finite-differencing two consecutive solutions,
    then orient it to be consistent with the previous tangent (dot > 0).

    Parameters
    ----------
    z_flat      : current extended state vector
    z_prev_flat : previous extended state vector
    tangent_prev : previous tangent (for orientation)

    Returns
    -------
    tangent : unit vector in extended state space
    """
    dz = z_flat - z_prev_flat
    norm = np.linalg.norm(dz)
    if norm < 1e-14:
        # Degenerate: reuse previous tangent
        return tangent_prev.copy()
    tangent = dz / norm
    # Orient consistently with previous direction
    if np.dot(tangent, tangent_prev) < 0:
        tangent = -tangent
    return tangent


def predict(z_flat: np.ndarray,
            tangent: np.ndarray,
            ds: float) -> np.ndarray:
    """
    Euler (secant) predictor:  z_pred = z + ds * tangent.
    """
    return z_flat + ds * tangent


def adapt_stepsize(ds: float,
                   newton_iters: int,
                   converged: bool,
                   target_iters: int = 4,
                   ds_min: float = 1e-5,
                   ds_max: float = 0.05,
                   ds_grow: float = 1.2,
                   ds_shrink: float = 0.5) -> float:
    """
    Adapt the arc-length step size.

    Logic:
    - Failure      → shrink by ds_shrink
    - Converged, many iters  → shrink slightly (hard step)
    - Converged, few iters   → grow by ds_grow (easy step)

    Parameters
    ----------
    ds           : current step size
    newton_iters : number of Newton iterations used
    converged    : whether Newton converged
    target_iters : target Newton iterations (grow if < target, shrink if > 2*target)
    ds_min, ds_max, ds_grow, ds_shrink : bounds and factors

    Returns
    -------
    ds_new : float
    """
    if not converged:
        ds_new = ds * ds_shrink
    elif newton_iters < target_iters:
        ds_new = ds * ds_grow
    elif newton_iters > 2 * target_iters:
        ds_new = ds * ds_shrink
    else:
        ds_new = ds  # no change

    return float(np.clip(ds_new, ds_min, ds_max))
