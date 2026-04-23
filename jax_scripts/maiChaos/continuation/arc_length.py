"""
Pseudo arc-length continuation for MHD RPOs.

Tracks a solution branch as the mean magnetic field B₀_y varies.

The augmented system at each step:

    G(z, λ)    = objective_RPO(unpack(z), params_with_b0(λ)) = 0
    N(z, λ; s) = (z_aug - z_aug_prev) · t_prev - ds          = 0

where ``z_aug = [flatten(input_dict), λ]`` and ``λ = b0_y``.

Newton is applied to the (n+1)-dimensional augmented system using a
matrix-free GMRES: the Jacobian of the augmented residual w.r.t. z_aug
is never explicitly formed; instead ``jax.jvp`` provides matrix-vector
products.

Conventions
-----------
- ``input_dict`` keys: ``{'fields', 'T', 'sx'}``
- ``param_dict``  keys include: ``'b0'``, ``'mask'``, ``'nu'``, ``'eta'``, ...
- ``b0_y`` is ``param_dict['b0'][1]`` (the y-component of the mean field).
- Dealiasing is applied after each Newton iteration (mirrors legacy newton.py).
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB  = os.path.join(_HERE, '..', '..')
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

import lib.loss_functions as loss_functions
from lib.linalg import gmres as lib_gmres

from .predictor import (
    flatten_state, unflatten_state,
    compute_unravel_fn, compute_tangent, predict, adapt_stepsize,
)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class ArcLengthStepResult:
    input_dict: Dict[str, Any]
    b0_y: float
    tangent: np.ndarray          # updated unit tangent in (z, b0_y) space
    ds_used: float               # arc-length step actually taken
    converged: bool
    n_iters: int
    residual_norm: float
    timed_out: bool = False
    history: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: build param_dict with a different b0_y
# ---------------------------------------------------------------------------

def _update_b0(param_dict: Dict, b0_y: float) -> Dict:
    """Return a shallow copy of param_dict with b0[1] set to b0_y."""
    new_pd = dict(param_dict)
    b0_old = param_dict['b0']
    new_pd['b0'] = jnp.array([float(b0_old[0]), float(b0_y)])
    return new_pd


# ---------------------------------------------------------------------------
# Augmented residual
# ---------------------------------------------------------------------------

def make_augmented_residual(param_dict_base: Dict, unravel_fn,
                             z_prev_aug: np.ndarray,
                             tangent_prev: np.ndarray,
                             ds: float,
                             obj_mode: str = "RPO"):
    """
    Return a function ``G_aug(z_aug) → residual_vector`` for the augmented
    continuation system.

    Parameters
    ----------
    param_dict_base : dict
        Physics parameters (b0 will be overwritten for each evaluation).
    unravel_fn : callable
        Unflattens the state portion of z_aug back to input_dict.
    z_prev_aug : (n+1,) array
        Previous augmented state vector.
    tangent_prev : (n+1,) array
        Previous unit tangent vector.
    ds : float
        Desired arc-length step.
    obj_mode : str
        Integration mode passed to objective_RPO (default "RPO").

    Returns
    -------
    G_aug : callable
        Takes z_aug (jnp array, shape (n+1,)) and returns residual
        (jnp array, shape (n+1,)).
    """
    z_prev_jnp   = jnp.array(z_prev_aug)
    tangent_jnp  = jnp.array(tangent_prev)
    ds_val       = float(ds)

    def G_aug(z_aug: jnp.ndarray) -> jnp.ndarray:
        # Split state and parameter
        z_flat = z_aug[:-1]
        b0_y   = z_aug[-1]

        # Unpack into input_dict
        input_dict = unravel_fn(z_flat)

        # Build param_dict with updated b0_y
        pd = _update_b0(param_dict_base, b0_y)

        # Evaluate MHD residual (objective_RPO)
        out_dict = loss_functions.objective_RPO(input_dict, pd)

        # Flatten the output dict to a vector (same structure as input_dict)
        F_flat, _ = jax.flatten_util.ravel_pytree(out_dict)

        # Arc-length constraint
        N = jnp.dot(z_aug - z_prev_jnp, tangent_jnp) - ds_val

        return jnp.concatenate([F_flat, jnp.array([N])])

    return G_aug


# ---------------------------------------------------------------------------
# Matrix-free JVP operator
# ---------------------------------------------------------------------------

def jvp_matvec(G_aug, z_aug: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Compute J(G_aug) @ v using JAX forward-mode AD.

    This avoids explicitly forming the (n+1)×(n+1) Jacobian.
    """
    _, Jv = jax.jvp(G_aug, (z_aug,), (v,))
    return Jv


# ---------------------------------------------------------------------------
# One arc-length step
# ---------------------------------------------------------------------------

def arc_length_step(
    input_dict:   Dict[str, Any],
    b0_y:         float,
    tangent_prev: np.ndarray,
    param_dict:   Dict[str, Any],
    ds:           float,
    gmres_m:      int   = 50,
    gmres_s_min:  float = 0.0,
    max_newton_iter: int   = 128,
    newton_tol:   float = 1e-10,
    max_time_hours: float = 10.0,
    obj_mode:     str   = "RPO",
) -> ArcLengthStepResult:
    """
    Advance one pseudo arc-length step.

    Parameters
    ----------
    input_dict   : current solution dict ``{'fields', 'T', 'sx'}``.
    b0_y         : current continuation parameter value.
    tangent_prev : unit tangent vector from the previous step (len = n+1).
    param_dict   : physics parameters (b0 is overridden internally).
    ds           : desired arc-length step.
    gmres_m      : Krylov dimension for GMRES.
    gmres_s_min  : singular-value threshold in GMRES (0 = lstsq).
    max_newton_iter : Newton iteration limit.
    newton_tol   : convergence tolerance on |G_aug|.
    max_time_hours : wall-clock timeout.
    obj_mode     : integration mode for objective_RPO.

    Returns
    -------
    ArcLengthStepResult
    """
    # ---- pack previous point ----
    unravel_fn = compute_unravel_fn(input_dict)
    z_prev_aug = flatten_state(input_dict, b0_y)

    # ---- predictor ----
    z_pred_aug = predict(z_prev_aug, tangent_prev, ds)

    # ---- build augmented residual ----
    G_aug = make_augmented_residual(
        param_dict, unravel_fn, z_prev_aug, tangent_prev, ds, obj_mode
    )
    G_aug_jit = jax.jit(G_aug)

    # Jitted JVP operator
    jvp_fn = jax.jit(lambda z, v: jvp_matvec(G_aug_jit, z, v))

    # ---- Newton-GMRES on augmented system ----
    z_aug = jnp.array(z_pred_aug)
    history = []
    start_time = time.time()
    converged = False
    timed_out = False
    res_norm  = float('inf')
    mask      = param_dict['mask']

    for i in range(max_newton_iter):
        # Timeout check (outside JIT)
        if time.time() - start_time > max_time_hours * 3600:
            timed_out = True
            break

        # Residual
        G_val = G_aug_jit(z_aug)
        res_norm = float(jnp.linalg.norm(G_val))

        print(f"  [arc_len] iter {i:3d}: |G| = {res_norm:.3e},  b0_y = {float(z_aug[-1]):.6f}")
        history.append((i, res_norm))

        if res_norm < newton_tol:
            converged = True
            break

        # Matrix-free GMRES: solve J @ dz = -G_val
        b_rhs = jnp.array(G_val)
        A_op  = lambda v: jvp_fn(z_aug, jnp.array(v))

        dz, gmres_relres = lib_gmres(A_op, b_rhs, gmres_m, gmres_s_min)

        # Update
        z_aug = z_aug - dz

        # Dealias the state portion (mirrors legacy newton.py)
        z_np    = np.array(z_aug)
        id_tmp  = unravel_fn(jnp.array(z_np[:-1]))
        f       = id_tmp['fields']
        f       = mask * jnp.fft.rfft2(f)
        f       = jnp.fft.irfft2(f)
        id_tmp['fields'] = f
        flat_dealiased, _ = jax.flatten_util.ravel_pytree(id_tmp)
        z_aug = jnp.concatenate([flat_dealiased, z_aug[-1:]])

    # ---- unpack result ----
    z_np   = np.array(z_aug)
    input_dict_new, b0_y_new = unflatten_state(z_np, unravel_fn)

    # ---- update tangent ----
    tangent_new = compute_tangent(z_np, z_prev_aug, tangent_prev)

    return ArcLengthStepResult(
        input_dict   = input_dict_new,
        b0_y         = b0_y_new,
        tangent      = tangent_new,
        ds_used      = ds,
        converged    = converged,
        n_iters      = i + 1,
        residual_norm= res_norm,
        timed_out    = timed_out,
        history      = history,
    )


# ---------------------------------------------------------------------------
# Initial tangent (for the very first step)
# ---------------------------------------------------------------------------

def compute_initial_tangent(
    input_dict: Dict[str, Any],
    b0_y:       float,
    param_dict: Dict[str, Any],
    db0:        float = 1e-4,
) -> np.ndarray:
    """
    Estimate the initial tangent by finite-differencing in the b0_y direction.

    We perturb b0_y by ``db0``, evaluate the residual Jacobian w.r.t. b0_y
    analytically, then build the nullspace direction of the Jacobian.

    In practice, for the first step we just point "in the direction of
    increasing b0_y" by setting d(b0_y)/ds > 0 and d(z)/ds ≈ 0.
    This is sufficient as a starting guess; the Newton corrector will fix it.

    Parameters
    ----------
    input_dict  : starting solution dict.
    b0_y        : current b0_y value.
    param_dict  : physics params.
    db0         : perturbation size for finite-difference (unused here, kept
                  for future higher-order estimates).

    Returns
    -------
    tangent : unit vector of length n+1, pointing in +b0_y direction.
    """
    flat, _ = jax.flatten_util.ravel_pytree(input_dict)
    n = len(np.array(flat))
    tangent = np.zeros(n + 1, dtype=np.float64)
    tangent[-1] = 1.0  # initial direction: increase b0_y
    return tangent
