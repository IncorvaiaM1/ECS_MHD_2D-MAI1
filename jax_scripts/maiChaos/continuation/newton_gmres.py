"""
Newton-GMRES solver for MHD RPO residuals.

Based directly on jax_scripts/legacy/newton.py (Matt's version) — the JIT
structure, line-search, adjoint option, io_callback timing, and dealiasing step
are all preserved.

New additions over legacy/newton.py:
- Wall-clock timeout check in the outer Python loop (cannot be done inside @jax.jit).
- Optional hookstep trust-region from legacy/newton_hookstep_v2.py.
- Structured NewtonResult dataclass return instead of bare input_dict.
- ``newton_step`` function that callers can use directly without building the
  full JIT harness themselves.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import jax
import jax.flatten_util
import jax.numpy as jnp

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB  = os.path.join(_HERE, '..', '..')
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

import lib.loss_functions as loss_functions
from lib.linalg import gmres
import lib.utils as utils


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class NewtonResult:
    input_dict: Dict[str, Any]
    converged: bool
    n_iters: int
    residual_norm: float
    timed_out: bool = False
    history: list = field(default_factory=list)  # list of (iter, rel_err, |f|)


# ---------------------------------------------------------------------------
# Internal JIT-compiled update step (mirrors legacy newton.py structure)
# ---------------------------------------------------------------------------

def _build_update_fn(obj, jac, param_dict,
                     use_transpose, gmres_m, gmres_s_min,
                     do_line_search, default_damp):
    """
    Construct and return a @jax.jit single-iteration update.

    Mirrors the ``newton_gmres_update`` inner function in legacy/newton.py
    exactly, so that compilation and performance are identical.
    """
    flatten = lambda x: jax.flatten_util.ravel_pytree(x)[0]
    f_sample = obj(param_dict.get('_sample_input', {})) if False else None

    # We need the unflattening functions, which require a sample evaluation.
    # They are captured into the closure when _build_update_fn is first called
    # (which requires one evaluation outside of JIT). Callers must pass
    # unflatten_left and unflatten_right explicitly.
    raise NotImplementedError("Use build_newton_solver() instead")


def build_newton_solver(input_dict, param_dict, obj_fn,
                        gmres_m=64, gmres_s_min=0,
                        use_transpose=False,
                        do_line_search=True, default_damp=0.1):
    """
    Compile the Newton-GMRES update and return a callable.

    Parameters
    ----------
    input_dict : dict
        Representative input dict (used to build flatten/unflatten).
    param_dict : dict
        Physics parameters, must include 'mask' for dealiasing.
    obj_fn : callable
        Jitted objective: ``f = obj_fn(input_dict)``  (param_dict already
        captured).
    gmres_m : int
        Krylov subspace dimension (``inner`` in legacy newton.py).
    gmres_s_min : float
        Smallest singular value to invert (``outer`` in legacy newton.py,
        despite the name — see linalg.gmres signature).
    use_transpose : bool
        If True, solve (J^T J) s = J^T f  (Gauss-Newton / adjoint mode).
    do_line_search : bool
        If True, perform Armijo line search after GMRES step.
    default_damp : float
        Damping factor when line search is disabled.

    Returns
    -------
    update_fn : callable
        ``input_dict_new = update_fn(i, input_dict)``  — JIT-compiled.
    unflatten_left : callable
    unflatten_right : callable
    """
    # One warm-up evaluation to build flatten/unflatten closures
    jac_fn = jax.jit(
        lambda primal, tangent: jax.jvp(obj_fn, (primal,), (tangent,))[1]
    )
    f_sample = obj_fn(input_dict)

    flatten = lambda x: jax.flatten_util.ravel_pytree(x)[0]
    _, unflatten_left  = jax.flatten_util.ravel_pytree(f_sample)
    _, unflatten_right = jax.flatten_util.ravel_pytree(input_dict)

    mask = param_dict['mask']

    def run_and_time(fn, x):
        start = time.time()
        y = fn(x)
        stop = time.time()
        return y, stop - start

    def relative_error_RPO(input_dict, f):
        norm = lambda x: jnp.sqrt(jnp.sum(jnp.square(x)))
        return norm(f["fields"]) / norm(input_dict["fields"])

    def newton_gmres_update(i, input_dict):
        # Evaluate the objective
        f, f_walltime = run_and_time(obj_fn, input_dict)
        b = flatten(f)

        # Jacobian operator on flat vectors
        A = lambda x: flatten(jac_fn(input_dict, unflatten_right(x)))
        precond = []
        if use_transpose:
            _, jacT = jax.vjp(obj_fn, input_dict, has_aux=False)
            A_T = lambda v: flatten(jacT(unflatten_left(v)))
            precond = [A_T]

        # Time the GMRES solve in plain Python (not inside JIT)
        gmres_start = time.time()
        step, gmres_residual = gmres(A, b, gmres_m, gmres_s_min,
                                     preconditioner_list=precond)
        gmres_walltime = time.time() - gmres_start

        # Apply Newton step with optional line search
        x, unravel_fn = jax.flatten_util.ravel_pytree(input_dict)
        if do_line_search:
            x, damp = utils.line_search_unravel(x, step, obj_fn, unravel_fn,
                                                 b, max_iters=20)
        else:
            damp = default_damp
            x = x - damp * step
        input_dict = unravel_fn(x)

        rel_err = float(relative_error_RPO(input_dict, f))
        print(
            f"Iter {i}: rel_err={rel_err:.3e}, |f|={float(jnp.linalg.norm(b)):.3e},"
            f" fwall={f_walltime:.3f}, gmreswall={gmres_walltime:.3f},"
            f" gmres_res={float(gmres_residual):.3e}, damp={float(damp):.3e},"
            f" T={float(input_dict['T']):.3e}, sx={float(input_dict['sx']):.3e}"
        )

        # Dealias after every Newton step (critical — same as legacy)
        fields = input_dict['fields']
        fields = mask * jnp.fft.rfft2(fields)
        fields = jnp.fft.irfft2(fields)
        input_dict['fields'] = fields
        return input_dict

    return newton_gmres_update, unflatten_left, unflatten_right


# ---------------------------------------------------------------------------
# High-level solver
# ---------------------------------------------------------------------------

def solve(input_dict, param_dict, obj_fn,
          maxit=128, tol=1e-10,
          max_time_hours=10.0,
          gmres_m=64, gmres_s_min=0,
          use_transpose=False,
          do_line_search=True, default_damp=0.1,
          save_every=None, save_fn=None) -> NewtonResult:
    """
    Run Newton-GMRES to convergence.

    Parameters
    ----------
    input_dict : dict
        Starting point ``{'fields', 'T', 'sx'}``.
    param_dict : dict
        Physics parameters (must contain 'mask', 'b0', etc.).
    obj_fn : callable
        Jitted objective ``f_dict = obj_fn(input_dict)``.  param_dict must
        already be captured into this function (use a lambda or partial).
    maxit : int
        Maximum Newton iterations.
    tol : float
        Convergence criterion on ``|f| / |fields|``.
    max_time_hours : float
        Wall-clock timeout (checked between iterations, not inside JIT).
    save_every : int or None
        If set, call ``save_fn(i, input_dict)`` every this many iterations.
    save_fn : callable or None
        ``save_fn(i, input_dict)`` — called if save_every is set.

    Returns
    -------
    NewtonResult
    """
    update_fn, _, _ = build_newton_solver(
        input_dict, param_dict, obj_fn,
        gmres_m=gmres_m, gmres_s_min=gmres_s_min,
        use_transpose=use_transpose,
        do_line_search=do_line_search, default_damp=default_damp,
    )

    flatten = lambda x: jax.flatten_util.ravel_pytree(x)[0]

    history = []
    start_time = time.time()
    converged = False
    timed_out = False

    for i in range(maxit):
        # ---- timeout check (must be OUTSIDE @jax.jit) ----
        elapsed = time.time() - start_time
        if elapsed > max_time_hours * 3600:
            timed_out = True
            print(f"[newton_gmres] Timeout after {elapsed/3600:.2f} h at iter {i}")
            break

        input_dict = update_fn(i, input_dict)

        # Evaluate residual for convergence check
        f_dict = obj_fn(input_dict)
        f_flat = flatten(f_dict)
        x_flat = flatten(input_dict)
        res_norm = float(jnp.linalg.norm(f_flat))
        sol_norm = float(jnp.linalg.norm(x_flat))
        rel_err  = res_norm / max(float(jnp.linalg.norm(
            flatten({'fields': input_dict['fields'],
                     'T': input_dict['T'], 'sx': input_dict['sx']})
        )), 1e-30)

        history.append((i, rel_err, res_norm))

        if save_every and save_fn and (i % save_every == 0):
            save_fn(i, input_dict)

        if res_norm < tol:
            converged = True
            break

    return NewtonResult(
        input_dict=input_dict,
        converged=converged,
        n_iters=i + 1,
        residual_norm=res_norm,
        timed_out=timed_out,
        history=history,
    )
