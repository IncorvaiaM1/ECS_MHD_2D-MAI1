"""
Floquet analysis for MHD RPOs.

Migrated from jax_scripts/legacy/floquet.py and made callable as a function
so it can be invoked from the sweep loop or as a standalone script.

The Floquet multipliers are estimated by power iteration of the linearised
map M = J_T (the monodromy operator), restricted to a ``block_size``-dimensional
subspace.  After convergence we return the Schur factor R = V @ (J_T V)^T where
V is the converged orthonormal tangent block.

The dominant multipliers are the eigenvalues of R.

References
----------
- Cvitanovic et al., "Chaos: Classical and Quantum" (chaosbook.org)
- Goldhirsch, Sulem, Orszag (1987): "Stability and Lyapunov stability of
  dynamical systems: A differential approach and a numerical method"
"""

import time
import os
import sys
from typing import Dict, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB  = os.path.join(_HERE, '..', '..')
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

import lib.mhd_jax as mhd_jax
import lib.dictionaryIO as dictionaryIO


def run_floquet(
    input_dict:  Dict[str, Any],
    param_dict:  Dict[str, Any],
    block_size:  int = 32,
    maxit:       int = 8,
    seed:        int = 0,
    tang_init:   Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Estimate the leading Floquet multipliers of an RPO.

    Parameters
    ----------
    input_dict  : RPO solution ``{'fields', 'T', 'sx'}``.
    param_dict  : physics parameters (must include 'steps').
    block_size  : number of tangent vectors to iterate.
    maxit       : number of power iterations.
    seed        : PRNG seed for initial random tangent block.
    tang_init   : optional pre-computed initial tangent block
                  of shape ``(block_size, 2, n, n)``.

    Returns
    -------
    dict with keys:
        R          : (block_size, block_size) Schur factor matrix (numpy)
        eigenvalues: (block_size,) dominant Floquet multipliers (numpy, complex)
        tang       : (block_size, 2, n, n) final tangent block (numpy)
        rel_error  : relative error in the periodic orbit (float)
    """
    jax.config.update("jax_enable_x64", True)

    f  = input_dict['fields']
    T  = input_dict['T']
    sx = input_dict['sx']

    if f.ndim == 4:
        f = f[0, ...]  # restrict to single-shooting

    steps = int(param_dict['steps'])
    dt    = float(T) / steps
    n     = f.shape[-1]
    precision = jnp.float64

    @jax.jit
    def forward(f):
        """Advance one full period and apply spatial shift."""
        f_ = jnp.fft.rfft2(f)
        f_ = mhd_jax.eark4(f_, dt, steps, param_dict)
        f_ = jnp.exp(-1j * param_dict['kx'] * sx) * f_
        return jnp.fft.irfft2(f_)

    # Verify periodicity
    f_out = forward(f)
    norm      = lambda x: jnp.linalg.norm(jnp.reshape(x, [-1]))
    rel_error = float(norm(f - f_out) / norm(f))
    print(f"[floquet] relative error in periodic orbit = {rel_error:.3e}")

    # Jacobian action via JVP, vmapped over the tangent block
    jac = jax.jit(jax.vmap(lambda tang: jax.jvp(forward, (f,), (tang,))[1]))

    # Initialise tangent block
    if tang_init is not None:
        tang = jnp.array(tang_init)
    else:
        key  = jax.random.PRNGKey(seed)
        tang = jax.random.normal(key, [block_size, 2, n, n], dtype=precision)

    # Power iteration with QR re-orthogonalisation
    for i in range(maxit):
        start = time.time()
        tang  = jac(tang)
        print(f"[floquet] iter {i}: walltime = {time.time() - start:.3f}s")

        tang = jnp.reshape(tang, [block_size, -1])
        tang, _ = jnp.linalg.qr(tang.transpose(), mode="reduced")
        tang    = tang.transpose()
        tang    = jnp.reshape(tang, [block_size, 2, n, n])

    # Estimate Schur factor R from the converged subspace
    t  = jnp.reshape(tang,       [block_size, -1])
    jt = jnp.reshape(jac(tang),  [block_size, -1])
    R  = np.array(t @ jt.transpose())

    eigenvalues = np.linalg.eigvals(R)
    # Sort descending by magnitude
    idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]

    print(f"[floquet] dominant multiplier |μ₁| = {np.abs(eigenvalues[0]):.4f}")

    return {
        "R":           R,
        "eigenvalues": eigenvalues,
        "tang":        np.array(tang),
        "rel_error":   rel_error,
    }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main(solution_file: str, output_path: Optional[str] = None,
         block_size: int = 32, maxit: int = 8):
    input_dict, param_dict = dictionaryIO.load_dicts(solution_file)
    data = run_floquet(input_dict, param_dict,
                       block_size=block_size, maxit=maxit)

    if output_path is None:
        base = os.path.splitext(solution_file)[0]
        output_path = base + "_floquet.npz"

    np.savez(output_path, **{k: np.array(v) for k, v in data.items()
                              if not isinstance(v, dict)})
    print(f"[floquet] Saved to {output_path}")
    return data
