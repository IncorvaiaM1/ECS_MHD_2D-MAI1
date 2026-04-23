"""
Automated arc-length continuation sweep over B₀_y.

For each seed solution in the converted solutions directory the sweep:

1.  Loads the seed solution.
2.  Runs Newton-GMRES to re-converge at the starting B₀_y (in case the seed
    is not fully converged).
3.  Steps in B₀_y using pseudo arc-length continuation until B₀_y reaches
    ``b0_target`` or continuation fails.
4.  Optionally runs Floquet analysis and animation at each converged point.
5.  Logs all results to a CSV and saves checkpoints for resumption.

Usage
-----
    from maiChaos.sweep.automated_sweep import run_sweep
    run_sweep("maiChaos/config/default_config.yaml")

or via the entry-point script::

    python jax_scripts/maiChaos/scripts/run_sweep.py \\
        --config jax_scripts/maiChaos/config/default_config.yaml
"""

import os
import sys
import time
import glob
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB  = os.path.join(_HERE, '..', '..')
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

import lib.mhd_jax as mhd_jax
import lib.loss_functions as loss_functions
import lib.dictionaryIO as dictionaryIO
import lib.utils as utils

from maiChaos.continuation.arc_length import (
    arc_length_step, compute_initial_tangent, ArcLengthStepResult,
)
from maiChaos.continuation.predictor import adapt_stepsize
from maiChaos.continuation.newton_gmres import solve as newton_solve
from maiChaos.sweep.checkpoint import Checkpoint
from maiChaos.io.data_manager import DataManager
from maiChaos.io.results_log import ResultsLog
from maiChaos.analysis.bifurcation_detect import detect_bifurcation


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config and return as a nested dict."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Domain setup
# ---------------------------------------------------------------------------

def build_param_dict(cfg: Dict, b0_y: float) -> Dict:
    """Construct param_dict from config."""
    phys = cfg['physics']
    n    = phys['n']

    jax.config.update("jax_enable_x64", True)
    precision = jnp.float64

    param_dict = mhd_jax.construct_domain(n, precision)
    y = param_dict['y']

    forcing = -4.0 * jnp.cos(4.0 * y)   # hard-coded Kolmogorov forcing
    b0 = jnp.array([float(phys['b0_start'][0]), float(b0_y)])

    steps     = int(phys.get('steps', 1024))
    ministeps = int(phys.get('ministeps', 32))
    assert steps % ministeps == 0

    param_dict.update({
        'nu':              float(phys['nu']),
        'eta':             float(phys['eta']),
        'b0':              b0,
        'forcing':         forcing,
        'steps':           steps,
        'ministeps':       ministeps,
        'num_checkpoints': steps // ministeps,
    })
    return param_dict


# ---------------------------------------------------------------------------
# Per-solution sweep
# ---------------------------------------------------------------------------

def _sweep_one_solution(
    sol_idx:     int,
    input_dict:  Dict,
    param_dict:  Dict,
    cfg:         Dict,
    dm:          DataManager,
    log:         ResultsLog,
    ckpt:        Checkpoint,
) -> None:
    """
    Run the full arc-length sweep for a single seed solution.

    Resumes from checkpoint if one exists for this sol_idx.
    """
    cont_cfg  = cfg['continuation']
    phys_cfg  = cfg['physics']
    gmres_cfg = cfg['gmres']
    ana_cfg   = cfg['analysis']

    b0_target = float(phys_cfg['b0_target'])
    ds         = float(cont_cfg['ds_init'])
    ds_min     = float(cont_cfg['ds_min'])
    ds_max     = float(cont_cfg['ds_max'])
    ds_grow    = float(cont_cfg['ds_grow'])
    ds_shrink  = float(cont_cfg['ds_shrink'])
    max_iter   = int(cont_cfg['max_newton_iter'])
    newton_tol = float(cont_cfg['newton_tol'])
    max_hours  = float(cont_cfg['max_time_hours'])
    gmres_m    = int(gmres_cfg['restart'])
    gmres_smin = 0.0

    # ----------------------------------------------------------------
    # Resume from checkpoint?
    # ----------------------------------------------------------------
    ckpt_state = ckpt.load()
    if ckpt_state is not None and ckpt_state.get('sol_idx') == sol_idx:
        print(f"[sweep] Resuming sol_{sol_idx:03d} from checkpoint "
              f"(b0_y={ckpt_state['b0_y']:.4f}, step={ckpt_state['step_count']})")
        b0_y    = float(ckpt_state['b0_y'])
        ds      = float(ckpt_state['ds'])
        tangent = ckpt_state['tangent']
        # Reload the last converged solution
        input_dict, param_dict = dictionaryIO.load_dicts(ckpt_state['solution_path'])
        param_dict = build_param_dict(cfg, b0_y)
        step_count = int(ckpt_state['step_count'])
    else:
        b0_y       = float(param_dict['b0'][1])
        tangent    = compute_initial_tangent(input_dict, b0_y, param_dict)
        step_count = 0

    # ----------------------------------------------------------------
    # Step 0: re-converge seed with Newton (may already be converged)
    # ----------------------------------------------------------------
    print(f"\n[sweep] sol_{sol_idx:03d}: re-converging seed at b0_y={b0_y:.4f}")
    obj_fn = jax.jit(
        lambda id_: loss_functions.objective_RPO(id_, param_dict)
    )
    nr = newton_solve(
        input_dict, param_dict, obj_fn,
        maxit=max_iter, tol=newton_tol,
        max_time_hours=max_hours,
        gmres_m=gmres_m, gmres_s_min=gmres_smin,
    )
    input_dict = nr.input_dict
    if not nr.converged:
        print(f"[sweep] Seed convergence failed — skipping sol_{sol_idx:03d}")
        return

    # Save seed solution
    path = dm.save_solution(sol_idx, b0_y, input_dict, param_dict)
    _maybe_run_analysis(sol_idx, b0_y, input_dict, param_dict, dm, ana_cfg)
    obs = _compute_observables(input_dict, param_dict)
    b0_x = float(param_dict['b0'][0])
    log.log(_make_log_row(sol_idx, b0_y, b0_x, nr, path, ds,
                          observables=obs))

    # ----------------------------------------------------------------
    # Arc-length continuation loop
    # ----------------------------------------------------------------
    prev_sol_norm = _solution_norm(input_dict)

    while b0_y < b0_target:
        print(f"\n[sweep] sol_{sol_idx:03d}  step {step_count:04d}:"
              f"  b0_y={b0_y:.5f},  ds={ds:.5e}")

        result: ArcLengthStepResult = arc_length_step(
            input_dict   = input_dict,
            b0_y         = b0_y,
            tangent_prev = tangent,
            param_dict   = param_dict,
            ds           = ds,
            gmres_m      = gmres_m,
            gmres_s_min  = gmres_smin,
            max_newton_iter = max_iter,
            newton_tol   = newton_tol,
            max_time_hours = max_hours,
        )

        ds = adapt_stepsize(
            ds           = ds,
            newton_iters = result.n_iters,
            converged    = result.converged,
            ds_min       = ds_min,
            ds_max       = ds_max,
            ds_grow      = ds_grow,
            ds_shrink    = ds_shrink,
        )

        if not result.converged:
            if ds < ds_min:
                print(f"[sweep] ds={ds:.2e} < ds_min — aborting sol_{sol_idx:03d}")
                break
            print(f"[sweep] Step failed, retrying with smaller ds={ds:.2e}")
            continue

        # Accept step
        input_dict = result.input_dict
        b0_y       = result.b0_y
        tangent    = result.tangent
        step_count += 1

        # Update param_dict with new b0_y
        param_dict = build_param_dict(cfg, b0_y)

        # Bifurcation detection
        cur_sol_norm = _solution_norm(input_dict)
        bif_flag = detect_bifurcation(
            prev_sol_norm, cur_sol_norm,
            threshold=float(ana_cfg['bifurcation_jump_threshold']),
        )
        if bif_flag:
            print(f"[sweep] *** Bifurcation flag at b0_y={b0_y:.4f} ***")

        # Save
        step_wall = time.time() - start_time
        path = dm.save_solution(sol_idx, b0_y, input_dict, param_dict)
        obs  = _compute_observables(input_dict, param_dict)
        b0_x = float(param_dict['b0'][0])
        log.log(_make_log_row(sol_idx, b0_y, b0_x, result, path, ds,
                              bif_flag=bif_flag,
                              wall_time_sec=step_wall,
                              observables=obs))

        # Checkpoint
        ckpt.save(sol_idx, b0_y, ds, tangent, path, step_count)

        # Analysis
        _maybe_run_analysis(sol_idx, b0_y, input_dict, param_dict, dm, ana_cfg)

        prev_sol_norm = cur_sol_norm

    print(f"[sweep] sol_{sol_idx:03d} sweep complete.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solution_norm(input_dict: Dict) -> float:
    import jax.flatten_util as fu
    flat, _ = fu.ravel_pytree(input_dict)
    return float(jnp.linalg.norm(flat))


def _compute_observables(input_dict: Dict, param_dict: Dict) -> Dict[str, float]:
    """
    Compute physical observables from a converged solution.

    Returns a dict with keys: energy, enstrophy, helicity.

    Definitions (spectral, Re-independent):
    - energy    = ½ ∑|f̂|²   (Parseval: ½‖fields‖² in real space)
    - enstrophy = ½ ‖ω‖²     (vorticity L² norm; fields[0] is vorticity)
    - helicity  = ∫ u·B dA   (cross-helicity, proxy via dot product of
                               uncurled fields in real space)
    """
    try:
        f    = input_dict['fields']                     # [2, n, n] real
        f_hat = jnp.fft.rfft2(f)                        # Fourier coefficients
        n     = f.shape[-1]

        energy    = float(0.5 * jnp.mean(jnp.square(f)))
        enstrophy = float(0.5 * jnp.mean(jnp.square(f[0])))

        # Cross-helicity: u · B in real space.
        # u = irfft2( i * ky * w_hat / k² ),  B = irfft2( i * ky * j_hat / k² )
        # As a quick proxy we use the real-space dot product of both uncurled fields.
        to_u  = param_dict['to_u']
        to_v  = param_dict['to_v']
        ux    = jnp.fft.irfft2(1j * to_u * f_hat[0])
        uy    = jnp.fft.irfft2(1j * to_v * f_hat[0])
        bx    = jnp.fft.irfft2(1j * to_u * f_hat[1])
        by_   = jnp.fft.irfft2(1j * to_v * f_hat[1])
        helicity = float(jnp.mean(ux * bx + uy * by_))
    except Exception:
        energy = enstrophy = helicity = float('nan')

    return {"energy": energy, "enstrophy": enstrophy, "helicity": helicity}


def _make_log_row(sol_idx, b0_y, b0_x, result, path, ds,
                  bif_flag=False, wall_time_sec=0.0,
                  observables: Optional[Dict] = None) -> dict:
    """Build one CSV row using the exact column names from the spec."""
    obs = observables or {}
    return {
        "sol_index":     sol_idx,
        "b0_x":          f"{b0_x:.6f}",
        "b0_y":          f"{b0_y:.6f}",
        "residual_norm": f"{result.residual_norm:.4e}",
        "n_newton_iters": result.n_iters,
        "wall_time_sec": f"{wall_time_sec:.2f}",
        "energy":        f"{obs.get('energy', ''):.6e}" if obs.get('energy') is not None else "",
        "enstrophy":     f"{obs.get('enstrophy', ''):.6e}" if obs.get('enstrophy') is not None else "",
        "helicity":      f"{obs.get('helicity', ''):.6e}" if obs.get('helicity') is not None else "",
        "bifurcation_flag": int(bif_flag),
        "ds_used":       f"{ds:.4e}",
        "output_path":   path,
    }


def _maybe_run_analysis(sol_idx, b0_y, input_dict, param_dict, dm, ana_cfg):
    """Optionally run Floquet analysis and animation."""
    if ana_cfg.get('run_floquet', False):
        try:
            from maiChaos.analysis.floquet import run_floquet
            floquet_data = run_floquet(input_dict, param_dict)
            dm.save_floquet(sol_idx, b0_y, floquet_data)
        except Exception as e:
            print(f"[sweep] Floquet analysis failed: {e}")

    if ana_cfg.get('run_animation', False):
        try:
            from maiChaos.analysis.animate import make_animation
            anim_path = dm.animation_path(sol_idx, b0_y)
            os.makedirs(os.path.dirname(anim_path), exist_ok=True)
            make_animation(input_dict, param_dict, anim_path,
                           fps=int(ana_cfg.get('animation_fps', 10)))
        except Exception as e:
            print(f"[sweep] Animation failed: {e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_sweep(config_path: str, sol_indices: Optional[List[int]] = None):
    """
    Run the full sweep over all converted seed solutions.

    Parameters
    ----------
    config_path : str
        Path to a YAML config file (see ``config/default_config.yaml``).
    sol_indices : list of int, optional
        If provided, only sweep these solution indices (0-based). Otherwise,
        sweep all ``.npz`` files found in ``paths.converted_solutions``.
    """
    cfg = load_config(config_path)
    jax.config.update("jax_enable_x64", True)

    paths_cfg = cfg['paths']
    out_dir   = paths_cfg['output_dir']
    conv_dir  = paths_cfg['converted_solutions']

    dm   = DataManager(out_dir)
    log  = ResultsLog(os.path.join(out_dir, "sweep_log.csv"))
    ckpt = Checkpoint(out_dir)

    # Discover seed solutions
    pattern  = os.path.join(conv_dir, "*.npz")
    npz_files = sorted(glob.glob(pattern))
    if not npz_files:
        print(f"[sweep] No .npz files found in {conv_dir}")
        return

    if sol_indices is not None:
        npz_files = [npz_files[i] for i in sol_indices if i < len(npz_files)]

    print(f"[sweep] Found {len(npz_files)} seed solutions in {conv_dir}")
    print(f"[sweep] Output directory: {out_dir}\n")

    for sol_idx, npz_path in enumerate(npz_files):
        print(f"\n{'='*60}")
        print(f"[sweep] Solution {sol_idx:03d}: {os.path.basename(npz_path)}")
        print(f"{'='*60}")

        try:
            input_dict, param_dict = dictionaryIO.load_dicts(npz_path)
        except Exception as e:
            print(f"[sweep] Failed to load {npz_path}: {e}")
            continue

        # Override b0 in param_dict with config starting value
        b0_start = cfg['physics']['b0_start']
        param_dict = build_param_dict(cfg, float(b0_start[1]))

        _sweep_one_solution(
            sol_idx, input_dict, param_dict, cfg,
            dm, log, ckpt,
        )

    print("\n[sweep] All solutions processed.")
