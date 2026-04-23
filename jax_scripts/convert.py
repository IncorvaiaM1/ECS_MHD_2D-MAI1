"""
Batch-convert Andrew's Re40 solutions from raw .npy files to .npz seeds.

Each raw solution pair (soln_array_Re40_N.npy + soln_meta_Re40_N.npy) is
converted to the (input_dict, param_dict) format expected by maiChaos and
saved as data/converted/Re40/soln_NNN.npz.

Usage
-----
    # Convert all (skip already-converted files):
    python jax_scripts/convert.py

    # Re-convert everything from scratch:
    python jax_scripts/convert.py --fresh

    # Convert a single index for testing:
    python jax_scripts/convert.py --index 50

    # Use paths from config yaml:
    python jax_scripts/convert.py --config jax_scripts/maiChaos/config/default_config.yaml
"""

import os
import sys
import glob
import argparse

import numpy as np
import jax.numpy as jnp
import jax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import lib.mhd_jax as mhd_jax
import lib.dictionaryIO as dictionaryIO


def convert_one(index: int, raw_dir: str):
    """
    Convert a single Andrew solution to (input_dict, param_dict).

    Parameters
    ----------
    index   : solution index (matches the filename suffix)
    raw_dir : directory containing soln_array_Re40_N.npy / soln_meta_Re40_N.npy
    """
    n = 128

    f = jnp.zeros([2, n, n])
    f0 = np.load(os.path.join(raw_dir, f"soln_array_Re40_{index}.npy"))
    f0 = jnp.fft.irfft2(f0)
    f = f.at[0, :, :].set(f0)

    nu  = 1 / 40
    eta = 1 / 40
    b0  = [0.0, 0.0]

    param_dict = mhd_jax.construct_domain(n, jnp.float64)
    y = param_dict['y']

    forcing = -4 * jnp.cos(4 * y)

    steps     = 1024
    ministeps = 32
    assert steps % ministeps == 0

    param_dict.update({
        'nu':      nu,
        'eta':     eta,
        'b0':      b0,
        'forcing': forcing,
        'ministeps':       ministeps,
        'num_checkpoints': steps // ministeps,
        'steps':           steps,
    })

    metadata = np.load(os.path.join(raw_dir, f"soln_meta_Re40_{index}.npy"))
    input_dict = {'fields': f, 'T': float(metadata[0]), 'sx': float(metadata[1])}
    param_dict.update({'rot': False, 'shift_reflect_ny': float(metadata[2])})

    return input_dict, param_dict


def _discover_indices(raw_dir: str):
    """Return sorted list of available solution indices in raw_dir."""
    pattern = os.path.join(raw_dir, "soln_meta_Re40_*.npy")
    meta_files = glob.glob(pattern)
    indices = sorted(
        int(os.path.basename(f)
              .replace("soln_meta_Re40_", "")
              .replace(".npy", ""))
        for f in meta_files
    )
    return indices


def convert_all(raw_dir: str, out_dir: str, skip_existing: bool = True):
    """Convert every solution in raw_dir and write .npz seeds to out_dir."""
    jax.config.update("jax_enable_x64", True)
    os.makedirs(out_dir, exist_ok=True)

    indices = _discover_indices(raw_dir)
    if not indices:
        print(f"[convert] No solution files found in {raw_dir}")
        return

    total = len(indices)
    print(f"[convert] Found {total} solutions in {raw_dir}")
    print(f"[convert] Output directory: {out_dir}\n")

    converted = skipped = failed = 0

    for idx in indices:
        out_path = os.path.join(out_dir, f"solution_{idx:03d}.npz")

        if skip_existing and os.path.isfile(out_path):
            skipped += 1
            continue

        try:
            input_dict, param_dict = convert_one(idx, raw_dir)
            dictionaryIO.save_dicts(out_path, input_dict, param_dict)
            converted += 1
            done = converted + skipped + failed
            print(f"[convert] [{done:3d}/{total}] solution_{idx:03d}.npz  "
                  f"T={float(input_dict['T']):.4f}  sx={float(input_dict['sx']):.4f}")
        except Exception as e:
            failed += 1
            print(f"[convert] FAILED index {idx}: {e}")

    print(f"\n[convert] Done — {converted} converted, {skipped} skipped, {failed} failed.")


def _resolve_paths_from_config(config_path: str):
    """Return (raw_dir, out_dir) resolved from a YAML config, or None on error."""
    try:
        import yaml
    except ImportError:
        print("[convert] Warning: PyYAML not installed — ignoring --config")
        return None, None
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        paths = cfg.get('paths', {})
        raw = paths.get('raw_solutions')
        out = paths.get('converted_solutions')
        raw_dir = os.path.join(_HERE, raw) if raw else None
        out_dir = os.path.join(_HERE, out) if out else None
        return raw_dir, out_dir
    except Exception as e:
        print(f"[convert] Warning: could not load config {config_path}: {e}")
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-convert Andrew's Re40 .npy solutions to .npz seeds."
    )
    parser.add_argument(
        "--raw-dir",
        default=os.path.join(_HERE, "Re40_all_for_matt"),
        help="Directory containing soln_array/meta .npy files.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_HERE, "data", "converted", "Re40"),
        help="Output directory for .npz files.",
    )
    parser.add_argument(
        "--config", default=None,
        help="YAML config to read paths.raw_solutions / paths.converted_solutions from.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Re-convert even if output file already exists.",
    )
    parser.add_argument(
        "--index", type=int, default=None,
        help="Convert only a single index (for testing).",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir

    if args.config:
        cfg_raw, cfg_out = _resolve_paths_from_config(args.config)
        if cfg_raw:
            raw_dir = cfg_raw
        if cfg_out:
            out_dir = cfg_out

    jax.config.update("jax_enable_x64", True)

    if args.index is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"solution_{args.index:03d}.npz")
        input_dict, param_dict = convert_one(args.index, raw_dir)
        dictionaryIO.save_dicts(out_path, input_dict, param_dict)
        print(f"[convert] Saved {out_path}")
    else:
        convert_all(raw_dir, out_dir, skip_existing=not args.fresh)
