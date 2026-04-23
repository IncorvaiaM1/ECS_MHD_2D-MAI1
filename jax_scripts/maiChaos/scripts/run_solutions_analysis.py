"""
Batch Floquet analysis and animation for a directory of MHD solution files.

Usage
-----
    # Process all .npz in a single leaf directory:
    python jax_scripts/maiChaos/scripts/run_solutions_analysis.py \\
        --solutions-dir solutions/Re40/

    # Recurse into Re*/ subdirectories of a top-level directory:
    python jax_scripts/maiChaos/scripts/run_solutions_analysis.py \\
        --solutions-dir solutions/

    # Only animate, skip Floquet:
    python jax_scripts/maiChaos/scripts/run_solutions_analysis.py \\
        --solutions-dir solutions/ --no-floquet

    # Force overwrite of existing outputs:
    python jax_scripts/maiChaos/scripts/run_solutions_analysis.py \\
        --solutions-dir solutions/ --overwrite
"""

import argparse
import glob
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

import numpy as np

from maiChaos.analysis.floquet import run_floquet
from maiChaos.analysis.animate import make_animation
import lib.dictionaryIO as dictionaryIO


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

def _find_npz_in_dir(directory: str) -> list:
    """
    Return sorted list of solution .npz files directly in directory.

    Excludes files whose stem ends with '_floquet' (safety guard for stray
    analysis outputs). Files inside subdirectories are not matched because
    glob('*.npz') does not descend.
    """
    results = []
    for path in sorted(glob.glob(os.path.join(directory, '*.npz'))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.endswith('_floquet'):
            continue
        results.append(path)
    return results


def find_solution_dirs(solutions_dir: str) -> list:
    """
    Return a list of leaf directories that contain solution .npz files.

    If solutions_dir itself contains .npz files directly, return it as a
    single-element list.  Otherwise scan one level of subdirectories.
    """
    solutions_dir = os.path.abspath(solutions_dir)

    direct = _find_npz_in_dir(solutions_dir)
    if direct:
        return [solutions_dir]

    leaf_dirs = []
    try:
        entries = sorted(os.scandir(solutions_dir), key=lambda e: e.name)
    except PermissionError as exc:
        print(f"[batch] Cannot read directory {solutions_dir}: {exc}")
        return []

    for entry in entries:
        if entry.is_dir() and _find_npz_in_dir(entry.path):
            leaf_dirs.append(entry.path)

    return leaf_dirs


# ---------------------------------------------------------------------------
# Output path convention
# ---------------------------------------------------------------------------

def compute_output_paths(sol_file: str):
    """Return (floquet_path, animation_path) for a given solution .npz."""
    sol_dir = os.path.dirname(sol_file)
    stem    = os.path.splitext(os.path.basename(sol_file))[0]
    floquet_path   = os.path.join(sol_dir, 'floquet',    f'{stem}_floquet.npz')
    animation_path = os.path.join(sol_dir, 'animations', f'{stem}.gif')
    return floquet_path, animation_path


# ---------------------------------------------------------------------------
# Per-solution processing
# ---------------------------------------------------------------------------

def process_solution(sol_file, run_floquet_flag, run_animate_flag,
                     block_size, maxit, fps, save_every, overwrite):
    floquet_path, animation_path = compute_output_paths(sol_file)
    stem = os.path.splitext(os.path.basename(sol_file))[0]

    print(f"\n{'='*60}")
    print(f"[batch] Solution: {sol_file}")
    print(f"{'='*60}")

    try:
        input_dict, param_dict = dictionaryIO.load_dicts(sol_file)
    except Exception:
        print(f"[batch] ERROR loading {sol_file}:")
        traceback.print_exc()
        return

    # ---- Floquet analysis ----
    if run_floquet_flag:
        if not overwrite and os.path.isfile(floquet_path):
            print(f"[batch]   floquet: SKIP (exists) -> {floquet_path}")
        else:
            print(f"[batch]   floquet: running (block_size={block_size}, maxit={maxit})")
            try:
                os.makedirs(os.path.dirname(floquet_path), exist_ok=True)
                data = run_floquet(input_dict, param_dict,
                                   block_size=block_size, maxit=maxit)
                np.savez(floquet_path,
                         **{k: np.array(v) for k, v in data.items()
                            if not isinstance(v, dict)})
                print(f"[batch]   floquet: saved -> {floquet_path}")
            except Exception:
                print(f"[batch]   floquet: ERROR for {stem}:")
                traceback.print_exc()

    # ---- Animation ----
    if run_animate_flag:
        if not overwrite and os.path.isfile(animation_path):
            print(f"[batch]   animate: SKIP (exists) -> {animation_path}")
        else:
            print(f"[batch]   animate: running (fps={fps}, save_every={save_every})")
            try:
                os.makedirs(os.path.dirname(animation_path), exist_ok=True)
                make_animation(input_dict, param_dict,
                               output_path=animation_path,
                               fps=fps, save_every=save_every)
            except Exception:
                print(f"[batch]   animate: ERROR for {stem}:")
                traceback.print_exc()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Batch Floquet analysis and animation for MHD solution directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--solutions-dir", "-d", required=True,
        help=(
            "Path to solutions directory: either a leaf dir containing .npz files "
            "directly (e.g. solutions/Re40/) or a top-level dir whose subdirs "
            "contain .npz files (e.g. solutions/)."
        ),
    )
    parser.add_argument(
        "--floquet", action=argparse.BooleanOptionalAction, default=True,
        help="Run Floquet analysis on each solution.",
    )
    parser.add_argument(
        "--animate", action=argparse.BooleanOptionalAction, default=True,
        help="Run animation on each solution.",
    )
    parser.add_argument("--block-size", type=int, default=32,
                        help="Number of Floquet vectors to iterate.")
    parser.add_argument("--maxit", type=int, default=8,
                        help="Number of power iterations.")
    parser.add_argument("--fps", type=int, default=10,
                        help="Animation frames per second.")
    parser.add_argument("--save-every", type=int, default=32,
                        help="Timesteps between animation frames.")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Re-run analyses even if output files already exist.")
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()

    solutions_dir = os.path.abspath(args.solutions_dir)
    if not os.path.isdir(solutions_dir):
        parser.error(f"--solutions-dir is not a directory: {solutions_dir}")

    leaf_dirs = find_solution_dirs(solutions_dir)
    if not leaf_dirs:
        print(f"[batch] No solution .npz files found under {solutions_dir}")
        sys.exit(0)

    all_solutions = []
    for leaf_dir in leaf_dirs:
        files = _find_npz_in_dir(leaf_dir)
        all_solutions.extend(files)
        print(f"[batch] {leaf_dir}: {len(files)} solution(s)")

    print(f"\n[batch] Total: {len(all_solutions)} solution(s)")
    print(f"[batch] Floquet: {'enabled' if args.floquet else 'disabled'}")
    print(f"[batch] Animate: {'enabled' if args.animate else 'disabled'}")
    print(f"[batch] Overwrite: {'yes' if args.overwrite else 'no'}")

    for sol_file in all_solutions:
        process_solution(
            sol_file         = sol_file,
            run_floquet_flag = args.floquet,
            run_animate_flag = args.animate,
            block_size       = args.block_size,
            maxit            = args.maxit,
            fps              = args.fps,
            save_every       = args.save_every,
            overwrite        = args.overwrite,
        )

    print(f"\n[batch] Done. Processed {len(all_solutions)} solution(s).")


if __name__ == "__main__":
    main()
