"""
Entry point: run the automated B₀_y sweep.

Usage
-----
    python jax_scripts/maiChaos/scripts/run_sweep.py
        # uses default_config.yaml, auto-resumes from checkpoint if present

    python jax_scripts/maiChaos/scripts/run_sweep.py --config my_config.yaml
        # custom config

    python jax_scripts/maiChaos/scripts/run_sweep.py --sol 5
        # run only solution index 5

    python jax_scripts/maiChaos/scripts/run_sweep.py --resume
        # force resume from checkpoint (same as default, explicit flag)

    python jax_scripts/maiChaos/scripts/run_sweep.py --fresh
        # ignore any existing checkpoint, start from scratch

    python jax_scripts/maiChaos/scripts/run_sweep.py --convert
        # batch-convert raw .npy seeds before sweeping (skips already-converted)

    python jax_scripts/maiChaos/scripts/run_sweep.py --convert --fresh-convert
        # re-convert all seeds from scratch, then sweep
"""

import argparse
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Ensure jax_scripts/ is on the path so lib/ is importable
_JAX_SCRIPTS = os.path.join(_HERE, '..', '..')
sys.path.insert(0, _JAX_SCRIPTS)

from maiChaos.sweep.automated_sweep import run_sweep, load_config
from maiChaos.sweep.checkpoint import Checkpoint


def _run_convert(cfg, fresh_convert: bool = False):
    """Batch-convert raw .npy seeds using convert.py logic."""
    from convert import convert_all

    paths = cfg.get('paths', {})
    raw_dir = os.path.join(_JAX_SCRIPTS, paths.get('raw_solutions', 'Re40_all_for_matt'))
    out_dir = os.path.join(_JAX_SCRIPTS, paths.get('converted_solutions', 'data/converted/Re40'))

    existing = glob.glob(os.path.join(out_dir, "*.npz"))
    if existing and not fresh_convert:
        print(f"[run_sweep] {len(existing)} converted file(s) already in {out_dir} "
              f"— skipping conversion (use --fresh-convert to redo).")
        return

    print(f"[run_sweep] Running batch conversion: {raw_dir} → {out_dir}")
    convert_all(raw_dir, out_dir, skip_existing=not fresh_convert)


def main():
    parser = argparse.ArgumentParser(
        description="Run MHD arc-length continuation sweep."
    )
    parser.add_argument(
        "--config", "-c",
        default=os.path.join(_HERE, "..", "config", "default_config.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--sol", "--sols", dest="sols", nargs="*", type=int, default=None,
        help="Indices of solutions to sweep (default: all).",
    )
    parser.add_argument(
        "--convert", action="store_true", default=False,
        help="Batch-convert raw .npy seeds before sweeping.",
    )
    parser.add_argument(
        "--fresh-convert", action="store_true", default=False,
        help="Re-convert all seeds even if already converted (implies --convert).",
    )

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume", action="store_true", default=False,
        help="Force resume from checkpoint (default behaviour when checkpoint exists).",
    )
    resume_group.add_argument(
        "--fresh", action="store_true", default=False,
        help="Ignore existing checkpoint and start from scratch.",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    # Run conversion step if requested (or if fresh-convert forces it)
    if args.convert or args.fresh_convert:
        _run_convert(cfg, fresh_convert=args.fresh_convert)

    # Handle --fresh: delete checkpoint before starting
    if args.fresh:
        ckpt_dir = cfg['paths']['output_dir']
        ckpt = Checkpoint(ckpt_dir)
        if ckpt.exists():
            ckpt.reset()
            print(f"[run_sweep] Checkpoint deleted — starting fresh.")
        else:
            print(f"[run_sweep] No checkpoint found — starting fresh.")

    # --resume is the default, but printing a confirmation is useful
    if args.resume:
        ckpt_dir = cfg['paths']['output_dir']
        ckpt = Checkpoint(ckpt_dir)
        state = ckpt.load()
        if state:
            print(f"[run_sweep] Resuming from checkpoint: "
                  f"sol={state.get('current_sol_index')}, "
                  f"b0_y={state.get('current_b0'):.4f}, "
                  f"timestamp={state.get('timestamp')}")
        else:
            print(f"[run_sweep] --resume specified but no checkpoint found — starting fresh.")

    run_sweep(args.config, sol_indices=args.sols)


if __name__ == "__main__":
    main()
