"""
Entry point: run Floquet analysis on a single solution file.

Usage
-----
    python jax_scripts/maiChaos/scripts/run_floquet.py \\
        --solution data/converted/Re40/solution_000.npz

    # With options
    python jax_scripts/maiChaos/scripts/run_floquet.py \\
        --solution solution.npz --block-size 64 --maxit 16 \\
        --output floquet_result.npz
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from maiChaos.analysis.floquet import main as floquet_main


def main():
    parser = argparse.ArgumentParser(
        description="Floquet analysis for an MHD RPO solution."
    )
    parser.add_argument("--solution", "-s", required=True,
                        help="Path to .npz solution file.")
    parser.add_argument("--output", "-o", default=None,
                        help="Output .npz path (default: <solution>_floquet.npz).")
    parser.add_argument("--block-size", type=int, default=32,
                        help="Number of Floquet vectors to iterate.")
    parser.add_argument("--maxit", type=int, default=8,
                        help="Number of power iterations.")
    args = parser.parse_args()

    floquet_main(
        solution_file=args.solution,
        output_path=args.output,
        block_size=args.block_size,
        maxit=args.maxit,
    )


if __name__ == "__main__":
    main()
