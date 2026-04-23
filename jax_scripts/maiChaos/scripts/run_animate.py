"""
Entry point: animate a single solution file.

Usage
-----
    python jax_scripts/maiChaos/scripts/run_animate.py \\
        --solution data/converted/Re40/solution_000.npz

    # With options
    python jax_scripts/maiChaos/scripts/run_animate.py \\
        --solution solution.npz --fps 10 --save-every 32 --output out.gif
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from maiChaos.analysis.animate import main as animate_main


def main():
    parser = argparse.ArgumentParser(
        description="Animate an MHD RPO solution."
    )
    parser.add_argument("--solution", "-s", required=True,
                        help="Path to .npz solution file.")
    parser.add_argument("--output", "-o", default=None,
                        help="Output .gif or .mp4 path.")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second.")
    parser.add_argument("--save-every", type=int, default=32,
                        help="Timesteps between frames.")
    args = parser.parse_args()

    animate_main(
        solution_file=args.solution,
        output_path=args.output,
        fps=args.fps,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
