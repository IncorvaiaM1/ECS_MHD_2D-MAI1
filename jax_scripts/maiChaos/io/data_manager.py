"""
Centralized save/load for all maiChaos outputs.

Wraps lib.dictionaryIO to provide path-management and a consistent API for
continuation results, Floquet data, and checkpoint state.
"""

import os
import sys

# Allow running from jax_scripts/ or from maiChaos/
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB  = os.path.join(_HERE, '..', '..')
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

import lib.dictionaryIO as dictionaryIO


class DataManager:
    """
    Manages all file I/O for a single sweep output directory.

    Directory layout created on demand::

        <output_dir>/
            sweep_log.csv
            checkpoint.json
            sol_<NNN>/
                B0_<val>/
                    solution.npz
                floquet/
                    B0_<val>_floquet.npz
                animations/
                    B0_<val>.gif

    Parameters
    ----------
    output_dir : str
        Root directory for this sweep (e.g. ``data/sweep_results/Re40/B0_sweep``).
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def sol_dir(self, sol_idx: int) -> str:
        return os.path.join(self.output_dir, f"sol_{sol_idx:03d}")

    def solution_path(self, sol_idx: int, b0_y: float) -> str:
        b0_str = f"B0_{b0_y:.4f}"
        return os.path.join(self.sol_dir(sol_idx), b0_str, "solution.npz")

    def floquet_path(self, sol_idx: int, b0_y: float) -> str:
        b0_str = f"B0_{b0_y:.4f}"
        return os.path.join(self.sol_dir(sol_idx), "floquet", f"{b0_str}_floquet.npz")

    def animation_path(self, sol_idx: int, b0_y: float) -> str:
        b0_str = f"B0_{b0_y:.4f}"
        return os.path.join(self.sol_dir(sol_idx), "animations", f"{b0_str}.gif")

    # ------------------------------------------------------------------
    # Solution save / load
    # ------------------------------------------------------------------

    def save_solution(self, sol_idx: int, b0_y: float,
                      input_dict, param_dict) -> str:
        """Save input_dict + param_dict to the canonical solution path."""
        path = self.solution_path(sol_idx, b0_y)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dictionaryIO.save_dicts(path, input_dict, param_dict)
        return path

    def load_solution(self, sol_idx: int, b0_y: float):
        """Load and return (input_dict, param_dict) from the canonical path."""
        path = self.solution_path(sol_idx, b0_y)
        return dictionaryIO.load_dicts(path)

    def solution_exists(self, sol_idx: int, b0_y: float) -> bool:
        return os.path.isfile(self.solution_path(sol_idx, b0_y))

    # ------------------------------------------------------------------
    # Floquet save / load
    # ------------------------------------------------------------------

    def save_floquet(self, sol_idx: int, b0_y: float, data: dict) -> str:
        """Save a dict of numpy arrays as a Floquet result."""
        import numpy as np
        path = self.floquet_path(sol_idx, b0_y)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **{k: np.array(v) for k, v in data.items()})
        return path

    def load_floquet(self, sol_idx: int, b0_y: float) -> dict:
        import numpy as np
        path = self.floquet_path(sol_idx, b0_y)
        loaded = np.load(path)
        return dict(loaded)

    # ------------------------------------------------------------------
    # Arbitrary npz save (for Floquet or other arrays)
    # ------------------------------------------------------------------

    def save_npz(self, path: str, data: dict) -> str:
        """Save any dict of arrays to an npz file at an explicit path."""
        import numpy as np
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **{k: np.array(v) for k, v in data.items()})
        return path
