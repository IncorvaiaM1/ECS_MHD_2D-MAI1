"""
JSON + npz checkpointing so a sweep can be resumed after interruption.

Checkpoint format (checkpoint.json) — matches spec exactly::

    {
        "current_sol_index": 3,
        "current_b0":        0.14,
        "last_converged_file": "data/sweep_results/.../sol_003/B0_0.1400/solution.npz",
        "ds_current":        0.003,
        "timestamp":         "2026-04-20T12:34:56",
        "tangent_path":      "data/sweep_results/.../checkpoint_tangent.npy",
        "step_count":        42
    }

The tangent vector is saved as a separate ``.npy`` because JSON cannot store
float arrays losslessly.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


class Checkpoint:
    """
    Save and load sweep checkpoints.

    Parameters
    ----------
    checkpoint_dir : str
        Directory where ``checkpoint.json`` and ``checkpoint_tangent.npy``
        are written.
    """

    FNAME_JSON    = "checkpoint.json"
    FNAME_TANGENT = "checkpoint_tangent.npy"

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._json_path    = os.path.join(checkpoint_dir, self.FNAME_JSON)
        self._tangent_path = os.path.join(checkpoint_dir, self.FNAME_TANGENT)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, sol_idx: int, b0_y: float, ds: float,
             tangent: np.ndarray, solution_path: str,
             step_count: int = 0, extra: Optional[Dict[str, Any]] = None):
        """
        Persist the current sweep state.

        Parameters
        ----------
        sol_idx       : index of the solution being tracked.
        b0_y          : current b0_y value.
        ds            : current arc-length step size.
        tangent       : current unit tangent vector.
        solution_path : path to the last saved solution.npz.
        step_count    : total number of continuation steps taken.
        extra         : any additional JSON-serialisable metadata.
        """
        np.save(self._tangent_path, tangent)

        state: Dict[str, Any] = {
            # --- spec-required keys ---
            "current_sol_index":   sol_idx,
            "current_b0":          float(b0_y),
            "last_converged_file": solution_path,
            "ds_current":          float(ds),
            "timestamp":           datetime.now().isoformat(timespec="seconds"),
            # --- extra bookkeeping ---
            "tangent_path":        self._tangent_path,
            "step_count":          step_count,
        }
        if extra:
            state.update(extra)

        with open(self._json_path, "w") as f:
            json.dump(state, f, indent=2)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        return os.path.isfile(self._json_path)

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Return the checkpoint state dict, or None if no checkpoint exists.

        The returned dict contains both the spec-canonical keys
        (``current_sol_index``, ``current_b0``, ``last_converged_file``,
        ``ds_current``, ``timestamp``) and convenience aliases
        (``sol_idx``, ``b0_y``, ``solution_path``, ``ds``).

        Also adds key ``'tangent'`` with the numpy array loaded from disk.
        """
        if not self.exists():
            return None

        with open(self._json_path, "r") as f:
            state = json.load(f)

        # Convenience aliases for internal code that uses shorter names
        state.setdefault("sol_idx",       state.get("current_sol_index"))
        state.setdefault("b0_y",          state.get("current_b0"))
        state.setdefault("solution_path", state.get("last_converged_file"))
        state.setdefault("ds",            state.get("ds_current"))

        tangent_path = state.get("tangent_path", self._tangent_path)
        if os.path.isfile(tangent_path):
            state["tangent"] = np.load(tangent_path)
        else:
            state["tangent"] = None

        return state

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Delete the checkpoint files (start fresh)."""
        for p in [self._json_path, self._tangent_path]:
            if os.path.isfile(p):
                os.remove(p)
