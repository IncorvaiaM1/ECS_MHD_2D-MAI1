"""
CSV append logger for sweep results.

Each call to ``log`` appends one row to a CSV so the sweep history is readable
without loading any .npz files.

Columns (exact match to spec)
------------------------------
sol_index        : seed solution index
b0_x             : mean magnetic field Bₓ (fixed at 0.0 in current setup)
b0_y             : mean magnetic field Bᵧ (the continuation parameter)
residual_norm    : final Newton residual ‖G‖
n_newton_iters   : number of Newton iterations
wall_time_sec    : wall-clock time for this step (seconds)
energy           : total field energy ½ ‖fields‖²
enstrophy        : vorticity enstrophy ½ ‖∇×u‖²  (proxy: ‖w‖²/2)
helicity         : cross-helicity ∫ u·B dA  (proxy from fields)
bifurcation_flag : 1 if a bifurcation event was detected, 0 otherwise
ds_used          : arc-length step actually taken
output_path      : relative path to the saved solution.npz
"""

import csv
import os
from typing import Any, Dict, List, Optional


class ResultsLog:
    """
    Append-mode CSV logger.

    Parameters
    ----------
    path : str
        Path to the CSV file (created on first write if absent).
    columns : list of str
        Column names. Used to write the header on first creation and to
        validate row dicts on write.
    """

    COLUMNS: List[str] = [
        "sol_index",
        "b0_x",
        "b0_y",
        "residual_norm",
        "n_newton_iters",
        "wall_time_sec",
        "energy",
        "enstrophy",
        "helicity",
        "bifurcation_flag",
        "ds_used",
        "output_path",
    ]

    def __init__(self, path: str, columns: Optional[List[str]] = None):
        self.path = path
        self.columns = columns if columns is not None else self.COLUMNS
        self._ensure_header()

    def _ensure_header(self):
        """Write header row if the file does not exist yet."""
        if not os.path.isfile(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()

    def log(self, row: Dict[str, Any]):
        """
        Append one row to the CSV.

        Keys not in ``self.columns`` are silently dropped; missing keys
        become empty strings.
        """
        clean = {k: row.get(k, "") for k in self.columns}
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(clean)

    def read_all(self) -> List[Dict[str, str]]:
        """Return the full CSV as a list of dicts."""
        if not os.path.isfile(self.path):
            return []
        with open(self.path, "r", newline="") as f:
            return list(csv.DictReader(f))
