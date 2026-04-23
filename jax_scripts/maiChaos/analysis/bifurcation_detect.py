"""
Bifurcation detection utilities.

Implements three detection strategies (all three from the spec):

1. **Jump detection**: flags if the solution norm changes by more than a
   threshold fraction between two consecutive continuation steps.  Heuristic
   proxy for a fold or branch-crossing.

2. **Fold-point detection**: a fold (saddle-node) occurs when the arc-length
   tangent's b0_y component changes sign — the parameter momentarily reverses.

3. **Branch-point detection via Floquet**: a branch point occurs when the
   leading Floquet multiplier crosses the unit circle (|μ| crosses 1.0).
   This requires Floquet data from successive steps.
"""

from typing import List, Optional
import numpy as np


def detect_bifurcation(
    prev_norm: float,
    curr_norm: float,
    threshold: float = 0.1,
) -> bool:
    """
    Return True if the solution norm jumped by more than ``threshold``.

    This is a simple heuristic: large jumps in ‖z‖ between successive
    continuation steps indicate a fold, branch-crossing, or numerical blowup.

    Parameters
    ----------
    prev_norm  : solution norm at the previous step.
    curr_norm  : solution norm at the current step.
    threshold  : relative change fraction that triggers the flag.
                 ``|curr - prev| / max(prev, 1e-12) > threshold``

    Returns
    -------
    bool
    """
    denom = max(abs(prev_norm), 1e-12)
    return abs(curr_norm - prev_norm) / denom > threshold


def detect_fold(
    tangent_history: List[np.ndarray],
) -> Optional[int]:
    """
    Detect a fold point (saddle-node) in the continuation history.

    A fold occurs when the b0_y component of the tangent changes sign.
    We scan pairs of consecutive tangents and return the index of the first
    sign flip, or None if no fold is found.

    Parameters
    ----------
    tangent_history : list of 1-D tangent arrays.  The last element of each
                      array is the b0_y component of the tangent.

    Returns
    -------
    int or None : index i such that tangent_history[i][-1] and
                  tangent_history[i+1][-1] have opposite signs, or None.
    """
    for i in range(len(tangent_history) - 1):
        dy_i   = float(tangent_history[i][-1])
        dy_ip1 = float(tangent_history[i + 1][-1])
        if dy_i * dy_ip1 < 0:
            return i
    return None


def detect_branch_point(
    floquet_prev: Optional[np.ndarray],
    floquet_curr: Optional[np.ndarray],
    unit_circle_tol: float = 0.05,
) -> bool:
    """
    Detect a branch point by checking if the leading Floquet multiplier
    crosses the unit circle between two successive continuation steps.

    A branch point (period-doubling, pitchfork, Hopf) occurs when
    max |μᵢ| crosses 1.0.  We flag it if the leading multiplier magnitude
    straddles 1.0 between the previous and current step.

    Parameters
    ----------
    floquet_prev    : complex array of Floquet multipliers from the previous
                      step (output of ``run_floquet``), or None if unavailable.
    floquet_curr    : complex array of Floquet multipliers from the current
                      step, or None.
    unit_circle_tol : tolerance band around |μ| = 1 for flagging.
                      A multiplier is ``on`` the unit circle if
                      ``|1 - |μ|| < unit_circle_tol``.

    Returns
    -------
    bool — True if a unit-circle crossing is detected.
    """
    if floquet_prev is None or floquet_curr is None:
        return False

    lead_prev = float(np.max(np.abs(floquet_prev)))
    lead_curr = float(np.max(np.abs(floquet_curr)))

    # Crossing: one side < 1, other side > 1 (with tolerance)
    crossed = (lead_prev < 1.0 - unit_circle_tol and
               lead_curr > 1.0 + unit_circle_tol) or \
              (lead_prev > 1.0 + unit_circle_tol and
               lead_curr < 1.0 - unit_circle_tol)

    # Or currently sitting right on the circle
    on_circle = abs(lead_curr - 1.0) < unit_circle_tol

    return crossed or on_circle


class BifurcationTracker:
    """
    Stateful tracker that accumulates tangent and Floquet history, flags events.

    Detects three types of events (per spec):
    - ``'jump'``   : solution norm jump > threshold
    - ``'fold'``   : tangent b0_y component sign flip
    - ``'branch'`` : leading Floquet multiplier crosses the unit circle

    Attributes
    ----------
    events : list of dict
        Each dict has keys: ``step``, ``b0_y``, ``type``.
    """

    def __init__(self, jump_threshold: float = 0.1,
                 floquet_tol: float = 0.05):
        self.jump_threshold  = jump_threshold
        self.floquet_tol     = floquet_tol
        self.tangent_history:  List[np.ndarray]          = []
        self.norm_history:     List[float]               = []
        self.b0y_history:      List[float]               = []
        self.floquet_history:  List[Optional[np.ndarray]] = []
        self.events:           list                      = []

    def update(self, step: int, b0_y: float,
               tangent: np.ndarray, sol_norm: float,
               floquet_multipliers: Optional[np.ndarray] = None):
        """
        Record a new step and check for all three bifurcation event types.

        Parameters
        ----------
        step                : continuation step index.
        b0_y                : current parameter value.
        tangent             : current unit tangent vector.
        sol_norm            : ‖input_dict‖ at this step.
        floquet_multipliers : complex array of Floquet multipliers (optional).
                              Pass ``None`` if Floquet was not computed.
        """
        # 1. Jump detection
        if self.norm_history:
            if detect_bifurcation(self.norm_history[-1], sol_norm,
                                  self.jump_threshold):
                self.events.append({
                    "step":      step,
                    "b0_y":      b0_y,
                    "type":      "jump",
                    "norm_prev": self.norm_history[-1],
                    "norm_curr": sol_norm,
                })
                print(f"[bifurcation] Jump at step {step}, b0_y={b0_y:.4f}")

        self.tangent_history.append(np.array(tangent))
        self.norm_history.append(sol_norm)
        self.b0y_history.append(b0_y)
        self.floquet_history.append(
            np.array(floquet_multipliers) if floquet_multipliers is not None
            else None
        )

        # 2. Fold detection (requires at least 2 tangents)
        if len(self.tangent_history) >= 2:
            fold_idx = detect_fold(self.tangent_history[-2:])
            if fold_idx is not None:
                self.events.append({
                    "step":  step,
                    "b0_y":  b0_y,
                    "type":  "fold",
                })
                print(f"[bifurcation] Fold at step {step}, b0_y={b0_y:.4f}")

        # 3. Branch point via Floquet (requires at least 2 Floquet results)
        if len(self.floquet_history) >= 2:
            f_prev = self.floquet_history[-2]
            f_curr = self.floquet_history[-1]
            if detect_branch_point(f_prev, f_curr, self.floquet_tol):
                self.events.append({
                    "step":  step,
                    "b0_y":  b0_y,
                    "type":  "branch",
                })
                print(f"[bifurcation] Branch point (Floquet) at step {step},"
                      f" b0_y={b0_y:.4f}")

    def summary(self) -> str:
        if not self.events:
            return "No bifurcation events detected."
        lines = [f"  step={e['step']:4d}  b0_y={e['b0_y']:.4f}  type={e['type']}"
                 for e in self.events]
        return "Bifurcation events:\n" + "\n".join(lines)
