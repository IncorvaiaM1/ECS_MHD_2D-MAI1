# maiChaos — MHD Arc-Length Continuation Package

`maiChaos` is the continuation and bifurcation analysis sub-package for the
ECS_MHD_2D_MAI1 project.  It tracks solution branches of the 2D MHD equations
as the mean magnetic field B₀_y varies, using pseudo arc-length continuation,
and detects bifurcations along the way.

---

## Module Descriptions

```
maiChaos/
├── config/
│   └── default_config.yaml     All tunable parameters in one file
│
├── continuation/
│   ├── arc_length.py           Pseudo arc-length stepper.
│   │                           Augmented system [G(z,λ); N(z,λ)] = 0
│   │                           solved with matrix-free Newton-GMRES.
│   ├── newton_gmres.py         Newton-GMRES solver, based on legacy/newton.py.
│   │                           JIT inner loop, wall-clock timeout, line search,
│   │                           optional hookstep, structured NewtonResult return.
│   └── predictor.py            Euler (secant) predictor, tangent computation,
│                               adaptive step-size control.
│
├── sweep/
│   ├── automated_sweep.py      Main automation loop. Loads seed solutions,
│   │                           re-converges, steps with arc-length, logs,
│   │                           checkpoints, optionally runs analysis.
│   └── checkpoint.py           JSON + .npy checkpointing for resume.
│
├── analysis/
│   ├── floquet.py              Floquet multiplier estimation via power iteration
│   │                           (vmapped JVP). Migrated from legacy/floquet.py.
│   ├── animate.py              GIF/MP4 animation of one RPO period.
│   │                           Migrated from legacy/animation.py.
│   └── bifurcation_detect.py   Three detectors: norm jump, fold (tangent sign
│                               flip), branch point (Floquet crosses unit circle).
│
├── io/
│   ├── data_manager.py         Canonical path management + save/load wrappers
│   │                           around lib/dictionaryIO.py.
│   └── results_log.py          Append-mode CSV logger. One row per converged step.
│
└── scripts/
    ├── run_sweep.py            python run_sweep.py [--config ...] [--sol N]
    │                                               [--resume | --fresh]
    ├── run_floquet.py          python run_floquet.py --solution sol.npz
    └── run_animate.py          python run_animate.py --solution sol.npz
```

---

## How to Run a Sweep

1. **Convert seed solutions** (once):
   ```bash
   cd jax_scripts
   python convert.py
   # writes data/converted/Re40/solution_NNN.npz
   ```

2. **Edit config** (optional):
   ```bash
   # jax_scripts/maiChaos/config/default_config.yaml
   # Adjust b0_target, ds_init, newton_tol, etc.
   ```

3. **Run the sweep**:
   ```bash
   cd jax_scripts
   python maiChaos/scripts/run_sweep.py
   # or with explicit config:
   python maiChaos/scripts/run_sweep.py --config maiChaos/config/default_config.yaml
   # or sweep only solutions 0 and 5:
   python maiChaos/scripts/run_sweep.py --sol 0 5
   ```

Expected output:
```
Loading checkpoint... none found. Starting fresh.
Processing solution 000: solution_000.npz
  re-converging seed at b0_y=0.0100
  Iter   0: rel_err=1.23e-03, |f|=4.56e-02, ...
  Iter   7: rel_err=2.11e-11, |f|=8.30e-13  ← converged
  Saved: data/sweep_results/Re40/B0_sweep/sol_000/B0_0.0100/solution.npz
  [arc_len] iter  0: |G| = 3.45e-02,  b0_y = 0.020000
  [arc_len] iter  4: |G| = 7.80e-12,  b0_y = 0.020011  ← converged
  Checkpoint saved.
  ...
```

---

## How to Resume an Interrupted Sweep

The sweep writes `data/sweep_results/Re40/B0_sweep/checkpoint.json` after
every successful step.  On the next run it resumes automatically:

```bash
python maiChaos/scripts/run_sweep.py          # resumes if checkpoint exists
python maiChaos/scripts/run_sweep.py --resume  # same, explicit
python maiChaos/scripts/run_sweep.py --fresh   # delete checkpoint, start over
```

The checkpoint stores:
```json
{
  "current_sol_index": 3,
  "current_b0":        0.14,
  "last_converged_file": "data/sweep_results/.../solution.npz",
  "ds_current":        0.003,
  "timestamp":         "2026-04-20T12:34:56"
}
```

---

## How to Add a New Observable to the CSV Log

1. Open `maiChaos/io/results_log.py` and add your column name to `COLUMNS`.

2. Open `maiChaos/sweep/automated_sweep.py`, find `_make_log_row()`,
   and add the computation:
   ```python
   def _make_log_row(sol_idx, b0_y, result, path, ds, bif_flag=False,
                     my_observable=None):
       return {
           ...,
           "my_column": f"{my_observable:.6f}" if my_observable else "",
       }
   ```

3. Compute the observable in `_sweep_one_solution()` after each converged step
   and pass it to `_make_log_row`.

The CSV header is written once on first creation, so delete `sweep_log.csv`
if you change columns mid-sweep.

---

## How to Extend the Floquet Analysis

`maiChaos/analysis/floquet.py::run_floquet()` returns:
```python
{
    "R":           np.ndarray,   # (block_size, block_size) Schur factor
    "eigenvalues": np.ndarray,   # dominant Floquet multipliers (complex)
    "tang":        np.ndarray,   # converged tangent block
    "rel_error":   float,        # periodic-orbit residual
}
```

To add a new quantity (e.g. Lyapunov exponents from the log of |μ|):

```python
from maiChaos.analysis.floquet import run_floquet

data = run_floquet(input_dict, param_dict, block_size=64, maxit=16)
lyapunov = np.log(np.abs(data["eigenvalues"])) / float(input_dict["T"])
```

To seed the next power iteration from a previous result (warm start):
```python
data2 = run_floquet(input_dict_new, param_dict_new,
                    tang_init=data["tang"])
```

---

## Architecture Notes

- **State vector**: `z = [flatten(input_dict), b0_y]` — b0_y is the last element.
- **Objective**: `lib.loss_functions.objective_RPO(input_dict, param_dict)`.
  `param_dict['b0']` is updated at each step; `b0_x = 0` stays fixed.
- **GMRES**: uses `lib.linalg.gmres` (Matt's custom implementation, not
  `jax.scipy`). JVP-based matrix-free matvec via `jax.jvp`.
- **Dealiasing**: `mask * rfft2(fields)` → `irfft2` is applied after **every**
  Newton iteration, both in `newton_gmres.py` and `arc_length.py`.
- **JIT boundary**: `@jax.jit` wraps single-iteration updates only.
  The outer Newton loop, timeout check, and convergence test are plain Python.
- **lib/ is read-only**: never modify anything in `jax_scripts/lib/`.
