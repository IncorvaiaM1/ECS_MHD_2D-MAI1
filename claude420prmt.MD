# Claude Code Prompt: MHD Arc-Length Continuation System (ECS_MHD_2D_MAI1)

---

## CONTEXT — READ THIS FIRST

This is a research codebase for finding Exact Coherent Structures (ECS) in 2D
Magnetohydrodynamics (MHD). The goal is to track solution branches as the mean
magnetic field B₀ varies, using pseudo arc-length continuation, and detect
bifurcations in the process.

The repo is a fork of mgolden30/ECS_MHD_2D. We are working exclusively in
ECS_MHD_2D_MAI1. Do NOT touch anything outside this repo.

Before writing any code, read these existing files in full:
- `jax_scripts/lib/mhd_jax.py`        ← MHD physics: timestepping, residuals, domain
- `jax_scripts/lib/dictionaryIO.py`    ← how solutions are saved/loaded (.npz format)
- `jax_scripts/lib/timestepping.py`    ← integrator
- `jax_scripts/lib/linalg.py`          ← custom GMRES + hookstep implementations
- `jax_scripts/lib/loss_functions.py`  ← objective_RPO and other loss functions
- `jax_scripts/lib/utils.py`           ← compile_objective_and_Jacobian, line_search_unravel
- `jax_scripts/newton.py`              ← PRIMARY Newton solver — base maiChaos on THIS
- `jax_scripts/newton_hookstep_v2.py`  ← hookstep variant — secondary, understand the trust region only
- `jax_scripts/floquet.py`             ← existing Floquet analysis (migrate)
- `jax_scripts/animation.py`           ← existing animation code (migrate)
- `jax_scripts/convert.py`             ← converts Andrew's Re=40 .npy → .npz format
- `jax_scripts/generate_turbulence.py` ← generates turbulent initial conditions
- `jax_scripts/arc_len_stuff/arc_len_main.py`              ← existing arc-length attempt
- `jax_scripts/arc_len_stuff/arclength_cont.py`            ← existing arc-length core
- `jax_scripts/arc_len_stuff/mhd_continuation_complete.py` ← most complete version so far
- `jax_scripts/arc_len_stuff/TEMPLATE_minimal.py`          ← minimal template

**Newton solver priority:** `newton.py` is the PRIMARY base — it is JIT-compiled,
has line search, adjoint/transpose option, and cleaner structure than the hookstep variants.
`newton_hookstep_v2.py` provides only the hookstep trust-region as an optional add-on.

Understand the full picture before writing anything.

---

## TASK: Reorganize and Build

### STEP 1 — New Folder Structure

Reorganize `jax_scripts/` into this structure. Move existing files, don't delete them.

```
jax_scripts/
│
├── lib/                          # UNCHANGED — core MHD library
│   ├── mhd_jax.py
│   ├── dictionaryIO.py
│   ├── timestepping.py
│   ├── linalg.py
│   ├── loss_functions.py
│   ├── adam.py
│   ├── preconditioners.py
│   ├── symmetry.py
│   └── utils.py
│
├── maiChaos/                     # ALL new continuation + analysis code lives here
│   │
│   ├── config/
│   │   └── default_config.yaml   # all tunable parameters in one place
│   │
│   ├── continuation/
│   │   ├── __init__.py
│   │   ├── arc_length.py         # JAX arc-length continuation core (NEW)
│   │   ├── newton_gmres.py       # Newton-GMRES with hookstep, migrated + improved
│   │   └── predictor.py          # tangent predictor, step-size control
│   │
│   ├── sweep/
│   │   ├── __init__.py
│   │   ├── automated_sweep.py    # main automation loop (NEW, replaces old automated_Newton.py)
│   │   └── checkpoint.py         # JSON + npz checkpointing, resume logic
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── floquet.py            # migrated + automated from jax_scripts/floquet.py
│   │   ├── animate.py            # migrated + automated from jax_scripts/animation.py
│   │   └── bifurcation_detect.py # branch detection, fold point flagging (NEW)
│   │
│   ├── io/
│   │   ├── __init__.py
│   │   ├── data_manager.py       # centralized save/load for all outputs (NEW)
│   │   └── results_log.py        # CSV append logger (NEW)
│   │
│   └── scripts/
│       ├── run_sweep.py          # entrypoint: python run_sweep.py --config default_config.yaml
│       ├── run_floquet.py        # entrypoint: analyze a specific solution
│       └── run_animate.py        # entrypoint: animate a specific solution
│
├── legacy/                       # old top-level scripts, moved here, untouched
│   ├── newton.py
│   ├── newton_hookstep.py
│   ├── newton_hookstep_v2.py
│   ├── floquet.py
│   ├── animation.py
│   ├── adjoint_descent.py
│   ├── look_for_attractor.py
│   ├── inspect_solution_files.py
│   ├── MHDSolutionExtractor.py
│   ├── NPYtoNPZConverter.py
│   ├── NPYtoNPZConverter1.py
│   ├── setup_MATLAB_animation.py
│   ├── upscale_state.py
│   └── visualize_state.py
│
├── legacy_arc_len/               # old arc-length attempts, moved here, untouched
│   ├── arc_len_main.py
│   ├── arclength_cont.py
│   ├── arclength_mhd_bifurcation.py
│   ├── mhd_continuation_complete.py
│   ├── mhd_integration_guide.py
│   ├── python_mat_aLen.py
│   ├── TEMPLATE_minimal.py
│   ├── DESCRIB_ARC_LEN_MAIN.md
│   └── describ_arc_len_main
│
└── convert.py                    # keep at top level (utility script)
```

**Data output structure** (outside jax_scripts, at repo root):

```
data/
├── README.md                     # explains folder structure
├── raw/
│   └── Re40/                     # Andrew's original .npy files stay in Re40_all_for_matt/
│                                 # (symlink or reference only — don't move the .npy files)
├── converted/
│   └── Re40/                     # converted .npz from convert.py
│       ├── solution_000.npz
│       ├── solution_001.npz
│       └── ...
└── sweep_results/
    └── Re40/
        └── B0_sweep/
            ├── sweep_log.csv           # master results table
            ├── checkpoint.json         # resume state
            ├── sol_000/
            │   ├── B0_0.010/
            │   │   └── solution.npz
            │   ├── B0_0.020/
            │   │   └── solution.npz
            │   ├── floquet/
            │   │   └── B0_0.010_floquet.npz
            │   └── animations/
            │       └── B0_0.010.gif
            ├── sol_001/
            │   └── ...
            └── bifurcation_diagram.png
```

---

### STEP 2 — `config/default_config.yaml`

```yaml
# MHD Continuation Configuration
physics:
  Re: 40
  nu: 0.025          # 1/Re
  eta: 0.025         # 1/Re (equal dissipation)
  n: 128             # grid size
  forcing: "-4*cos(4*y)"
  b0_start: [0.0, 0.01]   # initial mean magnetic field [bx, by]
  b0_target: 1.0           # max by to sweep to
  b0_step: 0.01            # initial increment

continuation:
  ds_init: 0.001           # initial arc-length step
  ds_min: 1.0e-5           # minimum step (give up if below this)
  ds_max: 0.05             # maximum step
  ds_grow: 1.2             # multiply ds by this on easy steps
  ds_shrink: 0.5           # multiply ds by this on failed steps
  max_newton_iter: 128
  newton_tol: 1.0e-10
  max_time_hours: 10.0     # wall-clock timeout per solution
  convergence_order: 8     # arc-length norm exponent (from notebook: ||sol - sol_prev||^{-ss})

gmres:
  restart: 50
  max_iter: 500
  tol: 1.0e-8

analysis:
  run_floquet: true
  run_animation: true
  animation_fps: 10
  bifurcation_jump_threshold: 0.1   # flag if solution norm jumps by this much

paths:
  raw_solutions: "Re40_all_for_matt"
  converted_solutions: "data/converted/Re40"
  output_dir: "data/sweep_results/Re40/B0_sweep"
```

---

### STEP 3 — Core Files to Implement

#### `maiChaos/continuation/arc_length.py`

Implement pseudo arc-length continuation. Key requirements:

**The augmented system** (from the notebook):
The continuation is parameterized by arc-length `s`. At each step we solve:

```
G(z, λ) = 0                          # original equations
N(z, λ) = (z - z_prev)·ż_prev + (λ - λ_prev)·λ̇_prev - ds = 0   # arc-length constraint
```

where `z` is the state vector and `λ = B₀_y` is the bifurcation parameter.

**State vector structure** (from notebook):
```
Input dict keys:  {T, sf, f, v, z, b̄}
Param dict keys:  {z, t/s, hx, hy, forcing, v, z, B₀}

F vector layout (what gets packed into z):
  f      → fluid field (sliced for T)
  s_f    → phase condition for f
  v      → velocity/vorticity
  ...    → (unused slots)
  b_x    → magnetic field x
  b_y    → magnetic field y (the continuation parameter λ)
  
Arc-length constraint: ||solution - solution_prev||  (Euclidean in state space)
```

**Algorithm:**
1. Tangent prediction: `z_pred = z_prev + ds * ż` where `ż` is the normalized tangent
2. Newton correction on the augmented system `[F; N]`
3. Use JAX `jax.linear_util` or `jax.scipy.sparse.linalg.gmres` for the linear solve
4. Use `jax.jacobian` for the Jacobian of F (NOT finite differences)
5. Adaptive step control: shrink `ds` on failure, grow on easy convergence

**Critical JAX implementation notes:**
- Use `jax.jit` on the residual evaluation
- Use `jax.jacfwd` for the Jacobian (forward-mode, cheaper for tall systems)
- The GMRES solve: use `jax.scipy.sparse.linalg.gmres` with a matrix-free
  Jacobian-vector product via `jax.jvp`
- Pack/unpack state vector to/from dict using `dictionaryIO.py` conventions

```python
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres
import sys
sys.path.append('../..')
from lib import mhd_jax, dictionaryIO

def jvp_matvec(residual_fn, z, v):
    """Matrix-free Jacobian-vector product using JAX forward-mode AD."""
    _, jvp_val = jax.jvp(residual_fn, (z,), (v,))
    return jvp_val

def arc_length_step(z_prev, tangent_prev, ds, residual_fn, params, config):
    """
    One arc-length continuation step.
    Returns (z_new, tangent_new, ds_used, converged, n_iters)
    """
    # 1. Predict
    z_pred = z_prev + ds * tangent_prev

    # 2. Define augmented residual
    def augmented_residual(z):
        F = residual_fn(z, params)
        N = jnp.dot(z - z_prev, tangent_prev) - ds
        return jnp.concatenate([F, jnp.array([N])])

    # 3. Newton-GMRES on augmented system
    # ... (full Newton loop)

    # 4. Update tangent via GMRES on Jacobian
    # ... 

    # 5. Orient tangent (dot with previous tangent > 0)
    # ...
```

#### `maiChaos/continuation/newton_gmres.py`

**Base this on `newton.py` (Matt's version) — it is the most current and performant.**
Do NOT use `newton_hookstep_v2.py` as the base; the hookstep is a secondary add-on.

Key things to preserve from `newton.py`:
- The `@jax.jit` compiled `newton_gmres_update` inner function
- The `use_transpose` / adjoint (Gauss-Newton) option
- Line search (`do_line_search`)
- `io_callback` timing pattern inside JIT
- `jax.debug.print` logging format
- Dealiasing step after every Newton iteration:
  ```python
  fields = param_dict['mask'] * jnp.fft.rfft2(fields)
  fields = jnp.fft.irfft2(fields)
  input_dict['fields'] = fields
  ```
- The flatten/unflatten pattern using `jax.flatten_util.ravel_pytree`

Key things to ADD on top of `newton.py`:
- **Wall-clock timeout**: check `time.time() - start_time > config.max_time_hours * 3600`
  at the top of each iteration; return early with `converged=False` if exceeded.
  (Note: you cannot do this inside `@jax.jit` — the timeout check must be in the
  outer Python loop that calls `newton_gmres_update`)
- **Hookstep trust region** from `newton_hookstep_v2.py`: optionally replace line search
  with `newton_gmres_hookstep_v2` from `lib.linalg` when `config.use_hookstep=True`
- **Structured return**: return a dataclass or dict instead of just `input_dict`:
  ```python
  return {
      "input_dict": input_dict,
      "converged": bool,
      "n_iters": int,
      "residual_history": list[float],
      "wall_time_sec": float,
      "final_residual": float,
  }
  ```
- **Convergence check**: after each iteration check `jnp.linalg.norm(b) < config.newton_tol`
  and break early (again in the outer Python loop, not inside JIT)

The Jacobian is already matrix-free via `jax.jvp` through `utils.compile_objective_and_Jacobian`
— preserve this exactly. Do NOT introduce finite differences.

Solvability condition (from notebook):
- `u` must be orthogonal to `b_y`
- `Jα = -J∂b_y` — one Jacobian evaluation gives the RHS for the bordered system
- This is handled naturally by GMRES on the augmented arc-length system in `arc_length.py`;
  `newton_gmres.py` itself doesn't need to implement this explicitly.

```python
def newton_gmres_solve(residual_fn, z0, param_dict, config, start_time):
    """
    Newton-GMRES with line search (and optional hookstep), JIT inner loop, timeout.
    Built from newton.py as base. 
    Returns: dict with input_dict, converged, n_iters, residual_history, wall_time_sec
    """
```

#### `maiChaos/sweep/automated_sweep.py`

This is the main loop. Full logic:

```python
"""
Automated B₀ sweep over Andrew's Re=40 solutions.

Algorithm per solution:
  1. Load solution_i from data/converted/Re40/solution_i.npz
  2. Run Newton-GMRES to converge it at B₀ = b0_start
  3. Once converged (residual < newton_tol), save to data/sweep_results/.../sol_i/B0_X/
  4. Take converged solution as new initial guess
  5. Increment B₀ by b0_step (arc-length continuation handles this)
  6. Repeat until B₀ = b0_target, non-convergence, or timeout
  7. On convergence at each B₀: append row to sweep_log.csv
  8. On non-convergence: try shrinking ds, retry; if still fails, move to next solution
  9. After each solution: run Floquet analysis and animation if enabled in config

Checkpointing:
  - checkpoint.json updated after every successful B₀ step
  - On startup: check if checkpoint.json exists, resume from there
  - checkpoint.json format:
    {
      "current_sol_index": 3,
      "current_b0": 0.15,
      "last_converged_file": "data/sweep_results/.../sol_003/B0_0.150/solution.npz",
      "ds_current": 0.008,
      "timestamp": "2026-04-20T12:34:56"
    }
"""
```

#### `maiChaos/io/results_log.py`

CSV logger. Every converged solution appends one row:

```
sol_index, b0_x, b0_y, residual_norm, n_newton_iters, wall_time_sec, 
energy, enstrophy, helicity, bifurcation_flag, ds_used, output_path
```

`bifurcation_flag` is set to `True` if the solution norm jumped by more than
`config.bifurcation_jump_threshold` from the previous step (crude but useful
for flagging candidate bifurcation points for later investigation).

#### `maiChaos/analysis/floquet.py`

Migrate from `jax_scripts/floquet.py`. Automate it:
- Takes a solution `.npz` and config
- Computes Floquet multipliers
- Saves to `floquet/B0_X_floquet.npz` alongside the solution
- Returns a summary dict (leading multiplier, stability flag)

#### `maiChaos/analysis/bifurcation_detect.py`

After each B₀ step, check for bifurcations:
- **Fold/turning point**: tangent's λ-component changes sign
- **Branch point**: leading Floquet multiplier crosses unit circle
- **Norm jump**: solution norm changes by more than threshold
- Log all flags to CSV and print warnings

#### `maiChaos/scripts/run_sweep.py`

```python
#!/usr/bin/env python
"""
Main entrypoint.

Usage:
    python run_sweep.py                          # uses default_config.yaml
    python run_sweep.py --config my_config.yaml  # custom config
    python run_sweep.py --sol 5                  # run only solution index 5
    python run_sweep.py --resume                 # force resume from checkpoint
    python run_sweep.py --fresh                  # ignore checkpoint, start fresh
"""
```

---

### STEP 4 — `data/README.md`

Write a clear README explaining the full data layout, what each file contains,
how to load a solution, and how to resume an interrupted sweep.

---

### STEP 5 — `maiChaos/README.md`

Write a README for the maiChaos package covering:
- Package overview and module descriptions
- How to run a sweep
- How to resume
- How to add a new observable to the CSV log
- How to extend the Floquet analysis

---

## IMPLEMENTATION PRIORITIES

Do these in order. If time/context runs short, prioritize earlier items.

1. **Folder reorganization** — move all legacy files first, verify nothing breaks
2. **`default_config.yaml`** — everything configurable from one file
3. **`data_manager.py`** and **`results_log.py`** — data layer first
4. **`checkpoint.py`** — resume logic
5. **`newton_gmres.py`** — improved Newton solver (builds on existing newton.py)
6. **`arc_length.py`** — the continuation core
7. **`predictor.py`** — tangent prediction and step control
8. **`automated_sweep.py`** — the main loop
9. **`floquet.py`** (migrated + automated)
10. **`animate.py`** (migrated + automated)
11. **`bifurcation_detect.py`**
12. **`run_sweep.py`**, **`run_floquet.py`**, **`run_animate.py`**
13. READMEs

---

## IMPORTANT IMPLEMENTATION NOTES

- **JAX precision**: always call `jax.config.update("jax_enable_x64", True)` at top of every script
- **JAX JIT**: wrap residual evaluations in `@jax.jit`; do NOT jit the outer Newton loop (Python-level control flow)
- **GMRES**: use `jax.scipy.sparse.linalg.gmres` with matrix-free matvec via `jax.jvp` — no explicit Jacobian matrix
- **Timeout**: use `time.time()` checks inside Newton loop — JAX doesn't have built-in timeouts
- **b0 format**: B₀ is a 2-vector `[b0_x, b0_y]`; continuation is in `b0_y`; `b0_x` stays fixed at 0.0
- **Solution format**: follow `dictionaryIO.py` exactly for all saves/loads — don't invent new formats
- **Arc-length norm**: from the notebook, the arc-length constraint uses `||solution - solution_prev||` in the full state-space norm including all fields
- **Power iteration fallback**: if GMRES fails, fall back to power iteration as noted in notebook: `(JS + I)v = v̄`
- **Don't break existing lib/**: `lib/` is shared with Matt's original code — only add, never modify

---

## WHAT SUCCESS LOOKS LIKE

After your implementation, this should work:

```bash
cd ECS_MHD_2D_MAI1/jax_scripts
python maiChaos/scripts/run_sweep.py

# Output:
# Loading checkpoint... none found. Starting fresh.
# Processing solution 0 / 179
#   Converting Re40_all_for_matt/soln_array_Re40_0.npy ...
#   Newton converging at B0_y=0.01 ... residual: 1.2e-3 → 4.1e-6 → 8.3e-11 ✓
#   Saved: data/sweep_results/Re40/B0_sweep/sol_000/B0_0.010/solution.npz
#   Arc-length step to B0_y=0.02 ...
#   Newton converging ... ✓
#   Checkpoint saved.
#   ...
# Processing solution 1 / 179
# ...
```

And after a sweep, `data/sweep_results/Re40/B0_sweep/sweep_log.csv` has a
complete record of every converged solution with all observables logged.
C:\Users\micha\PycharmProjects\ECS_MHD_2D-MAI1\claude420prmt.MD\
necessary task 