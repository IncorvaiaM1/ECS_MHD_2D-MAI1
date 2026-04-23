# Data Directory

This directory contains all data inputs and outputs for the MHD arc-length
continuation sweep.

## Layout

```
data/
├── raw/
│   └── Re40/               # Andrew's original .npy files (kept in Re40_all_for_matt/)
│                           # Reference only — do NOT move the .npy files.
├── converted/
│   └── Re40/               # Converted .npz files from jax_scripts/convert.py
│       ├── solution_000.npz
│       ├── solution_001.npz
│       └── ...
└── sweep_results/
    └── Re40/
        └── B0_sweep/
            ├── sweep_log.csv           # Master results table (one row per step)
            ├── checkpoint.json         # Resume state (JSON)
            ├── checkpoint_tangent.npy  # Tangent vector at last checkpoint
            ├── sol_000/
            │   ├── B0_0.0100/
            │   │   └── solution.npz
            │   ├── B0_0.0200/
            │   │   └── solution.npz
            │   ├── floquet/
            │   │   └── B0_0.0100_floquet.npz
            │   └── animations/
            │       └── B0_0.0100.gif
            ├── sol_001/
            │   └── ...
            └── bifurcation_diagram.png
```

## Converting raw solutions

```bash
cd jax_scripts
python convert.py
```

Output is written to `data/converted/Re40/`.

## Running the sweep

```bash
cd jax_scripts
python maiChaos/scripts/run_sweep.py --config maiChaos/config/default_config.yaml
```

## Resuming an interrupted sweep

The sweep automatically resumes from `checkpoint.json` if it exists.  To start
fresh, delete `checkpoint.json` and `checkpoint_tangent.npy`.

## CSV columns

| Column             | Description                                    |
|--------------------|------------------------------------------------|
| `sol_idx`          | Seed solution index                            |
| `b0_y`             | Mean magnetic field Bᵧ at this step            |
| `converged`        | Whether Newton converged                       |
| `newton_iters`     | Number of Newton iterations                    |
| `residual_norm`    | Final residual ‖G‖                             |
| `solution_norm`    | ‖z‖ of the solution vector                     |
| `ds_used`          | Arc-length step actually taken                 |
| `wall_time_s`      | Wall-clock time for this step (seconds)        |
| `bifurcation_flag` | 1 if a bifurcation event was detected          |
| `solution_path`    | Relative path to the saved solution.npz        |
