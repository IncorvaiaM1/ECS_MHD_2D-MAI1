# High-Level Overview of `arc_len_main.py`

This script performs **MHD bifurcation analysis** in three integrated stages. Here's what's happening:

## **Overall Goal**
Generate MHD turbulence → trace how solutions change as a bifurcation parameter varies → visualize the bifurcation diagram showing how solution branches connect.

---

## **STAGE 1: Generate Turbulent Initial Condition**
**What it does:** Creates an interesting starting state by simulating the MHD equations.

```
Random noise → Time integrate (512 steps) → Turbulent attractor state
```

**Key components:**
- **Domain construction**: Sets up spectral grid with wavenumbers, dealiasing masks
- **Random initialization**: Creates [2, 128, 128] state (vorticity + magnetic current)
- **Lawson-RK6 integration**: Efficiently solves MHD equations with exponential treatment of dissipation
- **Output**: Turbulent snapshot saved as visualization

**Why?** Starting with turbulent dynamics gives a realistic initial state with interesting nonlinear features that will reveal bifurcation structure.

---

## **STAGE 2: Use Turbulent State as Starting Point**
**What it does:** Prepares the turbulent state for continuation.

```
Fourier space state → Convert to physical space → Flatten into 1D vector
```

**Key components:**
- **IFFT conversion**: `jnp.fft.irfft2(f)` transforms from spectral to real space
- **Flattening**: Reshapes [2, 128, 128] → [32,768] element vector for continuation
- **Storage**: Saves state metadata

**Why?** Arc-length continuation needs a 1D vector representation. The flattened state is the starting point `x_0` for tracing the bifurcation curve.

---

## **STAGE 3: Arc-Length Continuation for Bifurcation Diagram**
**What it does:** Traces how the solution evolves as Reynolds number varies, revealing bifurcation structure.

### **The Math:**
Arc-length continuation solves two equations simultaneously:

```
F(x, Re) = 0              (governing equation)
||z - z_prev|| = ds       (arc-length constraint)
```

where `z = [x; Re]` is the extended state (state + parameter).

### **The Algorithm:**
```
For each continuation step:
  1. Linear extrapolation:   z_guess = 2*z_prev - z_prev_prev
  2. Newton iterations (≤32):
     - Compute residual g(z)
     - Finite difference Jacobian J
     - Solve J*dz = -g
     - Update z = z + dz
  3. Check convergence: ||g|| < threshold
  4. Store trajectory point
  5. Move to next Reynolds number
```

### **Key Parameters:**
| Parameter | Value | Role |
|-----------|-------|------|
| `ds_continuation` | 0.5 | Step size along curve |
| `maxit_continuation` | 32 | Max Newton iterations per step |
| `h_fd` | 1e-4 | Finite difference step for Jacobian |
| `n_continuation` | 128 | Max number of steps to trace |

### **Simple Model Used:**
```python
F_residual(x, Re) = (1/Re) * x * (1 - x²)
```
This is a simplified nonlinear model. A real implementation would use the MHD time derivative at equilibrium.

### **Output:**
- **trajectory**: Shape (32,769 × 128) - state + Reynolds number at each step
- **reynolds**: Parameter values traced
- **errors**: Residual norm (should be ≈ 1e-8)
- **newton_iters**: Newton iterations needed per step (should be ≈ 2-5)

---

## **STAGE 4: Visualization**

Creates a **4-panel figure** showing:

1. **Bifurcation Diagram** (top-left)
   - x-axis: Reynolds number
   - y-axis: State norm ||x||
   - Shows how solution amplitude varies with parameter
   - Reveals fold bifurcations (turning points), pitchfork bifurcations (branches split)

2. **Continuation Error** (top-right)
   - Log scale plot of ||g(z)||
   - Should decrease as continuation progresses
   - If increasing, indicates lost convergence

3. **Newton Convergence** (bottom-left)
   - Number of Newton iterations per step
   - Typically 2-5 iterations for well-conditioned problems
   - Spikes indicate near-singular Jacobian

4. **Summary Statistics** (bottom-right)
   - Grid resolution, Reynolds range
   - Total steps traced
   - Final errors and iteration counts

---

## **Data Flow Diagram**

```
┌─────────────────────┐
│  Random noise [2,128,128]
└──────────┬──────────┘
           │ Lawson-RK6 (512 steps)
           ▼
┌─────────────────────┐
│  Turbulent state [2,128,128] in Fourier space
└──────────┬──────────┘
           │ IFFT + Flatten
           ▼
┌─────────────────────┐
│  State vector [32,768] at Re=40
└──────────┬──────────┘
           │ Arc-length continuation (128 steps)
           ▼
┌─────────────────────┐
│  Solution branch [32,769 × 128]
│  (includes Re parameter)
└──────────┬──────────┘
           │ Visualization
           ▼
┌─────────────────────┐
│  4-panel bifurcation diagram
│  + NPZ export
└─────────────────────┘
```

---

## **Key Concepts**

| Concept | Meaning |
|---------|---------|
| **Arc-length** | Distance along the solution curve; ensures uniform stepping |
| **Bifurcation** | Point where solution branches split or merge (change topology) |
| **Continuation** | Technique to follow solution curves as parameters vary |
| **Newton's method** | Finds roots by iteratively improving guess |
| **Finite difference** | Approximates Jacobian without automatic differentiation |
| **Fold bifurcation** | Turning point where solution amplitude peaks/troughs |
| **Pitchfork** | Symmetric branching where stable and unstable solutions exchange |

---

## **Why This Matters for MHD**

This pipeline lets you:
1. **Find equilibria** by integrating turbulent dynamics
2. **Trace bifurcation branches** to understand solution structure
3. **Visualize the "spider web"** of interconnected solution branches
4. **Identify stability changes** by tracking when branches appear/disappear

This is essential for understanding **how MHD flows transition between different regimes** as Reynolds numbers change.

---

## **Implementation Details**

### **Dependencies:**
- `jax`, `jax.numpy`: Differentiable computation
- `numpy`: Classical numerical operations
- `matplotlib`: Visualization
- `lib.mhd_jax`: MHD spectral solver
- `lib.timestepping`: Runge-Kutta integrators
- `lib.dictionaryIO`: File I/O utilities

### **Output Files:**
```
figures/bifurcation/
  ├── 01_post_transient.png        # Turbulent state visualization
  └── 03_bifurcation_diagram.png   # 4-panel analysis figure

temp_data3/arc_len/
  ├── equilibrium_state.npz        # Starting state metadata
  └── bifurcation_results.npz      # Full continuation trajectory
```

### **Tuning Parameters for Your Work:**

If you want to modify the solver behavior:
- **Finer branches**: Decrease `ds_continuation` (e.g., 0.1 instead of 0.5)
- **More coverage**: Increase `n_continuation` (e.g., 512 instead of 128)
- **Better accuracy**: Decrease `h_fd` (e.g., 1e-5 instead of 1e-4)
- **Faster computation**: Increase `ds_continuation`, decrease `n_continuation`

---

## **Common Issues & Solutions**

| Issue | Cause | Solution |
|-------|-------|----------|
| Newton doesn't converge | Jacobian singular | Check `h_fd`, use smaller `ds` |
| Error grows | Lost solution branch | Reduce step size `ds_continuation` |
| Memory issues | State too large | Reduce `n` (grid resolution) |
| Slow execution | Too many steps | Increase `ds_continuation` step size |

---

## **Next Steps for Real MHD Work**

To apply this to actual MHD bifurcations:

1. **Replace `F_residual`** with actual MHD time derivative evaluation
2. **Use Jacobian autodiff** instead of finite differences (faster, more accurate)
3. **Add phase conditions** to constrain relative periodic orbits (RPOs)
4. **Implement adaptive step sizing** to handle bifurcation points
5. **Add symmetry handling** for shift-reflect symmetries in your domain

This code provides the framework—you just need to plug in your MHD equations!
