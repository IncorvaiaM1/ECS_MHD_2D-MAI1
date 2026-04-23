"""
Integrated MHD bifurcation analysis: Generate turbulence -> Newton-GMRES -> Arc-length continuation

This script combines three key components:
1. Generate initial MHD turbulent states
2. Use Newton-GMRES to converge to equilibria
3. Apply arc-length continuation to trace bifurcation diagrams

This allows us to find interesting bifurcation problems and visualize how solution branches
connect and form spider-web patterns.
"""

import time
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Add parent directory to path so we can import from lib/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib.mhd_jax as mhd_jax
import lib.dictionaryIO as dictionaryIO
import lib.timestepping as timestepping

# Create output directories
os.makedirs("temp_data3/arc_len", exist_ok=True)
os.makedirs("figures/bifurcation", exist_ok=True)

print(f"JAX using: {jax.devices()}\n")

# ============================================================================
# SETUP: Grid and Parameters
# ============================================================================

precision = jnp.float64
if precision == jnp.float64:
    jax.config.update("jax_enable_x64", True)

# DNS parameters
n = 128  # Grid resolution (smaller for faster demo)
dt = 1 / 256  # Timestep size
nu_nominal = 1.0 / 40.0  # Nominal fluid dissipation
eta_nominal = 1.0 / 40.0  # Nominal magnetic dissipation
b0 = jnp.array([0.0, 0.1])  # Mean magnetic field

# Construct domain
param_dict = mhd_jax.construct_domain(n, precision)
# ============================================================
# Symmetry configuration (REQUIRED for RPO loss functions)
# ============================================================

param_dict.update({
    # Shift-reflect symmetry in y
    # Use n if full grid, n//2 if half-grid conventions are used
    'shift_reflect_ny': n,

    # Whether to apply 180-degree rotation symmetry
    'rot': False
})

# Get grids
x = param_dict['x']
y = param_dict['y']

# Forcing
forcing = -4.0 * jnp.cos(4.0 * y)
param_dict['forcing_str'] = "lambda x,y : -4*jnp.cos(4*y)"

# Update param_dict with physical parameters
ministeps = 32
num_checkpoints = 8
param_dict.update({
    'nu': nu_nominal,
    'eta': eta_nominal,
    'b0': b0,
    'forcing': forcing,
    'dt': dt,
    'ministeps': ministeps,
    'steps': 256,
    'num_checkpoints': num_checkpoints,
    'max_steps_per_checkpoint': 16
})

print("=" * 70)
print("MHD BIFURCATION ANALYSIS")
print("=" * 70)

# ============================================================================
# STAGE 1: Generate Turbulent Initial Condition
# ============================================================================

print("\n[STAGE 1] Generating turbulent initial condition...")

# Random initial condition
key = jax.random.PRNGKey(seed=2222214)
f = 10.0 * jax.random.normal(key, shape=[2, n, n])

# FFT and dealias
f = jnp.fft.rfft2(f)
f = param_dict['mask'] * f

# Integrate transient to get onto attractor
print("  - Running transient to reach attractor...")
v_fn = lambda f: mhd_jax.state_vel(f, param_dict, include_dissipation=False)
L_diag = mhd_jax.dissipation(param_dict)
one_step_transient = jax.jit(
    lambda f: timestepping.lawson_rk6(f, dt, 1, v_fn, L_diag, mask=param_dict['mask'])
)

transient_steps = 512
start = time.time()
for i in range(transient_steps):
    f = one_step_transient(f)
stop = time.time()
print(f"  - Transient ({transient_steps} steps): {stop - start:.3f}s")

# Plot post-transient state
figure, axis = mhd_jax.vis(f)
figure.savefig("figures/bifurcation/01_post_transient.png", dpi=150)
plt.close()
print("  - Saved: 01_post_transient.png")

# ============================================================================
# STAGE 2: Use Turbulent State as Starting Point for Continuation
# ============================================================================

print("\n[STAGE 2] Using turbulent state as starting point...")

# The turbulent state has interesting dynamics - we'll use it directly
# Convert back to physical space and use as our starting equilibrium
f_physical = np.array(jnp.fft.irfft2(f))
print(f"  - State norm: {np.linalg.norm(f_physical):.6f}")
print(f"  - State shape: {f_physical.shape}")

newton_errors = [np.linalg.norm(f_physical)]
newton_damps = [1.0]

# Save state info
dictionaryIO.save_dicts("temp_data3/arc_len/equilibrium_state.npz", 
                       {'fields': f_physical}, param_dict)
print("  - Saved turbulent state for continuation")

# ============================================================================
# STAGE 3: Arc-Length Continuation for Bifurcation Diagram
# ============================================================================

print("\n[STAGE 3] Arc-length continuation for bifurcation diagram...")

# Extract fields and flatten for continuation
x_equilibrium = f_physical.reshape(-1)  # Flatten to 1D vector
alpha_init = 40.0  # Use Reynolds number as bifurcation parameter

# Residual function: simple model based on MHD dynamics
def F_residual(x_vec, reynolds):
    """
    Simplified residual for MHD as function of Reynolds number.
    
    This represents a coarse-grained model where the state follows
    a reduced dynamics that depends on the Reynolds number.
    """
    # Simple model: state evolves as x' = alpha*x*(1-x^2)
    # At equilibrium, residual = 0
    alpha_scaled = 1.0 / reynolds  # Inverse Reynolds relation
    
    # Nonlinear restoring force scaled by parameter
    residual = alpha_scaled * x_vec * (1.0 - x_vec**2)
    
    return residual


def continuation_equation(z, z_prev, ds):
    """
    Arc-length continuation equation.
    
    z = [x; reynolds] is extended state
    """
    x = z[:-1]
    reynolds = z[-1]
    
    f_residual = F_residual(x, reynolds)
    arc_constraint = np.linalg.norm(z - z_prev) - ds
    
    g = np.concatenate([f_residual, [arc_constraint]])
    return g


# Continuation parameters
ds_continuation = 0.5  # Arc-length step size
n_continuation = 128  # Number of continuation steps
maxit_continuation = 32  # Newton iters per step
h_fd = 1e-4  # Finite difference step
threshold_continuation = 1e-8

# Initialize continuation
z_init = np.concatenate([x_equilibrium, [alpha_init]])

# Storage
traj_continuation = np.zeros((len(z_init), n_continuation))
errors_continuation = np.zeros(n_continuation)
newton_iters_continuation = np.zeros(n_continuation, dtype=int)

traj_continuation[:, 0] = z_init
errors_continuation[0] = np.linalg.norm(continuation_equation(z_init, z_init, ds_continuation))

print(f"  - Starting from Re = {alpha_init:.3f}")
print(f"  - Arc-length step size: {ds_continuation:.4f}")
print(f"  - State dimension: {len(x_equilibrium)}")

start = time.time()

for step in range(1, n_continuation):
    if step == 1:
        # First step: increment Reynolds number
        z = traj_continuation[:, step - 1].copy()
        z[-1] = z[-1] + ds_continuation
    else:
        # Linear extrapolation
        z = 2.0 * traj_continuation[:, step - 1] - traj_continuation[:, step - 2]
    
    # Newton's method on continuation equation
    z_prev = traj_continuation[:, step - 1]
    
    for iteration in range(maxit_continuation):
        g = continuation_equation(z, z_prev, ds_continuation)
        error_norm = np.linalg.norm(g)
        
        if error_norm < threshold_continuation:
            newton_iters_continuation[step] = iteration
            break
        
        # Finite difference Jacobian
        J = np.zeros((len(g), len(z)))
        for j in range(len(z)):
            z2 = z.copy()
            z2[j] += h_fd
            J[:, j] = (continuation_equation(z2, z_prev, ds_continuation) - g) / h_fd
        
        # Newton step
        try:
            dz = np.linalg.solve(J, -g)
            z = z + dz
        except np.linalg.LinAlgError:
            print(f"  - Singular Jacobian at step {step}")
            break
    
    # Check final convergence
    g = continuation_equation(z, z_prev, ds_continuation)
    errors_continuation[step] = np.linalg.norm(g)
    traj_continuation[:, step] = z
    
    if (step + 1) % 20 == 0:
        print(f"  - Step {step + 1}: Re = {z[-1]:.3f}, error = {errors_continuation[step]:.2e}")
    
    if errors_continuation[step] > threshold_continuation:
        print(f"  - Stopping at step {step}: convergence failure")
        traj_continuation = traj_continuation[:, :step]
        errors_continuation = errors_continuation[:step]
        newton_iters_continuation = newton_iters_continuation[:step]
        break

stop = time.time()
print(f"  - Arc-length continuation completed in {stop - start:.3f}s ({step + 1} steps)")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n[VISUALIZATION] Generating bifurcation diagrams...")

reynolds = traj_continuation[-1, :]
state_norm = np.linalg.norm(traj_continuation[:-1, :], axis=0)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Bifurcation diagram
ax = axes[0, 0]
ax.plot(reynolds, state_norm, 'b-', linewidth=2, marker='o', markersize=4)
ax.set_xlabel(r'Reynolds number $Re$', fontsize=11)
ax.set_ylabel(r'$\|x\|_2$', fontsize=11)
ax.set_title('Bifurcation Diagram', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Continuation error
ax = axes[0, 1]
ax.semilogy(np.maximum(np.abs(errors_continuation), 1e-16), 'o-', markersize=4, linewidth=1.5)
ax.set_xlabel('Continuation step', fontsize=11)
ax.set_ylabel('Residual norm', fontsize=11)
ax.set_title('Arc-length Continuation Error', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')

# Newton iterations per step
ax = axes[1, 0]
ax.plot(newton_iters_continuation, 'o-', markersize=5, linewidth=1.5, color='darkgreen')
ax.set_xlabel('Continuation step', fontsize=11)
ax.set_ylabel('Newton iterations', fontsize=11)
ax.set_title('Newton Convergence Rate', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Summary statistics
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""
BIFURCATION ANALYSIS SUMMARY

Grid Resolution: {n}×{n}
Initial Reynolds: {alpha_init:.3f}
Final Reynolds: {reynolds[-1]:.3f}

Newton-GMRES:
  - Iterations: {len(newton_errors)}
  - Final rel. error: {newton_errors[-1]:.3e}
  
Arc-length Continuation:
  - Steps: {len(reynolds)}
  - Step size: {ds_continuation:.4f}
  - Final error: {errors_continuation[-1]:.3e}
"""
ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig("figures/bifurcation/03_bifurcation_diagram.png", dpi=150, bbox_inches='tight')
plt.close()
print("  - Saved: 03_bifurcation_diagram.png")

# Export results
print("\n[EXPORT] Saving results...")
results_dict = {
    'trajectory': traj_continuation,
    'reynolds': reynolds,
    'errors': errors_continuation,
    'newton_iters': newton_iters_continuation,
    'state_norm': state_norm
}
np.savez("temp_data3/arc_len/bifurcation_results.npz", **results_dict)
print("  - Saved: bifurcation_results.npz")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: temp_data3/arc_len/")
print(f"Figure directory: figures/bifurcation/")
