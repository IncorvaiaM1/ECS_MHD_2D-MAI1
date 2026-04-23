"""
Arc-length continuation solver for bifurcation analysis.

Direct translation of MATLAB arc-length continuation code to Python with JAX/NumPy.
Traces solution curves by solving F(x, alpha) = 0 subject to arc-length constraint.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jacobian
import matplotlib.pyplot as plt


# ============================================================================
# Problem Definition
# ============================================================================

def F(x, alpha):
    """
    Nonlinear problem with bifurcation parameter alpha.
    
    3-point finite-difference Bratu problem:
    -u'' = alpha * exp(u)
    """
    y = np.array([
        -2*x[0] + x[1] + alpha*np.exp(x[0]),
        x[0] - 2*x[1] + x[2] + alpha*np.exp(x[1]),
        x[1] - 2*x[2] + alpha*np.exp(x[2])
    ])
    return y


def continuation_function(z, z_previous, ds):
    """
    Arc-length continuation equation.
    
    z = [x; alpha] is the extended state
    Returns: [F(x, alpha); ||z - z_previous|| - ds]
    """
    x = z[:-1]
    alpha = z[-1]
    
    f_residual = F(x, alpha)
    arc_constraint = np.linalg.norm(z - z_previous) - ds
    
    f = np.concatenate([f_residual, [arc_constraint]])
    return f


# ============================================================================
# Arc-length Continuation
# ============================================================================

def arclength_continuation():
    """
    Main arc-length continuation solver.
    Traces the bifurcation curve for the Bratu problem.
    """
    
    # Define initial solution and initial value of bifurcation parameter
    alpha = 0.0
    x = np.array([0.0, 0.0, 0.0])
    
    # Check that we have a solution
    y = F(x, alpha)
    assert np.max(np.abs(y)) < 1e-10, f"Initial condition not a solution: {np.max(np.abs(y))}"
    
    # Continuation parameters
    ds = 0.001          # Arc-length step size
    n = 1024 * 16       # Number of steps
    maxit = 128         # Max number of Newton steps
    h = 1e-3            # Finite difference parameter
    threshold = 1e-12   # When is a solution good enough?
    
    # Storage
    traj = np.zeros((4, n))
    newton_steps = np.zeros(n, dtype=int)
    norm_of_error = np.zeros(n)
    
    traj[:, 0] = np.concatenate([x, [alpha]])
    
    # Main continuation loop
    for i in range(1, n):
        if i == 1:
            # First step: increment alpha
            z = traj[:, i-1].copy()
            z[-1] = z[-1] + ds
        else:
            # Linear extrapolation from two previous solutions
            z = 2*traj[:, i-1] - traj[:, i-2]
        
        # Define continuation function for this step
        def G(z_var):
            return continuation_function(z_var, traj[:, i-1], ds)
        
        # Newton's method
        for iteration in range(maxit):
            g = G(z)
            
            if np.linalg.norm(g) < threshold:
                newton_steps[i] = iteration
                break
            
            # Finite difference Jacobian
            J = np.zeros((len(g), len(z)))
            for j in range(len(z)):
                z2 = z.copy()
                z2[j] = z2[j] + h
                J[:, j] = (G(z2) - g) / h
            
            # Newton step
            try:
                dz = np.linalg.solve(J, -g)
                z = z + dz
            except np.linalg.LinAlgError:
                print(f"Singular Jacobian at step {i}")
                break
        
        # Check convergence
        g = G(z)
        if np.linalg.norm(g) > threshold:
            print(f"Did not converge at step {i}. Error: {np.linalg.norm(g):.2e}")
            n = i
            traj = traj[:, :i]
            newton_steps = newton_steps[:i]
            norm_of_error = norm_of_error[:i]
            break
        
        norm_of_error[i] = np.linalg.norm(g)
        traj[:, i] = z
        
        if (i+1) % 100 == 0:
            print(f"Step {i+1}/{n}: alpha = {z[-1]:.6f}, error = {norm_of_error[i]:.2e}")
    
    return traj, norm_of_error, newton_steps


# ============================================================================
# Visualization
# ============================================================================

def visualize_output(traj, norm_of_error, newton_steps):
    """
    Create 2x2 visualization of continuation results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    
    # Bifurcation diagram
    ax = axes[0, 0]
    observable = np.linalg.norm(traj[0:3, :], axis=0)
    ax.plot(traj[-1, :], observable, 'b-', linewidth=1.5, marker='o', markersize=3)
    ax.set_xlabel(r'Bifurcation parameter $\alpha$', fontsize=11)
    ax.set_ylabel(r'$\|x\|_2$', fontsize=11)
    ax.set_title('Bifurcation Diagram', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Error in objective
    ax = axes[0, 1]
    ax.semilogy(norm_of_error, 'o', markersize=4, linewidth=1)
    ax.set_xlabel('Continuation step', fontsize=11)
    ax.set_ylabel('Error in objective', fontsize=11)
    ax.set_title('Are we solving the nonlinear equations?', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # Newton iterations
    ax = axes[1, 0]
    ax.plot(newton_steps, 'o', markersize=5, linewidth=1, color='darkgreen')
    ax.set_xlabel('Continuation step', fontsize=11)
    ax.set_ylabel('Newton iterations', fontsize=11)
    ax.set_title('How many Newton iterations did we need?', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # State components
    ax = axes[1, 1]
    ax.plot(traj[-1, :], traj[0, :], '-', linewidth=1.5, label='x[0]', marker='o', markersize=2)
    ax.plot(traj[-1, :], traj[1, :], '-', linewidth=1.5, label='x[1]', marker='s', markersize=2)
    ax.plot(traj[-1, :], traj[2, :], '-', linewidth=1.5, label='x[2]', marker='^', markersize=2)
    ax.set_xlabel(r'Bifurcation parameter $\alpha$', fontsize=11)
    ax.set_ylabel('State components', fontsize=11)
    ax.set_title('Solution branches', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('arc_length_bifurcation.png', dpi=150, bbox_inches='tight')
    print("Figure saved: arc_length_bifurcation.png")
    plt.show()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Arc-length Continuation for Bratu Problem")
    print("=" * 70)
    print()
    
    traj, norm_of_error, newton_steps = arclength_continuation()
    
    print()
    print(f"Completed {traj.shape[1]} steps")
    print(f"Alpha range: [{traj[-1, 0]:.6f}, {traj[-1, -1]:.6f}]")
    print(f"Final error: {norm_of_error[-1]:.2e}")
    print()
    
    visualize_output(traj, norm_of_error, newton_steps)
