"""
Arc-length continuation for bifurcation analysis of MHD equilibria.

This module implements arc-length continuation using Newton's method
to trace solution branches of the MHD equations as a function of
a bifurcation parameter (Re, viscosity, magnetic field strength, etc).

Follows the standards and conventions of the ECS_MHD_2D repository.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jacobian
import matplotlib.pyplot as plt
from functools import partial
from typing import Callable, Tuple, Dict
import scipy.sparse.linalg as spla

import lib.mhd_jax as mhd_jax
import lib.dictionaryIO as dictionaryIO
import lib.timestepping as timestepping


class ArclengthContinuation:
    """
    Arc-length continuation solver for bifurcation problems.
    
    Traces solution curves of F(z, alpha) = 0 where z = [x; alpha] 
    is the extended state, by solving:
      F(x, alpha) = 0
      ||z - z_prev|| - ds = 0
    """
    
    def __init__(
        self,
        f: Callable,
        ds: float = 0.01,
        max_steps: int = 512,
        max_newton_iter: int = 64,
        newton_threshold: float = 1e-10,
        verbose: bool = False
    ):
        """
        Initialize continuation solver.
        
        Args:
            f: Residual function f(x, alpha) where alpha is bifurcation parameter
            ds: Arc-length step size
            max_steps: Maximum number of continuation steps
            max_newton_iter: Max Newton iterations per step
            newton_threshold: Convergence threshold for Newton method
            verbose: Print progress
        """
        self.f = f
        self.ds = ds
        self.max_steps = max_steps
        self.max_newton_iter = max_newton_iter
        self.newton_threshold = newton_threshold
        self.verbose = verbose
        
        # JAX autodiff for Jacobian
        self.jac_f = jacobian(f)
    
    def continuation_equation(
        self, 
        z: jnp.ndarray, 
        z_prev: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Define continuation equation.
        
        z = [x; alpha] extended state
        Returns [F(x, alpha); ||z - z_prev|| - ds]
        """
        x = z[:-1]
        alpha = z[-1]
        
        # Original problem
        f_residual = self.f(x, alpha)
        
        # Arc-length constraint
        arc_constraint = jnp.linalg.norm(z - z_prev) - self.ds
        
        return jnp.concatenate([f_residual, jnp.array([arc_constraint])])
    
    def continuation_equation_jac(
        self,
        z: jnp.ndarray,
        z_prev: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute Jacobian of continuation equation using autodiff."""
        def g_func(z_var):
            return self.continuation_equation(z_var, z_prev)
        
        return jacobian(g_func)(z)
    
    def newton_step(
        self,
        z: jnp.ndarray,
        z_prev: jnp.ndarray
    ) -> Tuple[jnp.ndarray, float, bool]:
        """
        Single Newton step on continuation equation.
        
        Returns:
            Updated z, error norm, success flag
        """
        g = self.continuation_equation(z, z_prev)
        error = np.linalg.norm(g)
        
        if error < self.newton_threshold:
            return z, error, True
        
        # Compute Jacobian
        J = np.array(self.continuation_equation_jac(z, z_prev))
        
        # Solve for Newton step
        try:
            dz = np.linalg.solve(J, -np.array(g))
            z_new = z + dz
            return z_new, error, True
        except np.linalg.LinAlgError:
            return z, error, False
    
    def solve(
        self,
        x_init: jnp.ndarray,
        alpha_init: float,
        direction: str = "forward"
    ) -> Dict:
        """
        Run arc-length continuation.
        
        Args:
            x_init: Initial state
            alpha_init: Initial bifurcation parameter value
            direction: "forward" or "backward"
            
        Returns:
            Dictionary with keys:
                - 'trajectory': Shape (n_dim+1, n_steps)
                - 'errors': Residual norm at each step
                - 'newton_iters': Newton iterations per step
                - 'alpha': Bifurcation parameter values
        """
        # Storage
        trajectory = np.zeros((len(x_init) + 1, self.max_steps))
        errors = np.zeros(self.max_steps)
        newton_iters = np.zeros(self.max_steps, dtype=int)
        
        # Initial condition
        z = jnp.concatenate([x_init, jnp.array([alpha_init])])
        trajectory[:, 0] = np.array(z)
        errors[0] = np.linalg.norm(self.f(x_init, alpha_init))
        
        # Direction along curve
        direction_mult = 1.0 if direction == "forward" else -1.0
        
        z_prev = np.array(z)
        z_next = np.array(z)
        z_next[-1] += direction_mult * self.ds
        
        converged = True
        actual_steps = self.max_steps
        
        for step in range(1, self.max_steps):
            # Newton solve
            for iteration in range(self.max_newton_iter):
                z_next, error, success = self.newton_step(
                    jnp.array(z_next), 
                    jnp.array(z_prev)
                )
                
                if error < self.newton_threshold:
                    newton_iters[step] = iteration + 1
                    break
            else:
                # Newton didn't converge
                if self.verbose:
                    print(f"Newton did not converge at step {step}. Error: {error:.2e}")
                converged = False
                actual_steps = step
                break
            
            if error > 1e-8:
                if self.verbose:
                    print(f"Stopping at step {step}: error too large ({error:.2e})")
                converged = False
                actual_steps = step
                break
            
            # Store
            trajectory[:, step] = np.array(z_next)
            errors[step] = error
            
            if self.verbose and step % 50 == 0:
                print(f"Step {step}: alpha={z_next[-1]:.6f}, error={error:.2e}")
            
            # Prepare next step
            z_prev = np.array(z_next)
            # Linear extrapolation
            z_next = 2 * np.array(z_next) - z_prev + np.array(z)
            z = z_prev
        
        return {
            'trajectory': trajectory[:, :actual_steps],
            'errors': errors[:actual_steps],
            'newton_iters': newton_iters[:actual_steps],
            'alpha': trajectory[-1, :actual_steps],
            'converged': converged
        }
    
    def visualize_results(
        self, 
        results: Dict,
        save_path: str = None,
        state_index: int = 0
    ):
        """
        Visualize continuation results.
        
        Args:
            results: Output from solve()
            save_path: If provided, save figure
            state_index: Which state component to highlight
        """
        traj = results['trajectory']
        alpha = results['alpha']
        errors = results['errors']
        newton_iters = results['newton_iters']
        
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        
        # Bifurcation diagram
        ax = axes[0, 0]
        state_norm = np.linalg.norm(traj[:-1, :], axis=0)
        ax.plot(alpha, state_norm, 'b-', linewidth=1.5, marker='o', markersize=3)
        ax.set_xlabel(r'Bifurcation parameter $\alpha$', fontsize=11)
        ax.set_ylabel(r'$\|x\|_2$', fontsize=11)
        ax.set_title('Bifurcation Diagram', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Residual convergence
        ax = axes[0, 1]
        ax.semilogy(np.maximum(np.abs(errors), 1e-16), 'o-', markersize=4, linewidth=1)
        ax.set_xlabel('Continuation step', fontsize=11)
        ax.set_ylabel('Residual norm', fontsize=11)
        ax.set_title('Continuation Equation Error', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        
        # Newton iterations
        ax = axes[1, 0]
        ax.plot(newton_iters, 'o-', markersize=5, linewidth=1, color='darkgreen')
        ax.set_xlabel('Continuation step', fontsize=11)
        ax.set_ylabel('Newton iterations', fontsize=11)
        ax.set_title('Newton Convergence Rate', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # State components
        ax = axes[1, 1]
        n_components = min(3, traj.shape[0] - 1)
        for i in range(n_components):
            ax.plot(alpha, traj[i, :], '-', linewidth=1.5, label=f'x[{i}]', marker='o', markersize=2)
        ax.set_xlabel(r'Bifurcation parameter $\alpha$', fontsize=11)
        ax.set_ylabel('State components', fontsize=11)
        ax.set_title('Solution Components', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {save_path}")
        
        plt.show()


# ============================================================================
# Test Problems
# ============================================================================

def bratu_3point(x: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """
    3-point Bratu problem. Classical fold bifurcation example.
    
    -u''(x) = alpha * exp(u(x)), u(0)=u(1)=0
    Discretized at 3 points with uniform spacing.
    """
    y = jnp.array([
        -2*x[0] + x[1] + alpha*jnp.exp(x[0]),
        x[0] - 2*x[1] + x[2] + alpha*jnp.exp(x[1]),
        x[1] - 2*x[2] + alpha*jnp.exp(x[2])
    ])
    return y


def pitchfork_bifurcation(x: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """
    Simple pitchfork bifurcation: alpha*x - x^3 = 0
    Symmetric branches emerge at alpha=0.
    """
    return jnp.array([alpha*x[0] - x[0]**3])


# ============================================================================
# Continuation with MHD
# ============================================================================

def create_mhd_steady_state_problem(
    n: int = 64,
    precision = jnp.float64,
    param_type: str = "reynolds"
) -> Tuple[Callable, Dict]:
    """
    Create a residual function for MHD steady states.
    
    Args:
        n: Grid resolution
        precision: float32 or float64
        param_type: "reynolds", "nu", "forcing_amplitude"
        
    Returns:
        Tuple of (residual_function, param_dict)
    """
    if precision == jnp.float64:
        jax.config.update("jax_enable_x64", True)
    
    # Set up domain and default parameters
    param_dict = mhd_jax.construct_domain(n, precision)
    
    # Physical parameters
    nu_nominal = 1.0 / 40.0  # Viscosity
    eta_nominal = 1.0 / 40.0  # Magnetic diffusivity
    b0_nominal = jnp.array([0.0, 0.0])
    
    # Forcing
    x = param_dict['x']
    y = param_dict['y']
    forcing = -4*jnp.cos(4*y)
    
    # Update param_dict with physical parameters
    param_dict.update({
        'nu': nu_nominal,
        'eta': eta_nominal,
        'b0': b0_nominal,
        'forcing': forcing,
        'forcing_str': "lambda x,y : -4*jnp.cos(4*y)"
    })
    
    def residual_steady_state(f_vec: jnp.ndarray, param_value: float) -> jnp.ndarray:
        """
        Compute residual for MHD steady state.
        
        For equilibrium: df/dt = 0
        This is a simplified residual. For full implementation,
        would compute actual MHD time derivative.
        """
        # Reshape state
        f = f_vec.reshape(2, n, n)
        
        # Update parameter
        if param_type == "reynolds":
            param_dict['nu'] = 1.0 / param_value
            param_dict['eta'] = 1.0 / param_value
        elif param_type == "nu":
            param_dict['nu'] = param_value
        elif param_type == "forcing_amplitude":
            param_dict['forcing'] = param_value * (-4*jnp.cos(4*y))
        
        # For demonstration, use simple quadratic potential
        # In practice, compute: df/dt = MHD_time_derivative(f, param_dict)
        residual = f_vec**3 - 0.5 * f_vec * param_value
        
        return residual
    
    return residual_steady_state, param_dict


def run_bratu_continuation():
    """Run continuation on 3-point Bratu problem."""
    print("=" * 70)
    print("Arc-length Continuation: 3-point Bratu Problem")
    print("=" * 70)
    
    jax.config.update("jax_enable_x64", True)
    
    # Initial condition
    x_init = jnp.array([0.0, 0.0, 0.0])
    alpha_init = 0.0
    
    # Create solver
    solver = ArclengthContinuation(
        f=bratu_3point,
        ds=0.01,
        max_steps=512,
        max_newton_iter=50,
        newton_threshold=1e-11,
        verbose=True
    )
    
    # Solve
    print("\nContinuing forward...")
    results_fwd = solver.solve(x_init, alpha_init, direction="forward")
    
    # Visualize
    print("\nGenerating visualization...")
    solver.visualize_results(results_fwd, save_path="bratu_bifurcation_diagram.png")
    
    return results_fwd


def run_pitchfork_continuation():
    """Run continuation on pitchfork bifurcation."""
    print("=" * 70)
    print("Arc-length Continuation: Pitchfork Bifurcation")
    print("=" * 70)
    
    jax.config.update("jax_enable_x64", True)
    
    # Start on upper branch
    x_init = jnp.array([1.5])
    alpha_init = 2.0
    
    # Create solver
    solver = ArclengthContinuation(
        f=pitchfork_bifurcation,
        ds=0.05,
        max_steps=256,
        max_newton_iter=50,
        newton_threshold=1e-11,
        verbose=False
    )
    
    # Continue forward
    print("\nContinuing forward...")
    results_fwd = solver.solve(x_init, alpha_init, direction="forward")
    
    # Continue backward
    print("Continuing backward...")
    results_bwd = solver.solve(x_init, alpha_init, direction="backward")
    
    # Combine trajectories
    traj_bwd_rev = np.fliplr(results_bwd['trajectory'])
    traj_combined = np.hstack([traj_bwd_rev, results_fwd['trajectory']])
    alpha_combined = np.hstack([np.flip(results_bwd['alpha']), results_fwd['alpha']])
    
    results_combined = {
        'trajectory': traj_combined,
        'alpha': alpha_combined,
        'errors': np.hstack([np.flip(results_bwd['errors']), results_fwd['errors']]),
        'newton_iters': np.zeros_like(alpha_combined, dtype=int),
        'converged': results_fwd['converged'] and results_bwd['converged']
    }
    
    # Visualize
    print("\nGenerating visualization...")
    solver.visualize_results(results_combined, save_path="pitchfork_bifurcation_diagram.png")
    
    return results_combined


def export_continuation_results(
    results: Dict,
    filename: str,
    param_dict: Dict = None
):
    """
    Export continuation results to NPZ file.
    
    Args:
        results: Output from continuation solver
        filename: Output filename
        param_dict: Optional parameter dictionary to save
    """
    save_dict = {
        'trajectory': results['trajectory'],
        'alpha': results['alpha'],
        'errors': results['errors'],
        'newton_iters': results['newton_iters'],
        'converged': np.array([results['converged']])
    }
    
    if param_dict is not None:
        # Can't directly save nested dicts, so just save key parameters
        pass
    
    np.savez(filename, **save_dict)
    print(f"Results exported to {filename}")


if __name__ == "__main__":
    print("\n")
    bratu_results = run_bratu_continuation()
    
    print("\n" * 2)
    pitchfork_results = run_pitchfork_continuation()
    
    # Export results
    print("\n" * 2)
    export_continuation_results(bratu_results, "bratu_continuation_results.npz")
