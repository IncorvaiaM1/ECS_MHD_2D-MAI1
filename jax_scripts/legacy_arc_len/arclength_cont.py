"""
MHD Bifurcation Analysis with Arc-Length Continuation
======================================================

This script adapts the arc-length continuation method for 2D MHD problems,
showing how to create spider-web-like bifurcation diagrams for MHD systems.

Author: Adapted for MHD/ECS
Date: 2026
"""

import jax
import jax.numpy as jnp
from jax import jacfwd, jit
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from typing import Callable, Tuple, Dict
from jax_scripts.arc_len_stuff.mhd_continuation_complete import ArcLengthContinuation


class MHDBifurcationAnalyzer:
    """
    Specialized bifurcation analyzer for MHD systems.
    
    This class extends the basic arc-length continuation to handle
    MHD-specific observables and visualizations.
    """
    
    def __init__(self, mhd_solver_func: Callable):
        """
        Initialize MHD bifurcation analyzer.
        
        Parameters:
        -----------
        mhd_solver_func : callable
            Function that takes (state, Reynolds_number) and returns residual
        """
        self.mhd_solver = mhd_solver_func
        self.results_cache = []
        
    def compute_mhd_observables(self, state: jnp.ndarray) -> Dict:
        """
        Compute various MHD observables from state.
        
        Parameters:
        -----------
        state : jnp.ndarray
            MHD state vector (vorticity, stream function, magnetic field, etc.)
            
        Returns:
        --------
        observables : dict
            Dictionary of computed observables
        """
        # Assuming state contains flattened fields
        n = int(np.sqrt(len(state) / 3))  # Assuming 3 fields (omega, psi, A)
        
        # Reshape to 2D fields
        omega = state[:n*n].reshape(n, n)
        psi = state[n*n:2*n*n].reshape(n, n)
        A = state[2*n*n:].reshape(n, n)
        
        observables = {
            'kinetic_energy': jnp.sum(omega**2),
            'magnetic_energy': jnp.sum(A**2),
            'total_energy': jnp.sum(omega**2) + jnp.sum(A**2),
            'enstrophy': jnp.sum(omega**4),
            'max_vorticity': jnp.max(jnp.abs(omega)),
            'max_current': jnp.max(jnp.abs(A)),
            'helicity_proxy': jnp.sum(omega * A),
        }
        
        return observables
    
    def run_multi_parameter_continuation(
        self, 
        x0: jnp.ndarray,
        param_ranges: Dict[str, Tuple[float, float]],
        n_branches: int = 10
    ):
        """
        Run continuation along multiple parameter directions to create
        spider-web bifurcation diagrams.
        
        Parameters:
        -----------
        x0 : jnp.ndarray
            Initial state
        param_ranges : dict
            Dictionary of parameter names and their ranges
        n_branches : int
            Number of branches to explore
        """
        results = []
        
        for param_name, (alpha_start, alpha_end) in param_ranges.items():
            print(f"\nExploring parameter: {param_name}")
            print(f"Range: [{alpha_start}, {alpha_end}]")
            
            # Run continuation for this parameter
            solver = ArcLengthContinuation(
                F=lambda x, alpha: self.mhd_solver(x, **{param_name: alpha}),
                x0=x0,
                alpha0=alpha_start,
                ds=(alpha_end - alpha_start) / 1000,
                n_steps=1000,
                verbose=False
            )
            
            result = solver.run()
            results.append({
                'parameter': param_name,
                'solver': solver,
                'result': result
            })
        
        self.results_cache = results
        return results
    
    def plot_spider_web_bifurcation(self, figsize=(14, 10)):
        """
        Create spider-web style bifurcation diagram with multiple observables.
        
        This creates the characteristic "spider web" appearance of MHD
        bifurcation diagrams where multiple solution branches emanate
        from bifurcation points.
        """
        if not self.results_cache:
            raise ValueError("Must run continuation first!")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main bifurcation diagram (takes up 2x2 space)
        ax_main = fig.add_subplot(gs[:2, :2])
        
        # Observable plots
        ax_ke = fig.add_subplot(gs[0, 2])
        ax_me = fig.add_subplot(gs[1, 2])
        ax_helicity = fig.add_subplot(gs[2, 0])
        ax_enstrophy = fig.add_subplot(gs[2, 1])
        ax_phase = fig.add_subplot(gs[2, 2])
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results_cache)))
        
        for i, result_dict in enumerate(self.results_cache):
            result = result_dict['result']
            alpha = result['alpha']
            states = result['states']
            
            # Compute observables for each point
            observables = []
            for j in range(states.shape[1]):
                obs = self.compute_mhd_observables(states[:, j])
                observables.append(obs)
            
            # Extract observable arrays
            ke = np.array([o['kinetic_energy'] for o in observables])
            me = np.array([o['magnetic_energy'] for o in observables])
            total_e = ke + me
            helicity = np.array([o['helicity_proxy'] for o in observables])
            enstrophy = np.array([o['enstrophy'] for o in observables])
            
            label = result_dict['parameter']
            color = colors[i]
            
            # Main bifurcation diagram: Total energy vs parameter
            ax_main.plot(alpha, total_e, '-', color=color, linewidth=2, 
                        label=label, alpha=0.7)
            
            # Individual observable plots
            ax_ke.plot(alpha, ke, '-', color=color, linewidth=1.5, alpha=0.7)
            ax_me.plot(alpha, me, '-', color=color, linewidth=1.5, alpha=0.7)
            ax_helicity.plot(alpha, helicity, '-', color=color, linewidth=1.5, alpha=0.7)
            ax_enstrophy.plot(alpha, enstrophy, '-', color=color, linewidth=1.5, alpha=0.7)
            
            # Phase space plot: Kinetic vs Magnetic energy
            ax_phase.plot(ke, me, '-', color=color, linewidth=2, alpha=0.7)
            ax_phase.scatter(ke[0], me[0], c=color, s=100, marker='o', 
                           edgecolors='black', zorder=5)
            ax_phase.scatter(ke[-1], me[-1], c=color, s=100, marker='s', 
                           edgecolors='black', zorder=5)
        
        # Format main plot
        ax_main.set_xlabel('Control Parameter', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Total Energy', fontsize=12, fontweight='bold')
        ax_main.set_title('MHD Bifurcation Diagram (Spider Web)', 
                         fontsize=14, fontweight='bold')
        ax_main.legend(loc='best', framealpha=0.9)
        ax_main.grid(True, alpha=0.3)
        
        # Format observable plots
        ax_ke.set_title('Kinetic Energy', fontsize=10, fontweight='bold')
        ax_ke.set_xlabel('Parameter', fontsize=9)
        ax_ke.grid(True, alpha=0.3)
        
        ax_me.set_title('Magnetic Energy', fontsize=10, fontweight='bold')
        ax_me.set_xlabel('Parameter', fontsize=9)
        ax_me.grid(True, alpha=0.3)
        
        ax_helicity.set_title('Cross Helicity Proxy', fontsize=10, fontweight='bold')
        ax_helicity.set_xlabel('Parameter', fontsize=9)
        ax_helicity.grid(True, alpha=0.3)
        
        ax_enstrophy.set_title('Enstrophy', fontsize=10, fontweight='bold')
        ax_enstrophy.set_xlabel('Parameter', fontsize=9)
        ax_enstrophy.grid(True, alpha=0.3)
        
        ax_phase.set_title('Energy Phase Space', fontsize=10, fontweight='bold')
        ax_phase.set_xlabel('Kinetic Energy', fontsize=9)
        ax_phase.set_ylabel('Magnetic Energy', fontsize=9)
        ax_phase.grid(True, alpha=0.3)
        
        return fig


# ============================================================================
# Example: Simplified 2D MHD System (Toy Model)
# ============================================================================

def toy_mhd_system(state: jnp.ndarray, Re: float, Rm: float = 1.0) -> jnp.ndarray:
    """
    A simplified toy model of 2D MHD equations.
    
    This is a low-dimensional model for demonstration. For real MHD,
    you would interface with your full spectral solver.
    
    Parameters:
    -----------
    state : jnp.ndarray
        State vector [vorticity_mode, magnetic_mode, ...]
    Re : float
        Reynolds number (control parameter)
    Rm : float
        Magnetic Reynolds number
        
    Returns:
    --------
    residual : jnp.ndarray
        Residual of the equilibrium equations
    """
    # For demonstration: a 2-mode truncation of MHD
    # state = [omega_1, psi_1, A_1, omega_2, psi_2, A_2]
    
    if len(state) < 6:
        # Expand to 6 components if needed
        state = jnp.concatenate([state, jnp.zeros(6 - len(state))])
    
    omega_1, psi_1, A_1, omega_2, psi_2, A_2 = state[:6]
    
    # Simplified MHD equilibrium equations (Galerkin truncation)
    # These are steady-state versions of:
    # d(omega)/dt = -(v·∇)omega + (B·∇)J + (1/Re)∇²omega
    # d(A)/dt = -(v·∇)A + (1/Rm)∇²A
    # ∇²psi = -omega (Poisson equation)
    
    k1, k2 = 1.0, 2.0  # Wavenumbers
    
    # Poisson relation: k²psi = -omega
    f1 = k1**2 * psi_1 + omega_1
    f2 = k2**2 * psi_2 + omega_2
    
    # Vorticity equation (simplified nonlinear terms)
    nonlinear_omega_1 = omega_1 * psi_2 - omega_2 * psi_1
    lorentz_force_1 = A_1 * A_2
    f3 = nonlinear_omega_1 - lorentz_force_1 - (k1**2 / Re) * omega_1
    
    # Induction equation
    nonlinear_A_1 = psi_1 * A_2 - psi_2 * A_1
    f4 = nonlinear_A_1 - (k1**2 / Rm) * A_1
    
    # Higher mode equations (simplified)
    f5 = k2**2 * omega_2 / Re + omega_1 * omega_2
    f6 = k2**2 * A_2 / Rm + A_1 * A_2
    
    residual = jnp.array([f1, f2, f3, f4, f5, f6])
    
    return residual


# ============================================================================
# Template for Integration with Your MHD Code
# ============================================================================

def create_mhd_continuation_wrapper(your_mhd_solver):
    """
    Template function showing how to wrap your MHD solver for continuation.
    
    Your MHD solver should have a function that computes F(state, params) = 0
    where state is the full spectral/grid representation of the MHD fields.
    
    Example:
    --------
    def your_mhd_residual(state, Re, Rm, Pr, ...):
        # Compute residual of MHD equilibrium equations
        # This might involve:
        # 1. Reshaping state to 2D fields
        # 2. Computing derivatives (spectral or finite difference)
        # 3. Evaluating nonlinear terms
        # 4. Returning residual
        return residual
    
    # Then wrap it:
    mhd_wrapper = create_mhd_continuation_wrapper(your_mhd_residual)
    
    # And use with continuation:
    solver = ArcLengthContinuation(
        F=lambda state, Re: your_mhd_residual(state, Re, Rm=1.0, ...),
        x0=initial_equilibrium,
        alpha0=100.0,  # Starting Reynolds number
        ds=1.0,
        n_steps=1000
    )
    """
    
    def wrapped_solver(state: jnp.ndarray, control_param: float, **kwargs):
        """
        Wrapper that calls your MHD solver.
        
        Parameters:
        -----------
        state : jnp.ndarray
            MHD state vector
        control_param : float
            The bifurcation parameter (Re, Rm, Ha, etc.)
        **kwargs : dict
            Additional fixed parameters
            
        Returns:
        --------
        residual : jnp.ndarray
            Residual of equilibrium equations
        """
        return your_mhd_solver(state, control_param, **kwargs)
    
    return wrapped_solver


if __name__ == "__main__":
    print("=" * 70)
    print("MHD Bifurcation Analysis with Arc-Length Continuation")
    print("=" * 70)
    print()
    
    # Example 1: Simple toy MHD system
    print("Example: Toy 2D MHD System")
    print("-" * 70)
    
    # Initial equilibrium (trivial solution)
    x0_mhd = jnp.array([0.1, -0.1, 0.05, 0.02, -0.02, 0.01])
    Re0 = 10.0
    
    # Create solver
    solver_mhd = ArcLengthContinuation(
        F=lambda x, Re: toy_mhd_system(x, Re, Rm=10.0),
        x0=x0_mhd,
        alpha0=Re0,
        ds=0.5,
        n_steps=200,
        max_newton_iter=128,
        tolerance=1e-10,
        verbose=True
    )
    
    # Run continuation
    results_mhd = solver_mhd.run()
    
    # Create custom observable for MHD
    def mhd_observable(state):
        """Compute total energy."""
        omega = state[0::3]  # Every 3rd element starting from 0
        A = state[2::3]      # Every 3rd element starting from 2
        return np.sqrt(np.sum(omega**2) + np.sum(A**2))
    
    # Plot results
    fig_mhd = solver_mhd.plot_results(observable_fn=mhd_observable)
    plt.savefig('/home/claude/mhd_bifurcation_simple.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved MHD bifurcation diagram to mhd_bifurcation_simple.png")
    
    # Example 2: Spider web diagram (simulated multi-parameter)
    print("\n" + "=" * 70)
    print("Example: Multi-Parameter Spider Web Diagram")
    print("-" * 70)
    
    fig_spider = plt.figure(figsize=(14, 10))
    ax = fig_spider.add_subplot(111)
    
    # Simulate multiple continuation runs with different starting points
    colors = plt.cm.rainbow(np.linspace(0, 1, 8))
    
    for i, (ds_sign, color) in enumerate(zip([1, -1, 1, -1, 1, -1, 1, -1], colors)):
        # Vary initial conditions and step direction
        x0_vary = x0_mhd * (1 + 0.2 * i) + 0.01 * np.random.randn(6)
        Re0_vary = Re0 + 5.0 * i
        
        try:
            solver_branch = ArcLengthContinuation(
                F=lambda x, Re: toy_mhd_system(x, Re, Rm=10.0),
                x0=x0_vary,
                alpha0=Re0_vary,
                ds=ds_sign * 0.5,
                n_steps=150,
                max_newton_iter=64,
                tolerance=1e-8,
                verbose=False
            )
            
            result = solver_branch.run()
            
            # Plot branch
            alpha = result['alpha']
            states = result['states']
            observable = np.array([mhd_observable(states[:, j]) for j in range(states.shape[1])])
            
            ax.plot(alpha, observable, '-', color=color, linewidth=2, alpha=0.7)
            ax.scatter(alpha[0], observable[0], c=[color], s=100, marker='o', 
                      edgecolors='black', zorder=5, label=f'Branch {i+1}')
            
        except Exception as e:
            print(f"Branch {i+1} encountered issue: {e}")
            continue
    
    ax.set_xlabel('Reynolds Number (Re)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Energy', fontsize=13, fontweight='bold')
    ax.set_title('MHD Bifurcation Spider Web Diagram', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', ncol=2, framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/home/claude/mhd_spider_web.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved spider web diagram to mhd_spider_web.png")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("MHD bifurcation analysis complete!")
    print("=" * 70)
    print("\nTo integrate with your MHD solver:")
    print("1. Create a function that computes F(state, Re) = 0")
    print("2. Ensure it works with JAX arrays")
    print("3. Use ArcLengthContinuation class")
    print("4. Explore bifurcations by varying control parameters")
    print("=" * 70)