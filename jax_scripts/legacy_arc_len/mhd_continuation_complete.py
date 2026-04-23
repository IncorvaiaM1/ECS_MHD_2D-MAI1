"""
COMPREHENSIVE ARC-LENGTH CONTINUATION FOR MHD BIFURCATION ANALYSIS
===================================================================

This package provides a complete arc-length continuation solver using JAX,
specifically designed for MHD systems. It includes:
1. Core continuation algorithm
2. Example bifurcation problems  
3. MHD-specific templates
4. Spider-web bifurcation diagram generators

Author: Based on MATLAB script, adapted for JAX
Date: January 2026
"""

import jax
import jax.numpy as jnp
from jax import jacfwd, jit, grad
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Callable, Tuple, Dict, List
import warnings


class ArcLengthContinuation:
    """
    Arc-length continuation solver with JAX autodiff.
    
    This class implements the pseudo arc-length continuation method for
    tracing solution branches through bifurcations. It uses JAX's automatic
    differentiation to compute Jacobians efficiently.
    """
    
    def __init__(self, F: Callable, x0: jnp.ndarray, alpha0: float, **kwargs):
        """
        Initialize continuation solver.
        
        Parameters:
        -----------
        F : callable
            Residual function F(x, alpha) that should equal zero
        x0 : jnp.ndarray
            Initial solution vector
        alpha0 : float
            Initial bifurcation parameter value
        **kwargs : dict
            Optional parameters:
            - ds: float (default 0.01) - Arc-length step size
            - n_steps: int (default 1000) - Number of continuation steps
            - max_newton_iter: int (default 50) - Max Newton iterations
            - tolerance: float (default 1e-10) - Convergence tolerance
            - verbose: bool (default True) - Print progress
        """
        self.F = F
        self.x0 = jnp.array(x0, dtype=jnp.float64)
        self.alpha0 = float(alpha0)
        
        # Parameters
        self.ds = kwargs.get('ds', 0.01)
        self.n_steps = kwargs.get('n_steps', 1000)
        self.max_newton_iter = kwargs.get('max_newton_iter', 50)
        self.tolerance = kwargs.get('tolerance', 1e-10)
        self.verbose = kwargs.get('verbose', True)
        
        # Results storage
        self.trajectory = None
        self.newton_iters = None
        self.residuals = None
        self.success = False
    
    def _continuation_system(self, z: jnp.ndarray, z_prev: jnp.ndarray) -> jnp.ndarray:
        """
        Augmented continuation system combining F=0 with arc-length constraint.
        
        System: [F(x, alpha); ||z - z_prev|| - |ds|] = 0
        """
        x = z[:-1]
        alpha = z[-1]
        
        # Original system
        f_orig = self.F(x, alpha)
        
        # Arc-length constraint
        arc_length = jnp.linalg.norm(z - z_prev) - jnp.abs(self.ds)
        
        return jnp.concatenate([f_orig, jnp.array([arc_length])])
    
    def _newton_solve(self, z_init: jnp.ndarray, z_prev: jnp.ndarray) -> Tuple:
        """
        Solve continuation system using Newton's method with JAX autodiff.
        
        Returns:
        --------
        z_final : solution
        n_iters : number of iterations
        final_res : final residual norm
        success : convergence flag
        """
        z = z_init
        
        for iter in range(self.max_newton_iter):
            # Compute residual
            res = self._continuation_system(z, z_prev)
            res_norm = float(jnp.linalg.norm(res))
            
            # Check convergence
            if res_norm < self.tolerance:
                return z, iter, res_norm, True
            
            # Compute Jacobian via automatic differentiation
            try:
                J = jacfwd(lambda w: self._continuation_system(w, z_prev))(z)
                
                # Newton update: solve J·Δz = -res
                delta_z = jnp.linalg.solve(J, -res)
                z = z + delta_z
                
            except Exception as e:
                return z, iter, res_norm, False
        
        # Check final residual
        res = self._continuation_system(z, z_prev)
        res_norm = float(jnp.linalg.norm(res))
        
        return z, self.max_newton_iter, res_norm, (res_norm < self.tolerance)
    
    def run(self) -> Dict:
        """
        Execute the continuation algorithm.
        
        Returns:
        --------
        results : dict
            Dictionary containing:
            - trajectory: full solution path [x; alpha]
            - newton_iters: iterations per step
            - residuals: residual norms
            - alpha: parameter values
            - states: state vectors
            - success: whether continuation completed
        """
        # Validate initial condition
        f0 = self.F(self.x0, self.alpha0)
        init_res = float(jnp.max(jnp.abs(f0)))
        
        if init_res > 1e-4:
            warnings.warn(f"Initial condition may not be a solution (residual={init_res:.2e})")
        
        # Initialize storage
        n_vars = len(self.x0) + 1
        traj = np.zeros((n_vars, self.n_steps), dtype=np.float64)
        iters = np.zeros(self.n_steps, dtype=np.int32)
        resids = np.zeros(self.n_steps, dtype=np.float64)
        
        # Initial point
        z0 = jnp.concatenate([self.x0, jnp.array([self.alpha0])])
        traj[:, 0] = np.array(z0)
        resids[0] = init_res
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Arc-Length Continuation")
            print(f"{'='*70}")
            print(f"Initial α = {self.alpha0:.6f}")
            print(f"Step size ds = {self.ds:.6f}")
            print(f"Steps = {self.n_steps}")
            print(f"{'-'*70}")
        
        # Main continuation loop
        for i in range(1, self.n_steps):
            # Generate initial guess
            if i == 1:
                # First step: perturb parameter
                z_guess = z0.at[-1].add(self.ds)
                z_prev = z0
            else:
                # Subsequent steps: linear extrapolation
                z_prev = jnp.array(traj[:, i-1])
                z_prev2 = jnp.array(traj[:, i-2])
                z_guess = 2.0 * z_prev - z_prev2
            
            # Solve via Newton
            z_new, n_iter, res_norm, success = self._newton_solve(z_guess, z_prev)
            
            if not success:
                if self.verbose:
                    print(f"\nConvergence failure at step {i} (residual={res_norm:.2e})")
                    print(f"Stopping continuation.")
                # Truncate arrays
                traj = traj[:, :i]
                iters = iters[:i]
                resids = resids[:i]
                break
            
            # Store results
            traj[:, i] = np.array(z_new)
            iters[i] = n_iter
            resids[i] = res_norm
            
            # Progress output
            if self.verbose and (i % max(1, self.n_steps // 10) == 0):
                print(f"Step {i:4d}: α={float(z_new[-1]):9.5f}, "
                      f"Newton={n_iter:2d}, res={res_norm:.1e}")
        
        if self.verbose:
            final_alpha = traj[-1, -1]
            alpha_range = [traj[-1, 0], final_alpha]
            avg_iters = np.mean(iters[iters > 0])
            
            print(f"{'-'*70}")
            print(f"Continuation complete!")
            print(f"Parameter range: α ∈ [{alpha_range[0]:.5f}, {alpha_range[1]:.5f}]")
            print(f"Average Newton iterations: {avg_iters:.2f}")
            print(f"{'='*70}\n")
        
        # Store and return
        self.trajectory = traj
        self.newton_iters = iters
        self.residuals = resids
        self.success = (i == self.n_steps - 1)
        
        return {
            'trajectory': traj,
            'newton_iters': iters,
            'residuals': resids,
            'alpha': traj[-1, :],
            'states': traj[:-1, :],
            'success': self.success
        }
    
    def plot(self, observable_fn: Callable = None, filename: str = None):
        """
        Create comprehensive visualization of continuation results.
        
        Parameters:
        -----------
        observable_fn : callable, optional
            Function to compute observable(state). Default is L2 norm.
        filename : str, optional
            Save figure to this file
        """
        if self.trajectory is None:
            raise RuntimeError("Must run continuation before plotting!")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        alpha = self.trajectory[-1, :]
        states = self.trajectory[:-1, :]
        
        # Compute observable
        if observable_fn is None:
            obs = np.linalg.norm(states, axis=0)
            obs_label = "||x||₂"
        else:
            obs = np.array([observable_fn(states[:, i]) for i in range(states.shape[1])])
            obs_label = "Observable"
        
        # 1. Bifurcation diagram
        ax = axes[0, 0]
        ax.plot(alpha, obs, 'b-', linewidth=2.5, alpha=0.8)
        ax.scatter(alpha[0], obs[0], c='green', s=150, marker='o', 
                   edgecolors='black', linewidths=2, zorder=5, label='Start')
        ax.scatter(alpha[-1], obs[-1], c='red', s=150, marker='s',
                   edgecolors='black', linewidths=2, zorder=5, label='End')
        ax.set_xlabel('Parameter α', fontsize=13, fontweight='bold')
        ax.set_ylabel(obs_label, fontsize=13, fontweight='bold')
        ax.set_title('Bifurcation Diagram', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 2. Convergence history
        ax = axes[0, 1]
        valid_res = self.residuals[self.residuals > 0]
        if len(valid_res) > 0:
            ax.semilogy(self.residuals, 'o-', markersize=5, linewidth=1, alpha=0.7)
            ax.axhline(self.tolerance, color='r', linestyle='--', linewidth=2,
                       label=f'Tolerance={self.tolerance:.0e}')
            ax.set_xlabel('Continuation step', fontsize=13)
            ax.set_ylabel('Residual norm', fontsize=13)
            ax.set_title('Convergence Quality', fontsize=15, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # 3. Newton iterations
        ax = axes[1, 0]
        ax.plot(self.newton_iters, 'o-', markersize=5, linewidth=1, 
                color='purple', alpha=0.7)
        ax.set_xlabel('Continuation step', fontsize=13)
        ax.set_ylabel('Newton iterations', fontsize=13)
        ax.set_title('Newton Performance', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 4. State space trajectory
        ax = axes[1, 1]
        n_dim = states.shape[0]
        
        if n_dim == 1:
            ax.plot(alpha, states[0, :], 'g-', linewidth=2.5, alpha=0.8)
            ax.set_xlabel('α', fontsize=13)
            ax.set_ylabel('x', fontsize=13)
            ax.set_title('State vs Parameter', fontsize=15, fontweight='bold')
        elif n_dim == 2:
            ax.plot(states[0, :], states[1, :], 'g-', linewidth=2.5, alpha=0.8)
            ax.scatter(states[0, 0], states[1, 0], c='green', s=150, marker='o',
                       edgecolors='black', linewidths=2, zorder=5)
            ax.scatter(states[0, -1], states[1, -1], c='red', s=150, marker='s',
                       edgecolors='black', linewidths=2, zorder=5)
            ax.set_xlabel('x₁', fontsize=13)
            ax.set_ylabel('x₂', fontsize=13)
            ax.set_title('Phase Space', fontsize=15, fontweight='bold')
        elif n_dim == 3:
            colors = plt.cm.viridis(np.linspace(0, 1, 3))
            for i in range(3):
                ax.plot(alpha, states[i, :], label=f'x_{i}', 
                        linewidth=2, color=colors[i])
            ax.set_xlabel('α', fontsize=13)
            ax.set_ylabel('State components', fontsize=13)
            ax.set_title('State Components', fontsize=15, fontweight='bold')
            ax.legend(fontsize=11)
        else:
            im = ax.imshow(states, aspect='auto', cmap='RdBu_r',
                          extent=[alpha[0], alpha[-1], n_dim-1, 0])
            plt.colorbar(im, ax=ax, label='State value')
            ax.set_xlabel('α', fontsize=13)
            ax.set_ylabel('State index', fontsize=13)
            ax.set_title('State Evolution Heatmap', fontsize=15, fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            print(f"Figure saved: {filename}")
        
        return fig


# =============================================================================
# EXAMPLE PROBLEMS
# =============================================================================

def bratu_1d(x: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """
    1D Bratu problem with 3-point finite difference.
    Classic bifurcation problem: -u'' = α·exp(u) with u(0)=u(1)=0
    """
    return jnp.array([
        -2*x[0] + x[1] + alpha * jnp.exp(x[0]),
        x[0] - 2*x[1] + x[2] + alpha * jnp.exp(x[1]),
        x[1] - 2*x[2] + alpha * jnp.exp(x[2])
    ])


def pitchfork(x: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """Supercritical pitchfork: α·x - x³ = 0"""
    return jnp.array([alpha * x[0] - x[0]**3])


def hopf_steady(x: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """
    Steady states of Hopf normal form.
    Dynamical system: dr/dt = α·r - r³, dθ/dt = ω
    In Cartesian: dx/dt = α·x - ω·y - x·(x²+y²), dy/dt = ω·x + α·y - y·(x²+y²)
    """
    omega = 2.0
    r_sq = x[0]**2 + x[1]**2
    return jnp.array([
        alpha*x[0] - omega*x[1] - x[0]*r_sq,
        omega*x[0] + alpha*x[1] - x[1]*r_sq
    ])


# =============================================================================
# MHD-SPECIFIC UTILITIES
# =============================================================================

def create_mhd_spider_web(
    F_list: List[Callable],
    x0_list: List[jnp.ndarray],
    alpha0_list: List[float],
    labels: List[str],
    **kwargs
) -> plt.Figure:
    """
    Create spider-web bifurcation diagram from multiple continuation branches.
    
    This is characteristic of MHD systems where multiple solution branches
    can emanate from bifurcation points, creating a web-like structure.
    
    Parameters:
    -----------
    F_list : list of callables
        List of residual functions for each branch
    x0_list : list of arrays
        Initial conditions for each branch
    alpha0_list : list of floats
        Initial parameters for each branch
    labels : list of str
        Labels for each branch
    **kwargs : dict
        Continuation parameters (ds, n_steps, etc.)
    
    Returns:
    --------
    fig : matplotlib figure
        Spider-web bifurcation diagram
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(F_list)))
    
    for i, (F, x0, alpha0, label) in enumerate(zip(F_list, x0_list, alpha0_list, labels)):
        try:
            solver = ArcLengthContinuation(F, x0, alpha0, verbose=False, **kwargs)
            results = solver.run()
            
            alpha = results['alpha']
            states = results['states']
            energy = np.linalg.norm(states, axis=0)
            
            # Plot branch
            ax.plot(alpha, energy, '-', color=colors[i], linewidth=2.5,
                    alpha=0.8, label=label)
            
            # Mark start and end
            ax.scatter(alpha[0], energy[0], c=[colors[i]], s=120, marker='o',
                       edgecolors='black', linewidths=1.5, zorder=5)
            ax.scatter(alpha[-1], energy[-1], c=[colors[i]], s=120, marker='s',
                       edgecolors='black', linewidths=1.5, zorder=5)
            
        except Exception as e:
            print(f"Branch '{label}' failed: {e}")
            continue
    
    ax.set_xlabel('Control Parameter', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Energy', fontsize=14, fontweight='bold')
    ax.set_title('MHD Bifurcation Spider Web', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ARC-LENGTH CONTINUATION DEMONSTRATION")
    print("="*70)
    
    # Example 1: Bratu problem
    print("\n1. Bratu Problem (1D finite difference)")
    print("-"*70)
    
    solver_bratu = ArcLengthContinuation(
        F=bratu_1d,
        x0=jnp.zeros(3),
        alpha0=0.0,
        ds=0.01,
        n_steps=400,
        tolerance=1e-10
    )
    
    res_bratu = solver_bratu.run()
    
    # fig_bratu = solver_bratu.plot(filename='/home/claude/bratu_demo.png')
    
    # Example 2: Pitchfork
    print("\n2. Pitchfork Bifurcation")
    print("-"*70)
    
    solver_pitch = ArcLengthContinuation(
        F=pitchfork,
        x0=jnp.array([1e-6]),  # Small perturbation
        alpha0=-0.2,
        ds=0.02,
        n_steps=150,
        tolerance=1e-10
    )
    
    res_pitch = solver_pitch.run()
    
    # fig_pitch = solver_pitch.plot(filename='/home/claude/pitchfork_demo.png')
    
    # Example 3: Spider web (multi-branch)
    print("\n3. Spider Web Bifurcation Diagram")
    print("-"*70)
    
    # Create multiple branches with different initial conditions
    branches_F = [bratu_1d] * 4
    branches_x0 = [
        jnp.zeros(3),
        jnp.array([0.01, 0.01, 0.01]),
        jnp.array([-0.01, 0.0, 0.01]),
        jnp.array([0.01, -0.01, 0.01])
    ]
    branches_alpha0 = [0.0, 0.02, 0.01, 0.03]
    branches_labels = ['Branch 1', 'Branch 2', 'Branch 3', 'Branch 4']
    
    fig_spider = create_mhd_spider_web(
        branches_F, branches_x0, branches_alpha0, branches_labels,
        ds=0.008, n_steps=300, tolerance=1e-10
    )
    
    # plt.savefig('/home/claude/spider_web_demo.png', dpi=200, bbox_inches='tight')
    print("Spider web diagram saved: spider_web_demo.png")
    
    plt.show()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nFiles created:")
    print("  - bratu_demo.png")
    print("  - pitchfork_demo.png")
    print("  - spider_web_demo.png")
    print("\nFor MHD integration, wrap your solver as:")
    print("  F_mhd = lambda state, Re: your_mhd_residual(state, Re, ...)")
    print("  solver = ArcLengthContinuation(F_mhd, x0, Re0, ...)")
    print("="*70 + "\n")