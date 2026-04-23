"""
INTEGRATION GUIDE: Arc-Length Continuation for ECS_MHD_2D Project
===================================================================

This guide shows how to integrate the arc-length continuation solver
with your 2D MHD spectral solver for bifurcation analysis and finding
Exact Coherent Structures (ECS).

Author: Integration Guide
Date: January 2026
"""

import jax.numpy as jnp
from mhd_continuation_complete import ArcLengthContinuation, create_mhd_spider_web
import matplotlib.pyplot as plt


# =============================================================================
# STEP 1: Wrapper for Your MHD Solver
# =============================================================================

def create_mhd_wrapper(your_solver_module):
    """
    Template for wrapping your MHD spectral solver.
    
    Your solver should have:
    - A function to compute the residual F(state, parameters) = 0
    - State representation (spectral coefficients or grid values)
    - Physical parameters (Re, Rm, Pr, Ha, etc.)
    
    Example structure of your_solver_module:
    ----------------------------------------
    
    class MHDSolver:
        def __init__(self, nx, ny, ...):
            # Initialize grid, operators, etc.
            pass
        
        def compute_residual(self, state, Re, Rm=1.0, ...):
            '''
            Compute F(state, Re) for equilibrium equations.
            
            Parameters:
            -----------
            state : jnp.ndarray
                Flattened state vector [omega_hat, psi_hat, A_hat, ...]
                where _hat denotes spectral coefficients
            Re : float
                Reynolds number (bifurcation parameter)
            Rm : float
                Magnetic Reynolds number
                
            Returns:
            --------
            residual : jnp.ndarray
                Residual of discretized MHD equations
            '''
            # Reshape state to spectral modes
            n_modes = len(state) // 3
            omega_hat = state[:n_modes]
            psi_hat = state[n_modes:2*n_modes]
            A_hat = state[2*n_modes:]
            
            # Compute residual of:
            # 1. Vorticity equation: (v·∇)ω - (B·∇)J - (1/Re)∇²ω = 0
            # 2. Poisson equation: ∇²ψ + ω = 0
            # 3. Induction equation: (v·∇)A - (1/Rm)∇²A = 0
            
            # [Your implementation here]
            
            return residual
    """
    
    def mhd_residual(state: jnp.ndarray, Re: float, **fixed_params) -> jnp.ndarray:
        """
        Wrapper function for continuation.
        
        This function should:
        1. Take state and bifurcation parameter (Re)
        2. Call your MHD solver's residual function
        3. Return the residual as a JAX array
        """
        # Example:
        # solver = fixed_params.get('solver')
        # Rm = fixed_params.get('Rm', 1.0)
        # residual = solver.compute_residual(state, Re, Rm)
        # return residual
        
        pass
    
    return mhd_residual


# =============================================================================
# STEP 2: Finding Initial Equilibria
# =============================================================================

def find_initial_equilibrium_example():
    """
    Before running continuation, you need an initial equilibrium.
    
    Methods:
    1. Trivial solution (often zero or simple base flow)
    2. Time-stepping until steady state
    3. Newton-GMRES on full nonlinear system
    4. Perturbation of known solution
    """
    
    # Example: Start from trivial solution
    # For MHD, this might be zero velocity and magnetic field
    nx, ny = 32, 32
    n_modes = nx * ny
    
    # Trivial equilibrium (usually works for small Re)
    x0_trivial = jnp.zeros(3 * n_modes)  # [omega, psi, A]
    
    # Or start from a known solution (e.g., from DNS)
    # x0_from_dns = load_dns_snapshot(...)
    
    return x0_trivial


# =============================================================================
# STEP 3: Running Continuation for Your MHD System
# =============================================================================

def mhd_continuation_example():
    """
    Example of running continuation for 2D MHD system.
    """
    
    # === Setup ===
    # Initialize your MHD solver
    # solver = MHDSolver(nx=32, ny=32, Lx=2*pi, Ly=2*pi)
    
    # Define residual function
    def F_mhd(state, Re):
        """
        MHD equilibrium residual.
        For your solver, this would call your Newton-GMRES residual function.
        """
        # Example placeholder
        # This should be replaced with your actual MHD residual
        Rm = 1.0  # Magnetic Reynolds number (fixed)
        # return solver.compute_residual(state, Re, Rm)
        
        # Dummy example for demonstration
        n = len(state) // 3
        omega, psi, A = state[:n], state[n:2*n], state[2*n:]
        
        # Simplified MHD residual (replace with your spectral code)
        r_omega = -omega/Re + 0.1*omega*psi  # Simplified vorticity
        r_psi = psi + omega  # Poisson equation
        r_A = -A/Rm + 0.1*A*psi  # Simplified induction
        
        return jnp.concatenate([r_omega, r_psi, r_A])
    
    # Initial equilibrium (at Re = 100)
    Re0 = 100.0
    n_modes = 10  # Small example
    x0 = jnp.zeros(3 * n_modes)  # Trivial solution
    
    # === Run Continuation ===
    print("\nRunning MHD bifurcation analysis...")
    print("="*70)
    
    solver = ArcLengthContinuation(
        F=F_mhd,
        x0=x0,
        alpha0=Re0,
        ds=5.0,  # Step in Reynolds number
        n_steps=100,
        max_newton_iter=50,
        tolerance=1e-8,
        verbose=True
    )
    
    results = solver.run()
    
    # === Analyze Results ===
    alpha_values = results['alpha']
    states = results['states']
    
    # Compute MHD observables
    kinetic_energy = []
    magnetic_energy = []
    
    for i in range(states.shape[1]):
        state = states[:, i]
        n = len(state) // 3
        omega = state[:n]
        A = state[2*n:]
        
        KE = float(jnp.sum(omega**2))
        ME = float(jnp.sum(A**2))
        
        kinetic_energy.append(KE)
        magnetic_energy.append(ME)
    
    # === Plot MHD-Specific Observables ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total energy
    axes[0, 0].plot(alpha_values, [k+m for k,m in zip(kinetic_energy, magnetic_energy)],
                    'b-', linewidth=2.5)
    axes[0, 0].set_xlabel('Reynolds Number', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Total Energy', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('MHD Bifurcation Diagram', fontsize=14, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # Kinetic vs Magnetic energy
    axes[0, 1].plot(kinetic_energy, magnetic_energy, 'g-', linewidth=2.5)
    axes[0, 1].set_xlabel('Kinetic Energy', fontsize=12)
    axes[0, 1].set_ylabel('Magnetic Energy', fontsize=12)
    axes[0, 1].set_title('Energy Phase Space', fontsize=14, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # Individual energies vs Re
    axes[1, 0].plot(alpha_values, kinetic_energy, label='Kinetic', linewidth=2)
    axes[1, 0].plot(alpha_values, magnetic_energy, label='Magnetic', linewidth=2)
    axes[1, 0].set_xlabel('Reynolds Number', fontsize=12)
    axes[1, 0].set_ylabel('Energy', fontsize=12)
    axes[1, 0].set_title('Energy Components', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Energy ratio
    ratio = [m/(k+1e-10) for k, m in zip(kinetic_energy, magnetic_energy)]
    axes[1, 1].plot(alpha_values, ratio, 'r-', linewidth=2.5)
    axes[1, 1].set_xlabel('Reynolds Number', fontsize=12)
    axes[1, 1].set_ylabel('Magnetic/Kinetic Energy Ratio', fontsize=12)
    axes[1, 1].set_title('Energy Partition', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/mhd_integration_example.png', dpi=200, bbox_inches='tight')
    print("\nSaved: mhd_integration_example.png")
    
    return results


# =============================================================================
# STEP 4: Multi-Parameter Continuation (Spider Web Diagrams)
# =============================================================================

def mhd_spider_web_example():
    """
    Create spider-web diagrams by exploring multiple parameter directions.
    
    In MHD, bifurcations often depend on multiple parameters:
    - Reynolds number (Re)
    - Magnetic Reynolds number (Rm)
    - Hartmann number (Ha)
    - Prandtl number (Pr)
    
    Spider web diagrams show how solution branches evolve in different
    parameter directions, revealing the complex bifurcation structure.
    """
    
    print("\nCreating MHD Spider Web Diagram...")
    print("="*70)
    
    # Define multiple branches starting from different equilibria
    # or going in different parameter directions
    
    # Example: Same system, different initial perturbations
    n_branches = 6
    branches_F = []
    branches_x0 = []
    branches_alpha0 = []
    branches_labels = []
    
    for i in range(n_branches):
        # Each branch has slightly different initial condition
        x0_perturbed = jnp.array([0.01 * (i+1), 0.0, 0.01 * (i+1)])
        
        branches_F.append(lambda x, alpha: jnp.array([
            alpha*x[0] - x[0]**3,
            x[0] - 2*x[1] + x[2],
            x[1] - 2*x[2]
        ]))
        branches_x0.append(x0_perturbed)
        branches_alpha0.append(0.1 * (i+1))
        branches_labels.append(f'Branch {i+1}')
    
    # Create spider web
    fig = create_mhd_spider_web(
        branches_F,
        branches_x0,
        branches_alpha0,
        branches_labels,
        ds=0.02,
        n_steps=100,
        tolerance=1e-8
    )
    
    plt.savefig('/home/claude/mhd_spider_web_integration.png', dpi=200, bbox_inches='tight')
    print("Saved: mhd_spider_web_integration.png\n")


# =============================================================================
# STEP 5: Tips for Your MHD Project
# =============================================================================

def integration_tips():
    """
    Key tips for successful continuation with your MHD solver:
    """
    
    tips = """
    INTEGRATION TIPS FOR ECS_MHD_2D PROJECT:
    ========================================
    
    1. INITIAL CONDITIONS:
       - Start continuation from a known equilibrium
       - Use time-stepping to find initial equilibria at low Re
       - Verify F(x0, Re0) ≈ 0 before starting
    
    2. STEP SIZE SELECTION:
       - Start with ds ~ 0.01 * Re for Reynolds number
       - Smaller steps near bifurcations
       - Increase step size in smooth regions
       - If continuation fails, reduce ds
    
    3. JAX COMPATIBILITY:
       - Ensure all operations use jax.numpy (not numpy)
       - Use jax.scipy for special functions
       - JIT-compile expensive functions for speed
       - Enable float64: export JAX_ENABLE_X64=1
    
    4. SPECTRAL METHODS:
       - State vector: flattened spectral coefficients
       - Include all fields: [omega_hat, psi_hat, A_hat, ...]
       - Dealiasing: use 2/3 rule or padding
       - Poisson solve: include in residual or eliminate psi
    
    5. CONVERGENCE ISSUES:
       - If Newton fails: check Jacobian accuracy
       - Use GMRES for large systems (not direct solve)
       - Preconditioner: approximate Jacobian inverse
       - Relaxation: under-relax Newton steps
    
    6. BIFURCATION DETECTION:
       - Monitor eigenvalues of Jacobian
       - Track solution norm changes
       - Look for turning points (dα/ds changes sign)
       - Branch switching at bifurcations
    
    7. OBSERVABLES FOR MHD:
       - Kinetic energy: ∫|∇ψ|² dx
       - Magnetic energy: ∫|A|² dx
       - Cross helicity: ∫(v·B) dx
       - Enstrophy: ∫|ω|² dx
       - Current: ∫|∇²A|² dx
    
    8. SPIDER WEB DIAGRAMS:
       - Start from multiple initial conditions
       - Vary different parameters (Re, Rm, Ha)
       - Connect branches at bifurcation points
       - Color-code by stability or energy
    
    9. EXACT COHERENT STRUCTURES (ECS):
       - Use continuation to track ECS families
       - Start from turbulent DNS snapshots
       - Apply symmetry constraints
       - Relative periodic orbits: add phase condition
    
    10. PERFORMANCE:
        - Use GPU acceleration with JAX
        - Parallelize multiple branches
        - Cache Jacobian for multiple RHS
        - Checkpoint large continuations
    
    EXAMPLE WORKFLOW:
    =================
    
    # 1. Initialize solver
    from your_mhd_solver import MHDSolver
    solver = MHDSolver(nx=64, ny=64)
    
    # 2. Find initial equilibrium
    x0 = solver.find_trivial_solution(Re=100)
    # OR
    x0 = solver.time_step_to_steady_state(Re=100)
    
    # 3. Define residual for continuation
    F = lambda state, Re: solver.compute_equilibrium_residual(state, Re, Rm=1.0)
    
    # 4. Run continuation
    from mhd_continuation_complete import ArcLengthContinuation
    
    cont = ArcLengthContinuation(F, x0, alpha0=100.0, ds=5.0, n_steps=100)
    results = cont.run()
    cont.plot(filename='my_bifurcation.png')
    
    # 5. Analyze
    Re_values = results['alpha']
    states = results['states']
    # Compute observables, detect bifurcations, etc.
    
    REFERENCES:
    ===========
    - Viswanath (2007): Recurrent motions within plane Couette turbulence
    - Kawahara & Kida (2001): Periodic motion in a plane Couette flow
    - Gibson et al. (2008): Visualizing the geometry of state space
    """
    
    print(tips)


# =============================================================================
# RUN EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MHD CONTINUATION INTEGRATION GUIDE")
    print("="*70 + "\n")
    
    # Print integration tips
    integration_tips()
    
    # Run MHD example
    print("\n" + "="*70)
    print("RUNNING MHD CONTINUATION EXAMPLE")
    print("="*70)
    results = mhd_continuation_example()
    
    # Create spider web
    print("\n" + "="*70)
    print("CREATING SPIDER WEB DIAGRAM")
    print("="*70)
    mhd_spider_web_example()
    
    print("\n" + "="*70)
    print("INTEGRATION GUIDE COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - mhd_integration_example.png")
    print("  - mhd_spider_web_integration.png")
    print("\nNext steps:")
    print("  1. Adapt this code to your MHD solver")
    print("  2. Verify F(x0, Re0) = 0 for your initial state")
    print("  3. Start with small systems (32x32) for testing")
    print("  4. Scale up once continuation is working")
    print("  5. Explore multi-parameter space for ECS")
    print("="*70 + "\n")