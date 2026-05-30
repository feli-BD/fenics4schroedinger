###################################################
#           main.py — Execution Pipeline
###################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from petsc4py import PETSc
from mpi4py import MPI
from solver_wire import Solver_wire

def run_simulation(geometry_type="straight"):
    print(f"\n==================================================")
    print(f"Starting Quantum Wire Simulation: {geometry_type.upper()}")
    print(f"==================================================")

    # 1. Initialize Solver parameters
    Lx = 20.0
    Ly = 4.0
    h = 0.15       # Characteristic element length
    nev = 120      # Request more eigenvalues to capture high-momentum dynamics
    
    sim_solver = Solver_wire(Lx=Lx, Ly=Ly, h=h, geometry_type=geometry_type, nev=nev)
    A, M = sim_solver.assemble()
    energies, eigvecs = sim_solver.solve(A, M)

    # 2. Extract Degree of Freedom (DOF) coordinates
    # Allows mapping of nodal array indexes directly to physical (x, y) space
    dof_coords = sim_solver.V.tabulate_dof_coordinates()[:, :2]
    x_coords = dof_coords[:, 0]
    y_coords = dof_coords[:, 1]
    n_dofs = len(x_coords)

    # 3. Define the Initial State (Gaussian Wavepacket localized on the left)
    x0, y0 = -5.0, 0.0     # Initial packet center
    sigma = 0.6            # Spatial width
    kx = 3.5               # Initial wave momentum pointing right along the wire

    # Evaluate the analytical profile over the mesh nodes
    psi_0 = np.exp(-((x_coords - x0)**2 + (y_coords - y0)**2) / (2 * sigma**2)) * np.exp(1j * kx * x_coords)
    
    # Normalize initial state using the Mass Matrix: ||psi||^2 = psi* . M . psi
    psi_0_vec = M.createVecRight()
    psi_0_vec.array[:] = psi_0
    M_psi_0 = M.createVecLeft()
    M.mult(psi_0_vec, M_psi_0)
    norm_sq = np.vdot(psi_0, M_psi_0.array)
    psi_0 = psi_0 / np.sqrt(norm_sq)
    
    # 4. Project Initial State onto Eigenmodes using Mass Matrix
    # c_n = <phi_n | psi_0> = phi_n^dagger * M * psi_0
    psi_0_vec.array[:] = psi_0
    M.mult(psi_0_vec, M_psi_0)
    
    c_n = []
    for n in range(len(energies)):
        cn_val = np.vdot(eigvecs[n], M_psi_0.array)
        c_n.append(cn_val)
    c_n = np.array(c_n)

    # Verify Projection Completeness (Should be close to 1.0)
    completeness = np.sum(np.abs(c_n)**2)
    print(f"[{geometry_type.upper()}] Initial State Projection Completeness: {completeness:.4f}")

    # 5. Time Evolution Grid Setup
    t_max = 6.0
    dt = 0.05
    times = np.arange(0.0, t_max + dt, dt)
    
    # Grid setup for 1D space-time visualization cross-sections
    x_grid = np.linspace(-Lx/2, Lx/2, 200)
    space_time_density = np.zeros((len(times), len(x_grid)))
    
    # Arrays tracking the physical Quantity of Interest (QoI)
    right_mask = x_coords > 0.0
    p_right_t = []

    # 6. Time Evolution Loop via Modal Superposition
    print("Propagating wavepacket through time...")
    for idx, t in enumerate(times):
        # Compute time-dependent expansion coefficient: c_n(t) = c_n * exp(-i * E_n * t)
        c_n_t = c_n * np.exp(-1j * energies * t)
        
        # Superimpose modes: psi(t) = Sum( c_n(t) * phi_n )
        psi_t = np.dot(c_n_t, eigvecs)
        
        # Calculate Nodal Probability Density
        prob_density = np.abs(psi_t)**2
        
        # Compute Quantity of Interest (Right-Half Survival Probability) using Mass Matrix
        psi_t_vec = M.createVecRight()
        psi_t_vec.array[:] = psi_t
        M_psi_t = M.createVecLeft()
        M.mult(psi_t_vec, M_psi_t)
        P_right = np.sum(np.real(np.conj(psi_t[right_mask]) * M_psi_t.array[right_mask]))
        p_right_t.append(P_right)
        
        # Project 2D unstructured mesh data onto a 1D x-axis profile for scannable plotting
        # Uses standard kernel-density binning along the channel length
        for i, x_val in enumerate(x_grid):
            mask = np.abs(x_coords - x_val) < 0.25
            if np.any(mask):
                space_time_density[idx, i] = np.mean(prob_density[mask])

    print(f"[{geometry_type.upper()}] Simulation Complete.")
    return times, x_grid, space_time_density, p_right_t, times


# ── Plotting Pipeline ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run both scenario variants programmatically
    times, x_grid, density_A, p_right_A, _ = run_simulation(geometry_type="straight")
    _, _, density_B, p_right_B, _ = run_simulation(geometry_type="constricted")

    # Generate the Diagnostic Dashboard for Variant B (The Bottleneck Wire)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Estimated physical Lieb-Robinson speed marker
    v_lr = 2.5 
    x_start = -5.0
    cone_right = x_start + v_lr * times
    
    # Panel 1: Linear Space-Time Density Plot
    im1 = axs[0].pcolormesh(x_grid, times, density_A, shading='auto', cmap='viridis')
    axs[0].plot(cone_right[cone_right <= 10], times[cone_right <= 10], 'w--', label='Estimated $v_{LR}$')
    axs[0].set_title("Space-Time Wavefront Density (Linear Scale)")
    axs[0].set_xlabel("Position Along Channel $x$")
    axs[0].set_ylabel("Time $t$")
    axs[0].set_xlim(-10, 10)
    axs[0].legend()
    fig.colorbar(im1, ax=axs[0], label=r"$\langle|\psi|^2 \rangle_y$")

    # Panel 2: Logarithmic Space-Time Density Plot (Exposes Light Cone Containment)
    # Floor near-zero float variations to -6.0 to match your dashboard style cleanly
    log_density = np.log10(density_B + 1e-6)
    im2 = axs[1].pcolormesh(x_grid, times, log_density, shading='auto', cmap='magma', vmin=-4, vmax=0)
    axs[1].plot(cone_right[cone_right <= 10], times[cone_right <= 10], 'w--', label='Estimated $v_{LR}$')
    axs[1].set_title(r"Space-Time Information Light Cone ($\log_{10}$ Scale)")
    axs[1].set_xlabel("Position Along Channel $x$")
    axs[1].set_ylabel("Time $t$")
    axs[1].set_xlim(-10, 10)
    axs[1].legend()
    fig.colorbar(im2, ax=axs[1], label="$ log 10 |psi|^2$")

    # Panel 3: Scenario Comparison (Your Declared Quantity of Interest)
    axs[2].plot(times, p_right_A, 'b-', lw=2.5, label="Variant A (Straight Wire)")
    axs[2].plot(times, p_right_B, 'r--', lw=2.5, label="Variant B (Constricted Bottleneck)")
    axs[2].set_title("Quantity of Interest: Right-Half Density Tracking")
    axs[2].set_xlabel("Time $t$")
    axs[2].set_ylabel("Probability Vector $P_{right}(t)$")
    axs[2].grid(True, linestyle=":")
    axs[2].legend()

    plt.savefig("../results/figures/quantum_wire_simulation.png", dpi=300)
    print("\n[SUCCESS] Figures successfully exported to 'results/figures/quantum_wire_simulation.png'")
    plt.show()