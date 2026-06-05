###################################################
#           Main
###################################################

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg") # issue in WSL vscode
from mpi4py import MPI
from dolfinx import fem, io
from src.solver import Solver
from src.mesh import make_square_mesh, make_circular_mesh, make_stadium_mesh


def main(msh_type: str = "square", num_packets: int = 1, L: float = 10.0, N: int = 60, nev: int = 150, radius: float = 5.0, target_h: float = 0.2, T: float = 5.0, frames: float = 120):
    print(f"--- Running Simulation Setup ---")
    print(f"Mesh Type: {msh_type} | Eigenvalues (nev): {nev}")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # ── Build mesh ───────────────────────────────────────────────────────────────
    print("Building mesh...")
  
    if msh_type == "square":
        msh, facet_tags = make_square_mesh(L=L, N=N)
    elif msh_type == "circular":
        msh, facet_tags = make_circular_mesh(radius=radius, target_h=target_h)
    elif msh_type == "stadium":
        msh, facet_tags = make_stadium_mesh(L_straight=L, radius=radius, target_h=target_h)
        total_time = (L / 2.0) + radius
    else:
        raise ValueError("Invalid mesh type. Choose 'square', 'circular', or 'stadium'.")

    # ── Build solver ──────────────────────────────────────────────────────────
    solver = Solver(msh = msh, facet_tags = facet_tags, L=L, N=N, nev=nev)
    print("Assembling matrices...")
    A, M = solver.assemble()

    # ── Eigenvalue problem ────────────────────────────────────────────────────
    print("\nSolving eigenvalue problem...")
    energies, eigvecs = solver.solve_eigenproblem(A, M)

    print("\nFirst 8 eigenvalues (FEM):")
    for i, E in enumerate(energies[:8]):
        print(f"  E_{i:02d} = {E:.6f}")
    

    # ── Write eigenstates to XDMF ─────────────────────────────────────────────
    eig_dir = os.path.join(output_dir, "eigenstates")
    os.makedirs(eig_dir, exist_ok=True)
    for i, vec in enumerate(eigvecs[:8]):      # save first 8 only
        psi = fem.Function(solver.V)
        # psi.x.array[:] = vec
        # psi.name = f"psi_{i}"
        rho_eigen = np.abs(vec)**2
        psi.x.array[:] = rho_eigen
        
        # Update the name so it renders correctly labeled in ParaView's pipeline
        psi.name = f"probability_density_{i}"
        with io.XDMFFile(MPI.COMM_WORLD,
                         f"{eig_dir}/eigenstate_{i:02d}_{msh_type}.xdmf", "w") as f:
            f.write_mesh(solver.msh)
            f.write_function(psi)


    # ── Wavepacket ────────────────────────────────────────────────────
    coords   = solver.msh.geometry.x
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    if num_packets == 1:
        wavepacket_configs = [{"x0": 0.0, "y0": 0.0, "sigma": 0.4, "kx": 0.0, "ky": 2.0}]
    elif num_packets == 2:
        wavepacket_configs = [
            {"x0": -L/4.0, "y0": 0.0, "sigma": 0.2, "kx":  10.0, "ky": 0.0},
            {"x0":  L/4.0, "y0": 0.0, "sigma": 0.2, "kx": -10.0, "ky": 0.0}
        ]
    elif num_packets == 3:
        wavepacket_configs = [
            {"x0": -L/4.0, "y0":  0.0, "sigma": 0.2, "kx":  10.0, "ky":  0.0}, 
            {"x0":  L/4.0, "y0":  0.0, "sigma": 0.2, "kx": -10.0, "ky":  0.0},
            {"x0":   0.0,  "y0": -L/4.0, "sigma": 0.2, "kx":   0.0, "ky": 10.0}  
        ]
    elif num_packets == 4:
        wavepacket_configs = [
        {"x0": -L/4.0, "y0":   0.0,  "sigma": 0.2, "kx":  10.0, "ky":   0.0}, # Left
        {"x0":  L/4.0, "y0":   0.0,  "sigma": 0.2, "kx": -10.0, "ky":   0.0}, # Right
        {"x0":   0.0,  "y0": -L/4.0, "sigma": 0.2, "kx":   0.0, "ky":  10.0}, # Bottom
        {"x0":   0.0,  "y0":  L/4.0, "sigma": 0.2, "kx":   0.0, "ky": -10.0}  # Top
    ]
    else:
        raise ValueError("Invalid number of wavepackets. Choose 1, 2, 3, or 4.")
    
    psi0_raw = np.zeros_like(x_coords, dtype=np.complex128)

    for packet in wavepacket_configs:
        r_squared = (x_coords - packet["x0"])**2 + (y_coords - packet["y0"])**2
        psi0_raw += np.exp(-r_squared / (2 * packet["sigma"]**2)) * np.exp(1j * (packet["kx"] * x_coords + packet["ky"] * y_coords))

    # ── Spectral time evolution ────────────────────────────────────────────────
    print("\nRunning spectral time evolution...")

    times, snapshots = solver.solve_time_evolution(M = M, energies=energies,eigvecs=eigvecs, psi0=psi0_raw, T=T, frames = frames)

    psi_eval_func = fem.Function(solver.V, name="Probability_Density")

    # Initialize the VTK file exporter
    vtk_path = os.path.join(output_dir, f"time_evolution/time_evol_{msh_type}_{num_packets}pkts.pvd")
    with io.VTKFile(solver.msh.comm, vtk_path, "w") as vtk:
        
        print("  Exporting snapshots for ParaView...")
        for ti, t in enumerate(times):
            # Calculate the complex wavefunction array for this specific time
            psi_t = snapshots[ti] 
            
            # Calculate the physical probability density: |psi|^2
            rho_t = np.abs(psi_t)**2
            
            # Assign the calculated real densities to the FEniCSx function vector
            psi_eval_func.x.array[:] = rho_t
            
            # Write the timestep to the VTK file series
            vtk.write_function(psi_eval_func, t)

    print(f"  [SUCCESS] ParaView compatible file saved to {vtk_path}")



# This block intercepts terminal execution
if __name__ == "__main__":
    # Initialize the argument parser with a clear description
    parser = argparse.ArgumentParser(
        description="FEniCS4Schroedinger: Finite Element Simulation for the 2D Schrödinger Equation."
    )

    # Add parameters matching your function defaults
    parser.add_argument(
        "--mesh_type", type=str, default="square", choices=["square", "circular", "stadium"],
        help="Type of domain geometry to build (default: %(default)s)"
    )

    parser.add_argument(
        "--num_packets", type=int, default=1,
        help="Number of Gaussian wavepackets to initialize (default: %(default)s)"
    )

    parser.add_argument(
        "--L", type=float, default=10.0,
        help="Full side length of the square domain (default: %(default)s)"
    )
    parser.add_argument(
        "--N", type=int, default=60,
        help="Number of grid divisions per side for the square (default: %(default)s)"
    )
    parser.add_argument(
        "--nev", type=int, default=150,
        help="Number of target eigenvalues to resolve with SLEPc (default: %(default)s)"
    )
    parser.add_argument(
        "--radius", type=float, default=5.0,
        help="Radius of the circular dot domain if geometry is circle (default: %(default)s)"
    )
    parser.add_argument(
        "--target_h", type=float, default=0.2,
        help="Target element mesh sizing h for the unstructured circle geometry (default: %(default)s)"
    )

    parser.add_argument(
        "--T", type=float, default=5.0,
        help="Total simulation time (default: %(default)s)"
    )
    parser.add_argument(
        "--frames", type=float, default=120,
        help="Number of frames to save (default: %(default)s)"
    )


    # Parse what the user typed in the terminal
    args = parser.parse_args()

    # Pass the parsed CLI variables straight into your main function
    main(
        msh_type=args.mesh_type,
        num_packets=args.num_packets,
        L=args.L,
        N=args.N,
        nev=args.nev,
        radius=args.radius,
        target_h=args.target_h,
        T=args.T,
        frames=args.frames
    )