###################################################
#          Lieb-Robinson Study
###################################################

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg") # issue in WSL vscode
from mpi4py import MPI
from dolfinx import fem, io
from src.solver import Solver
from src.lieb_robinson import compute_lieb_robinson
from src.mesh import make_square_mesh, make_circular_mesh, make_stadium_mesh


def run_lieb_robinson_study(msh_type: str = "square", L: float = 10.0, N: int = 60, nev: int = 150, radius: float = 5.0, target_h: float = 0.2, T: float = 8.0, dt: float = 0.005):
    print(f"--- Running Lieb-Robinson Study ---")
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
    solver = Solver(msh=msh, facet_tags=facet_tags, L=L, N=N, nev=nev)
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
        psi.x.array[:] = vec
        psi.name = f"psi_{i}"
        with io.XDMFFile(MPI.COMM_WORLD,
                         f"{eig_dir}/eigenstate_{i:02d}.xdmf", "w") as f:
            f.write_mesh(solver.msh)
            f.write_function(psi)


    # ── Lieb-Robinson  ────────────────────────────────────────────────────────────────
    print("\nComputing Lieb-Robinson (spectral)...")
    v_meas, v_anal, mu = compute_lieb_robinson(solver, energies, eigvecs, output_dir, M)

    print(f"\n  Measured v_LR  = {v_meas:.4f}")
    print(f"  Analytic v_LR  = {v_anal:.4f}")
    print(f"  Decay constant = {mu}")


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
        "-L", type=float, default=10.0,
        help="Full side length of the square domain (default: %(default)s)"
    )
    parser.add_argument(
        "-N", type=int, default=60,
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

    # Parse what the user typed in the terminal
    args = parser.parse_args()

    # Pass the parsed CLI variables straight into your main function
    run_lieb_robinson_study(
        msh_type=args.mesh_type,
        L=args.L,
        N=args.N,
        nev=args.nev,
        radius=args.radius,
        target_h=args.target_h
    )
