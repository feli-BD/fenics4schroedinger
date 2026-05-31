###################################################
#           Main
###################################################

import os
import numpy as np
import matplotlib
matplotlib.use("Agg") # issue in WSL vscode
from mpi4py import MPI
from dolfinx import fem, io
from src.solver import Solver
from src.lieb_robinson import compute_lieb_robinson


def main():
    L          = 10.0
    N          = 60
    nev        = 150
    T          = 8.0      # total simulation time
    dt         = 0.005    # Crank-Nicolson timestep
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # ── Build solver ──────────────────────────────────────────────────────────
    solver = Solver(L=L, N=N, nev=nev)
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

    # ── Initial wavepacket ────────────────────────────────────────────────────
    coords   = solver.msh.geometry.x
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    x0, y0, sigma = -L/4, 0.0, 0.8

    psi0_raw  = np.exp(-((x_coords - x0)**2 + (y_coords - y0)**2)
                        / (2 * sigma**2))
    psi0_raw /= np.linalg.norm(psi0_raw)

    # ── Lieb-Robinson  ────────────────────────────────────────────────────────────────
    print("\nComputing Lieb-Robinson (spectral)...")
    v_meas, v_anal, mu = compute_lieb_robinson(solver, energies, eigvecs, output_dir, M)

    print(f"\n  Measured v_LR  = {v_meas:.4f}")
    print(f"  Analytic v_LR  = {v_anal:.4f}")
    print(f"  Decay constant = {mu}")



if __name__ == "__main__":
    main()