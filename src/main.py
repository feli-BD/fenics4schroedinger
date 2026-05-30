######################################################
#               Main (where the magic happens)
######################################################

#           Imports
from mpi4py import MPI
import numpy as np
from pathlib import Path
import ufl
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import assemble_matrix
from petsc4py import PETSc
from slepc4py import SLEPc
from solver import Solver
from lieb_robinson import compute_lieb_robinson
import os


def main(output_dir = None):
    # ── Parameters ────────────────────────────────────────────────────────────
    L    = 10.0   # domain half-width (nm, ℏ = m = 1 units)
    N     = 60     # mesh resolution per side
    omega = 0.5    # harmonic trap frequency
    nev   = 1000    # number of eigenvalues to compute

    # ── Output directory ───────────────────────────────────────────────────────
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if output_dir is None:
        output_dir = output_dir = PROJECT_ROOT / "results"

    output_dir.mkdir(exist_ok=True)


    # ── Build & run solver ────────────────────────────────────────────────────
    solver = Solver(L=L, N=N, omega=omega, nev=nev)

    print("Assembling stiffness and mass matrices...")
    A, M = solver.assemble()

    print("Solving generalised eigenvalue problem...")
    energies, eigvecs = solver.solve(A, M)

    # ── Print eigenvalues ─────────────────────────────────────────────────────
    print("\nComputed eigenvalues (energies):")
    for i, E in enumerate(energies):
        print(f"  E_{i:02d} = {E:.6f}")

    # ── Analytic check (2D harmonic oscillator: E = omega*(nx+ny+1)) ──────────
    print("\nAnalytic eigenvalues (2D harmonic oscillator):")
    analytic = []
    for nx in range(6):
        for ny in range(6):
            analytic.append(omega * (nx + ny + 1))
    analytic = sorted(set(np.round(analytic, 6)))[:nev]
    for i, E in enumerate(analytic):
        print(f"  E_{i:02d} = {E:.6f}")

    # ── Relative error for first few modes ───────────────────────────────────
    print("\nRelative error (FEM vs analytic):")
    for i in range(min(8, len(energies), len(analytic))):
        err = abs(energies[i] - analytic[i]) / abs(analytic[i])
        print(f"  mode {i:02d}: {err:.2e}")


    # ── Write all eigenstates ─────────────────────────────────────────────────
    print(f"\nWriting {len(eigvecs)} eigenstates to {output_dir}/...")

    for i, vec in enumerate(eigvecs):
        psi = fem.Function(solver.V)
        psi.x.array[:] = vec
        psi.name = f"psi_{i}"

        filepath = f"{output_dir}/eigenstate_{i:02d}.xdmf"
        with io.XDMFFile(MPI.COMM_WORLD, filepath, "w") as xdmf:
            xdmf.write_mesh(solver.msh)
            xdmf.write_function(psi)

        print(f"  eigenstate_{i:02d}.xdmf  (E = {energies[i]:.6f})")

    print("\nDone. Open any eigenstate_XX.xdmf in Paraview with Xdmf3ReaderS.")


if __name__ == "__main__":
    main()