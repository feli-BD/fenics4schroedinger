###################################################
#           Verification — time evolution
#           Stationary eigenstate test
###################################################

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy.sparse import csr_matrix
from src.mesh import make_square_mesh
from src.solver import Solver


def run_time_evolution_verification(L=10.0, N=60, nev=50,T = 8.0, dt=0.05, output_dir="results/time_verification"):
    os.makedirs(output_dir, exist_ok=True)

    print( f"Running time evolution verification: L={L}, N={N}, T={T}, dt={dt}")

    msh, facet_tags = make_square_mesh(L=L, N=N)
    solver = Solver(msh=msh, facet_tags=facet_tags, L=L, N=N, nev=nev)
    A, M = solver.assemble()
    energies, eigvecs = solver.solve_eigenproblem(A, M)

    psi0 = eigvecs[0].copy()
    E0 = energies[0]

    print(f"Ground state energy: {E0:.8f}")

    times, snapshots = solver.solve_time_evolution(
        M=M,
        energies=energies,
        eigvecs=eigvecs,
        psi0=psi0,
    )

    indptr, indices, data = M.getValuesCSR()

    M_csr = csr_matrix(
        (data, indices, indptr),
        shape=M.getSize(),
    )

    norm_errors = []
    overlap_errors = []
    central_dof_index = len(psi0) // 2
    psi_numerical_trace = []
    psi_exact_trace = []

    for t, psi_num in zip(times, snapshots):

        # Exact solution:
        psi_exact = np.exp(-1j * E0 * t) * psi0

        norm = np.real(np.vdot(psi_num, M_csr @ psi_num))
        overlap = np.abs(np.vdot(psi_exact, M_csr @ psi_num))

        norm_errors.append(abs(norm - 1.0))
        overlap_errors.append(abs(overlap - 1.0))

        psi_numerical_trace.append(np.real(psi_num[central_dof_index]))
        psi_exact_trace.append(np.real(psi_exact[central_dof_index]))


    norm_errors = np.array(norm_errors)
    overlap_errors = np.array(overlap_errors)

    print()
    print(f"Max norm error     : {norm_errors.max():.3e}")
    print(f"Max overlap error  : {overlap_errors.max():.3e}")

   # ── Plots ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # LEFT PANEL: Direct comparison (Analytic vs FEM Time Trace)
    axes[0].plot(times, psi_exact_trace, "k-", marker="o", ms=4, lw=1.5, label="Analytic")
    axes[0].plot(times, psi_numerical_trace, "r--", marker="s", ms=3, lw=1.2, label="FEM")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(r"$\mathrm{Re}(\psi(t))$ at central DOF")
    axes[0].set_title("Ground-state wave oscillations")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # RIGHT PANEL: Log scale errors with a 1% reference threshold line
    axes[1].semilogy(times, overlap_errors, "b-", marker="s", ms=3, lw=1.2, label="Overlap Error")
    axes[1].semilogy(times, norm_errors, "g-", marker="o", ms=3, lw=1.2, label="Norm Conservation Error")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Relative error")
    axes[1].set_title("Relative error vs time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Matching layout typography block
    plt.suptitle(
        f"Time evolution verification — 2D particle in a box\n"
        f"$L={L}$,  $N={N}$,  P1 elements,  {N*N} cells",
        fontsize=11
    )

    outfile = os.path.join(output_dir, "time_evolution_verification.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved to {outfile}")

    return times, norm_errors, overlap_errors


if __name__ == "__main__":
    run_time_evolution_verification()