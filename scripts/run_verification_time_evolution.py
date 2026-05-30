###################################################
#           Verification — time evolution
#           Stationary eigenstate test
###################################################

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from src.solver import Solver


def run_time_evolution_verification(
    L=10.0,
    N=60,
    nev=50,
    T=10.0,
    dt=0.05,
    output_dir="results/time_verification",
):
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"Running time evolution verification:"
        f" L={L}, N={N}, T={T}, dt={dt}"
    )

    # --------------------------------------------------
    # Build solver
    # --------------------------------------------------
    solver = Solver(L=L, N=N, nev=nev)

    A, M = solver.assemble()

    energies, eigvecs = solver.solve_eigenproblem(A, M)

    # --------------------------------------------------
    # Use ground state as initial condition
    # --------------------------------------------------
    psi0 = eigvecs[0].copy()
    E0 = energies[0]

    print(f"Ground state energy: {E0:.8f}")

    # --------------------------------------------------
    # Evolve
    # --------------------------------------------------
    times, snapshots = solver.solve_time_evolution(
        M=M,
        energies=energies,
        eigvecs=eigvecs,
        psi0=psi0,
    )

    # --------------------------------------------------
    # Convert mass matrix once
    # --------------------------------------------------
    from scipy.sparse import csr_matrix

    indptr, indices, data = M.getValuesCSR()

    M_csr = csr_matrix(
        (data, indices, indptr),
        shape=M.getSize(),
    )

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------
    norm_errors = []
    overlap_errors = []

    for t, psi_num in zip(times, snapshots):

        # Exact solution:
        psi_exact = np.exp(-1j * E0 * t) * psi0

        norm = np.real(
            np.vdot(psi_num, M_csr @ psi_num)
        )

        overlap = np.abs(
            np.vdot(
                psi_exact,
                M_csr @ psi_num
            )
        )

        norm_errors.append(abs(norm - 1.0))
        overlap_errors.append(abs(overlap - 1.0))

    norm_errors = np.array(norm_errors)
    overlap_errors = np.array(overlap_errors)

    print()
    print(f"Max norm error     : {norm_errors.max():.3e}")
    print(f"Max overlap error  : {overlap_errors.max():.3e}")

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].semilogy(
        times,
        norm_errors,
        "bo-",
        ms=3,
        label="Norm error",
    )

    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(r"$|\|\psi\|_M^2 - 1|$")
    axes[0].set_title("Norm conservation")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(
        times,
        overlap_errors,
        "ro-",
        ms=3,
        label="Overlap error",
    )

    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(r"$|1-|\langle\psi_{exact}|\psi\rangle_M||$")
    axes[1].set_title("Agreement with exact solution")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        "Time evolution verification\n"
        "Ground-state propagation"
    )

    plt.tight_layout()

    outfile = os.path.join(
        output_dir,
        "time_evolution_verification.png",
    )

    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved to {outfile}")

    return times, norm_errors, overlap_errors


if __name__ == "__main__":
    run_time_evolution_verification()