###################################################
#           Verification — analytic vs FEM
#           Particle in a box
###################################################

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from src.solver import Solver
from src.mesh import make_square_mesh


def analytic_eigenvalues(L: float, nev: int = 20):
    half = L / 2.0
    energies = []
    for nx in range(1, 20):
        for ny in range(1, 20):
            E = (np.pi**2 / 2.0) * (nx**2 + ny**2) / L**2
            energies.append(E)
    return np.array(sorted(energies)[:nev])


def run_verification(L: float = 10.0, N: int = 60, nev: int = 20,
                     output_dir: str = "results/verification"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running verification: L={L}, N={N}, nev={nev}")
    msh, facet_tags = make_square_mesh(L=L, N=N)
    solver   = Solver(msh=msh, facet_tags=facet_tags, L=L, N=N, nev=nev)
    A, M     = solver.assemble()
    energies, _ = solver.solve_eigenproblem(A, M)

    analytic    = analytic_eigenvalues(L, nev=nev)
    n_compare   = min(len(energies), len(analytic))
    abs_err     = np.abs(energies[:n_compare] - analytic[:n_compare])
    rel_err     = abs_err / np.abs(analytic[:n_compare])

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'Mode':>4}  {'FEM':>10}  {'Analytic':>10}  "
          f"{'Abs err':>10}  {'Rel err':>10}")
    print("-" * 54)
    for i in range(n_compare):
        print(f"  {i:02d}  {energies[i]:10.6f}  {analytic[i]:10.6f}"
              f"  {abs_err[i]:10.2e}  {rel_err[i]:10.2e}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    modes = np.arange(n_compare)

    axes[0].plot(modes, analytic[:n_compare], "ko-", ms=5, lw=1.5,
                 label="Analytic")
    axes[0].plot(modes, energies[:n_compare], "r--s", ms=4, lw=1.2,
                 label="FEM")
    axes[0].set_xlabel("Mode index")
    axes[0].set_ylabel(r"Energy $E_n$")
    axes[0].set_title("FEM vs analytic eigenvalues")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(modes, rel_err, "bs-", ms=4, lw=1.2)
    axes[1].axhline(1e-2, color="r", linestyle="--", lw=1, label="1% threshold")
    axes[1].set_xlabel("Mode index")
    axes[1].set_ylabel("Relative error")
    axes[1].set_title("Relative error vs mode index")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Solver verification — 2D particle in a box\n"
        f"$L={L}$,  $N={N}$,  P1 elements,  {N*N} cells",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/verification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {output_dir}/verification.png")

    np.savetxt(
        f"{output_dir}/verification_table.csv",
        np.column_stack([modes, energies[:n_compare],
                         analytic[:n_compare], abs_err, rel_err]),
        header="mode,E_fem,E_analytic,abs_err,rel_err",
        delimiter=",",
        fmt=["%.0f", "%.8f", "%.8f", "%.4e", "%.4e"]
    )

    return energies, analytic, rel_err


if __name__ == "__main__":
    run_verification()