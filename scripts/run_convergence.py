###################################################
#           Mesh convergence study
#           run_convergence.py
###################################################

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from src.solver import Solver
from scripts.run_verification import analytic_eigenvalues


def run_convergence(L: float = 10.0,
                    mesh_sizes: list = [8, 16, 32, 64, 96],
                    nev: int = 10,
                    output_dir: str = "results/convergence"):
    os.makedirs(output_dir, exist_ok=True)

    analytic = analytic_eigenvalues(L, nev=nev)
    h_vals, l2_errors, max_errors = [], [], []

    for N in mesh_sizes:
        h = L / N
        print(f"\nN={N}, h={h:.4f}")

        solver      = Solver(L=L, N=N, nev=nev)
        A, M        = solver.assemble()
        energies, _ = solver.solve_eigenproblem(A, M)

        n_compare = min(len(energies), len(analytic))
        abs_err   = np.abs(energies[:n_compare] - analytic[:n_compare])
        rel_err   = abs_err / np.abs(analytic[:n_compare])

        l2_err  = np.sqrt(np.mean(abs_err**2))   # RMS eigenvalue error
        max_err = np.max(abs_err)                  # max eigenvalue error

        h_vals.append(h)
        l2_errors.append(l2_err)
        max_errors.append(max_err)

        print(f"  RMS eigenvalue error: {l2_err:.4e}")
        print(f"  Max eigenvalue error: {max_err:.4e}")

    h_vals    = np.array(h_vals)
    l2_errors = np.array(l2_errors)
    max_errors = np.array(max_errors)

    # ── Estimate convergence rates ────────────────────────────────────────────
    # Fit log(error) = p*log(h) + c  →  p is the convergence rate
    rate_l2  = np.polyfit(np.log(h_vals), np.log(l2_errors),  1)[0]
    rate_max = np.polyfit(np.log(h_vals), np.log(max_errors), 1)[0]
    print(f"\nEstimated convergence rate (RMS): {rate_l2:.2f}  (expect ~2.0 for P1)")
    print(f"Estimated convergence rate (max): {rate_max:.2f}  (expect ~2.0 for P1)")

    # ── Log-log plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(h_vals, l2_errors,  "bo-", ms=6, lw=1.5,
              label=fr"RMS error  (rate $\approx$ {rate_l2:.2f})")
    ax.loglog(h_vals, max_errors, "rs-", ms=6, lw=1.5,
              label=fr"Max error  (rate $\approx$ {rate_max:.2f})")

    # Reference O(h^2) line
    ref = h_vals**2 * (l2_errors[0] / h_vals[0]**2)
    ax.loglog(h_vals, ref, "k--", lw=1, label=r"$\mathcal{O}(h^2)$ reference")

    ax.set_xlabel(r"Mesh size $h = L/N$")
    ax.set_ylabel("Eigenvalue error")
    ax.set_title(
        "Mesh convergence — 2D particle in a box\n"
        f"$L={L}$,  P1 elements,  first {nev} eigenvalues"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {output_dir}/convergence.png")

    np.savetxt(
        f"{output_dir}/convergence_table.csv",
        np.column_stack([h_vals, l2_errors, max_errors]),
        header="h,rms_error,max_error",
        delimiter=","
    )

    return h_vals, l2_errors, max_errors


if __name__ == "__main__":
    run_convergence()