###################################################
#           Lieb-Robinson bounds
#           Particle in a box
###################################################

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dolfinx.io as io
import dolfinx.fem as fem
import src.solver as Solver

def analytic_lr_velocity(L: float, energies: np.ndarray):
    """
    For a free particle, the group velocity is v = dE/dk = k (in natural units).
    The maximum group velocity from the resolved modes gives the LR velocity:
        v_LR = sqrt(2 * E_max)
    since E = k^2/2  =>  k = sqrt(2E)  =>  v = k = sqrt(2E).
    """
    return float(np.sqrt(2.0 * energies.max()))


def compute_lieb_robinson(solver, energies, eigvecs, output_dir, M):
    """
    Computes and plots the Lieb-Robinson space-time diagram.

    Parameters
    ----------
    solver       : Solver instance
    energies     : eigenvalues from solve_eigenproblem
    eigvecs      : eigenvectors from solve_eigenproblem
    output_dir   : where to save the figure
    M            : PETSc mass matrix (for projection)
    """
    os.makedirs(output_dir, exist_ok=True)

    coords   = solver.msh.geometry.x
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    # ── 1. Initial Gaussian wavepacket — left quarter of box ──────────────────
    x0    = -solver.L / 4
    y0    = 0.0
    sigma = 0.8

    psi0_raw  = np.exp(-((x_coords - x0)**2 + (y_coords - y0)**2)
                        / (2 * sigma**2))

    # M-normalize the wavepacket
    vec_raw        = M.createVecRight(); vec_raw.array[:]  = psi0_raw
    vec_M_raw      = M.createVecLeft();  M.mult(vec_raw, vec_M_raw)
    norm_M         = np.sqrt(float(vec_raw.array @ vec_M_raw.array))
    psi0           = psi0_raw / norm_M

    # ── 2. Projection onto eigenbasis ─────────────────────────────────────────
    vec_psi0       = M.createVecRight(); vec_psi0.array[:] = psi0
    vec_M_psi0     = M.createVecLeft();  M.mult(vec_psi0, vec_M_psi0)
    coeffs         = eigvecs @ vec_M_psi0.array

    completeness   = float(np.sum(np.abs(coeffs)**2))
    print(f"  Projection completeness: {completeness:.4f}  (want > 0.95)")
    if completeness < 0.80:
        print("  WARNING: increase nev for reliable spectral evolution")

    # ── 3. Time points and wavefunction snapshots ─────────────────────────────

    # TODO: combine this spectral evolution with the one in solver.solve_time_evolution to avoid code duplication

    times = np.linspace(0, solver.L / 2, 120)   # T ~ L/2 for propagation

    def psi_at_time(t):
        phases = np.exp(-1j * energies * t)
        return (coeffs * phases) @ eigvecs

    print("  Computing spectral time evolution...")
    snapshots = np.array([psi_at_time(t) for t in times])

    # ── 4. Probe points — y-averaged density along x-axis ────────────────────
    probe_xs = np.linspace(-solver.L / 2 * 0.9, solver.L / 2 * 0.9, 120)
    dx_probe = probe_xs[1] - probe_xs[0]
    dy_half  = solver.L / 2 * 0.3    # average over central 30% width

    print("  Building space-time density...")
    density_xt = np.zeros((len(times), len(probe_xs)))

    for ti, psi_t in enumerate(snapshots):
        rho = np.abs(psi_t)**2
        for xi, px in enumerate(probe_xs):
            mask = (
                (np.abs(x_coords - px) < dx_probe) &
                (np.abs(y_coords)      < dy_half)
            )
            density_xt[ti, xi] = rho[mask].mean() if mask.any() else 0.0

    # ── 5. Wavefront detection ────────────────────────────────────────────────
    threshold     = 1e-4 * density_xt.max()
    front_r, front_l, front_t = [], [], []

    for ti, t in enumerate(times[1:], 1):
        row  = density_xt[ti]
        mask = row > threshold
        if mask.any():
            front_r.append(probe_xs[mask].max())
            front_l.append(probe_xs[mask].min())
            front_t.append(t)

    front_t = np.array(front_t)
    front_r = np.array(front_r)
    front_l = np.array(front_l)

    # Fit LR velocity from both fronts
    v_LR = 1.0
    if len(front_t) > 3:
        v_right = np.polyfit(front_t, front_r, 1)[0]
        v_left  = np.polyfit(front_t, front_l, 1)[0]
        v_LR    = (v_right - v_left) / 2
        print(f"  v_right = {v_right:.4f},  v_left = {v_left:.4f}")
        print(f"  Measured v_LR = {v_LR:.4f}")

    # Analytic LR velocity
    v_analytic = analytic_lr_velocity(solver.L, energies)
    print(f"  Analytic v_LR = sqrt(2*E_max) = {v_analytic:.4f}")

    # ── 6. Exponential decay fit outside cone ─────────────────────────────────
    t_idx    = int(0.65 * len(times))
    t_slice  = times[t_idx]
    profile  = density_xt[t_idx]
    x_rel    = probe_xs - (x0 + v_LR * t_slice)

    mu, slope, intercept = None, None, None
    outside  = x_rel > 0.5
    if outside.sum() > 4:
        valid = profile[outside] > 1e-12
        if valid.sum() > 4:
            slope, intercept = np.polyfit(
                x_rel[outside][valid],
                np.log(profile[outside][valid]), 1
            )
            mu = -slope
            print(f"  Decay constant mu = {mu:.4f}")

    # ── 7. Professional Publication-Quality Plots ─────────────────────────────
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    t_arr = np.linspace(0, times[-1], 300)

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL 1: Space-Time Heatmap with Logarithmic Contours
    # ═════════════════════════════════════════════════════════════════════════
    # We use a log scale for the heatmap to expose the faint wavefront structure
    log_density = np.log10(density_xt + 1e-10)
    
    im0 = axes[0].imshow(
        log_density, aspect="auto", origin="lower",
        extent=[probe_xs[0], probe_xs[-1], times[0], times[-1]],
        cmap="magma", vmin=-6, vmax=log_density.max()
    )
    
    # Overlay discrete logarithmic contours to show the shape of the cone clearly
    contours = axes[0].contour(
        probe_xs, times, log_density, 
        levels=[-4, -3, -2], colors="white", alpha=0.5, linestyles=":"
    )
    axes[0].clabel(contours, inline=True, fmt=lambda x: f"$10^{{{int(x)}}}$", fontsize=8)

    # Draw the theoretical and measured bounds
    axes[0].plot(x0 + v_LR * t_arr, t_arr, "w--", lw=2, label=fr"Meas. $v_{{LR}} = {v_LR:.2f}$")
    axes[0].plot(x0 - v_LR * t_arr, t_arr, "w--", lw=2)
    axes[0].plot(x0 + v_analytic * t_arr, t_arr, "c-.", lw=1.2, label=fr"Analytic Limit $v = {v_analytic:.2f}$")
    axes[0].plot(x0 - v_analytic * t_arr, t_arr, "c-.", lw=1.2)
    
    axes[0].set_xlim(probe_xs[0], probe_xs[-1])
    axes[0].set_ylim(times[0], times[-1])
    axes[0].set_xlabel("Spatial Coordinate $x$", fontsize=11)
    axes[0].set_ylabel("Time $t$", fontsize=11)
    axes[0].set_title("A: Information Light Cone (Log Scale)", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper right", frameon=True, facecolor="black", framealpha=0.6, labelcolor="white")
    
    cbar = plt.colorbar(im0, ax=axes[0])
    cbar.set_label(r"$\log_{10} \langle|\psi|^2\rangle_y$", fontsize=10)

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL 2: Snapshot Evolution (How the wave spreads over time)
    # ═════════════════════════════════════════════════════════════════════════
    # Pick 4 evenly spaced times to demonstrate the dynamics intuitively
    sample_indices = [0, int(len(times)*0.25), int(len(times)*0.5), int(len(times)*0.75)]
    colors = ["#34495e", "#e74c3c", "#3394d4", "#2ecc71"]
    
    for idx, col in zip(sample_indices, colors):
        axes[1].plot(
            probe_xs, density_xt[idx], 
            color=col, lw=1.8, 
            label=f"$t = {times[idx]:.2f}$"
        )
    
    axes[1].set_xlabel("Spatial Coordinate $x$", fontsize=11)
    axes[1].set_ylabel(r"Probability Density $\langle|\psi|^2\rangle_y$", fontsize=11)
    axes[1].set_title("B: Spatial Profiles over Time", fontsize=12, fontweight="bold")
    axes[1].set_xlim(probe_xs[0], probe_xs[-1])
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL 3: Quantifying the Exponential Tail (Lieb-Robinson Leakage)
    # ═════════════════════════════════════════════════════════════════════════
    axes[2].semilogy(x_rel, profile + 1e-10, "k-", lw=2, label=f"FEM Data ($t={t_slice:.2f}$)")
    axes[2].axvline(0, color="r", linestyle="--", lw=1.5, label="Cone Edge ($x_0 + v_{LR}t$)")
    
    if mu is not None and slope is not None:
        x_fit = x_rel[outside]
        axes[2].semilogy(
            x_fit,
            np.exp(intercept + slope * x_fit),
            "b--", lw=2, label=fr"Fit: $e^{{-\mu x}}$ ($\mu={mu:.2f}$)"
        )
    
    axes[2].set_xlabel(r"Distance from Wavefront Edge: $x - (x_0 + v_{LR}\,t)$", fontsize=11)
    axes[2].set_ylabel(r"Log Probability Density $\langle|\psi|^2\rangle_y$", fontsize=11)
    axes[2].set_title("C: Exponential Leakage Tail", fontsize=12, fontweight="bold")
    axes[2].set_xlim(x_rel.min(), x_rel.max())
    axes[2].set_ylim(1e-7, density_xt.max() * 2)
    axes[2].legend(loc="upper right", frameon=True)
    axes[2].grid(True, which="both", linestyle="--", alpha=0.5)

    # Supertitle layout adjustments
    plt.suptitle(
        f"Verification of Lieb-Robinson Bounds — 2D Particle in a Box (Spectral Engine)\n"
        f"Domain $L={solver.L}$, Mesh $N={solver.N}$ | Resolved Modes: {len(energies)} | "
        fr"Extracted Bounds: $v_{{LR}}={v_LR:.3f}$, $\mu={mu:.3f}$",
        fontsize=13, fontweight="bold", y=0.98
    )
    
    plt.tight_layout()
    
    label = "spectral_enhanced"
    outpath = os.path.join(output_dir, f"lieb_robinson_{label}.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SUCCESS] Enhanced visualization saved to {outpath}")

    psi_eval_func = fem.Function(solver.V, name="Probability_Density")
    cone_func     = fem.Function(solver.V, name="Lieb_Robinson_Cone")

    dof_coords = solver.V.tabulate_dof_coordinates()
    x_dofs     = dof_coords[:, 0]

    # Initialize the VTK file exporter
    vtk_path = os.path.join(output_dir, "wavepacket/wavepacket_evolution.pvd")
    with io.VTKFile(solver.msh.comm, vtk_path, "w") as vtk:
        
        print("  Exporting snapshots for ParaView...")
        for ti, t in enumerate(times):
            # Calculate the complex wavefunction array for this specific time
            psi_t = snapshots[ti] 
            
            # Calculate the physical probability density: |psi|^2
            rho_t = np.abs(psi_t)**2
            
            # Assign the calculated real densities to the FEniCSx function vector
            psi_eval_func.x.array[:] = rho_t

            cone_radius = v_LR * t
            inside_cone = np.abs(x_dofs - x0) <= cone_radius
            
            # Assign 1.0 to DOFs inside the cone, 0.0 to those outside
            cone_field = np.where(inside_cone, 1.0, 0.0)
            cone_func.x.array[:] = cone_field
            
            # Write the timestep to the VTK file series
            vtk.write_function(psi_eval_func, t)
            vtk.write_function(cone_func, t)

    print(f"  [SUCCESS] ParaView compatible file saved to {vtk_path}")


    return v_LR, v_analytic, mu