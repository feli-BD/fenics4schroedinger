#               0. Imports 
import os
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, io
from solver import Solver

def compute_lieb_robinson(solver, energies, eigvecs, output_dir, M):

    #    1. Get mesh coordinates 
    coords = solver.msh.geometry.x          # shape (n_nodes, 3) — z is always 0
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    #    2. Initial Gaussian wavepacket (localized on the left)  
    x0, y0 = -2.0, 0.0     # center of initial wavepacket
    sigma  = 1.2            # width

    psi0 = np.exp(-((x_coords - x0)**2 + (y_coords - y0)**2) / (2 * sigma**2))
    psi0 /= np.linalg.norm(psi0)

    #    3. Project onto eigenbasis: c_n = <phi_n | psi0>   
    # eigvecs shape: (n_modes, n_dofs)
    vec_psi0 = M.createVecRight()
    vec_psi0.array[:] = psi0
    
    vec_M_psi0 = M.createVecLeft()
    M.mult(vec_psi0, vec_M_psi0) # Computes M * psi0
    
    # Now project using the properly weighted array
    coeffs = eigvecs @ vec_M_psi0.array 

    print(f"  Projection completeness: {np.sum(np.abs(coeffs)**2):.4f}  (want ~1.0)")

    #    4. Time evolution     
    times = np.linspace(0, 10, 120)

    def psi_at_time(t):
        phases = np.exp(-1j * energies * t)
        return (coeffs * phases) @ eigvecs      # shape (n_dofs,)

    #    5. Probe points along x-axis (y ≈ 0)    
    probe_xs   = np.linspace(-4.5, 4.5, 60)
    probe_nodes = []
    for px in probe_xs:
        dists = (x_coords - px)**2 + (y_coords - 0.0)**2
        probe_nodes.append(np.argmin(dists))

    #    6. Build space-time density matrix       
    print("  Computing space-time density...")
    density_xt = np.zeros((len(times), len(probe_xs)))

    for ti, t in enumerate(times):
        psi_t = psi_at_time(t)
        rho   = np.abs(psi_t)**2
        for xi, node in enumerate(probe_nodes):
            density_xt[ti, xi] = rho[node]

    #    7. Estimate LR velocity from wavefront   
    # Find the furthest x where density exceeds a threshold at each time
    threshold  = 1e-4 * density_xt.max()
    front_x_right, front_x_left, front_t = [], [], []

    for ti, t in enumerate(times[1:], 1):
        row  = density_xt[ti]
        mask = row > threshold
        if mask.any():
            front_x_right.append(probe_xs[mask].max())   # rightward
            front_x_left.append(probe_xs[mask].min())    # leftward
            front_t.append(t)

    front_t       = np.array(front_t)
    front_x_right = np.array(front_x_right)
    front_x_left  = np.array(front_x_left)

    # Fit both fronts and average for a symmetric v_LR
    if len(front_t) > 3:
        v_right = np.polyfit(front_t, front_x_right, 1)[0]
        v_left  = np.polyfit(front_t, front_x_left,  1)[0]
        v_LR    = (v_right - v_left) / 2      # average of both arms
        print(f"  v_right = {v_right:.4f},  v_left = {v_left:.4f}")
        print(f"  Estimated LR velocity: v_LR = {v_LR:.4f}")

    #    8. Plot       
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Left: linear scale space-time density ---
    im0 = axes[0].imshow(
        density_xt, aspect="auto", origin="lower",
        extent=[probe_xs[0], probe_xs[-1], times[0], times[-1]],
        cmap="inferno"
    )
    plt.colorbar(im0, ax=axes[0], label=r"$|\psi(x,t)|^2$")
    t_arr = np.linspace(0, times[-1], 200)
    axes[0].plot(x0 + v_LR * t_arr, t_arr, "w--", lw=1.5, label=f"$v_{{LR}}={v_LR:.2f}$")
    axes[0].plot(x0 - v_LR * t_arr, t_arr, "w--", lw=1.5)
    # axes[0].scatter(front_x_right, front_t, s=4, c="cyan", label="wavefront x")
    # axes[0].scatter(front_x_left, front_t, s=4, c="cyan", label="wavefront y")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    axes[0].set_title("Space-time density")
    axes[0].legend(fontsize=8)

    # --- Middle: log scale (shows exponential decay outside cone) ---
    log_density = np.log10(density_xt + 1e-12)
    im1 = axes[1].imshow(
        log_density, aspect="auto", origin="lower",
        extent=[probe_xs[0], probe_xs[-1], times[0], times[-1]],
        cmap="viridis", vmin=-10, vmax=log_density.max()
    )
    plt.colorbar(im1, ax=axes[1], label=r"$\log_{10}|\psi|^2$")
    axes[1].plot(x0 + v_LR * t_arr, t_arr, "r--", lw=1.5, label="LR cone")
    axes[1].plot(x0 - v_LR * t_arr, t_arr, "r--", lw=1.5)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    axes[1].set_title("Log scale: decay outside cone")
    axes[1].legend(fontsize=8)

    # --- Right: decay profile at fixed time (slice through the cone) ---
    t_slice_idx = len(times) // 2
    t_slice     = times[t_slice_idx]
    profile     = density_xt[t_slice_idx]
    x_rel       = probe_xs - (x0 + v_LR * t_slice)   # distance from cone edge

    axes[2].semilogy(x_rel, profile + 1e-12, "k-", lw=1.5)
    axes[2].axvline(0, color="r", linestyle="--", label="LR cone edge")
    axes[2].set_xlabel(r"$x - (x_0 + v_{LR} t)$")
    axes[2].set_ylabel(r"$|\psi(x,t)|^2$  (log scale)")
    axes[2].set_title(f"Spatial profile at t = {t_slice:.1f}")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        f"Lieb-Robinson bounds — 2D harmonic oscillator\n"
        f"$\\omega={solver.omega}$,  $v_{{LR}}={v_LR:.3f}$,  "
        f"{len(energies)} eigenmodes",
        fontsize=11
    )
    plt.tight_layout()

    outpath = f"{output_dir}/lieb_robinson.png"
    plt.savefig(outpath, dpi=150)
    # plt.show()
    print(f"  Saved to {outpath}")