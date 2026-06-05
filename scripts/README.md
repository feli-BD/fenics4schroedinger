## Scripts and Execution Guide

The project's computational core is modularized inside the `scripts/` package directory. Below is the full configuration and execution workflow breakdown for each script file:

### 1. Primary Entrypoint: `scripts.main`
Handles the core framework execution pipeline: building requested meshes via Gmsh, assembling global FEM operators, solving the underlying SLEPc eigenproblem, and evaluating analytical spectral time-evolution.

#### Command-Line Arguments Reference Table
| Argument Flag | Data Type | Default Value | Allowed Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--mesh_type` | `str` | `"square"` | `"square"`, `"circular"`, `"stadium"` | Sets the geometric configuration and boundary kernel of the simulation workspace. |
| `--num_packets` | `int` | `1` | `1`, `2`, `3`, `4` | Determines the wavepacket initialization layout dictionary (adds shifted coordinates and opposing momentum components). |
| `--L` | `float` | `10.0` | Any positive `float` | Specifies the full horizontal side length of the square, or the straight line corridor distance for the stadium. |
| `--N` | `int` | `60` | Any positive `int` | Grid division segments per axis. Determines the triangulation resolution for structured domains ($\mathcal{O}(N^2)$ DOF). |
| `--nev` | `int` | `150` | Any positive `int` | Target eigenvalues / eigenmodes to solve and extract from the sparse stiffness matrix via the SLEPc Krylov-Schur solver. |
| `--radius` | `float` | `5.0` | Any positive `float` | Defines the boundary radius for circular disks or the semicircular end-caps bounding the stadium geometry. |
| `--target_h` | `float` | `0.2` | Any positive `float` | Target element spacing parameter ($h$) passed to Gmsh to govern unstructured Delaunay refinement grid density. |
| `--T` | `float` | `5.0` | Any positive `float` | The global physical time horizon limits for the analytical spectral propagation loop ($t \in [0, T]$). |
| `--frames` | `float` | `120` | Any positive value | Total number of uniform snapshot slices evaluated and written to the master `.pvd` visualization folder. |

#### Hardcoded Wavepacket Presets Reference Table

> [!WARNING]
> **Currently Unavailable CLI Arguments:** Modifying individual wavepacket parameters (such as structural widths $\sigma$, starting centers $(x_0, y_0)$, or directional momentum vectors $(k_x, k_y)$) directly from the terminal is **not yet supported**. These configurations are determined automatically inside `main.py` by switching the `--num_packets` count integer.

Because individual wavepacket controls are hardcoded, changing the `--num_packets` flag switches between the following explicit initial states:

| Packet Configuration | Wavepacket Identity | Center $(x_0, y_0)$ | Envelope Width ($\sigma$) | Momentum $(k_x, k_y)$ |
| :--- | :--- | :--- | :--- | :--- |
| **`--num_packets 1`** | Single Center | $(0.0, 0.0)$ | $0.4$ | $(2.0, 2.0)$ |
| **`--num_packets 2`** | Packet 1 (Left)<br>Packet 2 (Right) | $(-L/4.0, 0.0)$<br>$(L/4.0, 0.0)$ | $0.2$<br>$0.2$ | $(10.0, 0.0)$<br>$(-10.0, 0.0)$ |
| **`--num_packets 3`** | Packet 1 (Left)<br>Packet 2 (Right)<br>Packet 3 (Bottom) | $(-L/4.0, 0.0)$<br>$(L/4.0, 0.0)$<br>$(0.0, -L/4.0)$ | $0.2$<br>$0.2$<br>$0.2$ | $(10.0, 0.0)$<br>$(-10.0, 0.0)$<br>$(0.0, 10.0)$ |
| **`--num_packets 4`** | Packet 1 (Left)<br>Packet 2 (Right)<br>Packet 3 (Bottom)<br>Packet 4 (Top) | $(-L/4.0, 0.0)$<br>$(L/4.0, 0.0)$<br>$(0.0, -L/4.0)$<br>$(0.0, L/4.0)$ | $0.2$<br>$0.2$<br>$0.2$<br>$0.2$ | $(10.0, 0.0)$<br>$(-10.0, 0.0)$<br>$(0.0, 10.0)$<br>$(0.0, -10.0)$ |

#### Execution Examples:
* **Square Domain:** `python3 -m scripts.main --mesh_type square`
* **Circular Domain:** `python3 -m scripts.main --mesh_type circular`
* **Stadium Geometry (Chaotic Collision):** `python3 -m scripts.main --mesh_type stadium --num_packets 2 --T 10.0 --frames 500`

---

### 2. Analysis and Optimization Scripts

#### `scripts.run_lieb_robinson_study`
Computes the empirical information speed limit ($v_{\text{LR}}$) and out-of-cone leakage rates. It instantiates a single wavepacket non-symmetrically offset at $x_0 = -L/4$ to maximize the open physical propagation path.
```bash
python3 -m scripts.run_lieb_robinson_study
``` 

#### `scripts.run_convergence`
Executes an automated refinement loop across a scaling array of grid densities ($N = [20, 40, 60, 80, 100]$) to confirm the mathematical accuracy of the spatial discretizations.
```bash
python3 -m scripts.run_convergence
``` 

#### `scripts.run_verification.py` & `scripts.run_verification_time_evolution.py`
These runtime test configurations serve as validation sanity checks.
```bash
python3 -m scripts.run_verification
python3 -m scripts.run_verification_time_evolution
``` 

### The results
| Simulation Run  | Generation Script | Target Output Directory | What is saved |
| :--- | :--- | :--- | :--- | 
| Symmetric Particle in a box | main.py | eigenstates and particle_in_box_time_evol | The eigenstates are given as xdmf files and the particle_in_box_time is intended to be viewed as a pvd | 
| Lieb-Robinson Study | run_lieb_robinson_study.py | lieb_robinson and wavepacket | In Lieb_Robinson/ three matplotlib plots are saved and in wavepacket/ there is a pvd file for the time evolution but also for the lieb-robinson light cone evolution in time. | 
| Verfication eigenvalue solver | run_verification.py | verification | Saves a matplotlib plot  | 
| Verfication time evolution | run_verification_time_evolution.py | time_verification | Saves a matplotlib plot  |
| Convergenve study | run_convergence.py | convergence | Saves a matplotlib plot  |