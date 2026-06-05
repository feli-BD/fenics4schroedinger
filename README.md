# FEniCS4Schroedinger

A FEM-based simulation using **FEniCSx**, **SLEPc**, and **PETSc** to investigate the quantum dynamics of a 2D wavepacket. This project explores Lieb-Robinson bounds in the context of a particle in a 2D box with fixed boundary conditions. 

## Quantum Stadium

A two-dimensional quantum stadium is simulated. Two symmetric high-momentum wavepackets ($|k_x| = 10.0$) are initialized and evolved over time. 
The curved semicircular caps break the spatial symmetry of the wavefunctions, rapidly scattering the localized packets into a complex, distributed quantum interference profile across the cavity.

![Particle in a Stadium Evolution](media/Result_GIFS/stadium_2pkt_colliding_rm.gif)

### Replicate this simulation:
To run this exact multi-packet chaotic scattering configuration, execute the following command in your terminal:
```bash
python main.py --mesh_type stadium --num_packets 2 --L 10.0 --radius 5.0 --target_h 0.2 --nev 150 --T 5.0 --frames 120
```

## Requirements & Installation

This project is built to run smoothly inside a **Linux/WSL2** environment

### Dependencies
* Python 3.10+
* FEniCSx (`dolfinx`)
* `mpi4py`, `petsc4py`, `slepc4py`
* `numpy`, `matplotlib`

### Verify the stack is working
    python -c "import dolfinx, gmsh, petsc4py, mpi4py; print('FEniCSx stack ok')"

### Recommended Setup via Conda
The most reliable method to install the underlying C-libraries (PETSc/SLEPc/DOLFINx) alongside Python is via `conda-forge`:

```bash
# Create the environment from the file
conda env create -f environment.yml

# Activate the dedicated environment workspace
conda activate fenics-env
```

## Structure
```text
FEniCS4Schroedinger/
├── src/
│   ├── __init__.py
│   ├── mesh.py                             # Handles square mesh generation and facet tagging
│   ├── solver.py                           # FEniCSx Matrix assembly & SLEPc eigenproblem engine 
│   └── lieb_robinson.py                    # Light-cone extraction, polyfitting, and 1D profiling
├── scripts/
│   ├── __init__.py
│   ├── main.py                             # Runs the solver with a 2D square mesh with a spectral based time evolution
│   ├── run_convergence.py                  # Runs the convergence study with 4 different meshes
│   ├── run_lieb_robinson_study.py          # Runs the solver and also lieb_robinson.py extracting the light-cone
│   ├── run_verification_time_evolution.py  # Runs the verification of the time evolution comparing the numerical results to the analytical solution
│   └── run_verification.py                 # Runs the verification of the solver comparing the numerical results to the analytical solution
├── media/                                  # Animation from paraview
├── results/                                # Post-processed plots (.png) and ParaView streams (.pvd / .xmdf)
├── report/                                 # Short report of the project
└── README.md
```

## Usage
For more details see `scripts/README.md`.

### Main Time Evolution 

```bash
# Quick default run (Square domain, single wavepacket)
python3 -m scripts.main

# Run a circular domain configuration with a fine mesh and custom eigenvalue resolution
python3 -m scripts.main --mesh_type circular --radius 5.0 --target_h 0.12 --nev 200

# Run a high-energy dual-wavepacket collision inside a chaotic Bunimovich Stadium
python3 -m scripts.main --mesh_type stadium --num_packets 2 -L 6.0 --radius 3.0  --nev 250

### Run Lieb Robinson for a particle in a 2D box 
python3 -m scripts.run_lieb_robinson_study
```

### Verify solver
```bash
# Eigenvalue FEM: 
python3 -m scripts.run_verification

#Time Evolution: 
python3 -m scripts.run_verification_time_evolution
```

### Convergence study
```bash
python3 -m scripts.run_convergence
```

### The results
| Simulation Run  | Generation Script | Target Output Directory | What is saved |
| :--- | :--- | :--- | :--- | 
| Symmetric Particle in a box | main.py | eigenstates and particle_in_box_time_evol | The eigenstates are given as xdmf files and the particle_in_box_time is intended to be viewed as a pvd | 
| Lieb-Robinson Study | run_lieb_robinson_study.py | lieb_robinson and wavepacket | In Lieb_Robinson/ three matplotlib plots are saved and in wavepacket/ there is a pvd file for the time evolution but also for the lieb-robinson light cone evolution in time. | 
| Verfication eigenvalue solver | run_verification.py | verification | Saves a matplotlib plot  | 
| Verfication time evolution | run_verification_time_evolution.py | time_verification | Saves a matplotlib plot  |
| Convergenve study | run_convergence.py | convergence | Saves a matplotlib plot  |

Visualizsation of results can either be a matplotlib plot or xdmf / pvd files that can be viewed in paraview. For the pvd files in paraview select probability_density.


## Authors and acknowledgment
Felicia Erb

Special thanks to the FEniCSx and SLEPc development communities and to the MSSD Lecturers: Dr. Lambert Theisen and Dr. Georgii Oblapenko.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


