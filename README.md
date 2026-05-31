# FEniCS4Schroedinger

An FEM-based simulation using **FEniCSx**, **SLEPc**, and **PETSc** to investigate the quantum dynamics of a 2D wavepacket. This project explores Lieb-Robinson bounds in the context of a particle in a 2D box with fixed boundary conditions. 

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.


## Particle in 2D Box
When a localized Gaussian wavepacket is initialized at the exact center of a hard-wall square domain ($V=0$, Homogeneous Dirichlet boundaries), it expands as an isotropic ring. Upon impacting all four walls simultaneously, symmetric reflections interfere constructively to form a sharp, quantized $3 \times 3$ grid pattern of probability density hot-spots.

<p align="center">
  <video src="media/particle_in_box.mp4" width="500" autoplay loop muted></video>
</p>

## Lieb-Robinson Bounds for particle in 2D Box
By breaking geometric symmetry (placing the initial wavepacket off-center at $x_0 = -L/4$), we isolate a clear, directional wavefront corridor. To verify the **Lieb-Robinson Speed Limit ($v_{LR}$)**, the simulation uses a dynamic visibility threshold mask (left video) alongside a twin boundary-contour tracker (right video). Information is strictly prohibited from leaking beyond the linear expanding light cone wall, except for a minor exponentially suppressed tail.

<p align="center">
  <video src="media/particle2_formatted.mp4" width="600" autoplay loop muted></video>
  <video src="media/LR_inbox_formated.mp4" width="600" autoplay loop muted></video> 
</p>


## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.
## Requirements & Installation

This project is built to run smoothly inside a **Linux/WSL2** environment

### Dependencies
* Python 3.10+
* FEniCSx (`dolfinx`)
* `mpi4py`, `petsc4py`, `slepc4py`
* `numpy`, `matplotlib`

### Recommended Setup via Conda
The most reliable method to install the underlying C-libraries (PETSc/SLEPc/DOLFINx) alongside Python is via `conda-forge`:

### Create a dedicated environment with the FEniCSx stack
conda create -n fenics-env -c conda-forge python=3.10 dolfinx mpich petsc4py slepc4py matplotlib numpy

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

###  Simple test run of particle in a 2D box
python scripts.main or python3 scripts.main

### Run Lieb Robinson for a particle in a 2D box 
python scripts.run_lieb_robinson_study or python3 scripts.run_lieb_robinson_study

### Verify solver

Eigenvalue FEM: python scripts.run_verification or python3 scripts.run_verification

Time Evolution: python scripts.run_verification_time_evolution or python3 scripts.run_verification_time_evolution

### Convergence study
python scripts.run_convergence or python3 scripts.run_convergence
 
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


