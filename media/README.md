## Technical Pipeline: How to Produce the Animations & GIFs

All simulation videos (`.mp4`) and animated graphics (`.gif`) in this repository follow a standardized, reproducible 3-step pipeline:

### Step 1: Execute the Finite Element Simulation
Run the appropriate Python script with your chosen parameters (see `./scripts/README.md`). This generates  data containers:
* Static eigenvalues and eigenstates are written to `.xdmf` (indexed XML metadata) and `.h5` (heavy binary data) formats.
* Time-evolution probability density sequences are serialized into a `.pvd` master collection file linked to underlying time-step snapshots.

##### Step 2: Render in ParaView Using Statefiles (`.pvsm`)
Pre-configured statefiles are provided in the `/Paraview_statefiles` directory.


1. Open **ParaView**.
2. Click `File` $\rightarrow$ `Load State...` and navigate to the `/Paraview_statefiles` folder.
3. Choose the specific `.pvsm` file matching your study scenario:
| Statefile Name | Target Data Output | Visualization Purpose |
| :--- | :--- | :--- |
| `Eigenstates_square.pvsm` | `results/eigenstates/*.xdmf` | Renders static, spatial probability distributions ($|\phi_n|^2$) of the isolated energy eigenstates to inspect structural node geometry. |
| `Time_Evolution_square.pvsm` | `results/time_evolution/*square*.pvd` | Maps the orthogonal boundary interactions, showing isotropic wave expansions transforming into quantized checkerboard patterns. |
| `Time_Evolution_circular.pvsm` | `results/time_evolution/*circular*.pvd` | Configures focused radial views optimized for inspecting the concentric, self-interfering wave reflection rings within the circular domain. |
| `Time_Evolution_stadium.pvsm` | `results/time_evolution/*stadium*.pvd` | Adjusts aspect ratios and color scaling thresholds to best illustrate dual-packet high-momentum collisions and chaotic mixing across the stadium geometry. |
| `Time_Evolution_LR.pvsm` | `results/time_evolution/*_LR*.pvd` | Sets up 2D and 1D sliced views of an asymmetric wavepacket launch to visually monitor the leading information wavefront over time. |
| `Time_Evolution_LR_withCone.pvsm` | `results/time_evolution/*_LR*.pvd` | Overlays the analytical light-cone boundary line and threshold limit masks directly onto the space-time dataset for instant verification of speed constraints. |

4. When prompted by the *Choose File Names* dialog, select **Search files under specified directory** or manually choose your newly generated data folder (`results/`).
5. *(Optional)* To eliminate raw unrendered background patches, go to the **Properties Panel**, scroll down to **View**, and set **Background** from `Single Color` to a solid **Black** or **White** to match your host layout.
6. Click `File` $\rightarrow$ `Save Animation...` to output an high-definition `.mp4` video.

### Step 3: Convert Video to Web-Optimized GIFs Using FFmpeg
To compile your exported video files into high-quality, web-friendly loops for markdown rendering without massive file bloat, execute the following command in your terminal:

```bash
ffmpeg -i input_simulation.mp4 -vf "fps=15,scale=700:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" output_loop.gif
```

