###################################################
#           Mesh Generation (Gmsh API)
###################################################

from mpi4py import MPI
from dolfinx import io
from dolfinx.io import gmsh as gmshio
import gmsh


def make_mesh(Lx: float = 20.0, Ly: float = 4.0, h: float = 0.2,
              geometry_type: str = "straight"):
    """
    Builds either a straight or constricted quantum wire mesh using the
    Gmsh OpenCASCADE kernel and converts it directly into FEniCSx structures.

    Parameters
    ----------
    Lx : float  — wire length along x
    Ly : float  — wire width along y
    h  : float  — target element size (controls mesh density)
    geometry_type : "straight" or "constricted"

    Returns
    -------
    msh        : dolfinx Mesh
    facet_tags : MeshTags for boundary facets (tag=2 → Dirichlet wall)
    """

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    model = gmsh.model
    model.add("quantum_wire")

    x_min, x_max = -Lx / 2.0, Lx / 2.0
    y_min, y_max = -Ly / 2.0,  Ly / 2.0

    # ── Geometry ──────────────────────────────────────────────────────────────
    if geometry_type == "straight":
        rect = model.occ.addRectangle(x_min, y_min, 0, Lx, Ly)
        model.occ.synchronize()

    elif geometry_type == "constricted":
        main_channel = model.occ.addRectangle(x_min, y_min, 0, Lx, Ly)

        w_cut = Ly * 0.30
        h_cut = Ly * 0.35

        cut_top = model.occ.addRectangle(-w_cut / 2, y_max - h_cut, 0, w_cut, h_cut)
        cut_bot = model.occ.addRectangle(-w_cut / 2, y_min,         0, w_cut, h_cut)

        # Boolean cut — outDimTags returned but not needed
        model.occ.cut([(2, main_channel)], [(2, cut_top), (2, cut_bot)])
        model.occ.synchronize()

    else:
        raise ValueError(f"Unknown geometry_type: {geometry_type!r}")

    # ── Physical groups ───────────────────────────────────────────────────────
    # Must be set AFTER synchronize() and BEFORE mesh generation

    surfaces = model.getEntities(dim=2)
    assert len(surfaces) == 1, f"Expected 1 surface, got {len(surfaces)}"
    model.addPhysicalGroup(2, [surfaces[0][1]], tag=1)
    model.setPhysicalName(2, 1, "domain")

    # getBoundary returns (dim, tag) pairs — oriented=False avoids sign issues
    boundaries   = model.getBoundary(surfaces, oriented=False, combined=True)
    boundary_ids = [b[1] for b in boundaries]
    model.addPhysicalGroup(1, boundary_ids, tag=2)
    model.setPhysicalName(1, 2, "wall")

    # ── Mesh sizing ───────────────────────────────────────────────────────────
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h * 0.5)   # allow refinement near corners

    # Optional: extra refinement at constriction corners
    if geometry_type == "constricted":
        gmsh.option.setNumber("Mesh.MeshSizeMin", h * 0.3)

    model.mesh.generate(2)
    model.mesh.optimize("Netgen")    # improves triangle quality

    # ── Convert to FEniCSx ────────────────────────────────────────────────────
    # model_to_mesh returns (msh, cell_tags, facet_tags)
    msh_data = gmshio.model_to_mesh(
        model, MPI.COMM_WORLD, rank=0, gdim=2
    )

    gmsh.finalize()

    msh_data[0].name        = "quantum_wire"
    msh_data[2].name = "facet_tags"

    return msh_data[0], msh_data[2]