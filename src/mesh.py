###################################################
#           Mesh Generation (Gmsh API)
###################################################

from mpi4py import MPI
from dolfinx import io
from dolfinx.io import gmsh as gmshio
import gmsh


def make_mesh(L: float, N: int):
    """
    Builds a square domain [-L/2, L/2]^2 using the Gmsh API
    and converts it directly into FEniCSx structures.

    Parameters
    ----------
    L : float  — full side length of the square domain
    N : int    — number of mesh divisions per side (controls element size h = L/N)

    Returns
    -------
    msh        : dolfinx Mesh
    facet_tags : MeshTags for boundary facets (tag=2 → Dirichlet wall)
    """

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    model = gmsh.model
    model.add("schrodinger_domain")

    x_min, x_max = -L / 2.0,  L / 2.0
    y_min, y_max = -L / 2.0,  L / 2.0

    # ── Geometry ──────────────────────────────────────────────────────────────
    rect = model.occ.addRectangle(x_min, y_min, 0, L, L)
    model.occ.synchronize()

    # ── Physical groups (must be after synchronize, before mesh.generate) ─────
    surfaces = model.getEntities(dim=2)
    assert len(surfaces) == 1, f"Expected 1 surface, got {len(surfaces)}"
    model.addPhysicalGroup(2, [surfaces[0][1]], tag=1)
    model.setPhysicalName(2, 1, "domain")

    boundaries   = model.getBoundary(surfaces, oriented=False, combined=True)
    boundary_ids = [b[1] for b in boundaries]
    model.addPhysicalGroup(1, boundary_ids, tag=2)
    model.setPhysicalName(1, 2, "wall")

    # ── Mesh sizing — h = L/N to match the create_rectangle N parameter ───────
    h = L / N
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)

    model.mesh.generate(2)
    model.mesh.optimize("Netgen")

    # ── Convert to FEniCSx ────────────────────────────────────────────────────
    msh_data = gmshio.model_to_mesh(
        model, MPI.COMM_WORLD, rank=0, gdim=2
    )

    gmsh.finalize()

    msh_data[0].name        = "schrodinger_domain"
    msh_data[2].name = "facet_tags"

    return msh_data[0], msh_data[2]