#################################################
#            FEniCSx Mesh Generation
#    Square, Circular, and Stadium Domains
#
##################################################

from mpi4py import MPI
from dolfinx import io
import gmsh
from dolfinx.io import gmshio as gmshio


def make_square_mesh(L: float, N: int):
    """Builds a square domain [-L/2, L/2]^2 using the Gmsh API."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    model = gmsh.model
    model.add("square_domain")

    x_min, y_min = -L / 2.0, -L / 2.0

    rect = model.occ.addRectangle(x_min, y_min, 0, L, L)
    model.occ.synchronize()

    surfaces = model.getEntities(dim=2)
    model.addPhysicalGroup(2, [surfaces[0][1]], tag=1)
    model.setPhysicalName(2, 1, "domain")

    boundaries = model.getBoundary(surfaces, oriented=False, combined=True)
    boundary_ids = [b[1] for b in boundaries]
    model.addPhysicalGroup(1, boundary_ids, tag=2)
    model.setPhysicalName(1, 2, "wall")

    h = L / N
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)

    model.mesh.generate(2)
    model.mesh.optimize("Netgen")

    msh_data = gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=2)
    gmsh.finalize()

    msh_data[0].name = "square_domain"
    msh_data[2].name = "facet_tags"
    return msh_data[0], msh_data[2]

def make_circular_mesh(radius: float, target_h: float):
    """Builds a circular quantum dot domain with radius R."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    model = gmsh.model
    model.add("circular_domain")

    # Draw disk centered at (0,0)
    disk = model.occ.addDisk(0.0, 0.0, 0.0, radius, radius)
    model.occ.synchronize()

    # Tag physical items
    surfaces = model.getEntities(dim=2)
    model.addPhysicalGroup(2, [surfaces[0][1]], tag=1)
    model.setPhysicalName(2, 1, "domain")

    boundaries = model.getBoundary(surfaces, oriented=False, combined=True)
    boundary_ids = [b[1] for b in boundaries]
    model.addPhysicalGroup(1, boundary_ids, tag=2)
    model.setPhysicalName(1, 2, "wall")

    gmsh.option.setNumber("Mesh.MeshSizeMax", target_h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", target_h)

    model.mesh.generate(2)
    model.mesh.optimize("Netgen")

    msh_data = gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=2)
    gmsh.finalize()

    msh_data[0].name = "square_domain"
    msh_data[2].name = "facet_tags"
    return msh_data[0], msh_data[2]

def make_stadium_mesh(L_straight: float, radius: float, target_h: float):
    """
    Builds a 2D Quantum Stadium domain using the Gmsh OpenCASCADE kernel.
    
    Parameters
    ----------
    L_straight : float — Length of the straight parallel walls
    radius     : float — Radius of the semicircular end caps
    target_h   : float — Target element mesh sizing h
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    model = gmsh.model
    model.add("stadium_domain")

    # 1. Geometry Creation via OpenCASCADE
    # Central rectangle: centered horizontally, stretching from -L_straight/2 to +L_straight/2
    rect = model.occ.addRectangle(-L_straight / 2.0, -radius, 0, L_straight, 2.0 * radius)
    
    # Left disk and Right disk
    disk_left = model.occ.addDisk(-L_straight / 2.0, 0.0, 0.0, radius, radius)
    disk_right = model.occ.addDisk(L_straight / 2.0, 0.0, 0.0, radius, radius)
    
    # Fuse them all together into a single continuous stadium surface
    # The first tuple (2, rect) is the tool/object, followed by the modification list
    fused_shape, _ = model.occ.fuse([(2, rect)], [(2, disk_left), (2, disk_right)])
    model.occ.synchronize()

    # 2. Tagging Physical Groups for FEniCSx
    # Get the resulting 2D surface tag
    surfaces = model.getEntities(dim=2)
    model.addPhysicalGroup(2, [surfaces[0][1]], tag=1)
    model.setPhysicalName(2, 1, "domain")

    # Extract the external 1D boundary wire wrapping around the entire stadium
    boundaries = model.getBoundary(surfaces, oriented=False, combined=True)
    boundary_ids = [b[1] for b in boundaries]
    model.addPhysicalGroup(1, boundary_ids, tag=2) # Tag 2 matches your Dirichlet wall logic
    model.setPhysicalName(1, 2, "wall")

    # 3. Mesh Generation & Optimization
    gmsh.option.setNumber("Mesh.MeshSizeMax", target_h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", target_h)

    model.mesh.generate(2)
    model.mesh.optimize("Netgen")

    # Convert to FEniCSx mesh objects
    msh_data = gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=2)
    gmsh.finalize()

    msh_data[0].name = "square_domain"
    msh_data[2].name = "facet_tags"
    return msh_data[0], msh_data[2]