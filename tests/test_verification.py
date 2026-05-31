###################################################
#           tests/test_verification.py
###################################################

import numpy as np
import pytest
import sys
import os
from scipy.sparse import csr_matrix
from src.mesh import make_circular_mesh, make_square_mesh
from src.solver import Solver

# ── Mesh Unit Test ──────────────────────────────
def test_square_mesh_exists():
    msh, facet_tags = make_square_mesh(L=1.0, N=20)

    assert msh.topology.dim == 2
    assert facet_tags is not None

def test_circle_radius():
    R = 1.0
    msh, _ = make_circular_mesh(R, 0.1)

    coords = msh.geometry.x[:, :2]
    rmax = np.sqrt((coords**2).sum(axis=1)).max()

    assert np.isclose(rmax, R, atol=0.05)

def test_boundary_tag_exists():
    msh, facet_tags = make_square_mesh(1.0, 20)

    wall_facets = facet_tags.find(2)

    assert len(wall_facets) > 0

# ── Solver Unit Test ──────────────────────────────

def test_boundary_conditions_created():
    msh, facet_tags = make_square_mesh(1.0, 20)

    solver = Solver(msh, facet_tags, L=1.0, N=20)

    assert len(solver.bc_dofs) > 0


def test_boundary_dofs_on_boundary():
    msh, facet_tags = make_square_mesh(1.0, 20)

    solver = Solver(msh, facet_tags, L=1.0, N=20)

    coords = solver.V.tabulate_dof_coordinates()

    boundary_coords = coords[solver.bc_dofs]

    assert np.any(np.isclose(np.abs(boundary_coords[:,0]), 0.5))

def test_matrix_sizes():
    msh, facet_tags = make_square_mesh(1.0, 20)

    solver = Solver(msh, facet_tags, 1.0, 20)

    A, M = solver.assemble()

    assert A.getSize() == M.getSize()

def test_mass_diagonal_positive():
    msh, facet_tags = make_square_mesh(1.0, 20)

    solver = Solver(msh, facet_tags, 1.0, 20)

    _, M = solver.assemble()

    diag = M.getDiagonal().array

    assert np.all(diag >= 0)
