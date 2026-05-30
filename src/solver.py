###################################################
#           FEniCSx Solver
###################################################

#           Imports
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
import gmsh
from mesh import make_mesh

    
class Solver:

    def __init__(self, L: float, N: int, omega: float = 0.5, nev: int = 100):
        self.L     = L
        self.N     = N
        self.omega = omega
        self.nev   = nev

        # Build mesh, function space, BCs, and forms on construction
        self.msh, self.facet_tags = make_mesh(self.L, self.N)
        self.V            = fem.functionspace(self.msh, ("Lagrange", 1))
        self.bc, self.bc_dofs = self._boundary_conditions()
        self.a_form, self.m_form = self._variational_forms()


    # ── Boundary conditions ───────────────────────────────────────────────────
    def _boundary_conditions(self):
        tdim = self.msh.topology.dim
        self.msh.topology.create_connectivity(tdim - 1, tdim)
        boundary_facets = mesh.exterior_facet_indices(self.msh.topology)
        boundary_dofs   = fem.locate_dofs_topological(
            self.V, tdim - 1, boundary_facets
        )
        bc = fem.dirichletbc(
            PETSc.ScalarType(0.0), boundary_dofs, self.V
        )
        return bc, boundary_dofs

    # ── Variational forms ─────────────────────────────────────────────────────
    def _variational_forms(self):
        x     = ufl.SpatialCoordinate(self.msh)
        # 2D harmonic oscillator:
        V_pot = 0.5 * self.omega**2 * (x[0]**2 + x[1]**2)
        

        # V0 = 20.0  # Barrier height
        # a = 1.0    # Lattice spacing

        # Periodic potential instead of harmonic trap
        # V_pot = V0 * (ufl.sin(np.pi * x[0] / a)**2 + ufl.sin(np.pi * x[1] / a)**2)


        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)          # was TrialFunction — typo fixed

        a = (0.5 * ufl.dot(ufl.grad(u), ufl.grad(v)) + V_pot * u * v) * ufl.dx
        m = u * v * ufl.dx

        return fem.form(a), fem.form(m)

    # ── Assemble ──────────────────────────────────────────────────────────────
    def assemble(self):
        A = fem.petsc.assemble_matrix(self.a_form, bcs=[self.bc])
        A.assemble()

        M = fem.petsc.assemble_matrix(self.m_form, bcs=[self.bc])
        M.assemble()

        # Zero the mass-matrix rows/cols for BC dofs so they don't
        # contribute spurious near-zero eigenvalues
        bc_dofs = np.asarray(self.bc_dofs, dtype=PETSc.IntType)
        diag    = M.getDiagonal()
        diag.setValues(bc_dofs, np.zeros(bc_dofs.size, dtype=PETSc.ScalarType))
        diag.assemble()
        M.setDiagonal(diag)
        M.assemblyBegin()
        M.assemblyEnd()

        return A, M

    # ── Solve ─────────────────────────────────────────────────────────────────
    def solve(self, A, M):
        eps = SLEPc.EPS().create(MPI.COMM_WORLD)
        eps.setOperators(A, M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        eps.setTarget(0.0)
        # eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eps.setDimensions(nev=self.nev)

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)   

        eps.setFromOptions()
        eps.solve()

        n_conv = eps.getConverged()
        print(f"Converged eigenvalues: {n_conv}")

        energies = []
        eigvecs  = []
        for i in range(min(self.nev, n_conv)):
            vr, vi = A.createVecs()
            lam = eps.getEigenpair(i, vr, vi)
            energies.append(lam.real)
            eigvecs.append(np.array(vr))

        return np.array(energies), np.array(eigvecs)