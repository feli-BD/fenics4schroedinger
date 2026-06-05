###################################################
#           FEniCSx Solver
#           Time-independent + Crank-Nicolson
#           Particle in a box (V=0)
###################################################

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix

class Solver:

    def __init__(self, msh, facet_tags, L: float, N: int, nev: int = 100):
        """
        Parameters
        ----------
        L   : float — full side length of square domain [-L/2, L/2]^2
        N   : int   — mesh divisions per side, h = L/N
        nev : int   — number of eigenvalues for the eigenproblem solver
        """
        self.L   = L
        self.N   = N
        self.nev = nev
        self.msh = msh
        self.facet_tags = facet_tags

        self.V                       = fem.functionspace(self.msh, ("Lagrange", 1))
        self.bc, self.bc_dofs        = self._boundary_conditions()
        self.a_form, self.m_form     = self._variational_forms()


    # ── Boundary conditions ───────────────────────────────────────────────────
    def _boundary_conditions(self):
        boundary_dofs = fem.locate_dofs_topological(
            self.V,
            self.msh.topology.dim - 1,
            self.facet_tags.find(2)
        )
        bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, self.V)
        return bc, boundary_dofs

    # ── Variational forms — free particle V=0 ────────────────────────────────
    def _variational_forms(self):
        """
        Weak form of the free-particle Hamiltonian:
            a(u,v) = 1/2 * grad(u).grad(v) dx     [stiffness — kinetic only]
            m(u,v) = u v dx                         [mass]

        Strong form:  -1/2 * nabla^2 psi = E psi
        Analytic eigenvalues: E_{nx,ny} = pi^2/2 * (nx^2 + ny^2) / (L/2)^2
        """
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        a = 0.5 * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        m = u * v * ufl.dx

        return fem.form(a), fem.form(m)

    # ── Assemble A and M ──────────────────────────────────────────────────────
    def assemble(self):
        A = assemble_matrix(self.a_form, bcs=[self.bc])
        A.assemble()

        M = assemble_matrix(self.m_form, bcs=[self.bc])
        M.assemble()

        # Zero BC dofs in M to suppress spurious near-zero eigenvalues
        bc_dofs = np.asarray(self.bc_dofs, dtype=PETSc.IntType)
        diag    = M.getDiagonal()
        diag.setValues(bc_dofs, np.zeros(bc_dofs.size, dtype=PETSc.ScalarType))
        diag.assemble()
        M.setDiagonal(diag)
        M.assemblyBegin()
        M.assemblyEnd()

        return A, M

    # ── Eigenvalue solver ─────────────────────────────────────────────────────
    def solve_eigenproblem(self, A, M):
        """
        Solves A u = E M u (GHEP) for the nev smallest eigenvalues.
        Used for verification and for the spectral time evolution.
        """
        eps = SLEPc.EPS().create(MPI.COMM_WORLD)
        eps.setOperators(A, M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        eps.setTarget(0.0)
        eps.setDimensions(nev=self.nev)

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        ksp = st.getKSP()
        ksp.setType("preonly")
        ksp.getPC().setType("lu")

        eps.setFromOptions()
        eps.solve()

        n_conv = eps.getConverged()
        print(f"  Converged eigenvalues: {n_conv}")

        energies, eigvecs = [], []
        for i in range(min(self.nev, n_conv)):
            vr, vi = A.createVecs()
            lam    = eps.getEigenpair(i, vr, vi)
            energies.append(lam.real)
            eigvecs.append(np.array(vr))

        return np.array(energies), np.array(eigvecs)

    # ── Time evolution  ──────────────────────────────────────
    def solve_time_evolution(self, M, energies,eigvecs,psi0, T, frames):
        """
        Spectral evolution of

            i dψ/dt = H ψ

        using

            ψ(t) = Σ_n c_n exp(-i E_n t) φ_n

        Parameters
        ----------
        M         : PETSc mass matrix
        energies  : eigenvalues
        eigvecs   : eigenvectors returned by solve_eigenproblem
        psi0      : initial state
        T         : final time
        dt        : timestep
        save_every: save every N steps
        """
        vec_raw = M.createVecRight()
        vec_raw.array[:] = psi0

        vec_M_raw = M.createVecLeft()
        M.mult(vec_raw, vec_M_raw)

        norm_M = np.sqrt(
            np.real(np.vdot(vec_raw.array, vec_M_raw.array))
        )

        psi0 = psi0 / norm_M

        # ── 2. Projection onto eigenbasis ─────────────────────────────────────────
        vec_psi0       = M.createVecRight(); vec_psi0.array[:] = psi0
        vec_M_psi0     = M.createVecLeft();  M.mult(vec_psi0, vec_M_psi0)
        coeffs         = eigvecs @ vec_M_psi0.array
        # times = np.linspace(0, self.L / 2, 120)   # T ~ L/2 for propagation
        times = np.linspace(0, T, int(frames))   # T ~ L for full reflection

        def psi_at_time(t):
            phases = np.exp(-1j * energies * t)
            return (coeffs * phases) @ eigvecs

        print("  Computing spectral time evolution...")
        snapshots = np.array([psi_at_time(t) for t in times])

        return np.array(times), np.array(snapshots)