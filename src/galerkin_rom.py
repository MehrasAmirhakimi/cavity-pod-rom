"""Intrusive POD-Galerkin reduced-order model of the vorticity equation.

State:      omega = omega_ref + Phi a,       Phi (N, r) orthonormal POD modes.
Stream fn:  psi   = psi_ref + Psi a,         Psi[:, k] solves lap(psi) = -Phi[:, k].
Walls:      Thom's formula is affine in psi. The lid term (-2/dy) belongs to
            omega_ref; the modes have homogeneous wall values. So boundary
            conditions hold for every a.

Projecting the central-difference right-hand side onto the modes gives

    da/dt = c + A a + B(a, a),      B(a, a)_i = sum_jk B_ijk a_j a_k.

c, A and B are computed once (offline). The online cost per step is O(r^3)
and does not depend on the grid size.

Time stepping uses backward Euler with the same dt as the full solver, so the
ROM and the full solver share the same time discretisation. Each step is
solved by Newton's method on the r x r system.

Optional DEIM term (Stage 3): the full solver's upwind dissipation
|u|/(2dx) d2x(omega) + |v|/(2dy) d2y(omega) is not polynomial. With DEIM it is
evaluated at m selected grid points only, so the ROM can reproduce the
upwind scheme without a full-grid evaluation online.
"""
import time

import numpy as np

from .operators import PoissonSolver, field_terms


class GalerkinROM:
    def __init__(self, g, Re, Phi, omega_ref, poisson=None):
        t0 = time.perf_counter()
        self.g, self.Re = g, Re
        self.Phi = np.ascontiguousarray(Phi)
        self.r = r = Phi.shape[1]
        self.omega_ref = omega_ref
        self.poisson = poisson or PoissonSolver(g)

        ref = field_terms(g, omega_ref[:, None], self.poisson, lid=True)
        mod = field_terms(g, self.Phi, self.poisson, lid=False)
        self.ref, self.mod = ref, mod
        u0, v0 = ref["u"], ref["v"]                      # (N, 1)
        # convective and viscous parts kept apart, so Re can be changed online
        self.c_conv = (Phi.T @ (-u0 * ref["dx"] - v0 * ref["dy"]))[:, 0]
        self.c_visc = (Phi.T @ ref["lap"])[:, 0]
        self.A_conv = Phi.T @ (-(u0 * mod["dx"] + mod["u"] * ref["dx"])
                               - (v0 * mod["dy"] + mod["v"] * ref["dy"]))
        self.A_visc = Phi.T @ mod["lap"]
        self.set_Re(Re)
        B = np.empty((r, r, r))
        for j in range(r):
            Q = -(mod["u"][:, j:j + 1] * mod["dx"] + mod["v"][:, j:j + 1] * mod["dy"])
            B[:, j, :] = Phi.T @ Q
        self.B = B
        self.deim = None
        self.offline_time = time.perf_counter() - t0

    # ------------------------------------------------------------------
    @classmethod
    def truncated(cls, full, r):
        """Cheap copy of a larger ROM keeping the first r modes (same basis order)."""
        new = cls.__new__(cls)
        new.g, new.Re, new.poisson = full.g, full.Re, full.poisson
        new.Phi, new.r, new.omega_ref = full.Phi[:, :r], r, full.omega_ref
        new.ref = full.ref
        new.mod = {k: v[:, :r] for k, v in full.mod.items()}
        new.c_conv, new.c_visc = full.c_conv[:r], full.c_visc[:r]
        new.A_conv, new.A_visc = full.A_conv[:r, :r], full.A_visc[:r, :r]
        new.B = full.B[:r, :r, :r]
        new.set_Re(full.Re)
        new.deim, new.offline_time = None, full.offline_time
        return new

    def set_Re(self, Re):
        """Re enters only through the viscous term, which is linear in 1/Re."""
        self.Re = Re
        self.c = self.c_conv + self.c_visc / Re
        self.A = self.A_conv + self.A_visc / Re

    # ------------------------------------------------------------------
    def add_deim(self, U_nl, points):
        """Add the hyper-reduced upwind dissipation term.

        U_nl: (N, m) POD basis of the dissipation-term snapshots.
        points: (m,) DEIM interpolation indices.
        """
        p = np.asarray(points)
        G = self.Phi.T @ U_nl @ np.linalg.inv(U_nl[p, :])      # (r, m), offline
        self._set_nonlinear(p, G)

    def add_upwind_full(self):
        """Upwind dissipation evaluated on the whole grid (no hyper-reduction).

        Exact Galerkin projection of the full solver's scheme. The online cost
        then grows with the grid size; it is the reference for DEIM.
        """
        self._set_nonlinear(np.arange(self.g.N), self.Phi.T)

    def _set_nonlinear(self, p, G):
        g = self.g
        self.deim = dict(
            p=p, G=G,
            u0=self.ref["u"][p, 0], U=self.mod["u"][p, :],
            v0=self.ref["v"][p, 0], V=self.mod["v"][p, :],
            x0=self.ref["d2x"][p, 0] / (2 * g.dx), X=self.mod["d2x"][p, :] / (2 * g.dx),
            y0=self.ref["d2y"][p, 0] / (2 * g.dy), Y=self.mod["d2y"][p, :] / (2 * g.dy),
        )

    def _deim_term(self, a):
        d = self.deim
        u = d["u0"] + d["U"] @ a
        v = d["v0"] + d["V"] @ a
        dx2 = d["x0"] + d["X"] @ a
        dy2 = d["y0"] + d["Y"] @ a
        f = np.abs(u) * dx2 + np.abs(v) * dy2
        J = (np.sign(u)[:, None] * d["U"] * dx2[:, None] + np.abs(u)[:, None] * d["X"]
             + np.sign(v)[:, None] * d["V"] * dy2[:, None] + np.abs(v)[:, None] * d["Y"])
        return d["G"] @ f, d["G"] @ J

    # ------------------------------------------------------------------
    def f(self, a):
        Ba = self.B @ a                          # (r, r): sum_k B_ijk a_k
        out = self.c + self.A @ a + Ba @ a
        J = self.A + Ba + np.einsum("ijk,j->ik", self.B, a)
        if self.deim is not None:
            fd, Jd = self._deim_term(a)
            out, J = out + fd, J + Jd
        return out, J

    def integrate(self, a0, dt, nsteps, newton_tol=1e-12, max_newton=30):
        """Backward Euler. Returns coefficients (nsteps+1, r), wall time, Newton counts."""
        a = np.array(a0, dtype=float)
        traj = np.empty((nsteps + 1, self.r))
        traj[0] = a
        I = np.eye(self.r)
        newton = np.zeros(nsteps, int)
        t0 = time.perf_counter()
        for n in range(nsteps):
            an = a.copy()
            for k in range(max_newton):
                fa, J = self.f(a)
                G = a - an - dt * fa
                da = np.linalg.solve(I - dt * J, -G)
                a = a + da
                if np.linalg.norm(da) <= newton_tol * (1 + np.linalg.norm(a)):
                    break
            else:
                raise RuntimeError(f"Newton did not converge at step {n}")
            if not np.all(np.isfinite(a)):
                raise FloatingPointError(f"ROM blew up at step {n}")
            newton[n] = k + 1
            traj[n + 1] = a
        return traj, time.perf_counter() - t0, newton

    # ------------------------------------------------------------------
    def reconstruct(self, traj, what="omega"):
        """Full fields (nsteps+1, N) for omega, psi, u or v."""
        ref = self.omega_ref if what == "omega" else self.ref[what][:, 0]
        basis = self.Phi if what == "omega" else self.mod[what]
        return ref[None, :] + traj @ basis.T

    def project(self, omega):
        """Coefficients of full fields (k, N) -> (k, r)."""
        return (omega - self.omega_ref[None, :]) @ self.Phi
