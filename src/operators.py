"""Semi-discrete operators of the vorticity transport equation.

The full solver advances

    (omega^{n+1} - omega^n) / dt = F_upwind(omega^{n+1}),

with omega on the interior nodes, psi from lap(psi) = -omega, central
velocities, and wall vorticity from Thom's formula. Here we write the
right-hand side as an explicit function of omega. Two forms are used:

* ``rhs_central``: central differences for convection. It is a quadratic
  polynomial in omega, so it can be projected exactly onto a POD basis
  (intrusive Galerkin ROM).
* ``rhs_upwind``: the full solver's first-order upwind form. It equals the
  central form plus a numerical-dissipation term
  ``|u|/(2 dx) d2x(omega) + |v|/(2 dy) d2y(omega)``, which is not polynomial
  (absolute values). It is used to check the full solver and for DEIM.
"""
import numpy as np
import scipy.sparse.linalg as spla

from .solver import Grid, poisson_system, velocities


class PoissonSolver:
    """Exact (LU) solve of lap(psi) = -omega with psi = 0 on the walls."""

    def __init__(self, g: Grid):
        A, _ = poisson_system(g, np.zeros(g.N))
        self.g = g
        self.lu = spla.splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A")

    def __call__(self, omega):
        omega = np.asarray(omega)
        b = -self.g.dx ** 2 * omega
        if b.ndim == 1:
            return self.lu.solve(b)
        return self.lu.solve(np.ascontiguousarray(b))


def velocities_many(g: Grid, Psi):
    """Velocities for each column of Psi (N, k). Walls psi = 0."""
    k = Psi.shape[1]
    P = np.zeros((g.n + 2, g.n + 2, k))
    P[1:-1, 1:-1, :] = Psi.reshape(g.n, g.n, k)
    v = -(P[1:-1, 2:] - P[1:-1, :-2]) / (2 * g.dx)
    u = (P[2:, 1:-1] - P[:-2, 1:-1]) / (2 * g.dy)
    return u.reshape(g.N, k), v.reshape(g.N, k)


def extended(g: Grid, Omega, Psi, lid):
    """Vorticity on (n+2, n+2, k) with Thom wall values (corners unused).

    ``lid=True`` adds the moving-lid term -2/dy on the top wall. It belongs
    to the reference (lifting) field only; POD modes use ``lid=False``.
    """
    k = Omega.shape[1]
    n = g.n
    W = np.zeros((n + 2, n + 2, k))
    W[1:-1, 1:-1] = Omega.reshape(n, n, k)
    P = Psi.reshape(n, n, k)
    W[1:-1, 0] = 2 / g.dx ** 2 * (0 - P[:, 0])
    W[1:-1, -1] = 2 / g.dx ** 2 * (0 - P[:, -1])
    W[0, 1:-1] = 2 / g.dy ** 2 * (0 - P[0, :])
    W[-1, 1:-1] = 2 / g.dy ** 2 * (0 - P[-1, :]) - (2 / g.dy if lid else 0.0)
    return W


def stencils(g: Grid, W):
    """Central first derivatives, second differences and Laplacian (N, k)."""
    C = W[1:-1, 1:-1]
    dxw = (W[1:-1, 2:] - W[1:-1, :-2]) / (2 * g.dx)
    dyw = (W[2:, 1:-1] - W[:-2, 1:-1]) / (2 * g.dy)
    d2x = W[1:-1, 2:] - 2 * C + W[1:-1, :-2]
    d2y = W[2:, 1:-1] - 2 * C + W[:-2, 1:-1]
    lap = d2x / g.dx ** 2 + d2y / g.dy ** 2
    k = W.shape[2]
    f = lambda a: a.reshape(g.N, k)  # noqa: E731
    return dict(dx=f(dxw), dy=f(dyw), d2x=f(d2x), d2y=f(d2y), lap=f(lap))


def field_terms(g: Grid, Omega, poisson: PoissonSolver, lid):
    """All linear quantities derived from vorticity columns (N, k)."""
    Omega = np.atleast_2d(Omega.T).T if Omega.ndim == 1 else Omega
    Psi = poisson(Omega)
    if Psi.ndim == 1:
        Psi = Psi[:, None]
    U, V = velocities_many(g, Psi)
    st = stencils(g, extended(g, Omega, Psi, lid))
    return dict(omega=Omega, psi=Psi, u=U, v=V, **st)


def rhs(g: Grid, omega, Re, poisson, scheme="central"):
    """Semi-discrete right-hand side F(omega) for one field."""
    t = field_terms(g, omega[:, None], poisson, lid=True)
    u, v = t["u"][:, 0], t["v"][:, 0]
    F = -u * t["dx"][:, 0] - v * t["dy"][:, 0] + t["lap"][:, 0] / Re
    if scheme == "upwind":
        F = F + upwind_dissipation(g, u, v, t["d2x"][:, 0], t["d2y"][:, 0])
    return F


def upwind_dissipation(g: Grid, u, v, d2x, d2y):
    """Numerical dissipation of first-order upwind (non-polynomial term)."""
    return np.abs(u) / (2 * g.dx) * d2x + np.abs(v) / (2 * g.dy) * d2y


def check_velocities(g: Grid, psi):
    """Same central velocities as the full solver (used in tests)."""
    return velocities(g, psi)
