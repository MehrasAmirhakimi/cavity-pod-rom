"""Stage 3: POD-Galerkin ROM of the passive temperature field, driven by the flow ROM.

Full solver (one-way coupling, solved after each flow step):

    (T^{n+1} - T^n)/dt = -u.grad(T) [upwind] + lap(T)/(Re Pr),   u = u^{n+1}.

Boundary conditions: lid T = 1, bottom T = 0, adiabatic side walls (T_0 = T_1).
We work with the interior nodes; the wall values are affine in the interior
values (copies for the side walls, constants top and bottom).

ROM:  T = T_ref + Theta b, velocities from the flow ROM u = u_ref + U a.
Upwind = central + |u|/(2dx) d2x T + |v|/(2dy) d2y T. The central part gives
constant, linear and bilinear (a x b) terms, computed offline. The |u| part is
hyper-reduced with DEIM, as in the flow ROM. Given a^{n+1}, each backward-Euler
step is a small linear q x q solve. The lid Nusselt number is linear in T, so
it is evaluated directly from b.
"""
import time

import numpy as np

from .operators import stencils
from .solver import Grid


def interior(g: Grid, T_full):
    """(k, N2) full temperature fields -> (k, N) interior values."""
    T_full = np.atleast_2d(T_full)
    return T_full.reshape(-1, g.jm + 1, g.im + 1)[:, 1:-1, 1:-1].reshape(len(T_full), g.N)


def extended_T(g: Grid, Tint, lid):
    """Interior temperature columns (N, k) -> (n+2, n+2, k) with wall values."""
    n, k = g.n, Tint.shape[1]
    W = np.zeros((n + 2, n + 2, k))
    C = Tint.reshape(n, n, k)
    W[1:-1, 1:-1] = C
    W[1:-1, 0] = C[:, 0]          # adiabatic: T_0 = T_1
    W[1:-1, -1] = C[:, -1]
    W[0, 1:-1] = 0.0              # bottom T = 0
    W[-1, 1:-1] = 1.0 if lid else 0.0
    return W


def temperature_terms(g, Tint, lid):
    return stencils(g, extended_T(g, Tint, lid))


def rhs_T(g, Tint, u, v, Re, Pr):
    """Upwind semi-discrete right-hand side for one interior field (for tests)."""
    s = temperature_terms(g, Tint[:, None], lid=True)
    s = {k: v_[:, 0] for k, v_ in s.items()}
    return (-u * s["dx"] - v * s["dy"] + np.abs(u) / (2 * g.dx) * s["d2x"]
            + np.abs(v) / (2 * g.dy) * s["d2y"] + s["lap"] / (Re * Pr))


def nusselt_rows(g):
    """Indices of the interior nodes next to the lid."""
    return np.arange(g.N)[g.jnode == g.jm - 2]


class ThermalROM:
    def __init__(self, g, Re, Pr, Theta, T_ref, flow_rom, nl_basis=None, nl_points=None):
        """nl_basis=None evaluates the upwind term on the whole grid (no DEIM)."""
        t0 = time.perf_counter()
        self.g, self.Re, self.Pr = g, Re, Pr
        self.Theta, self.T_ref = Theta, T_ref
        q = self.q = Theta.shape[1]
        fr = flow_rom
        r = fr.r
        u0, v0 = fr.ref["u"][:, 0], fr.ref["v"][:, 0]
        U, V = fr.mod["u"], fr.mod["v"]
        sr = {k: x[:, 0] for k, x in temperature_terms(g, T_ref[:, None], lid=True).items()}
        sm = temperature_terms(g, Theta, lid=False)
        k = 1.0 / (Re * Pr)
        Th = Theta.T
        # central convection + diffusion: c0 + Ca a + K b + sum_j a_j M_j b
        self.c0 = Th @ (-u0 * sr["dx"] - v0 * sr["dy"] + k * sr["lap"])
        self.Ca = Th @ (-(U * sr["dx"][:, None]) - (V * sr["dy"][:, None]))       # (q, r)
        self.K = Th @ (-(u0[:, None] * sm["dx"]) - (v0[:, None] * sm["dy"]) + k * sm["lap"])
        M = np.empty((r, q, q))
        for j in range(r):
            M[j] = Th @ (-(U[:, j:j + 1] * sm["dx"]) - (V[:, j:j + 1] * sm["dy"]))
        self.M = M
        # DEIM for the upwind dissipation |u| d2x T/(2dx) + |v| d2y T/(2dy)
        if nl_basis is None:                      # full-grid evaluation
            p = np.arange(g.N)
            self.G = Th
        else:
            p = np.asarray(nl_points)
            self.G = Th @ nl_basis @ np.linalg.inv(nl_basis[p, :])                 # (q, m)
        self.p = p
        self.up0, self.Up, self.vp0, self.Vp = u0[p], U[p], v0[p], V[p]
        self.x0, self.X = sr["d2x"][p] / (2 * g.dx), sm["d2x"][p] / (2 * g.dx)
        self.y0, self.Y = sr["d2y"][p] / (2 * g.dy), sm["d2y"][p] / (2 * g.dy)
        # Nusselt number: Nu = mean((1 - T_top_row)/dy)
        rows = nusselt_rows(g)
        self.nu0 = np.mean((1 - T_ref[rows]) / g.dy)
        self.nu_b = -Theta[rows].mean(axis=0) / g.dy
        self.offline_time = time.perf_counter() - t0

    def integrate(self, b0, a_traj, dt):
        """Backward Euler driven by flow coefficients a_traj (nsteps+1, r)."""
        n = len(a_traj) - 1
        b = np.array(b0, float)
        out = np.empty((n + 1, self.q))
        out[0] = b
        I = np.eye(self.q)
        t0 = time.perf_counter()
        for i in range(n):
            a = a_traj[i + 1]
            au = np.abs(self.up0 + self.Up @ a)
            av = np.abs(self.vp0 + self.Vp @ a)
            c = self.c0 + self.Ca @ a + self.G @ (au * self.x0 + av * self.y0)
            K = (self.K + np.tensordot(a, self.M, axes=1)
                 + self.G @ (au[:, None] * self.X + av[:, None] * self.Y))
            b = np.linalg.solve(I - dt * K, b + dt * c)
            out[i + 1] = b
        return out, time.perf_counter() - t0

    def reconstruct(self, b_traj):
        return self.T_ref[None, :] + b_traj @ self.Theta.T

    def nusselt(self, b_traj):
        return self.nu0 + b_traj @ self.nu_b
