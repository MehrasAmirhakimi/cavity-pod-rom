"""Vectorised stream function-vorticity solver for the lid-driven cavity.

This module reproduces the numerical scheme of the original scripts
``legacy/CavityFlow_CSR_er_Temp2.py`` (transient) and
``legacy/CavityFlow_Steady_CSR_er_Temp2.py`` (steady):

* uniform grid with ``im`` intervals per side (``im + 1`` nodes, ``dx = 1/im``);
* vorticity transport with implicit first-order upwind convection and
  central diffusion (backward Euler in time for the transient case);
* wall vorticity from Thom's formula, lagged inside the Picard loop;
* stream function from the 5-point Poisson equation ``lap(psi) = -omega``;
* velocities ``u = dpsi/dy``, ``v = -dpsi/dx`` by central differences;
* Picard loop with under-relaxation (0.2 by default) until the largest
  relative change of omega, psi, u and v is below the tolerance;
* passive temperature (one-way coupling), solved once per time step.

The only change is in how the matrices are built: the Python loops are
replaced by NumPy array operations. The coefficient formulas are written in
the same order as in the original code, so the assembled matrices are
identical bit for bit (see ``tests/test_regression.py``).

Two linear solvers are offered:

``linear_solver="gmres"``
    The original call ``scipy.sparse.linalg.gmres(A, b, x0)`` with SciPy's
    default tolerance (relative residual 1e-5). Use this to reproduce the
    original results.
``linear_solver="direct"``
    Sparse LU. It solves the same discrete equations to machine precision.
    The stream-function matrix is constant, so it is factorised once.

``scheme="central"`` replaces the upwind convection by central differences.
It is not part of the original code. It is used only as a diagnostic, to
separate the reduced-order-model error from the upwind/central modelling gap.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
@dataclass
class Grid:
    """Uniform cavity grid, as in the original code (``H = 1``, ``im = jm``)."""

    im: int
    H: float = 1.0
    jm: int = field(init=False)
    dx: float = field(init=False)
    dy: float = field(init=False)
    beta: float = field(init=False)
    n: int = field(init=False)      # interior nodes per side
    N: int = field(init=False)      # interior unknowns (omega, psi, u, v)
    N2: int = field(init=False)     # temperature unknowns (with boundary nodes)

    def __post_init__(self):
        self.jm = self.im
        self.dx = self.H / self.im
        self.dy = self.H / self.jm
        self.beta = self.dx / self.dy
        self.n = self.im - 1
        self.N = (self.im - 1) * (self.jm - 1)
        self.N2 = (self.im + 1) * (self.jm + 1)
        idx = np.arange(self.N)
        self.inode = idx % (self.im - 1)          # x index of interior node (0..n-1)
        self.jnode = idx // (self.im - 1)         # y index of interior node (0..n-1)
        idx2 = np.arange(self.N2)
        self.inode2 = idx2 % (self.im + 1)
        self.jnode2 = idx2 // (self.im + 1)
        self.x = np.linspace(0, self.H, self.im + 1)   # node coordinates incl. walls
        self.y = np.linspace(0, self.H, self.jm + 1)

    # reshape helpers: interior vector <-> (n, n) array indexed [j, i]
    def to2d(self, f):
        return np.reshape(f, (self.jm - 1, self.im - 1))

    def full_field(self, f, wall=0.0):
        """Interior vector -> (jm+1, im+1) array with constant wall value."""
        F = np.full((self.jm + 1, self.im + 1), wall, dtype=float)
        F[1:-1, 1:-1] = self.to2d(f)
        return F


# ---------------------------------------------------------------------------
# Matrix assembly (vectorised)
# ---------------------------------------------------------------------------
def _csr(grid_N, rows, cols, data):
    # Same construction path as the original code: COO -> CSR (canonical order).
    return sp.csr_matrix((data, (rows, cols)), shape=(grid_N, grid_N))


def _five_point(g: Grid, diag, cE, cW, cN, cS):
    """Assemble a 5-point matrix on the interior nodes (walls eliminated)."""
    i = np.arange(g.N)
    hasE = g.inode < g.im - 2
    hasW = g.inode > 0
    hasN = g.jnode < g.jm - 2
    hasS = g.jnode > 0
    rows = np.concatenate([i, i[hasE], i[hasW], i[hasN], i[hasS]])
    cols = np.concatenate([i, i[hasE] + 1, i[hasW] - 1,
                           i[hasN] + (g.im - 1), i[hasS] - (g.im - 1)])
    data = np.concatenate([diag, cE[hasE], cW[hasW], cN[hasN], cS[hasS]])
    return _csr(g.N, rows, cols, data)


def vorticity_system(g: Grid, Re, u, v, psi, omega_prev, dt=None,
                     scheme="upwind", saiwall=0.0):
    """Matrix and right-hand side of the vorticity equation.

    ``dt=None`` gives the steady equations of the steady script.
    ``u, v, psi`` are the lagged Picard iterates.
    """
    dx, dy, beta = g.dx, g.dy, g.beta
    mxu, mnu = np.maximum(u, 0), np.maximum(-u, 0)
    mxv, mnv = np.maximum(v, 0), np.maximum(-v, 0)
    if scheme == "upwind":
        if dt is not None:   # transient script, same expressions and order
            diag = 1 + dt / dx * (mxu + mnu + beta * mxv + beta * mnv
                                  + 2 / (Re * dx) + 2 * beta ** 2 / (Re * dx))
            aE = dt / dx * mnu + dt / (Re * dx ** 2)
            aW = dt / dx * mxu + dt / (Re * dx ** 2)
            aN = beta * dt / dx * mnv + beta ** 2 * dt / (Re * dx ** 2)
            aS = beta * dt / dx * mxv + beta ** 2 * dt / (Re * dx ** 2)
            b = omega_prev.copy()
        else:                # steady script
            diag = (mxu + mnu + beta * mxv + beta * mnv
                    + 2 / (Re * dx) + 2 * beta ** 2 / (Re * dx))
            aE = mnu + 1 / (Re * dx)
            aW = mxu + 1 / (Re * dx)
            aN = beta * mnv + beta ** 2 / (Re * dx)
            aS = beta * mxv + beta ** 2 / (Re * dx)
            b = np.zeros(g.N)
    elif scheme == "central":
        s = dt / dx if dt is not None else 1.0         # steady rows are scaled by dx
        d = dt / (Re * dx ** 2) if dt is not None else 1 / (Re * dx)
        aE = s * (-u / 2) + d
        aW = s * (u / 2) + d
        aN = s * beta * (-v / 2) + beta ** 2 * d
        aS = s * beta * (v / 2) + beta ** 2 * d
        diag = np.full(g.N, (1.0 if dt is not None else 0.0) + 2 * d + 2 * beta ** 2 * d)
        b = omega_prev.copy() if dt is not None else np.zeros(g.N)
    else:
        raise ValueError(scheme)

    A = _five_point(g, diag, -aE, -aW, -aN, -aS)
    # wall vorticity (Thom), added in the same order as the original loop
    L, R = g.inode == 0, g.inode == g.im - 2
    B, T = g.jnode == 0, g.jnode == g.jm - 2
    b[L] += 2 / (dx ** 2) * aW[L] * (saiwall - psi[L])
    b[R] += 2 / (dx ** 2) * aE[R] * (saiwall - psi[R])
    b[B] += 2 / (dy ** 2) * aS[B] * (saiwall - psi[B])
    b[T] += 2 / (dy ** 2) * aN[T] * (saiwall - psi[T] - 1 * dy)
    return A, b


def poisson_system(g: Grid, omega, saiwall=0.0):
    """Matrix and RHS of ``lap(psi) = -omega`` (rows scaled by dx**2)."""
    beta = g.beta
    one = np.ones(g.N)
    A = _five_point(g, -2 * (1 + beta ** 2) * one, one, one,
                    beta ** 2 * one, beta ** 2 * one)
    b = -g.dx ** 2 * omega
    if saiwall != 0.0:
        L, R = g.inode == 0, g.inode == g.im - 2
        B, T = g.jnode == 0, g.jnode == g.jm - 2
        b[L] += -saiwall
        b[R] += -saiwall
        b[B] += -beta ** 2 * saiwall
        b[T] += -beta ** 2 * saiwall
    return A, b


def velocities(g: Grid, psi, saiwall=0.0):
    """Central-difference velocities at interior nodes."""
    P = g.full_field(psi, saiwall)
    v = -(P[1:-1, 2:] - P[1:-1, :-2]) / (2 * g.dx)
    u = (P[2:, 1:-1] - P[:-2, 1:-1]) / (2 * g.dy)
    return u.ravel(), v.ravel()


def temperature_system(g: Grid, Re, Pr, u, v, T_prev=None, dt=None,
                       btemp=None):
    """Temperature matrix and RHS on all (im+1)^2 nodes.

    Side walls adiabatic (T_0 = T_1), bottom T = 0, lid T = 1.
    """
    dx, beta, im = g.dx, g.beta, g.im
    i2, j2 = g.inode2, g.jnode2
    idx = np.arange(g.N2)
    left, right = i2 == 0, i2 == im
    inner_x = ~(left | right)
    bottom = inner_x & (j2 == 0)
    top = inner_x & (j2 == g.jm)
    inter = inner_x & ~(bottom | top)
    ip = (j2[inter] - 1) * (im - 1) + i2[inter] - 1
    uu, vv = u[ip], v[ip]
    mxu, mnu = np.maximum(uu, 0), np.maximum(-uu, 0)
    mxv, mnv = np.maximum(vv, 0), np.maximum(-vv, 0)
    k = Re * Pr
    if dt is not None:
        dg = 1 + dt / dx * (mxu + mnu + beta * mxv + beta * mnv
                            + 2 / (k * dx) + 2 * beta ** 2 / (k * dx))
        cE = -(dt / dx * mnu + dt / (k * dx ** 2))
        cW = -(dt / dx * mxu + dt / (k * dx ** 2))
        cN = -(beta * dt / dx * mnv + beta ** 2 * dt / (k * dx ** 2))
        cS = -(beta * dt / dx * mxv + beta ** 2 * dt / (k * dx ** 2))
    else:
        dg = (mxu + mnu + beta * mxv + beta * mnv
              + 2 / (k * dx) + 2 * beta ** 2 / (k * dx))
        cE = -mnu - 1 / (k * dx)
        cW = -(mxu + 1 / (k * dx))
        cN = -(beta * mnv + beta ** 2 / (k * dx))
        cS = -(beta * mxv + beta ** 2 / (k * dx))
    ii = idx[inter]
    rows = np.concatenate([idx[left], idx[left], idx[right], idx[right],
                           idx[bottom], idx[top], ii, ii, ii, ii, ii])
    cols = np.concatenate([idx[left], idx[left] + 1, idx[right], idx[right] - 1,
                           idx[bottom], idx[top], ii, ii + 1, ii - 1,
                           ii + (im + 1), ii - (im + 1)])
    nl, nr = left.sum(), right.sum()
    data = np.concatenate([np.ones(nl), -np.ones(nl), np.ones(nr), -np.ones(nr),
                           np.ones(bottom.sum()), np.ones(top.sum()),
                           dg, cE, cW, cN, cS])
    A = _csr(g.N2, rows, cols, data)
    if btemp is None:
        btemp = np.zeros(g.N2)
    btemp[top] = 1
    if dt is not None:
        btemp[inter] = T_prev[inter]
    return A, btemp


# ---------------------------------------------------------------------------
# Linear solvers
# ---------------------------------------------------------------------------
class _Linear:
    def __init__(self, kind):
        if kind not in ("gmres", "direct"):
            raise ValueError(kind)
        self.kind = kind
        self._psi_lu = None
        self.info = []      # GMRES info flags (0 = converged)

    def solve(self, A, b, x0=None):
        if self.kind == "gmres":
            if x0 is None:
                x = spla.gmres(A, b)
            else:
                x = spla.gmres(A, b, x0)
            self.info.append(x[1])
            return x[0]
        return direct_solve(A, b)

    def solve_psi(self, A, b, x0):
        if self.kind == "gmres":
            return self.solve(A, b, x0)
        if self._psi_lu is None:   # constant matrix: factorise once
            self._psi_lu = spla.splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A")
        return self._psi_lu.solve(b)


def direct_solve(A, b):
    """Sparse LU solve (SuperLU, minimum-degree ordering on A^T + A)."""
    return spla.splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A").solve(b)


def _rel(new, old):
    return np.sqrt(np.sum((new - old) ** 2) / np.sum(new ** 2))


def _picard(g, Re, lin, u, v, psi, omega, omega_prev, dt, scheme, tol,
            komega, ksai, max_iter, history=None):
    """One Picard loop (as in the original code). Returns the new state."""
    uold, vold, saiold, omegaold = u, v, psi, omega
    er, k = 1.0, 0
    while er > tol:
        k += 1
        A, b = vorticity_system(g, Re, uold, vold, saiold, omega_prev, dt, scheme)
        omeganew = lin.solve(A, b, omegaold)
        er_omega = _rel(omeganew, omegaold)
        omeganew = omegaold + komega * (omeganew - omegaold)

        Ap, bp = poisson_system(g, omeganew)
        sainew = lin.solve_psi(Ap, bp, saiold)
        er_sai = _rel(sainew, saiold)
        sainew = saiold + ksai * (sainew - saiold)

        unew, vnew = velocities(g, sainew)
        er_u, er_v = _rel(unew, uold), _rel(vnew, vold)
        er = max(er_omega, er_sai, er_u, er_v)
        if history is not None:
            history.append((er_omega, er_sai, er_u, er_v))
        omegaold, saiold, uold, vold = (omeganew.copy(), sainew.copy(),
                                        unew.copy(), vnew.copy())
        if k >= max_iter:
            raise RuntimeError(f"Picard loop did not converge in {max_iter} iterations "
                               f"(last change {er:.3e})")
    return uold, vold, saiold, omegaold, k


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
def run_transient(Re=100, Pr=7, im=129, tf=200.0, nm=1000, linear_solver="direct",
                  scheme="upwind", tol=1e-6, komega=0.2, ksai=0.2,
                  temperature=True, max_iter=100000, verbose=False):
    """Transient run from rest, as in CavityFlow_CSR_er_Temp2.py.

    Returns a dict with arrays of shape (nm+1, N) for omega, psi, u, v and
    (nm+1, N2) for T, plus Picard iteration counts and wall-clock times.
    """
    g = Grid(im)
    dt = tf / nm
    lin = _Linear(linear_solver)
    out = {k: np.zeros((nm + 1, g.N)) for k in ("omega", "psi", "u", "v")}
    if temperature:
        out["T"] = np.zeros((nm + 1, g.N2))
    iters = np.zeros(nm, int)
    t_step = np.zeros(nm)
    t_temp = np.zeros(nm)
    btemp = np.zeros(g.N2)
    history = []
    for n in range(nm):
        t0 = time.perf_counter()
        u, v, psi, omega, k = _picard(
            g, Re, lin, out["u"][n], out["v"][n], out["psi"][n], out["omega"][n],
            out["omega"][n], dt, scheme, tol, komega, ksai, max_iter, history)
        out["omega"][n + 1], out["psi"][n + 1] = omega, psi
        out["u"][n + 1], out["v"][n + 1] = u, v
        iters[n] = k
        t1 = time.perf_counter()
        t_step[n] = t1 - t0
        if temperature:
            At, btemp = temperature_system(g, Re, Pr, u, v, out["T"][n], dt, btemp)
            out["T"][n + 1] = lin.solve(At, btemp, out["T"][n])
            t_temp[n] = time.perf_counter() - t1
        if verbose and (n % verbose == 0):
            print(f"step {n+1:5d}  t={dt*(n+1):7.2f}  picard={k:4d}  {t_step[n]:.2f}s",
                  flush=True)
    out.update(grid=g, dt=dt, t=np.linspace(0, tf, nm + 1), picard_iters=iters,
               wall_flow=t_step, wall_temp=t_temp, history=np.array(history),
               gmres_info=np.array(lin.info), params=dict(
                   Re=Re, Pr=Pr, im=im, tf=tf, nm=nm, linear_solver=linear_solver,
                   scheme=scheme, tol=tol, komega=komega, ksai=ksai))
    return out


def run_steady(Re=100, Pr=7, im=129, linear_solver="direct", scheme="upwind",
               tol=1e-6, komega=0.2, ksai=0.2, temperature=True, init=None,
               max_iter=200000):
    """Steady run, as in CavityFlow_Steady_CSR_er_Temp2.py.

    ``init`` may be a dict with omega, psi, u, v to start the Picard loop
    (the original starts from rest; the converged answer is the same
    discrete solution, only the iteration count changes).
    """
    g = Grid(im)
    lin = _Linear(linear_solver)
    z = np.zeros(g.N)
    if init is None:
        u0, v0, p0, w0 = z, z.copy(), z.copy(), z.copy()
    else:
        u0, v0, p0, w0 = (init[k].copy() for k in ("u", "v", "psi", "omega"))
    history = []
    t0 = time.perf_counter()
    u, v, psi, omega, k = _picard(g, Re, lin, u0, v0, p0, w0, None, None, scheme,
                                  tol, komega, ksai, max_iter, history)
    out = dict(omega=omega, psi=psi, u=u, v=v, grid=g, picard_iters=k,
               history=np.array(history), wall=time.perf_counter() - t0)
    if temperature:
        At, bt = temperature_system(g, Re, Pr, u, v, dt=None)
        out["T"] = lin.solve(At, bt)
    out["gmres_info"] = np.array(lin.info)
    out["params"] = dict(Re=Re, Pr=Pr, im=im, linear_solver=linear_solver,
                         scheme=scheme, tol=tol, komega=komega, ksai=ksai)
    return out


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------
def wall_vorticity_field(g: Grid, omega, psi, saiwall=0.0):
    """(jm+1, im+1) vorticity field with Thom wall values, as in the original plots."""
    S = g.full_field(psi, saiwall)
    W = g.full_field(omega, 0.0)
    W[:, 0] = 2 / g.dx ** 2 * (saiwall - S[:, 1])
    W[:, -1] = 2 / g.dx ** 2 * (saiwall - S[:, -2])
    W[0, :] = 2 / g.dy ** 2 * (saiwall - S[1, :])
    W[-1, :] = 2 / g.dy ** 2 * (saiwall - S[-2, :] - 1 * g.dy)
    return W


def lid_nusselt(g: Grid, T):
    """Average lid Nusselt number.

    The original code prints ``Nuavg = mean(-(T_lid - T_below)/dy)``, which is
    negative because heat flows from the hot lid into the cavity. We return
    the positive value ``mean((T_lid - T_below)/dy)`` over the interior lid
    nodes; it is the original number with the sign flipped.
    """
    TT = np.reshape(T, (g.jm + 1, g.im + 1))
    return np.sum((TT[-1, 1:-1] - TT[-2, 1:-1]) / g.dy) / (g.im - 1)


def centrelines(g: Grid, u, v):
    """u on the vertical centreline x = 0.5 and v on the horizontal one y = 0.5.

    With an odd number of intervals (im = 129) the centreline falls between
    two grid lines, so we take the mean of the two neighbouring lines
    (linear interpolation). Wall values (no slip, lid u = 1) are included.
    """
    U = g.full_field(u)
    U[-1, :] = 1.0
    V = g.full_field(v)
    xc = 0.5 * g.H
    if g.im % 2 == 0:
        m = g.im // 2
        uc, vc = U[:, m], V[m, :]
    else:
        m = g.im // 2
        w = (xc - g.x[m]) / g.dx
        uc = (1 - w) * U[:, m] + w * U[:, m + 1]
        vc = (1 - w) * V[m, :] + w * V[m + 1, :]
    return g.y, uc, g.x, vc
