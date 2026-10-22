"""Consistency tests for the operators, POD and the Galerkin ROM (small grids)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import solver, operators, pod  # noqa: E402
from src.galerkin_rom import GalerkinROM  # noqa: E402


@pytest.fixture(scope="module")
def run17():
    return solver.run_transient(Re=100, Pr=7, im=17, tf=2.0, nm=10,
                                linear_solver="direct", temperature=False)


@pytest.mark.parametrize("scheme", ["upwind", "central"])
def test_full_solver_satisfies_semi_discrete_equation(scheme):
    """Converged backward-Euler steps satisfy (w1 - w0)/dt = F(w1)."""
    r = solver.run_transient(Re=100, im=17, tf=2.0, nm=10, scheme=scheme,
                             linear_solver="direct", temperature=False, tol=1e-12)
    g = r["grid"]
    ps = operators.PoissonSolver(g)
    for n in (1, 5, 10):
        lhs = (r["omega"][n] - r["omega"][n - 1]) / r["dt"]
        F = operators.rhs(g, r["omega"][n], 100, ps, scheme=scheme)
        assert np.linalg.norm(lhs - F) / np.linalg.norm(lhs) < 1e-8


def test_pod_svd_equals_method_of_snapshots(run17):
    X = (run17["omega"] - run17["omega"][-1]).T
    P1, s1 = pod.pod_svd(X)
    P2, s2 = pod.pod_snapshots(X)
    k = 5
    assert np.allclose(s1[:k], s2[:k], rtol=1e-8)
    assert np.allclose(np.abs(P1[:, :k].T @ P2[:, :k]), np.eye(k), atol=1e-6)


def test_galerkin_tensors_equal_projected_rhs(run17):
    """c + A a + B(a,a) must equal Phi^T F_central(omega_ref + Phi a) exactly."""
    g = run17["grid"]
    wref = run17["omega"][-1]
    X = (run17["omega"] - wref).T
    Phi, s = pod.pod_svd(X)
    Phi = Phi[:, :8]
    rom = GalerkinROM(g, 100, Phi, wref)
    rng = np.random.default_rng(0)
    for _ in range(3):
        a = rng.normal(size=8) * s[:8] / np.sqrt(X.shape[1])
        f_rom, J = rom.f(a)
        F = operators.rhs(g, wref + Phi @ a, 100, rom.poisson, scheme="central")
        assert np.allclose(f_rom, Phi.T @ F, rtol=1e-10, atol=1e-10 * np.abs(f_rom).max())
        # Jacobian against finite differences
        e = 1e-6 * np.linalg.norm(a)
        Jfd = np.column_stack([(rom.f(a + e * ei)[0] - rom.f(a - e * ei)[0]) / (2 * e)
                               for ei in np.eye(8)])
        assert np.allclose(J, Jfd, rtol=1e-5, atol=1e-6 * np.abs(J).max())


def test_deim_with_all_points_is_exact(run17):
    """With all grid points as DEIM points, the hyper-reduced term is exact."""
    g = run17["grid"]
    wref = run17["omega"][-1]
    Phi = pod.pod_svd((run17["omega"] - wref).T)[0][:, :6]
    rom = GalerkinROM(g, 100, Phi, wref)
    rom.add_deim(np.eye(g.N), np.arange(g.N))
    a = Phi.T @ (run17["omega"][3] - wref)
    f_rom, _ = rom.f(a)
    F = operators.rhs(g, wref + Phi @ a, 100, rom.poisson, scheme="upwind")
    assert np.allclose(f_rom, Phi.T @ F, rtol=1e-9, atol=1e-9 * np.abs(f_rom).max())


def test_reynolds_switch_matches_new_rom(run17):
    g = run17["grid"]
    wref = run17["omega"][-1]
    Phi = pod.pod_svd((run17["omega"] - wref).T)[0][:, :6]
    rom = GalerkinROM(g, 100, Phi, wref)
    rom.set_Re(130)
    ref = GalerkinROM(g, 130, Phi, wref)
    a = Phi.T @ (run17["omega"][4] - wref)
    assert np.allclose(rom.f(a)[0], ref.f(a)[0], rtol=1e-12, atol=1e-12)


def test_temperature_operator_and_thermal_rom():
    from src import thermal_rom as trm
    r = solver.run_transient(Re=100, Pr=7, im=17, tf=2.0, nm=10, linear_solver="direct",
                             tol=1e-12)
    g = r["grid"]
    Ti = trm.interior(g, r["T"])
    # 1) full solver satisfies the semi-discrete temperature equation
    for n in (1, 5, 10):
        lhs = (Ti[n] - Ti[n - 1]) / r["dt"]
        F = trm.rhs_T(g, Ti[n], r["u"][n], r["v"][n], 100, 7)
        assert np.linalg.norm(lhs - F) / np.linalg.norm(lhs) < 1e-8
    # 2) thermal ROM terms equal the projected right-hand side (all points as DEIM points)
    wref = r["omega"][-1]
    Phi = pod.pod_svd((r["omega"] - wref).T)[0][:, :5]
    fr = GalerkinROM(g, 100, Phi, wref)
    Theta = pod.pod_svd((Ti - Ti[-1]).T)[0][:, :6]
    tr = trm.ThermalROM(g, 100, 7, Theta, Ti[-1], fr, np.eye(g.N), np.arange(g.N))
    a = Phi.T @ (r["omega"][3] - wref)
    b = Theta.T @ (Ti[3] - Ti[-1])
    au = np.abs(tr.up0 + tr.Up @ a)
    av = np.abs(tr.vp0 + tr.Vp @ a)
    c = tr.c0 + tr.Ca @ a + tr.G @ (au * tr.x0 + av * tr.y0)
    K = tr.K + np.tensordot(a, tr.M, axes=1) + tr.G @ (au[:, None] * tr.X + av[:, None] * tr.Y)
    u = fr.ref["u"][:, 0] + fr.mod["u"] @ a
    v = fr.ref["v"][:, 0] + fr.mod["v"] @ a
    F = trm.rhs_T(g, Ti[-1] + Theta @ b, u, v, 100, 7)
    assert np.allclose(c + K @ b, Theta.T @ F, rtol=1e-9, atol=1e-9 * np.abs(Theta.T @ F).max())
    # 3) Nusselt number from b equals the full formula
    T_full = r["T"][3].copy()
    T_rec = Ti[-1] + Theta @ b
    nu_rom = tr.nusselt(b[None, :])[0]
    TT = T_full.reshape(g.jm + 1, g.im + 1)
    TT[1:-1, 1:-1] = T_rec.reshape(g.n, g.n)
    assert np.isclose(nu_rom, solver.lid_nusselt(g, TT.ravel()), rtol=1e-12)


def test_thermal_rom_full_grid_option_equals_identity_deim():
    from src import thermal_rom as trm
    r = solver.run_transient(Re=100, Pr=7, im=17, tf=2.0, nm=10, linear_solver="direct")
    g = r["grid"]
    Ti = trm.interior(g, r["T"])
    Phi = pod.pod_svd(r["omega"].T)[0][:, :5]
    fr = GalerkinROM(g, 100, Phi, np.zeros(g.N))
    Theta = pod.pod_svd(Ti.T)[0][:, :6]
    a = trm.ThermalROM(g, 100, 7, Theta, np.zeros(g.N), fr)
    b = trm.ThermalROM(g, 100, 7, Theta, np.zeros(g.N), fr, np.eye(g.N), np.arange(g.N))
    assert np.allclose(a.G, b.G) and np.allclose(a.K, b.K)
