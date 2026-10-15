"""Stage 1: intrusive POD-Galerkin ROM of the transient Re = 100 run.

Needs the full-order runs (see scripts/run_all.sh):
    data/fom_upwind_Re100_im129.npz, data/fom_central_Re100_im129.npz,
    data/fom_upwind_Re80_im129_t20.npz, data/fom_upwind_Re125_im129_t20.npz
Writes results/stage1.npz (all numbers and curves used in the figures).
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import operators, pod, solver  # noqa: E402
from src.galerkin_rom import GalerkinROM  # noqa: E402

D = lambda f: os.path.join(ROOT, "data", f)  # noqa: E731
NW = 100                      # snapshot window: steps 0..100 (t = 0..20)
R_LIST = [3, 5, 6, 8, 10, 15, 20, 30, 40]
R_MAX = max(R_LIST)
T_PROFILES = [1.0, 2.0, 5.0, 20.0]
R_PROFILES = [5, 10]
DT = 0.2


def timed_integrate(rom, a0, nsteps, repeats=5):
    best, out = np.inf, None
    for _ in range(repeats):
        tr, wt, nn = rom.integrate(a0, DT, nsteps)
        if wt < best:
            best, out = wt, (tr, nn)
    return out[0], best, out[1]


def rel_err(Wr, W):
    """Relative L2 error per time step (t > 0, since omega(0) = 0)."""
    return np.linalg.norm(Wr[1:] - W[1:], axis=1) / np.linalg.norm(W[1:], axis=1)


def main():
    res = {}
    g = solver.Grid(129)
    up = np.load(D("fom_upwind_Re100_im129.npz"))
    W, t = up["omega"], up["t"]
    ws = W[-1]
    res["t_full"] = t
    res["dist_to_steady"] = np.linalg.norm(W - ws, axis=1) / np.linalg.norm(ws)
    res["picard_iters"] = up["picard_iters"]
    res["last_step_change"] = np.linalg.norm(W[-1] - W[-2]) / np.linalg.norm(ws)
    res["fom_wall_flow_total"] = up["wall_flow"].sum()
    res["fom_wall_window"] = up["wall_flow"][:NW].sum()       # steps 1..NW
    res["fom_wall_full_run"] = float(up["total_wall"])
    tw = t[:NW + 1]
    res["t_window"] = tw

    # ---------------- POD -------------------------------------------------
    X = (W[:NW + 1] - ws).T
    Phi, s = pod.pod_svd(X)
    Phi2, s2 = pod.pod_snapshots(X)
    res["sv_check"] = np.max(np.abs(s[:10] - s2[:10]) / s[:10])
    res["s"] = s
    res["energy"] = pod.cumulative_energy(s)
    res["r_energy"] = np.array([pod.rank_for_energy(s, th) for th in (0.99, 0.999, 0.9999)])
    res["modes"] = Phi[:, :4].T.copy()
    nrm = np.linalg.norm(W[1:NW + 1], axis=1)
    res["proj_err"] = np.array([pod.projection_error(X[:, 1:], Phi, r) / nrm for r in R_LIST])

    # ---------------- ROMs ------------------------------------------------
    t0 = time.perf_counter()
    big = GalerkinROM(g, 100, Phi[:, :R_MAX], ws)
    res["offline_galerkin_s"] = time.perf_counter() - t0
    ps = big.poisson
    # nonlinear (upwind dissipation) snapshots and DEIM basis
    t0 = time.perf_counter()
    ft = operators.field_terms(g, W[:NW + 1].T, ps, lid=True)
    NL = operators.upwind_dissipation(g, ft["u"], ft["v"], ft["d2x"], ft["d2y"])
    Un, sn = pod.pod_svd(NL)
    res["offline_deim_s"] = time.perf_counter() - t0
    res["s_nl"] = sn
    a0 = Phi[:, :R_MAX].T @ (W[0] - ws)

    keys = ["central", "deim", "full_upwind"]
    err = {k: [] for k in keys}
    wall = {k: [] for k in keys}
    traj = {}
    deim_pts = {}
    for r in R_LIST:
        for k in keys:
            rom = GalerkinROM.truncated(big, r)
            if k == "deim":
                m = 2 * r
                p = pod.deim_points(Un[:, :m])
                rom.add_deim(Un[:, :m], p)
                deim_pts[r] = p
            elif k == "full_upwind":
                rom.add_upwind_full()
            reps = 1 if k == "full_upwind" else 5
            tr, wt, _ = timed_integrate(rom, a0[:r], NW, reps)
            err[k].append(rel_err(rom.reconstruct(tr), W[:NW + 1]))
            wall[k].append(wt)
            if r in R_PROFILES and k in ("central", "deim"):
                traj[(k, r)] = (rom, tr)
        print(f"r={r:2d}  central max {err['central'][-1].max():.2e}  "
              f"DEIM max {err['deim'][-1].max():.2e}  full {err['full_upwind'][-1].max():.2e}  "
              f"DEIM online {wall['deim'][-1]*1e3:.1f} ms", flush=True)
    for k in keys:
        res[f"err_{k}"] = np.array(err[k])
        res[f"wall_{k}"] = np.array(wall[k])
    res["r_list"] = np.array(R_LIST)
    res["deim_points_r10"] = deim_pts[10]

    # ---------------- centreline profiles ---------------------------------
    for (k, r), (rom, tr) in traj.items():
        U = rom.reconstruct(tr, "u")
        V = rom.reconstruct(tr, "v")
        for tp in T_PROFILES:
            n = int(round(tp / DT))
            y, uc, x, vc = solver.centrelines(g, U[n], V[n])
            res[f"prof_u_{k}_r{r}_t{tp:g}"] = uc
            res[f"prof_v_{k}_r{r}_t{tp:g}"] = vc
    for tp in T_PROFILES:
        n = int(round(tp / DT))
        y, uc, x, vc = solver.centrelines(g, up["u"][n], up["v"][n])
        res[f"prof_u_fom_t{tp:g}"], res[f"prof_v_fom_t{tp:g}"] = uc, vc
    res["prof_y"], res["prof_x"] = y, x
    res["t_profiles"] = np.array(T_PROFILES)

    # ---------------- diagnostic: central ROM vs central full solver -------
    ce = np.load(D("fom_central_Re100_im129.npz"))
    Wc = ce["omega"]
    wcs = Wc[-1]
    res["steady_gap_central_upwind"] = np.linalg.norm(wcs - ws) / np.linalg.norm(ws)
    res["psi_min_upwind"], res["psi_min_central"] = up["psi"][-1].min(), ce["psi"][-1].min()
    Xc = (Wc[:NW + 1] - wcs).T
    Phic, _ = pod.pod_svd(Xc)
    bigc = GalerkinROM(g, 100, Phic[:, :R_MAX], wcs, poisson=ps)
    a0c = Phic[:, :R_MAX].T @ (Wc[0] - wcs)
    nrmc = np.linalg.norm(Wc[1:NW + 1], axis=1)
    ec, pc = [], []
    for r in R_LIST:
        rom = GalerkinROM.truncated(bigc, r)
        tr, _, _ = rom.integrate(a0c[:r], DT, NW)
        ec.append(rel_err(rom.reconstruct(tr), Wc[:NW + 1]))
        pc.append(pod.projection_error(Xc[:, 1:], Phic, r) / nrmc)
    res["err_central_vs_centralfom"] = np.array(ec)
    res["proj_err_centralfom"] = np.array(pc)

    # ---------------- long-time run beyond the window (r = 10, DEIM) ------
    rom = GalerkinROM.truncated(big, 10)
    rom.add_deim(Un[:, :20], deim_pts[10])
    tr, _, _ = rom.integrate(a0[:10], DT, len(t) - 1)
    res["err_longtime_deim_r10"] = rel_err(rom.reconstruct(tr), W)
    romc = GalerkinROM.truncated(big, 10)
    tr, _, _ = romc.integrate(a0[:10], DT, len(t) - 1)
    res["err_longtime_central_r10"] = rel_err(romc.reconstruct(tr), W)

    # ---------------- other Reynolds numbers (same basis, Re changed) -----
    for Re in (80, 125):
        f = D(f"fom_upwind_Re{Re}_im129_t20.npz")
        if not os.path.exists(f):
            print("missing", f)
            continue
        WR = np.load(f)["omega"]
        nR = np.linalg.norm(WR[1:], axis=1)
        e, pe = [], []
        for r in R_LIST:
            rom = GalerkinROM.truncated(big, r)
            rom.add_deim(Un[:, :2 * r], deim_pts[r])
            rom.set_Re(Re)
            tr, _, _ = rom.integrate(a0[:r], DT, NW)
            e.append(rel_err(rom.reconstruct(tr), WR))
            pe.append(np.linalg.norm((WR[1:] - ws) - ((WR[1:] - ws) @ Phi[:, :r]) @ Phi[:, :r].T,
                                     axis=1) / nR)
        res[f"err_deim_Re{Re}"] = np.array(e)
        res[f"proj_err_Re{Re}"] = np.array(pe)

    # ---------------- held-out Re = 100 from a basis trained at Re = 80 and 125 ----
    f80, f125 = D("fom_upwind_Re80_im129_t20.npz"), D("fom_upwind_Re125_im129_t20.npz")
    if os.path.exists(f80) and os.path.exists(f125):
        W80, W125 = np.load(f80)["omega"], np.load(f125)["omega"]
        ref = 0.5 * (W80[-1] + W125[-1])
        X2 = np.hstack([(W80 - ref).T, (W125 - ref).T])
        Phi2r, _ = pod.pod_svd(X2)
        big2 = GalerkinROM(g, 100, Phi2r[:, :R_MAX], ref, poisson=ps)
        ft2 = operators.field_terms(g, np.hstack([W80.T, W125.T]), ps, lid=True)
        Un2, _ = pod.pod_svd(operators.upwind_dissipation(g, ft2["u"], ft2["v"],
                                                           ft2["d2x"], ft2["d2y"]))
        a02 = Phi2r[:, :R_MAX].T @ (0 - ref)
        W100 = W[:NW + 1]
        e, pe = [], []
        for r in R_LIST:
            rom = GalerkinROM.truncated(big2, r)
            rom.add_deim(Un2[:, :2 * r], pod.deim_points(Un2[:, :2 * r]))
            rom.set_Re(100)
            tr, _, _ = rom.integrate(a02[:r], DT, NW)
            e.append(rel_err(rom.reconstruct(tr), W100))
            P = Phi2r[:, :r]
            pe.append(np.linalg.norm((W100[1:] - ref) - ((W100[1:] - ref) @ P) @ P.T, axis=1)
                      / nrm)
        res["err_deim_Re100_from_80_125"] = np.array(e)
        res["proj_err_Re100_from_80_125"] = np.array(pe)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    np.savez(os.path.join(ROOT, "results", "stage1.npz"), **res)
    print("saved results/stage1.npz")


if __name__ == "__main__":
    main()
