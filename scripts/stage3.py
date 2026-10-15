"""Stage 3: temperature POD and a coupled thermal ROM for the lid Nusselt number.

The flow ROM (r = 10, DEIM m = 20, basis from 0 <= t <= 20) drives a
temperature ROM over the whole run 0 <= t <= 200, because the temperature
needs much longer to settle than the flow (Pr = 7).
Writes results/stage3.npz.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import operators, pod, solver  # noqa: E402
from src import thermal_rom as trm  # noqa: E402
from src.galerkin_rom import GalerkinROM  # noqa: E402

DT, NW, R_FLOW = 0.2, 100, 10
SV_FLOOR = 1e-8   # DEIM modes with sigma_k/sigma_1 below this are left out (see README)
Q_LIST = [5, 10, 15, 20, 30, 40]


def main():
    g = solver.Grid(129)
    d = np.load(os.path.join(ROOT, "data", "fom_upwind_Re100_im129.npz"))
    W, t = d["omega"], d["t"]
    nsteps = len(t) - 1
    res = dict(t=t, q_list=np.array(Q_LIST), r_flow=R_FLOW)

    # flow ROM (as in Stage 1)
    ws = W[-1]
    Phi, _ = pod.pod_svd((W[:NW + 1] - ws).T)
    flow = GalerkinROM(g, 100, Phi[:, :R_FLOW], ws)
    ft = operators.field_terms(g, W[:NW + 1].T, flow.poisson, lid=True)
    Un, _ = pod.pod_svd(operators.upwind_dissipation(g, ft["u"], ft["v"], ft["d2x"], ft["d2y"]))
    flow.add_deim(Un[:, :2 * R_FLOW], pod.deim_points(Un[:, :2 * R_FLOW]))
    a_traj, t_flow, _ = flow.integrate(Phi[:, :R_FLOW].T @ (W[0] - ws), DT, nsteps)
    res["flow_rom_wall"] = t_flow

    # temperature snapshots (interior nodes) and POD
    Ti = trm.interior(g, d["T"])
    Ts = Ti[-1]
    res["T_dist_to_steady"] = np.linalg.norm(Ti - Ts, axis=1) / np.linalg.norm(Ts)
    XT = (Ti - Ts).T
    Theta, sT = pod.pod_svd(XT)
    res["s_T"] = sT
    res["r_energy_T"] = np.array([pod.rank_for_energy(sT, th) for th in (0.99, 0.999, 0.9999)])
    res["T_modes"] = Theta[:, :4].T.copy()

    # snapshots of the thermal upwind term (full-solver u, v, T) for DEIM
    st = trm.temperature_terms(g, Ti.T, lid=True)
    NLT = (np.abs(d["u"].T) / (2 * g.dx) * st["d2x"] + np.abs(d["v"].T) / (2 * g.dy) * st["d2y"])
    UT, sNT = pod.pod_svd(NLT)
    res["s_nl_T"] = sNT

    nu_fom = np.array([solver.lid_nusselt(g, T) for T in d["T"]])
    res["nu_fom"] = nu_fom
    nrm = np.linalg.norm(Ti[1:], axis=1)
    m_cap = int(np.sum(sNT / sNT[0] >= SV_FLOOR))
    res["m_cap"] = m_cap

    # evidence of the instability without the cap: q = 40 with m = 80 DEIM points
    rom = trm.ThermalROM(g, 100, 7, Theta[:, :40], Ts, flow, UT[:, :80], pod.deim_points(UT[:, :80]))
    with np.errstate(all="ignore"):
        bt, _ = rom.integrate(Theta[:, :40].T @ (Ti[0] - Ts), a_traj, DT)
        e = np.linalg.norm(rom.reconstruct(bt)[1:] - Ti[1:], axis=1) / np.linalg.norm(Ti[1:], axis=1)
    res["err_T_q40_m80_uncapped"] = e
    print(f"uncapped q=40, m=80: max error {np.nanmax(e):.2e}, first step with error > 1: "
          f"{int(np.argmax(e > 1)) + 1 if np.nanmax(e) > 1 else None}; cap gives m <= {m_cap}")

    errs, perrs, nus, walls, ms = [], [], [], [], []
    for q in Q_LIST:
        m = min(2 * q, m_cap)
        ms.append(m)
        rom = trm.ThermalROM(g, 100, 7, Theta[:, :q], Ts, flow, UT[:, :m],
                             pod.deim_points(UT[:, :m]))
        b0 = Theta[:, :q].T @ (Ti[0] - Ts)
        best = np.inf
        for _ in range(3):
            bt, wt = rom.integrate(b0, a_traj, DT)
            best = min(best, wt)
        Tr = rom.reconstruct(bt)
        errs.append(np.linalg.norm(Tr[1:] - Ti[1:], axis=1) / nrm)
        perrs.append(pod.projection_error(XT[:, 1:], Theta, q) / nrm)
        nus.append(rom.nusselt(bt))
        walls.append(best)
        print(f"q={q:2d}  T max err {errs[-1].max():.2e}  final {errs[-1][-1]:.2e}  "
              f"proj max {perrs[-1].max():.2e}  Nu final ROM {nus[-1][-1]:.4f} FOM {nu_fom[-1]:.4f}  "
              f"max |dNu| (t>=1) {np.abs(nus[-1][5:] - nu_fom[5:]).max():.3e}  {best*1e3:.0f} ms",
              flush=True)
    res["err_T"], res["proj_err_T"] = np.array(errs), np.array(perrs)
    res["m_list"] = np.array(ms)
    res["nu_rom"], res["thermal_rom_wall"] = np.array(nus), np.array(walls)
    res["fom_wall_flow_total"] = d["wall_flow"].sum()
    res["fom_wall_temp_total"] = d["wall_temp"].sum()
    np.savez(os.path.join(ROOT, "results", "stage3.npz"), **res)
    print("saved results/stage3.npz")


if __name__ == "__main__":
    main()
