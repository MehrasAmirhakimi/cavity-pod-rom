"""Stage 4b: lid Nusselt number at unseen Re with adapted flow and temperature bases.

For each method (order 0, ITSGM 2, ORBI 1, ORBI 2, global), the same method
adapts both the vorticity basis (r = 10) and the temperature basis (q modes).
"global r=30" uses a rank-30 global vorticity basis, where the global flow ROM
does best (Stage 4), so the comparison is not tilted towards low rank.
The flow ROM drives the thermal ROM (one-way coupling, as in the full solver).
Both upwind terms are evaluated on the whole grid, so only the bases differ.
Snapshots are raw (zero lifting): the lid enters through the wall values
(Thom's formula for vorticity, T = 1 for temperature).

Test Re: 35, 70, 80, 125, 150, 280; window 0 <= t <= 20. Writes results/stage4_nu.npz.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src import operators, pod, solver  # noqa: E402
from src import thermal_rom as trm  # noqa: E402
from src.basis_interp import ORBI, itsgm  # noqa: E402
from src.galerkin_rom import GalerkinROM  # noqa: E402
from stage4_orbi import NW, DT, TEST, TRAIN, neighbours  # noqa: E402

R_FLOW = 10
R_FLOW_GLOBAL_HI = 30   # the global basis is also run at the rank where it does best
Q_LIST = [5, 10, 15, 20, 30]
METHODS = ["order 0", "ITSGM 2", "ORBI 1", "ORBI 2", "global", "global r=30"]
PR = 7
T_START = 5          # Nusselt error measured from t = 1 (step 5), as in Stage 3


def load(Re):
    f = os.path.join(ROOT, "data", f"fom_upwind_Re{Re}_im129_t20.npz")
    if Re == 100 and not os.path.exists(f):
        f = os.path.join(ROOT, "data", "fom_upwind_Re100_im129.npz")
    d = np.load(f)
    return d["omega"][:NW + 1], d["T"][:NW + 1]


def adapt(method, local, glob, Re, r):
    pair, triple = neighbours(Re)
    lx = np.log10(Re)
    if method == "order 0":
        return local[pair[0]][:, :r]
    if method.startswith("global"):
        return glob[:, :r]
    nodes = pair if method.endswith("1") else triple
    B = [local[x][:, :r] for x in nodes]
    if method.startswith("ORBI"):
        return ORBI(B, np.log10(nodes))(lx)
    return itsgm(B, np.log10(nodes), lx)


def main(out="stage4_nu.npz", im=129):
    g = solver.Grid(im)
    ps = operators.PoissonSolver(g)
    W, T = {}, {}
    for Re in TRAIN + TEST:
        w, t_full = load(Re)
        W[Re], T[Re] = w, trm.interior(g, t_full)
    qmax = max(Q_LIST)
    locW = {Re: pod.pod_svd(W[Re].T)[0][:, :R_FLOW] for Re in TRAIN}
    gW_hi = pod.pod_svd(np.hstack([W[Re].T for Re in TRAIN]))[0][:, :R_FLOW_GLOBAL_HI]
    locT = {Re: pod.pod_svd(T[Re].T)[0][:, :qmax] for Re in TRAIN}
    gW = pod.pod_svd(np.hstack([W[Re].T for Re in TRAIN]))[0][:, :R_FLOW]
    gT = pod.pod_svd(np.hstack([T[Re].T for Re in TRAIN]))[0][:, :qmax]
    zero = np.zeros(g.N)
    res = dict(test=np.array(TEST), q_list=np.array(Q_LIST), methods=np.array(METHODS),
               r_flow=R_FLOW, t_start=T_START * DT)
    shape = (len(METHODS), len(Q_LIST), len(TEST))
    errT, projT, nu_rel, nu_abs = (np.full(shape, np.nan) for _ in range(4))
    nu_fom_all = []
    for it, Re in enumerate(TEST):
        Tfull = load(Re)[1]
        nu_fom = np.array([solver.lid_nusselt(g, x) for x in Tfull])
        nu_fom_all.append(nu_fom)
        Ti = T[Re]
        nrm = np.linalg.norm(Ti[1:], axis=1)
        for im, meth in enumerate(METHODS):
            if meth == "global r=30":
                X = gW_hi
            else:
                X = adapt(meth, locW, gW, Re, R_FLOW)
            flow = GalerkinROM(g, Re, X, zero, poisson=ps)
            flow.add_upwind_full()
            a_traj, _, _ = flow.integrate(X.T @ W[Re][0], DT, NW)
            for iq, q in enumerate(Q_LIST):
                Th = adapt(meth, locT, gT, Re, q)
                rom = trm.ThermalROM(g, Re, PR, Th, zero, flow)
                bt, _ = rom.integrate(Th.T @ Ti[0], a_traj, DT)
                errT[im, iq, it] = (np.linalg.norm(rom.reconstruct(bt)[1:] - Ti[1:], axis=1) / nrm).max()
                R = Ti[1:].T - Th @ (Th.T @ Ti[1:].T)
                projT[im, iq, it] = (np.linalg.norm(R, axis=0) / nrm).max()
                dnu = np.abs(rom.nusselt(bt) - nu_fom)[T_START:]
                nu_abs[im, iq, it] = dnu.max()
                nu_rel[im, iq, it] = (dnu / nu_fom[T_START:]).max()
                if q == 10:
                    res[f"nu_{meth.replace(' ', '').replace('=', '')}_Re{Re}"] = rom.nusselt(bt)
        print(f"Re={Re:4d}  " + "  ".join(
            f"{m}: Nu {100*nu_rel[i, Q_LIST.index(10), it]:.2f}%" for i, m in enumerate(METHODS)), flush=True)
    res.update(errT=errT, projT=projT, nu_rel=nu_rel, nu_abs=nu_abs, nu_fom=np.array(nu_fom_all),
               t=np.linspace(0, NW * DT, NW + 1))
    np.savez(os.path.join(ROOT, "results", out), **res)
    print(f"saved results/{out}")


if __name__ == "__main__":
    main()
