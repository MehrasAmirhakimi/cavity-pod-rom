"""Stage 2: non-intrusive parametric POD of steady solutions over Re.

Needs data/steady/steady_Re*.npz (scripts/run_steady_sweep.py) and the Ghia
et al. (1982) centreline data in src/ghia.py. Writes results/stage2.npz.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import ghia, solver  # noqa: E402
from src.parametric_rom import ParametricPOD  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from run_steady_sweep import TEST, TRAIN  # noqa: E402

R_LIST = [2, 3, 4, 6, 8, 10]
R_MAIN = 8
N_SAMPLES = 500


def load(Re):
    return np.load(os.path.join(ROOT, "data", "steady", f"steady_Re{Re}.npz"))


def main():
    g = solver.Grid(129)
    res = dict(Re_train=np.array(TRAIN), Re_test=np.array(TEST), r_list=np.array(R_LIST),
               r_main=R_MAIN)
    data = {Re: load(Re) for Re in TRAIN + TEST}
    for Re in TRAIN + TEST:
        res[f"residual_Re{Re}"] = float(data[Re]["residual"])
    S = {f: np.array([data[Re][f] for Re in TRAIN]) for f in ("omega", "psi", "T")}
    St = {f: np.array([data[Re][f] for Re in TEST]) for f in ("omega", "psi", "T")}
    rel = lambda a, b: np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)  # noqa: E731

    # QoIs of the full solutions
    res["psimin_fom_test"] = St["psi"].min(axis=1)
    res["nu_fom_test"] = np.array([solver.lid_nusselt(g, T) for T in St["T"]])
    res["psimin_fom_train"] = S["psi"].min(axis=1)
    res["nu_fom_train"] = np.array([solver.lid_nusselt(g, T) for T in S["T"]])

    # field errors versus r
    for f in ("omega", "psi", "T"):
        e_rbf, e_gpr, e_prj = [], [], []
        for r in R_LIST:
            m = ParametricPOD(TRAIN, S[f], r)
            e_rbf.append(rel(m.field(m.coef_rbf(TEST)), St[f]))
            e_gpr.append(rel(m.field(m.coef_gpr(TEST)[0]), St[f]))
            e_prj.append(rel(m.field(m.project(St[f])), St[f]))
        res[f"err_rbf_{f}"], res[f"err_gpr_{f}"], res[f"err_proj_{f}"] = map(np.array, (e_rbf, e_gpr, e_prj))
        mfull = ParametricPOD(TRAIN, S[f], R_MAIN)
        res[f"s_{f}"] = mfull.s_all
        print(f"{f:5s} r={R_MAIN}: RBF max {res[f'err_rbf_{f}'][R_LIST.index(R_MAIN)].max():.2e}  "
              f"GPR max {res[f'err_gpr_{f}'][R_LIST.index(R_MAIN)].max():.2e}  "
              f"projection max {res[f'err_proj_{f}'][R_LIST.index(R_MAIN)].max():.2e}", flush=True)

    # QoIs at r = R_MAIN, on test points and on a dense Re line
    mp = ParametricPOD(TRAIN, S["psi"], R_MAIN)
    mT = ParametricPOD(TRAIN, S["T"], R_MAIN)
    Re_line = np.logspace(1, 3, 121)
    res["Re_line"] = Re_line

    def qois(Re_eval):
        out = {}
        out["psimin_rbf"] = mp.field(mp.coef_rbf(Re_eval)).min(axis=1)
        out["nu_rbf"] = np.array([solver.lid_nusselt(g, T) for T in mT.field(mT.coef_rbf(Re_eval))])
        out["psimin_gpr"] = mp.field(mp.coef_gpr(Re_eval)[0]).min(axis=1)
        out["nu_gpr"] = np.array([solver.lid_nusselt(g, T) for T in mT.field(mT.coef_gpr(Re_eval)[0])])
        lo_p, hi_p, lo_n, hi_n = [], [], [], []
        for Re in np.atleast_1d(Re_eval):
            pm = mp.gpr_samples(Re, N_SAMPLES).min(axis=1)
            nu = np.array([solver.lid_nusselt(g, T) for T in mT.gpr_samples(Re, N_SAMPLES)])
            lo_p.append(np.percentile(pm, 2.5)); hi_p.append(np.percentile(pm, 97.5))
            lo_n.append(np.percentile(nu, 2.5)); hi_n.append(np.percentile(nu, 97.5))
        out["psimin_gpr_lo"], out["psimin_gpr_hi"] = np.array(lo_p), np.array(hi_p)
        out["nu_gpr_lo"], out["nu_gpr_hi"] = np.array(lo_n), np.array(hi_n)
        return out

    for k, v in qois(TEST).items():
        res[f"{k}_test"] = v
    for k, v in qois(Re_line).items():
        res[f"{k}_line"] = v
    for k in ("psimin", "nu"):
        inside = ((res[f"{k}_fom_test"] >= res[f"{k}_gpr_lo_test"])
                  & (res[f"{k}_fom_test"] <= res[f"{k}_gpr_hi_test"]))
        res[f"{k}_gpr_coverage"] = int(inside.sum())
    # coefficient-level GPR band for mode 1 and 2 of psi
    mu, sd = mp.coef_gpr(Re_line)
    res["psi_coef_train"], res["psi_coef_mu"], res["psi_coef_sd"] = mp.coef, mu, sd
    res["psi_coef_test"] = mp.project(St["psi"])
    res["psi_coef_rbf"] = mp.coef_rbf(Re_line)
    res["gp_kernels_psi"] = np.array([str(gp.kernel_) for gp in mp.gps])

    # centrelines and Ghia comparison (full solver and ROM)
    def centre(psi):
        u, v = solver.velocities(g, psi)
        return solver.centrelines(g, u, v)

    for Re in (100, 1000):
        y, uc, x, vc = centre(data[Re]["psi"])
        res[f"cl_u_fom_Re{Re}"], res[f"cl_v_fom_Re{Re}"] = uc, vc
        gy, gu, gx, gv = ghia.table(Re)
        res[f"ghia_y_Re{Re}"], res[f"ghia_u_Re{Re}"] = gy, gu
        res[f"ghia_x_Re{Re}"], res[f"ghia_v_Re{Re}"] = gx, gv
        du = np.interp(gy, y, uc) - gu
        dv = np.interp(gx, x, vc) - gv
        res[f"ghia_maxdiff_u_fom_Re{Re}"] = np.max(np.abs(du))
        res[f"ghia_maxdiff_v_fom_Re{Re}"] = np.max(np.abs(dv))
    res["cl_y"], res["cl_x"] = y, x
    # ROM at the held-out Re = 100 (not in TRAIN)
    assert 100 not in TRAIN
    for name, coef in (("rbf", mp.coef_rbf(100)), ("gpr", mp.coef_gpr(100)[0])):
        _, uc, _, vc = centre(mp.field(coef)[0])
        res[f"cl_u_{name}_Re100"], res[f"cl_v_{name}_Re100"] = uc, vc
        gy, gu, gx, gv = ghia.table(100)
        res[f"ghia_maxdiff_u_{name}_Re100"] = np.max(np.abs(np.interp(gy, y, uc) - gu))
        res[f"ghia_maxdiff_v_{name}_Re100"] = np.max(np.abs(np.interp(gx, x, vc) - gv))
        res[f"cl_maxdiff_u_{name}_vs_fom_Re100"] = np.max(np.abs(uc - res["cl_u_fom_Re100"]))
        res[f"cl_maxdiff_v_{name}_vs_fom_Re100"] = np.max(np.abs(vc - res["cl_v_fom_Re100"]))
    res["psimin_fom_Re1000"] = data[1000]["psi"].min()

    np.savez(os.path.join(ROOT, "results", "stage2.npz"), **res)
    print("saved results/stage2.npz")


if __name__ == "__main__":
    main()
