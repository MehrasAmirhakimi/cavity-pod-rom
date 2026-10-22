"""Print the results tables (Markdown) straight from results/*.npz.

Every number in README.md comes from this script.
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: dict(np.load(os.path.join(ROOT, "results", f)))  # noqa: E731


def stage1():
    r = R("stage1.npz")
    fw = r["fom_wall_window"]
    print(f"Full solver, steps 1-100 (0 < t <= 20): {fw:.1f} s; whole run {r['fom_wall_full_run']:.1f} s")
    print(f"r for 99/99.9/99.99% energy: {r['r_energy'].tolist()}; "
          f"SVD vs method of snapshots (first 10 sigma): {r['sv_check']:.1e}")
    print(f"Steady states, upwind vs central: {100*r['steady_gap_central_upwind']:.2f} %; "
          f"min psi upwind {r['psi_min_upwind']:.6f}, central {r['psi_min_central']:.6f}")
    print(f"Offline: Galerkin tensors (r=40) {r['offline_galerkin_s']:.2f} s, "
          f"DEIM snapshots+POD {r['offline_deim_s']:.2f} s\n")
    print("| r | projection error | central ROM | upwind-DEIM ROM (m = 2r) | DEIM ROM time | speed-up |")
    print("|---|---|---|---|---|---|")
    for i, rr in enumerate(r["r_list"]):
        print(f"| {rr} | {r['proj_err'][i].max():.1e} | {r['err_central'][i].max():.1e} | "
              f"{r['err_deim'][i].max():.1e} | {1e3*r['wall_deim'][i]:.1f} ms | "
              f"{fw / r['wall_deim'][i]:,.0f}× |")
    print("\nCheck: central ROM vs central full solver, max error:",
          ", ".join(f"r={rr}: {e.max():.1e}" for rr, e in zip(r["r_list"], r["err_central_vs_centralfom"])))
    print("Full-grid upwind ROM, max error:",
          ", ".join(f"r={rr}: {e.max():.1e}" for rr, e in zip(r["r_list"], r["err_full_upwind"])))
    print(f"Long run to t=200, r=10: DEIM max {r['err_longtime_deim_r10'].max():.1e}, final "
          f"{r['err_longtime_deim_r10'][-1]:.1e}; central final {r['err_longtime_central_r10'][-1]:.2e}")
    for k, lab in (("Re80", "Re=100 basis at Re=80"), ("Re125", "Re=100 basis at Re=125"),
                   ("Re100_from_80_125", "Re=80+125 basis at Re=100")):
        e, p = r[f"err_deim_{k}"].max(axis=1), r[f"proj_err_{k}"].max(axis=1)
        i = list(r["r_list"]).index(10)
        print(f"{lab}: r=10 ROM {e[i]:.2e}, projection {p[i]:.2e}; best ROM over r {e.min():.2e}")


def stage2():
    r = R("stage2.npz")
    i = list(r["r_list"]).index(int(r["r_main"]))
    print(f"\nStage 2: train Re {r['Re_train'].tolist()}, test Re {r['Re_test'].tolist()}, r = {r['r_main']}")
    print("max steady residual:", f"{max(r[k] for k in r if k.startswith('residual_')):.1e}\n")
    print("| field | RBF | GPR mean | projection |")
    print("|---|---|---|---|")
    for f, lab in (("omega", "vorticity"), ("psi", "stream function"), ("T", "temperature")):
        print(f"| {lab} | {100*r['err_rbf_'+f][i].max():.3f} % | {100*r['err_gpr_'+f][i].max():.3f} % | "
              f"{100*r['err_proj_'+f][i].max():.3f} % |")
    print("\n| Re (held out) | min psi, full | RBF | GPR [95% band] | Nu, full | RBF | GPR [95% band] |")
    print("|---|---|---|---|---|---|---|")
    for j, Re in enumerate(r["Re_test"]):
        print(f"| {Re} | {r['psimin_fom_test'][j]:.5f} | {r['psimin_rbf_test'][j]:.5f} | "
              f"{r['psimin_gpr_test'][j]:.5f} [{r['psimin_gpr_lo_test'][j]:.5f}, {r['psimin_gpr_hi_test'][j]:.5f}] | "
              f"{r['nu_fom_test'][j]:.3f} | {r['nu_rbf_test'][j]:.3f} | "
              f"{r['nu_gpr_test'][j]:.3f} [{r['nu_gpr_lo_test'][j]:.3f}, {r['nu_gpr_hi_test'][j]:.3f}] |")
    print(f"GPR band coverage: min psi {r['psimin_gpr_coverage']}/8, Nu {r['nu_gpr_coverage']}/8")
    print("max |RBF - full|: min psi %.1e, Nu %.1e" % (
        np.abs(r["psimin_rbf_test"] - r["psimin_fom_test"]).max(),
        np.abs(r["nu_rbf_test"] - r["nu_fom_test"]).max()))
    print("max |GPR - full|: min psi %.1e, Nu %.1e" % (
        np.abs(r["psimin_gpr_test"] - r["psimin_fom_test"]).max(),
        np.abs(r["nu_gpr_test"] - r["nu_fom_test"]).max()))
    print("Ghia max |diff|: full Re=100 u %.4f v %.4f; full Re=1000 u %.3f v %.3f; "
          "RBF ROM Re=100 u %.4f v %.4f; GPR ROM Re=100 u %.4f v %.4f" % tuple(float(r[k]) for k in (
              "ghia_maxdiff_u_fom_Re100", "ghia_maxdiff_v_fom_Re100", "ghia_maxdiff_u_fom_Re1000",
              "ghia_maxdiff_v_fom_Re1000", "ghia_maxdiff_u_rbf_Re100", "ghia_maxdiff_v_rbf_Re100",
              "ghia_maxdiff_u_gpr_Re100", "ghia_maxdiff_v_gpr_Re100")))
    print("ROM vs full centrelines Re=100: RBF u %.1e v %.1e" % (
        r["cl_maxdiff_u_rbf_vs_fom_Re100"], r["cl_maxdiff_v_rbf_vs_fom_Re100"]))


def stage3():
    r = R("stage3.npz")
    t = r["t"]
    print(f"\nStage 3: T within 1e-4 of steady from t = "
          f"{t[np.argmax(r['T_dist_to_steady'] < 1e-4)]:.1f}; q for 99/99.9/99.99%: {r['r_energy_T'].tolist()}; "
          f"DEIM cap m <= {r['m_cap']}")
    e = r["err_T_q40_m80_uncapped"]
    print(f"Uncapped q=40, m=80: first step error > 1: {int(np.argmax(e > 1)) + 1} "
          f"(t = {t[int(np.argmax(e > 1)) + 1]:.1f}); final error {np.nanmax(e):.1e}")
    fom = r["fom_wall_flow_total"] + r["fom_wall_temp_total"]
    print(f"Full solver whole run: flow {r['fom_wall_flow_total']:.1f} s + temperature "
          f"{r['fom_wall_temp_total']:.1f} s; flow ROM (r=10) {r['flow_rom_wall']:.3f} s\n")
    print("| q | m | max T error | projection | max abs Nu error (t ≥ 1) | ROM time (flow + T) | speed-up |")
    print("|---|---|---|---|---|---|---|")
    for i, q in enumerate(r["q_list"]):
        tot = r["flow_rom_wall"] + r["thermal_rom_wall"][i]
        print(f"| {q} | {r['m_list'][i]} | {r['err_T'][i].max():.1e} | {r['proj_err_T'][i].max():.1e} | "
              f"{np.abs(r['nu_rom'][i][5:] - r['nu_fom'][5:]).max():.1e} | {tot:.3f} s | {fom/tot:,.0f}× |")
    print(f"steady Nu (full solver, t=200): {r['nu_fom'][-1]:.4f}")


def stage4():
    r = R("stage4.npz")
    M, Rl, E = list(r["methods"]), list(r["r_list"]), list(r["eval_Re"])
    ti = [E.index(t) for t in r["test"]]
    print(f"\nStage 4: train Re {r['train'].tolist()}, test Re {r['test'].tolist()}")
    for key, ttl in (("rom_max", "Galerkin ROM"), ("proj_max", "best fit in basis")):
        print(f"\n{ttl}: max relative error of omega over 0 < t <= 20, worst of the 6 test Re (median)\n")
        print("| method | " + " | ".join(f"r = {x}" for x in Rl) + " |")
        print("|---|" + "---|" * len(Rl))
        for i, m in enumerate(M):
            cells = []
            for j in range(len(Rl)):
                v = r[key][i, j, ti]
                w = np.nanmax(v)
                cells.append("diverged" if not np.isfinite(w) else f"{w:.1e} ({np.nanmedian(v):.1e})")
            print(f"| {m} | " + " | ".join(cells) + " |")
    for q in (1, 2):
        a = r["proj_max"][M.index(f"ITSGM {q}")][:, ti] / r["proj_max"][M.index(f"ORBI {q}")][:, ti]
        print(f"order {q}: ITSGM/ORBI best-fit ratio, median over test Re, per r: "
              + ", ".join(f"r={x}: {np.median(a[j]):.2f}" for j, x in enumerate(Rl)))
    j10 = Rl.index(10)
    print(f"largest ORBI rotation angle: r=10 {np.nanmax(r['max_angle'][:, j10]):.2f} rad, "
          f"r=30 {np.nanmax(r['max_angle'][:, Rl.index(30)]):.2f} rad (pi = 3.14)")
    print(f"ORBI 2 adaptation time (r=10, median): {1e3*np.nanmedian(r['t_adapt'][M.index('ORBI 2'), j10, ti]):.1f} ms; "
          f"operator assembly {np.nanmedian(r['t_build'][:, j10, ti]):.3f} s")
    print(f"DEIM (r=10, common DEIM basis), m = {r['deim_m_list'].tolist()}: worst over test Re "
          f"ORBI 2 {[f'{x:.1e}' for x in r['deim_err'][0].max(axis=1)]}, "
          f"global {[f'{x:.1e}' for x in r['deim_err'][1].max(axis=1)]}; "
          f"online time {1e3*r['deim_wall'].min():.0f}-{1e3*r['deim_wall'].max():.0f} ms; "
          f"full solver, same 100 steps: {r['fom_wall_test'].min():.0f}-{r['fom_wall_test'].max():.0f} s")
    n = R("stage4_nu.npz")
    Mn, Q = list(n["methods"]), list(n["q_list"])
    print(f"\nLid Nusselt number, flow r = {n['r_flow']} (global also at r = 30): max relative error over "
          f"{n['t_start']:g} <= t <= 20, worst of the 6 test Re (median)\n")
    print("| method | " + " | ".join(f"q = {x}" for x in Q) + " |")
    print("|---|" + "---|" * len(Q))
    for i, m in enumerate(Mn):
        v = n["nu_rel"][i]
        print(f"| {m} | " + " | ".join(f"{100*np.nanmax(v[j]):.1f} % ({100*np.nanmedian(v[j]):.1f} %)"
                                        for j in range(len(Q))) + " |")


if __name__ == "__main__":
    stage1()
    stage2()
    stage3()
    stage4()
