"""Figures for the README and the notebook. Each function takes a results dict."""
import os

import matplotlib.pyplot as plt
import numpy as np

# Fixed colour per entity (categorical palette, validated for colour-vision deficiency).
C = dict(fom="#0b0b0b", deim="#2a78d6", central="#eb6834", proj="#1baf7a",
         full="#008300", diag="#6250d6", gpr="#2a78d6", rbf="#eb6834", ghia="#e34948",
         grey="#52514e")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 10,
    "axes.labelsize": 10, "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e",
})


def _save(fig, name, outdir):
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, name), bbox_inches="tight")
    return fig


# ------------------------------------------------------------------ Stage 1
def s1_steady_approach(r, outdir=None):
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.6, 1]))
    t = r["t_full"]
    for a in ax:
        a.axvspan(0, 20, color="#2a78d6", alpha=0.07, lw=0)
    ax[0].semilogy(t[:-1], r["dist_to_steady"][:-1], color=C["fom"])
    ax[0].set_ylabel(r"$\|\omega(t)-\omega_s\| / \|\omega_s\|$")
    ax[0].text(10, 1e-8, "snapshot window\n0 ≤ t ≤ 20", ha="center", fontsize=8.5,
               color=C["grey"])
    ax[0].set_ylim(1e-9, 3)
    ax[1].plot(t[1:], r["picard_iters"], color=C["fom"])
    ax[1].set_ylabel("Picard iterations")
    ax[1].set_xlabel("t")
    ax[1].set_xlim(0, 60)
    ax[0].set_title("Re = 100 full run: approach to steady state (t > 60 not shown)")
    return _save(fig, "s1_steady_approach.png", outdir)


def s1_pod_spectrum(r, outdir=None):
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))
    s, sn = r["s"], r["s_nl"]
    k = np.arange(1, len(s) + 1)
    ax[0].semilogy(k, s / s[0], "o-", ms=3, color=C["deim"], label="vorticity snapshots")
    ax[0].semilogy(k, sn / sn[0], "s-", ms=3, color=C["central"], label="upwind-term snapshots (DEIM)")
    ax[0].set_xlabel("index k")
    ax[0].set_ylabel(r"$\sigma_k/\sigma_1$")
    ax[0].set_ylim(1e-17, 2)
    ax[0].legend()
    ax[0].set_title("Singular values")
    E = r["energy"]
    ax[1].semilogy(k[:30], np.clip(1 - E[:30], 1e-30, None), "o-", ms=3, color=C["deim"])
    ax[1].set_ylim(1e-17, 3)
    for th, rr in zip((1e-2, 1e-3, 1e-4), r["r_energy"]):
        ax[1].axhline(th, color=C["grey"], lw=0.8, ls=":")
        ax[1].text(29, th * 1.6, f"{100*(1-th):g}% energy: r = {rr}", fontsize=8.5,
                   color=C["grey"], ha="right", va="bottom")
    ax[1].set_xlabel("number of modes r")
    ax[1].set_ylabel("1 − cumulative energy")
    ax[1].set_title("Energy not captured by r modes")
    fig.tight_layout()
    return _save(fig, "s1_pod_spectrum.png", outdir)


def mode_panels(fields, titles, n, outdir=None, name="modes.png", suptitle=None):
    fig, ax = plt.subplots(1, len(fields), figsize=(2.6 * len(fields), 2.8))
    x = np.linspace(0, 1, n + 2)[1:-1]
    for a, f, ti in zip(ax, fields, titles):
        F = f.reshape(n, n)
        m = np.percentile(np.abs(F), 99)        # corner peaks would hide the interior
        a.contourf(x, x, np.clip(F, -m, m), levels=np.linspace(-m, m, 25), cmap="RdBu_r")
        a.set_aspect("equal")
        a.set_title(ti)
        a.set_xticks([0, 0.5, 1])
        a.set_yticks([0, 0.5, 1])
        a.grid(False)
    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    return _save(fig, name, outdir)


def s1_error_vs_time(r, outdir=None, rs=(5, 10)):
    fig, ax = plt.subplots(1, len(rs), figsize=(8.6, 3.3), sharey=True)
    t = r["t_window"][1:]
    rl = list(r["r_list"])
    for a, rr in zip(ax, rs):
        i = rl.index(rr)
        a.semilogy(t, r["err_central"][i], "--", color=C["central"], label="central Galerkin ROM")
        a.semilogy(t, r["err_deim"][i], color=C["deim"], label="upwind-DEIM Galerkin ROM")
        a.semilogy(t, r["proj_err"][i], ":", color=C["proj"], lw=2, label="best fit in basis (projection)")
        a.set_title(f"r = {rr}")
        a.set_xlabel("t")
    ax[0].set_ylabel(r"relative $L_2$ error of $\omega$")
    ax[0].legend(loc="lower left")
    fig.tight_layout()
    return _save(fig, "s1_error_vs_time.png", outdir)


def s1_error_vs_r(r, outdir=None):
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.4))
    rl = r["r_list"]
    mx = lambda k: r[k].max(axis=1)  # noqa: E731
    ax[0].semilogy(rl, mx("err_central"), "o--", color=C["central"], label="central ROM vs upwind full solver")
    ax[0].semilogy(rl, mx("err_central_vs_centralfom"), "^-.", color=C["diag"],
                   label="central ROM vs central full solver (check)")
    ax[0].semilogy(rl, mx("err_deim"), "o-", color=C["deim"], label="upwind-DEIM ROM vs upwind full solver")
    ax[0].semilogy(rl, mx("proj_err"), "s:", color=C["proj"], label="projection error (lower bound)")
    ax[0].set_xlabel("number of modes r")
    ax[0].set_ylabel("max relative error over 0 < t ≤ 20")
    ax[0].set_ylim(1e-9, 1)
    ax[0].set_title("Error versus r")
    sp_d = r["fom_wall_window"] / r["wall_deim"]
    sp_f = r["fom_wall_window"] / r["wall_full_upwind"]
    ax[1].semilogy(rl, sp_d, "o-", color=C["deim"], label="upwind-DEIM ROM (m = 2r points)")
    ax[1].semilogy(rl, sp_f, "o--", color=C["full"], label="upwind ROM, full-grid evaluation")
    ax[1].set_xlabel("number of modes r")
    ax[1].set_ylabel("speed-up (full-solver time / ROM time)")
    ax[1].set_title("Online speed-up, same 100 time steps")
    ax[1].legend(loc="center right")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    return _save(fig, "s1_error_vs_r.png", outdir)


def s1_centrelines(r, outdir=None, rr=5):
    tps = r["t_profiles"]
    fig, ax = plt.subplots(2, len(tps), figsize=(10.5, 5.6))
    y, x = r["prof_y"], r["prof_x"]
    for j, tp in enumerate(tps):
        a = ax[0, j]
        a.plot(r[f"prof_u_fom_t{tp:g}"], y, color=C["fom"], label="full solver (upwind)")
        a.plot(r[f"prof_u_central_r{rr}_t{tp:g}"], y, "--", color=C["central"], label=f"central ROM, r = {rr}")
        a.plot(r[f"prof_u_deim_r{rr}_t{tp:g}"], y, ":", lw=2.2, color=C["deim"], label=f"upwind-DEIM ROM, r = {rr}")
        a.set_title(f"t = {tp:g}")
        a.set_xlabel("u at x = 0.5")
        b = ax[1, j]
        b.plot(x, r[f"prof_v_fom_t{tp:g}"], color=C["fom"])
        b.plot(x, r[f"prof_v_central_r{rr}_t{tp:g}"], "--", color=C["central"])
        b.plot(x, r[f"prof_v_deim_r{rr}_t{tp:g}"], ":", lw=2.2, color=C["deim"])
        b.set_xlabel("x")
        if j == 0:
            a.set_ylabel("y")
            b.set_ylabel("v at y = 0.5")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(h_pad=1.5)
    return _save(fig, "s1_centrelines.png", outdir)


def s1_generalisation(r, outdir=None):
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.4))
    t = r["t_full"][1:]
    ax[0].semilogy(t, r["err_longtime_central_r10"], "--", color=C["central"], label="central ROM, r = 10")
    ax[0].semilogy(t, r["err_longtime_deim_r10"], color=C["deim"], label="upwind-DEIM ROM, r = 10")
    ax[0].axvline(20, color=C["grey"], lw=0.8, ls=":")
    ax[0].text(22, 3e-3, "end of\nsnapshot window", fontsize=8, color=C["grey"])
    ax[0].set_xlabel("t")
    ax[0].set_ylabel(r"relative $L_2$ error of $\omega$")
    ax[0].set_title("Run to t = 200 (10× the window)")
    ax[0].legend(loc="center right")
    rl = r["r_list"]
    cases = [("err_deim_Re80", "proj_err_Re80", "Re = 100 basis → Re = 80", C["central"]),
             ("err_deim_Re125", "proj_err_Re125", "Re = 100 basis → Re = 125", C["proj"]),
             ("err_deim_Re100_from_80_125", "proj_err_Re100_from_80_125",
              "Re = 80 + 125 basis → Re = 100", C["deim"])]
    for ke, kp, lab, col in cases:
        if ke in r:
            ax[1].semilogy(rl, r[ke].max(axis=1), "o-", color=col, label=lab + " (ROM)")
            ax[1].semilogy(rl, r[kp].max(axis=1), ":", color=col, lw=1.4)
    ax[1].set_xlabel("number of modes r")
    ax[1].set_ylabel("max relative error, 0 < t ≤ 20")
    ax[1].set_title("Other Re (dotted: projection error)")
    ax[1].legend(fontsize=7.8)
    fig.tight_layout()
    return _save(fig, "s1_generalisation.png", outdir)


# ------------------------------------------------------------------ Stage 2
def s2_field_error_vs_r(r, outdir=None):
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    rl = r["r_list"]
    for a, f, lab in zip(ax, ("omega", "psi", "T"), ("vorticity ω", "stream function ψ", "temperature T")):
        a.semilogy(rl, r[f"err_rbf_{f}"].max(axis=1), "o-", color=C["rbf"], label="RBF")
        a.semilogy(rl, r[f"err_gpr_{f}"].max(axis=1), "o-", color=C["gpr"], label="GPR mean")
        a.semilogy(rl, r[f"err_proj_{f}"].max(axis=1), "s:", color=C["proj"], label="projection (lower bound)")
        a.set_title(lab)
        a.set_xlabel("number of modes r")
    ax[0].set_ylabel("max relative L2 error\nover 8 held-out Re")
    ax[0].legend()
    fig.tight_layout()
    return _save(fig, "s2_field_error_vs_r.png", outdir)


def s2_qoi(r, outdir=None):
    fig, ax = plt.subplots(2, 2, figsize=(9.6, 6.0), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.5, 1]))
    Rl, Rt, Rtr = r["Re_line"], r["Re_test"], r["Re_train"]
    for j, (k, lab) in enumerate((("psimin", "primary vortex strength  min ψ"),
                                  ("nu", "average lid Nusselt number"))):
        a = ax[0, j]
        a.fill_between(Rl, r[f"{k}_gpr_lo_line"], r[f"{k}_gpr_hi_line"], color=C["gpr"], alpha=0.18,
                       lw=0, label="GPR 95% band")
        a.plot(Rl, r[f"{k}_gpr_line"], color=C["gpr"], label="GPR mean")
        a.plot(Rl, r[f"{k}_rbf_line"], "--", color=C["rbf"], label="RBF")
        a.plot(Rtr, r[f"{k}_fom_train"], "o", color=C["fom"], ms=5, label="full solver, training Re")
        a.plot(Rt, r[f"{k}_fom_test"], "o", mfc="white", mec=C["fom"], ms=6, mew=1.4,
               label="full solver, held-out Re")
        a.set_xscale("log")
        a.set_title(lab)
        b = ax[1, j]
        lo = r[f"{k}_gpr_test"] - r[f"{k}_gpr_lo_test"]
        hi = r[f"{k}_gpr_hi_test"] - r[f"{k}_gpr_test"]
        b.axhline(0, color=C["grey"], lw=0.8)
        b.errorbar(Rt * 1.03, r[f"{k}_gpr_test"] - r[f"{k}_fom_test"], yerr=[lo, hi], fmt="o",
                   color=C["gpr"], ms=4, capsize=2, label="GPR mean ± 95% band")
        b.plot(Rt / 1.03, r[f"{k}_rbf_test"] - r[f"{k}_fom_test"], "s", color=C["rbf"], ms=4, label="RBF")
        b.set_xscale("log")
        b.set_xlabel("Re")
        b.set_ylabel("ROM − full solver")
    ax[0, 0].legend(fontsize=7.8)
    ax[1, 0].legend(fontsize=7.8)
    fig.tight_layout()
    return _save(fig, "s2_qoi.png", outdir)


def s2_coefficients(r, outdir=None, modes=(0, 1, 2)):
    fig, ax = plt.subplots(1, len(modes), figsize=(10.5, 3.2))
    Rl = r["Re_line"]
    for a, k in zip(ax, modes):
        mu, sd = r["psi_coef_mu"][:, k], r["psi_coef_sd"][:, k]
        a.fill_between(Rl, mu - 1.96 * sd, mu + 1.96 * sd, color=C["gpr"], alpha=0.18, lw=0)
        a.plot(Rl, mu, color=C["gpr"], label="GPR mean ± 1.96σ")
        a.plot(Rl, r["psi_coef_rbf"][:, k], "--", color=C["rbf"], label="RBF")
        a.plot(r["Re_train"], r["psi_coef_train"][:, k], "o", color=C["fom"], ms=5, label="training")
        a.plot(r["Re_test"], r["psi_coef_test"][:, k], "o", mfc="white", mec=C["fom"], ms=6, mew=1.4,
               label="held-out (projected)")
        a.set_xscale("log")
        a.set_xlabel("Re")
        a.set_title(f"ψ mode {k+1} coefficient")
    ax[0].legend(fontsize=7.8)
    fig.tight_layout()
    return _save(fig, "s2_coefficients.png", outdir)


def s2_ghia(r, outdir=None):
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.6))
    y, x = r["cl_y"], r["cl_x"]
    for j, Re in enumerate((100, 1000)):
        a, b = ax[2 * j], ax[2 * j + 1]
        a.plot(r[f"cl_u_fom_Re{Re}"], y, color=C["fom"], label="full solver (upwind, 129 intervals)")
        b.plot(x, r[f"cl_v_fom_Re{Re}"], color=C["fom"])
        if Re == 100:
            a.plot(r["cl_u_rbf_Re100"], y, "--", color=C["rbf"], label="parametric ROM (RBF), Re held out")
            b.plot(x, r["cl_v_rbf_Re100"], "--", color=C["rbf"])
        a.plot(r[f"ghia_u_Re{Re}"], r[f"ghia_y_Re{Re}"], "o", mfc="white", mec=C["ghia"], mew=1.4, ms=5,
               label="Ghia et al. (1982)")
        b.plot(r[f"ghia_x_Re{Re}"], r[f"ghia_v_Re{Re}"], "o", mfc="white", mec=C["ghia"], mew=1.4, ms=5)
        a.set_xlabel("u at x = 0.5")
        a.set_ylabel("y")
        b.set_xlabel("x")
        b.set_ylabel("v at y = 0.5")
        a.set_title(f"Re = {Re}")
        b.set_title(f"Re = {Re}")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return _save(fig, "s2_ghia.png", outdir)


# ------------------------------------------------------------------ Stage 3
def s3_thermal(r, outdir=None, qs=(10, 20, 30)):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    t = r["t"]
    ql = list(r["q_list"])
    cols = (C["central"], C["deim"], C["proj"])
    for q, col in zip(qs, cols):
        i = ql.index(q)
        ax[0].semilogy(t[1:], r["err_T"][i], color=col, label=f"q = {q} (m = {r['m_list'][i]})")
        ax[0].semilogy(t[1:], r["proj_err_T"][i], ":", color=col, lw=1.3)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("relative L2 error of T")
    ax[0].set_title("Temperature (dotted: projection error)")
    ax[0].legend(fontsize=7.8)
    ax[1].plot(t[1:], r["nu_fom"][1:], color=C["fom"], lw=2.4, label="full solver")
    for q, col in zip(qs[:2], cols[:2]):
        i = ql.index(q)
        ax[1].plot(t[1:], r["nu_rom"][i][1:], "--", color=col, label=f"coupled ROM, q = {q}")
    ax[1].set_xscale("log")
    ax[1].set_ylim(0, 40)
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("average lid Nusselt number")
    ax[1].set_title("Nusselt number history")
    ax[1].legend(fontsize=7.8)
    i = ql.index(40)
    e_bad = r["err_T_q40_m80_uncapped"]
    ax[2].semilogy(t[1:][np.isfinite(e_bad)], e_bad[np.isfinite(e_bad)], color=C["grey"],
                   label="q = 40, m = 80 DEIM points")
    ax[2].semilogy(t[1:], r["err_T"][i], color=C["deim"], label=f"q = 40, m = {r['m_list'][i]} (capped)")
    ax[2].set_xlabel("t")
    ax[2].set_title("DEIM size: instability and fix")
    ax[2].legend(fontsize=7.8)
    fig.tight_layout()
    return _save(fig, "s3_thermal.png", outdir)
