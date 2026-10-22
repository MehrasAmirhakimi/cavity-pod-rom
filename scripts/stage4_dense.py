"""Stage 4 note: does denser training fix ORBI at high rank?

Re-runs the Stage 4 comparison at the two Re that stay unseen (80 and 125) with
two training sets built from the same transient runs (0 <= t <= 20):

  original: Re = 25, 50, 100, 200, 400            (spacing: factor 2 in Re)
  denser:   Re = 25, 35, 50, 70, 100, 150, 200, 280, 400   (factor about 1.4)

The denser set uses four former test Re as training data, so only 80 and 125
remain as tests. Same methods and settings as scripts/stage4_orbi.py (raw
snapshots, zero lifting, upwind term on the whole grid).

Also stores two diagnostics: the energy left out by r local modes, and how well
the k-th local mode at Re = 100 matches the k-th mode at Re = 150.
Writes results/stage4_dense.npz.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import stage4_orbi as s4  # noqa: E402
from src import operators, pod, solver  # noqa: E402
from src.basis_interp import ORBI, itsgm  # noqa: E402
from src.galerkin_rom import GalerkinROM  # noqa: E402

SETS = {"original": [25, 50, 100, 200, 400],
        "denser": [25, 35, 50, 70, 100, 150, 200, 280, 400]}
TESTS = [80, 125]
R_LIST = [5, 10, 15, 20, 30]
METHODS = ["order 0", "ORBI 1", "ORBI 2", "ITSGM 2", "global"]


def main():
    g = solver.Grid(129)
    ps = operators.PoissonSolver(g)
    zero = np.zeros(g.N)
    allre = sorted(set(sum(SETS.values(), [])) | set(TESTS))
    data = {Re: s4.load_omega(Re) for Re in allre}
    shape = (len(SETS), len(TESTS), len(R_LIST), len(METHODS))
    rom_err, proj_err, angle = (np.full(shape, np.nan) for _ in range(3))
    for iset, (name, train) in enumerate(SETS.items()):
        s4.TRAIN = train                       # used by s4.neighbours()
        loc = {Re: pod.pod_svd(data[Re].T)[0][:, :max(R_LIST)] for Re in train}
        Ug = pod.pod_svd(np.hstack([data[Re].T for Re in train]))[0]
        for it, Re in enumerate(TESTS):
            W = data[Re]
            Y = W[1:].T
            nrm = np.linalg.norm(W[1:], axis=1)
            pair, triple = s4.neighbours(Re)
            for ir, r in enumerate(R_LIST):
                for im, m in enumerate(METHODS):
                    if m == "order 0":
                        X = loc[pair[0]][:, :r]
                    elif m == "global":
                        X = Ug[:, :r]
                    else:
                        nodes = pair if m.endswith("1") else triple
                        B = [loc[x][:, :r] for x in nodes]
                        if m.startswith("ORBI"):
                            o = ORBI(B, np.log10(nodes))
                            X = o(np.log10(Re))
                            angle[iset, it, ir, im] = o.max_angle
                        else:
                            X = itsgm(B, np.log10(nodes), np.log10(Re))
                    proj_err[iset, it, ir, im] = (np.linalg.norm(Y - X @ (X.T @ Y), axis=0) / nrm).max()
                    rom = GalerkinROM(g, Re, X, zero, poisson=ps)
                    rom.add_upwind_full()
                    try:
                        tr, _, _ = rom.integrate(X.T @ W[0], s4.DT, s4.NW)
                        e = np.linalg.norm(rom.reconstruct(tr)[1:] - W[1:], axis=1) / nrm
                        rom_err[iset, it, ir, im] = e.max()
                    except (FloatingPointError, RuntimeError):
                        rom_err[iset, it, ir, im] = np.inf
            print(f"{name:8s} Re={Re}: neighbours {pair} / {triple}  ORBI 2 ROM: "
                  + ", ".join(f"r={r}: {rom_err[iset, it, ir, METHODS.index('ORBI 2')]:.1e}"
                              for ir, r in enumerate(R_LIST)), flush=True)

    # diagnostics: energy left out by r local modes, and mode matching between neighbours
    r_diag = np.arange(1, 31)
    tail = {}
    for Re in (70, 100, 150):
        sv = pod.pod_svd(data[Re].T)[1]
        tail[Re] = np.array([np.sqrt(np.sum(sv[r:] ** 2) / np.sum(sv ** 2)) for r in r_diag])
    svg = pod.pod_svd(np.hstack([data[Re].T for Re in SETS["original"]]))[1]
    tail_global = np.array([np.sqrt(np.sum(svg[r:] ** 2) / np.sum(svg ** 2)) for r in r_diag])
    U1 = pod.pod_svd(data[100].T)[0][:, :30]
    U2 = pod.pod_svd(data[150].T)[0][:, :30]
    C = np.abs(U1.T @ U2)
    np.savez(os.path.join(ROOT, "results", "stage4_dense.npz"),
             sets=np.array(list(SETS)), train_original=np.array(SETS["original"]),
             train_denser=np.array(SETS["denser"]), tests=np.array(TESTS),
             r_list=np.array(R_LIST), methods=np.array(METHODS),
             rom_err=rom_err, proj_err=proj_err, max_angle=angle, r_diag=r_diag,
             tail_Re70=tail[70], tail_Re100=tail[100], tail_Re150=tail[150],
             tail_global=tail_global, cos_kk_100_150=np.diag(C).copy(),
             best_match_100_150=np.argmax(C, axis=1) + 1)
    print("saved results/stage4_dense.npz")


if __name__ == "__main__":
    main()
