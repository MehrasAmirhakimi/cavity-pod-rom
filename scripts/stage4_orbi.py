"""Stage 4: adapting the transient POD basis to a new Reynolds number.

Compares, for the intrusive POD-Galerkin ROM of Stage 1 (upwind scheme):
  * order 0      - basis of the nearest training Re (no adaptation)
  * ITSGM 1 / 2  - interpolation on the tangent space to the Grassmann manifold
  * ORBI 1 / 2   - ordered reduced basis interpolation (Goutaudier et al., 2023)
  * global       - one POD basis of all training snapshots (SNAPC in the paper)

Training Re: 25, 50, 100, 200, 400. Test Re (never used for training):
35, 70, 80, 125, 150, 280. All runs: 0 <= t <= 20, dt = 0.2 (100 steps).

Order 1 uses the two training Re that bracket the target; order 2 adds the
next-nearest one. The reference point of the interpolation is the nearest
training Re. Interpolation variable: log10(Re).

Snapshots are used raw (not centred), as in the paper. The lifting field is
omega_ref = 0 in the interior: the moving lid enters only through Thom's wall
vorticity of the reference (-2/dy on the lid), so the boundary conditions hold
for every basis and nothing has to be interpolated except the basis itself.
(A first version interpolated a t = 20 reference field between training Re.
Its interpolation error is not in any basis and put a floor of about 1 % under
all methods, so it was dropped.)

Main ROM: Galerkin projection of the full upwind scheme, evaluated on the whole
grid. This isolates the effect of the basis. A DEIM version (common DEIM basis
from all training Re) gives the online cost.

Writes results/stage4.npz. Needs the runs from scripts/run_orbi_foms.sh.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import operators, pod, solver  # noqa: E402
from src.basis_interp import ORBI, itsgm  # noqa: E402
from src.galerkin_rom import GalerkinROM  # noqa: E402

TRAIN = [25, 50, 100, 200, 400]
TEST = [35, 70, 80, 125, 150, 280]
R_LIST = [3, 5, 8, 10, 15, 20, 30]
METHODS = ["order 0", "ITSGM 1", "ORBI 1", "ITSGM 2", "ORBI 2", "global"]
NW, DT = 100, 0.2
R_DEIM = 10


def load_omega(Re):
    f = os.path.join(ROOT, "data", f"fom_upwind_Re{Re}_im129_t20.npz")
    if Re == 100 and not os.path.exists(f):
        f = os.path.join(ROOT, "data", "fom_upwind_Re100_im129.npz")
    W = np.load(f)["omega"][:NW + 1]
    assert W.shape[0] == NW + 1
    return W


def neighbours(Re):
    """Bracketing pair (order 1) and three nearest (order 2) training Re, nearest first."""
    lx, lt = np.log10(Re), np.log10(TRAIN)
    k = int(np.searchsorted(TRAIN, Re))
    pair = [TRAIN[k - 1], TRAIN[k]] if 0 < k < len(TRAIN) else [TRAIN[min(k, len(TRAIN) - 1)]]
    order = list(np.array(TRAIN)[np.argsort(np.abs(lt - lx))])
    triple = list(pair) + [r for r in order if r not in pair][:1]
    pair.sort(key=lambda r: abs(np.log10(r) - lx))
    triple.sort(key=lambda r: abs(np.log10(r) - lx))
    return pair, triple


def main(out="stage4.npz", im=129):
    g = solver.Grid(im)
    ps = operators.PoissonSolver(g)
    data = {Re: load_omega(Re) for Re in TRAIN + TEST}
    zero = np.zeros(g.N)
    refs = {Re: zero for Re in TRAIN}
    rmax = max(R_LIST)
    local = {}
    for Re in TRAIN:
        U, s = pod.pod_svd((data[Re] - refs[Re]).T)
        local[Re] = U[:, :rmax]
    Xg = np.hstack([(data[Re] - refs[Re]).T for Re in TRAIN])
    Ug, sg = pod.pod_svd(Xg)
    res = dict(train=np.array(TRAIN), test=np.array(TEST), r_list=np.array(R_LIST),
               methods=np.array(METHODS), s_global=sg)

    # common DEIM basis of the upwind term (all training snapshots)
    ft = operators.field_terms(g, np.hstack([data[Re].T for Re in TRAIN]), ps, lid=True)
    Un, sn = pod.pod_svd(operators.upwind_dissipation(g, ft["u"], ft["v"], ft["d2x"], ft["d2y"]))
    res["s_nl"] = sn
    del ft

    evalRe = sorted(TRAIN + TEST)
    res["eval_Re"] = np.array(evalRe)
    shape = (len(METHODS), len(R_LIST), len(evalRe))
    proj_max, rom_max, proj_fro = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
    t_adapt, t_build, angles = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
    rom_curves = {}
    for ie, Re in enumerate(evalRe):
        W = data[Re]
        wref = zero
        Y = (W[1:] - wref).T                              # snapshots to approximate, t > 0
        nrm = np.linalg.norm(W[1:], axis=1)
        pair, triple = neighbours(Re)
        lx = np.log10(Re)
        for ir, r in enumerate(R_LIST):
            for im, meth in enumerate(METHODS):
                t0 = time.perf_counter()
                if meth == "order 0":
                    X = local[pair[0]][:, :r]
                elif meth == "global":
                    X = Ug[:, :r]
                else:
                    nodes_re = pair if meth.endswith("1") else triple
                    if len(nodes_re) < (2 if meth.endswith("1") else 3):
                        continue
                    B = [local[x][:, :r] for x in nodes_re]
                    nodes = np.log10(nodes_re)
                    if meth.startswith("ORBI"):
                        o = ORBI(B, nodes)                 # offline part
                        t0 = time.perf_counter()
                        X = o(lx)                          # online part
                        angles[im, ir, ie] = o.max_angle
                    else:
                        X = itsgm(B, nodes, lx)
                t_adapt[im, ir, ie] = time.perf_counter() - t0
                R = Y - X @ (X.T @ Y)
                proj_max[im, ir, ie] = (np.linalg.norm(R, axis=0) / nrm).max()
                proj_fro[im, ir, ie] = np.linalg.norm(R) / np.linalg.norm(Y)
                t0 = time.perf_counter()
                rom = GalerkinROM(g, Re, X, wref, poisson=ps)
                rom.add_upwind_full()
                t_build[im, ir, ie] = time.perf_counter() - t0
                try:
                    tr, _, _ = rom.integrate(X.T @ (W[0] - wref), DT, NW)
                    e = np.linalg.norm(rom.reconstruct(tr)[1:] - W[1:], axis=1) / nrm
                except (FloatingPointError, RuntimeError):
                    e = np.full(NW, np.inf)
                rom_max[im, ir, ie] = e.max()
                if r == 10:
                    rom_curves[(meth, Re)] = e
            print(f"Re={Re:4d} r={r:2d}  " + "  ".join(
                f"{m}: {rom_max[i, ir, ie]:.1e}/{proj_max[i, ir, ie]:.1e}"
                for i, m in enumerate(METHODS)), flush=True)
    res.update(proj_max=proj_max, rom_max=rom_max, proj_fro=proj_fro, t_adapt=t_adapt,
               t_build=t_build, max_angle=angles)
    for (m, Re), e in rom_curves.items():
        res[f"curve_{m.replace(' ', '')}_Re{Re}"] = e

    # hyper-reduced (DEIM) version for ORBI 2 and global at r = R_DEIM, with timings
    ir = R_LIST.index(R_DEIM)
    cap = int(np.sum(sn / sn[0] >= 1e-8))
    m_list = [m for m in (2 * R_DEIM, 4 * R_DEIM, 6 * R_DEIM) if m <= cap]
    res["deim_m_list"], res["deim_cap"] = np.array(m_list), cap
    deim = np.full((2, len(m_list), len(TEST)), np.nan)
    deim_wall = np.full((2, len(m_list), len(TEST)), np.nan)
    for it, Re in enumerate(TEST):
        W = data[Re]
        wref = zero
        nrm = np.linalg.norm(W[1:], axis=1)
        _, triple = neighbours(Re)
        Xo = ORBI([local[x][:, :R_DEIM] for x in triple], np.log10(triple))(np.log10(Re))
        for k, X in enumerate((Xo, Ug[:, :R_DEIM])):
            base = GalerkinROM(g, Re, X, wref, poisson=ps)
            for jm, m in enumerate(m_list):
                rom = GalerkinROM.truncated(base, R_DEIM)
                rom.set_Re(Re)
                rom.add_deim(Un[:, :m], pod.deim_points(Un[:, :m]))
                best, e = np.inf, None
                for _ in range(3):
                    tr, wt, _ = rom.integrate(X.T @ (W[0] - wref), DT, NW)
                    best = min(best, wt)
                e = np.linalg.norm(rom.reconstruct(tr)[1:] - W[1:], axis=1) / nrm
                deim[k, jm, it], deim_wall[k, jm, it] = e.max(), best
        print(f"DEIM Re={Re}: ORBI 2 " + ", ".join(f"m={m}: {deim[0, j, it]:.1e}" for j, m in enumerate(m_list))
              + " | global " + ", ".join(f"m={m}: {deim[1, j, it]:.1e}" for j, m in enumerate(m_list)), flush=True)
    res.update(deim_err=deim, deim_wall=deim_wall)

    # full-solver cost of the same 100 steps at the test Re
    fw = []
    for Re in TEST:
        f = os.path.join(ROOT, "data", f"fom_upwind_Re{Re}_im129_t20.npz")
        fw.append(np.load(f)["wall_flow"].sum() if os.path.exists(f) else np.nan)
    res["fom_wall_test"] = np.array(fw)
    np.savez(os.path.join(ROOT, "results", out), **res)
    print(f"saved results/{out}")


if __name__ == "__main__":
    main()
