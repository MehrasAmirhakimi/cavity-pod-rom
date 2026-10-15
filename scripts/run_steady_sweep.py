"""Steady full-order solutions over a range of Re (Stage 2 snapshots).

Same discrete equations as legacy/CavityFlow_Steady_CSR_er_Temp2.py, solved
with exact (LU) linear solves. Each Re starts from the converged solution of
the previous (lower) Re; this changes only the number of Picard iterations,
not the converged discrete solution.

Usage:  python scripts/run_steady_sweep.py
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import operators, solver  # noqa: E402

TRAIN = [10, 20, 40, 75, 150, 200, 300, 400, 600, 800, 1000]
TEST = [30, 60, 100, 125, 250, 500, 700, 900]
TOL = 1e-9


def main():
    out_dir = os.path.join(ROOT, "data", "steady")
    os.makedirs(out_dir, exist_ok=True)
    g = solver.Grid(129)
    ps = operators.PoissonSolver(g)
    init = None
    for Re in sorted(TRAIN + TEST):
        f = os.path.join(out_dir, f"steady_Re{Re}.npz")
        if os.path.exists(f):
            d = np.load(f)
            init = {k: d[k] for k in ("omega", "psi", "u", "v")}
            continue
        t0 = time.perf_counter()
        r = solver.run_steady(Re=Re, Pr=7, im=129, linear_solver="direct", tol=TOL,
                              init=init)
        # residual of the steady discrete equations, psi recomputed exactly
        F = operators.rhs(g, r["omega"], Re, ps, scheme="upwind")
        scale = np.linalg.norm(operators.rhs(g, np.zeros(g.N), Re, ps, scheme="upwind"))
        res = np.linalg.norm(F) / scale
        np.savez(f, omega=r["omega"], psi=r["psi"], u=r["u"], v=r["v"], T=r["T"],
                 Re=Re, iters=r["picard_iters"], wall=r["wall"], residual=res,
                 nu_lid=solver.lid_nusselt(g, r["T"]))
        init = {k: r[k] for k in ("omega", "psi", "u", "v")}
        print(f"Re={Re:5d}  Picard {r['picard_iters']:6d}  {time.perf_counter()-t0:7.1f} s  "
              f"residual {res:.1e}  psi_min {r['psi'].min():.6f}  "
              f"Nu_lid {solver.lid_nusselt(g, r['T']):.4f}", flush=True)


if __name__ == "__main__":
    main()
