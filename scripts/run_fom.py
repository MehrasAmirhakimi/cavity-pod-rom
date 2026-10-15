"""Run the full-order transient solver and save the snapshots.

Usage:  python scripts/run_fom.py --scheme upwind --im 129 --out data/fom_upwind_Re100_im129.npz
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import solver  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--Re", type=float, default=100)
p.add_argument("--Pr", type=float, default=7)
p.add_argument("--im", type=int, default=129)
p.add_argument("--tf", type=float, default=200.0)
p.add_argument("--nm", type=int, default=1000)
p.add_argument("--scheme", default="upwind")
p.add_argument("--solver", default="direct")
p.add_argument("--out", required=True)
a = p.parse_args()

t0 = time.perf_counter()
r = solver.run_transient(Re=a.Re, Pr=a.Pr, im=a.im, tf=a.tf, nm=a.nm,
                         linear_solver=a.solver, scheme=a.scheme, verbose=10)
total = time.perf_counter() - t0
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.savez(a.out, omega=r["omega"], psi=r["psi"], u=r["u"], v=r["v"], T=r["T"],
         t=r["t"], picard_iters=r["picard_iters"], wall_flow=r["wall_flow"],
         wall_temp=r["wall_temp"], total_wall=total, **{f"p_{k}": v for k, v in r["params"].items()})
print(f"done in {total:.1f} s, total Picard iterations {r['picard_iters'].sum()}")
