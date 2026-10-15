#!/usr/bin/env bash
# Reproduce every result from scratch (about 30 min on 2 CPU cores).
# Full-order data go to data/ (about 1.4 GB, not in git); results/ and figures/ are small.
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest -q tests
python scripts/run_fom.py --scheme upwind  --out data/fom_upwind_Re100_im129.npz
python scripts/run_fom.py --scheme central --out data/fom_central_Re100_im129.npz
python scripts/run_fom.py --Re 80  --tf 20 --nm 100 --out data/fom_upwind_Re80_im129_t20.npz
python scripts/run_fom.py --Re 125 --tf 20 --nm 100 --out data/fom_upwind_Re125_im129_t20.npz
python scripts/run_steady_sweep.py
python scripts/stage1.py
python scripts/stage2.py
python scripts/stage3.py
python scripts/make_tables.py
