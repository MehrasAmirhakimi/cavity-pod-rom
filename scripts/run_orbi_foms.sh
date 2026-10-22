#!/usr/bin/env bash
# Transient full-order runs for the ORBI test (0 <= t <= 20, 100 steps, two at a time).
# Training Re: 25, 50, 100, 200, 400 (Re = 100 comes from the Stage 1 run).
# Test Re: 35, 70, 80, 125, 150, 280 (80 and 125 come from Stage 1).
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
printf "%s\n" 400 200 280 150 50 70 25 35 | xargs -P 2 -I{} sh -c \
  '[ -f data/fom_upwind_Re{}_im129_t20.npz ] || python scripts/run_fom.py --Re {} --tf 20 --nm 100 --out data/fom_upwind_Re{}_im129_t20.npz > data/log_fom_Re{}.txt 2>&1'
