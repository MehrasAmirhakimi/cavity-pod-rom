"""Regression tests: vectorised solver vs the original loop-based scripts.

The original scripts in ``legacy/`` are executed as they are. We only change
the parameter lines (Re, grid size, number of steps) and stop before the
plotting section. The vectorised solver, in GMRES mode, must give the same
fields to round-off.

Run with:  python -m pytest -q tests
"""
import contextlib
import io
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import solver  # noqa: E402

LEGACY = os.path.join(ROOT, "legacy")


def _run_legacy(fname, params):
    """Execute a legacy script with new parameter values; return its namespace."""
    src = open(os.path.join(LEGACY, fname)).read()
    src = src.split("\nx=np.linspace(0,1,im+1)")[0]          # drop plotting part
    for name, val in params.items():
        src, nsub = re.subn(rf"^{name}=.*$", f"{name}={val}", src, flags=re.M)
        assert nsub == 1, f"parameter line {name}= not found exactly once"
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, fname, "exec"), ns)
    return ns


def _relmax(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


@pytest.fixture(scope="module")
def transient_pair():
    p = dict(Re=100, Pr=7, im=33, tf=1.0, nm=5)
    ref = _run_legacy("CavityFlow_CSR_er_Temp2.py",
                      dict(Re=p["Re"], im=p["im"], jm=p["im"], tf=p["tf"], nm=p["nm"]))
    new = solver.run_transient(**p, linear_solver="gmres")
    return ref, new


def test_transient_matches_original_to_roundoff(transient_pair):
    ref, new = transient_pair
    for key, rkey in [("omega", "omega"), ("psi", "sai"), ("u", "u"), ("v", "v"),
                      ("T", "temp")]:
        err = _relmax(new[key], ref[rkey])
        assert err < 1e-12, f"{key}: max relative difference {err:.2e}"
    # same number of Picard iterations in total
    assert int(new["picard_iters"].sum()) == int(ref["j"])


def test_steady_matches_original_to_roundoff():
    p = dict(Re=100, im=33)
    ref = _run_legacy("CavityFlow_Steady_CSR_er_Temp2.py",
                      dict(Re=p["Re"], im=p["im"], jm=p["im"]))
    # this legacy file uses relaxation 0.1 and tolerance 1e-8
    new = solver.run_steady(Re=100, Pr=7, im=33, linear_solver="gmres",
                            tol=1e-8, komega=0.1, ksai=0.1)
    for key, rkey in [("omega", "omega"), ("psi", "sai"), ("u", "u"), ("v", "v"),
                      ("T", "temp")]:
        err = _relmax(new[key], ref[rkey])
        assert err < 1e-12, f"{key}: max relative difference {err:.2e}"
    assert new["picard_iters"] == int(ref["j"])


def test_direct_solver_close_to_gmres(transient_pair):
    """The direct solver solves the same equations more accurately.

    It is not round-off identical to GMRES (tolerance 1e-5), but close.
    """
    _, gm = transient_pair
    di = solver.run_transient(Re=100, Pr=7, im=33, tf=1.0, nm=5,
                              linear_solver="direct")
    assert _relmax(di["omega"], gm["omega"]) < 1e-3
