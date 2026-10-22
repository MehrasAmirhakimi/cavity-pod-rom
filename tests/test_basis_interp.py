"""Tests for ORBI and ITSGM (src/basis_interp.py) on random bases."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.basis_interp import ORBI, itsgm, lagrange_weights  # noqa: E402


def _bases(n=200, p=4, q=2, eps=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X0 = np.linalg.qr(rng.normal(size=(n, p)))[0]
    out = [X0]
    for _ in range(q):
        out.append(np.linalg.qr(X0 + eps * rng.normal(size=(n, p)))[0])
    return out


def _proj_dist(A, B):
    """Distance between the subspaces spanned by A and B (2-norm of P_A - P_B)."""
    return np.linalg.norm(A @ A.T - B @ B.T, 2)


def test_lagrange_weights():
    L = lagrange_weights([0.0, 1.0, 3.0], 1.0)
    assert np.allclose(L, [0, 1, 0])
    assert np.isclose(lagrange_weights([0.0, 1.0, 3.0], 2.2).sum(), 1.0)


@pytest.mark.parametrize("q", [1, 2])
def test_orbi_reproduces_training_bases(q):
    """At node i, ORBI returns X_i F_i: the training basis up to mode signs."""
    B = _bases(q=q)
    nodes = np.arange(q + 1, dtype=float)
    orbi = ORBI(B, nodes)
    for i, Xi in enumerate(B):
        X = orbi(nodes[i])
        assert np.allclose(X, Xi * orbi.flips[i], atol=1e-10)


@pytest.mark.parametrize("q", [1, 2])
def test_outputs_are_orthonormal(q):
    B = _bases(q=q)
    nodes = np.arange(q + 1, dtype=float)
    orbi = ORBI(B, nodes)
    for x in (0.3, 0.77, 1.5):
        for X in (orbi(x), itsgm(B, nodes, x)):
            assert np.allclose(X.T @ X, np.eye(X.shape[1]), atol=1e-10)


def test_itsgm_reproduces_training_subspaces():
    B = _bases(q=2)
    nodes = np.array([0.0, 1.0, 2.0])
    for i, Xi in enumerate(B):
        assert _proj_dist(itsgm(B, nodes, nodes[i]), Xi) < 1e-10


def test_orbi_equals_itsgm_for_principal_vector_bases():
    """Paper, Section 4.5 (Eq. 28): if the inputs are bases of principal vectors,
    ORBI and ITSGM give the same subspace. This checks both implementations."""
    X0, X1 = _bases(q=1)
    P, c, Qt = np.linalg.svd(X0.T @ X1)
    Xa, Xb = X0 @ P, X1 @ Qt.T          # principal vectors (cosines c > 0)
    assert np.all(c > 0)
    orbi = ORBI([Xa, Xb], [0.0, 1.0])
    for s in (0.25, 0.5, 0.8):
        assert _proj_dist(orbi(s), itsgm([Xa, Xb], [0.0, 1.0], s)) < 1e-9


def test_orbi_differs_from_itsgm_for_pod_ordered_bases():
    """For general (POD-ordered) bases the two methods give different subspaces."""
    X0, X1 = _bases(q=1, seed=3)
    orbi = ORBI([X0, X1], [0.0, 1.0])
    assert _proj_dist(orbi(0.5), itsgm([X0, X1], [0.0, 1.0], 0.5)) > 1e-3
