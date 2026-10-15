"""Proper orthogonal decomposition (POD) utilities.

On the uniform grid used here the discrete L2 inner product is the Euclidean
one times dx*dy, so the plain SVD of the snapshot matrix gives L2-optimal
modes. Relative errors do not depend on the constant weight.
"""
import numpy as np


def pod_svd(X):
    """Thin SVD of the snapshot matrix X (N, m). Returns modes and singular values."""
    Phi, s, _ = np.linalg.svd(X, full_matrices=False)
    return Phi, s


def pod_snapshots(X):
    """Method of snapshots (Sirovich 1987): eigenvectors of the m x m correlation matrix.

    Gives the same modes as the thin SVD (up to sign). Kept to cross-check.
    """
    C = X.T @ X
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, V = np.clip(lam[order], 0, None), V[:, order]
    keep = lam > lam[0] * 1e-28
    s = np.sqrt(lam[keep])
    Phi = X @ V[:, keep] / s
    return Phi, s


def cumulative_energy(s):
    e = s ** 2
    return np.cumsum(e) / np.sum(e)


def rank_for_energy(s, threshold):
    """Smallest r whose cumulative energy (sum of s^2) reaches the threshold."""
    return int(np.searchsorted(cumulative_energy(s), threshold) + 1)


def projection_error(X, Phi, r):
    """Norm of the part of each column of X outside span(Phi[:, :r]) (best-approximation error)."""
    P = Phi[:, :r]
    R = X - P @ (P.T @ X)
    return np.linalg.norm(R, axis=0)


def deim_points(U):
    """Greedy DEIM point selection (Chaturantabut & Sorensen, SIAM J. Sci. Comput. 2010).

    U: (N, m) basis of the nonlinear-term snapshots. Returns m row indices.
    """
    N, m = U.shape
    p = [int(np.argmax(np.abs(U[:, 0])))]
    for j in range(1, m):
        c = np.linalg.solve(U[p, :j], U[p, j])
        res = U[:, j] - U[:, :j] @ c
        p.append(int(np.argmax(np.abs(res))))
    return np.array(p)
