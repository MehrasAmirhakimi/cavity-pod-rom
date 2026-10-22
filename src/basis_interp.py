"""POD basis adaptation to a new parameter value: ORBI and ITSGM.

Implements, for orthonormal bases X_0, ..., X_q in R^{n x p} computed at
parameters mu_0, ..., mu_q:

* ITSGM (interpolation on the tangent space to the Grassmann manifold),
  order q: Goutaudier, Nobile & Schiffmann (2023), Eqs. (16)-(19).
* ORBI (ordered reduced basis interpolation), order q: same paper,
  Algorithms 1, 2 and 4. The rotation Q_0i in SO(m) maps the reference basis
  X_0 onto X_i F_i (F_i flips mode signs), its principal real logarithm is
  interpolated with Lagrange polynomials, and the result is mapped back with
  the matrix exponential: X(mu) = W expm(A(mu)) Y_0.

Reference: D. Goutaudier, F. Nobile, J. Schiffmann, A new method to interpolate
POD reduced bases - Application to the parametric model order reduction of a
gas bearings supported rotor, Int. J. Numer. Methods Eng. 124(18), 4141-4170,
2023. https://doi.org/10.1002/nme.7305

Implementation notes (my choices, not from the paper):
* The real principal logarithm of Q_0i is computed with scipy.linalg.logm
  instead of the direct rotation decomposition (Algorithm 3). Both give the
  unique real skew-symmetric principal logarithm when -1 is not an eigenvalue
  of Q_0i; we check this and report the largest rotation angle.
* The basis W of X_0 + ... + X_q is the thin SVD of [X_0 ... X_q], keeping
  singular values above 1e-10 of the largest.
"""
import numpy as np
from scipy.linalg import expm, logm, null_space


def lagrange_weights(nodes, x):
    """Lagrange basis polynomials L_i(x) for the given nodes."""
    nodes = np.asarray(nodes, float)
    L = np.ones(len(nodes))
    for i, xi in enumerate(nodes):
        for j, xj in enumerate(nodes):
            if j != i:
                L[i] *= (x - xj) / (xi - xj)
    return L


# ----------------------------------------------------------------------------- ITSGM
def itsgm(bases, nodes, x, ref=0):
    """ITSGM of order q = len(bases) - 1 (Eqs. 16-19). Returns an n x p basis."""
    X0 = bases[ref]
    L = lagrange_weights(nodes, x)
    T = np.zeros_like(X0)
    for i, Xi in enumerate(bases):
        if i == ref:
            continue
        M = X0.T @ Xi
        Z = (Xi - X0 @ M) @ np.linalg.inv(M)                   # Eq. (16)
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        T += L[i] * (U * np.arctan(s)) @ Vt                     # Eqs. (17)-(18)
    U, th, Vt = np.linalg.svd(T, full_matrices=False)
    V = Vt.T
    return (X0 @ V) * np.cos(th) @ V.T + (U * np.sin(th)) @ V.T   # Eq. (19)


# ----------------------------------------------------------------------------- ORBI
def _flipping(Y0, Yi, Y0p, Yip):
    """Algorithm 2: sign flips F so that det([Yi F, Yip] [Y0, Y0p]^T) = +1."""
    cos_g = np.einsum("ij,ij->j", Y0, Yi)
    f = np.where(cos_g < 0, -1.0, 1.0)
    Yb0 = np.hstack([Y0, Y0p])
    if np.linalg.det(np.hstack([Yi * f, Yip]) @ Yb0.T) < 0:
        k = int(np.argmin(np.einsum("ij,ij->j", Y0, Yi * f)))
        f[k] = -f[k]
    return f


def _rotation(Y0, Yi):
    """Algorithm 1, lines 3-7: rotation Q_0i in SO(m) with Q_0i Y0 = Yi F."""
    Y0p, Yip = null_space(Y0.T), null_space(Yi.T)
    if Y0p.shape[1] == 0:                       # W = X_0 = X_i (no complement)
        f = _flipping(Y0, Yi, Y0p, Yip)
        return (Yi * f) @ Y0.T, f
    P2, _, Q2t = np.linalg.svd(Y0p.T @ Yip)
    Y0p, Yip = Y0p @ P2, Yip @ Q2t.T            # principal vectors in the complements
    f = _flipping(Y0, Yi, Y0p, Yip)
    Q = np.hstack([Yi * f, Yip]) @ np.hstack([Y0, Y0p]).T
    return Q, f


class ORBI:
    """Ordered reduced basis interpolation of order q (Algorithm 4).

    Offline (constructor): W, Y_0 and the logarithms A'_i of Q_0i.
    Online (__call__): X(mu) = W expm(sum_i L_i(mu) A'_i) Y_0.
    """

    def __init__(self, bases, nodes, ref=0):
        self.nodes = np.asarray(nodes, float)
        self.ref = ref
        Wf, s, _ = np.linalg.svd(np.hstack(bases), full_matrices=False)
        self.W = Wf[:, s > 1e-10 * s[0]]
        self.Y0 = self.W.T @ bases[ref]
        m = self.W.shape[1]
        self.logs, self.flips, self.max_angle = [], [], 0.0
        for i, Xi in enumerate(bases):
            if i == ref:
                self.logs.append(np.zeros((m, m)))
                self.flips.append(np.ones(Xi.shape[1]))
                continue
            Q, f = _rotation(self.Y0, self.W.T @ Xi)
            ang = np.abs(np.angle(np.linalg.eigvals(Q))).max()
            if ang > np.pi - 1e-6:
                raise ValueError("-1 is an eigenvalue of Q_0i: no unique principal logarithm")
            A = logm(Q)
            if np.abs(A.imag).max() > 1e-8:
                raise ValueError("logarithm of Q_0i is not real")
            A = A.real
            self.logs.append(0.5 * (A - A.T))
            self.flips.append(f)
            self.max_angle = max(self.max_angle, ang)

    def __call__(self, x):
        L = lagrange_weights(self.nodes, x)
        A = sum(Li * Ai for Li, Ai in zip(L, self.logs))
        return self.W @ (expm(A) @ self.Y0)
