"""Non-intrusive parametric POD of steady solutions over the Reynolds number.

Offline: steady snapshots at training Re -> POD basis (after removing the
mean) -> modal coefficients a_k(Re) at the training points.
Online: interpolate a_k at a new Re and rebuild the field. Two interpolants:

* radial basis functions (scipy RBFInterpolator);
* Gaussian-process regression (scikit-learn), one GP per mode, which also
  gives a standard deviation for each coefficient.

The interpolation variable is log10(Re), because the fields change faster at
low Re on a linear scale.
"""
import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


class ParametricPOD:
    def __init__(self, Re_train, snapshots, r, rbf_kernel="cubic", gp_restarts=10, seed=0):
        """snapshots: (n_train, N) steady fields, one row per Re in Re_train."""
        self.Re_train = np.asarray(Re_train, float)
        self.x = np.log10(self.Re_train)[:, None]
        S = np.asarray(snapshots)
        self.mean = S.mean(axis=0)
        U, s, Vt = np.linalg.svd((S - self.mean).T, full_matrices=False)
        self.s_all = s
        self.r = r
        self.Phi = U[:, :r]
        self.coef = (S - self.mean) @ self.Phi             # (n_train, r)
        self.rbf = RBFInterpolator(self.x, self.coef, kernel=rbf_kernel)
        self.gps = []
        for k in range(r):
            kern = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(0.5, (1e-1, 1e1))
                    + WhiteKernel(1e-8, (1e-12, 1e-1)))
            gp = GaussianProcessRegressor(kern, normalize_y=True,
                                          n_restarts_optimizer=gp_restarts,
                                          random_state=seed + k)
            gp.fit(self.x, self.coef[:, k])
            self.gps.append(gp)

    def coef_rbf(self, Re):
        return self.rbf(np.log10(np.atleast_1d(Re))[:, None])

    def coef_gpr(self, Re):
        x = np.log10(np.atleast_1d(Re))[:, None]
        out = [gp.predict(x, return_std=True) for gp in self.gps]
        mu = np.column_stack([m for m, _ in out])
        sd = np.column_stack([s for _, s in out])
        return mu, sd

    def field(self, coef):
        return self.mean[None, :] + np.atleast_2d(coef) @ self.Phi.T

    def project(self, fields):
        return (np.atleast_2d(fields) - self.mean) @ self.Phi

    def gpr_samples(self, Re, n=500, seed=0):
        """Fields sampled from the GP posterior of the coefficients (modes independent)."""
        mu, sd = self.coef_gpr(Re)
        rng = np.random.default_rng(seed)
        a = mu[0] + sd[0] * rng.standard_normal((n, self.r))
        return self.field(a)
