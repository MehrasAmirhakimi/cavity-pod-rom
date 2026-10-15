"""Centreline velocities of Ghia, Ghia & Shin (1982) for the lid-driven cavity.

Reference (checked against the Crossref record of the DOI):
U. Ghia, K. N. Ghia, C. T. Shin, "High-Re solutions for incompressible flow
using the Navier-Stokes equations and a multigrid method",
Journal of Computational Physics 48(3), 387-411, 1982.
https://doi.org/10.1016/0021-9991(82)90058-4

The numbers were taken from a machine-readable copy of Tables I and II
(barbagroup cuIBM/PetIBM: data/ghia_et_al_1982_lid_driven_cavity.dat) and
cross-checked against a second, independent copy (PN-CourseWork/
02689-AdvancedNumericalAlgorithmP3: data/validation/ghia/*.csv). For Re = 100
all values agree. For Re = 1000 the two copies disagree at one point, v at
x = 0.5 (0.02426 vs 0.02526), so that point is left out. Re = 400 is not
included because only one copy was available.

u is tabulated on the vertical centreline x = 0.5, v on the horizontal
centreline y = 0.5. The lid moves in +x at y = 1.
"""
import numpy as np

_Y = np.array([0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1])
_X = np.array([0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1])
_U = {100: np.array([0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.1015, -0.15662, -0.2109, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1]),
      1000: np.array([0.0, -0.18109, -0.20196, -0.2222, -0.2973, -0.38289, -0.27805, -0.10648, -0.0608, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 0.57492, 0.65928, 1])}
_V = {100: np.array([0.0, 0.09233, 0.10091, 0.1089, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.0]),
      1000: np.array([0.0, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075, 0.32235, 0.02426, -0.31966, -0.42665, -0.5155, -0.39188, -0.33714, -0.27669, -0.21388, 0.0])}
_V_EXCLUDE = {1000: 0.5}   # disputed point (see module docstring)


def table(Re):
    """Return (y, u, x, v) for Re = 100 or 1000, sorted by coordinate."""
    y, u = _Y.copy(), _U[Re].copy()
    x, v = _X.copy(), _V[Re].copy()
    if Re in _V_EXCLUDE:
        keep = ~np.isclose(x, _V_EXCLUDE[Re])
        x, v = x[keep], v[keep]
    iy, ix = np.argsort(y), np.argsort(x)
    return y[iy], u[iy], x[ix], v[ix]
