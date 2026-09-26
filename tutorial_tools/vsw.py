"""Vector spherical waves (VSWs) with normalized vector spherical harmonics.

    M_lm = (i / sqrt(l(l+1))) curl( r z_l(kr) Y_lm ) = z_l(kr) X_lm,
    N_lm = (1/k) curl M_lm
         = i [ sqrt(l(l+1)) z_l/(kr) Y_lm rhat + (1/sqrt(l(l+1))) (1/(kr)) d(kr z_l)/d(kr) grad_s Y_lm ],

where X_lm = L Y_lm / sqrt(l(l+1)) (Jackson), grad_s = thetahat d_theta + phihat (1/sin) d_phi,
and z_l is j_l (regular), h_l^(1) (outgoing) or h_l^(2) (incoming).  With this
normalization the tutorial's N_{1,0} has |N_{1,0}(0)| = (2/3) sqrt(3/(8 pi)).
Also curl N = k M, so for E = N the magnetic field is H = M/(i Z0).
"""
import numpy as np
from scipy.special import sph_harm_y, spherical_jn, spherical_yn


def _radial(l, x, kind):
    j = spherical_jn(l, x)
    jp = spherical_jn(l, x, derivative=True)
    if kind == "reg":
        return j, jp
    y = spherical_yn(l, x)
    yp = spherical_yn(l, x, derivative=True)
    s = 1 if kind == "out" else -1
    return j + s * 1j * y, jp + s * 1j * yp


def cart_to_sph(p):
    x, y, z = p.T
    r = np.sqrt(x**2 + y**2 + z**2)
    th = np.arccos(np.clip(z / np.where(r > 0, r, 1), -1, 1))
    ph = np.arctan2(y, x)
    return r, th, ph


def unit_vectors(th, ph):
    rh = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    thh = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)], -1)
    phh = np.stack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)], -1)
    return rh, thh, phh


def vsw(l, m, points, k=1.0, kind="reg"):
    """Return (M, N) at points (P,3), each (P,3) complex, Cartesian components."""
    r, th, ph = cart_to_sph(np.asarray(points, float))
    x = k * r
    Y, dY = sph_harm_y(l, m, th, ph, diff_n=1)
    dY_th = dY[..., 0]
    dY_ph = 1j * m * Y
    rh, thh, phh = unit_vectors(th, ph)
    sinth = np.where(np.abs(np.sin(th)) > 1e-300, np.sin(th), 1e-300)
    grad_s = thh * dY_th[:, None] + phh * (dY_ph / sinth)[:, None]
    L = np.sqrt(l * (l + 1))
    X = (-1j / L) * np.cross(rh, grad_s)             # X_lm = -i rhat x grad_s Y / L
    z, zp = _radial(l, x, kind)
    M = z[:, None] * X
    d_xz = z + x * zp                                  # d(x z)/dx
    N = 1j * (L * (z / x)[:, None] * Y[:, None] * rh + (1 / L) * (d_xz / x)[:, None] * grad_s)
    return M, N


def curl_numeric(F, points, h=1e-5):
    """Central-difference curl of a vector field F(points)->(P,3)."""
    p = np.asarray(points, float)
    J = np.zeros(p.shape + (3,), complex)              # J[:, i, j] = d F_i / d x_j
    for j in range(3):
        e = np.zeros(3); e[j] = h
        J[:, :, j] = (F(p + e) - F(p - e)) / (2 * h)
    return np.stack([J[:, 2, 1] - J[:, 1, 2], J[:, 0, 2] - J[:, 2, 0], J[:, 1, 0] - J[:, 0, 1]], -1)
