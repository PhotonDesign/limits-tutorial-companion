"""Plane-wave (angular-spectrum) representations of focused fields.

A field built from propagating plane waves with k_z >= 0 is written (tutorial Eq. PWEv)

    E(x) = sum_p  int d^2k_par  c_p(k_par) e_p(k_par) exp(i k.x),

with |k| = k and two transverse polarization vectors e_p(k_par).  The z-directed power
is P = (2 pi^2 / Z0) sum_p int d^2k_par (k_z/k) |c_p|^2.

Everything here is evaluated by brute-force quadrature over the spherical cap of
allowed propagation directions, with *explicit* polarization vectors, so that it is
independent of the polarization-sum identity sum_p e_p e_p^dagger = I - khat khat^T
used in the analytical derivations.

Useful change of variables: d^2k_par = k^2 cos(theta) dOmega, so
    d^2k_par (k/k_z) = k^2 dOmega      (power-normalized weight),
    d^2k_par         = k^2 cos(theta) dOmega  (amplitude-normalized weight).
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def cap_quadrature(theta_max, n_theta=200, n_phi=256, hemisphere=+1):
    """Quadrature nodes/weights for integrals over a spherical cap of directions.

    Returns unit vectors khat (N,3), polar angle theta (N,), azimuth phi (N,), and
    solid-angle weights w (N,) such that sum(w f) ~ int_cap f dOmega.
    Gauss-Legendre in theta, uniform (spectrally accurate, periodic) in phi.
    hemisphere=+1 gives k_z >= 0, hemisphere=-1 gives the mirror cap k_z <= 0.
    """
    x, wx = leggauss(n_theta)
    th = 0.5 * theta_max * (x + 1.0)
    wth = 0.5 * theta_max * wx * np.sin(th)
    ph = 2.0 * np.pi * np.arange(n_phi) / n_phi
    wph = 2.0 * np.pi / n_phi
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    W = np.outer(wth, np.full(n_phi, wph))
    TH, PH, W = TH.ravel(), PH.ravel(), W.ravel()
    khat = np.stack([np.sin(TH) * np.cos(PH),
                     np.sin(TH) * np.sin(PH),
                     hemisphere * np.cos(TH)], axis=1)
    return khat, TH, PH, W


def polarization_basis(khat):
    """Explicit orthonormal transverse polarizations (TE 's', TM 'p') for each khat.

    e_s = (-sin phi, cos phi, 0) (azimuthal unit vector) and e_p = e_s x khat, so that
    (e_s, e_p, khat) is a right-handed orthonormal triad for either hemisphere.
    Returns array (N, 2, 3).
    """
    kx, ky, kz = khat.T
    kpar = np.hypot(kx, ky)
    # azimuthal unit vector; at normal incidence choose phi = 0
    cph = np.where(kpar > 1e-14, kx / np.where(kpar > 0, kpar, 1), 1.0)
    sph = np.where(kpar > 1e-14, ky / np.where(kpar > 0, kpar, 1), 0.0)
    e_s = np.stack([-sph, cph, np.zeros_like(kx)], axis=1)
    e_p = np.cross(e_s, khat)
    return np.stack([e_s, e_p], axis=1)


def optimal_coefficients(khat, q, normalization):
    """Optimal plane-wave coefficients (tutorial Eqs. cOpt, alphaOpt), with explicit
    polarization vectors: c_p = (k/k_z)^n [q^* . e_p]^*, with n = 1 for
    'power' normalization and n = 0 for 'amplitude' normalization.
    Returned as the *vector* amplitude sum_p c_p e_p, shape (N, 3), together with
    the per-polarization coefficients (N, 2).
    """
    E = polarization_basis(khat)                       # (N,2,3)
    proj = np.einsum("j,npj->np", np.conj(q), E)      # q^* . e_p
    c = np.conj(proj)
    if normalization == "power":
        c = c / np.abs(khat[:, 2])[:, None]            # k/k_z
    elif normalization != "amplitude":
        raise ValueError(normalization)
    vec = np.einsum("np,npj->nj", c, E)
    return c, vec


def field(points, khat, w, vec, k=1.0):
    """E(x) = int d^2k_par  vec(k_par) exp(i k.x), with vec = sum_p c_p e_p.

    The cap quadrature is in solid angle, so d^2k_par = k^2 |cos theta| dOmega is
    applied internally.  points: (M,3) array of positions.
    """
    phase = np.exp(1j * k * points @ khat.T)          # (M,N)
    jac = k**2 * np.abs(khat[:, 2]) * w                # d^2k_par weights
    return phase @ (jac[:, None] * vec)


def power_z(khat, w, c, k=1.0, Z0=1.0):
    """z-directed power P = (2 pi^2/Z0) sum_p int d^2k_par (k_z/k)|c_p|^2."""
    jac = k**2 * np.abs(khat[:, 2]) * w
    return 2 * np.pi**2 / Z0 * np.sum(jac * np.abs(khat[:, 2]) * np.sum(np.abs(c)**2, axis=1))


def amplitude_norm(khat, w, c, k=1.0, Z0=1.0):
    """'Amplitude normalization' written with the same prefactor as the power:
    P_amp = (2 pi^2/Z0) sum_p int d^2k_par |c_p|^2."""
    jac = k**2 * np.abs(khat[:, 2]) * w
    return 2 * np.pi**2 / Z0 * np.sum(jac * np.sum(np.abs(c)**2, axis=1))


def focal_matrix(theta_max, normalization, n_theta=400, n_phi=256):
    """3x3 matrix M whose Rayleigh quotient q^dag M q gives the maximal
    eps0|q^*.E(0)|^2 / P (times 2 pi^2 c).  Built with *explicit* polarization vectors:
        M = int d^2k_par  w(k)  sum_p e_p e_p^dagger,
    w = k/k_z for power normalization and w = 1 for amplitude normalization.
    Returned in units of k^2 (k = 1).
    """
    khat, th, ph, W = cap_quadrature(theta_max, n_theta, n_phi)
    E = polarization_basis(khat)
    jac = np.abs(khat[:, 2]) * W                       # d^2k_par / k^2
    if normalization == "power":
        wt = jac / np.abs(khat[:, 2])
    else:
        wt = jac
    return np.einsum("n,npi,npj->ij", wt, E, np.conj(E))
