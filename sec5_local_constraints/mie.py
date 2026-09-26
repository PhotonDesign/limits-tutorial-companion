"""Compact Mie theory for a homogeneous sphere in vacuum (Bohren & Huffman conventions,
e^{-i omega t}, Im m > 0 for loss), plus the eigenvalues rho_{n,p} of Im Gamma_0 on a
spherical domain (Sec. 5.1 of the tutorial, Eq. CSOTBound).

Units: all cross-sections are returned in units of 1/k^2 unless a wavenumber k is given.
Channel convention: a_n = electric multipole = TM = N-waves; b_n = magnetic = TE = M-waves.

    sigma_ext = (2 pi/k^2) sum_n (2n+1) Re(a_n + b_n)
    sigma_sca = (2 pi/k^2) sum_n (2n+1) (|a_n|^2 + |b_n|^2)

Note (2 pi/k^2) = lambda^2/(2 pi), the per-channel prefactor of Eq. CSvswChannelBound.
"""
from functools import lru_cache
import numpy as np
from scipy.special import spherical_jn, spherical_yn


@lru_cache(maxsize=16)
def _leggauss(n):
    return np.polynomial.legendre.leggauss(n)


def n_max(x):
    """Wiscombe-type truncation order for size parameter x."""
    x = np.max(np.abs(np.atleast_1d(x)))
    return int(np.ceil(x + 4.05 * x ** (1 / 3) + 2)) + 2


def mie_coefficients(m, x, nmax=None):
    """Mie coefficients a_n, b_n (n = 1..nmax) for relative index m and size parameter
    x = k R.  m, x may be arrays of equal shape (vectorized over e.g. frequency).
    Returns (a, b) of shape (..., nmax)."""
    m = np.atleast_1d(np.asarray(m, complex))
    x = np.atleast_1d(np.asarray(x, float))
    m, x = np.broadcast_arrays(m, x)
    if nmax is None:
        nmax = n_max(x)
    n = np.arange(1, nmax + 1)
    mx = m * x
    # logarithmic derivative D_n(mx) by downward recurrence (stable for complex m)
    nstart = int(max(nmax, np.max(np.abs(mx)))) + 16
    D = np.zeros(mx.shape + (nstart + 1,), complex)
    for k in range(nstart, 0, -1):
        D[..., k - 1] = k / mx - 1.0 / (D[..., k] + k / mx)
    Dn = D[..., 1:nmax + 1]
    # Riccati-Bessel functions psi_n = x j_n, xi_n = x h_n^(1), for n = 0..nmax
    nn = np.arange(0, nmax + 1)
    xx = x[..., None]
    psi = xx * spherical_jn(nn, xx)
    xi = psi + 1j * xx * spherical_yn(nn, xx)
    mm = m[..., None]
    fa = Dn / mm + n / xx
    fb = Dn * mm + n / xx
    a = (fa * psi[..., 1:] - psi[..., :-1]) / (fa * xi[..., 1:] - xi[..., :-1])
    b = (fb * psi[..., 1:] - psi[..., :-1]) / (fb * xi[..., 1:] - xi[..., :-1])
    return a, b


def cross_sections(m, x, nmax=None, per_channel=False):
    """Mie efficiencies in units of 1/k^2 (multiply by 1/k^2 for SI).
    Returns dict with ext, sca, abs (summed), and if per_channel, arrays (..., nmax)
    for each channel 'ext_TM','ext_TE','sca_TM','sca_TE','abs_TM','abs_TE'."""
    a, b = mie_coefficients(m, x, nmax)
    n = np.arange(1, a.shape[-1] + 1)
    w = 2 * np.pi * (2 * n + 1)
    out = {}
    ch = dict(ext_TM=w * a.real, ext_TE=w * b.real,
              sca_TM=w * abs(a) ** 2, sca_TE=w * abs(b) ** 2)
    ch["abs_TM"] = ch["ext_TM"] - ch["sca_TM"]
    ch["abs_TE"] = ch["ext_TE"] - ch["sca_TE"]
    for q in ("ext", "sca", "abs"):
        out[q] = (ch[q + "_TM"] + ch[q + "_TE"]).sum(-1)
    if per_channel:
        out.update(ch)
    return out


def rho_ImGamma0(kR, nmax, nquad=400):
    """Eigenvalues of Im Gamma_0 on a sphere of radius R (dimensionless, units eps0 = 1),
    for the regular vector spherical waves with orthonormal angular parts:
        Im Gamma_0(x,x') = k^3 sum_{nmp} w_nmp(x) w_nmp(x')^dagger
        rho_{n,TE} = int_0^{kR} j_n(t)^2 t^2 dt
        rho_{n,TM} = int_0^{kR} [n(n+1) (j_n/t)^2 + ((t j_n)'/t)^2] t^2 dt
    (independent of m).  Small-kR check: rho_{1,TM} -> (2/9)(kR)^3 = k^3 V/(6 pi).
    Returns (rho_TE, rho_TM), each of shape (nmax,)."""
    t, wq = _leggauss(nquad)
    t = 0.5 * kR * (t + 1)
    wq = 0.5 * kR * wq
    n = np.arange(1, nmax + 1)[:, None]
    jn = spherical_jn(n, t)
    jnm1 = spherical_jn(n - 1, t)
    dtj = t * jnm1 - n * jn                       # d/dt [t j_n(t)]
    rTE = (jn ** 2 * t ** 2) @ wq
    rTM = ((n * (n + 1) * jn ** 2 + dtj ** 2)) @ wq   # t^2 cancels the 1/t^2
    return rTE, rTM


def planewave_ImGamma0_quadform(kR, nth=4000):
    """Direct evaluation of psi^dagger (Im Gamma_0) psi for a unit x-polarized plane wave
    restricted to a sphere of radius R (k = 1), from the plane-wave representation
        Im Gamma_0(x,x') = (k^3/16 pi^2) int dOmega (I - khat khat) e^{i k khat.(x-x')},
    giving (k^3/16 pi^2) int F(q)^2 pi (1 + cos^2 th) sin th dth,
    F(q) = 3V j_1(qR)/(qR), q = 2k sin(th/2).  Used to validate rho_ImGamma0 and the
    plane-wave coefficients sum_m |a_nmp|^2 = 2 pi (2n+1):
        psi^dag ImGamma0 psi = (2 pi/k^3) sum_n (2n+1)(rho_TE^2 + rho_TM^2)."""
    R = kR
    V = 4 * np.pi * R ** 3 / 3
    th, wq = _leggauss(nth)
    th = 0.5 * np.pi * (th + 1)
    wq = 0.5 * np.pi * wq
    q = 2 * np.sin(th / 2)
    qR = np.maximum(q * R, 1e-12)
    F = 3 * V * spherical_jn(1, qR) / qR
    return (1 / (16 * np.pi ** 2)) * np.sum(wq * F ** 2 * np.pi * (1 + np.cos(th) ** 2) * np.sin(th))
