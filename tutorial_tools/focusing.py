"""Optimal free-space focusing (tutorial Sec. 3.1): closed forms and semi-analytic
field evaluations.

Geometry: plane waves arrive from the cap of directions 0 <= theta' <= theta_max about
+z ("one" hemisphere) or, for 4pi-type excitation, from that cap and its mirror image
about the focal plane z = 0 ("both").  Target polarization q = xhat unless stated.

The optimal field for target polarization q is (see derivations, Sec. 1)

    E(x) = k^2 int_cap dOmega  w(theta') (I - khat khat^T) q  exp(i k khat.x),

with w = 1 for power normalization (uniform in solid angle) and w = |cos theta'| for
amplitude normalization.  For q = xhat the azimuthal integrals are done analytically
with the Jacobi-Anger/Hansen-Bessel identity int e^{i u cos phi} cos(n phi) dphi
= 2 pi i^n J_n(u), leaving 1D integrals over theta':

  E_x = pi k^2      int dth sin th w e^{ikz cos th} [(2 - s^2) J0(a s) + s^2 cos2phi J2(a s)]
  E_y = pi k^2 sin2phi int dth sin th w e^{ikz cos th}  s^2 J2(a s)
  E_z = -2 pi i k^2 cos phi int dth sin th w e^{ikz cos th} s cos th J1(a s)

with s = sin th, a = k rho, (rho, phi, z) cylindrical coordinates of the field point.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import jv, spherical_jn


# ----------------------------------------------------------------------------
# Focal-point intensity maxima (tutorial Eqs. OptPowerNorm, OptAmpNorm)
# All in units of k^2/(2 pi c), i.e. these return (2 pi c/k^2) eps0 |E(0)|^2 / P.
# ----------------------------------------------------------------------------
def bound_power_inplane(theta):
    c = np.cos(theta)
    return 4 / 3 - c - c**3 / 3


def bound_power_z(theta):
    c = np.cos(theta)
    return 2 * (2 / 3 - c + c**3 / 3)


def bound_amp_inplane(theta):
    s = np.sin(theta)
    return s**2 * (1 - s**2 / 4)


def bound_amp_z(theta):
    return np.sin(theta)**4 / 2


def amp_optimal_per_true_power(theta):
    """(2 pi c/k^2) eps0|E(0)|^2 / P for the *amplitude-optimal* coefficients when P
    is the physical z-directed power (not the amplitude norm).  Closed form:
        s^4 (1 - s^2/4)^2 / [2 (4/15 - c^3/6 - c^5/10)].
    """
    s, c = np.sin(theta), np.cos(theta)
    return s**4 * (1 - s**2 / 4)**2 / (2 * (4 / 15 - c**3 / 6 - c**5 / 10))


# ----------------------------------------------------------------------------
# Semi-analytic fields (1D quadrature in theta')
# ----------------------------------------------------------------------------
def _theta_nodes(theta_max, hemispheres, n):
    x, wx = leggauss(n)
    th = 0.5 * theta_max * (x + 1)
    wt = 0.5 * theta_max * wx
    if hemispheres == "one":
        return th, wt
    if hemispheres == "both":
        return np.concatenate([th, np.pi - th]), np.concatenate([wt, wt])
    raise ValueError(hemispheres)


def optimal_field_xpol(rho, phi, z, theta_max, normalization="power",
                       hemispheres="one", k=1.0, n=400):
    """Optimal field for q = xhat at points (rho, phi, z) (broadcastable arrays).
    Returns (Ex, Ey, Ez).  Units: E(0)_x = k^2 * (solid-angle integral), so that
    E_x(0) = pi k^2 * (2pi-normalized bound) for power normalization."""
    rho, phi, z = np.broadcast_arrays(*(np.asarray(v, float) for v in (rho, phi, z)))
    th, wt = _theta_nodes(theta_max, hemispheres, n)
    s, c = np.sin(th), np.cos(th)
    w = np.ones_like(th) if normalization == "power" else np.abs(c)
    a = k * rho[..., None] * s
    ph = np.exp(1j * k * z[..., None] * c)
    base = wt * s * w * ph
    J0, J1, J2 = jv(0, a), jv(1, a), jv(2, a)
    Ex = np.pi * k**2 * np.sum(base * ((2 - s**2) * J0 + s**2 * np.cos(2 * phi)[..., None] * J2), -1)
    Ey = np.pi * k**2 * np.sin(2 * phi) * np.sum(base * s**2 * J2, -1)
    Ez = -2j * np.pi * k**2 * np.cos(phi) * np.sum(base * s * c * J1, -1)
    return Ex, Ey, Ez


# ----------------------------------------------------------------------------
# Closed forms
# ----------------------------------------------------------------------------
def j1_over(a):
    a = np.asarray(a, float)
    out = np.empty_like(a)
    small = np.abs(a) < 1e-4
    out[small] = 1 / 3 - a[small]**2 / 30
    out[~small] = spherical_jn(1, a[~small]) / a[~small]
    return out


def J2_over(a):
    a = np.asarray(a, float)
    out = np.empty_like(a)
    small = np.abs(a) < 1e-4
    out[small] = a[small] / 8
    out[~small] = jv(2, a[~small]) / a[~small]
    return out


def field_2pi_power_closed(rho, phi, k=1.0):
    """Closed-form focal-plane (z = 0) field of the 2pi power-optimal beam, q = xhat:
        E_x = 2 pi k^2 [ sin^2(phi) j0(a) + (3 cos^2(phi) - 1) j1(a)/a ]
        E_y = 2 pi k^2 sin(phi) cos(phi) j2(a)
        E_z = -2 pi i k^2 cos(phi) J2(a)/a,          a = k rho.
    The in-plane part is exactly half of the 4pi (full-sphere) optimum,
    i.e. of 4 pi k^2 [ (j0 - j1/a) I + (3 j1/a - j0) rhat rhat ] xhat.
    """
    a = k * np.asarray(rho, float)
    j0 = spherical_jn(0, a)
    j1a = j1_over(a)
    j2 = spherical_jn(2, a)
    Ex = 2 * np.pi * k**2 * (np.sin(phi)**2 * j0 + (3 * np.cos(phi)**2 - 1) * j1a)
    Ey = 2 * np.pi * k**2 * np.sin(phi) * np.cos(phi) * j2
    Ez = -2j * np.pi * k**2 * np.cos(phi) * J2_over(a)
    return Ex, Ey, Ez


def intensity_paraxial(u):
    """Airy pattern, normalized to 1 at u = k rho sin(theta) = 0 (Eq. E2Paraxial)."""
    u = np.asarray(u, float)
    out = np.ones_like(u)
    nz = np.abs(u) > 1e-8
    out[nz] = (2 * jv(1, u[nz]) / u[nz])**2
    return out


def intensity_2pi_tutorial(a):
    """Tutorial Eq. (E2TwoPi), normalized to 1 at the origin:
    [4 (j1/a)^2 + (J2/a)^2] / (4/9).  Valid along the polarization axis (phi = 0)."""
    return (4 * j1_over(a)**2 + J2_over(a)**2) / (4 / 9)


def intensity_4pi_perp(a):
    """Tutorial Eq. (E2optFourPi), normalized: (j0 - j1/a)^2 / (2/3)^2."""
    a = np.asarray(a, float)
    return ((spherical_jn(0, a) - j1_over(a)) / (2 / 3))**2


def intensity_4pi_par(a):
    """4pi optimum along the polarization axis: (2 j1/a)^2 / (2/3)^2."""
    return (2 * j1_over(a) / (2 / 3))**2


# ----------------------------------------------------------------------------
# Profile metrics
# ----------------------------------------------------------------------------
def half_max_radius(f, x_max, n=20001):
    """Smallest x > 0 with f(x) = 1/2 f(0) (f evaluated on a grid, refined by brentq)."""
    from scipy.optimize import brentq
    x = np.linspace(0, x_max, n)
    y = f(x) / f(np.array([0.0]))[0]
    i = np.argmax(y < 0.5)
    if i == 0:
        return np.nan
    return brentq(lambda t: f(np.array([t]))[0] / f(np.array([0.0]))[0] - 0.5, x[i - 1], x[i])


def first_minimum(f, x_max, n=20001):
    """First local minimum of f on (0, x_max): returns (x_min, f(x_min)/f(0))."""
    from scipy.optimize import minimize_scalar
    x = np.linspace(1e-6, x_max, n)
    y = f(x)
    i = np.argmax((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:])) + 1
    r = minimize_scalar(lambda t: f(np.array([t]))[0], bracket=(x[i - 1], x[i], x[i + 1]))
    return r.x, r.fun / f(np.array([0.0]))[0]


def field_amp_closed_inplane(rho, phi, sin_theta, k=1.0):
    """Closed-form in-plane focal-plane field (z = 0) of the amplitude-optimal beam for
    any NA s = sin(theta), target xhat (arguments of J_n are a s, a = k rho):
        E_x = pi k^2 [ (2 s - s^3) J1/a + 2 s^2 J2/a^2 + cos(2 phi) s^3 J3/a ]
        E_y = pi k^2 sin(2 phi) s^3 J3/a
    (E_z = -2 pi i k^2 cos(phi) int_0^s b^2 sqrt(1-b^2) J1(a b) db has a closed form only at
    s = 1, where it equals -2 pi i k^2 cos(phi) j2(a)/a.)"""
    a = k * np.asarray(rho, float)
    s = sin_theta
    x = a * s
    J1, J2, J3 = jv(1, x), jv(2, x), jv(3, x)
    Ex = np.pi * k**2 * ((2 * s - s**3) * J1 / a + 2 * s**2 * J2 / a**2 + np.cos(2 * phi) * s**3 * J3 / a)
    Ey = np.pi * k**2 * np.sin(2 * phi) * s**3 * J3 / a
    return Ex, Ey


def field_amp_2pi_Ez_closed(rho, phi, k=1.0):
    a = k * np.asarray(rho, float)
    return -2j * np.pi * k**2 * np.cos(phi) * spherical_jn(2, a) / a
