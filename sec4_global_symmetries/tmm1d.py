"""Minimal 1D transfer-matrix (characteristic-matrix) solver for planar stacks in vacuum.

Conventions (time dependence exp(-i omega t), so Im(eps) > 0 is loss):
  * The stack occupies 0 < z < L = sum(d); vacuum on both sides.
  * In-plane wavevector kx is conserved; k_z = sqrt(eps k0^2 - kx^2) with Im k_z >= 0.
  * U = E_y (s/TE) or H_y (p/TM); V = (1/(i k0)) dU/dz for TE, (1/(i k0 eps)) dU/dz for TM.
    (U, V) are both proportional to tangential fields and hence continuous at interfaces.
  * Left incidence: U = exp(i kz0 z) + r exp(-i kz0 z) for z < 0 and
    U = t_face exp(i kz0 (z - L)) for z > L.  Hence t_face is referenced to the EXIT face
    (Yeh's convention, also used by multilayer_film.m).  The forward-scattering amplitude
    that enters the optical theorem is t_ext = t_face * exp(-i kz0 L).

Used by extinction_symmetry_slab.py and tr_field_ratio_bounds.py.
"""
import numpy as np


def _kz(eps, k0, kx):
    kz = np.sqrt(eps * k0**2 - kx**2 + 0j)
    return np.where(kz.imag < 0, -kz, kz)


def _q(eps, k0, kx, pol):
    kz = _kz(eps, k0, kx)
    return kz / k0 if pol == "s" else kz / (eps * k0)


def layer_matrix(eps, d, k0, kx=0.0, pol="s"):
    """2x2 matrix m (arrays of shape (..., 2, 2)) with (U, V)(z + d) = m (U, V)(z)."""
    kz = _kz(eps, k0, kx)
    q = _q(eps, k0, kx, pol)
    c, s = np.cos(kz * d), np.sin(kz * d)
    m = np.empty(np.shape(k0) + (2, 2), complex)
    m[..., 0, 0] = c
    m[..., 0, 1] = 1j * s / q
    m[..., 1, 0] = 1j * q * s
    m[..., 1, 1] = c
    return m


def stack_matrix(eps_list, d_list, k0, kx=0.0, pol="s"):
    """Total left-to-right matrix P = m_N ... m_1."""
    k0 = np.asarray(k0, float)
    P = np.broadcast_to(np.eye(2, dtype=complex), k0.shape + (2, 2)).copy()
    for e, d in zip(eps_list, d_list):
        P = layer_matrix(e, d, k0, kx, pol) @ P
    return P


def rt(eps_list, d_list, k0, kx=0.0, pol="s"):
    """Reflection r and exit-face transmission t_face for incidence from the LEFT.
    Returns (r, t_face, kz0)."""
    k0 = np.asarray(k0, float)
    P = stack_matrix(eps_list, d_list, k0, kx, pol)
    a, b, c, d = P[..., 0, 0], P[..., 0, 1], P[..., 1, 0], P[..., 1, 1]
    kz0 = _kz(1.0, k0, kx)
    q0 = kz0 / k0                        # same for s and p in vacuum
    t = 2 * q0 / (q0 * (a + d) - b * q0**2 - c)   # uses det P = 1
    r = t * (d - b * q0) - 1
    return r, t, kz0


def rt_both_sides(eps_list, d_list, k0, kx=0.0, pol="s"):
    """(r1, t1, r2, t2, kz0): '1' = incidence from the left (front), '2' = from the right."""
    r1, t1, kz0 = rt(eps_list, d_list, k0, kx, pol)
    r2, t2, _ = rt(eps_list[::-1], d_list[::-1], k0, kx, pol)
    return r1, t1, r2, t2, kz0


def extinction(t_face, kz0, L):
    """Physical extinction per unit area / incident flux, 2 Re(1 - t_ext) with
    t_ext = t_face exp(-i kz0 L)."""
    return 2 * np.real(1 - t_face * np.exp(-1j * kz0 * L))


def interior_fields(eps_list, d_list, k0, z, side="left"):
    """Normal-incidence E_y(z) at points 0 <= z <= L for a unit-amplitude wave incident from
    `side`.  Left: incoming exp(i k0 z) (amplitude 1 at z = 0).  Right: incoming
    exp(-i k0 (z - L)) (amplitude 1 at z = L).  k0 scalar; z array."""
    eps_list = np.asarray(eps_list, complex)
    d_list = np.asarray(d_list, float)
    L = d_list.sum()
    if side == "right":
        E = interior_fields(eps_list[::-1], d_list[::-1], k0, L - np.asarray(z), "left")
        return E
    r, t, _ = rt(eps_list, d_list, k0)
    U = np.array([1 + r, (1 - r)], complex)       # (U, V) at z = 0 (q0 = 1)
    edges = np.concatenate([[0.0], np.cumsum(d_list)])
    edges[-1] = np.inf                                   # absorb round-off at z = L
    z = np.clip(np.asarray(z, float), 0.0, None)
    out = np.full(z.shape, np.nan, complex)
    for j, (e, d) in enumerate(zip(eps_list, d_list)):
        sel = (z >= edges[j]) & (z < edges[j + 1]) & np.isnan(out.real)
        if np.any(sel):
            dz = z[sel] - edges[j]
            kz = np.sqrt(e * k0**2 + 0j)
            q = kz / k0
            out[sel] = U[0] * np.cos(kz * dz) + 1j * U[1] / q * np.sin(kz * dz)
        U = layer_matrix(e, d, np.float64(k0)) @ U
    return out
