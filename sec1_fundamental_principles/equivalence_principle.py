"""Sec. 1.1: sign conventions of Maxwell's equations with magnetic currents and of the
surface equivalence principle, checked by brute-force radiation of the equivalent currents
from a sphere.  Time dependence e^{-i w t}; natural units eps0 = mu0 = c = 1 (so k = w).

Tutorial claims checked (tutorial.tex line numbers):
  [eq:MaxwellDiffEq] l.138  curl H + i w eps E = J_E,   -curl E + i w mu H = J_M
  [eq:TETD]          l.572  work done on the field = -int (J_e . E + J_m . H)
  [eq:SEP]           l.164  (J_e, J_m) = [[0, -n x], [n x, 0]] [(E', H') - (E, H)] delta_S
  [eq:ECinc]         l.190  J_e = -n x H_inc, J_m = n x E_inc radiate E_inc inside S, 0 outside
  [eq:ECscat]        l.210  J_e = n x H_scat, J_m = -n x E_scat radiate 0 inside S, E_scat outside

Method.  Free-space fields of electric and magnetic point currents are written down
(E_J = i w G.J, H_J = grad g x J, E_M = -grad g x M, H_M = i w G.M) and then PINNED to the
tutorial's conventions independently of the equivalence principle:
  (i)  finite-difference residuals of the two curl equations of [eq:MaxwellDiffEq] away from
       the source (fixes E_M relative to H_M and H_J relative to E_J);
  (ii) the work done by each source, -1/2 Re(J*.E) and -1/2 Re(M*.H) (the time-harmonic form
       of [eq:TETD]), equals the outward Poynting flux through a large sphere and is positive
       (fixes the overall signs and the outgoing boundary condition).
The currents of [eq:ECinc] and [eq:ECscat] are then placed on a sphere (Gauss-Legendre x
trapezoid quadrature) and radiated.  A control run with the sign of J_m flipped fails at O(1).

Run:  python equivalence_principle.py    (under 1 s; no figure)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

K = 2.5          # k a with sphere radius a = 1


def _geom(r, rp, k):
    R = r[:, None, :] - rp[None, :, :]
    d = np.linalg.norm(R, axis=-1)
    Rh = R / d[..., None]
    kr = k * d
    g = np.exp(1j * kr) / (4 * np.pi * d)
    A = 1 + 1j / kr - 1 / kr**2
    B = -1 - 3j / kr + 3 / kr**2
    gp = (1j * k - 1 / d) * g                                 # dg/dR
    return Rh, g, A, B, gp


def fields(r, rp, J=None, M=None, wts=None, k=K):
    """E, H at points r (P,3) radiated by electric/magnetic currents J, M (S,3) at rp (S,3),
    each multiplied by quadrature weight wts (S,).  Outgoing free-space Green's functions."""
    r, rp = np.atleast_2d(r), np.atleast_2d(rp)
    wts = np.ones(len(rp)) if wts is None else wts
    Rh, g, A, B, gp = _geom(r, rp, k)
    w = k
    E = np.zeros(r.shape, complex)
    H = np.zeros(r.shape, complex)
    if J is not None:
        Jw = J * wts[:, None]
        GJ = g[..., None] * (A[..., None] * Jw[None] + B[..., None] * Rh * np.sum(Rh * Jw[None], -1)[..., None])
        E += 1j * w * GJ.sum(1)
        H += (gp[..., None] * np.cross(Rh, Jw[None])).sum(1)
    if M is not None:
        Mw = M * wts[:, None]
        GM = g[..., None] * (A[..., None] * Mw[None] + B[..., None] * Rh * np.sum(Rh * Mw[None], -1)[..., None])
        H += 1j * w * GM.sum(1)
        E += -(gp[..., None] * np.cross(Rh, Mw[None])).sum(1)
    return E, H


# ------------------------------------------------------------------ (i), (ii) grounding
def curl(F, x, h=1e-4):
    """Central-difference curl of a vector field F: (P,3)->(P,3) at the points x."""
    out = np.zeros(x.shape, complex)
    d = []
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        d.append((F(x + e) - F(x - e)) / (2 * h))            # d/dx_j of all components
    out[:, 0] = d[1][:, 2] - d[2][:, 1]
    out[:, 1] = d[2][:, 0] - d[0][:, 2]
    out[:, 2] = d[0][:, 1] - d[1][:, 0]
    return out


def maxwell_residuals(seed=0):
    """Relative residuals of [eq:MaxwellDiffEq] (source-free region) for J and M point sources."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.5, 1.5, (6, 3)) + np.array([2.0, 0, 0])
    src = np.zeros((1, 3))
    q = rng.normal(size=(1, 3)) + 1j * rng.normal(size=(1, 3))
    out = {}
    for name, kw in (("J", dict(J=q)), ("M", dict(M=q))):
        Ef = lambda y: fields(y, src, **kw)[0]
        Hf = lambda y: fields(y, src, **kw)[1]
        E, H = fields(x, src, **kw)
        r1 = curl(Hf, x) + 1j * K * E                        # curl H + i w eps E = 0
        r2 = -curl(Ef, x) + 1j * K * H                       # -curl E + i w mu H = 0
        scale = np.abs(K * E).max() + np.abs(K * H).max()
        out[name] = max(np.abs(r1).max(), np.abs(r2).max()) / scale
    return out


def sphere_quadrature(a=1.0, n_th=48, n_ph=96):
    u, wu = np.polynomial.legendre.leggauss(n_th)
    ph = np.arange(n_ph) * 2 * np.pi / n_ph
    U, PH = np.meshgrid(u, ph, indexing="ij")
    S = np.sqrt(1 - U**2)
    nhat = np.stack([S * np.cos(PH), S * np.sin(PH), U], -1).reshape(-1, 3)
    wts = (a**2 * wu[:, None] * np.full(n_ph, 2 * np.pi / n_ph)[None, :]).ravel()
    return a * nhat, nhat, wts


def power_balance(seed=1):
    """Work done by a unit J and a unit M point source vs outward Poynting flux at R = 7."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(1, 3)) + 1j * rng.normal(size=(1, 3))
    q = q / np.linalg.norm(q)
    src = np.zeros((1, 3))
    x0 = np.array([[1e-3, 0, 0]])                           # Im G is smooth at the source
    rp, nh, w = sphere_quadrature(a=7.0, n_th=60, n_ph=120)
    out = {}
    for name, kw in (("J", dict(J=q)), ("M", dict(M=q))):
        E0, H0 = fields(x0, src, **kw)
        work = -0.5 * np.real(np.vdot(q[0], E0[0] if name == "J" else H0[0]))
        E, H = fields(rp, src, **kw)
        flux = 0.5 * np.real(np.sum(np.cross(E, np.conj(H)) * nh, -1) @ w)
        out[name] = (work, flux)
    return out, K**2 / (12 * np.pi)


# ------------------------------------------------------------------ equivalence principle
def plane_wave(x, khat, pol):
    ph = np.exp(1j * K * x @ khat)[:, None]
    E = pol[None, :] * ph
    H = np.cross(khat, pol)[None, :] * ph                  # Z0 = 1, from -curl E + i w H = 0
    return E, H


def test_ECinc(sign_m=+1, seed=2):
    rng = np.random.default_rng(seed)
    khat = rng.normal(size=3)
    khat /= np.linalg.norm(khat)
    pol = np.cross(khat, rng.normal(size=3)) + 1j * np.cross(khat, rng.normal(size=3))
    pol /= np.linalg.norm(pol)
    rp, nh, w = sphere_quadrature()
    Ei, Hi = plane_wave(rp, khat, pol)
    Je = -np.cross(nh, Hi)                                   # [eq:ECinc]
    Jm = sign_m * np.cross(nh, Ei)
    xin = rng.normal(size=(20, 3))
    xin *= (0.6 * rng.uniform(0, 1, 20)**(1 / 3) / np.linalg.norm(xin, axis=1))[:, None]
    xout = rng.normal(size=(20, 3))
    xout *= (rng.uniform(1.7, 4.0, 20) / np.linalg.norm(xout, axis=1))[:, None]
    Ein, Hin = fields(xin, rp, Je, Jm, w)
    Eout, Hout = fields(xout, rp, Je, Jm, w)
    Eref, Href = plane_wave(xin, khat, pol)
    return (max(np.abs(Ein - Eref).max(), np.abs(Hin - Href).max()),
            max(np.abs(Eout).max(), np.abs(Hout).max()))


def test_ECscat(sign_m=+1, seed=3):
    rng = np.random.default_rng(seed)
    r0 = np.array([[0.2, -0.1, 0.15]])                       # "scatterer": a point current inside S
    J0 = rng.normal(size=(1, 3)) + 1j * rng.normal(size=(1, 3))
    rp, nh, w = sphere_quadrature()
    Es, Hs = fields(rp, r0, J=J0)
    Je = np.cross(nh, Hs)                                    # [eq:ECscat]
    Jm = -sign_m * np.cross(nh, Es)
    xin = rng.normal(size=(20, 3))
    xin *= (0.5 * rng.uniform(0, 1, 20)**(1 / 3) / np.linalg.norm(xin, axis=1))[:, None]
    xin = xin[np.linalg.norm(xin - r0, axis=1) > 0.05]
    xout = rng.normal(size=(20, 3))
    xout *= (rng.uniform(1.7, 4.0, 20) / np.linalg.norm(xout, axis=1))[:, None]
    Ein, Hin = fields(xin, rp, Je, Jm, w)
    Eout, Hout = fields(xout, rp, Je, Jm, w)
    Eref, Href = fields(xout, r0, J=J0)
    scale = np.abs(Eref).max()
    return (max(np.abs(Ein).max(), np.abs(Hin).max()) / scale,
            max(np.abs(Eout - Eref).max(), np.abs(Hout - Href).max()) / scale)


def main():
    print("=" * 80)
    print(f"Sec. 1.1  Maxwell sign conventions and the surface equivalence principle (ka = {K})")
    print("=" * 80)
    res = maxwell_residuals()
    print("(i)  [eq:MaxwellDiffEq] source-free residuals (central differences, h = 1e-4):"
          f"  J source {res['J']:.1e}, M source {res['M']:.1e}")
    pb, exact = power_balance()
    print("(ii) work done by unit point sources ([eq:TETD], time-harmonic) vs outward Poynting flux:")
    for name in ("J", "M"):
        wk, fl = pb[name]
        print(f"     {name}: -1/2 Re(q*.field) = {wk:.8f},  flux at R = 7: {fl:.8f},  k^2/(12 pi) = {exact:.8f}")
    print("\nSphere a = 1, 48 x 96 quadrature; 20 random points inside (r <= 0.6) and outside (1.7 <= r <= 4)")
    print(f"{'currents':>40} | {'inside':>22} | {'outside':>22}")
    ei, eo = test_ECinc(+1)
    print(f"{'[eq:ECinc]  -n x H_inc, +n x E_inc':>40} | {'|F - F_inc| = ' + format(ei, '.1e'):>22} | {'|F| = ' + format(eo, '.1e'):>22}")
    ei, eo = test_ECinc(-1)
    print(f"{'control: J_m sign flipped':>40} | {'|F - F_inc| = ' + format(ei, '.1e'):>22} | {'|F| = ' + format(eo, '.1e'):>22}")
    ei, eo = test_ECscat(+1)
    print(f"{'[eq:ECscat] +n x H_sca, -n x E_sca':>40} | {'|F|/|E_s| = ' + format(ei, '.1e'):>22} | {'|F - F_s|/|E_s| = ' + format(eo, '.1e'):>22}")
    ei, eo = test_ECscat(-1)
    print(f"{'control: J_m sign flipped':>40} | {'|F|/|E_s| = ' + format(ei, '.1e'):>22} | {'|F - F_s|/|E_s| = ' + format(eo, '.1e'):>22}")
    print("  -> [eq:SEP]/[eq:ECinc]/[eq:ECscat] are consistent with [eq:MaxwellDiffEq] and e^{-iwt}.")


if __name__ == "__main__":
    main()
