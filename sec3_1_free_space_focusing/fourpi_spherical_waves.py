"""Sec. 3.1.3: focusing from the full 4pi solid angle, in a vector-spherical-wave basis.

Checks:
  1. VSW implementation: numerical curl of M_lm equals k N_lm (finite differences).
  2. At the origin, among all regular VSWs with l <= 6, only the electric dipole waves
     N_{1,m} are nonzero; |N_{1,m}(0)| = (2/3) sqrt(3/(8 pi)) for each m.
  3. The incoming part (1/2) N^(2)_{1,0} carries P_in = 1/(8 Z0 k^2) (Poynting flux through
     a sphere, computed by quadrature at several radii).
  4. Hence U(0)/P_in = (eps0/4)|N_{1,0}(0)|^2 / P_in = k^2/(3 pi c)   [Eq. FourPiBound],
     exactly twice the 2pi bound k^2/(6 pi c)   [Eq. TwoPiBound].
  5. The 4pi optimum is the uniform-solid-angle (Herglotz) wave
        E(x) = k^2 int dOmega (I - khat khat) q e^{ik khat.x} = 16 pi^2 k Im G0(x,0) q,
     proportional to N^reg_{1,0} for q = zhat; its incoming power obtained from the VSW
     coefficient equals (2 pi^2/(Z0 k^2)) int |a|^2 dOmega, i.e. twice the z-directed
     power of the 2pi (one-hemisphere) optimum -- the "doubling" argument of the text.
  6. Equatorial profile: N_{1,0}(theta = pi/2) = -sqrt(3/(8 pi)) (j0 - j1/x) thetahat... with
     overall phase i in our convention [Eq. N10xy].

Run: python fourpi_spherical_waves.py  (~3 s)
"""
import os
import sys
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import spherical_jn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tutorial_tools.vsw import vsw, curl_numeric
from tutorial_tools.angular_spectrum import cap_quadrature

K = 1.0


def sphere_quadrature(n_th=60, n_ph=120):
    x, w = leggauss(n_th)
    th = np.arccos(x)
    ph = 2 * np.pi * np.arange(n_ph) / n_ph
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    W = np.outer(w, np.full(n_ph, 2 * np.pi / n_ph)).ravel()
    TH, PH = TH.ravel(), PH.ravel()
    rh = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], 1)
    return rh, W


def check_curl(seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(6, 3)) * 2
    err = 0
    for l, m in [(1, 0), (1, 1), (2, -1), (3, 2)]:
        for kind in ("reg", "out"):
            Mfun = lambda p, l=l, m=m, kind=kind: vsw(l, m, p, K, kind)[0]
            N = vsw(l, m, pts, K, kind)[1]
            curlM = curl_numeric(Mfun, pts, h=1e-5)
            err = max(err, np.abs(curlM - K * N).max() / np.abs(N).max())
    return err


def origin_amplitudes(lmax=6, r=1e-7):
    p = np.array([[r, 0.3 * r, 0.2 * r]])
    rows = []
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            M, N = vsw(l, m, p, K, "reg")
            rows.append((l, m, np.linalg.norm(M), np.linalg.norm(N)))
    return rows


def incoming_power(l=1, m=0, radii=(3.0, 30.0, 300.0), Z0=1.0):
    """P_in of (1/2) N^(2)_lm: inward Poynting flux through spheres of given radii."""
    rh, W = sphere_quadrature()
    out = []
    for R in radii:
        pts = R * rh
        M2, N2 = vsw(l, m, pts, K, "in")
        E = 0.5 * N2
        H = 0.5 * M2 / (1j * Z0)
        Sr = 0.5 * np.real(np.sum(np.cross(E, np.conj(H)) * rh, 1))
        out.append(-np.sum(Sr * W) * R**2)
    return np.array(out)


def herglotz_field(points, q, n_theta=120, n_phi=128):
    kh, _, _, W1 = cap_quadrature(np.pi / 2, n_theta, n_phi, +1)
    kh2, _, _, W2 = cap_quadrature(np.pi / 2, n_theta, n_phi, -1)
    kh = np.concatenate([kh, kh2]); W = np.concatenate([W1, W2])
    a = q[None, :] - kh * (kh @ q)[:, None]            # (I - khat khat) q
    ph = np.exp(1j * K * points @ kh.T)
    return K**2 * ph @ (W[:, None] * a), kh, W, a


def imG0(points, k=K):
    """Im of the free-space dyadic Green's function G0(x,0) = (I + grad grad/k^2) e^{ikr}/(4 pi r):
    Im G0 = (k/4pi) [ (j0 - j1/x) I + (3 j1/x - j0) rhat rhat ]."""
    r = np.linalg.norm(points, axis=1)
    x = k * r
    j0, j1 = spherical_jn(0, x), spherical_jn(1, x)
    rh = points / r[:, None]
    A = (j0 - j1 / x)
    B = (3 * j1 / x - j0)
    return (k / (4 * np.pi)) * (A[:, None, None] * np.eye(3) + B[:, None, None] * rh[:, :, None] * rh[:, None, :])


def main():
    print("=" * 78)
    print("Sec. 3.1.3: 4pi focusing with vector spherical waves")
    print("=" * 78)
    print(f"1. max |curl M - k N| / |N| (finite differences, several l,m,kinds): {check_curl():.1e}")
    rows = origin_amplitudes()
    nz = [(l, m, a, b) for (l, m, a, b) in rows if max(a, b) > 1e-6]
    print(f"2. regular VSWs with l <= 6 having |field(0)| > 1e-6: {[(l, m) for l, m, _, _ in nz]}")
    ref = 2 / 3 * np.sqrt(3 / (8 * np.pi))
    for l, m, a, b in nz:
        print(f"     (l,m)=({l},{m}): |M(0)| = {a:.2e}, |N(0)| = {b:.10f}  vs (2/3)sqrt(3/(8pi)) = {ref:.10f}")
    N0 = vsw(1, 0, np.array([[0, 0, 1e-7]]), K)[1][0]
    print(f"     direction of N_10(0): {np.round(N0 / np.linalg.norm(N0), 8)} (i.e. zhat, times phase i)")
    Pin = incoming_power()
    print(f"3. P_in of (1/2) N^(2)_10 at R = 3, 30, 300: {Pin}  vs 1/(8 Z0 k^2) = {1/8:.10f}")
    U_over_P = 0.25 * ref**2 / Pin[-1]
    print(f"4. U(0)/P_in = {U_over_P:.10f} k^2/c  vs 1/(3 pi) = {1/(3*np.pi):.10f};  2pi bound 1/(6 pi) = {1/(6*np.pi):.10f}")
    # 5. Herglotz / Im G / N10 proportionality
    rng = np.random.default_rng(5)
    pts = rng.normal(size=(8, 3)) * 1.5
    Eh, kh, W, a = herglotz_field(pts, np.array([0, 0, 1.0]))
    N10 = vsw(1, 0, pts, K)[1]
    ratio = Eh / N10
    G = imG0(pts)
    Eg = 16 * np.pi**2 * K * G[:, :, 2]
    print(f"5. Herglotz wave (q = zhat) vs 16 pi^2 k Im G0(x,0) zhat: max rel. diff {np.abs(Eh - Eg).max()/np.abs(Eh).max():.1e}")
    mask = np.abs(N10) > 1e-3 * np.abs(N10).max()
    cst = ratio[mask]
    print(f"   Herglotz / N_10^reg ratio: constant = {cst[0]:.10f}, spread {np.abs(cst - cst[0]).max():.1e}")
    Pin_herg = abs(cst[0])**2 * Pin[-1]
    Pin_stat = 2 * np.pi**2 / K**2 * np.sum(W * np.sum(np.abs(K**2 * a)**2, 1))
    E0 = herglotz_field(np.zeros((1, 3)), np.array([0, 0, 1.0]))[0][0]
    print(f"   P_in(Herglotz) from VSW coefficient = {Pin_herg:.8f};  (2 pi^2/k^2) int |a|^2 dOmega = {Pin_stat:.8f};"
          f"  16 pi^3/3 = {16*np.pi**3/3:.8f}")
    print(f"   |E(0)|^2/P_in = {abs(E0[2])**2/Pin_herg:.10f}  (x 2pi c/k^2 -> {2*np.pi*abs(E0[2])**2/Pin_herg:.10f}; 2pi optimum 4/3, 4pi optimum 8/3)")
    # 6. equatorial profile
    x = np.linspace(0.1, 10, 7)
    pts = np.stack([x, 0 * x, 0 * x], 1)
    N = vsw(1, 0, pts, K)[1]
    pred = -np.sqrt(3 / (8 * np.pi)) * (spherical_jn(1, x) - np.sin(x)) / x   # z-component, tutorial Eq. (N10xy)
    print(f"6. N_10 on equator: z-comp / (i * Eq. N10xy z-comp) max deviation from 1: {np.abs(N[:,2]/(1j*pred) - 1).max():.1e};"
          f" x,y comps max {np.abs(N[:, :2]).max():.1e}")


if __name__ == "__main__":
    main()
