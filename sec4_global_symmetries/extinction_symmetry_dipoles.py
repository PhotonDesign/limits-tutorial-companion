"""Sec. 4.1: 3D extinction symmetry with a coupled-dipole cluster.

Tutorial claims checked (tutorial.tex line numbers as of 2026-09):
  1. [~1425-1442] For ANY reciprocal scatterer (T = T^T), sigma_ext(k, e) = sigma_ext(-k, e^*)
     (time-reversed incident wave), for any polarization (incl. elliptical) and direction,
     even though absorption and scattering individually differ.
  2. [~1437-1440] the proof uses only T = T^T: a NON-reciprocal (gyrotropic, alpha != alpha^T)
     but passive cluster violates the equality, while "generalized reciprocity" holds:
     sigma_ext[alpha](k, e) = sigma_ext[alpha^T](-k, e^*) (~1618).

Model: N point dipoles (loss graded along z, so front/back absorption differ), p_i = alpha_i E_loc,i, E_loc,i = E_inc(r_i) + sum_{j!=i} G(r_i, r_j) p_j,
eps0 = 1, |E0| = 1, k = 2 pi (lengths in wavelengths).  alpha_i^{-1} = alpha0_i^{-1} - i k^3/(6 pi)
(radiative correction) with passive alpha0_i ((alpha0 - alpha0^dag)/2i >= 0) => absorption >= 0.
  sigma_ext = k sum_i Im(E_inc,i^* . p_i)
  sigma_abs = k sum_i [Im(E_loc,i^* . p_i) - k^3/(6 pi) |p_i|^2]
  sigma_sca = far-field integral of |f|^2, f = k^2/(4 pi) sum_i (I - rr) p_i e^{-ik r.r_i}
Energy conservation sigma_ext = sigma_abs + sigma_sca is checked as a code validation.

Run:  python extinction_symmetry_dipoles.py   (about 2 s; no figure)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

K = 2 * np.pi


def green_dyadic(r_vec, k=K):
    """Free-space dyadic G (eps0 = 1): E = G p."""
    r = np.linalg.norm(r_vec)
    rh = r_vec / r
    kr = k * r
    pref = k**2 * np.exp(1j * kr) / (4 * np.pi * r)
    a = 1 + 1j / kr - 1 / kr**2
    b = -1 - 3j / kr + 3 / kr**2
    return pref * (a * np.eye(3) + b * np.outer(rh, rh))


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.diag(r))


def make_cluster(rng, n=8, box=1.0, dmin=0.25, reciprocal=True, g=0.6):
    """Random asymmetric cluster.  Reciprocal: alpha0 = R diag(a) R^T (complex symmetric).
    Gyrotropic: alpha0 = R [[a, i g a0, 0], [-i g a0, a, 0], [0, 0, b]] R^T (alpha0^T != alpha0)."""
    pos = []
    while len(pos) < n:
        p = rng.uniform(0, box, 3)
        if all(np.linalg.norm(p - q) > dmin for q in pos):
            pos.append(p)
    pos = np.array(pos)
    alphas = []
    scale = 0.06                                   # |alpha0| k^3/(6 pi) ~ 0.2-1.6 (strong multiple scattering)
    for i in range(n):
        R = random_rotation(rng)
        loss = (pos[i, 2] / box)**3                # graded loss: lossy at z ~ box, ~lossless at z ~ 0
        a = scale * (rng.uniform(0.3, 2.0, 3) + 1j * (0.005 + loss * rng.uniform(0.5, 2.0, 3)))
        if reciprocal:
            A0 = R @ np.diag(a) @ R.T
        else:
            A0 = R @ np.array([[a[0], 1j * g * scale, 0], [-1j * g * scale, a[0], 0], [0, 0, a[2]]]) @ R.T
        alphas.append(np.linalg.inv(np.linalg.inv(A0) - 1j * K**3 / (6 * np.pi) * np.eye(3)))
    return pos, alphas


def solve(pos, alphas, khat, e):
    """Return dipoles p (n,3), local fields (n,3), incident fields (n,3)."""
    n = len(pos)
    A = np.zeros((3 * n, 3 * n), complex)
    for i in range(n):
        A[3 * i:3 * i + 3, 3 * i:3 * i + 3] = np.linalg.inv(alphas[i])
        for j in range(n):
            if i != j:
                A[3 * i:3 * i + 3, 3 * j:3 * j + 3] = -green_dyadic(pos[i] - pos[j])
    Einc = e[None, :] * np.exp(1j * K * pos @ khat)[:, None]
    p = np.linalg.solve(A, Einc.ravel()).reshape(n, 3)
    Eloc = np.array([np.linalg.solve(alphas[i], p[i]) for i in range(n)])
    return p, Eloc, Einc


def sphere_quadrature(n_theta=48, n_phi=96):
    x, w = np.polynomial.legendre.leggauss(n_theta)
    th = np.arccos(x)
    ph = np.arange(n_phi) * 2 * np.pi / n_phi
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    W = np.outer(w, np.full(n_phi, 2 * np.pi / n_phi))
    rh = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1).reshape(-1, 3)
    return rh, W.ravel()


def cross_sections(pos, alphas, khat, e, quad=None):
    p, Eloc, Einc = solve(pos, alphas, khat, e)
    ext = K * np.sum(np.imag(np.sum(Einc.conj() * p, 1)))
    ab = K * np.sum(np.imag(np.sum(Eloc.conj() * p, 1)) - K**3 / (6 * np.pi) * np.sum(np.abs(p)**2, 1))
    rh, w = sphere_quadrature() if quad is None else quad
    phase = np.exp(-1j * K * rh @ pos.T)                     # (Nq, n)
    P = phase @ p                                            # sum_i p_i e^{-ik r.r_i}, (Nq, 3)
    Pt = P - rh * np.sum(rh * P, 1)[:, None]
    sca = np.sum(w * np.sum(np.abs(Pt)**2, 1)) * (K**2 / (4 * np.pi))**2
    return ext, ab, sca


def random_direction_polarization(rng):
    """Random direction within ~35 deg of +z (along the loss gradient) and random elliptical e."""
    khat = np.array([0, 0, 1.0]) + 0.4 * rng.normal(size=3); khat /= np.linalg.norm(khat)
    u = np.cross(khat, rng.normal(size=3)); u /= np.linalg.norm(u)
    v = np.cross(khat, u)
    chi, delta = rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
    e = np.cos(chi) * u + np.exp(1j * delta) * np.sin(chi) * v   # general elliptical
    return khat, e


def run_trials(reciprocal, n_trials=6, seed=1):
    rng = np.random.default_rng(seed)
    quad = sphere_quadrature()
    rows = []
    for _ in range(n_trials):
        pos, alphas = make_cluster(rng, reciprocal=reciprocal)
        khat, e = random_direction_polarization(rng)
        c1 = cross_sections(pos, alphas, khat, e, quad)
        c2 = cross_sections(pos, alphas, -khat, e.conj(), quad)
        cT = cross_sections(pos, [a.T for a in alphas], -khat, e.conj(), quad)   # transposed medium
        rows.append((c1, c2, cT))
    return rows


def main():
    print("=" * 96)
    print("Sec. 4.1  3D extinction symmetry, coupled-dipole clusters (8 anisotropic dipoles in 1 lambda^3)")
    print("=" * 96)
    hdr = (f"{'':>12} | {'ext(k,e)':>9} {'ext(-k,e*)':>10} {'rel diff':>9} | {'abs1':>7} {'abs2':>7} |"
           f" {'sca1':>7} {'sca2':>7} | {'|ext-abs-sca|/ext':>17} | {'ext[a^T](-k,e*)':>15}")
    out = {}
    for label, rec in (("reciprocal", True), ("gyrotropic", False)):
        print(hdr)
        rows = run_trials(rec)
        rel, cons, gen = [], [], []
        for c1, c2, cT in rows:
            rel.append(abs(c1[0] - c2[0]) / abs(c1[0]))
            cons.append(max(abs(c1[0] - c1[1] - c1[2]) / c1[0], abs(c2[0] - c2[1] - c2[2]) / c2[0]))
            gen.append(abs(c1[0] - cT[0]) / abs(c1[0]))
            print(f"{label:>12} | {c1[0]:9.4f} {c2[0]:10.4f} {rel[-1]:9.1e} | {c1[1]:7.4f} {c2[1]:7.4f} |"
                  f" {c1[2]:7.4f} {c2[2]:7.4f} | {cons[-1]:17.1e} | {cT[0]:15.4f}")
        print(f"  {label}: max rel |ext(k,e) - ext(-k,e*)| = {max(rel):.1e};"
              f" max rel |ext[alpha](k,e) - ext[alpha^T](-k,e*)| = {max(gen):.1e};"
              f" energy-conservation residual {max(cons):.1e}\n")
        out[label] = dict(rel=np.array(rel), gen=np.array(gen), cons=np.array(cons))
    return out


if __name__ == "__main__":
    main()
