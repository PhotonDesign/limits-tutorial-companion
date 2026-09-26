"""Sec. 3.1.4: energy cost of sub-diffraction ("super-oscillatory") focal spots.

Band-limited field on a line (tutorial Eq. EPW2D at z = 0):  E(x) = int_{-km}^{km} c(k) e^{ikx} dk,
with field energy  int |E|^2 dx = 2 pi int |c|^2 dk  (Parseval).  Minimizing int |c|^2 subject
to linear point constraints E(x_i) = b_i gives  min = b^dag G^{-1} b  with the band-limited Gram
matrix G_ij = int e^{ik(x_i - x_j)} dk = 2 sin(km (x_i - x_j))/(x_i - x_j).  This makes the
tutorial's scaling arguments exact:

  two points,   E(x+delta) - E(x) = 1:        min int|c|^2 = 1/(4km - 4 sin(km delta)/delta)
                                                         -> 3 / (2 km^3 delta^2)      [E >~ 1/delta^2]
  three points, E(0) = 1, E(+-delta) = 0:     min int|c|^2 -> 45 / (2 km^5 delta^4)   [E >~ 1/delta^4]
      i.e. relative to the unconstrained optimum 1/(2 km):  penalty -> 45/(km delta)^4.

2D (focal plane, paraxial scalar, circular pupil |k_par| <= km): unit field at the origin and a
zero ring at radius G.  Only the m = 0 angular harmonic matters and the maximal Strehl ratio
(peak intensity at fixed power, relative to the Airy optimum) has the closed form

    S(u) = 1 - 4 J1(u)^2 / ( u^2 [J0(u)^2 + J1(u)^2] ),   u = km G,
         -> u^4/192   as u -> 0,         S = 1 at the Airy zero u = 3.8317.

(the G^4 law of tutorial Fig. Superres(b)).  Checked against a brute-force discretization.

Run: python superoscillation.py   (~2 s; writes figures/superoscillation.*)
"""
import os
import sys
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import j0, j1, jn_zeros

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
U0 = jn_zeros(1, 1)[0]          # 3.8317..., Airy first zero (k rho sin theta)


def gram_1d(x, km=1.0):
    d = x[:, None] - x[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        G = np.where(np.abs(d) > 0, 2 * np.sin(km * d) / np.where(d == 0, 1, d), 2 * km)
    return G


def min_energy_1d(points, values, km=1.0):
    """Exact min of int |c|^2 dk s.t. E(points) = values (band-limited to |k| <= km).
    Uses a high-precision solve through the explicit quadrature representation to avoid
    the ill-conditioning of the sinc Gram matrix at small delta."""
    # Represent c on Gauss-Legendre nodes; the constraint matrix is A_ij = w_j e^{i k_j x_i}.
    x, w = leggauss(400)
    k = km * x
    wk = km * w
    A = np.exp(1j * np.outer(points, k)) * np.sqrt(wk)[None, :]   # acts on a = sqrt(w) c
    # min |a|^2 s.t. A a = b  ->  a = A^+ b  (minimum-norm solution via SVD-based lstsq)
    a, *_ = np.linalg.lstsq(A, values.astype(complex), rcond=None)
    return np.sum(np.abs(a)**2)


def two_point_exact(delta, km=1.0):
    return 1.0 / (4 * km - 4 * np.sin(km * delta) / delta)


def three_point(delta, km=1.0):
    pts = np.array([-delta, 0.0, delta])
    b = np.array([0.0, 1.0, 0.0])
    G = gram_1d(pts, km)
    exact = np.real(b @ np.linalg.solve(G, b))
    return exact


def strehl_ring(u):
    u = np.asarray(u, float)
    return 1 - 4 * j1(u)**2 / (u**2 * (j0(u)**2 + j1(u)**2))


def strehl_ring_bruteforce(G, km=1.0, n=600):
    """Maximize |E(0)|^2 / int|c|^2 over rotationally symmetric pupils c(k) on |k| <= km,
    subject to E(rho = G) = 0, by discretizing k on Gauss-Legendre nodes (weights 2 pi k dk)."""
    x, w = leggauss(n)
    k = 0.5 * km * (x + 1)
    wk = 0.5 * km * w * 2 * np.pi * k
    v0 = np.sqrt(wk)                      # E(0)   = sum wk c = <v0, a>,   a = sqrt(wk) c
    vG = np.sqrt(wk) * j0(k * G)          # E(G)   = sum wk c J0(kG)
    # project v0 orthogonally to vG; Strehl = |P v0|^2 / |v0|^2
    p = v0 - vG * (vG @ v0) / (vG @ vG)
    return (p @ p) / (v0 @ v0)


def main(make_figure=True):
    km = 1.0
    print("=" * 78)
    print("Sec. 3.1.4: super-oscillation energy penalties (km = 1)")
    print("=" * 78)
    print("1D, two-point constraint E(x+d) - E(x) = 1:")
    for d in [1.0, 0.3, 0.1, 0.03]:
        ex = two_point_exact(d)
        num = min_energy_1d(np.array([0.0, d]), np.array([-0.5, 0.5]))  # symmetric split, same min
        num2 = min_energy_1d(np.array([0.0, d]), np.array([0.0, 1.0]))
        print(f"  km d = {d:5.2f}: exact {ex:.6e}  (quadrature {min(num, num2):.6e})  asymptote 3/(2 d^2) = {1.5/d**2:.6e}")
    print("\n1D, three-point constraint E(0) = 1, E(+-d) = 0; penalty = min energy / unconstrained min:")
    print(f"  {'km d':>6} {'penalty (exact)':>16} {'45/(km d)^4':>12} {'ratio':>7}")
    for d in [np.pi, 0.75 * np.pi, np.pi / 2, 0.3, 0.1, 0.03]:
        pen = three_point(d) / (1 / (2 * km))
        pen_q = min_energy_1d(np.array([-d, 0, d]), np.array([0, 1, 0])) / (1 / (2 * km))
        print(f"  {d:6.3f} {pen:16.6e} {45/d**4:12.4e} {pen/(45/d**4):7.4f}   (quadrature {pen_q:.6e})")
    print("  (km d = pi is the sinc zero: penalty 1.  Halving the spot, km d = pi/2, costs",
          f"{three_point(np.pi/2)*2:.2f}x.)")

    print("\n2D ring (paraxial scalar, circular pupil):  S(u) closed form vs brute force")
    print(f"  {'G/G_Airy':>9} {'S closed':>12} {'S brute':>12} {'u^4/192':>12} {'1/S':>10}")
    for g in [1.0, 0.8, 0.5, 0.3, 0.1, 0.05]:
        u = g * U0
        sc, sb = strehl_ring(u), strehl_ring_bruteforce(u)
        print(f"  {g:9.2f} {sc:12.6e} {sb:12.6e} {u**4/192:12.6e} {1/sc:10.2f}")
    print(f"  small-G law: S ~ (U0^4/192) (G/G_Airy)^4 = {U0**4/192:.4f} (G/G_Airy)^4")
    print(f"  factor-2 reduction costs {1/strehl_ring(U0/2):.1f}x (tutorial: '16X');"
          f" factor-10 costs {1/strehl_ring(U0/10):.3g}x (tutorial: '10,000X')")

    if make_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = np.logspace(-2, 0, 200)
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
        ax[0].loglog(g, strehl_ring(g * U0), "C0", label="exact $S_{\\max}(G)$ (scalar, paraxial)")
        ax[0].loglog(g, (g * U0)**4 / 192, "k--", lw=0.8, label="$(k_mG)^4/192$")
        ax[0].set_xlabel("$G/G_{\\rm Airy}$")
        ax[0].set_ylabel("maximal Strehl ratio")
        ax[0].legend(frameon=False, fontsize=8)
        d = np.logspace(-1.5, np.log10(np.pi), 200)
        pen = np.array([three_point(x) * 2 for x in d])
        ax[1].loglog(d / np.pi, pen, "C3", label="exact 1D three-point penalty")
        ax[1].loglog(d / np.pi, 45 / d**4, "k--", lw=0.8, label="$45/(k_m\\delta)^4$")
        ax[1].set_xlabel("$k_m\\delta/\\pi$")
        ax[1].set_ylabel("energy / unconstrained energy")
        ax[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        os.makedirs(FIGDIR, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIGDIR, f"superoscillation.{ext}"), dpi=200)
        print("\nwrote figures/superoscillation.*")


if __name__ == "__main__":
    main()
