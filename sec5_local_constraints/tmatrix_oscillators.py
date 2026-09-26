"""Sec. 5.2.3: oscillator representation of the volume T matrix and its sum rules.

Tutorial (tutorial.tex l.1900-1911), following Zhang, Monticone & Miller, Nat. Commun. 14,
7724 (2023):
  [Tosc]  T(w) = (2/pi) lim_{g->0} int_0^inf [X(wi) + (wi/w) Y(wi)] / (wi^2 - w^2 - i g w) dwi
  with   X >= 0,  -X <= Y <= X,  int_0^inf X dwi <= w_p^2 I_V,  int_0^inf X/wi^2 dwi <= T_0.
  High-frequency limit (l.1898): T(w -> inf) ~ -(w_p^2/w^2) I_V.

What the KK derivation actually gives (Hermitian/anti-Hermitian split, e^{-iwt}):
  omega T(omega) = (1/pi) int_R Z(wi)/(wi - w - i0) dwi,  Z = w Im T (Hermitian, PSD for w > 0),
  X = (Z + Z^T)/2 (real symmetric), Y = (Z - Z^T)/2 (Hermitian, imaginary, skew-symmetric),
  => T(w) = (2/pi) int_0^inf [X + (wi/w) Y]/(wi^2 - w^2 - i0 w) dwi   [same as Tosc, no i on Y]
  => int_0^inf X dwi = (pi/2) w_p^2 I_V,   int_0^inf X/wi^2 dwi = (pi/2) T_0.
So WITH the 2/pi prefactor the sum-rule constants carry pi/2, and the printed inequalities
are violated by every material (e.g. a single Lorentz oscillator) by the factor pi/2.  The
paper's Eq. (3) has the 2/pi; its Eq. (6) rescales X, Y -> (pi/2) X, Y and drops the 2/pi,
after which int X <= w_p^2 I_D and int X/wi^2 <= T_0.  The tutorial mixes the two.

Tests (units w_p = 1):
  A. 1x1 Lorentz "T" = chi: int X = pi/2 (not <= 1).
  B. Reciprocal quasi-static coupled-dipole T matrix (6 pixels, 18x18, Lorentz material):
     X >= 0, Y = 0, int X = (pi/2) I, int X/w^2 = (pi/2) T(0), and Tosc reconstructs T(w).
  C. Nonreciprocal gyrotropic Lorentz pixel (3x3): X +- Y >= 0, Tosc form (no i on Y)
     reconstructs T(w) in the upper half plane; the variant with i(wi/w)Y does not.

Run:  python tmatrix_oscillators.py   (about 3 s; writes figures/sec5_tmatrix_oscillators.png/pdf)
"""
import os
import sys
import numpy as np
from scipy.integrate import quad, quad_vec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sec5_local_constraints.vie_conservation_laws import ReH, ImH, lattice

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
WP = 1.0


def lorentz_chi(w, w0=1.0, g=0.2, wp=WP):
    return wp ** 2 / (w0 ** 2 - w ** 2 - 1j * g * w)


def static_Gamma(pos, v):
    """Quasi-static coupled-dipole Green's matrix (k -> 0): off-diagonal v(3rr - I)/(4pi r^3),
    diagonal -1/3 (sphere depolarization).  Real symmetric."""
    N = len(pos)
    G = np.zeros((3 * N, 3 * N))
    for i in range(N):
        G[3 * i:3 * i + 3, 3 * i:3 * i + 3] = -np.eye(3) / 3
        for j in range(i + 1, N):
            r = pos[i] - pos[j]; R = np.linalg.norm(r); rr = np.outer(r, r) / R ** 2
            B = v * (3 * rr - np.eye(3)) / (4 * np.pi * R ** 3)
            G[3 * i:3 * i + 3, 3 * j:3 * j + 3] = B
            G[3 * j:3 * j + 3, 3 * i:3 * i + 3] = B
    return G


class QuasiStaticT:
    """T(w) = -(Gamma_s + xi(w))^{-1}, xi = -1/chi(w), Lorentz chi.  Exactly causal/passive."""

    def __init__(self, shape=(3, 2, 1), d=1.0, w0=2.0, g=0.15):
        self.pos = lattice(*shape, d)
        self.G = static_Gamma(self.pos, 0.6 * d ** 3)
        self.w0, self.g = w0, g
        lam = np.linalg.eigvalsh(self.G)
        self.res = np.sqrt(w0 ** 2 - WP ** 2 * lam)      # oscillator frequencies (need > 0)
        self.n = self.G.shape[0]

    def __call__(self, w):
        chi = lorentz_chi(w, self.w0, self.g)
        return -np.linalg.inv(self.G - np.eye(self.n) / chi)

    @property
    def T0(self):
        return self(0.0).real


class GyrotropicT:
    """Single pixel, damped bound charge in a static B field along z:
    [(w0^2 - w^2 - i g w) I - i w wc J] x = -(e/m) E,  T = chi = wp^2 M^{-1},
    J = [z x] (skew).  Nonreciprocal (T != T^T), passive, T -> -wp^2/w^2 I."""

    def __init__(self, w0=1.0, g=0.2, wc=0.6):
        self.w0, self.g, self.wc = w0, g, wc
        self.J = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], float)
        self.res = [w0 - wc / 2, w0, w0 + wc / 2]

    def __call__(self, w):
        M = (self.w0 ** 2 - w ** 2 - 1j * self.g * w) * np.eye(3) - 1j * w * self.wc * self.J
        return WP ** 2 * np.linalg.inv(M)

    @property
    def T0(self):
        return self(0.0).real


def XY(Tfun, w):
    Z = w * ImH(Tfun(w))
    return 0.5 * (Z + Z.T), 0.5 * (Z - Z.T)


def sum_rules(Tfun, res):
    """int_0^inf X dw and int_0^inf X/w^2 dw (quad_vec with resonance breakpoints)."""
    pts = sorted(res)
    edges = [0.0] + [p for r in pts for p in (0.8 * r, r, 1.2 * r)] + [4 * max(pts)]
    edges = sorted(set(edges))
    I1 = sum(quad_vec(lambda w: XY(Tfun, w)[0], a, b, epsabs=1e-13, epsrel=1e-11)[0]
             for a, b in zip(edges[:-1], edges[1:]))
    I1 += quad_vec(lambda w: XY(Tfun, w)[0], edges[-1], np.inf, epsabs=1e-13, epsrel=1e-11)[0]
    f2 = lambda w: XY(Tfun, w)[0] / w ** 2 if w > 0 else XY(Tfun, 1e-12)[0] / 1e-24
    I2 = sum(quad_vec(f2, max(a, 1e-9), b, epsabs=1e-13, epsrel=1e-11)[0]
             for a, b in zip(edges[:-1], edges[1:]))
    I2 += quad_vec(f2, edges[-1], np.inf, epsabs=1e-13, epsrel=1e-11)[0]
    return I1, I2


def reconstruct(Tfun, w, with_i=False):
    """(2/pi) int_0^inf [X + (wi/w) Y]/(wi^2 - w^2) dwi for complex w in the UHP."""
    c = 1j if with_i else 1.0

    def f(wi):
        X, Y = XY(Tfun, wi)
        return (X + c * (wi / w) * Y) / (wi ** 2 - w ** 2)
    val = quad_vec(f, 0, 50, epsabs=1e-12, epsrel=1e-11, limit=4000)[0] \
        + quad_vec(f, 50, np.inf, epsabs=1e-12, epsrel=1e-11)[0]
    return 2 / np.pi * val


def single_lorentz():
    I1 = quad(lambda w: w * lorentz_chi(w).imag, 0, np.inf, limit=400)[0]
    I2 = quad(lambda w: lorentz_chi(w).imag / w if w > 0 else 0.0, 0, np.inf, limit=400)[0]
    return I1, I2, lorentz_chi(0.0).real


def main(make_figure=True):
    print("=" * 86)
    print("Sec. 5.2.3 volume-T-matrix oscillator representation (units w_p = 1)")
    print("=" * 86)
    I1, I2, T0 = single_lorentz()
    print("A. Single Lorentz oscillator T = chi = 1/(1 - w^2 - 0.2 i w) (saturates the f-sum rule):")
    print(f"   int_0^inf X dw = int w Im chi dw = {I1:.10f} = (pi/2) w_p^2 = {np.pi / 2:.10f}"
          f"   -> violates printed 'int X <= w_p^2 I_V' by {I1:.4f}x")
    print(f"   int_0^inf X/w^2 dw = {I2:.10f} = (pi/2) T0 = {np.pi / 2 * T0:.10f}")

    qt = QuasiStaticT()
    print(f"\nB. Reciprocal quasi-static coupled-dipole T matrix, {qt.n}x{qt.n}, oscillators at "
          f"w = {qt.res.min():.3f} ... {qt.res.max():.3f}:")
    wgrid = np.linspace(0.01, 5, 400)
    minX = min(np.linalg.eigvalsh(XY(qt, w)[0]).min() for w in wgrid)
    maxY = max(np.abs(XY(qt, w)[1]).max() for w in wgrid)
    print(f"   min eig X over grid = {minX:.2e} (>= 0);  max |Y| = {maxY:.1e} (reciprocal -> 0)")
    J1, J2 = sum_rules(qt, qt.res)
    I = np.eye(qt.n)
    print(f"   int X dw        = c I with c = {np.trace(J1).real / qt.n:.8f};  max|int X - (pi/2) I| = {np.abs(J1 - np.pi / 2 * I).max():.1e}")
    print(f"   int X/w^2 dw    : max|int X/w^2 - (pi/2) T0| / max|T0| = {np.abs(J2 - np.pi / 2 * qt.T0).max() / np.abs(qt.T0).max():.1e}")
    ev = np.linalg.eigvalsh(J1.real - WP ** 2 * I)
    print(f"   printed 'int X <= w_p^2 I_V': eigenvalues of int X - I = {ev.min():.4f} ... {ev.max():.4f} (> 0: VIOLATED)")
    wt = 1.3 + 0.25j
    Tr = reconstruct(qt, wt)
    print(f"   Tosc (2/pi prefactor) reconstruction at w = {wt}: max|T_rec - T| / max|T| = {np.abs(Tr - qt(wt)).max() / np.abs(qt(wt)).max():.1e}")

    gt = GyrotropicT()
    print("\nC. Nonreciprocal gyrotropic Lorentz pixel (w0 = 1, gamma = 0.2, wc = 0.6):")
    minZ = min(np.linalg.eigvalsh(sum(XY(gt, w))).min() for w in wgrid)
    minXmY = min(np.linalg.eigvalsh(XY(gt, w)[0] - XY(gt, w)[1]).min() for w in wgrid)
    X1, Y1 = XY(gt, 0.9)
    print(f"   min eig (X+Y) = {minZ:.2e}, min eig (X-Y) = {minXmY:.2e}  (both >= 0: -X <= Y <= X)")
    print(f"   X real symmetric: |Im X| = {np.abs(X1.imag).max():.1e};  Y Hermitian skew: |Y + Y^T| = {np.abs(Y1 + Y1.T).max():.1e},"
          f" |Re Y| = {np.abs(Y1.real).max():.1e}, |Im Y| = {np.abs(Y1.imag).max():.3f}")
    for wt in (1.1 + 0.3j, 0.4 + 0.1j):
        Tex = gt(wt)
        e0 = np.abs(reconstruct(gt, wt) - Tex).max() / np.abs(Tex).max()
        e1 = np.abs(reconstruct(gt, wt, with_i=True) - Tex).max() / np.abs(Tex).max()
        print(f"   w = {wt}: Tosc as in tutorial [X + (wi/w)Y] rel. error {e0:.1e};  variant [X + i(wi/w)Y] rel. error {e1:.2f}")
    G1, G2 = sum_rules(gt, gt.res)
    print(f"   int X dw diag = {np.diag(G1).real.round(8)} (pi/2 = {np.pi / 2:.8f});  int X/w^2 vs (pi/2)T0: "
          f"{np.abs(G2 - np.pi / 2 * gt.T0).max():.1e}")
    print("\n   => with the 2/pi prefactor of [Tosc]: int X = (pi/2) w_p^2 I_V and int X/w^2 = (pi/2) T_0 (equalities");
    print("      for a given V; inequalities over a design domain).  Either drop 2/pi (Zhang et al. Eq. 6) or")
    print("      put pi/2 on the right-hand sides.")

    if make_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        w = np.linspace(1e-3, 8, 4000)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        Xd = np.array([np.linalg.eigvalsh(XY(qt, x)[0].real) for x in w])
        ax[0].plot(w, Xd, lw=0.8)
        ax[0].set_xlabel(r"$\omega_i/\omega_p$"); ax[0].set_ylabel(r"eigenvalues of $\mathbb{X}(\omega_i)$")
        ax[0].set_title("18x18 quasi-static T matrix: X >= 0", fontsize=9)
        tr = np.array([np.trace(XY(qt, x)[0]).real for x in w]) / qt.n
        run = np.concatenate([[0], np.cumsum(0.5 * (tr[1:] + tr[:-1]) * np.diff(w))])
        chi1 = w * lorentz_chi(w).imag
        run1 = np.concatenate([[0], np.cumsum(0.5 * (chi1[1:] + chi1[:-1]) * np.diff(w))])
        ax[1].plot(w, run, "C0", label=r"$\mathrm{Tr}\int_0^{\Omega}\mathbb{X}\,d\omega_i/(N\omega_p^2)$, cluster")
        ax[1].plot(w, run1, "C1--", label=r"$\int_0^{\Omega}\omega\,\mathrm{Im}\chi\,d\omega/\omega_p^2$, single Lorentz")
        ax[1].axhline(np.pi / 2, color="k", lw=0.8, label=r"$\pi/2$")
        ax[1].axhline(1, color="C3", lw=0.8, ls=":", label="printed bound 1")
        ax[1].set_xlabel(r"$\Omega/\omega_p$"); ax[1].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        os.makedirs(FIGDIR, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIGDIR, f"sec5_tmatrix_oscillators.{ext}"), dpi=200)
        print(f"\nwrote {os.path.normpath(os.path.join(FIGDIR, 'sec5_tmatrix_oscillators.pdf'))}")


if __name__ == "__main__":
    main()
