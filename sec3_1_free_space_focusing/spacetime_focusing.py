"""Sec. 3.1.5: the space-time diffraction limit.

Amplitude normalization (the tutorial's choice): c(k, omega) uniform on |k| <= |omega|/c,
|omega| <= omega_max.  Checks:
  1. Eq. (ST1d): the closed form
        psi_1D(x,t) ∝ (1/x) [ sin^2(wm (x-ct)/2c)/(x-ct) + sin^2(wm (x+ct)/2c)/(x+ct) ]
     against direct 2D quadrature of int dw int dk e^{i(kx - wt)}.
  2. First zero at t = 0: 2 pi c/wm (space-time) vs pi c/wm (single frequency wm): ratio 2.
  3. 3D vector case (tutorial equation after Fig. SpacetimeFocus):
        psi_3D(rho, 0) ∝ int_0^wm [j0(w rho/c) - j1(w rho/c)/(w rho/c)] dw
                       = (c / 2 rho) [ Si(U) + j1(U) ],   U = wm rho / c     (closed form, new)
     This profile has NO zeros (Si(U) - |j1(U)| >= 0.33 for U >= 0.5, and both are positive
     below), and decays as [pi/2 - 2 cos(U)/U]/U: a positive pi/(2U) tail with an O(U^-2)
     ripple (shallow minima at U = 6.88, 13.3, ...).  The two 1D features highlighted in the
     text (faster side-lobe decay, doubled first-zero radius) do not carry over to 3D.
  4. The result depends on the normalization.  With physical energy normalization the
     optimal spectra are weighted (2D: c ∝ k/k_z per frequency; 3D: a ∝ omega^2), giving
        line focus (t = 0):  J1(U)/U   vs single-frequency J0(U)   (first-zero ratio 1.593)
        3D (t = 0):       [2 sin U - U cos U - Si(U)] / U^3  vs  j0 - j1/U.

Run: python spacetime_focusing.py  (~3 s; writes figures/spacetime_focusing.*)
"""
import os
import sys
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import sici, spherical_jn, j0, j1, jn_zeros
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def psi1d_closed(x, t, wm=1.0, c=1.0):
    def term(s):
        s = np.asarray(s, float)
        out = np.where(np.abs(s) > 1e-12, np.sin(wm * s / (2 * c))**2 / np.where(s == 0, 1, s), 0.0)
        return out
    return 4 * c / x * (term(x - c * t) + term(x + c * t))


def psi1d_quad(x, t, wm=1.0, c=1.0, n=400):
    """int_{-wm}^{wm} dw int_{-|w|/c}^{|w|/c} dk e^{i(kx - wt)} by tensor Gauss-Legendre."""
    u, wu = leggauss(n)
    total = 0.0
    for sgn in (+1, -1):
        w = 0.5 * wm * (u + 1) * sgn
        ww = 0.5 * wm * wu
        kk = np.abs(w)[:, None] / c * u[None, :]
        wk = np.abs(w)[:, None] / c * wu[None, :]
        total = total + np.sum(ww[:, None] * wk * np.exp(1j * (kk * x - w[:, None] * t)))
    return total


def psi3d_closed(U):
    """Normalized to 1 at U = 0:  (3/4) [Si(U) + j1(U)] / U."""
    U = np.asarray(U, float)
    si = sici(U)[0]
    return np.where(U > 1e-8, 0.75 * (si + spherical_jn(1, U)) / np.where(U > 0, U, 1), 1.0)


def psi3d_quad(U, n=400):
    x, w = leggauss(n)
    u = 0.5 * U * (x + 1)
    ww = 0.5 * U * w
    f = spherical_jn(0, u) - np.where(u > 1e-8, spherical_jn(1, u) / np.where(u > 0, u, 1), 1 / 3)
    return 1.5 * np.sum(ww * f) / U          # (1/U) int_0^U (j0 - j1/u) du, normalized by 2/3


def single_freq_3d(U):
    U = np.asarray(U, float)
    return 1.5 * (spherical_jn(0, U) - np.where(U > 1e-8, spherical_jn(1, U) / np.where(U > 0, U, 1), 1 / 3))


def energy_weighted_3d(U):
    U = np.asarray(U, float)
    si = sici(U)[0]
    return np.where(U > 1e-3, 4.5 * (2 * np.sin(U) - U * np.cos(U) - si) / np.where(U > 0, U, 1)**3, 1.0)


def energy_weighted_2d_quad(x, wm=1.0, c=1.0, n=300):
    """2D line, energy normalization: int_0^wm dw int_{-w/c}^{w/c} dk (k0/kz) e^{ikx},
    k0 = w/c; substituting k = k0 sin(t) removes the edge singularity."""
    u, wu = leggauss(n)
    w = 0.5 * wm * (u + 1); ww = 0.5 * wm * wu
    t = 0.5 * np.pi * u; wt = 0.5 * np.pi * wu            # t in (-pi/2, pi/2)
    k0 = w / c
    inner = np.sum(wt[None, :] * k0[:, None] * np.cos(k0[:, None] * np.sin(t)[None, :] * x), axis=1)
    return np.sum(ww * inner)


def energy_weighted_3d_quad(U, n=400):
    """(9/2) (1/U^3) int_0^U u^2 (j0 - j1/u) du, normalized to 1 at U = 0."""
    x, w = leggauss(n)
    u = 0.5 * U * (x + 1); ww = 0.5 * U * w
    return 4.5 * np.sum(ww * u**2 * (spherical_jn(0, u) - spherical_jn(1, u) / u)) / U**3


def fwhm_amplitude(f, umax=6):
    """half width at which the FIELD falls to 1/sqrt(2) (intensity half max)."""
    g = lambda u: f(np.array([u]))[0]**2 - 0.5
    x = np.linspace(1e-6, umax, 4000)
    y = f(x)**2 - 0.5
    i = np.argmax(y < 0)
    return brentq(g, x[i - 1], x[i])


def main(make_figure=True):
    print("=" * 78)
    print("Sec. 3.1.5: space-time focusing (wm = c = 1)")
    print("=" * 78)
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(6):
        x, t = rng.uniform(-8, 8), rng.uniform(-8, 8)
        a, b = psi1d_closed(np.array(x), np.array(t)), psi1d_quad(x, t)
        errs.append(abs(a - b) / abs(psi1d_closed(np.array(1e-6), np.array(0.0))))
    print(f"1. Eq. (ST1d) closed form vs direct quadrature at random (x,t): max rel. err {max(errs):.1e}")
    xs = np.linspace(0.05, 10, 20000)
    st = psi1d_closed(xs, 0 * xs)
    sf = np.sin(xs) / xs
    z_st = xs[np.argmax(st < 1e-6 * st[0])] if np.any(st < 1e-6 * st[0]) else np.nan
    z_st = brentq(lambda x: np.sin(x / 2), 5, 7)   # exact zero of sin^2(x/2): 2 pi
    print(f"2. first zero at t=0: space-time {z_st:.6f} = 2pi;  single-frequency {np.pi:.6f} = pi;  ratio {z_st/np.pi:.6f}")
    Us = np.array([0.5, 2.0, 5.0, 12.0, 30.0])
    d = max(abs(psi3d_closed(np.array([u]))[0] - psi3d_quad(u)) for u in Us)
    print(f"3. 3D: closed form (3/4)[Si(U)+j1(U)]/U vs quadrature: max abs diff {d:.1e}")
    U = np.linspace(1e-3, 200, 200001)
    p3 = psi3d_closed(U)
    Uc = np.linspace(0.5, 500, 1000001)
    gap = sici(Uc)[0] - np.abs(spherical_jn(1, Uc))
    print(f"   no zeros: min_(U >= 0.5) [Si(U) - |j1(U)|] = {gap.min():.4f} > 0 (and Si, j1 > 0 for U < 0.5);"
          f"  U*psi at U=200: {200*p3[-1]:.4f} (-> 3pi/8 = {3*np.pi/8:.4f})")
    Ul = np.linspace(1, 30, 290001); yl = psi3d_closed(Ul)
    im = np.where((yl[1:-1] < yl[:-2]) & (yl[1:-1] < yl[2:]))[0] + 1
    print(f"   shallow side-lobe minima (no zeros) at U = {np.round(Ul[im][:4], 3)}, psi/psi0 = {np.round(yl[im][:4], 4)}")
    h3 = fwhm_amplitude(psi3d_closed)
    h1 = fwhm_amplitude(single_freq_3d)
    print(f"   intensity half-width at half-max (units c/wm): space-time {h3:.4f}, single-frequency {h1:.4f}; ratio {h3/h1:.3f}")
    # 1D half widths
    f1 = lambda x: psi1d_closed(x, 0 * x) / psi1d_closed(np.array([1e-9]), np.array([0.0]))
    fs = lambda x: np.sinc(x / np.pi)
    print(f"   1D intensity HWHM: space-time {fwhm_amplitude(f1):.4f}, single-frequency {fwhm_amplitude(fs):.4f}")
    print("4. Energy-normalized alternatives:")
    xs2 = np.array([0.5, 2.0, 5.5, 9.0])
    q2 = np.array([energy_weighted_2d_quad(x) for x in xs2]) / energy_weighted_2d_quad(1e-9)
    c2 = 2 * j1(xs2) / xs2
    print(f"   line focus: direct quadrature vs 2 J1(U)/U: max abs diff {np.abs(q2 - c2).max():.1e}")
    q3 = np.array([energy_weighted_3d_quad(u) for u in (0.5, 3.0, 7.0, 15.0)])
    c3 = energy_weighted_3d(np.array([0.5, 3.0, 7.0, 15.0]))
    print(f"   3D: direct quadrature vs 9[2 sin U - U cos U - Si(U)]/(2U^3): max abs diff {np.abs(q3 - c3).max():.1e}")
    print(f"   line focus (2D propagation): J1(U)/U first zero {jn_zeros(1,1)[0]:.4f} vs J0 first zero {jn_zeros(0,1)[0]:.4f}: ratio {jn_zeros(1,1)[0]/jn_zeros(0,1)[0]:.4f}")
    ze = brentq(lambda u: 2 * np.sin(u) - u * np.cos(u) - sici(u)[0], 3.0, 4.5)
    z1 = brentq(lambda u: single_freq_3d(np.array([u]))[0], 2, 3.5)
    print(f"   3D: omega^2-weighted profile has first zero at U = {ze:.4f} vs single-frequency {z1:.4f}: ratio {ze/z1:.3f}")

    if make_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
        x = np.linspace(-4 * np.pi, 4 * np.pi, 2001) + 1e-9
        ax[0].plot(x / np.pi, f1(np.abs(x)), "C0", label="space-time (Eq. ST1d)")
        ax[0].plot(x / np.pi, np.sinc(x / np.pi), "C1", label="single frequency")
        ax[0].set_xlabel(r"$\omega_{\max}x/(\pi c)$"); ax[0].set_title("1D, amplitude norm.", fontsize=9)
        u = np.linspace(1e-3, 25, 2000)
        ax[1].plot(u / np.pi, psi3d_closed(u), "C0", label=r"space-time: $\frac{3}{4}[\mathrm{Si}(U)+j_1(U)]/U$")
        ax[1].plot(u / np.pi, single_freq_3d(u), "C1", label=r"single freq.: $\frac{3}{2}(j_0-j_1/U)$")
        ax[1].set_xlabel(r"$U/\pi=\omega_{\max}\rho/(\pi c)$"); ax[1].set_title("3D, amplitude norm.", fontsize=9)
        ax[2].plot(u / np.pi, energy_weighted_3d(u), "C0", label=r"space-time, $\omega^2$-weighted")
        ax[2].plot(u / np.pi, single_freq_3d(u), "C1", label="single frequency")
        ax[2].set_xlabel(r"$U/\pi$"); ax[2].set_title("3D, energy norm.", fontsize=9)
        for a in ax:
            a.axhline(0, color="gray", lw=0.5)
            a.legend(frameon=False, fontsize=7, loc="upper right")
        ax[0].legend(frameon=False, fontsize=7, loc="upper left")
        ax[0].set_ylim(-0.3, 1.45)
        ax[0].set_ylabel(r"field at $t=0$ (normalized)")
        fig.tight_layout()
        os.makedirs(FIGDIR, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIGDIR, f"spacetime_focusing.{ext}"), dpi=200)
        print("\nwrote figures/spacetime_focusing.*")


if __name__ == "__main__":
    main()
