"""Secs. 5.4-5.5 (sec:polarizabilities, sec:plasmonicQ): polarizability limits, the
Thal/Chu-type Q bound for small scatterers, and the geometry independence of plasmonic Q.
Units: eps0 = c = 1, sphere radius a = 1 (so omega = k a = x).  e^{-i omega t}.
Mie coefficients from sec5_local_constraints/mie.py (Bohren-Huffman); alpha = i (6 pi/k^3) a_1
(i.e. tutorial c_1^TM = -a_1).

Tutorial claims (tutorial.tex line numbers):
  l.2026  Sipe-Kranendonk: -Im[1/alpha_e] >= k^3/(6 pi eps0)  (equality if lossless).
  l.2030  [eq:polar_corr] alpha~ = (alpha_qs^-1 - i k^3/(6 pi eps0))^-1.
  l.2045  [eq:alpha_e] alpha_e = -i (6 pi eps0/k^3) c_1^TM, exact, -> [eq:polar_corr] as ka -> 0.
  l.2053  max |alpha_e| = 6 pi eps0/k^3 (resonance, alpha purely imaginary).
  l.2057  [eq:alpha_e_DC] static alpha_e(0) <= 4 pi eps0 a^3 (monotonicity; |eps(0)| -> inf).
  l.2064  Delta omega = omega0 k^3 V/(2 pi) for alpha_qs = 3 eps0 V w0^2/(w0^2 - w^2).
  l.2066  [eq:Q_Thal] Q = (3/2)/(k0 a)^3.
  l.2154 fn  "a spherical plasmonic scatterer of radius a [lossless metal] would operate exactly
          at the Chu-Thal limit".
  l.2097  [eq:kin_energy] U_K = (1/4) eps0 a^3 wp^2/(gamma^2 + w^2) |E|^2 (V = a^3).
  l.2107-2117  int eps E.E dV = 0 for a quasi-static mode  [eq:dielectric_metal].
  l.2150  Q = omega/gamma_m for any subwavelength structure of one Drude metal (+ dielectrics),
          independent of geometry and size; cf. Wang-Shen Q = omega (d eps'/d omega)/(2 eps'').

Checks:
  A. Sipe-Kranendonk, max|alpha| and passivity on 3000 random lossy/lossless Mie spheres.
  B. Exact Mie alpha vs [eq:polar_corr] as ka -> 0 (relative error ~ (ka)^2).
  C. Static bound: Mie alpha(omega -> 0) for eps(0) = 10..1e6 and for a lossy Drude metal;
     5000 random quasi-static coated spheres (eps_core, eps_shell in [1, 1e4]).
  D. Q of a LOSSLESS Drude sphere from the exact Mie a_1 (half-power points of |alpha|^2):
     Q (ka)^3/1.5 -> 1 for eps_inf = 1 (reproduces 1.0056, 1.0006, 1.0000 at ka = 0.171, 0.058,
     0.017), but -> (eps_inf + 2)/3 for eps_inf != 1 (e.g. 2 for eps_inf = 4): the footnote's
     "exactly at the Chu-Thal limit" holds for eps_inf = 1 only.
  E. Plasmonic Q: complex poles (Newton) and absorption FWHM for a sphere (vacuum, glass host),
     spheroids (L = 0.1, 0.05) and a coated sphere (two modes), for eps_inf = 1 and 5:
     Re(w~)/(2|Im w~|) = Re(w~)/gamma to machine precision; FWHM-Q agrees to O(gamma^2).
     Exact Mie for small lossy Drude spheres: Q -> w/gamma as ka -> 0, with
     1/Q ~= gamma/w + 1/Q_rad.
  F. [eq:dielectric_metal] and energy balance for the sphere mode (eps = -2): numerical
     quadrature gives int_out |E|^2 = 2 V |E_in|^2, so U_K = U_{m,E} + U_d at w = wp/sqrt 3.

Run:  python polarizability_q_limits.py   (about 3 s; writes figures/sec5_4_9_q_limits.png/pdf)
"""
import os
import sys
import numpy as np
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sec5_local_constraints.mie import mie_coefficients

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def a1(m, x):
    a, _ = mie_coefficients(np.atleast_1d(m), np.atleast_1d(x), nmax=1)
    return a[..., 0]


def alpha_mie(m, x):
    return 1j * 6 * np.pi / x ** 3 * a1(m, x)           # a = 1, k = x


def alpha_qs(eps):
    return 4 * np.pi * (eps - 1) / (eps + 2)


def alpha_rc(eps, x):
    return 1 / (1 / alpha_qs(eps) - 1j * x ** 3 / (6 * np.pi))


def index(eps):
    m = np.sqrt(np.asarray(eps, complex))
    return np.where(m.imag < 0, -m, m)


# ----------------------------------------------------------------------------------------
def check_passivity(N=3000, seed=1):
    rng = np.random.default_rng(seed)
    x = 10 ** rng.uniform(-2, 1, N)
    eps = rng.uniform(-10, 20, N) + 1j * np.where(rng.random(N) < 0.2, 0.0, 10 ** rng.uniform(-4, 1, N))
    A = a1(index(eps), x)
    sk = (1 / A).real                          # = -Im(1/alpha) / (k^3/6pi)
    lossless = eps.imag == 0
    return sk.min(), np.abs(sk[lossless] - 1).max(), np.abs(A).max()


def check_polar_corr():
    rows = []
    for eps in (4.0, -5 + 1j, -2 + 0.5j):
        for x in (0.3, 0.1, 0.03, 0.01):
            am, ar = alpha_mie(index(eps), x)[0], alpha_rc(eps, x)
            rows.append((eps, x, abs(am / ar - 1)))
    return rows


def coated_sphere_static(eps_core, eps_shell, f):
    """Quasi-static polarizability (units 4 pi a^3) of a coated sphere, core volume fraction f."""
    e1, e2 = eps_core, eps_shell
    num = (e2 - 1) * (e1 + 2 * e2) + f * (e1 - e2) * (1 + 2 * e2)
    den = (e2 + 2) * (e1 + 2 * e2) + f * (2 * e2 - 2) * (e1 - e2)
    return num / den


def check_static(N=5000, seed=2):
    rng = np.random.default_rng(seed)
    e1 = 10 ** rng.uniform(0, 4, N)
    e2 = 10 ** rng.uniform(0, 4, N)
    f = rng.uniform(0, 1, N)
    a = coated_sphere_static(e1, e2, f)
    # monotonicity: increase the core permittivity by 10%
    a_up = coated_sphere_static(1.1 * e1, e2, f)
    mie = [(e, (alpha_mie(index(e), 1e-3)[0] / (4 * np.pi))) for e in (10.0, 1e2, 1e4, 1e6)]
    w = 1e-4                                        # lossy Drude metal at low frequency
    eps_d = 1 - 1.0 / (w ** 2 + 1j * 0.01 * w)
    mie.append(("Drude wp=1, g=0.01 at w=1e-4", alpha_mie(index(eps_d), w)[0] / (4 * np.pi)))
    return a.max(), (a_up - a).min(), mie


# ----------------------------------------------------------------------------------------
def thal_ratio(ka0, einf=1.0):
    """Exact-Mie Q of a lossless Drude sphere (a = 1) with quasi-static resonance at ka0."""
    wp = ka0 * np.sqrt(einf + 2)
    t = lambda w: (1 / a1(index(einf - wp ** 2 / w ** 2), w)[0]).imag
    lo, hi = 0.85 * ka0, 1.15 * ka0
    wr = brentq(t, lo, hi, xtol=1e-15, rtol=1e-15)
    wm = brentq(lambda w: t(w) - 1, lo, hi, xtol=1e-15, rtol=1e-15)
    wq = brentq(lambda w: t(w) + 1, lo, hi, xtol=1e-15, rtol=1e-15)
    Q = wr / abs(wq - wm)
    return Q * wr ** 3 / 1.5, wr, abs(a1(index(einf - wp ** 2 / wr ** 2), wr)[0])


def check_lorentzian_bandwidth(V=4 * np.pi / 3 * 1e-3, w0=1.0):
    """l.2064: FWHM of |alpha~|^2 with alpha_qs = 3 V w0^2/(w0^2 - w^2), k = w (c = 1)."""
    al = lambda w: 1 / ((w0 ** 2 - w ** 2) / (3 * V * w0 ** 2) - 1j * w ** 3 / (6 * np.pi))
    peak = minimize_scalar(lambda w: -abs(al(w)) ** 2, bounds=(0.99, 1.01), method="bounded",
                           options=dict(xatol=1e-14))
    pk = abs(al(peak.x)) ** 2
    lo = brentq(lambda w: abs(al(w)) ** 2 - pk / 2, 0.9, peak.x)
    hi = brentq(lambda w: abs(al(w)) ** 2 - pk / 2, peak.x, 1.1)
    return hi - lo, w0 * w0 ** 3 * V / (2 * np.pi), np.sqrt(pk), 6 * np.pi / peak.x ** 3


# ----------------------------------------------------------------------------------------
def drude(w, wp=1.0, g=0.01, einf=1.0):
    return einf - wp ** 2 / (w ** 2 + 1j * g * w)


GEOMS = {
    "sphere, vacuum": dict(den=lambda e: e + 2, num=lambda e: e - 1),
    "sphere, glass host (eps_d = 2.25)": dict(den=lambda e: e + 4.5, num=lambda e: e - 2.25),
    "prolate spheroid L = 0.1": dict(den=lambda e: 1 + 0.1 * (e - 1), num=lambda e: e - 1),
    "prolate spheroid L = 0.05, host 1.77": dict(den=lambda e: 1.77 + 0.05 * (e - 1.77), num=lambda e: e - 1.77),
    "coated sphere core 2.1, f = 0.5 (mode 1)": dict(
        den=lambda e: (e + 2) * (2.1 + 2 * e) + 2 * 0.5 * (e - 1) * (2.1 - e),
        num=lambda e: (e - 1) * (2.1 + 2 * e) + 0.5 * (2.1 - e) * (1 + 2 * e), mode=0),
    "coated sphere core 2.1, f = 0.5 (mode 2)": dict(
        den=lambda e: (e + 2) * (2.1 + 2 * e) + 2 * 0.5 * (e - 1) * (2.1 - e),
        num=lambda e: (e - 1) * (2.1 + 2 * e) + 0.5 * (2.1 - e) * (1 + 2 * e), mode=1),
}


def complex_pole(den, g, einf, guess):
    w = complex(guess)
    F = lambda w: den(drude(w, 1.0, g, einf))
    for _ in range(60):
        h = 1e-7 * (1 + abs(w))
        dF = (F(w + h) - F(w - h)) / (2 * h)
        step = F(w) / dF
        w -= step
        if abs(step) < 1e-15:
            break
    return w


def plasmonic_q(geom, g=0.01, einf=1.0):
    den, num = geom["den"], geom["num"]
    w = np.linspace(0.02, 1.2 * np.sqrt(1 / max(einf, 1)) + 0.5, 400001)
    sab = w * np.imag(num(drude(w, 1.0, g, einf)) / den(drude(w, 1.0, g, einf)))
    # locate resonance peaks
    pk = np.where((sab[1:-1] > sab[:-2]) & (sab[1:-1] > sab[2:]) & (sab[1:-1] > 0.05 * sab.max()))[0] + 1
    i = pk[geom.get("mode", 0)]
    wp = complex_pole(den, g, einf, w[i])
    # FWHM of the absorption peak
    half = sab[i] / 2
    l = i
    while sab[l] > half:
        l -= 1
    r = i
    while sab[r] > half:
        r += 1
    fw = (w[r] - w[r - 1]) * (sab[r - 1] - half) / (sab[r - 1] - sab[r]) + w[r - 1] - (
        w[l] + (w[l + 1] - w[l]) * (half - sab[l]) / (sab[l + 1] - sab[l]))
    Q_pole = wp.real / (2 * abs(wp.imag))
    wr = w[i]
    epsp = (drude(wr + 1e-6, 1, g, einf).real - drude(wr - 1e-6, 1, g, einf).real) / 2e-6
    Q_ws = wr * epsp / (2 * drude(wr, 1, g, einf).imag)
    return dict(wr=wp.real, Q_pole=Q_pole, Q_fwhm=wr / fw, Q_gamma=wp.real / g, Q_ws=Q_ws, im=wp.imag)


def mie_lossy_q(ka0, g=0.01, einf=1.0):
    """FWHM-Q of |a_1|^2 for a lossy Drude sphere (a = 1) with QS resonance near ka0."""
    wp = ka0 * np.sqrt(einf + 2)
    gg = g * wp
    A2 = lambda w: abs(a1(index(einf - wp ** 2 / (w ** 2 + 1j * gg * w)), w)[0]) ** 2
    pk = minimize_scalar(lambda w: -A2(w), bounds=(0.8 * ka0, 1.2 * ka0), method="bounded",
                         options=dict(xatol=1e-13 * ka0))
    h = A2(pk.x) / 2
    lo = brentq(lambda w: A2(w) - h, 0.8 * ka0, pk.x, xtol=1e-15)
    hi = brentq(lambda w: A2(w) - h, pk.x, 1.25 * ka0, xtol=1e-15)
    Q = pk.x / (hi - lo)
    Q_abs = pk.x / gg
    Q_rad = 1.5 / pk.x ** 3
    return Q, Q_abs, 1 / (1 / Q_abs + 1 / Q_rad), pk.x


def check_energy_balance(nq=60, h=1e-5):
    """Quasi-static sphere mode (a = 1): interior potential -r cos(th) (E_in = z), exterior
    potential -cos(th)/r^2 (continuous at r = a; normal D continuous iff eps_m = -2).  The
    exterior field is obtained by finite-differencing the potential and |E|^2 is integrated by
    Gauss-Legendre quadrature in u = 1/r and cos(th).  Returns int_out |E|^2/(V |E_in|^2),
    int eps |E|^2 dV (eps_m = -2, eps_d = 1), and the normal-D jump at r = a."""
    phi = lambda r, th: -np.cos(th) / r ** 2
    u, wu = np.polynomial.legendre.leggauss(nq)
    u = 0.5 * (u + 1)
    wu = 0.5 * wu                                        # u in (0, 1), r = 1/u, dV_r = r^2 dr = du/u^4
    c, wc = np.polynomial.legendre.leggauss(nq)
    R, C = np.meshgrid(1 / u, c, indexing="ij")
    TH = np.arccos(C)
    Er = -(phi(R + h, TH) - phi(R - h, TH)) / (2 * h)
    Et = -(phi(R, TH + h) - phi(R, TH - h)) / (2 * h) / R
    E2 = Er ** 2 + Et ** 2
    I_out = 2 * np.pi * np.sum((wu / u ** 4)[:, None] * wc[None, :] * E2)
    V = 4 * np.pi / 3
    Dn_out = -(phi(1 + h, 0.0) - phi(1 - h, 0.0)) / (2 * h)  # E_r just outside at the pole
    Dn_in = -2.0 * 1.0                                        # eps_m E_r(inside), eps_m = -2
    return I_out / V, -2.0 * V + I_out, Dn_out - Dn_in


def main(make_figure=True):
    print("=" * 90)
    print("Secs. 5.4-5.5  polarizability limits, Thal Q, plasmonic Q (eps0 = c = a = 1)")
    print("=" * 90)
    mn, lossless_dev, amax = check_passivity()
    print(f"A. 3000 random spheres: min [-Im(1/alpha)]/(k^3/6pi) = {mn:.6f} (>= 1);"
          f" lossless max |ratio - 1| = {lossless_dev:.1e}; max |alpha|/(6pi/k^3) = {amax:.6f} (<= 1)")

    print("\nB. exact Mie alpha vs [eq:polar_corr]:")
    for eps, x, err in check_polar_corr():
        print(f"   eps = {eps!s:10s} ka = {x:5.2f}: |alpha_Mie/alpha_rc - 1| = {err:.2e}")

    amax_c, dmono, mie = check_static()
    print(f"\nC. static bound: 5000 random coated spheres: max alpha(0)/(4 pi a^3) = {amax_c:.6f};"
          f" min change when eps_core increases 10% = {dmono:.2e} (>= 0, monotone)")
    for e, v in mie:
        print(f"   Mie ka = 1e-3, eps = {e!s:28s}: alpha/(4 pi a^3) = {v.real:.6f}{v.imag:+.1e}i")

    dw, dw_th, amx, amx_th = check_lorentzian_bandwidth()
    print(f"\nD. l.2064 FWHM of |alpha~|^2 for the LC-type alpha_qs (V = 4pi/3 x 1e-3): {dw:.6e}"
          f" vs w0 k^3 V/(2pi) = {dw_th:.6e};  peak |alpha| = {amx:.4f} vs 6pi/k^3 = {amx_th:.4f}")
    print("   Lossless Drude sphere, exact Mie, Q from half-power points of |alpha|^2:")
    thal = {}
    for einf in (1.0, 4.0):
        for ka0 in (0.171, 0.058, 0.017):
            rat, wr, apk = thal_ratio(ka0, einf)
            thal[(einf, ka0)] = rat
            print(f"   eps_inf = {einf:.0f}, k a = {wr:.4f}: Q (ka)^3 / 1.5 = {rat:.4f}   (|a_1| at peak = {apk:.6f})"
                  + ("" if einf == 1 else f"   [(eps_inf+2)/3 = {(einf+2)/3:.3f}]"))

    print("\nE. Plasmonic Q (Drude wp = 1, gamma = 0.01), quasi-static structures")
    print(f"   {'structure':42s} {'eps_inf':>7} {'Re w~':>8} {'Im w~':>10} {'Q pole':>9} {'Re w~/g':>9}"
          f" {'Q FWHM':>9} {'Q W-S':>9}")
    rows = []
    for einf in (1.0, 5.0):
        for name, geom in GEOMS.items():
            r = plasmonic_q(geom, 0.01, einf)
            rows.append((name, einf, r))
            print(f"   {name:42s} {einf:7.1f} {r['wr']:8.5f} {r['im']:10.7f} {r['Q_pole']:9.4f} {r['Q_gamma']:9.4f}"
                  f" {r['Q_fwhm']:9.4f} {r['Q_ws']:9.4f}")
    print("   -> Im w~ = -gamma/2 for every structure: Q = Re(w~)/gamma, geometry only sets Re w~.")
    print("   Exact Mie, lossy Drude sphere (gamma/wp = 0.01, eps_inf = 1):")
    mie_rows = []
    for ka0 in (0.3, 0.1, 0.03, 0.01):
        Q, Qa, Qc, wr = mie_lossy_q(ka0)
        mie_rows.append((wr, Q, Qa, Qc))
        print(f"   ka = {wr:.4f}: Q_Mie = {Q:9.4f}; w/gamma = {Qa:9.4f}; (gamma/w + (ka)^3/1.5)^-1 = {Qc:9.4f}")

    rat, s, jump = check_energy_balance()
    print(f"\nF. sphere mode: int_out |E|^2 / (V |E_in|^2) = {rat:.8f} (2);  int eps |E|^2 at eps_m = -2: {s:.1e};"
          f" normal-D jump {jump:.1e}")
    wr = 1 / np.sqrt(3)
    UK = 0.5 * (1 / wr ** 2) * 4 * np.pi / 3              # (1/2) eps0 wp^2/w^2 E^2 V, gamma = 0
    UmE = 0.5 * 4 * np.pi / 3
    Ud = 0.5 * rat * 4 * np.pi / 3
    print(f"   lossless Drude at w = wp/sqrt3: U_K = {UK:.6f}, U_m,E + U_d = {UmE + Ud:.6f}")
    print("   Eq. kin_energy: (1/4) m |v|^2 N V with v = -eE/(m(gamma - i w)) = (1/4) eps0 V wp^2/(gamma^2 + w^2)|E|^2 (identity)")

    if make_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.3))
        kas = np.logspace(np.log10(0.01), np.log10(0.4), 12)
        for einf, c in ((1.0, "C0"), (4.0, "C1")):
            ax[0].semilogx(kas, [thal_ratio(k, einf)[0] for k in kas], c + "o-", ms=3,
                           label=rf"lossless Drude, $\varepsilon_\infty$ = {einf:.0f}")
            ax[0].axhline((einf + 2) / 3, color=c, ls=":", lw=1)
        ax[0].set_xlabel(r"$ka$ at resonance")
        ax[0].set_ylabel(r"$Q\,(ka)^3/1.5$")
        ax[0].legend(fontsize=7, frameon=False)
        ax[0].set_title("exact Mie vs [eq:Q_Thal]", fontsize=9)
        for name, einf, r in rows:
            ax[1].plot(r["Q_gamma"], r["Q_pole"], "o" if einf == 1 else "s", ms=4)
        ax[1].plot([d[2] for d in mie_rows], [d[1] for d in mie_rows], "k^", ms=5,
                   label="exact Mie sphere, ka = 0.3-0.01 (radiation lowers Q at large ka)")
        q = np.array([15, 95])
        ax[1].plot(q, q, "k:", lw=1)
        ax[1].set_xlabel(r"$\mathrm{Re}\,\tilde\omega/\gamma$")
        ax[1].set_ylabel(r"$Q$ (pole; Mie: FWHM)")
        ax[1].set_title("plasmonic Q, 6 structures x 2 $\\varepsilon_\\infty$", fontsize=9)
        ax[1].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        os.makedirs(FIGDIR, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIGDIR, f"sec5_4_9_q_limits.{ext}"), dpi=200)
        print(f"\nwrote {os.path.normpath(os.path.join(FIGDIR, 'sec5_4_9_q_limits.pdf'))}")


if __name__ == "__main__":
    main()
