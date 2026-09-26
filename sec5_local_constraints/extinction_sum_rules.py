"""Sec. 5.2 / 5.2.1: plane-wave extinction sum rules, checked in SI units.

Tutorial claims (tutorial.tex line numbers):
  [GenDisp]  l.1793  Re s(w) = (2/pi) PV int_0^inf Im s(w') / (w'^2 - w^2) dw'   (as printed)
             vs [KKRe] l.368 / [KKExt] l.1813, which carry w' in the numerator.
  [SRExt0]   l.1822  int_0^inf sigma_ext(lambda) d lambda = pi^2 (e^dag alpha e)
             (alpha = static polarizability in volume units, p = eps0 alpha E).
             l.1825 calls this "the scattering cross-section integrated over all wavelengths".
  [SRExtInf] l.1833  int_0^inf sigma_ext(w) dw = pi w_p^2 V / 2        (as printed)
             Dimensional analysis and the text at l.1846 ("(2c/Omega) int sigma dw -> pi w_p^2")
             say it should be pi w_p^2 V / (2c).
  l.1844     "protons and neutrons" contribute negligibly to the plasma-frequency constant.

Tests:
  A. Quasi-static Drude-Lorentz spheroids, alpha = V chi/(1 + L chi) (exactly causal),
     sigma_ext = k Im alpha, SI units: both sum rules by adaptive quadrature.
  B. Same with the radiative correction alpha/(1 - i k^3 alpha/6pi): the static rule holds
     only to ~0.2% and the high-frequency integral comes out ~2x too large -- a known
     artifact of the point-dipole radiation-reaction model (runaway pole at w = i w_c in the
     upper half plane, w_c = 6 pi c^3/(V w_p^2); its coherent Thomson plateau
     sigma_sca = V^2 w_p^4/(6 pi c^4 (1 + w^2/w_c^2)) integrates to exactly pi w_p^2 V/(2c)),
     NOT a tutorial error.  Radiative correction is also unnecessary for sigma_ext >= 0:
     without it, sigma_ext = k Im alpha >= 0 already.
  C. Full Mie theory (own code, mie.py): Drude-Lorentz sphere R = 50 nm, high-frequency
     rule; lossy Lorentz-dielectric sphere, static rule for extinction and (for contrast)
     the scattering-only integral.
  D. GenDisp as printed vs the correct KK relation, for a Lorentz oscillator.
  E. Plasma-frequency contributions of nuclei vs electrons (l.1844).

Run:  python extinction_sum_rules.py   (about 10 s; writes figures/sec5_extinction_sum_rules.png/pdf)
"""
import os
import sys
import numpy as np
import warnings
from scipy.integrate import quad, IntegrationWarning

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sec5_local_constraints.mie import cross_sections, n_max

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
C0 = 299792458.0
HBAR_EV = 6.582119569e-16          # eV s
EV = 1 / HBAR_EV                   # rad/s per eV
HC_EVNM = 1239.841984              # eV nm  (k[1/nm] = E[eV] / 197.327)


# ------------------------------------------------------------------ materials (eV units)
class DrudeLorentz:
    """chi(w) = -wD^2/(w^2 + i gD w) + sum_j fj^2/(wj^2 - w^2 - i gj w), all in eV.
    High-frequency limit chi -> -w_p^2/w^2 with w_p^2 = wD^2 + sum_j fj^2."""

    def __init__(self, wD=0.0, gD=0.1, osc=()):
        self.wD, self.gD, self.osc = wD, gD, tuple(osc)

    def chi(self, w):
        w = np.asarray(w, complex)
        out = np.zeros_like(w)
        if self.wD:
            out = out - self.wD ** 2 / (w ** 2 + 1j * self.gD * w)
        for f, w0, g in self.osc:
            out = out + f ** 2 / (w0 ** 2 - w ** 2 - 1j * g * w)
        return out

    @property
    def wp2(self):
        return self.wD ** 2 + sum(f ** 2 for f, _, _ in self.osc)

    @property
    def chi0(self):
        """Static chi (inf for a conductor)."""
        return np.inf if self.wD else sum(f ** 2 / w0 ** 2 for f, w0, _ in self.osc)

    def resonances(self):
        return [w0 for _, w0, _ in self.osc]


SILVERLIKE = DrudeLorentz(wD=9.0, gD=0.1, osc=[(5.0, 5.0, 1.0)])          # w_p = 10.30 eV
DIELECTRIC = DrudeLorentz(wD=0.0, osc=[(8.0, 4.0, 0.8), (6.0, 9.0, 2.0)])   # chi(0) = 4.444


def depolarization_prolate(a, b):
    """L along the long (a) axis of a prolate spheroid with semi-axes a > b = c."""
    if abs(a - b) < 1e-12 * a:
        return 1 / 3
    e = np.sqrt(1 - (b / a) ** 2)
    return (1 - e ** 2) / e ** 2 * (np.log((1 + e) / (1 - e)) / (2 * e) - 1)


# ------------------------------------------------------------------ A/B: quasi-static
def alpha_qs(w_eV, mat, V, L, radiative=False):
    """Quasi-static polarizability (m^3), optional radiative correction."""
    chi = mat.chi(w_eV)
    a = V * chi / (1 + L * chi)
    if radiative:
        k = w_eV * EV / C0
        a = a / (1 - 1j * k ** 3 * a / (6 * np.pi))
    return a


def sigma_ext_qs(w_eV, mat, V, L, radiative=False):
    k = np.asarray(w_eV) * EV / C0
    return k * np.imag(alpha_qs(w_eV, mat, V, L, radiative))


def _breakpoints(mat, L, wmax):
    pts = [0.0]
    for w0 in mat.resonances():
        pts.append(w0)
    # plasmon-like poles of alpha: solve 1 + L chi = 0 on a fine real grid (minimum of |.|)
    w = np.linspace(1e-3, wmax, 200001)
    f = np.abs(1 + L * mat.chi(w))
    idx = np.where((f[1:-1] < f[:-2]) & (f[1:-1] < f[2:]))[0] + 1
    pts += list(w[idx])
    return sorted(set(p for p in pts if p < wmax))


def integrate_eV(f, mat, L, wmax=2000.0):
    """int_0^inf f(w) dw (w in eV) with breakpoints near resonances; tail by quad(inf)."""
    pts = _breakpoints(mat, L, 50.0)
    edges = [0.0]
    for p in pts[1:]:
        edges += [p * 0.9, p, p * 1.1]
    edges = sorted(set(e for e in edges if e > 0)) + [50.0, wmax]
    edges = [0.0] + edges
    tot, err = 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        v, e = quad(f, lo, hi, limit=400, epsabs=0, epsrel=1e-12)
        tot += v; err += e
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)   # tail is tiny; roundoff-limited
        v, e = quad(f, wmax, np.inf, limit=200, epsabs=0, epsrel=1e-10)
    return tot + v, err + e


def static_sum_rule_qs(mat, V, L, radiative=False):
    """int sigma_ext d lambda  (m^3) vs pi^2 alpha(0).  d lambda = 2 pi c dw/w^2."""
    f = lambda w: sigma_ext_qs(w, mat, V, L, radiative) * 2 * np.pi * C0 / (w * EV) ** 2 * EV
    val, err = integrate_eV(f, mat, L)
    alpha0 = V / L if np.isinf(mat.chi0) else V * mat.chi0 / (1 + L * mat.chi0)
    return val, np.pi ** 2 * alpha0


def highfreq_sum_rule_qs(mat, V, L, radiative=False, wmax=2000.0):
    """int sigma_ext dw (m^2 rad/s) vs pi w_p^2 V/(2c) and the printed pi w_p^2 V/2."""
    f = lambda w: sigma_ext_qs(w, mat, V, L, radiative) * EV
    val, err = integrate_eV(f, mat, L, wmax=wmax)
    wp2 = mat.wp2 * EV ** 2
    return val, np.pi * wp2 * V / (2 * C0), np.pi * wp2 * V / 2


# ------------------------------------------------------------------ C: Mie
def mie_sweep(mat, R_nm, w_eV, chunk=250):
    """sigma_ext, sigma_sca (nm^2) on a frequency grid (eV), per-chunk truncation."""
    w_eV = np.asarray(w_eV, float)
    ext = np.empty_like(w_eV); sca = np.empty_like(w_eV)
    for s in range(0, len(w_eV), chunk):
        w = w_eV[s:s + chunk]
        k = w / (HC_EVNM / (2 * np.pi))                 # 1/nm
        x = k * R_nm
        m = np.sqrt(1 + mat.chi(w))
        m = np.where(m.imag < 0, -m, m)
        cs = cross_sections(m, x, nmax=n_max(x))
        ext[s:s + chunk] = cs["ext"] / k ** 2
        sca[s:s + chunk] = cs["sca"] / k ** 2
    return ext, sca


def _grid(segments):
    return np.unique(np.concatenate([np.linspace(a, b, n) for a, b, n in segments]))


def mie_highfreq_sum_rule(mat=SILVERLIKE, R_nm=50.0, wmax_eV=400.0):
    """Running (2c/V) int_0^W sigma_ext dw / (pi w_p^2) with a C/w^2 tail extrapolation."""
    w = _grid([(1e-3, 2.0, 400), (2.0, 12.0, 4000), (12.0, 60.0, 2000), (60.0, wmax_eV, 2000)])
    ext, sca = mie_sweep(mat, R_nm, w)
    V = 4 * np.pi * R_nm ** 3 / 3
    run = np.concatenate([[0], np.cumsum(0.5 * (ext[1:] + ext[:-1]) * np.diff(w))])  # nm^2 eV
    # pi w_p^2 V/(2c) expressed in nm^2 eV (w in eV/hbar):  pi wp^2[eV^2] V[nm^3] / (2 hbar c[eV nm])
    norm = np.pi * mat.wp2 * V / (2 * (HC_EVNM / (2 * np.pi)))
    C_tail = ext[-1] * w[-1] ** 2
    slope_check = (ext[-200] * w[-200] ** 2) / C_tail
    total = run[-1] + C_tail / w[-1]
    return dict(w=w, ext=ext, sca=sca, running=run / norm, total=total / norm,
                tail_fraction=(C_tail / w[-1]) / norm, tail_flatness=slope_check)


def mie_static_sum_rule(mat=DIELECTRIC, R_nm=20.0):
    """int sigma_{ext,sca} d lambda vs pi^2 alpha(0), alpha(0) = 4 pi R^3 (eps-1)/(eps+2)."""
    w = _grid([(1e-3, 1.0, 400), (1.0, 20.0, 6000), (20.0, 200.0, 3000)])
    ext, sca = mie_sweep(mat, R_nm, w)
    lam_jac = 2 * np.pi * (HC_EVNM / (2 * np.pi)) / w ** 2      # d lambda [nm] per d w [eV]
    I = lambda s: np.trapezoid(s * lam_jac, w)
    # low-frequency analytic remainder (sigma ~ w^2 -> integrand const) and high tail (~1/w^4)
    low = ext[0] * lam_jac[0] * w[0]
    eps0 = 1 + mat.chi0
    target = np.pi ** 2 * 4 * np.pi * R_nm ** 3 * (eps0 - 1) / (eps0 + 2)
    return dict(ext=(I(ext) + low) / target, sca=I(sca) / target, target_nm3=target,
                absorption=(I(ext) - I(sca)) / target)


# ------------------------------------------------------------------ D: GenDisp
def kk_check(w, w0=1.0, wp=1.0, g=0.3):
    """Re h(w) for Lorentz h = wp^2/(w0^2 - w^2 - i g w), vs the correct KK relation
    (2/pi) PV int w' Im h/(w'^2 - w^2) and GenDisp as printed (no w' in numerator)."""
    h = lambda x: wp ** 2 / (w0 ** 2 - x ** 2 - 1j * g * x)
    Imh = lambda x: np.imag(h(x))
    W = 50.0
    # 1/(x^2 - w^2) = 1/((x - w)(x + w)) -> Cauchy weight 1/(x - w)
    kk = quad(lambda x: x * Imh(x) / (x + w), 0, W, weight="cauchy", wvar=w, limit=400)[0] \
        + quad(lambda x: x * Imh(x) / (x ** 2 - w ** 2), W, np.inf)[0]
    gd = quad(lambda x: Imh(x) / (x + w), 0, W, weight="cauchy", wvar=w, limit=400)[0] \
        + quad(lambda x: Imh(x) / (x ** 2 - w ** 2), W, np.inf)[0]
    return np.real(h(w)), 2 / np.pi * kk, 2 / np.pi * gd


# ------------------------------------------------------------------ E: nuclei
def plasma_contributions(Z, A):
    """w_p^2 contributions per atom: electrons Z e^2/m_e; nucleus (Z e)^2/M; neutrons 0.
    Returns nucleus/electron ratio and the 'free proton' ratio m_e/m_p."""
    me, mu, mp = 9.1093837e-31, 1.66053907e-27, 1.67262192e-27
    return Z * me / (A * mu), me / mp


def main(make_figure=True):
    print("=" * 86)
    print("Sec. 5.2.1 extinction sum rules (SI units; frequencies in eV/hbar)")
    print("=" * 86)
    print(f"materials: Drude-Lorentz 'silver-like' w_p = {np.sqrt(SILVERLIKE.wp2):.3f} eV (conductor);"
          f" Lorentz dielectric w_p = {np.sqrt(DIELECTRIC.wp2):.3f} eV, chi(0) = {DIELECTRIC.chi0:.4f}")

    print("\nA. Quasi-static spheroids (no radiative correction; exactly causal):")
    print(f"  {'material':10s} {'a x b (nm)':>11s} {'L':>7s} | {'int s dlam / pi^2 alpha0':>24s} |"
          f" {'int s dw / (pi wp^2 V/2c)':>25s} {'int s dw / (pi wp^2 V/2) [printed]':>35s}")
    rows = []
    for name, mat in (("metal", SILVERLIKE), ("dielectric", DIELECTRIC)):
        for a, b in ((10, 10), (30, 10), (60, 10)):
            a_m, b_m = a * 1e-9, b * 1e-9
            V = 4 * np.pi * a_m * b_m ** 2 / 3
            L = depolarization_prolate(a, b)
            s0, t0 = static_sum_rule_qs(mat, V, L)
            si, ti, tp = highfreq_sum_rule_qs(mat, V, L)
            rows.append((name, a, b, L, s0 / t0, si / ti, si / tp))
            print(f"  {name:10s} {a:4d} x {b:<4d} {L:7.4f} | {s0 / t0:24.10f} | {si / ti:25.10f} {si / tp:35.4e}")
    print(f"  -> printed SRExtInf is off by exactly 1/c = {1 / C0:.4e} s/m (dimensionally wrong); corrected form holds.")
    print("     Static rule depends on shape via alpha0 = V/L (conductor); high-frequency rule is shape-independent.")

    print("\nB. Same, with radiative correction alpha/(1 - i k^3 alpha/6pi)  [model caveat, not a tutorial error]:")
    a_m, b_m = 30e-9, 10e-9
    V = 4 * np.pi * a_m * b_m ** 2 / 3; L = depolarization_prolate(30, 10)
    s0, t0 = static_sum_rule_qs(SILVERLIKE, V, L, radiative=True)
    si, ti, _ = highfreq_sum_rule_qs(SILVERLIKE, V, L, radiative=True, wmax=1e6)
    wc = 6 * np.pi * C0 ** 3 / (V * SILVERLIKE.wp2 * EV ** 2) / EV
    print(f"  static: int s dlam / pi^2 alpha0 = {s0 / t0:.6f};  high-freq: int s dw / (pi wp^2 V/2c) = {si / ti:.4f}  (plateau alone gives exactly 1)")
    print(f"  (point-dipole Thomson plateau sigma_sca -> V^2 wp^4/(6 pi c^4) persists to w_c = 6pi c^3/(V wp^2) = {wc:.0f} eV;"
          f" the dipole model is invalid above kR ~ 1, so use Mie for the high-frequency rule)")

    print("\nC. Full Mie theory:")
    hf = mie_highfreq_sum_rule()
    iw = [np.searchsorted(hf["w"], x) for x in (5, 12, 30, 100, 399.9)]
    print(f"  silver-like sphere R = 50 nm: (2c/V) int_0^W sigma_ext dw / (pi w_p^2) at W = "
          + ", ".join(f"{hf['w'][i]:.0f} eV: {hf['running'][i]:.4f}" for i in iw))
    print(f"  + C/w^2 tail beyond 400 eV ({hf['tail_fraction']:.4f}; sigma w^2 flatness {hf['tail_flatness']:.3f})"
          f" -> total = {hf['total']:.6f}   (corrected SRExtInf: 1)")
    st = mie_static_sum_rule()
    print(f"  lossy Lorentz-dielectric sphere R = 20 nm, pi^2 alpha(0) = {st['target_nm3']:.4e} nm^3:")
    print(f"    int sigma_ext d lambda / pi^2 alpha0 = {st['ext']:.6f}   [SRExt0 holds]")
    print(f"    int sigma_sca d lambda / pi^2 alpha0 = {st['sca']:.4f}   [l.1825 'scattering cross-section' is wrong: it is extinction]")

    print("\nD. GenDisp (l.1793) vs KKRe/KKExt, Lorentz w0 = 1, wp = 1, gamma = 0.3:")
    print(f"  {'w':>5s} {'Re h exact':>12s} {'KK with w-prime':>16s} {'GenDisp printed':>16s}")
    kkrows = []
    for w in (0.5, 0.9, 1.5, 3.0):
        e, kk, gd = kk_check(w)
        kkrows.append((w, e, kk, gd))
        print(f"  {w:5.2f} {e:12.6f} {kk:16.6f} {gd:16.6f}")

    r_si, r_p = plasma_contributions(14, 28.0855)
    print("\nE. Plasma-frequency weight of nuclei (l.1844): neutrons carry no charge -> contribute exactly 0;")
    print(f"   Si nucleus/electrons = Z m_e/M = {r_si:.2e};  'free proton' m_e/m_p = {r_p:.2e}")

    if make_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
        w = np.linspace(0.05, 30, 6000)
        for (a, b), c in zip(((10, 10), (30, 10), (60, 10)), ("C0", "C1", "C2")):
            V = 4 * np.pi * (a * 1e-9) * (b * 1e-9) ** 2 / 3
            L = depolarization_prolate(a, b)
            s = sigma_ext_qs(w, SILVERLIKE, V, L)
            run = np.concatenate([[0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(w) * EV)])
            ax[0].plot(w, run / (np.pi * SILVERLIKE.wp2 * EV ** 2 * V / (2 * C0)), c,
                       label=f"quasi-static spheroid {a}x{b} nm")
        ax[0].plot(hf["w"], hf["running"], "k", lw=1.5, label="Mie sphere R = 50 nm")
        ax[0].axhline(1, color="gray", ls="--", lw=0.8)
        ax[0].set_xscale("log"); ax[0].set_xlim(0.5, 400)
        ax[0].set_xlabel(r"upper limit $\hbar\omega_{\max}$ (eV)")
        ax[0].set_ylabel(r"$\frac{2c}{\pi\omega_p^2 V}\int_0^{\omega_{\max}}\sigma_{\rm ext}\,d\omega$")
        ax[0].legend(fontsize=7, frameon=False)
        ax[1].semilogy(hf["w"], hf["ext"] / (np.pi * 50 ** 2), "k", label=r"$\sigma_{\rm ext}/\pi R^2$")
        ax[1].semilogy(hf["w"], hf["sca"] / (np.pi * 50 ** 2), "C3--", label=r"$\sigma_{\rm sca}/\pi R^2$")
        ax[1].set_xscale("log"); ax[1].set_xlim(0.5, 400); ax[1].set_ylim(1e-4, 30)
        ax[1].set_xlabel(r"$\hbar\omega$ (eV)"); ax[1].legend(fontsize=7, frameon=False)
        ax[1].set_title("Mie, silver-like Drude-Lorentz sphere", fontsize=9)
        fig.tight_layout()
        os.makedirs(FIGDIR, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIGDIR, f"sec5_extinction_sum_rules.{ext}"), dpi=200)
        print(f"\nwrote {os.path.normpath(os.path.join(FIGDIR, 'sec5_extinction_sum_rules.pdf'))}")


if __name__ == "__main__":
    main()
