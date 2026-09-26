"""Sec. 1.4 (Causality): Kramers-Kronig (KK) relations, their derivative form, the
conductor correction and the universal high-frequency asymptote, checked by adaptive
principal-value quadrature (time dependence e^{-i w t}).

Tutorial claims checked (tutorial.tex line numbers):
  [eq:KKRe]  l.368  Re h(w) =  (2/pi)  PV int_0^inf W Im h(W) / (W^2 - w^2) dW
  [eq:KKIm]  l.373  Im h(w) = -(2w/pi) PV int_0^inf   Re h(W) / (W^2 - w^2) dW
  [eq:dKKRe] l.388  d Re h/dw = (4w/pi) int_0^inf W Im h(W) / (W^2 - w^2)^2 dW in a
                    transparency window, hence > 0 for passive media (l.391, "normal dispersion")
  l.396      the standard KK relations also hold for causal, stable but NON-passive media
  l.397      conductors: KK for Re chi is unchanged; KK for Im chi acquires "+ sigma_0/omega"
  [eq:asymp] l.401  chi(w) -> -e^2 N/(m eps0 w^2) = -w_p^2/w^2 at high frequency

Tests:
  A. Random lossy multi-Lorentz insulators, one of them with an inverted (gain) oscillator:
     both KK relations at several frequencies (adaptive QAWC quadrature, to ~1e-9).
  B. dKKRe on a medium whose absorption is confined to two smooth, compactly supported
     bands: formula vs a 4th-order finite difference of the KK-computed Re chi, in the
     transparency windows below, between and above the bands; sign > 0.
  C. Drude + Lorentz conductor (silver-like parameters, eV units): KKRe holds unmodified;
     KKIm misses exactly sigma_0/(eps0 w).  The printed "sigma_0/omega" has units F/m and is
     missing 1/eps0 (fine only if chi means eps - eps0 in absolute units, which eq:asymp rules out).
  D. w^2 chi(w) -> -sum_j f_j (the oscillator strengths add up to w_p^2).

Run:  python kramers_kronig.py    (under 1 s; no figure)
"""
import os
import sys
import warnings
import numpy as np
from scipy.integrate import quad, IntegrationWarning

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EPS0 = 8.8541878128e-12          # F/m
EV = 1.0 / 6.582119569e-16       # rad/s per eV


# ------------------------------------------------------------------------ material models
class Oscillators:
    """chi(w) = sum_j f_j/(w_j^2 - w^2 - i g_j w) - fD/(w^2 + i gD w).
    f_j < 0 is an inverted (gain) oscillator: still causal and stable (poles in the LHP)."""

    def __init__(self, osc=(), fD=0.0, gD=0.0):
        self.osc, self.fD, self.gD = [tuple(o) for o in osc], fD, gD

    def chi(self, w):
        w = np.asarray(w, complex)
        out = np.zeros_like(w)
        for f, w0, g in self.osc:
            out = out + f / (w0**2 - w**2 - 1j * g * w)
        if self.fD:
            out = out - self.fD / (w**2 + 1j * self.gD * w)
        return out

    def re(self, w):
        return float(np.real(self.chi(w)))

    def im(self, w):
        return float(np.imag(self.chi(w)))

    def w_im(self, w):
        """W Im chi(W), written so that it stays finite at W -> 0 for the Drude term."""
        out = 0.0
        for f, w0, g in self.osc:
            out += f * g * w**2 / ((w0**2 - w**2)**2 + (g * w)**2)
        if self.fD:
            out += self.fD * self.gD / (w**2 + self.gD**2)
        return out

    @property
    def wp2(self):
        return sum(f for f, _, _ in self.osc) + self.fD

    @property
    def points(self):
        return [w0 for _, w0, _ in self.osc]


class BumpBands:
    """Absorption confined to bands: Im chi(W) = A sin^2(pi (W-a)/(b-a)) on [a, b], zero
    elsewhere (odd in W).  Re chi is obtained from KKRe numerically (no closed form used)."""

    def __init__(self, bands):
        self.bands = [tuple(b) for b in bands]

    def im(self, W):
        out = 0.0
        for A, a, b in self.bands:
            if a < W < b:
                out += A * np.sin(np.pi * (W - a) / (b - a))**2
        return out

    def w_im(self, W):
        return W * self.im(W)

    @property
    def points(self):
        return sorted({x for _, a, b in self.bands for x in (a, b)})


# ------------------------------------------------------------------------ KK quadratures
def _pv(g, w, upper, points):
    """PV int_0^inf g(W)/(W - w) dW, with g smooth near W = w.  QAWC on [w-d, w+d],
    ordinary adaptive quadrature elsewhere, analytic-free infinite tail."""
    d = min(0.5 * w, 0.25)
    lo, hi = w - d, w + d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        mid = quad(g, lo, hi, weight="cauchy", wvar=w, limit=500, epsabs=1e-13, epsrel=1e-12)[0]
        pl = [p for p in points if 0 < p < lo]
        left = quad(lambda x: g(x) / (x - w), 0, lo, points=pl or None, limit=500,
                    epsabs=1e-13, epsrel=1e-12)[0]
        pr = [p for p in points if hi < p < upper]
        right = quad(lambda x: g(x) / (x - w), hi, upper, points=pr or None, limit=500,
                     epsabs=1e-13, epsrel=1e-12)[0]
        tail = quad(lambda x: g(x) / (x - w), upper, np.inf, limit=500,
                    epsabs=1e-14, epsrel=1e-12)[0]
    return left + mid + right + tail


def kk_re(model, w, upper=60.0):
    """Right-hand side of [eq:KKRe]: (2/pi) PV int W Im chi/(W^2 - w^2)."""
    g = lambda W: model.w_im(W) / (W + w)
    return 2 / np.pi * _pv(g, w, upper, model.points)


def kk_im(model, w, upper=60.0):
    """Right-hand side of [eq:KKIm]: -(2w/pi) PV int Re chi/(W^2 - w^2)."""
    g = lambda W: model.re(W) / (W + w)
    return -2 * w / np.pi * _pv(g, w, upper, model.points)


def dkk_re(model, w):
    """Right-hand side of [eq:dKKRe] (no principal value: w must lie in a transparency window)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        val = sum(quad(lambda W: model.w_im(W) / (W**2 - w**2)**2, a, b, limit=400,
                       epsabs=1e-14, epsrel=1e-12)[0] for _, a, b in model.bands)
    return 4 * w / np.pi * val


# ------------------------------------------------------------------------ tests
def random_insulator(seed, n_osc=4, with_gain=False):
    rng = np.random.default_rng(seed)
    osc = [(rng.uniform(0.2, 2.0), rng.uniform(0.5, 4.0), rng.uniform(0.05, 0.6)) for _ in range(n_osc)]
    if with_gain:
        osc.append((-0.3, 2.2, 0.2))            # inverted Lorentzian: causal, stable, active
    return Oscillators(osc)


def test_A(freqs=(0.3, 1.1, 2.7, 5.0), seeds=(0, 1, 2)):
    rows = []
    for s in seeds:
        m = random_insulator(s, with_gain=(s == 2))
        for w in freqs:
            rows.append((s, w, m.re(w), kk_re(m, w), m.im(w), kk_im(m, w)))
    return rows


def test_B(ws=(0.4, 1.9, 4.5)):
    m = BumpBands([(0.8, 1.0, 1.6), (0.5, 2.4, 3.5)])
    rows = []
    for w in ws:
        h = 2e-3                                   # fourth-order central difference
        f = lambda x: kk_re(m, x, upper=10)
        fd = (-f(w + 2 * h) + 8 * f(w + h) - 8 * f(w - h) + f(w - 2 * h)) / (12 * h)
        rows.append((w, fd, dkk_re(m, w)))
    return rows


def silver_like():
    """eV units: Drude hbar*wD = 9.0 eV, hbar*gD = 0.021 eV, plus one interband oscillator."""
    return Oscillators([(20.0, 4.5, 1.0)], fD=9.0**2, gD=0.021)


def test_C(ws=(0.5, 1.5, 3.0)):
    m = silver_like()
    sigma0 = EPS0 * m.fD * EV / m.gD                  # S/m  (eps0 wD^2/gD, frequencies in rad/s)
    rows = []
    for w in ws:
        resid_im = m.im(w) - kk_im(m, w)              # dimensionless
        rows.append(dict(w=w, re=m.re(w), kkre=kk_re(m, w), resid=resid_im,
                         sig_eps0_w=sigma0 / (EPS0 * w * EV), sig_w=sigma0 / (w * EV)))
    return rows, sigma0


def test_D(m=None, w=3e3):
    m = m or random_insulator(0)
    return float(np.real(w**2 * m.chi(w))), -m.wp2


def main():
    print("=" * 86)
    print("Sec. 1.4  Kramers-Kronig relations (e^{-iwt}), adaptive PV quadrature")
    print("=" * 86)
    print("A. random multi-Lorentz media (seed 2 includes an inverted/gain oscillator, l.396)")
    print(f"{'seed':>4} {'w':>5} | {'Re chi':>12} {'KKRe rhs':>12} {'rel err':>9} | {'Im chi':>12} {'KKIm rhs':>12} {'rel err':>9}")
    for s, w, re, kre, im, kim in test_A():
        print(f"{s:4d} {w:5.2f} | {re:12.8f} {kre:12.8f} {abs(re-kre)/abs(re):9.1e} |"
              f" {im:12.8f} {kim:12.8f} {abs(im-kim)/abs(im):9.1e}")

    print("\nB. [eq:dKKRe] in transparency windows (absorption only in [1,1.6] and [2.4,3.5])")
    print(f"{'w':>5} | {'FD of KKRe Re chi':>18} {'dKKRe formula':>14} {'rel err':>9}  sign")
    for w, fd, f in test_B():
        print(f"{w:5.2f} | {fd:18.10f} {f:14.10f} {abs(fd-f)/abs(f):9.1e}  {'>0' if f > 0 else '<=0'}")

    rows, sigma0 = test_C()
    print(f"\nC. conductor (Drude 9.0 eV / 0.021 eV + Lorentz): sigma_0 = eps0 w_D^2/g_D = {sigma0:.3e} S/m")
    print(f"{'w[eV]':>5} | {'Re chi':>12} {'KKRe rhs':>12} | {'Im chi - KKIm':>14} {'s0/(eps0 w)':>12} {'ratio':>10} | printed s0/w [F/m]")
    for r in rows:
        print(f"{r['w']:5.2f} | {r['re']:12.6f} {r['kkre']:12.6f} | {r['resid']:14.6f} {r['sig_eps0_w']:12.6f}"
              f" {r['resid']/r['sig_eps0_w']:10.8f} | {r['sig_w']:.3e}")
    print("  -> KKRe unmodified; KKIm needs + sigma_0/(eps0 w) (dimensionless chi), not + sigma_0/w (l.397)")

    a, b = test_D()
    print(f"\nD. [eq:asymp] w^2 Re chi at w = 3e3: {a:.6f}   vs  -w_p^2 = -sum f_j = {b:.6f}")


if __name__ == "__main__":
    main()
