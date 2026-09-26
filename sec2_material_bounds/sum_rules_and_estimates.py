"""Secs. 2.1, 2.3, 2.4, 2.6, 2.7: the f-sum rule and its companions, the loss-bandwidth bound,
the Thomas-Reiche-Kuhn (TRK) sum rule, and the order-of-magnitude numbers quoted in the text.
Time dependence e^{-i w t}, chi(w) = int chi(t) e^{i w t} dt.

Tutorial claims checked (tutorial.tex line numbers):
  [eq:f-sum rule] l.748  int_0^inf w Im chi dw = pi w_p^2 / 2
  footnote l.746  "int_0^inf Re chi(w) dw = 0 ... the average value of Re chi over the entire
                  spectrum is zero"
  footnote l.757-760  Schwinger form int_{-inf}^{inf} -i w chi dw = pi w_p^2, read as the
                  polarization current at t = 0+ after an impulse E ~ delta(t), which depends only
                  on electron charge and mass ("frictional and binding forces have not time")
  l.833      loss over a band: Im chi <= pi w_p^2 / (2 Dw w_c)
  l.847      Lorentz-model glass: Im k ~ 10 cm^-1 -> "power attenuation on the order of 90%" in 1 mm
  footnote l.849  BCS gap Delta(0) ~ 1.764 k_B T_c
  l.891-899  generalized TRK: sum_n [w_n - (w_p + w_q)/2] x_pn x_nq = hbar/(2 m_e) delta_pq
  l.926      W ~ N U0 ~ 10^3 - 10^5 J/cm^3 for N = 1e22-1e23 cm^-3, U0 = 1-10 eV
  l.928      U_m = 2 m d^2/t^2; mechanical energy density < 1 J/cm^3 beyond "a few us" and
             < W beyond "tens of ns" for d = tens of um, rho = 1e-3 - 1e-2 kg/cm^3

Findings: all CONFIRMED except the l.746 footnote, which holds only for insulators: for a
conductor int_0^inf Re chi dw = -pi sigma_0/(2 eps0) (Altarelli et al. 1972 superconvergence
rule), e.g. every Drude metal has Re chi < 0 at all w.  Minor: the inverse Fourier transform
"evaluated at t = 0" returns the midpoint w_p^2/2 of the jump 0 -> w_p^2 of d chi/dt; the
t = 0+ value is w_p^2 (checked by QAWF quadrature).

Run:  python sum_rules_and_estimates.py    (under 1 s; no figure)
"""
import os
import sys
import warnings
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sec1_fundamental_principles.kramers_kronig import Oscillators


def _quad(f, a, b, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        return quad(f, a, b, limit=2000, epsabs=1e-13, epsrel=1e-11, **kw)[0]


def media():
    ins = Oscillators([(1.3, 1.0, 0.2), (0.7, 2.6, 0.5), (2.0, 4.0, 0.9)])
    met = Oscillators([(0.8, 2.0, 0.4)], fD=4.0, gD=0.3)
    return ins, met


def f_sum(m):
    top = max(m.points + [1.0]) + 10
    return _quad(m.w_im, 0, top, points=m.points) + _quad(m.w_im, top, np.inf)


def re_integral(m):
    top = max(m.points + [1.0]) + 10
    return _quad(m.re, 0, top, points=m.points) + _quad(m.re, top, np.inf)


def polarization_current(m, t):
    """J(t)/(eps0 E0) for E(t) = E0 delta(t):  (1/pi) int_0^inf w [Im chi cos wt - Re chi sin wt] dw.
    t = 0 is the symmetric (principal-value-at-infinity) integral of the footnote."""
    if t == 0:
        return f_sum(m) / np.pi
    s = np.sign(t)
    a = _quad(m.w_im, 0, np.inf, weight="cos", wvar=abs(t))
    b = _quad(lambda w: w * m.re(w), 0, np.inf, weight="sin", wvar=abs(t))
    return (a - s * b) / np.pi


def chi_dot_0plus(m):
    """Analytic time-domain slope: Lorentz f e^{-gt/2} sin(nu t)/nu -> f;  Drude (f/g)(1 - e^{-gt}) -> f."""
    return sum(f for f, _, _ in m.osc) + m.fD


def box_loss_bound(wp=3.0, wc=2.0, dw=0.5):
    """A box absorber Im chi = H on [wc - dw/2, wc + dw/2] with H at the l.833 bound exhausts the f-sum rule."""
    H = np.pi * wp**2 / (2 * dw * wc)
    fsum = _quad(lambda w: w * H, wc - dw / 2, wc + dw / 2)
    return H, fsum, np.pi * wp**2 / 2


def loss_bound_random(n=300, seed=0, wc=2.0, dw=0.5):
    """Random passive multi-Lorentz media: min_band Im chi / (pi w_p^2/(2 dw wc)) must be <= 1."""
    rng = np.random.default_rng(seed)
    wv = np.linspace(wc - dw / 2, wc + dw / 2, 101)
    worst = 0.0
    for _ in range(n):
        k = rng.integers(1, 5)
        m = Oscillators(zip(rng.uniform(0.1, 2, k), rng.uniform(0.5, 4, k), rng.uniform(0.05, 2, k)))
        ratio = np.imag(m.chi(wv)).min() / (np.pi * m.wp2 / (2 * dw * wc))
        worst = max(worst, ratio)
    return worst


def trk(n_grid=4000, L=8.0, pairs=((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 3))):
    """1D electron (hbar = m = 1) in an asymmetric anharmonic well, finite differences.
    Returns sum_n [E_n - (E_p + E_q)/2] x_pn x_nq for each (p, q); TRK predicts delta_pq / 2."""
    x = np.linspace(-L, L, n_grid)
    h = x[1] - x[0]
    V = 0.5 * x**2 + 0.15 * x**3 + 0.05 * x**4
    d = 1 / h**2 + V
    e = -0.5 / h**2 * np.ones(n_grid - 1)
    E, Psi = eigh_tridiagonal(d, e)
    out = {}
    for p, q in pairs:
        xp = (Psi[:, p] * x) @ Psi
        xq = (Psi[:, q] * x) @ Psi
        out[(p, q)] = float(np.sum((E - 0.5 * (E[p] + E[q])) * xp * xq))
    return out, h


def numbers():
    out = {}
    out["glass"] = 1 - np.exp(-2 * 10.0 * 0.1)                       # Im k = 10 /cm, L = 1 mm = 0.1 cm
    out["bcs"] = np.pi * np.exp(-np.euler_gamma)
    eV = 1.602176634e-19
    out["W"] = (1e22 * 1 * eV, 1e23 * 10 * eV)                        # J/cm^3
    t_for = lambda u, rho, d: d * np.sqrt(2 * rho / u)                # u = 2 rho d^2/t^2 (SI)
    rows = []
    for rho in (1e3, 1e4):                                            # kg/m^3 (1e-3, 1e-2 kg/cm^3)
        for d in (10e-6, 50e-6):
            rows.append((rho, d, t_for(1e6, rho, d), t_for(1e9, rho, d)))   # u = 1 J/cm^3, 1e3 J/cm^3
    out["mech"] = rows
    # kinematics: constant acceleration from rest over d in time t
    m, d, t = 1.0, 1.0, 1.0
    a = 2 * d / t**2
    out["Um"] = 0.5 * m * (a * t)**2 / (m * d**2 / t**2)
    return out


def main():
    print("=" * 88)
    print("Sec. 2  sum rules, loss-bandwidth bound, TRK, and quoted numbers")
    print("=" * 88)
    ins, met = media()
    for name, m in (("insulator", ins), ("conductor", met)):
        print(f"{name}: f-sum int w Im chi = {f_sum(m):.10f}   pi w_p^2/2 = {np.pi * m.wp2 / 2:.10f}")
    print("\nfootnote l.746:  int_0^inf Re chi dw")
    print(f"  insulator: {re_integral(ins):+.3e}   (claimed 0)")
    print(f"  conductor: {re_integral(met):+.10f}   -pi sigma_0/(2 eps0) = -pi fD/(2 gD) = {-np.pi * met.fD / (2 * met.gD):+.10f}")

    print("\nfootnote l.757-760: impulse response J(t)/(eps0 E0) = (1/2pi) int -i w chi e^{-i w t} dw")
    for name, m in (("insulator", ins), ("conductor", met)):
        vals = [(t, polarization_current(m, t)) for t in (-1e-3, 0.0, 1e-3, 1e-2)]
        print(f"  {name}: " + ",  ".join(f"t={t:+.0e}: {v:.6f}" for t, v in vals)
              + f";  w_p^2 = {m.wp2:.4f}, analytic d chi/dt(0+) = {chi_dot_0plus(m):.4f}")
    print("  -> 0 before, w_p^2 just after (independent of w_j, g_j); the symmetric integral at t = 0 is the midpoint w_p^2/2")

    H, fs, target = box_loss_bound()
    print(f"\nl.833: box absorber at the bound, Im chi = {H:.6f}: f-sum {fs:.10f} = pi w_p^2/2 = {target:.10f} (tight)")
    print(f"       300 random multi-Lorentz media: max of min_band Im chi / bound = {loss_bound_random():.4f} (<= 1)")

    t, h = trk()
    print(f"\nl.891-899 TRK (hbar = m = 1, anharmonic well, grid h = {h:.4f}; predicted delta_pq/2):")
    print("  " + ",  ".join(f"(p,q)={k}: {v:+.6f}" for k, v in t.items()))

    n = numbers()
    print("\nquoted numbers:")
    print(f"  l.847  Im k = 10 /cm over 1 mm: power attenuation 1 - e^(-2) = {n['glass']:.3f}  ('on the order of 90%')")
    print(f"  l.849  BCS pi e^(-gamma_E) = {n['bcs']:.4f}  (1.764)")
    print(f"  l.926  W = N U0: {n['W'][0]:.2e} to {n['W'][1]:.2e} J/cm^3  ('10^3 - 10^5 J/cm^3')")
    print(f"  l.928  U_m / (m d^2/t^2) = {n['Um']:.3f}  (2)")
    for rho, d, t1, t2 in n["mech"]:
        print(f"         rho = {rho / 1e6:.0e} kg/cm^3, d = {d * 1e6:.0f} um: u < 1 J/cm^3 for t > {t1 * 1e6:.2f} us;"
              f"  u < 1e3 J/cm^3 for t > {t2 * 1e9:.0f} ns")


if __name__ == "__main__":
    main()
