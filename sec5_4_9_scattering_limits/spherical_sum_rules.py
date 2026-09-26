"""Sec. 5.7 (sec:cloaking), causality relations and sum rules for spherical scattering
coefficients s_n = 1 - 2 a_n (Bohren-Huffman a_n, b_n; tutorial c_n = (s_n - 1)/2 = -a_n).
Units: sphere radius a = c = 1, so omega = x = k a.  Time dependence e^{-i omega t}.

Tutorial claims (tutorial.tex line numbers):
  l.2312  [eq:KK_scattering]
          Re[s_n(w) e^{2iwa/c}] = 1 + (w/pi) PV int Im[s_n(w) e^{2iw'a/c}] / (w'(w'-w)) dw'
          (as printed: s_n(w) inside the integral).
  l.2323  Blaschke product B_n = prod_i (1 - w/w_i)/(1 - w/w_i^*) over UHP zeros of s_n.
  l.2332  low-frequency expansion of h_n = -i log(s_n e^{2iwa/c}/B_n):
          h_n = 2a w/c + 2 sum_i w/Im w_{n,i} + O(w^2)                     (as printed)
  l.2340  [eq:sum_rule_sn]  (2/pi) int_0^inf ln(1/|s_n|)/w^2 dw <= 2a/c + 2 sum_i Im(1/w_{n,i})
  l.2350  |int_0^inf ln|s_n| / w^2 dw| <= pi a/c;   l.2353  |int ln|s_n(lambda)| d lambda| <= 2 pi^2 a
  l.2360  int_{w1}^{w2} dw/w^2 = (B/w0)/(1 - B^2/4) >= B/w0;  [eq:s_bound] |ln S0| B/pi <= k0 a
  l.2430  s_n = 2 c_n + 1; lossless |c_n| in [0, 1]; passive c_n in the disk |c + 1/2| <= 1/2.

Checks:
  A. PEC sphere, every multipole n = 1..6, TE and TM: f_n = s_n e^{2ix} is an exact all-pass
     rational function.  Its UHP zeros (roots of reverse-Bessel-type polynomials) give a
     Blaschke product with f_n/B_n = 1 to machine precision, so h_n = 0 identically and a_1 = 0.
     Then 2 + 2 sum Im(1/x_i) = 0 (i.e. sum Im(1/x_i) = -a/c for every n), consistent with
     l.2340 (tight: LHS = 0 for a lossless sphere), while the printed l.2332 coefficient
     2 + 2 sum 1/Im x_i is positive and wrong.
  B. [eq:KK_scattering] checked by principal-value quadrature for the PEC TE1 coefficient and for
     a lossy Lorentz sphere (own n = 1 Mie code, cross-checked against sec5_local_constraints/
     mie.py): holds with s_n(w') in the integrand.  The printed s_n(w) version is not even
     convergent: its integrand ~ Im s_n(w)/w'^2 as w' -> 0.
  C. [eq:sum_rule_sn] inequality for lossy spheres (n = 1, TM and TE): ratio of
     (2/pi) int ln(1/|s|)/x^2 dx to 2a/c is <= 1; also the lambda form.
  D. Algebra of l.2360 / [eq:s_bound].

Run:  python spherical_sum_rules.py   (about 3 s)
"""
import os
import sys
from math import factorial
import numpy as np
from scipy.special import spherical_jn, spherical_yn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sec5_local_constraints.mie import mie_coefficients


# ----------------------------------------------------------------------------------------
# A. PEC sphere: exact all-pass structure
# ----------------------------------------------------------------------------------------
def qpoly(n):
    """Coefficients (highest power first) of q_n(x) = x^n sum_k i^k (n+k)!/(k!(n-k)!) (2x)^-k,
    so that h_n^(1)(x) = (-i)^(n+1) e^{ix} q_n(x) / x^(n+1)."""
    return np.array([1j ** k * factorial(n + k) / (factorial(k) * factorial(n - k)) / 2 ** k
                     for k in range(n + 1)], complex)


def ppoly(n):
    """p_n(x) = i x q + x q' - n q, so that (x h_n^(1))' = (-i)^(n+1) e^{ix} p_n(x)/x^(n+1)."""
    q = np.poly1d(qpoly(n))
    p = np.poly1d([1j, 0]) * q + np.poly1d([1, 0]) * q.deriv() - n * q
    return p.coeffs


def pec_f(n, x, pol):
    """f = s e^{2ix} from Mie PEC coefficients (independent of the polynomial route):
    TE: b_n = j_n/h_n,  TM: a_n = psi_n'/xi_n'."""
    j = spherical_jn(n, x)
    y = spherical_yn(n, x)
    h = j + 1j * y
    if pol == "TE":
        c = j / h
    else:
        jp = spherical_jn(n, x, derivative=True)
        yp = spherical_yn(n, x, derivative=True)
        psi_p = j + x * jp
        xi_p = h + x * (jp + 1j * yp)
        c = psi_p / xi_p
    return (1 - 2 * c) * np.exp(2j * x)


def pec_allpass(n, pol, x=np.linspace(0.05, 20, 400)):
    coeffs = qpoly(n) if pol == "TE" else ppoly(n)
    poles = np.roots(coeffs)                    # zeros of h^(1) or (x h^(1))': LHP
    zeros = poles.conj()                        # UHP zeros of s_n
    B = np.prod((1 - x[:, None] / zeros) / (1 - x[:, None] / zeros.conj()), axis=1)
    f = pec_f(n, x, pol)
    return dict(n=n, pol=pol, nz=len(zeros), maxdev=np.max(np.abs(f / B - 1)),
                min_im=zeros.imag.min(), sum_im_inv=np.sum((1 / zeros).imag),
                sum_inv_im=np.sum(1 / zeros.imag), lossless=np.max(np.abs(np.abs(f) - 1)))


# ----------------------------------------------------------------------------------------
# own n = 1 Mie coefficient (complex argument via sin/cos), for long frequency grids
# ----------------------------------------------------------------------------------------
def _psi1(z):
    return np.sin(z) / z - np.cos(z)


def _dpsi1(z):
    return np.cos(z) / z - np.sin(z) / z ** 2 + np.sin(z)


def _xi1(x):
    return _psi1(x) + 1j * (-np.cos(x) / x - np.sin(x))      # x h_1^(1)(x)


def _dxi1(x):
    return _dpsi1(x) + 1j * (np.sin(x) / x + np.cos(x) / x ** 2 - np.cos(x))


def mie_ab1(m, x):
    """BH a_1, b_1 for relative index m (arrays)."""
    mx = m * x
    a = (m * _psi1(mx) * _dpsi1(x) - _psi1(x) * _dpsi1(mx)) / (m * _psi1(mx) * _dxi1(x) - _xi1(x) * _dpsi1(mx))
    b = (_psi1(mx) * _dpsi1(x) - m * _psi1(x) * _dpsi1(mx)) / (_psi1(mx) * _dxi1(x) - m * _xi1(x) * _dpsi1(mx))
    return a, b


def lorentz_index(x, wp=3.0, w0=1.2, g=0.4, einf=1.0):
    eps = einf + wp ** 2 / (w0 ** 2 - x ** 2 - 1j * g * x)
    m = np.sqrt(eps)
    return np.where(m.imag < 0, -m, m)


MATERIALS = [dict(name="Lorentz  wp=3,  w0=1.2, g=0.4", wp=3.0, w0=1.2, g=0.4),
             dict(name="Lorentz  wp=6,  w0=2.0, g=2.0", wp=6.0, w0=2.0, g=2.0),
             dict(name="Drude-ish wp=8, w0=0.05, g=1.0", wp=8.0, w0=0.05, g=1.0),
             dict(name="weak     wp=1,  w0=1.0, g=0.1", wp=1.0, w0=1.0, g=0.1)]


def fvals(x, mat, which="TM"):
    m = lorentz_index(x, mat["wp"], mat["w0"], mat["g"])
    a, b = mie_ab1(m, x)
    c = a if which == "TM" else b
    return (1 - 2 * c) * np.exp(2j * x)


# ----------------------------------------------------------------------------------------
# B. KK relation with a once-subtracted PV integral
# ----------------------------------------------------------------------------------------
def kk_check(f, xlist, X=400.0, dx=2e-3, typo=False):
    """Re f(x) vs 1 + (2x^2/pi) PV int_0^inf G(x')/(x'^2 - x^2) dx',  G = Im f(x')/x'.
    typo=True evaluates the printed version Im[s(x) e^{2ix'}]/x' (s frozen at x)."""
    xp = np.arange(dx / 2, X, dx)
    fp = f(xp)
    out = []
    for x in xlist:
        fx = f(np.array([x]))[0]
        if typo:
            s_x = fx * np.exp(-2j * x)
            G = (s_x * np.exp(2j * xp)).imag / xp
            Gx = fx.imag / x
        else:
            G = fp.imag / xp
            Gx = fx.imag / x
        integrand = (G - Gx) / (xp ** 2 - x ** 2)
        I = np.sum(integrand) * dx
        I += -Gx * np.log((X + x) / (X - x)) / (2 * x)      # tail of the -G(x) piece
        rhs = 1 + 2 * x ** 2 / np.pi * I
        out.append((x, fx.real, rhs))
    return out


# ----------------------------------------------------------------------------------------
# C. sum rule for lossy spheres
# ----------------------------------------------------------------------------------------
def sum_rule_ratio(mat, which="TM", X=2000.0):
    u = np.linspace(np.log(1e-3), np.log(X), 200001)
    x = np.exp(u)
    s = fvals(x, mat, which) * np.exp(-2j * x)
    integrand = -np.log(np.abs(s)) / x ** 2 * x
    lhs = 2 / np.pi * np.trapezoid(integrand, u)
    lam_integral = 2 * np.pi * np.trapezoid(integrand, u)     # int ln(1/|s|) d lambda, lambda = 2 pi/x
    return lhs / 2.0, lam_integral / (2 * np.pi ** 2), np.max(np.abs(s))


def main():
    print("=" * 86)
    print("Sec. 5.7  spherical scattering coefficients: KK relation, Blaschke term, sum rule (a = c = 1)")
    print("=" * 86)
    print("\nA. PEC sphere: f_n = s_n e^{2ix} vs Blaschke product of its UHP zeros")
    print(f"   {'n':>2} {'pol':>3} {'#zeros':>6} {'max|f/B-1|':>11} {'||f|-1|':>8} {'sum Im(1/x_i)':>14}"
          f" {'a1 correct':>11} {'a1 printed':>11}")
    for n in range(1, 7):
        for pol in ("TE", "TM"):
            r = pec_allpass(n, pol)
            print(f"   {n:2d} {pol:>3} {r['nz']:6d} {r['maxdev']:11.1e} {r['lossless']:8.1e}"
                  f" {r['sum_im_inv']:14.10f} {2 + 2*r['sum_im_inv']:11.1e} {2 + 2*r['sum_inv_im']:11.4f}")
    print("   -> h_n = -i log(f/B) = 0 identically; low-frequency coefficient is 2a/c + 2 sum Im(1/w_i) = 0,")
    print("      matching l.2340 with equality.  The printed 2a/c + 2 sum 1/Im w_i (l.2332, l.2388) is wrong.")

    # cross-check own n=1 Mie with mie.py
    x = np.linspace(0.1, 30, 50)
    m = lorentz_index(x)
    a_ref, b_ref = mie_coefficients(m, x, nmax=1)
    a1, b1 = mie_ab1(m, x)
    print(f"\n   own n=1 Mie vs sec5_local_constraints/mie.py: max|da| = {np.max(abs(a1-a_ref[:,0])):.1e},"
          f" max|db| = {np.max(abs(b1-b_ref[:,0])):.1e}")

    print("\nB. [eq:KK_scattering]: Re f(x) vs 1 + (x/pi) PV int Im f(x')/(x'(x'-x)) dx'")
    pec = lambda x: pec_f(1, x, "TE")
    mat = MATERIALS[0]
    lor = lambda x: fvals(x, mat, "TM")
    print("   (grid x' = dx/2, 3dx/2, ...; the printed integrand ~ Im s_n(w)/w'^2 as w' -> 0, so it diverges")
    print("    logarithmically and its value depends on the low-frequency cutoff dx/2)")
    for lab, f in (("PEC TE1", pec), ("Lorentz sphere TM1", lor)):
        good = kk_check(f, [0.5, 1.5, 4.0])
        good2 = kk_check(f, [0.5, 1.5, 4.0], dx=5e-4)
        bad = kk_check(f, [0.5, 1.5, 4.0], typo=True)
        bad2 = kk_check(f, [0.5, 1.5, 4.0], dx=5e-4, typo=True)
        print(f"   {lab}:  [dx = 2e-3 | dx = 5e-4]")
        for (x, re, rhs), (_, _, rhs2), (_, _, rbad), (_, _, rbad2) in zip(good, good2, bad, bad2):
            print(f"      x = {x:3.1f}: Re f = {re:+.5f}   KK with s(w') = {rhs:+.5f} | {rhs2:+.5f}"
                  f"   printed s(w) = {rbad:+.5f} | {rbad2:+.5f}")

    print("\nC. [eq:sum_rule_sn] for lossy Lorentz spheres (n = 1): (2/pi) int ln(1/|s|)/x^2 dx  /  (2a/c)")
    for mat in MATERIALS:
        for which in ("TM", "TE"):
            r, rl, smax = sum_rule_ratio(mat, which)
            print(f"   {mat['name']}  {which}: ratio = {r:.4f}  (lambda form / 2 pi^2 a = {rl:.4f}; max|s| = {smax:.4f})")

    print("\nD. l.2360: int_{w0(1-B/2)}^{w0(1+B/2)} dw/w^2 = (B/w0)/(1 - B^2/4)")
    for B in (0.1, 0.5, 1.0):
        w0 = 2.0
        lhs = 1 / (w0 * (1 - B / 2)) - 1 / (w0 * (1 + B / 2))
        print(f"   B = {B}: direct {lhs:.10f}, formula {(B/w0)/(1-B**2/4):.10f}")


if __name__ == "__main__":
    main()
