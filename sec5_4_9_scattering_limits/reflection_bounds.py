"""Sec. 5.7 (sec:cloaking), bounds on plane-wave / guided-wave reflection: Bode-Fano,
Rozanov, and the reflection-phase KK relation.  Units: c = eps0 = mu0 = eta0 = 1, R = C = 1,
slab thickness d = 1.  Time dependence e^{-i omega t}; Laplace variable s = -i omega.

Tutorial claims (tutorial.tex line numbers):
  l.2407  "For a parallel LC load, the load impedance is Z(omega) = R/(1 - i omega R C)"
          (the load is a parallel RC; Z is right).
  l.2407  Gamma(omega) = -1 - 2i/(RC) omega^-1 + O(omega^-2)          (as printed)
          h(omega)     = A - 2/(RC) omega^-1 + 2 sum_i omega^-1 Im omega_i + O(omega^-2)
  l.2410  [eq:Bode_Fano]  |int_0^inf ln|Gamma| d omega| <= pi/(RC);  [eq:Gamma_bound_BF]
  l.2420  |Gamma| = |Gamma'| (source side vs load side); expansion independent of network.
  l.2388  Rozanov: Gamma = -1 - i(omega/c) 2 sum mu_s d + O(omega^2)
          h = pi + (omega/c) 2 sum mu_s d + 2 sum_i omega/Im omega_i + O(omega^2)   (as printed)
  l.2391  |int_0^inf ln|Gamma(lambda)| d lambda| <= 2 pi^2 sum mu_s d;  [eq:Gamma_bound]
  l.2381  [eq:KK_reflection] arg Gamma = -(2 omega/pi) PV int_0^inf ln|Gamma(w')|/(w'^2 - w^2) dw'

Checks:
  A. Parallel-RC load: direct evaluation of (Gamma + 1) omega RC at large omega -> +2i (not -2i).
     The printed h expansion (-2/(RC)) is consistent with +2i, i.e. only the Gamma line has the
     sign slip.
  B. Random lossless LC ladders between a source R and the RC load, built as exact rational
     functions of s.  |Gamma_in| = |Gamma_load-side| on the real axis; the load-side Gamma' has
     the network-independent expansion -1 + 2i/(omega RC) (the source-side Gamma_in does not:
     it tends to +1 behind a series-L input, and its UHP zeros differ); the Bode-Fano integral obeys Fano's
     identity  int_0^inf ln(1/|Gamma|) d omega = pi/(RC) - pi sum_{UHP zeros} Im omega_i  to
     quadrature accuracy; the bound pi/(RC) and the band form |ln Gamma_0| Delta omega <= pi/RC
     hold for every network.
  C. Rozanov: PEC-backed Debye and magnetic-Lorentz slabs: low-frequency coefficient
     (Gamma + 1)/(-2 i omega d) -> mu_s, and the all-wavelength integral <= 2 pi^2 mu_s d.
  D. The Blaschke term in h.  For a lossless nondispersive PEC-backed slab (index n) the UHP
     zeros of Gamma are known in closed form, omega_m = [pi/2 + m pi + i atanh(1/n)]/(n d).
     The correct low-frequency term of -i log(Gamma/B) is 2 omega sum Im(1/omega_m) (this is
     what Eq. sum_rule_sn at l.2340 uses); it converges to -2 omega d/c, so that a_1 = 0 and the
     lossless sum rule (LHS = 0) is tight.  The printed 2 sum omega/Im omega_m has the wrong sign
     and diverges linearly with the number of zeros.
  E. [eq:KK_reflection] for a vacuum/Lorentz-medium interface: holds for r = (n-1)/(n+1); for the
     E-field / transmission-line convention Gamma = (Z_l - Z_0)/(Z_l + Z_0) = (1-n)/(1+n) used
     later in the section, arg Gamma = (KK integral) + pi.

Run:  python reflection_bounds.py   (about 5 s; writes figures/sec5_4_9_reflection_bounds.png/pdf)
"""
import os
import sys
import warnings
import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy.integrate import quad, IntegrationWarning

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
S = Poly([0.0, 1.0])


# ----------------------------------------------------------------------------------------
# A. parallel RC load
# ----------------------------------------------------------------------------------------
def z_parallel_rc(w, R=1.0, C=1.0):
    """1/(1/R + 1/Z_C) with Z_C = 1/(-i w C) = i/(w C) for e^{-i w t}."""
    return 1.0 / (1.0 / R + 1.0 / (1j / (w * C)))


def gamma_rc(w, R=1.0, C=1.0):
    z = z_parallel_rc(w, R, C) / R          # Z_0 = R
    return (z - 1) / (z + 1)


def rc_highfreq_coefficient(w=(1e2, 1e3, 1e4), R=1.0, C=1.0):
    """(Gamma + 1) * omega R C at large omega (tutorial: -2i).  Also (Re h - pi) omega R C
    with h = -i log Gamma (tutorial: -2)."""
    w = np.asarray(w, float)
    g = gamma_rc(w, R, C)
    h = -1j * np.log(g)
    return (g + 1) * w * R * C, (h.real - np.pi) * w * R * C


# ----------------------------------------------------------------------------------------
# B. lossless ladders, rational in s
# ----------------------------------------------------------------------------------------
def random_ladder(rng, nsec):
    """Elements ordered from the load toward the source; the element next to the load is a
    series L (a shunt C there would simply add to the load capacitance)."""
    el = []
    for _ in range(nsec):
        el.append(("L", 10 ** rng.uniform(-0.7, 0.7)))
        el.append(("C", 10 ** rng.uniform(-0.7, 0.7)))
    if rng.random() < 0.5:
        el.append(("L", 10 ** rng.uniform(-0.7, 0.7)))
    return el


def _add(N, D, kind, v):
    if kind == "L":
        return N + v * S * D, D                  # series inductor: Z -> Z + sL
    return N, D + v * S * N                      # shunt capacitor: Y -> Y + sC


def z_input(elems, R=1.0, C=1.0):
    """Impedance seen by the source (N/D polynomials in s)."""
    N, D = Poly([R]), Poly([1.0, R * C])         # R || C
    for kind, v in elems:
        N, D = _add(N, D, kind, v)
    return N, D


def z_back(elems, R=1.0, C=1.0):
    """Impedance seen by the load resistor R looking back into C || (network + source R)."""
    N, D = Poly([R]), Poly([1.0])
    for kind, v in reversed(elems):
        N, D = _add(N, D, kind, v)
    return N, D + C * S * N


def gamma_from_z(N, D, R=1.0):
    return N - R * D, N + R * D                  # numerator, denominator polynomials


def evalg(num, den, w):
    s = -1j * np.asarray(w)
    return num(s) / den(s)


def bode_fano_integral(num, den):
    f = lambda w: -np.log(abs(evalg(num, den, w)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        I1 = quad(f, 0, 1, limit=400)[0]
        I2 = quad(f, 1, 20, limit=400)[0]
        I3 = quad(f, 20, np.inf, limit=400)[0]
    return I1 + I2 + I3


def uhp_zero_sum(num):
    """sum Im(omega_i) over zeros in Im omega > 0  (<=> Re s > 0)."""
    r = num.roots()
    return float(np.sum(r.real[r.real > 1e-9]))


def widest_band(num, den, G0, w=np.linspace(1e-3, 12, 24001)):
    """Width of the widest contiguous band with |Gamma| <= G0."""
    ok = np.abs(evalg(num, den, w)) <= G0
    best = cur = 0
    for v in ok:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best * (w[1] - w[0])


def ladder_study(n_networks=40, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_networks):
        el = random_ladder(rng, rng.integers(1, 4))
        gi = gamma_from_z(*z_input(el))
        gb = gamma_from_z(*z_back(el))
        wt = np.linspace(0.01, 30, 400)
        mag_diff = np.max(np.abs(np.abs(evalg(*gi, wt)) - np.abs(evalg(*gb, wt))))
        wl = np.array([1e3, 1e4])
        coef = (evalg(*gb, wl) + 1) * wl          # R = C = 1
        I = bode_fano_integral(*gb)
        zsum = uhp_zero_sum(gb[0])
        G0 = 0.3
        band = widest_band(*gb, G0)
        rows.append(dict(I=I, fano=np.pi - np.pi * zsum, zsum=zsum, mag_diff=mag_diff,
                         coef=coef[-1], band_product=abs(np.log(G0)) * band, nel=len(el),
                         gin_inf=evalg(*gi, np.array([1e5]))[0],
                         fano_in=np.pi - np.pi * uhp_zero_sum(gi[0])))
    return rows


# ----------------------------------------------------------------------------------------
# C, D. Rozanov
# ----------------------------------------------------------------------------------------
def gamma_pec_slab(w, eps, mu, d=1.0):
    n = np.sqrt(eps * mu + 0j)
    n = np.where(n.imag < 0, -n, n)
    eta = np.sqrt(mu / eps + 0j)
    eta = np.where(eta.real < 0, -eta, eta)
    zin = -1j * eta * np.tan(n * w * d)
    return (zin - 1) / (zin + 1)


DEBYE = dict(name="Debye dielectric eps = 2 + 8/(1 - i w), mu = 1", mu_s=1.0,
             eps=lambda w: 2 + 8 / (1 - 1j * w), mu=lambda w: 1 + 0 * w)
MAGNETIC = dict(name="magnetic Lorentz mu_s = 10 (w_m = 0.3, g = 0.5), eps = 5", mu_s=10.0,
                eps=lambda w: 5 + 0 * w,
                mu=lambda w: 1 + 9 * 0.09 / (0.09 - w ** 2 - 0.5j * w))


def rozanov_check(mat, d=1.0):
    w0 = np.array([1e-4, 1e-3])
    g = gamma_pec_slab(w0, mat["eps"](w0), mat["mu"](w0), d)
    coef = ((g + 1) / (-2j * w0 * d))
    # all-wavelength integral = 2 pi c int_0^inf ln(1/|Gamma|)/w^2 dw, log grid
    u = np.linspace(np.log(1e-5), np.log(3e3), 400001)
    w = np.exp(u)
    gg = gamma_pec_slab(w, mat["eps"](w), mat["mu"](w), d)
    integrand = -np.log(np.abs(gg)) / w ** 2 * w          # d w = w du
    I_lambda = 2 * np.pi * np.trapezoid(integrand, u)
    return coef[0], I_lambda, 2 * np.pi ** 2 * mat["mu_s"] * d


def lossless_slab_zeros(n, d=1.0, M=1000):
    m = np.arange(-M, M)
    return (np.pi / 2 + m * np.pi + 1j * np.arctanh(1 / n)) / (n * d)


def blaschke_terms(n=2.0, d=1.0, Ms=(10, 100, 1000, 10000)):
    """Partial sums (over 2M zeros) of the correct and the printed low-frequency terms."""
    out = []
    for M in Ms:
        z = lossless_slab_zeros(n, d, M)
        out.append((M, np.sum((1 / z).imag), np.sum(1 / z.imag)))
    z = lossless_slab_zeros(n, d, 20)
    resid = np.max(np.abs(gamma_pec_slab(z, n ** 2 + 0 * z, 1 + 0 * z, d)))
    return out, resid


def h_slope_numeric(n=2.0, d=1.0, M=20000, dw=1e-4):
    """Low-frequency slope of h = -i log(Gamma/B) with B from the analytic zeros (truncated).
    For a lossless slab Gamma/B has unit modulus; the slope should be a_1 = 0."""
    z = lossless_slab_zeros(n, d, M)
    w = np.array([dw])
    G = gamma_pec_slab(w, n ** 2 + 0 * w, 1 + 0 * w, d)
    logB = np.sum(np.log((1 - w[:, None] / z) / (1 - w[:, None] / z.conj())), axis=1)
    h = -1j * (np.log(-G) - logB)          # log(-Gamma): take out the constant i pi
    return (h.real / dw)[0]


# ----------------------------------------------------------------------------------------
# E. reflection phase KK relation
# ----------------------------------------------------------------------------------------
def lorentz_eps(w, wp=1.5, w0=1.0, g=0.3):
    return 1 + wp ** 2 / (w0 ** 2 - w ** 2 - 1j * g * w)


def kk_reflection_phase(wlist, sign=+1, W=2e4, npts=2_000_001):
    """theta(w) = -(2w/pi) int_0^inf [ln|r(w')| - ln|r(w)|]/(w'^2 - w^2) dw' (subtracted form),
    r = sign * (n-1)/(n+1).  Tail beyond W uses ln|r| ~ ln(wp^2/4) - 2 ln w'."""
    def r(w):
        n = np.sqrt(lorentz_eps(w))
        n = np.where(n.imag < 0, -n, n)
        return sign * (n - 1) / (n + 1)
    u = np.linspace(np.log(1e-6), np.log(W), npts)
    wp_ = np.exp(u)
    lr = np.log(np.abs(r(wp_)))
    out = []
    for w in wlist:
        lw = np.log(np.abs(r(np.array([w]))))[0]
        den = wp_ ** 2 - w ** 2
        f = (lr - lw) / np.where(np.abs(den) < 1e-14, 1e-14, den) * wp_
        I = np.trapezoid(f, u)
        # small-w' part [0, 1e-6] and tail (W, inf)
        I += (np.log(np.abs(r(np.array([1e-7]))))[0] - lw) * 1e-6 / (-w ** 2)
        c = np.log(1.5 ** 2 / 4) - lw
        I += c / W - 2 * (np.log(W) + 1) / W
        theta = -2 * w / np.pi * I
        out.append((w, theta, np.angle(r(np.array([w])))[0]))
    return out


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def main(make_figure=True):
    print("=" * 84)
    print("Sec. 5.7  Bode-Fano, Rozanov and reflection-phase relations (c = R = C = d = 1)")
    print("=" * 84)
    print("\nA. Parallel RC load, Z = R/(1 - i w R C) [l.2407 calls it 'parallel LC']")
    zc = z_parallel_rc(np.array([0.7]))[0]
    print(f"   Z(0.7) from 1/(1/R + 1/Z_C) = {zc:.6f};  R/(1 - 0.7i) = {1/(1-0.7j):.6f}")
    cg, ch = rc_highfreq_coefficient()
    for w, a, b in zip((1e2, 1e3, 1e4), cg, ch):
        print(f"   w RC = {w:7.0f}:  (Gamma+1) w RC = {a.real:+.4f}{a.imag:+.4f}i   "
              f"(Re h - pi) w RC = {b:+.4f}")
    print("   -> Gamma = -1 + 2i/(w RC) + ...  (tutorial prints -1 - 2i/(w RC));"
          " h coefficient -2/(RC) as printed")
    I0 = bode_fano_integral(*gamma_from_z(Poly([1.0]), Poly([1.0, 1.0])))
    print(f"   bare load: int_0^inf ln(1/|Gamma|) dw = {I0:.8f}   pi/(RC) = {np.pi:.8f}")

    print("\nB. Random lossless LC ladders between source R and the RC load (R = C = 1)")
    rows = ladder_study()
    print(f"   {'#el':>3} {'int ln(1/|G|)dw':>16} {'pi - pi sum Im w_i':>19} {'sum Im w_i':>10}"
          f" {'(G+1)w RC':>16} {'|lnG0| dw':>9}")
    for r in rows[:10]:
        print(f"   {r['nel']:3d} {r['I']:16.8f} {r['fano']:19.8f} {r['zsum']:10.5f}"
              f" {r['coef'].real:+7.4f}{r['coef'].imag:+7.4f}i {r['band_product']:9.4f}")
    err = max(abs(r["I"] - r["fano"]) for r in rows)
    print(f"   ... {len(rows)} networks: max |integral - Fano identity| = {err:.1e};"
          f" max integral/(pi/RC) = {max(r['I'] for r in rows)/np.pi:.6f};"
          f" max |ln G0| dw /(pi/RC) = {max(r['band_product'] for r in rows)/np.pi:.4f} (G0 = 0.3)")
    print(f"   max ||Gamma_in| - |Gamma_load-side|| on real axis = {max(r['mag_diff'] for r in rows):.1e}")
    nplus = sum(r["gin_inf"].real > 0 for r in rows)
    bad_in = sum(abs(r["I"] - r["fano_in"]) > 1e-6 for r in rows)
    print(f"   source-side Gamma_in(w -> inf) = +1 for {nplus}/{len(rows)} networks (series-L input), and Fano's identity"
          f" with Gamma_in's own UHP zeros fails for {bad_in}/{len(rows)}:")
    print("   the network-independent expansion (l.2420) belongs to the load-side Gamma', not to Gamma at the source.")

    print("\nC. Rozanov bound, PEC-backed slabs (d = 1)")
    for mat in (DEBYE, MAGNETIC):
        coef, I, B = rozanov_check(mat)
        print(f"   {mat['name']}")
        print(f"      (Gamma+1)/(-2 i w d) at w = 1e-4: {coef.real:.5f}{coef.imag:+.5f}i  (mu_s = {mat['mu_s']})")
        print(f"      int_0^inf ln(1/|Gamma|) d lambda = {I:.4f}  <=  2 pi^2 mu_s d = {B:.4f}  (ratio {I/B:.4f})")

    print("\nD. Blaschke term: lossless PEC-backed slab, n = 2, UHP zeros w_m = [pi/2 + m pi + i atanh(1/n)]/(n d)")
    parts, resid = blaschke_terms()
    print(f"   max |Gamma(w_m)| over 40 zeros = {resid:.1e}")
    print(f"   {'2M zeros':>9} {'sum Im(1/w_m)':>15} {'sum 1/Im w_m':>14}")
    for M, a, b in parts:
        print(f"   {2*M:9d} {a:15.6f} {b:14.1f}")
    a1 = 2 * 1.0 + 2 * parts[-1][1]
    print(f"   correct a_1 = 2d/c + 2 sum Im(1/w_m) -> {a1:.2e} (exact 0: sum = -d/c);"
          f" numerical slope of h: {h_slope_numeric():.2e}")
    print("   printed a_1 = 2d/c + 2 sum 1/Im w_m diverges (and has the wrong sign)")

    print("\nE. [eq:KK_reflection], vacuum / Lorentz medium (wp = 1.5, w0 = 1, g = 0.3)")
    for sign, lab in ((+1, "r = (n-1)/(n+1)"), (-1, "r = (1-n)/(1+n)")):
        res = kk_reflection_phase([0.3, 0.9, 1.3, 3.0], sign)
        s = "  ".join(f"w={w}: KK {t:+.4f} arg {a:+.4f}" for w, t, a in res)
        dev = max(abs(wrap(t - a)) for w, t, a in res)
        print(f"   {lab}: {s}\n      max |KK - arg| (mod 2 pi) = {dev:.3f}")

    if make_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
        w = np.logspace(0, 4, 200)
        g = gamma_rc(w)
        ax[0].semilogx(w, ((g + 1) * w).imag, "C0-", label=r"Im$[(\Gamma+1)\omega RC]$")
        ax[0].axhline(2, color="C0", ls=":", lw=1)
        ax[0].axhline(-2, color="C3", ls="--", lw=1, label="printed (l.2407)")
        ax[0].set_xlabel(r"$\omega RC$")
        ax[0].set_ylim(-2.5, 2.5)
        ax[0].legend(fontsize=7, frameon=False)
        ax[0].set_title("parallel-RC load", fontsize=9)
        ax[1].plot([r["fano"] / np.pi for r in rows], [r["I"] / np.pi for r in rows], "o", ms=4)
        ax[1].plot([0, 1], [0, 1], "k:", lw=1)
        ax[1].set_xlabel(r"$1 - \sum_i \mathrm{Im}\,\omega_i\, RC$")
        ax[1].set_ylabel(r"$\int_0^\infty \ln|\Gamma|^{-1} d\omega \,/\, (\pi/RC)$")
        ax[1].set_title("random lossless ladders", fontsize=9)
        Ms = np.array([p[0] for p in parts])
        ax[2].loglog(2 * Ms, [abs(p[1] + 1) for p in parts], "C0o-", label=r"$|\sum \mathrm{Im}(1/\omega_m) + d/c|$")
        ax[2].loglog(2 * Ms, [p[2] for p in parts], "C3s--", label=r"$\sum 1/\mathrm{Im}\,\omega_m$ (printed)")
        ax[2].set_xlabel("number of UHP zeros kept")
        ax[2].legend(fontsize=7, frameon=False)
        ax[2].set_title("lossless slab on PEC, n = 2", fontsize=9)
        fig.tight_layout()
        os.makedirs(FIGDIR, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIGDIR, f"sec5_4_9_reflection_bounds.{ext}"), dpi=200)
        print(f"\nwrote {os.path.normpath(os.path.join(FIGDIR, 'sec5_4_9_reflection_bounds.pdf'))}")


if __name__ == "__main__":
    main()
