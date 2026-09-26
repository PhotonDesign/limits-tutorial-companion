"""Sec. 1.5 (Passivity) and 1.5.1 (Herglotz functions): bounded-real scattering matrices,
the Poisson representation, the Herglotz (Nevanlinna) representation of omega*T(omega), and
the Herglotz sum rules.  Time dependence e^{-i w t}; natural units c = 1.

Tutorial claims checked (tutorial.tex line numbers):
  l.428-431  passive S(w): analytic in the UHP, I - S^dag S >= 0 in the UHP, S*(w) = S(-w*)
  [eq:SMatRep] l.434  S(w0 + i g) = (g/pi) int S(w') / ((w' - w0)^2 + g^2) dw'
  l.437      "scatterers of nonzero size can excite outgoing coefficients before ... Recovering
             strong passivity requires size-dependent phase shifts" in the definition of S
  l.451-455  passive T: w T(w) analytic, Im[w T(w)] >= 0 in the UHP, w* T*(w) = -[w T(w)]_{-w*}
  l.461      C = Re[i T(i)], D = lim_{y->inf} T(iy);  l.463  w T(w) = int dOmega(l)/(l - w)
  [eq:h_sum_rules] l.489, [eq:h_sum_rule_p1] l.505, [eq:h_sum_rule_p0] l.511, with the
             asymptotic coefficients of l.476-478 and the parity statement of l.483

Tests:
  A. Two-port S matrix of a lossy Lorentz slab (thickness L), ports referenced at the slab
     faces: sigma_max(S) <= 1 on a UHP grid, the reality symmetry, and [eq:SMatRep] by
     quadrature at several UHP points.  Same S referenced at the slab CENTRE (S_c = S e^{-iwL},
     a pure size-dependent phase): sigma_max > 1 in the UHP and [eq:SMatRep] fails, which is
     exactly the l.437 caveat.
  B. 2x2 reciprocal passive polarizability alpha(w) = sum R_j/(w_j^2 - w^2 - i g_j w),
     R_j >= 0: Im[w alpha] >= 0 in the UHP, the symmetry, C = D = 0, and the Stieltjes form
     w alpha(w) = int (1/pi) Im[l alpha(l)] dl/(l - w).  An indefinite R_j (gain) breaks it.
  C. h(w) = w eps(w).  Asymptotic coefficients a_n, b_n extracted numerically on the imaginary
     axis; parity (odd real, even imaginary); sum rules p = 0 and p = 1 for an insulator; for a
     conductor a_0 is imaginary (N_0 = 0), so only p = 0 applies and the p = 1 integral diverges.

Run:  python passivity_herglotz.py    (about 1 s; writes figures/sec1_passivity_slab.png/pdf)
"""
import os
import sys
import warnings
import numpy as np
from scipy.integrate import quad, quad_vec, IntegrationWarning

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
SLAB = dict(f=3.0, w0=2.0, g=0.3, L=1.0)


# ------------------------------------------------------------------------ A. slab S matrix
def eps_lorentz(w, f, w0, g):
    return 1 + f / (w0**2 - w**2 - 1j * g * w)


def slab_S(w, f=SLAB["f"], w0=SLAB["w0"], g=SLAB["g"], L=SLAB["L"], reference="faces"):
    """Normal-incidence 2x2 S matrix [[r, t], [t, r]] of a slab in vacuum, ports referenced at
    the slab faces.  r and t are even in n, so they are entire functions of eps (no branch
    choice).  reference='centre' moves both reference planes to the slab centre."""
    w = np.asarray(w, complex)
    n = np.sqrt(eps_lorentz(w, f, w0, g))
    r1 = (1 - n) / (1 + n)
    e = np.exp(2j * n * w * L)
    r = r1 * (1 - e) / (1 - r1**2 * e)
    t = (1 - r1**2) * np.exp(1j * n * w * L) / (1 - r1**2 * e)
    S = np.array([[r, t], [t, r]])
    S = np.moveaxis(S, [0, 1], [-2, -1])
    if reference == "centre":
        S = S * np.exp(-1j * w * L)[..., None, None]
    return S


def sigma_max(S):
    return np.linalg.norm(S, ord=2, axis=(-2, -1))


def poisson(Sfun, w0, g, n=400000, subtract_delay=None):
    """(g/pi) int S(w')/((w'-w0)^2 + g^2) dw' via w' = w0 + g tan(theta), midpoint rule.
    subtract_delay=L removes e^{i w' L} from the transmission entries (its Poisson integral
    is exactly e^{i (w0 + i g) L}) to speed up convergence of the oscillatory tail."""
    th = (np.arange(n) + 0.5) / n * np.pi - np.pi / 2
    wp = w0 + g * np.tan(th)
    S = Sfun(wp)
    if subtract_delay is not None:
        d = np.exp(1j * wp * subtract_delay)
        S = S - d[:, None, None] * np.array([[0, 1], [1, 0]])
    val = S.mean(axis=0)                                  # (1/pi) * (pi/n) * sum
    if subtract_delay is not None:
        val = val + np.exp(1j * (w0 + 1j * g) * subtract_delay) * np.array([[0, 1], [1, 0]])
    return val


def test_A():
    L = SLAB["L"]
    wr = np.linspace(-12, 12, 4801)
    re_axis = sigma_max(slab_S(wr)).max()
    X, Y = np.meshgrid(np.linspace(-6, 6, 241), np.linspace(1e-3, 4, 160))
    Z = X + 1j * Y
    uhp_faces = sigma_max(slab_S(Z)).max()
    uhp_centre = sigma_max(slab_S(Z, reference="centre")).max()
    zs = np.array([0.7 + 0.2j, -1.3 + 0.9j, 2.1 + 0.05j])
    sym = np.abs(slab_S(-zs.conj()) - slab_S(zs).conj()).max()
    rows = []
    for z in [1.2 + 0.3j, 2.0 + 0.8j, -0.4 + 1.5j]:
        exact = slab_S(z)
        num = poisson(lambda x: slab_S(x), z.real, z.imag, subtract_delay=L)
        exact_c = slab_S(z, reference="centre")
        num_c = poisson(lambda x: slab_S(x, reference="centre"), z.real, z.imag)
        rows.append((z, np.abs(num - exact).max(), sigma_max(exact_c), np.abs(num_c - exact_c).max()))
    return dict(re_axis=re_axis, uhp_faces=uhp_faces, uhp_centre=uhp_centre, sym=sym, rows=rows)


# ------------------------------------------------------------------------ B. matrix Herglotz
def random_alpha(seed=0, n_osc=3, active=False):
    """2x2 alpha(w) = sum_j R_j/(w_j^2 - w^2 - i g_j w) with R_j = V V^T >= 0 (passive,
    reciprocal).  active=True adds an inverted (gain) oscillator R = -0.5 I at w = 3.5."""
    rng = np.random.default_rng(seed)
    terms = []
    for j in range(n_osc):
        V = rng.normal(size=(2, 2))
        terms.append((V @ V.T, rng.uniform(0.5, 2.5), rng.uniform(0.1, 0.5)))
    if active:
        terms.append((-0.5 * np.eye(2), 3.5, 0.1))

    def alpha(w):
        w = np.asarray(w, complex)
        out = np.zeros(w.shape + (2, 2), complex)
        for R, w0, g in terms:
            out = out + (1 / (w0**2 - w**2 - 1j * g * w))[..., None, None] * R
        return out
    alpha.terms = terms
    return alpha


def antiherm(A):
    return (A - np.conj(np.swapaxes(A, -1, -2))) / 2j


def test_B(seed=0):
    X, Y = np.meshgrid(np.linspace(-4, 4, 161), np.geomspace(1e-3, 3, 40))
    z = (X + 1j * Y).ravel()
    out = {}
    for active in (False, True):
        a = random_alpha(seed, active=active)
        h = z[:, None, None] * a(z)
        out["min_eig_active" if active else "min_eig"] = np.linalg.eigvalsh(antiherm(h)).min()
    a = random_alpha(seed)
    hz = lambda w: np.asarray(w)[..., None, None] * a(w)
    out["sym"] = np.abs(np.conj(hz(z[:50])) + hz(-np.conj(z[:50]))).max()
    out["C"] = np.abs(np.real(1j * a(1j))).max()
    out["D"] = np.abs(a(1e6j)).max()
    zt = np.array([0.8 + 0.4j, -1.7 + 0.1j, 0.2 + 2.0j])
    errs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        for zz in zt:
            dens = lambda l: (np.imag(l * a(l)) / np.pi) / (l - zz)        # Im elementwise = anti-Herm. part (alpha symmetric)
            val = quad_vec(dens, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-11, limit=2000)[0]
            errs.append(np.abs(val - zz * a(zz)).max())
    out["stieltjes"] = max(errs)
    return out


# ------------------------------------------------------------------------ C. Herglotz sum rules
class Medium:
    """eps(w) = 1 + sum f_j/(w_j^2 - w^2 - i g_j w) - fD/(w^2 + i gD w)."""

    def __init__(self, osc, fD=0.0, gD=1.0):
        self.osc, self.fD, self.gD = osc, fD, gD

    def eps(self, w):
        w = np.asarray(w, complex)
        e = 1 + sum(f / (w0**2 - w**2 - 1j * g * w) for f, w0, g in self.osc)
        return e - (self.fD / (w**2 + 1j * self.gD * w) if self.fD else 0)

    def h(self, w):
        return np.asarray(w, complex) * self.eps(w)

    def x_im_eps(self, x):                                   # x Im eps(x) = Im h(x), finite at 0
        v = sum(f * g * x**2 / ((w0**2 - x**2)**2 + (g * x)**2) for f, w0, g in self.osc)
        return v + (self.fD * self.gD / (x**2 + self.gD**2) if self.fD else 0.0)


def coefficients(m):
    """Asymptotic coefficients of h(z), read off numerically along the imaginary axis z = iy by
    least-squares Laurent fits (no model knowledge used):
      z -> 0  :  z h(z)     = a_-1 + a_0 z + a_1 z^2 + a_2 z^3 + ...
      z -> inf:  h(z)/z     = b_1 + b_0 s + b_-1 s^2 + b_-2 s^3 + ...,   s = 1/z."""
    y = np.geomspace(2e-4, 2e-3, 12)
    z = 1j * y
    V = np.vander(z / z[0], 6, increasing=True)
    c = np.linalg.lstsq(V, z * m.h(z), rcond=None)[0] / z[0]**np.arange(6)
    out = dict(a_m1=c[0], a0=c[1], a1=c[2], a2=c[3])
    y = np.geomspace(60, 600, 14)
    s = 1 / (1j * y)
    V = np.vander(s / s[0], 9, increasing=True)
    c = np.linalg.lstsq(V, m.h(1j * y) / (1j * y), rcond=None)[0] / s[0]**np.arange(9)
    out.update(b1=c[0], b0=c[1], bm1=c[2], bm2=c[3])
    return out


def sum_rule(m, p, lower=0.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        pts = [w0 for _, w0, _ in m.osc]
        f = (lambda x: m.x_im_eps(x) / x**(2 * p))
        lo = max(lower, 0.0)
        mid = max(pts) + 5
        v = quad(f, lo, mid, points=[q for q in pts if q > lo], limit=800, epsabs=1e-14, epsrel=1e-12)[0]
        v += quad(f, mid, np.inf, limit=800, epsabs=1e-14, epsrel=1e-12)[0]
    return 2 / np.pi * v


def test_C():
    ins = Medium([(1.3, 1.0, 0.2), (0.7, 2.6, 0.5), (2.0, 4.0, 0.9)])
    met = Medium([(0.8, 2.0, 0.4)], fD=4.0, gD=0.05)
    ci, cm = coefficients(ins), coefficients(met)
    res = dict(ins=ci, met=cm)
    res["ins_p0"] = (sum_rule(ins, 0), (ci["a_m1"] - ci["bm1"]).real, sum(f for f, _, _ in ins.osc))
    eps0 = 1 + sum(f / w0**2 for f, w0, _ in ins.osc)
    res["ins_p1"] = (sum_rule(ins, 1), (ci["a1"] - ci["b1"]).real, eps0 - 1)
    res["met_p0"] = (sum_rule(met, 0), (cm["a_m1"] - cm["bm1"]).real, 0.8 + 4.0)
    res["met_p1_growth"] = [(d, sum_rule(met, 1, lower=d)) for d in (1e-2, 1e-3, 1e-4)]
    z = np.array([0.3 + 0.2j, -2 + 1j, 5 + 0.1j])
    res["parity"] = max(np.abs(ins.h(z) + np.conj(ins.h(-np.conj(z)))).max(),
                        np.abs(met.h(z) + np.conj(met.h(-np.conj(z)))).max())
    return res


# ------------------------------------------------------------------------ output
def make_figure(path_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = np.linspace(0, 2.5, 300)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.1))
    for w0, ls in [(1.2, "-"), (2.5, "--")]:
        z = w0 + 1j * g
        ax[0].plot(g, sigma_max(slab_S(z)), "C0" + ls, label=rf"faces, Re$\,\omega$={w0}")
        ax[0].plot(g, sigma_max(slab_S(z, reference="centre")), "C3" + ls, label=rf"centre, Re$\,\omega$={w0}")
    ax[0].axhline(1, color="k", lw=0.6)
    ax[0].set_xlabel(r"Im $\omega$ ($c/L$)")
    ax[0].set_ylabel(r"$\sigma_{\max}[\mathbb{S}(\omega)]$")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=7, frameon=False)
    wr = np.linspace(0, 8, 800)
    S = slab_S(wr)
    ax[1].plot(wr, np.abs(S[:, 0, 0])**2, label=r"$|r|^2$")
    ax[1].plot(wr, np.abs(S[:, 0, 1])**2, label=r"$|t|^2$")
    ax[1].plot(wr, 1 - np.abs(S[:, 0, 0])**2 - np.abs(S[:, 0, 1])**2, label="absorption")
    ax[1].set_xlabel(r"$\omega$ ($c/L$)")
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(path_stem + "." + ext, dpi=200)
    plt.close(fig)


def main(make_fig=True):
    print("=" * 84)
    print("Sec. 1.5  Passivity: bounded-real S, Poisson representation, Herglotz functions")
    print("=" * 84)
    A = test_A()
    print(f"A. Lorentz slab (f={SLAB['f']}, w0={SLAB['w0']}, g={SLAB['g']}, L={SLAB['L']}), ports at the faces")
    print(f"   max sigma_max(S): real axis {A['re_axis']:.6f};  UHP grid {A['uhp_faces']:.6f}  (<= 1: bounded-real)")
    print(f"   reality symmetry  max|S(-w*) - S(w)*| = {A['sym']:.1e}")
    print(f"   same slab, ports referenced at the slab centre (S e^(-iwL)): UHP max sigma_max = {A['uhp_centre']:.3g}")
    print(f"   {'w0 + i g':>14} | {'[eq:SMatRep] err, faces':>24} | {'sigma_max(S_c)':>14} {'[eq:SMatRep] err, centre':>25}")
    for z, e, sc, ec in A["rows"]:
        print(f"   {z.real:6.2f}{z.imag:+6.2f}i | {e:24.1e} | {sc:14.4f} {ec:25.3f}")

    B = test_B()
    print("\nB. 2x2 reciprocal polarizability alpha = sum R_j/(w_j^2 - w^2 - i g_j w)")
    print(f"   min eig Im[w alpha] over a 161x40 UHP grid: passive (R_j >= 0) {B['min_eig']:.3e};"
          f" plus one gain oscillator {B['min_eig_active']:.3e}")
    print(f"   symmetry max|[w a]* + [w a]_(-w*)| = {B['sym']:.1e};  C = Re[i a(i)]: {B['C']:.1e};"
          f"  |a(i 1e6)| = {B['D']:.1e} (D = 0)")
    print(f"   Stieltjes form w a(w) = int (1/pi) Im[l a(l)] dl/(l - w): max error {B['stieltjes']:.1e}")

    C = test_C()
    ci, cm = C["ins"], C["met"]
    print("\nC. h(w) = w eps(w).  Insulator (3 Lorentz oscillators):")
    print(f"   a_-1 = {ci['a_m1']:.2e}, a_0 = {ci['a0']:.2e}, a_1 = {ci['a1']:.6f}, a_2 = {ci['a2']:.4f}"
          f"  (odd real, even imaginary)")
    print(f"   b_1 = {ci['b1']:.6f}, b_-1 = {ci['bm1']:.6f}, b_-2 = {ci['bm2']:.4f}   => N_0 = N_inf = 1, p in {{0, 1}}")
    v, c, t = C["ins_p0"]
    print(f"   p=0: (2/pi) int Im h dx = {v:.8f};  a_-1 - b_-1 = {c:.8f};  sum f_j = w_p^2 = {t:.8f}")
    v, c, t = C["ins_p1"]
    print(f"   p=1: (2/pi) int Im h/x^2 dx = {v:.8f};  a_1 - b_1 = {c:.8f};  eps(0) - 1 = {t:.8f}"
          f"  (<= a_1 = {ci['a1'].real:.6f}, eq:h_sum_rule_p1)")
    print("   Conductor (Drude fD=4, gD=0.05 + Lorentz):")
    print(f"   a_-1 = {cm['a_m1']:.2e}, a_0 = {cm['a0']:.4f}  (= i fD/gD = {4/0.05:.0f}i, imaginary => N_0 = 0, only p = 0)")
    v, c, t = C["met_p0"]
    print(f"   p=0: (2/pi) int Im h dx = {v:.8f};  a_-1 - b_-1 = {c:.8f};  w_p^2 = {t:.8f}")
    print("   p=1 integral from x = d: " + ", ".join(f"d={d:.0e}: {v:.4g}" for d, v in C["met_p1_growth"])
          + "  (diverges ~ 1/d, as the p-range restriction requires)")
    print(f"   parity h(z) = -h*(-z*): max error {C['parity']:.1e}")

    if make_fig:
        stem = os.path.join(FIGDIR, "sec1_passivity_slab")
        make_figure(stem)
        print(f"\nwrote {os.path.normpath(stem)}.pdf/png")


if __name__ == "__main__":
    main()
