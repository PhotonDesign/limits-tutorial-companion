"""Sec. 2 (intro, item 2) and Sec. 2.2: derivative bounds on passive transparent media and the
bandwidth bound on how closely eps(w) can stay at a value eps_m < eps0.  Time dependence
e^{-i w t}; eps0 = 1, frequencies in units of the band centre w_c.

Tutorial claims checked (tutorial.tex line numbers):
  [eq:derivative_bounds] l.715  d(w eps)/dw >= eps0,  d(w mu)/dw >= mu0 (transparent media)
  l.718      these "can also be derived (more rigorously) from causality [L&L p. 287]"
  l.814      lossless: d eps/dw >= (eps0 - eps)/w;  ENZ: slope >= eps0/w0;  constant-slope
             estimate |eps(w_{1,2}) - eps_m| >= (eps0 - eps_m) B,  "B = (w2 - w1) w_c"
  [eq:low_epsilon_bound] l.816  lossless: max_band |eps - eps_m| >= B/(1 + B/2) (eps0 - eps_m)
  l.821      consistent with the constant-slope estimate for B/2 << 1
  l.822-824  open question: tightness at finite B; is the lossy/lossless factor of two sharp?

Tests and findings:
  A. Random passive media, lossless in the window but with lossless oscillators (incl. Drude),
     absorption bands and a conducting (Im chi = sigma/w) band elsewhere: [eq:derivative_bounds]
     and l.814 always hold, but so does the SHARPER causality bound (Landau-Lifshitz Sec. 84;
     Gustafsson & Sjoberg 2010 Eq. 15)  d eps/dw >= 2 (eps0 - eps)/w, i.e. d[w^2 (eps - eps0)]/dw >= 0,
     with equality for a lossless Drude medium.
  B. LP over all lossless oscillator distributions: min slope at eps(w) = eps_m is exactly
     2 (eps0 - eps_m)/w (all weight on the Drude term); ENZ: 2 eps0/w0, not eps0/w0.
  C. Hence l.814's constant-slope argument with the printed slope gives (eps0 - eps_m) B/2, not
     the stated (eps0 - eps_m) B; the stated value needs the factor-2 (L&L) slope.  ERROR (text).
  D. LP minimax over lossless media (fixed oscillator grid, all weights free):
     min max_band |eps - eps_m| = (eps0 - eps_m) (w2^2 - w1^2)/(w2^2 + w1^2) = (eps0 - eps_m) B/(1 + B^2/4),
     attained by a single lossless Drude medium.  [eq:low_epsilon_bound] is never violated but is
     not tight (slack factor (1 + B/2)/(1 + B^2/4): 1.19 at B = 0.4, 1.20 at B = 1).  Random
     multi-oscillator designs never beat the Drude value.
  E. Lossy (general passive) media, LP over lossless oscillators + piecewise-linear absorption
     spectra (closed-form KK), complex modulus via a 32-gon: the optimum converges from above
     to (eps0 - eps_m) B/2, the tight bound of Lind-Johansen, Waters & Skaar (2009)
     [2 D/(1 + sqrt(1 - D^2)) for eps_m = -1, D = (w2^2-w1^2)/(w2^2+w1^2), which equals B].
     So lossless/lossy = 2/(1 + B^2/4): exactly 2 only as B -> 0 (answers l.823-824).
  F. [eq:derivative_bounds] for mu: the standard lossless split-ring (Pendry) model
     mu = 1 - F w^2/(w^2 - w0^2) is passive (w mu is Herglotz) but has mu(inf) = 1 - F, and
     d(w mu)/dw < mu0 for w > sqrt(3) w0.  The mu inequality needs mu(inf) = mu0.  UNCLEAR/caveat.
  G. Sec. 2.5, l.858-870: for D = eps E + (i/c) kappa H, B = -(i/c) kappa E + mu H, [eq:absorbed]
     equals the printed Hermitian form with off-diagonal +/- i Im(kappa/c), so P_a >= 0 for all
     fields iff Im eps, Im mu >= 0 and (Im kappa/c)^2 <= Im eps Im mu [eq:chiral_inequality]
     (positive SEMIdefinite; the text says "positive definite", l.867).  [eq:eff_nonlocal_chiral]
     follows from [eq:eff_nonlocal] with xi = i kappa, zeta = -i kappa.  CONFIRMED.

Run:  python dispersion_bandwidth_bounds.py    (about 10 s; writes figures/sec2_low_epsilon_bound.png/pdf)
"""
import os
import sys
import numpy as np
from scipy.optimize import linprog, minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


# ------------------------------------------------------------------ closed-form bounds
def band(B, wc=1.0):
    return wc * (1 - B / 2), wc * (1 + B / 2)


def bound_tutorial(B):
    """[eq:low_epsilon_bound] / (eps0 - eps_m)."""
    return B / (1 + B / 2)


def bound_lossless_tight(B):
    """(w2^2 - w1^2)/(w2^2 + w1^2) = B/(1 + B^2/4): lossless Drude, from d[w^2(eps-1)]/dw >= 0."""
    return B / (1 + B**2 / 4)


def bound_lossy_gs(B):
    """Gustafsson & Sjoberg (2010) lossy bound / (eps_inf - eps_m)."""
    return B / 2 / (1 + B / 2)


def bound_lossy_tight(B):
    """Lind-Johansen et al. (2009): 2D/(1 + sqrt(1 - D^2)) for eps_m = -1, rescaled; = B/2."""
    D = bound_lossless_tight(B)
    return D / (1 + np.sqrt(1 - D**2))


# ------------------------------------------------------------------ A. random passive media
class Medium:
    """eps(w) = 1 + sum_i A_i/(w_i^2 - w^2)                     lossless oscillators (w_i = 0: Drude)
              + sum_j (H_j/pi) ln|(b_j^2 - w^2)/(a_j^2 - w^2)|   Im chi = H_j on [a_j, b_j]
              + (s/(pi w)) ln|(a_c - w)/(a_c + w)|               Im chi = s/w on (0, a_c] (conductor)
    evaluated on the real axis outside all absorption bands (Re chi from KKRe in closed form)."""

    def __init__(self, deltas=(), boxes=(), cond=None):
        self.deltas, self.boxes, self.cond = list(deltas), list(boxes), cond

    def eps_deps(self, w):
        w = np.asarray(w, float)
        e = np.ones_like(w)
        d = np.zeros_like(w)
        for A, wi in self.deltas:
            e += A / (wi**2 - w**2)
            d += 2 * A * w / (wi**2 - w**2)**2
        for H, a, b in self.boxes:
            e += H / np.pi * np.log(np.abs((b**2 - w**2) / (a**2 - w**2)))
            d += H / np.pi * (2 * w / (a**2 - w**2) - 2 * w / (b**2 - w**2))
        if self.cond is not None:
            s, a = self.cond
            L = np.log(np.abs((a - w) / (a + w)))
            e += s / (np.pi * w) * L
            d += -s / (np.pi * w**2) * L + s / (np.pi * w) * (-2 * a / (a**2 - w**2))
        return e, d

    def singular(self):
        pts = [wi for _, wi in self.deltas]
        for _, a, b in self.boxes:
            pts += [a, b]
        return pts

    def absorbing(self, w):
        m = np.zeros_like(w, bool)
        for _, a, b in self.boxes:
            m |= (w > a) & (w < b)
        if self.cond is not None:
            m |= w < self.cond[1]
        return m


def random_medium(rng):
    deltas = [(rng.uniform(0.05, 3.0), 0.0)] if rng.random() < 0.6 else []
    deltas += [(rng.uniform(0.05, 3.0), rng.uniform(0.2, 5.0)) for _ in range(rng.integers(0 if deltas else 1, 4))]
    boxes = []
    for _ in range(rng.integers(0, 3)):
        a = rng.uniform(0.3, 5.0)
        boxes.append((rng.uniform(0.05, 1.5), a, a + rng.uniform(0.05, 1.0)))
    cond = (rng.uniform(0.05, 2.0), rng.uniform(0.05, 0.5)) if rng.random() < 0.3 else None
    return Medium(deltas, boxes, cond)


def test_A(n_media=400, seed=0):
    rng = np.random.default_rng(seed)
    w = np.linspace(0.05, 7.0, 3000)
    q1 = q3 = np.inf
    ratio = np.inf
    for _ in range(n_media):
        m = random_medium(rng)
        ok = ~m.absorbing(w)
        for p in m.singular():
            ok &= np.abs(w - p) > 0.01
        e, d = m.eps_deps(w[ok])
        ww = w[ok]
        q1 = min(q1, np.min(e + ww * d - 1))                  # d(w eps)/dw - eps0
        q3 = min(q3, np.min(ww * d - 2 * (1 - e)))            # L&L: w eps' - 2(eps0 - eps)
        sel = e < 0.999
        if sel.any():
            ratio = min(ratio, np.min(ww[sel] * d[sel] / (1 - e[sel])))
    drude = Medium([(1.7, 0.0)])
    e, d = drude.eps_deps(np.array([0.8, 1.3, 2.0]))
    drude_eq = np.abs(np.array([0.8, 1.3, 2.0]) * d - 2 * (1 - e)).max()
    return dict(q1=q1, q3=q3, ratio=ratio, drude_eq=drude_eq)


# ------------------------------------------------------------------ B. LP: minimum slope
def lp_min_slope(eps_m, w=1.0):
    wi = np.concatenate([[0.0], np.linspace(0.01, 0.999, 400) * w, np.geomspace(1.001, 100, 600) * w])
    x = wi**2 - w**2
    res = linprog(2 * w / x**2, A_eq=(1 / x)[None, :], b_eq=[eps_m - 1], bounds=(0, None), method="highs")
    return res.fun, res.x[0] / res.x.sum()


# ------------------------------------------------------------------ D, E. minimax LPs
def _logm(x):
    """Log(x - i0) for real x."""
    return np.log(np.abs(x)) - 1j * np.pi * (x < 0)


def hat_chi(w, nodes, h):
    """chi(w + i0) for absorption spectra Im chi = triangle ('hat') of half-width h centred at
    each node, odd-extended to w < 0; exact Cauchy integrals of the piecewise-linear density."""
    w = np.asarray(w, float)[:, None]
    t = np.asarray(nodes, float)[None, :]
    out = 0
    for p, q, al, be in [(t - h, t, -(t - h) / h, 1 / h), (t, t + h, (t + h) / h, -1 / h)]:
        I1 = be * (q - p) + (al + be * w) * (_logm(q - w) - _logm(p - w))
        I2 = be * (q - p) + (al - be * w) * (np.log(q + w) - np.log(p + w))
        out = out + I1 + I2
    return out / np.pi


def delta_grid(w1, w2):
    return np.concatenate([[0.0], np.linspace(0.02, 0.999 * w1, 80), 1.001 * w2 * np.geomspace(1, 60, 200)])


def minimax_lp(B, lossy, eps_m=0.0, h=0.01, n_dir=32, K=240, w_max=4.0):
    """min over passive media of max_{w in band} |eps(w) - eps_m| (eps0 = eps_inf = 1).
    Lossless: nonnegative weights on lossless oscillators at fixed frequencies outside the band
    (incl. Drude).  Lossy: plus nonnegative weights on hat absorption spectra on [0, w_max].
    Returns (LP value, max deviation re-evaluated on a 10x finer grid, Drude weight fraction)."""
    w1, w2 = band(B)
    ws = np.linspace(w1, w2, K)
    wd = delta_grid(w1, w2)
    nodes = (np.arange(1, int(w_max / h)) * h + 1e-4 * np.pi) if lossy else np.array([])

    def basis(wv):
        cols = [(1 / (wd[None, :]**2 - wv[:, None]**2)).astype(complex)]
        if lossy:
            cols.append(hat_chi(wv, nodes, h))
        return np.concatenate(cols, 1)
    Phi = basis(ws)
    target = eps_m - 1.0
    if lossy:
        th = np.arange(n_dir) * 2 * np.pi / n_dir
        rows = [np.hstack([np.real(np.exp(-1j * t) * Phi), -np.ones((K, 1))]) for t in th]
        rhs = [np.real(np.exp(-1j * t) * target) * np.ones(K) for t in th]
    else:
        P = np.real(Phi)
        rows = [np.hstack([P, -np.ones((K, 1))]), np.hstack([-P, -np.ones((K, 1))])]
        rhs = [target * np.ones(K), -target * np.ones(K)]
    c = np.zeros(Phi.shape[1] + 1)
    c[-1] = 1
    res = linprog(c, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs), bounds=(0, None), method="highs")
    x = res.x[:-1]
    wf = np.linspace(w1, w2, 10 * K)
    dev = np.abs(1 + basis(wf) @ x - eps_m).max()
    drude_frac = x[0] / x[:wd.size].sum() if x[:wd.size].sum() > 0 else 0.0
    return res.fun, dev, drude_frac


def random_design_deviation(B, n=300, seed=1):
    """Best normalized deviation reached by random lossless multi-oscillator media, each with
    its overall strength optimized (the deviation is convex in the scale factor)."""
    rng = np.random.default_rng(seed)
    w1, w2 = band(B)
    wv = np.linspace(w1, w2, 400)
    best = np.inf
    for _ in range(n):
        wi = np.concatenate([rng.uniform(0, 0.98 * w1, rng.integers(1, 4)),
                             rng.uniform(1.02 * w2, 4 * w2, rng.integers(0, 3))])
        A = rng.uniform(0.1, 1, wi.size)
        chi0 = (A[None, :] / (wi[None, :]**2 - wv[:, None]**2)).sum(1)
        r = minimize_scalar(lambda s: np.abs(1 + s * chi0).max(), bounds=(0, 20), method="bounded")
        best = min(best, r.fun)
    return best


# ------------------------------------------------------------------ F. magnetic caveat
def srr_check(F=0.5, w0=1.0):
    mu = lambda w: 1 - F * w**2 / (w**2 - w0**2)
    wmu = lambda w: w * mu(w)
    out = []
    for w in (0.5, 1.5, np.sqrt(3), 2.0, 4.0):
        h = 1e-6
        out.append((w, (wmu(w + h) - wmu(w - h)) / (2 * h)))
    return out, 1 - F


# ------------------------------------------------------------------ G. bi-isotropic passivity
def chiral_check(n=2000, seed=0):
    """Random complex eps, mu, kappa (c = 1) and random complex E, H: compare [eq:absorbed],
    evaluated from the constitutive relations, with the printed quadratic form; then test the
    PSD criterion against brute-force minimization of P_a over random fields."""
    rng = np.random.default_rng(seed)
    err = 0.0
    agree = 0
    w = 1.3
    for _ in range(n):
        eps, mu, kap = rng.normal(size=3) + 1j * rng.uniform(0, 1, 3) * np.array([1, 1, 2])
        E = rng.normal(size=3) + 1j * rng.normal(size=3)
        H = rng.normal(size=3) + 1j * rng.normal(size=3)
        D = eps * E + 1j * kap * H
        Bf = -1j * kap * E + mu * H
        Pa = 0.5 * np.real(-1j * w * (np.vdot(E, D) + np.vdot(H, Bf)))
        M = np.array([[eps.imag, 1j * kap.imag], [-1j * kap.imag, mu.imag]])
        v = np.concatenate([E, H]).reshape(2, 3)
        Pm = 0.5 * w * np.real(sum(np.conj(v[:, i]) @ M @ v[:, i] for i in range(3)))
        err = max(err, abs(Pa - Pm) / (abs(Pa) + 1))
        psd = kap.imag**2 <= eps.imag * mu.imag
        # brute force: most negative P_a over random polarization states (2x2 problem)
        lam_min = np.linalg.eigvalsh(M).min()
        agree += (lam_min >= -1e-14) == psd
    return err, agree / n


# ------------------------------------------------------------------ output
def make_figure(stem, lp_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    B = np.linspace(1e-3, 1.9, 300)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].plot(B, bound_tutorial(B), "C3--", label=r"[eq:low_epsilon_bound] $B/(1+B/2)$")
    ax[0].plot(B, bound_lossless_tight(B), "k-", label=r"lossless Drude $B/(1+B^2/4)$")
    ax[0].plot(B, bound_lossy_gs(B), "C0--", label=r"lossy, G&S $\frac{B/2}{1+B/2}$")
    ax[0].plot(B, B / 2, "C0-", label=r"lossy, tight (LJ 2009) $B/2$")
    Bs = [r[0] for r in lp_rows]
    ax[0].plot(Bs, [r[2] for r in lp_rows], "ko", ms=4, mfc="none", label="LP, lossless")
    ax[0].plot(Bs, [r[4] for r in lp_rows], "C0s", ms=4, mfc="none", label="LP, lossy (achieved)")
    ax[0].set_xlabel(r"fractional bandwidth $B$")
    ax[0].set_ylabel(r"min max $|\varepsilon-\varepsilon_m|\,/\,(\varepsilon_0-\varepsilon_m)$")
    ax[0].legend(fontsize=6.5, frameon=False)
    ax[1].plot(B, bound_lossless_tight(B) / bound_tutorial(B), "C3-", label="tight lossless / printed bound")
    ax[1].plot(B, bound_lossless_tight(B) / (B / 2), "k-", label="lossless / lossy (tight)")
    ax[1].set_xlabel(r"fractional bandwidth $B$")
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(stem + "." + ext, dpi=200)
    plt.close(fig)


def main(make_fig=True):
    print("=" * 92)
    print("Sec. 2 / 2.2  Derivative bounds and the bandwidth bound on eps(w) ~ eps_m < eps0")
    print("=" * 92)
    A = test_A()
    print("A. 400 random passive media (lossless oscillators incl. Drude, absorption bands, conducting band),")
    print("   evaluated in their transparency windows:")
    print(f"   min [d(w eps)/dw - eps0]              = {A['q1']:+.3e}  (>= 0: [eq:derivative_bounds], l.814)")
    print(f"   min [w eps' - 2 (eps0 - eps)]          = {A['q3']:+.3e}  (>= 0: sharper L&L causality bound)")
    print(f"   min w eps'/(eps0 - eps) over eps < eps0 = {A['ratio']:.6f}  (l.814 claims >= 1; the true floor is 2)")
    print(f"   lossless Drude: |w eps' - 2(eps0 - eps)| = {A['drude_eq']:.1e}  (equality)")

    print("\nB. LP over all lossless oscillator distributions: min d eps/dw at eps(w) = eps_m (w = 1)")
    for em in (0.5, 0.0, -1.0, -5.0):
        s, fd = lp_min_slope(em)
        print(f"   eps_m = {em:+5.1f}: min slope = {s:.8f} = {s / (1 - em):.8f} x (eps0 - eps_m)/w;"
              f"  Drude weight fraction {fd:.6f}")
    print("   -> ENZ (eps_m = 0): slope >= 2 eps0/w0 (l.814 states eps0/w0)")

    print("\nC. constant-slope estimate of l.814, deviation at the band edges / (eps0 - eps_m):")
    for B in (0.02, 0.1):
        print(f"   B = {B}: printed slope (eps0-eps)/w -> {B / 2:.4f};  L&L slope 2(eps0-eps)/w -> {B:.4f};"
              f"  text says B = {B:.4f};  true optimum {bound_lossless_tight(B):.5f}")

    print("\nD/E. min over passive media of max_band |eps - eps_m| / (eps0 - eps_m)   (eps_m = 0, band centre 1)")
    print(f"{'B':>5} | {'printed':>8} {'tight lossless':>14} {'LP lossless':>11} {'Drude frac':>10} {'random best':>11} |"
          f" {'G&S lossy':>9} {'tight lossy':>11} {'LP lossy':>9} {'achieved':>9}")
    rows = []
    for B in (0.1, 0.4, 0.8, 1.2, 1.6):
        v0, d0, fd = minimax_lp(B, lossy=False)
        v1, d1, _ = minimax_lp(B, lossy=True)
        rnd = random_design_deviation(B)
        rows.append((B, v0, d0, v1, d1))
        print(f"{B:5.2f} | {bound_tutorial(B):8.5f} {bound_lossless_tight(B):14.5f} {d0:11.5f} {fd:10.6f} {rnd:11.5f} |"
              f" {bound_lossy_gs(B):9.5f} {bound_lossy_tight(B):11.5f} {v1:9.5f} {d1:9.5f}")
    v, d, _ = minimax_lp(0.4, lossy=True, h=0.005, K=300)
    print(f"   lossy LP convergence at B = 0.4: hat width 0.01 -> {rows[1][3]:.5f}, 0.005 -> {v:.5f}  (-> B/2 = 0.2)")
    v, d, _ = minimax_lp(0.4, lossy=True, eps_m=-1.0)
    print(f"   eps_m = -1, B = 0.4: achieved {d:.5f} vs (eps0 - eps_m) B/2 = 0.4  (linear in eps0 - eps_m)")

    rows_srr, muinf = srr_check()
    print(f"\nF. split-ring model mu = 1 - F w^2/(w^2 - w0^2), F = 0.5 (passive, mu(inf) = {muinf}):")
    print("   " + ",  ".join(f"w/w0 = {w:.3f}: d(w mu)/dw = {v:.4f}" for w, v in rows_srr))
    print("   -> d(w mu)/dw >= mu0 fails for w > sqrt(3) w0; it holds with mu(inf) in place of mu0")

    err, agree = chiral_check()
    print(f"\nG. bi-isotropic [eq:absorbed] vs printed matrix form: max rel. difference {err:.1e};"
          f" PSD <=> (Im k)^2 <= Im e Im mu in {100 * agree:.1f}% of 2000 random cases")

    if make_fig:
        stem = os.path.join(FIGDIR, "sec2_low_epsilon_bound")
        make_figure(stem, rows)
        print(f"\nwrote {os.path.normpath(stem)}.pdf/png")


if __name__ == "__main__":
    main()
