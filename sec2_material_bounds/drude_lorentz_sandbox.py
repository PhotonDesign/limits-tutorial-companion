"""Secs. 2.1-2.2: interactive Drude-Lorentz sandbox.  Drag the resonance frequency, plasma
frequency (oscillator strength), damping, Drude share and bandwidth of a one-oscillator medium
and watch it respect the tutorial's causality/sum-rule bounds.  Time dependence e^{-i w t};
eps0 = 1; all frequencies in units of the probe/band-centre frequency w_c = 1.

Model (the f-sum budget w_p^2 is split between a Drude term and one Lorentz oscillator, i.e.
[eq:chi_causal_repres] with two oscillators and c_D + c_L = 1 [eq:normalized_f_sum]):
    eps(w) = 1 - c_D w_p^2/(w^2 + i g w) + c_L w_p^2/(w0^2 - w^2 - i g w),   c_L = 1 - c_D.

Tutorial results overlaid (equations used exactly as printed; tutorial LaTeX labels in []):
  [eq:f-sum rule]  int_0^inf w Im chi(w) dw = pi w_p^2/2.  Checked by quadrature of the total
                   w Im eps(w) for the current parameters (g > 0; for g = 0, Im eps is a sum of
                   delta functions whose weights exhaust w_p^2 by construction).
  [eq:max_n]       (n^2 - 1)^2/n <= w_p^2 n'/w  (Sec. 2.1, transparent media), evaluated at w = w_c
                   with n = Re sqrt(eps), n' = dn/dw.  Shown in the (n', n) plane.
  [eq:low_epsilon_bound]  lossless media, eps_m < eps0, band [w1, w2] with B = (w2 - w1)/w_c:
                   max_band |eps - eps_m| >= B/(1 + B/2) (eps0 - eps_m)   (Sec. 2.2).
                   The model's figure of merit is R = inf_{eps_m < eps0} max_band |eps - eps_m|/(eps0 - eps_m)
                   (the target value the model tracks best); the bound says R >= B/(1 + B/2).

Findings (printed by the snapshot mode):
  A. f-sum rule reproduced to < 1e-5 for all parameters (quadrature on a tan-mapped grid).
  B. A single lossless oscillator below resonance (c_D = 0, w0 > w_c, g = 0) SATURATES [eq:max_n]
     identically: (n^2-1)^2/n = w_p^2 n'/w for every w0 and w_p, i.e. dragging w0 slides the model
     along the bound (more index costs more dispersion).  Any Drude share c_D > 0 drops it below.
     With damping g > 0 the medium is not transparent at w_c and [eq:max_n] can be exceeded by a
     relative amount of order the loss tangent (worst 1.001 for Im eps/(Re eps - 1) < 0.01), so
     the sandbox greys out the point when g > 0.
  C. [eq:low_epsilon_bound] is respected over the full lossless grid; the closest approach is the
     lossless Drude limit, R = B/(1 + B^2/4) (slack 1.19 at B = 0.4; cf. the open question after
     [eq:low_epsilon_bound] and dispersion_bandwidth_bounds.py, test D).  Damped single-oscillator
     models also stay above it on the grid tested, although the tutorial states it for lossless media.

Run:
  python drude_lorentz_sandbox.py              interactive sliders (needs a GUI matplotlib backend)
  python drude_lorentz_sandbox.py --snapshot   print the checks and save two figures
The snapshot writes figures/sec2_drude_lorentz_sandbox_{index,loweps}.png/pdf (resonance above /
below the band).  The snapshot mode is also used automatically under a non-interactive backend (e.g. MPLBACKEND=Agg,
as in run_all.py).  About 5 s.
"""
import os
import sys
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
WC = 1.0                          # probe frequency / band centre (unit of frequency)


@dataclass(frozen=True)
class Params:
    w0: float = 1.5               # Lorentz resonance frequency
    wp: float = 1.0               # plasma frequency: total f-sum budget w_p^2
    g: float = 0.02               # damping (same for both terms)
    fD: float = 0.0               # Drude share c_D of the oscillator strength


DEFAULT = Params()
DEFAULT_B = 0.4
# states saved by the snapshot mode: resonance above the band (Sec. 2.1 regime: high index,
# lossless so that [eq:max_n] applies) and below it (Sec. 2.2 regime: Re eps < eps0 on the band)
SNAPSHOT_STATES = {"index": (Params(1.5, 1.0, 0.0, 0.0), 0.4),
                   "loweps": (Params(0.4, 1.2, 0.02, 0.0), 0.4)}


# ------------------------------------------------------------------ model
def eps(w, p):
    """Complex permittivity; Im eps >= 0 for w > 0 (passive)."""
    w = np.asarray(w, float) + 0j
    cD, cL = p.fD, 1 - p.fD
    with np.errstate(divide="ignore", invalid="ignore"):
        drude = -cD * p.wp**2 / (w**2 + 1j * p.g * w) if cD > 0 else 0
        lor = cL * p.wp**2 / (p.w0**2 - w**2 - 1j * p.g * w) if cL > 0 else 0
    return 1 + drude + lor


def deps(w, p):
    """d eps/dw (analytic)."""
    w = np.asarray(w, float) + 0j
    cD, cL = p.fD, 1 - p.fD
    with np.errstate(divide="ignore", invalid="ignore"):
        drude = cD * p.wp**2 * (2 * w + 1j * p.g) / (w**2 + 1j * p.g * w)**2 if cD > 0 else 0
        lor = cL * p.wp**2 * (2 * w + 1j * p.g) / (p.w0**2 - w**2 - 1j * p.g * w)**2 if cL > 0 else 0
    return drude + lor


# ------------------------------------------------------------------ [eq:f-sum rule]
def fsum_integral(p, w_max=1e6, n_map=4000):
    """int_0^inf w Im eps(w) dw by trapezoidal quadrature of the TOTAL model on the union of
    tan-mapped grids (w = w_i + g tan(phi), which resolves each Lorentzian line uniformly in phi)
    and a log grid, truncated at w_max (tail ~ w_p^2 g/w_max, i.e. < 1e-6 relative here).
    Returns None for g = 0 (Im eps is then a sum of delta functions)."""
    if p.g <= 0:
        return None
    parts = [np.geomspace(1e-9, w_max, 4000)]
    centres = ([0.0] if p.fD > 0 else []) + ([p.w0] if p.fD < 1 else [])
    for wi in centres:
        phi = np.linspace(np.arctan(-wi / p.g), np.pi / 2, n_map)[:-1]
        parts.append(wi + p.g * np.tan(phi))
    w = np.unique(np.concatenate(parts))
    w = w[(w >= 1e-9) & (w <= w_max)]
    return np.trapezoid(w * eps(w, p).imag, w)


def fsum_ratio(p):
    """Computed integral / (pi w_p^2 / 2); None if g = 0."""
    I = fsum_integral(p)
    return None if I is None else I / (np.pi * p.wp**2 / 2)


# ------------------------------------------------------------------ [eq:max_n]
def max_n_sides(p, w=WC):
    """(LHS, RHS, n, n') of [eq:max_n], (n^2-1)^2/n <= w_p^2 n'/w, at frequency w, using
    n = Re sqrt(eps) and n' = Re dn/dw.  Returns None unless n > 1 and n' > 0 (the bound is a
    statement about the largest index of a normally dispersive transparent medium)."""
    e = eps(w, p)
    nc = np.sqrt(e)
    with np.errstate(divide="ignore", invalid="ignore"):
        npr = (deps(w, p) / (2 * nc)).real
    n = nc.real
    if not (np.isfinite(n) and n > 1 and npr > 0):
        return None
    return (n**2 - 1)**2 / n, p.wp**2 * npr / w, n, npr


def max_n_curve(wp, w=WC, n_max=6.0):
    """Saturated [eq:max_n]: n' = w (n^2 - 1)^2/(n w_p^2), parametrized by n."""
    n = np.linspace(1 + 1e-4, n_max, 400)
    return w * (n**2 - 1)**2 / (n * wp**2), n


# ------------------------------------------------------------------ [eq:low_epsilon_bound]
def bound_low_eps(B):
    """[eq:low_epsilon_bound] / (eps0 - eps_m), as printed: B/(1 + B/2)."""
    return B / (1 + B / 2)


def band(B, wc=WC):
    return wc * (1 - B / 2), wc * (1 + B / 2)


def low_eps_merit(e):
    """R = inf over eps_m < 1 of max_k |e_k - eps_m|/(1 - eps_m) for samples e_k of eps on the band,
    and the minimizing eps_m (None if the infimum is the eps_m -> -inf limit, R = 1).
    The ratio is quasiconvex in eps_m (convex numerator, positive affine denominator), so a coarse
    scan in s = ln(1 - eps_m) followed by ternary search finds the global minimum."""
    e = np.asarray(e, complex)
    f = lambda s: np.max(np.abs(e - (1 - np.exp(s)))) / np.exp(s)
    s_grid = np.linspace(-14, 14, 300)
    vals = np.abs(e[None, :] - (1 - np.exp(s_grid))[:, None]).max(1) / np.exp(s_grid)
    k = int(np.argmin(vals))
    lo, hi = s_grid[max(k - 1, 0)], s_grid[min(k + 1, s_grid.size - 1)]
    for _ in range(60):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if f(m1) < f(m2):
            hi = m2
        else:
            lo = m1
    s = (lo + hi) / 2
    R = f(s)
    if R >= 1:
        return 1.0, None
    return R, 1 - np.exp(s)


def low_eps_merit_lossless(e):
    """Closed form of low_eps_merit for real e: with a = min e, b = max e, the optimum target is
    the midpoint eps_m = (a + b)/2, R = (b - a)/(2 - a - b), if b < 1; otherwise R = 1 (eps_m -> -inf)."""
    a, b = np.min(e), np.max(e)
    return ((b - a) / (2 - a - b), (a + b) / 2) if b < 1 else (1.0, None)


def model_low_eps(p, B, K=400, closed_form=True):
    """(R, eps_m*) of the model on the band of fractional width B centred on w_c.  Lossless models
    use the closed form (validated against the numerical minimization in tests/)."""
    w1, w2 = band(B)
    w = np.linspace(w1, w2, K)
    e = eps(w, p)
    if p.g == 0:
        e = e.real
        if closed_form:
            return low_eps_merit_lossless(e)
    return low_eps_merit(e)


# ------------------------------------------------------------------ parameter sweeps
GRID_W0 = np.linspace(0.0, 3.0, 31)
GRID_WP = np.array([0.2, 0.5, 1.0, 1.5, 2.0, 3.0])
GRID_FD = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
GRID_B = np.array([0.1, 0.4, 1.0, 1.6])


def sweep_low_eps(gammas=(0.0,), w0_grid=GRID_W0):
    """min over the grid of R_model / bound (>= 1 means the bound is respected), and its argmin."""
    worst, arg = np.inf, None
    for g in gammas:
        for B in GRID_B:
            for w0 in w0_grid:
                for wp in GRID_WP:
                    for fD in GRID_FD:
                        p = Params(w0, wp, g, fD)
                        R, _ = model_low_eps(p, B, K=200)
                        r = R / bound_low_eps(B)
                        if r < worst:
                            worst, arg = r, (B, p)
    return worst, arg


def sweep_max_n(gammas=(0.0,), max_loss_tangent=np.inf):
    """max over the grid of LHS/RHS of [eq:max_n] at w_c (<= 1 means respected); count of points."""
    worst, arg, count = 0.0, None, 0
    for g in gammas:
        for w0 in np.linspace(1.02, 4.0, 60):
            for wp in GRID_WP:
                for fD in GRID_FD:
                    p = Params(w0, wp, g, fD)
                    s = max_n_sides(p)
                    if s is None:
                        continue
                    e = eps(WC, p)
                    if e.imag / (e.real - 1) > max_loss_tangent:
                        continue
                    count += 1
                    r = s[0] / s[1]
                    if r > worst:
                        worst, arg = r, p
    return worst, arg, count


# ------------------------------------------------------------------ figure
W_PLOT = np.linspace(1e-3, 3.0, 3000)
B_PLOT = np.linspace(0.02, 1.6, 50)


def draw(axes, p, B):
    """Redraw the four panels for parameters p and fractional bandwidth B; returns (R, bound, max_n sides)."""
    a_re, a_im, a_n, a_b = axes
    for ax in axes:
        ax.cla()
    w = W_PLOT
    e = eps(w, p)
    w1, w2 = band(B)
    lossless = p.g == 0
    R, em = model_low_eps(p, B)
    hb = bound_low_eps(B)
    lossy_note = "" if lossless else r" (stated for lossless media; $\gamma>0$ shown)"

    # (a) Re eps with the [eq:low_epsilon_bound] corridor
    re = e.real.copy()
    re[np.abs(re) > 50] = np.nan
    a_re.plot(w, re, "k-", lw=1.4)
    a_re.axhline(1, color="0.6", lw=0.8, ls=":")
    a_re.axhline(0, color="0.6", lw=0.6)
    a_re.axvspan(w1, w2, color="C0", alpha=0.12, lw=0)
    if em is not None:
        half = hb * (1 - em)
        a_re.fill_between([w1, w2], em - half, em + half, color="C3", alpha=0.30, lw=0)
        a_re.plot([w1, w2], [em, em], "C3--", lw=1)
        corridor = (r"Re $\varepsilon$ must leave the red corridor $\varepsilon_m\pm\frac{B}{1+B/2}(\varepsilon_0-\varepsilon_m)$"
                    f" in the band;\nbest target $\\varepsilon_m$ = {em:.3f}")
    else:
        corridor = r"no target $\varepsilon_m<\varepsilon_0$ is tracked on this band (Re $\varepsilon\geq\varepsilon_0$ or a pole in it)" + "\n"
    a_re.set_title(f"(a) band B = {B:.2f} (blue), [eq:low_epsilon_bound]{lossy_note}\n" + corridor,
                   fontsize=7.5, loc="left")
    a_re.set_xlim(0, 3)
    a_re.set_ylim(-8, 8)
    a_re.set_xlabel(r"$\omega/\omega_c$")
    a_re.set_ylabel(r"Re $\varepsilon$")

    # (b) Im eps and the f-sum rule
    a_im.set_yscale("log")
    if lossless:
        if p.fD < 1:
            a_im.vlines(p.w0, 1e-4, 1e3, color="k", lw=1.4)
        if p.fD > 0:
            a_im.vlines(0.01, 1e-4, 1e3, color="k", lw=1.4)
        fs = (r"$\gamma = 0$: Im $\varepsilon$ is a sum of $\delta$-functions whose weights exhaust $\omega_p^2$;"
              "\n" + r"drag $\gamma > 0$ for a numerical check of [eq:f-sum rule]")
    else:
        a_im.plot(w, e.imag, "k-", lw=1.4)
        fs = (r"(b) [eq:f-sum rule]: $\int_0^\infty \omega\,$Im$\,\varepsilon\,d\omega$ = $\pi\omega_p^2/2$;"
              f"\nquadrature / ($\\pi\\omega_p^2/2$) = {fsum_ratio(p):.6f}")
    a_im.axvspan(w1, w2, color="C0", alpha=0.12, lw=0)
    a_im.set_title(fs if not lossless else "(b) " + fs, fontsize=7.5, loc="left")
    a_im.set_xlim(0, 3)
    a_im.set_ylim(1e-4, 1e3)
    a_im.set_xlabel(r"$\omega/\omega_c$")
    a_im.set_ylabel(r"Im $\varepsilon$")

    # (c) [eq:max_n] in the (n', n) plane at w = w_c
    x, y = max_n_curve(p.wp)
    a_n.fill_between(x, y, 6, color="C3", alpha=0.18, lw=0)
    a_n.plot(x, y, "C3-", lw=1.4)
    s = max_n_sides(p)
    if s is not None:
        lhs, rhs, n, npr = s
        a_n.plot(npr * WC, n, "o", ms=7, mfc="k" if lossless else "none", mec="k")
        txt = f"model: n = {n:.3f}, LHS/RHS = {lhs / rhs:.4f}"
        if not lossless:
            txt += r" (hollow: $\gamma>0$, not transparent at $\omega_c$)"
    else:
        txt = r"model: n $\leq$ 1 at $\omega_c$ (the bound concerns n > 1)"
    a_n.set_title("(c) [eq:max_n] at $\\omega_c$: $(n^2-1)^2/n \\leq \\omega_p^2 n'/\\omega$"
                  " (red: equality; shaded: forbidden)\n" + txt, fontsize=7.5, loc="left")
    a_n.set_xscale("log")
    a_n.set_xlim(1e-3, 1e3)
    a_n.set_ylim(1, 6)
    a_n.set_xlabel(r"dispersion $\omega_c\,dn/d\omega$")
    a_n.set_ylabel(r"index $n(\omega_c)$")

    # (d) [eq:low_epsilon_bound] vs bandwidth
    Rb = np.array([model_low_eps(p, b, K=200)[0] for b in B_PLOT])
    a_b.fill_between(B_PLOT, 0, bound_low_eps(B_PLOT), color="C3", alpha=0.18, lw=0)
    a_b.plot(B_PLOT, bound_low_eps(B_PLOT), "C3-", lw=1.4, label=r"bound $B/(1+B/2)$")
    a_b.plot(B_PLOT, Rb, "k-", lw=1.4, label="model")
    a_b.plot(B, R, "o", ms=7, mfc="k" if lossless else "none", mec="k")
    a_b.axvline(B, color="C0", lw=0.8, ls=":")
    a_b.set_title(f"(d) [eq:low_epsilon_bound] vs B{lossy_note}\n"
                  f"at B = {B:.2f}: model R = {R:.4f} $\\geq$ bound {hb:.4f} (ratio {R / hb:.3f})",
                  fontsize=7.5, loc="left")
    a_b.set_xlim(0, 1.6)
    a_b.set_ylim(0, 1.05)
    a_b.set_xlabel(r"fractional bandwidth $B$")
    a_b.set_ylabel(r"$R = \mathrm{min}_{\varepsilon_m}\mathrm{max}_{\rm band}|\varepsilon-\varepsilon_m|/(\varepsilon_0-\varepsilon_m)$",
                   fontsize=8)
    a_b.legend(loc="lower right", fontsize=7, frameon=False)
    return R, hb, s


def make_figure(p, B):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.4))
    draw(ax.ravel(), p, B)
    fig.suptitle(f"Drude-Lorentz sandbox: $\\omega_0$ = {p.w0:.2f}, $\\omega_p$ = {p.wp:.2f}, "
                 f"$\\gamma$ = {p.g:.3f}, Drude share $c_D$ = {p.fD:.2f}, B = {B:.2f}  (units of $\\omega_c$)",
                 fontsize=9)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ interactive mode
def interactive():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.30, hspace=0.45, wspace=0.25)
    axes = ax.ravel()
    specs = [("w0", r"$\omega_0$ (resonance)", 0.0, 3.0, DEFAULT.w0),
             ("wp", r"$\omega_p$ (strength)", 0.1, 3.0, DEFAULT.wp),
             ("g", r"$\gamma$ (damping)", 0.0, 0.5, DEFAULT.g),
             ("fD", r"Drude share $c_D$", 0.0, 1.0, DEFAULT.fD),
             ("B", r"bandwidth $B$", 0.02, 1.6, DEFAULT_B)]
    sliders = {}
    for i, (key, label, lo, hi, v0) in enumerate(specs):
        sax = fig.add_axes([0.20, 0.22 - 0.04 * i, 0.60, 0.025])
        sliders[key] = Slider(sax, label, lo, hi, valinit=v0)

    def update(_=None):
        p = Params(sliders["w0"].val, sliders["wp"].val, sliders["g"].val, sliders["fD"].val)
        draw(axes, p, sliders["B"].val)
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)
    update()
    fig.suptitle("Drude-Lorentz sandbox (Secs. 2.1-2.2): drag the sliders", fontsize=10)
    plt.show()


# ------------------------------------------------------------------ snapshot mode
def main(make_fig=True):
    print("=" * 92)
    print("Secs. 2.1-2.2  Drude-Lorentz sandbox: one oscillator (+ Drude share) vs the tutorial's bounds")
    print("=" * 92)
    print("Units: w_c = 1, eps0 = 1.  eps = 1 - c_D wp^2/(w^2 + i g w) + (1 - c_D) wp^2/(w0^2 - w^2 - i g w)")

    states = [DEFAULT, Params(1.5, 1.0, 0.0, 0.0), Params(0.6, 1.5, 0.05, 0.0), Params(2.5, 2.0, 0.3, 0.5),
              Params(0.0, 1.0, 0.02, 1.0), Params(0.0, 1.0, 0.0, 1.0)]

    print("\nA. [eq:f-sum rule]  int_0^inf w Im eps dw  vs  pi wp^2/2  (trapezoid on tan-mapped grid, w <= 1e6)")
    print(f"   {'w0':>5} {'wp':>5} {'g':>6} {'c_D':>5} | {'integral':>12} {'pi wp^2/2':>12} {'ratio - 1':>11}")
    for p in states + [Params(1.2, 0.5, 1e-3, 0.3), Params(2.0, 3.0, 0.5, 0.0)]:
        I = fsum_integral(p)
        if I is None:
            print(f"   {p.w0:5.2f} {p.wp:5.2f} {p.g:6.3f} {p.fD:5.2f} | {'delta fns':>12} {np.pi * p.wp**2 / 2:12.6f}"
                  f" {'exact':>11}")
        else:
            t = np.pi * p.wp**2 / 2
            print(f"   {p.w0:5.2f} {p.wp:5.2f} {p.g:6.3f} {p.fD:5.2f} | {I:12.6f} {t:12.6f} {I / t - 1:+11.2e}")

    print("\nB. [eq:max_n] at w = w_c:  (n^2 - 1)^2/n  <=  wp^2 n'/w")
    print(f"   {'w0':>5} {'wp':>5} {'g':>6} {'c_D':>5} | {'n':>7} {'n_prime':>9} {'LHS':>10} {'RHS':>10} {'LHS/RHS':>9}")
    for p in [DEFAULT, Params(1.5, 1.0, 0.0, 0.0), Params(1.1, 1.0, 0.0, 0.0), Params(3.0, 2.5, 0.0, 0.0),
              Params(1.5, 1.0, 0.0, 0.3), Params(1.5, 1.0, 0.0, 0.6), Params(1.2, 1.0, 0.1, 0.0)]:
        s = max_n_sides(p)
        if s is None:
            print(f"   {p.w0:5.2f} {p.wp:5.2f} {p.g:6.3f} {p.fD:5.2f} |  n <= 1: bound not applicable")
            continue
        lhs, rhs, n, npr = s
        print(f"   {p.w0:5.2f} {p.wp:5.2f} {p.g:6.3f} {p.fD:5.2f} | {n:7.4f} {npr:9.4f} {lhs:10.5f} {rhs:10.5f} {lhs / rhs:9.6f}")
    w0, a0, c0 = sweep_max_n()
    print(f"   lossless grid ({c0} points, w0 in [1.02, 4], 6 wp, c_D in 0..1): max LHS/RHS = {w0:.12f}")
    print("   -> a single lossless oscillator (c_D = 0) saturates [eq:max_n] exactly; c_D > 0 falls below.")
    w1_, a1, c1 = sweep_max_n(gammas=(1e-3, 0.01, 0.05, 0.1), max_loss_tangent=0.01)
    print(f"   damped, loss tangent Im eps/(Re eps - 1) < 0.01 ({c1} points): max LHS/RHS = {w1_:.6f}"
          f"  at w0 = {a1.w0:.2f}, wp = {a1.wp:.1f}, g = {a1.g}")
    print("   -> CAVEAT: with g > 0 the medium absorbs at w_c and [eq:max_n] (a transparency bound) can be")
    print("      exceeded by O(loss tangent); the sandbox greys out the model point when g > 0.")

    print("\nC. [eq:low_epsilon_bound]  R = min_{eps_m < eps0} max_band |eps - eps_m|/(eps0 - eps_m)  >=  B/(1 + B/2)")
    print(f"   {'w0':>5} {'wp':>5} {'g':>6} {'c_D':>5} {'B':>5} | {'eps_m*':>9} {'R model':>9} {'bound':>9} {'R/bound':>8}")
    for p, B in [(DEFAULT, DEFAULT_B), SNAPSHOT_STATES["loweps"], (Params(0.4, 1.2, 0.0, 0.0), 0.4), (Params(0.6, 1.5, 0.0, 0.0), 0.4), (Params(0.3, 1.5, 0.0, 0.0), 0.4),
                 (Params(0.0, 1.0, 0.0, 1.0), 0.4), (Params(0.0, 1.0, 0.0, 1.0), 1.0), (Params(0.0, 1.0, 0.0, 1.0), 0.1),
                 (Params(0.6, 1.5, 0.0, 0.5), 0.4), (Params(0.0, 1.0, 0.1, 1.0), 0.4)]:
        R, em = model_low_eps(p, B)
        hb = bound_low_eps(B)
        ems = "  -inf" if em is None else f"{em:9.4f}"
        print(f"   {p.w0:5.2f} {p.wp:5.2f} {p.g:6.3f} {p.fD:5.2f} {B:5.2f} | {ems:>9} {R:9.5f} {hb:9.5f} {R / hb:8.4f}")
    for B in (0.1, 0.4, 1.0):
        print(f"   lossless Drude limit, B = {B}: R = (w2^2 - w1^2)/(w2^2 + w1^2) = {B / (1 + B**2 / 4):.5f}"
              f" vs bound {bound_low_eps(B):.5f}")
    r0, (B0, p0) = sweep_low_eps()
    print(f"   lossless grid (31 w0 in [0, 3] x 6 wp x 5 c_D x B in {GRID_B.tolist()}): min R/bound = {r0:.6f}"
          f"  (B = {B0}, w0 = {p0.w0}, c_D = {p0.fD})")
    r1, (B1, p1) = sweep_low_eps(gammas=(0.01, 0.1, 1.0), w0_grid=GRID_W0[::3])
    print(f"   damped grid (11 w0, g in 0.01, 0.1, 1), outside the bound's lossless hypothesis: min R/bound = {r1:.6f}"
          f"  (B = {B1}, w0 = {p1.w0}, g = {p1.g}, c_D = {p1.fD})")
    print("   -> the bound is never violated; the closest approach is the lossless Drude limit (slack")
    print("      (1 + B/2)/(1 + B^2/4)), the open question after [eq:low_epsilon_bound].")

    if make_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIGDIR, exist_ok=True)
        print()
        for tag, (p, B) in SNAPSHOT_STATES.items():
            fig = make_figure(p, B)
            stem = os.path.join(FIGDIR, f"sec2_drude_lorentz_sandbox_{tag}")
            for ext in ("pdf", "png"):
                fig.savefig(stem + "." + ext, dpi=200)
            plt.close(fig)
            print(f"wrote {os.path.normpath(stem)}.pdf/png")


def _gui_available():
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if backend in ("agg", "pdf", "ps", "svg", "pgf", "cairo", "template") or "inline" in backend:
        return False
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    return True


if __name__ == "__main__":
    if "--snapshot" in sys.argv or not _gui_available():
        main()
    else:
        interactive()
