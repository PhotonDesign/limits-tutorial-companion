"""Sec. 2.1: how large can Re chi (and n) be at fixed dispersion?  The oscillator-strength
linear program is solved directly and compared with the single-oscillator closed forms and
the refractive-index bounds as printed.

Tutorial claims checked (tutorial.tex line numbers):
  [eq:chi_causal_repres], [eq:normalized_f_sum]  l.763-769  Re chi = sum c_i w_p^2/(w_i^2 - w^2),
             sum c_i = 1, c_i >= 0
  l.775-782  LP: max Re chi s.t. d Re chi/dw <= chi', sum c_i = 1, c_i >= 0
  l.783      optimum = a single oscillator, c_0 = 1, w_0^2 = w^2 (1 + sqrt(2 w_p^2/(w^3 chi')))
  [eq:max_n]       l.795  ((n^2 - 1)/n)^2 <= w_p^2 n'/w       (as printed)
  [eq:max_n_large] l.800  n <= (w_p^2 n'/w)^{1/3}, with n' = chi'/(2n)
  l.803      "a ten-fold increase in refractive index ... would require a 10^3 increase in
             electron density"

Findings.
  * LP optimum = closed form (to the grid resolution), attained by one oscillator (or the two
    grid points bracketing w_0), at the printed w_0.  CONFIRMED.
  * The optimum satisfies (n^2 - 1)^2 / n = w_p^2 n'/w EXACTLY (Shim, Monticone & Miller,
    Adv. Mater. 2021, Eq. 7).  The printed [eq:max_n] has (n^2 - 1)^2 / n^2 on the left: it is
    looser by a factor n and does not imply [eq:max_n_large] (it gives n ~ (w_p^2 n'/w)^{1/2}).
    For silicon at 1550 nm (n = 3.48) the printed form allows n <= 7.25; the correct form gives
    n <= 3.87 (and R^{1/3} = 3.70).
  * [eq:max_n_large] is the large-n asymptote of the correct bound, not a strict bound: the
    exact optimum exceeds (w_p^2 n'/w)^{1/3} by a factor (1 - 1/n^2)^{-2/3} (4% at n = 4).

Run:  python max_refractive_index.py    (about 2 s; writes figures/sec2_max_refractive_index.png/pdf)
"""
import os
import sys
import numpy as np
from scipy.optimize import linprog, brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
C0 = 299792458.0
EPS0 = 8.8541878128e-12
QE = 1.602176634e-19
ME = 9.1093837015e-31


# ------------------------------------------------------------------ the LP of l.775-782
def oscillator_grid(w=1.0, n_above=4000, n_below=600):
    """Transition frequencies w_i, parametrized by x_i = w_i^2 - w^2 on geometric grids on
    both sides of w (x = -w^2 is the Drude term w_i = 0)."""
    xa = np.geomspace(1e-5, 1e7, n_above) * w**2
    xb = -np.geomspace(1e-5, 1.0, n_below) * w**2
    return np.concatenate([xb, xa])


def lp_max_chi(wp, chip, w=1.0, grid=None):
    """max sum c_i f_i(w) s.t. sum c_i f_i'(w) <= chi', sum c_i = 1, c_i >= 0."""
    x = oscillator_grid(w) if grid is None else grid
    f = wp**2 / x
    fp = 2 * w * wp**2 / x**2
    res = linprog(-f, A_ub=fp[None, :], b_ub=[chip], A_eq=np.ones((1, x.size)), b_eq=[1.0],
                  bounds=(0, None), method="highs")
    c = res.x
    support = np.flatnonzero(c > 1e-9)
    w_i = np.sqrt(x[support] + w**2)
    return -res.fun, w_i, c[support]


def chi_max_closed(wp, chip, w=1.0):
    return wp * np.sqrt(chip / (2 * w))


def w0_closed(wp, chip, w=1.0):
    return w * np.sqrt(1 + np.sqrt(2 * wp**2 / (w**3 * chip)))


# ------------------------------------------------------------------ refractive-index forms
def n_bound_correct(R):
    """Largest n with (n^2 - 1)^2 / n <= R  (Shim et al. Eq. 7)."""
    return brentq(lambda n: (n**2 - 1)**2 / n - R, 1.0, 10 * R + 10)


def n_bound_printed(R):
    """Largest n with ((n^2 - 1)/n)^2 <= R  ([eq:max_n] as printed)."""
    return brentq(lambda n: ((n**2 - 1) / n)**2 - R, 1.0, 10 * R + 10)


def n_bound_cuberoot(R):
    return R**(1 / 3)


# ------------------------------------------------------------------ real materials
def si_index(lam_um):
    """Crystalline Si, 293 K (H. H. Li, J. Phys. Chem. Ref. Data 9, 561 (1980) Sellmeier form)."""
    l2 = lam_um**2
    return np.sqrt(11.6858 + 0.939816 / l2 + 0.00810461 * 1.1071**2 / (l2 - 1.1071**2))


def silica_index(lam_um):
    """Fused silica (I. H. Malitson, JOSA 55, 1205 (1965))."""
    l2 = lam_um**2
    return np.sqrt(1 + 0.6961663 * l2 / (l2 - 0.0684043**2) + 0.4079426 * l2 / (l2 - 0.1162414**2)
                   + 0.8974794 * l2 / (l2 - 9.896161**2))


MATERIALS = {
    # name: (index function, valence-electron density [m^-3])
    "Si": (si_index, 4 * 4.994e28),                      # 4 valence e- per atom
    "SiO2": (silica_index, 16 * 2.203e3 / 60.08e-3 * 6.02214076e23),   # 16 e- per SiO2 unit
}


def material_point(name, lam_um=1.55):
    nf, Ne = MATERIALS[name]
    n = float(nf(lam_um))
    h = 1e-4
    dn_dlam = (nf(lam_um + h) - nf(lam_um - h)) / (2 * h) * 1e6          # per metre
    lam = lam_um * 1e-6
    w = 2 * np.pi * C0 / lam
    dn_dw = -dn_dlam * lam**2 / (2 * np.pi * C0)
    wp2 = Ne * QE**2 / (EPS0 * ME)
    R = wp2 * dn_dw / w
    return dict(n=n, dn_dw=dn_dw, wp_eV=np.sqrt(wp2) * 6.582119569e-16, R=R,
                correct=n_bound_correct(R), printed=n_bound_printed(R), cube=n_bound_cuberoot(R))


def make_figure(stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    chip = np.geomspace(0.05, 200, 60)
    for k, wp in enumerate([2.0, 5.0, 15.0]):
        ax[0].plot(chip, chi_max_closed(wp, chip), f"C{k}-", lw=1, label=rf"closed form, $\omega_p$={wp:g}$\omega$")
        cs = chip[::6]
        ax[0].plot(cs, [lp_max_chi(wp, c)[0] for c in cs], f"C{k}o", ms=3, mfc="none")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"allowed dispersion $\chi'\,\omega$")
    ax[0].set_ylabel(r"max Re $\chi(\omega)$ (LP: circles)")
    ax[0].legend(fontsize=7, frameon=False)
    R = np.geomspace(1, 1e4, 200)
    ax[1].plot(R, [n_bound_correct(r) for r in R], "k-", label=r"$(n^2-1)^2/n \leq R$ (LP optimum)")
    ax[1].plot(R, [n_bound_printed(r) for r in R], "C3--", label=r"$((n^2-1)/n)^2 \leq R$ (printed)")
    ax[1].plot(R, R**(1 / 3), "C0:", label=r"$R^{1/3}$ (eq:max_n_large)")
    for name, mk in (("Si", "s"), ("SiO2", "^")):
        p = material_point(name)
        ax[1].plot(p["R"], p["n"], "k" + mk, ms=5, label=f"{name}, 1550 nm")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$R = \omega_p^2 n'/\omega$")
    ax[1].set_ylabel(r"max $n$")
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(stem + "." + ext, dpi=200)
    plt.close(fig)


def main(make_fig=True):
    print("=" * 96)
    print("Sec. 2.1  Maximum Re chi / refractive index at fixed dispersion: LP vs closed forms (w = 1)")
    print("=" * 96)
    print(f"{'w_p':>5} {'chi_':>7} | {'LP max chi':>12} {'closed':>12} {'rel gap':>8} | {'#osc':>4} "
          f"{'LP w_i':>18} {'w_0 (l.783)':>11} | {'n':>7} {'(n2-1)^2/n / R':>14} {'printed / R':>11}")
    for wp, chip in [(2.0, 0.3), (2.0, 5.0), (5.0, 1.0), (15.0, 0.5), (15.0, 50.0)]:
        v, wi, ci = lp_max_chi(wp, chip)
        cf = chi_max_closed(wp, chip)
        n = np.sqrt(1 + v)
        npr = chip / (2 * n)
        R = wp**2 * npr / 1.0
        ws = ", ".join(f"{x:.4f}" for x in wi)
        print(f"{wp:5.1f} {chip:7.2f} | {v:12.6f} {cf:12.6f} {(cf - v) / cf:8.1e} | {len(wi):4d} {ws:>18} "
              f"{w0_closed(wp, chip):11.4f} | {n:7.4f} {((n**2 - 1)**2 / n) / R:14.6f} {((n**2 - 1) / n)**2 / R:11.6f}")
    print("  (#osc = 2: the two grid points bracketing w_0; gap -> 0 as the grid is refined)")
    print("  -> optimum satisfies (n^2-1)^2/n = w_p^2 n'/w; printed LHS ((n^2-1)/n)^2 is smaller by the factor n")

    print("\nReal materials at 1550 nm (valence-electron density; n' = dn/dw from Sellmeier fits):")
    print(f"{'':6} {'n':>7} {'hbar w_p [eV]':>13} {'R':>8} | {'bound: Shim Eq.7':>16} {'printed eq:max_n':>16} {'R^(1/3)':>8}")
    for name in MATERIALS:
        p = material_point(name)
        print(f"{name:6} {p['n']:7.4f} {p['wp_eV']:13.2f} {p['R']:8.3f} | {p['correct']:16.4f} {p['printed']:16.4f} {p['cube']:8.4f}")
    print("\n[eq:max_n_large] is asymptotic: exact bound / R^(1/3) = "
          + ", ".join(f"{n_bound_correct(n**3 * (1 - 1 / n**2)**2) / n_bound_cuberoot(n**3 * (1 - 1 / n**2)**2):.4f} (n={n})"
                      for n in (2, 4, 10)))
    ratio = (40**3 * (1 - 1 / 40**2)**2) / (4**3 * (1 - 1 / 4**2)**2)
    print(f"l.803: 10x in n at fixed w, n' needs 10^3 x N_e for large n; exact form, n = 4 -> 40: x{ratio:.0f}")

    if make_fig:
        stem = os.path.join(FIGDIR, "sec2_max_refractive_index")
        make_figure(stem)
        print(f"\nwrote {os.path.normpath(stem)}.pdf/png")


if __name__ == "__main__":
    main()
