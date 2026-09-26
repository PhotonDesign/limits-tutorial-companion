"""Sec. 4.1: extinction symmetry of reciprocal scatterers, 1D graded lossy slab (fig:ExtSym).

Tutorial claims checked (tutorial.tex line numbers as of 2026-09):
  1. [~1425-1440] P_ext(psi_inc) = P_ext(psi_inc^*) for any reciprocal scatterer; for a slab,
     front (psi_1) and rear (psi_2 = psi_1^*) illumination give identical extinction
     2 Re(1 - t) while absorption and scattering differ ("at all frequencies", caption (c)).
  2. [~1444] geometry: Re n linear from 3.5 (front) to 1 (rear), parabolic Im n peaking at 0.2;
     original MATLAB: testSymmetricExtinction.m (L = 1, 100 staircase layers, f = 0.25..5 c/L).
  3. Extension: oblique incidence (theta = 30, 60 deg) and both s and p polarizations.

FINDING (figure, not text): the MATLAB script computes Pext = 2 Re(1 - t) with Yeh's t, which
is referenced to the EXIT face.  The optical-theorem amplitude is t_ext = t exp(-i k L).  The
plotted "extinction" (and scattering = ext - abs) curves in panel (c) therefore are not the
physical extinction for f < ~2 c/L (e.g. f = 0.25: plotted 3.35, true 2.48).  The front/rear
equality -- the point of the figure -- holds for both quantities, since t_face is itself
reciprocal.  Sanity check: an EMPTY slab (n = 1) has 2 Re(1 - t_face) = 2(1 - cos kL) != 0.

Run:  python extinction_symmetry_slab.py   (about 1 s; writes figures/sec4_extinction_symmetry_slab.pdf/png)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import tmm1d  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def graded_slab(n_layers=100, L=1.0, n_front=3.5, im_peak=0.2):
    """Staircase discretization at layer midpoints, exactly as testSymmetricExtinction.m.
    Returns (eps list front->rear, thicknesses)."""
    pos = np.linspace(0, L, n_layers + 1)
    x = 0.5 * (pos[1:] + pos[:-1])
    n = 1 + (n_front - 1) * (1 - x / L) + 1j * im_peak * 4 / L**2 * x * (L - x)
    return n**2, np.diff(pos)


def cross_sections(eps, d, f, theta_deg=0.0, pol="s"):
    """Front (1) and rear (2) incidence: extinction, absorption, scattering, reflectance,
    plus the MATLAB-convention quantity 2 Re(1 - t_face).  f in units of c/L (k0 = 2 pi f)."""
    k0 = 2 * np.pi * np.asarray(f, float)
    kx = k0 * np.sin(np.deg2rad(theta_deg))
    L = np.sum(d)
    out = {}
    r1, t1, r2, t2, kz0 = tmm1d.rt_both_sides(eps, d, k0, kx, pol)
    for tag, r, t in (("1", r1, t1), ("2", r2, t2)):
        ext = tmm1d.extinction(t, kz0, L)
        A = 1 - np.abs(r)**2 - np.abs(t)**2            # vacuum on both sides
        out["ext" + tag], out["abs" + tag], out["sca" + tag] = ext, A, ext - A
        out["R" + tag], out["t" + tag] = np.abs(r)**2, t
        out["ext_matlab" + tag] = 2 * np.real(1 - t)
        # independent scattering: |r|^2 + |t_ext - 1|^2 (reflected + forward scattered flux)
        out["sca_direct" + tag] = np.abs(r)**2 + np.abs(t * np.exp(-1j * kz0 * L) - 1)**2
    return out


def empty_slab_check(f):
    eps, d = np.ones(10, complex), np.full(10, 0.1)
    cs = cross_sections(eps, d, f)
    return np.abs(cs["ext1"]).max(), np.abs(cs["ext_matlab1"]).max()


def main(make_figure=True):
    print("=" * 84)
    print("Sec. 4.1  extinction symmetry for a graded lossy slab (fig:ExtSym)")
    print("=" * 84)
    eps, d = graded_slab()
    f = np.linspace(0.25, 5.0, 300)                  # MATLAB: w = linspace(2pi/4, 2pi/0.2, 300)
    print(f"{'theta':>5} {'pol':>3} | {'max|ext1-ext2|':>14} {'max|ext_ml1-ml2|':>16} | {'max|A1-A2|':>10}"
          f" {'max|sca1-sca2|':>14} {'max|R1-R2|':>10} | {'max|sca-sca_direct|':>19}")
    worst = 0.0
    for th in (0.0, 30.0, 60.0):
        for pol in ("s", "p"):
            cs = cross_sections(eps, d, f, th, pol)
            de = np.abs(cs["ext1"] - cs["ext2"]).max()
            worst = max(worst, de)
            print(f"{th:5.0f} {pol:>3} | {de:14.1e} {np.abs(cs['ext_matlab1'] - cs['ext_matlab2']).max():16.1e} |"
                  f" {np.abs(cs['abs1'] - cs['abs2']).max():10.3f} {np.abs(cs['sca1'] - cs['sca2']).max():14.3f}"
                  f" {np.abs(cs['R1'] - cs['R2']).max():10.3f} |"
                  f" {max(np.abs(cs['sca1'] - cs['sca_direct1']).max(), np.abs(cs['sca2'] - cs['sca_direct2']).max()):19.1e}")
    print(f"  -> extinction equal to {worst:.1e} at every frequency, angle, polarization;"
          " absorption/scattering/reflection differ by O(0.1-0.7).")

    # random asymmetric lossy stack as a second instance
    rng = np.random.default_rng(5)
    eps_r = (rng.uniform(1, 4, 15) + 1j * rng.uniform(0, 0.3, 15))**2
    d_r = rng.uniform(0.02, 0.15, 15)
    cs = cross_sections(eps_r, d_r, f, 45.0, "p")
    print(f"  random 15-layer lossy stack, 45 deg, p: max|ext1-ext2| = {np.abs(cs['ext1'] - cs['ext2']).max():.1e},"
          f" max|A1-A2| = {np.abs(cs['abs1'] - cs['abs2']).max():.3f}")

    print("\nFigure-(c) quantity: MATLAB 2 Re(1 - t_face) vs physical 2 Re(1 - t_face e^{-ikL}), normal incidence")
    cs = cross_sections(eps, d, f)
    print(f"{'f (c/L)':>8} | {'ext (true)':>10} {'ext (MATLAB)':>12} | {'A1 (front)':>10} {'A2 (rear)':>9} |"
          f" {'sca1 true':>9} {'sca1 MATLAB':>11}")
    for fi in (0.25, 0.457, 0.647, 0.90, 1.10, 2.0, 3.0, 5.0):
        i = np.argmin(np.abs(f - fi))
        print(f"{f[i]:8.3f} | {cs['ext1'][i]:10.3f} {cs['ext_matlab1'][i]:12.3f} | {cs['abs1'][i]:10.3f}"
              f" {cs['abs2'][i]:9.3f} | {cs['sca1'][i]:9.3f} {cs['ext_matlab1'][i] - cs['abs1'][i]:11.3f}")
    print("  (published panel (c): black x at f = 0.25 is ~3.35, first minimum ~0.85 near f = 0.46,"
          " next peak ~2.9 near f = 0.65 -> matches the MATLAB column, not the physical one)")
    e_true, e_ml = empty_slab_check(f)
    print(f"  empty slab (n = 1): max ext true = {e_true:.1e}; max MATLAB quantity = {e_ml:.3f}")
    eps_f, d_f = graded_slab(1000)
    cs_f = cross_sections(eps_f, d_f, f)
    print(f"  discretization: 100 vs 1000 layers, max |ext difference| = {np.abs(cs_f['ext1'] - cs['ext1']).max():.1e}")

    if make_figure:
        make_plot(f, cs, eps, d)
    return cs


def make_plot(f, cs, eps, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.cumsum(d) - d / 2
    n = np.sqrt(eps)
    ax[0].plot(x, n.real, "b", label="Re n")
    ax[0].plot(x, n.imag, "r", label="Im n")
    ax[0].set_xlabel("position x (L)"); ax[0].set_ylabel("refractive index"); ax[0].legend()
    ax[0].set_title("graded slab (front at x = 0)")
    for a, key, ttl in ((ax[1], "ext", "physical: 2 Re(1 - t e$^{-ikL}$)"),
                        (ax[2], "ext_matlab", "as in published fig.: 2 Re(1 - t$_{face}$)")):
        a.plot(f, cs["abs1"], "r--", lw=1, label=r"abs, $\psi_1$")
        a.plot(f, cs["abs2"], "b--", lw=1, label=r"abs, $\psi_2$")
        a.plot(f, cs[key + "1"] - cs["abs1"], "r-.", lw=1, label=r"scat, $\psi_1$")
        a.plot(f, cs[key + "2"] - cs["abs2"], "b-.", lw=1, label=r"scat, $\psi_2$")
        a.plot(f, cs[key + "1"], "k-", lw=1.2, label=r"ext, $\psi_1$")
        a.plot(f[::4], cs[key + "2"][::4], "kx", ms=4, label=r"ext, $\psi_2$")
        a.set_ylim(0, 4); a.set_xlim(0, 5)
        a.set_xlabel("frequency f (c/L)"); a.set_ylabel(r"$\sigma/A$"); a.set_title(ttl, fontsize=10)
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"sec4_extinction_symmetry_slab.{ext}"), dpi=150)
    print("\nwrote figures/sec4_extinction_symmetry_slab.pdf/png")


if __name__ == "__main__":
    main()
