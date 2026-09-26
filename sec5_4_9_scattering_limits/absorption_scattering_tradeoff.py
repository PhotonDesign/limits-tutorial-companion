"""Sec. 5.7 (sec:cloaking), single-frequency tradeoffs between absorption and scattering.
Mie theory (sec5_local_constraints/mie.py, Bohren-Huffman conventions, e^{-i omega t}).

Tutorial claims (tutorial.tex line numbers):
  l.2230-2248  sigma_scat, sigma_ext, sigma_abs in terms of s_n (s_n = 1 - 2 a_n):
               (pi/2k^2) sum (2n+1)|s_n - 1|^2,  (pi/k^2) sum (2n+1)(1 - Re s_n),
               (pi/2k^2) sum (2n+1)(1 - |s_n|^2)   (TM + TE).
  l.2228 fn    c_n = (s_n - 1)/2 lies in the disk |c + 1/2| <= 1/2.
  l.2252-2257  (sigma_abs/sigma_scat)_n = (1 - |s|^2)/|s - 1|^2; maximum absorption when
               absorption = scattering (s_n = 0): (sigma_abs)_n <= (2n+1) lambda^2/(8 pi).
  l.2268       Andersen inequality, as printed:
                   sigma_abs + sigma_scat <= sqrt( sigma_scat D_s(k_i) / pi )
               with D_s the directivity of the scattered field in the forward direction.

Checks:
  A. The s_n forms of the three cross sections reproduce mie.py to machine precision.
  B. Per-channel absorption maximum (2n+1) lambda^2/(8 pi) at s = 0 (brute scan of the disk)
     and absorption = scattering there.
  C. Andersen inequality.  The printed form equates an area to a length: rescaling the length
     unit by 10^3 (m -> mm) multiplies LHS/RHS by 10^3, so its truth value depends on units.
     From the optical theorem, sigma_ext = (4 pi/k^2) Re S(0) <= (4 pi/k^2)|S(0)| and
     D_s = 4 pi |S(0)|^2/(k^2 sigma_sca):
                   sigma_abs + sigma_sca <= lambda sqrt( sigma_sca D_s / pi ),
     with equality iff the forward amplitude S(0) is real.  Checked on 4000 random lossy
     spheres (x = 0.05-15, Re m = 0.2-4, Im m = 1e-3-3) and on metal spheres at resonance.
     Corollary (maximize over sigma_sca): sigma_abs <= lambda^2 D_s/(4 pi) (effective-area form).

Run:  python absorption_scattering_tradeoff.py   (about 3 s; writes figures/sec5_4_9_andersen.png/pdf)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sec5_local_constraints.mie import mie_coefficients, cross_sections

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def cross_sections_s(m, x):
    """Cross sections (units 1/k^2) from s_n = 1 - 2 a_n, 1 - 2 b_n via l.2230-2248."""
    a, b = mie_coefficients(m, x)
    n = np.arange(1, a.shape[-1] + 1)
    sTM, sTE = 1 - 2 * a, 1 - 2 * b
    w = 2 * n + 1
    sca = np.pi / 2 * np.sum(w * (abs(sTM - 1) ** 2 + abs(sTE - 1) ** 2), -1)
    ext = np.pi * np.sum(w * (1 - sTM.real + 1 - sTE.real), -1)
    ab = np.pi / 2 * np.sum(w * (1 - abs(sTE) ** 2 + 1 - abs(sTM) ** 2), -1)
    return dict(sca=sca, ext=ext, abs=ab)


def forward_amplitude(m, x):
    """S(0) = (1/2) sum (2n+1)(a_n + b_n);  d sigma/d Omega (forward) = |S(0)|^2/k^2."""
    a, b = mie_coefficients(m, x)
    n = np.arange(1, a.shape[-1] + 1)
    return 0.5 * np.sum((2 * n + 1) * (a + b), -1)


def andersen_quantities(m, x, lam=1.0):
    """Returns sigma_ext, sigma_sca, sigma_abs, D_s in length units where lambda = lam."""
    k = 2 * np.pi / lam
    cs = cross_sections(m, x)
    S0 = forward_amplitude(m, x)
    ext, sca, ab = cs["ext"] / k ** 2, cs["sca"] / k ** 2, cs["abs"] / k ** 2
    D = 4 * np.pi * abs(S0) ** 2 / (k ** 2 * sca)
    return ext, sca, ab, D, S0


def random_spheres(N=4000, seed=7):
    rng = np.random.default_rng(seed)
    x = 10 ** rng.uniform(np.log10(0.05), np.log10(15), N)
    m = rng.uniform(0.2, 4, N) + 1j * 10 ** rng.uniform(-3, np.log10(3), N)
    return m, x


def andersen_study(N=4000):
    m, x = random_spheres(N)
    out = {}
    for lam, unit in ((1.0, "lambda"), (1e-6, "m (lambda = 1 um)"), (1e-3, "mm (lambda = 1 um)"),
                      (1e3, "nm (lambda = 1 um)")):
        ext, sca, ab, D, S0 = andersen_quantities(m, x, lam)
        printed = ext / np.sqrt(sca * D / np.pi)
        corrected = ext / (lam * np.sqrt(sca * D / np.pi))
        eff_area = ab / (lam ** 2 * D / (4 * np.pi))
        out[unit] = dict(printed=printed, corrected=corrected, eff_area=eff_area,
                         phase=np.angle(S0), m=m, x=x, sca=sca, D=D, ext=ext, lam=lam)
    return out


def disk_scan(n=1, npts=801):
    """Per-channel sigma_abs in units of (2n+1) lambda^2 over the passive disk |s| <= 1."""
    u = np.linspace(-1, 1, npts)
    S = u[:, None] + 1j * u[None, :]
    S = S[abs(S) <= 1]
    k = 2 * np.pi
    sab = np.pi / (2 * k ** 2) * (1 - abs(S) ** 2)            # per (2n+1), lambda = 1
    ssc = np.pi / (2 * k ** 2) * abs(S - 1) ** 2
    i = np.argmax(sab)
    return sab[i], S[i], ssc[i]


def main(make_figure=True):
    print("=" * 84)
    print("Sec. 5.7  absorption / scattering tradeoffs (Mie)")
    print("=" * 84)
    m, x = random_spheres(200)
    cs = cross_sections(m, x)
    css = cross_sections_s(m, x)
    err = max(np.max(abs(cs[q] - css[q]) / np.maximum(abs(cs[q]), 1e-30)) for q in ("ext", "sca", "abs"))
    print(f"A. s_n forms of sigma_sca/ext/abs vs mie.py over 200 random spheres: max rel. diff = {err:.1e}")
    a, b = mie_coefficients(m, x)
    print(f"   passive disk |c + 1/2| <= 1/2 (c = -a_n): max |c + 1/2| = {max(abs(-a + 0.5).max(), abs(-b + 0.5).max()):.6f}")

    smax, sarg, ssc = disk_scan()
    print(f"\nB. max per-channel sigma_abs/((2n+1) lambda^2) over the disk = {smax:.6f} at s = {sarg:.3f};"
          f" 1/(8 pi) = {1/(8*np.pi):.6f}; sigma_sca there = {ssc:.6f} (= sigma_abs)")

    print("\nC. Andersen inequality over 4000 random lossy spheres (x = 0.05-15)")
    res = andersen_study()
    print(f"   {'length unit':22s} {'max printed LHS/RHS':>20} {'min printed':>12} {'max corrected':>14} {'max sabs/(l^2 D/4pi)':>21}")
    for unit, r in res.items():
        print(f"   {unit:22s} {r['printed'].max():20.4e} {r['printed'].min():12.4e} {r['corrected'].max():14.8f}"
              f" {r['eff_area'].max():21.6f}")
    r = res["lambda"]
    i = np.argmax(r["corrected"])
    print(f"   corrected form: equality when S(0) is real; best case |arg S(0)| = {abs(r['phase'][i]):.2e},"
          f" ratio {r['corrected'][i]:.8f} (m = {r['m'][i]:.3f}, x = {r['x'][i]:.3f})")
    print("   -> printed form: LHS/RHS scales as (length unit)^1, so it is dimensionally inconsistent;")
    print("      the optical-theorem form sigma_abs + sigma_sca <= lambda sqrt(sigma_sca D_s/pi) holds for all spheres.")

    # a plasmonic example in SI
    lam = 500e-9
    xs = 2 * np.pi * 30e-9 / lam
    ms = np.sqrt(-2.2 + 0.3j)
    ext, sca, ab, D, S0 = andersen_quantities(np.array([ms]), np.array([xs]), lam)
    print(f"\n   Example: 30-nm-radius sphere, eps = -2.2 + 0.3i, lambda = 500 nm: sigma_ext = {ext[0]*1e18:.0f} nm^2,"
          f" sigma_sca = {sca[0]*1e18:.0f} nm^2, D_s = {D[0]:.3f}")
    print(f"      printed RHS sqrt(sigma_sca D/pi) = {np.sqrt(sca[0]*D[0]/np.pi)*1e9:.1f} (in nm, if lengths in nm)"
          f" -> 'violated' in nm, 'satisfied' in m ({np.sqrt(sca[0]*D[0]/np.pi):.2e} m vs {ext[0]:.2e} m^2)")
    print(f"      corrected RHS lambda sqrt(sigma_sca D/pi) = {lam*np.sqrt(sca[0]*D[0]/np.pi)*1e18:.0f} nm^2"
          f" >= sigma_ext = {ext[0]*1e18:.0f} nm^2")

    if make_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.3))
        rhs = r["lam"] * np.sqrt(r["sca"] * r["D"] / np.pi)
        ax[0].loglog(rhs, r["ext"], ".", ms=2)
        lim = [rhs.min(), rhs.max()]
        ax[0].loglog(lim, lim, "k-", lw=1)
        ax[0].set_xlabel(r"$\lambda\sqrt{\sigma_{\rm sca} D_s/\pi}$  ($\lambda = 1$)")
        ax[0].set_ylabel(r"$\sigma_{\rm abs}+\sigma_{\rm sca}$")
        ax[0].set_title("corrected Andersen inequality", fontsize=9)
        ax[1].semilogx(r["x"], r["corrected"], ".", ms=2)
        ax[1].axhline(1, color="k", lw=1)
        ax[1].set_xlabel("size parameter $ka$")
        ax[1].set_ylabel(r"LHS / RHS $= \cos\arg S(0)$")
        fig.tight_layout()
        os.makedirs(FIGDIR, exist_ok=True)
        for ext_ in ("pdf", "png"):
            fig.savefig(os.path.join(FIGDIR, f"sec5_4_9_andersen.{ext_}"), dpi=200)
        print(f"\nwrote {os.path.normpath(os.path.join(FIGDIR, 'sec5_4_9_andersen.pdf'))}")


if __name__ == "__main__":
    main()
