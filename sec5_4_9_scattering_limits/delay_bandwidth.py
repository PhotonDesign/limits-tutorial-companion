"""Sec. 5.9 (sec:delay-bandwidth), delay-bandwidth products.  Units: c = 1.

Tutorial claims (tutorial.tex line numbers):
  l.2507  Fourier "uncertainty principle": FWHMs of a pulse in time and frequency satisfy
          Delta t Delta omega >= 2.
  l.2515-2517  single resonator, da/dt = -i w0 a - gamma a + kappa s+: |a|^2 FWHM = 2 gamma,
          amplitude lifetime Delta T = 1/gamma, so Delta T Delta omega = 2 (Delta T = 2Q/w_c).
  l.2519-2524  [eq:Tucker_delay_band]  Delta T Delta omega <= 2 pi (L/lambda_c)(n_max - n_min),
          "assumed that the fractional bandwidth is not too large, Delta omega/omega_c << 2".
  l.2534-2539  [eq:Miller_delay_band] Delta T Delta omega <= (pi/sqrt 3)(L/lambda_c) eta_max;
          "exceeds [Tucker] if eps_max >~ 6".
  l.2557  metalens: Delta omega <= (kappa c/F) sqrt(1-NA^2)/(1 - sqrt(1-NA^2)).
  l.2561  "... achromatic metalens designs is shown in Fig. meta-optic(d)" (caption: panel (b)).

Checks:
  A. Tucker: maximize L Delta k over all straight dispersion segments of width Delta omega
     (centred on omega_c) lying between the light lines k = n_min w/c and n_max w/c (brute-force
     LP over endpoints).  Exact result:
        Delta T Delta omega = 2 pi (L/lambda_c) [ (n_max - n_min) + (n_max + n_min) Delta omega/(2 omega_c) ],
     so dropping the second term needs Delta omega/omega_c << 2 (n_max - n_min)/(n_max + n_min),
     not Delta omega/omega_c << 2.  Without that, the printed '<=' is violated by the linear-
     dispersion construction it is derived from.
  B. FWHM time-bandwidth products of transform-limited pulses (numerical FFT): Gaussian 4 ln2 =
     2.77, sech^2 1.98, one-sided exponential ln2 = 0.69, rectangular 5.57.  There is no universal
     lower bound 2 for FWHM products; the rigorous statement is for rms widths
     (Delta t_rms Delta omega_rms >= 1/2).
  C. Resonator: numerical pulse propagation through (i) a one-port lossless resonator (reflection)
     and (ii) a two-port in-line resonator (transmission).  Amplitude lifetime 1/gamma, |a|^2
     FWHM 2 gamma, measured pulse delays 2/gamma and 1/gamma: the product is 2 with the
     1/e-lifetime definition used in the text, 4 for the one-port group delay.
  D. Miller vs Tucker crossover (n_min = background = 1): ratio (sqrt(eps)+1)/(2 sqrt 3) = 1 at
     eps = (2 sqrt3 - 1)^2 = 6.07.
  E. Metalens: required delay spread (F/c)(1/sqrt(1-NA^2) - 1) reproduces the l.2557 bound.

Run:  python delay_bandwidth.py   (about 2 s; writes figures/sec5_4_9_delay_bandwidth.png/pdf)
"""
import os
import sys
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


# ----------------------------------------------------------------------------------------
# A. Tucker
# ----------------------------------------------------------------------------------------
def tucker_max_lp(nmax, nmin, frac_bw, wc=1.0, L=1.0):
    """max L (k2 - k1) s.t. nmin w_i <= k_i <= nmax w_i, w1,2 = wc (1 -+ frac/2) (straight
    segment between two points of the convex cone lies inside it). Returns Delta T Delta w."""
    w1, w2 = wc * (1 - frac_bw / 2), wc * (1 + frac_bw / 2)
    res = linprog(c=[1.0, -1.0], bounds=[(nmin * w1, nmax * w1), (nmin * w2, nmax * w2)])
    return L * (-res.fun)


def tucker_exact(nmax, nmin, frac_bw, wc=1.0, L=1.0):
    return L * wc * ((nmax - nmin) + (nmax + nmin) * frac_bw / 2)


def tucker_printed(nmax, nmin, wc=1.0, L=1.0):
    return 2 * np.pi * L / (2 * np.pi / wc) * (nmax - nmin)


# ----------------------------------------------------------------------------------------
# B. FWHM products
# ----------------------------------------------------------------------------------------
def fwhm(x, y):
    y = y / y.max()
    above = np.where(y >= 0.5)[0]
    i0, i1 = above[0], above[-1]
    # linear interpolation at both edges
    xl = x[i0] if i0 == 0 else x[i0 - 1] + (0.5 - y[i0 - 1]) * (x[i0] - x[i0 - 1]) / (y[i0] - y[i0 - 1])
    xr = x[i1] if i1 == len(x) - 1 else x[i1] + (y[i1] - 0.5) * (x[i1 + 1] - x[i1]) / (y[i1] - y[i1 + 1])
    return xr - xl


def pulse_products(N=2 ** 22, T=4000.0):
    t = (np.arange(N) - N // 2) * (T / N)
    dt = t[1] - t[0]
    w = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, dt))
    shapes = {
        "Gaussian": np.exp(-t ** 2 / 2),
        "sech (sech^2 intensity)": 2 * np.exp(-np.abs(t)) / (1 + np.exp(-2 * np.abs(t))),
        "one-sided exponential": np.where(t >= 0, np.exp(-np.abs(t) / 2), 0.0),
        "rectangular": np.where(np.abs(t) <= 0.5, 1.0, 0.0),
    }
    out = {}
    for name, E in shapes.items():
        It = abs(E) ** 2
        Ew = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E)))
        Iw = abs(Ew) ** 2
        p_fwhm = fwhm(t, It) * fwhm(w, Iw)
        # rms widths (intensity-weighted std devs); the one-sided exponential and rectangular
        # pulses have power spectra ~ 1/w^2, so their rms bandwidth diverges
        def rms(x, I):
            m = np.sum(x * I) / np.sum(I)
            return np.sqrt(np.sum((x - m) ** 2 * I) / np.sum(I))
        r = rms(t, It) * rms(w, Iw) if name in ("Gaussian", "sech (sech^2 intensity)") else None
        out[name] = (p_fwhm, r)
    return out


# ----------------------------------------------------------------------------------------
# C. resonator delays by pulse propagation
# ----------------------------------------------------------------------------------------
def resonator_delays(gamma=1.0, w0=0.0, sig_w=0.02, N=2 ** 16, T=4000.0):
    t = (np.arange(N) - N // 2) * (T / N)
    dt = t[1] - t[0]
    w = 2 * np.pi * np.fft.fftfreq(N, dt)
    s_in = np.exp(-t ** 2 * sig_w ** 2 / 2)                  # envelope, bandwidth << gamma
    S = np.fft.fft(np.fft.ifftshift(s_in))
    # e^{-i w t}: detuning Omega = w (baseband).  a = kappa s/(gamma - i(w - w0))
    one_port = (gamma + 1j * (w - w0)) / (gamma - 1j * (w - w0))  # lossless, kappa = sqrt(2 gamma)
    two_port = gamma / (gamma - 1j * (w - w0))
    out = {}
    for name, H in (("one-port reflection", one_port), ("two-port transmission", two_port)):
        # physics convention S(w) = int s e^{+i w t} dt; numpy's fft gives S(-w), so the output
        # spectrum in numpy ordering is H(-w) S_np(w) = conj(H(w)) S_np(w) (real impulse response)
        y = np.fft.fftshift(np.fft.ifft(S * np.conj(H)))
        cen = lambda f: np.sum(t * abs(f) ** 2) / np.sum(abs(f) ** 2)
        out[name + " delay"] = cen(y) - cen(s_in)
    # lifetime: free decay of a(t)
    tt = np.linspace(0, 5, 5001)
    a = np.exp(-gamma * tt)
    out["1/e amplitude lifetime"] = tt[np.argmin(abs(a - np.exp(-1)))]
    ww = np.linspace(-5, 5, 200001)
    out["|a|^2 FWHM"] = fwhm(ww, 1 / (ww ** 2 + gamma ** 2))
    return out


def main(make_figure=True):
    print("=" * 90)
    print("Sec. 5.9  delay-bandwidth products (c = 1)")
    print("=" * 90)
    print("A. Tucker bound: Delta T Delta omega in units of 2 pi L/lambda_c")
    print(f"   {'n_max':>5} {'n_min':>5} {'dw/wc':>6} {'LP max':>9} {'exact':>9} {'printed':>9} {'excess':>7}"
          f" {'2(dn)/(n+n)':>11}")
    rows = []
    for nmax, nmin in ((3.5, 1.45), (2.0, 1.0), (1.5, 1.45)):
        for fb in (0.02, 0.2, 0.5):
            lp = tucker_max_lp(nmax, nmin, fb)
            ex = tucker_exact(nmax, nmin, fb)
            pr = tucker_printed(nmax, nmin)
            rows.append((nmax, nmin, fb, lp, ex, pr))
            print(f"   {nmax:5.2f} {nmin:5.2f} {fb:6.2f} {lp:9.4f} {ex:9.4f} {pr:9.4f} {lp/pr-1:7.1%}"
                  f" {2*(nmax-nmin)/(nmax+nmin):11.3f}")
    print("   -> the dropped term (n_max+n_min) dw/(2 wc) is negligible only if dw/wc << 2(n_max-n_min)/(n_max+n_min)")

    print("\nB. FWHM (intensity) time-bandwidth products Delta t Delta omega of transform-limited pulses")
    prods = pulse_products()
    ref = {"Gaussian": 4 * np.log(2), "sech (sech^2 intensity)": 2 * np.pi * 0.3148,
           "one-sided exponential": np.log(2), "rectangular": 2 * np.pi * 0.8859}
    for name, (p, r) in prods.items():
        rs = f"{r:.4f}" if r is not None else "diverges (spectrum ~ 1/w^2)"
        print(f"   {name:26s}: FWHM product {p:.4f}  (tabulated {ref[name]:.4f});  rms product {rs}")
    print("   -> 'Delta t Delta omega >= 2' is not a theorem for FWHMs (one-sided exponential: ln 2 = 0.69);"
          " Gaussian gives 4 ln 2 = 2.77.")

    print("\nC. Single resonator (gamma = 1), pulse bandwidth << gamma")
    rd = resonator_delays()
    for k, v in rd.items():
        print(f"   {k:24s}: {v:.4f}")
    print("   -> 1/e lifetime x FWHM = 2 (text's definition); one-port group delay x FWHM = 4;"
          " two-port transmission delay x FWHM = 2")

    print("\nD. Miller [eq:Miller_delay_band] vs Tucker [eq:Tucker_delay_band], n_min = n_b = 1")
    eps = np.array([2.0, 4.0, 6.0, 6.07, 8.0, 12.0])
    ratio = (np.pi / np.sqrt(3)) * (eps - 1) / (2 * np.pi * (np.sqrt(eps) - 1))
    for e, r in zip(eps, ratio):
        print(f"   eps_max = {e:5.2f}: Miller/Tucker = {r:.4f}")
    print(f"   crossover eps = (2 sqrt3 - 1)^2 = {(2*np.sqrt(3)-1)**2:.4f}  (text: ~6)")

    print("\nE. Metalens bound l.2557: required delay spread Delta T = (F/c)(1/sqrt(1-NA^2) - 1)")
    F, kap = 1.0, 3.0
    for NA in (0.1, 0.5, 0.9):
        dT = F * (1 / np.sqrt(1 - NA ** 2) - 1)
        fmla = kap / F * np.sqrt(1 - NA ** 2) / (1 - np.sqrt(1 - NA ** 2))
        print(f"   NA = {NA}: kappa/Delta T = {kap/dT:.6f}, formula = {fmla:.6f}")

    if make_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
        fb = np.linspace(0, 0.6, 61)
        for (nmax, nmin), c in zip(((3.5, 1.45), (2.0, 1.0), (1.5, 1.45)), ("C0", "C1", "C2")):
            ax[0].plot(fb, tucker_exact(nmax, nmin, fb) / tucker_printed(nmax, nmin), c + "-",
                       label=f"$n$ = {nmax}/{nmin}")
        ax[0].axhline(1, color="k", lw=1, ls=":")
        ax[0].set_xlabel(r"$\Delta\omega/\omega_c$")
        ax[0].set_ylabel("max $\\Delta T\\Delta\\omega$ / [eq:Tucker]")
        ax[0].set_yscale("log")
        ax[0].legend(fontsize=7, frameon=False)
        names = list(prods)
        ax[1].bar(range(len(names)), [prods[n][0] for n in names], color="C0")
        ax[1].axhline(2, color="C3", ls="--", lw=1, label="'>= 2' (l.2507)")
        ax[1].set_xticks(range(len(names)))
        ax[1].set_xticklabels(["Gauss", "sech$^2$", "1-sided exp", "rect"], fontsize=8)
        ax[1].set_ylabel(r"FWHM $\Delta t\,\Delta\omega$")
        ax[1].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        os.makedirs(FIGDIR, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIGDIR, f"sec5_4_9_delay_bandwidth.{ext}"), dpi=200)
        print(f"\nwrote {os.path.normpath(os.path.join(FIGDIR, 'sec5_4_9_delay_bandwidth.pdf'))}")


if __name__ == "__main__":
    main()
