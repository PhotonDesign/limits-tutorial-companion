"""Secs. 3.1.2-3.1.3: optimal focused fields across the focal plane, and their widths.

Three independent evaluations of the power-optimal field for target polarization xhat:
  (i)   brute-force 2D quadrature over the cap of plane-wave directions, with explicit
        TE/TM polarization vectors and the optimal coefficients of Eq. (alphaOpt);
  (ii)  1D quadrature after doing the azimuthal integrals analytically (Hansen-Bessel);
  (iii) for 2pi solid angle, the closed form derived in derivations.pdf, Sec. 2:
          E_x = 2 pi k^2 [ sin^2(phi) j0(a) + (3cos^2(phi) - 1) j1(a)/a ]
          E_y = 2 pi k^2 sin(phi) cos(phi) j2(a)
          E_z = -2 pi i k^2 cos(phi) J2(a)/a ,        a = k rho.
Findings reproduced here (see derivations.pdf / VALIDATION.md):
  * Tutorial Eq. (E2TwoPi), 4 (j1/a)^2 + (J2/a)^2, is the intensity along the
    polarization axis (phi = 0) only.  The focal spot is elliptical; perpendicular to the
    polarization (phi = pi/2) the intensity is (j0 - j1/a)^2, identical to the 4pi
    optimum of Eq. (E2optFourPi).
  * E_z is proportional to cos(phi) (antisymmetric in x), not rotationally symmetric.
  * In the focal plane, the in-plane field of the 2pi optimum is exactly one half of the
    4pi optimum (the lower hemisphere contributes an identical E_par and an opposite E_z).

Run: python focal_plane_fields.py    (~20 s; writes figures/focal_plane_*.pdf/png)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tutorial_tools.angular_spectrum import cap_quadrature, optimal_coefficients, field
from tutorial_tools.focusing import (optimal_field_xpol, field_2pi_power_closed,
                                     intensity_paraxial, intensity_2pi_tutorial,
                                     intensity_4pi_perp, intensity_4pi_par,
                                     half_max_radius, first_minimum)

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
K = 2 * np.pi  # wavelength = 1


def intensity_fn(theta, normalization, hemispheres, direction, n=200, component="all"):
    """Returns f(r) = |E|^2 along a line through the focus ('x' = polarization axis,
    'y' = in-plane perpendicular, 'z' = optical axis), r in wavelengths."""
    def f(r):
        r = np.atleast_1d(np.asarray(r, float))
        zero = np.zeros_like(r)
        if direction == "x":
            E = optimal_field_xpol(r, zero, zero, theta, normalization, hemispheres, k=K, n=n)
        elif direction == "y":
            E = optimal_field_xpol(r, zero + np.pi / 2, zero, theta, normalization, hemispheres, k=K, n=n)
        else:
            E = optimal_field_xpol(zero, zero, r, theta, normalization, hemispheres, k=K, n=n)
        comps = E if component == "all" else E[:2]
        return sum(np.abs(c)**2 for c in comps)
    return f


def check_three_ways(n_pts=12, seed=3):
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0, 3, n_pts)
    phi = rng.uniform(0, 2 * np.pi, n_pts)
    z = np.zeros(n_pts)
    pts = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], 1)
    th = np.pi / 2
    kh, _, _, W = cap_quadrature(th, 300, 256)
    c, vec = optimal_coefficients(kh, np.array([1, 0, 0], complex), "power")
    E_brute = field(pts, kh, W, vec, k=K)
    E_semi = np.stack(optimal_field_xpol(rho, phi, z, th, "power", "one", k=K), 1)
    E_closed = np.stack(field_2pi_power_closed(rho, phi, k=K), 1)
    scale = np.abs(E_semi).max()
    return (np.abs(E_brute - E_semi).max() / scale, np.abs(E_closed - E_semi).max() / scale)


def check_half_of_4pi(n_pts=12, seed=4):
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0, 3, n_pts)
    phi = rng.uniform(0, 2 * np.pi, n_pts)
    z = np.zeros(n_pts)
    E2 = np.stack(optimal_field_xpol(rho, phi, z, np.pi / 2, "power", "one", k=K), 1)
    E4 = np.stack(optimal_field_xpol(rho, phi, z, np.pi / 2, "power", "both", k=K), 1)
    return np.abs(E2[:, :2] - 0.5 * E4[:, :2]).max() / np.abs(E4).max(), np.abs(E4[:, 2]).max() / np.abs(E4).max()


def width_table():
    rows = []
    th2 = np.pi / 2
    cases = [("2pi power", th2, "power", "one"), ("2pi amplitude", th2, "amplitude", "one"),
             ("4pi power", th2, "power", "both")]
    for label, th, norm, hem in cases:
        for d in ("x", "y", "z"):
            f = intensity_fn(th, norm, hem, d)
            xmax = 3.0 if d == "z" else 1.5
            hm = half_max_radius(f, xmax, n=301)
            try:
                xm, fm = first_minimum(f, xmax, n=301)
            except Exception:
                xm, fm = np.nan, np.nan
            rows.append((label, d, 2 * hm, xm, fm))
    return rows


def fwhm_vs_na(sin_list, normalization, direction, component="all"):
    out = []
    for s in sin_list:
        th = np.arcsin(s)
        f = intensity_fn(th, normalization, "one", direction, component=component)
        out.append(2 * half_max_radius(f, 1.5 / s, n=151) * s)   # in units of lambda/sin(theta)
    return np.array(out)


def main(make_figure=True):
    print("=" * 78)
    print("Focal-plane fields of the optimal 2pi / 4pi beams (target polarization xhat)")
    print("=" * 78)
    e1, e2 = check_three_ways()
    print(f"2pi power: brute-force (explicit polarizations) vs 1D-quadrature: rel. err {e1:.1e}")
    print(f"2pi power: new closed form (with phi dependence) vs 1D-quadrature: rel. err {e2:.1e}")
    h1, h2 = check_half_of_4pi()
    print(f"z = 0 plane: |E_par(2pi) - E_par(4pi)/2| / |E| = {h1:.1e};  |E_z(4pi)|/|E| = {h2:.1e}")

    a = np.linspace(1e-3, 8, 400)
    Ix = sum(np.abs(c)**2 for c in field_2pi_power_closed(a / K, 0 * a, k=K))
    Iy = sum(np.abs(c)**2 for c in field_2pi_power_closed(a / K, 0 * a + np.pi / 2, k=K))
    Ix /= Ix[0]; Iy /= Iy[0]
    print(f"Tutorial Eq. (E2TwoPi) vs true 2pi intensity: along phi=0 max diff {np.abs(Ix - intensity_2pi_tutorial(a)).max():.1e};"
          f" along phi=pi/2 max diff {np.abs(Iy - intensity_2pi_tutorial(a)).max():.2f}")
    print(f"True 2pi intensity along phi=pi/2 vs 4pi Eq. (E2optFourPi): max diff {np.abs(Iy - intensity_4pi_perp(a)).max():.1e}")

    print("\nWidths (lambda = 1).  FWHM = full width at half max of |E|^2; 'first min' = first local")
    print("minimum of |E|^2 (a true zero only if I/I0 ~ 0):")
    print(f"  {'case':16s} {'axis':>4s} {'FWHM':>8s} {'first min':>10s} {'I/I0 there':>11s}")
    print(f"  {'paraxial (Airy)':16s} {'any':>4s} {2*1.6163399/(2*np.pi):8.4f} {3.8317060/(2*np.pi):10.4f} {0:11.1e}   (x lambda/sin theta)")
    for label, d, fw, xm, fm in width_table():
        print(f"  {label:16s} {d:>4s} {fw:8.4f} {xm:10.4f} {fm:11.1e}")
    print("  (tutorial FocusingFourPi table: 2pi 0.63 / 0.77 lambda; 4pi 0.40 / 0.44 lambda)")

    s_list = np.concatenate([np.linspace(0.05, 0.9, 18), np.linspace(0.92, 1.0, 9)])
    fw = {}
    for norm in ("power", "amplitude"):
        for d in ("x", "y"):
            fw[(norm, d)] = fwhm_vs_na(s_list, norm, d)
        fw[(norm, "x", "par")] = fwhm_vs_na(s_list, norm, "x", component="par")
    print("\nFWHM (units lambda/sin theta) vs NA [along x = polarization axis; along y]:")
    for i, s in enumerate(s_list):
        if i % 3 == 0 or s > 0.95:
            print(f"  sin th = {s:5.3f}: power x {fw[('power','x')][i]:.4f}  y {fw[('power','y')][i]:.4f} |"
                  f" amplitude x {fw[('amplitude','x')][i]:.4f}  y {fw[('amplitude','y')][i]:.4f}")

    if make_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIGDIR, exist_ok=True)
        # --- (1) corrected focal-plane maps (cf. tutorial Fig. PWFocusing(d)) ---
        xv = np.linspace(-1, 1, 201)
        X, Y = np.meshgrid(xv, xv, indexing="xy")
        R, PHI = np.hypot(X, Y), np.arctan2(Y, X)
        Ex, Ey, Ez = field_2pi_power_closed(R, PHI, k=K)
        I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
        E0 = Ex[100, 100].real
        fig, ax = plt.subplots(1, 4, figsize=(11, 2.9))
        panels = [(Ex.real / E0, "RdBu_r", (-1, 1), r"Re $E_x$"), (Ez.imag / E0, "RdBu_r", (-0.4, 0.4), r"Im $E_z$"),
                  (np.abs(Ey) / E0, "viridis", (0, float(np.abs(Ey).max() / E0)), r"$|E_y|$"), (I / I.max(), "hot_r", (0, 1), r"$|\mathbf{E}|^2$")]
        for a_, (F, cm, lim, ttl) in zip(ax, panels):
            im = a_.imshow(F, extent=(-1, 1, -1, 1), origin="lower", cmap=cm, vmin=lim[0], vmax=lim[1])
            a_.contour(X, Y, I / I.max(), levels=[0.5], colors="k", linewidths=0.8) if ttl.startswith("$|\\mathbf") else None
            a_.set_title(ttl)
            a_.set_xlabel(r"$x/\lambda$")
            plt.colorbar(im, ax=a_, fraction=0.046)
        ax[0].set_ylabel(r"$y/\lambda$")
        fig.suptitle(r"$2\pi$ power-optimal focus, $\hat{\mathbf{x}}$ target: focal plane $z=0$ (half-max contour on $|E|^2$)", fontsize=10)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIGDIR, f"focal_plane_maps_2pi.{ext}"), dpi=200)
        # --- (2) line cuts ---
        r = np.linspace(-1, 1, 801)
        fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
        u = 2 * np.pi * r
        ax[0].plot(r, intensity_paraxial(np.abs(u)), "C0", label=r"paraxial (Airy), $\rho\sin\theta/\lambda$")
        ax[0].plot(r, intensity_2pi_tutorial(np.abs(u)), "C3", label=r"$2\pi$, along $x$ (pol. axis) = Eq. E2TwoPi")
        ax[0].plot(r, intensity_4pi_perp(np.abs(u)), "k", label=r"$4\pi$, $\perp$ pol. = Eq. E2optFourPi")
        ax[0].plot(r, intensity_4pi_perp(np.abs(u)), "C1--", label=r"$2\pi$, along $y$ ($\perp$ pol.) [identical]")
        ax[0].plot(r, intensity_4pi_par(np.abs(u)), "C2:", lw=2, label=r"$4\pi$, along $x$ (pol. axis)")
        ax[0].set_xlabel(r"radial position ($\lambda$)")
        ax[0].set_ylabel(r"$|\mathbf{E}|^2$ (normalized)")
        ax[0].legend(fontsize=6, frameon=False, loc="upper left")
        ax[0].set_xlim(-1, 1)
        ax[0].set_ylim(0, 1.45)
        ax[1].plot(s_list, fw[("power", "x")], "C0-", label="power norm, along x")
        ax[1].plot(s_list, fw[("power", "x", "par")], "C0:", label=r"power norm, along x, $|E_\parallel|^2$")
        ax[1].plot(s_list, fw[("power", "y")], "C0--", label="power norm, along y")
        ax[1].plot(s_list, fw[("amplitude", "x")], "C3-", label="amplitude norm, along x")
        ax[1].plot(s_list, fw[("amplitude", "x", "par")], "C3:", label=r"amplitude norm, along x, $|E_\parallel|^2$")
        ax[1].plot(s_list, fw[("amplitude", "y")], "C3--", label="amplitude norm, along y")
        ax[1].axhline(2 * 1.6163399 / (2 * np.pi), color="gray", lw=0.6)
        ax[1].set_xlabel(r"numerical aperture $\sin\theta$")
        ax[1].set_ylabel(r"FWHM $\times\sin\theta/\lambda$")
        ax[1].legend(fontsize=6.5, frameon=False)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIGDIR, f"focal_plane_profiles.{ext}"), dpi=200)
        print(f"\nwrote figures/focal_plane_maps_2pi.* and figures/focal_plane_profiles.*")


if __name__ == "__main__":
    main()
