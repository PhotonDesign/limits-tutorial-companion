"""Label-light replacement panels for the tutorial's focusing figures (for assembly in
PowerPoint).  Writes vector PDFs plus 300-dpi PNGs to figures/tutorial_panels/.

  PWFocusing_d_corrected : 2pi power-optimal focal-plane maps Re E_x, Im E_z, |E|^2
                           (the published panel draws E_z and |E|^2 rotationally symmetric;
                           the correct E_z ∝ cos(phi) is antisymmetric in x and |E|^2 is elliptical)
  PWFocusing_e_corrected : left, radial intensity cuts incl. the 2pi cut perpendicular to the
                           polarization; right, FWHM vs NA along (solid) and perpendicular to
                           (dashed) the polarization
  FocusingFourPi_left    : cuts for the FocusingFourPi comparison figure, same color code as
                           the published figure (paraxial blue, 2pi red, 4pi black)

Run: python tutorial_figure_panels.py   (~15 s)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tutorial_tools.focusing import (field_2pi_power_closed, intensity_paraxial,
                                     intensity_2pi_tutorial, intensity_4pi_perp, intensity_4pi_par)
from focal_plane_fields import fwhm_vs_na

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "tutorial_panels")
K = 2 * np.pi


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")


def panel_d(plt):
    xv = np.linspace(-1, 1, 301)
    X, Y = np.meshgrid(xv, xv, indexing="xy")
    Ex, Ey, Ez = field_2pi_power_closed(np.hypot(X, Y), np.arctan2(Y, X), k=K)
    E0 = (4 * np.pi / 3) * K**2
    I = (np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2) / E0**2
    fig, ax = plt.subplots(1, 3, figsize=(9, 2.8))
    for a, F, cm, lim in zip(ax, [Ex.real / E0, Ez.imag / E0, I],
                             ["RdBu", "RdBu", "hot_r"], [(-1, 1), (-1, 1), (0, 1)]):
        im = a.imshow(F, extent=(-1, 1, -1, 1), origin="lower", cmap=cm, vmin=lim[0], vmax=lim[1])
        a.set_axis_off()
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.03, ticks=[lim[0], lim[1]])
    fig.tight_layout()
    save(fig, "PWFocusing_d_corrected")
    # same, but Im E_z on its own color scale (it is weak on the E_x scale)
    fig, a = plt.subplots(figsize=(3, 2.8))
    m = np.abs(Ez.imag / E0).max()
    im = a.imshow(Ez.imag / E0, extent=(-1, 1, -1, 1), origin="lower", cmap="RdBu", vmin=-m, vmax=m)
    a.set_axis_off()
    plt.colorbar(im, ax=a, fraction=0.046, pad=0.03, ticks=[-round(m, 2), round(m, 2)])
    save(fig, "PWFocusing_d_ImEz_ownscale")


def panel_e(plt, s_list):
    r = np.linspace(-1, 1, 1001)
    u = 2 * np.pi * np.abs(r)
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 2.9))
    ax[0].plot(r, intensity_paraxial(u), color="#1f3fd1")
    ax[0].plot(r, intensity_2pi_tutorial(u), color="#e0301e")
    ax[0].plot(r, intensity_4pi_perp(u), color="#e0301e", ls="--")      # 2pi, perpendicular cut
    ax[0].plot(r, intensity_4pi_perp(u), color="k", ls=(0, (1, 2.5)), lw=1.6)
    ax[0].set_xlim(-1, 1); ax[0].set_ylim(0, 1.02)
    fw = {(n, d, c): fwhm_vs_na(s_list, n, d, component=c)
          for n in ("power", "amplitude") for d in ("x", "y") for c in ("all", "par") if not (d == "y" and c == "par")}
    for n, col in [("power", "#1f3fd1"), ("amplitude", "#e0301e")]:
        ax[1].plot(s_list, fw[(n, "x", "all")], color=col)
        ax[1].plot(s_list, fw[(n, "x", "par")], color=col, ls=":")
        ax[1].plot(s_list, fw[(n, "y", "all")], color=col, ls="--")
    ax[1].set_xlim(0, 1)
    fig.tight_layout()
    save(fig, "PWFocusing_e_corrected")
    return fw


def fourpi_left(plt):
    r = np.linspace(-1, 1, 1001)
    u = 2 * np.pi * np.abs(r)
    fig, a = plt.subplots(figsize=(3.4, 2.6))
    a.plot(r, intensity_paraxial(u), color="#1f3fd1")
    a.plot(r, intensity_2pi_tutorial(u), color="#e0301e")
    a.plot(r, intensity_4pi_perp(u), color="#e0301e", ls="--")
    a.plot(r, intensity_4pi_par(u), color="k", ls=":")
    a.plot(r, intensity_4pi_perp(u), color="k", lw=0.9)
    a.set_xlim(-1, 1); a.set_ylim(0, 1.02)
    fig.tight_layout()
    save(fig, "FocusingFourPi_left")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s_list = np.concatenate([np.linspace(0.02, 0.9, 23), np.linspace(0.91, 1.0, 10)])
    panel_d(plt)
    panel_e(plt, s_list)
    fourpi_left(plt)
    print(f"wrote panels to {os.path.normpath(OUT)}")
    print("line code: blue = paraxial / power norm., red = 2pi / amplitude norm., black = 4pi;")
    print("  solid = along the polarization axis, dashed = perpendicular to it (in the focal plane),")
    print("  dotted (right panel) = |E_par|^2 only; black dotted (4pi left) = along the polarization axis.")


if __name__ == "__main__":
    main()
