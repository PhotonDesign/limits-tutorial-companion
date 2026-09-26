"""Sec. 3.1.2: maximum focal-point intensity from plane waves within a cone of
half-angle theta (numerical aperture sin theta), for power and amplitude normalization.

Checks (tutorial equation labels in brackets):
  1. The 3x3 matrix M = int d^2k_par w(k) sum_p e_p e_p^dag, built with EXPLICIT TE/TM
     polarization vectors (no use of the identity sum_p e_p e_p^dag = I - khat khat),
     is diagonal; its eigenvalues reproduce the closed forms
        power:     x,y: 4/3 - cos - cos^3/3        z: 2(2/3 - cos + cos^3/3)   [OptPowerNorm]
        amplitude: x,y: s^2 (1 - s^2/4)            z: s^4/2                    [OptAmpNorm]
     in units of k^2/(2 pi c) for eps0 |E(0)|^2 / P.  Hence in-plane polarization is the
     global optimum, with (x - z) gap = cos(theta) sin^2(theta) (power normalization).
  2. Brute force: a random discretization of the cap (random directions, random
     polarization basis rotation) with the Rayleigh-quotient maximum lambda_max(A^dag A)
     computed by an eigensolver agrees with the closed forms.
  3. Limits: paraxial -> theta^2; 2pi -> 4/3 (power) i.e. U/P = k^2/(6 pi c) [TwoPiBound];
     power/amplitude ratio at 2pi = 16/9.
  4. NEW: the amplitude-optimal beam measured per unit *physical* power delivers
     s^4 (1 - s^2/4)^2 / [2 (4/15 - c^3/6 - c^5/10)], = 135/128 at 2pi, so the
     power-optimal beam beats it by a factor 1.264 (not 16/9) at equal physical power.

Run:  python focal_intensity_bounds.py    (about 5 s; writes figures/focal_intensity_bounds.pdf/png)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tutorial_tools.angular_spectrum import (cap_quadrature, polarization_basis,
                                             focal_matrix, optimal_coefficients,
                                             field, power_z, amplitude_norm)
from tutorial_tools.focusing import (bound_power_inplane, bound_power_z,
                                     bound_amp_inplane, bound_amp_z,
                                     amp_optimal_per_true_power)

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def closed_forms(theta):
    return dict(power_x=bound_power_inplane(theta), power_z=bound_power_z(theta),
                amp_x=bound_amp_inplane(theta), amp_z=bound_amp_z(theta))


def check_matrix(theta):
    """Eigen-decomposition of the explicit-polarization focal matrices.
    Returns dict of normalized eigenvalues (units k^2/(2 pi c)) and off-diagonal size."""
    out = {}
    for norm in ("power", "amplitude"):
        M = focal_matrix(theta, norm) / np.pi          # -> units of k^2/(2 pi c)
        out[norm + "_eig"] = np.sort(np.linalg.eigvalsh(M))[::-1]
        out[norm + "_offdiag"] = np.abs(M - np.diag(np.diag(M))).max()
        out[norm + "_diag"] = np.real(np.diag(M))
    return out


def brute_force_rayleigh(theta, normalization, n_dirs=60000, seed=0):
    """Random point cloud on the cap (Monte-Carlo quadrature), random rotation of the
    polarization basis at each direction, then lambda_max of the 3x3 matrix A A^dag
    where A maps normalized coefficients to E(0).  Returns (2 pi c/k^2) eps0|E|^2/P."""
    rng = np.random.default_rng(seed)
    # uniform sampling of the cap in solid angle
    u = rng.uniform(np.cos(theta), 1.0, n_dirs)
    ph = rng.uniform(0, 2 * np.pi, n_dirs)
    s = np.sqrt(1 - u**2)
    khat = np.stack([s * np.cos(ph), s * np.sin(ph), u], 1)
    area = 2 * np.pi * (1 - np.cos(theta))
    w = np.full(n_dirs, area / n_dirs)                 # dOmega weights
    E = polarization_basis(khat)
    rot = rng.uniform(0, 2 * np.pi, n_dirs)            # arbitrary basis rotation
    e1 = np.cos(rot)[:, None] * E[:, 0] + np.sin(rot)[:, None] * E[:, 1]
    e2 = -np.sin(rot)[:, None] * E[:, 0] + np.cos(rot)[:, None] * E[:, 1]
    basis = np.concatenate([e1, e2], 0)                # 2N x 3
    dk2 = np.concatenate([w * u, w * u])               # d^2k_par = cos(theta) dOmega (k=1)
    kz = np.concatenate([u, u])
    # E(0) = sum_j c_j e_j dk2_j.  Power P = 2 pi^2 sum_j dk2_j kz_j |c_j|^2 (Z0 = 1).
    # Amplitude "power" = 2 pi^2 sum_j dk2_j |c_j|^2.  Normalize c_j = a_j / sqrt(2 pi^2 dk2_j g_j)
    g = kz if normalization == "power" else np.ones_like(kz)
    A = (basis * (dk2 / np.sqrt(2 * np.pi**2 * dk2 * g))[:, None]).T   # 3 x 2N
    lam = np.linalg.eigvalsh(A @ A.conj().T).max()     # max eps0|E(0)|^2/P (c = eps0 = 1)
    return lam * 2 * np.pi                              # -> units k^2/(2 pi c)


def check_true_power_of_amp_optimum(theta):
    """Synthesize the amplitude-optimal coefficients explicitly and compute
    (2 pi c/k^2) eps0 |E_x(0)|^2 / P with P the physical z-power."""
    khat, TH, PH, W = cap_quadrature(theta, 400, 256)
    c, vec = optimal_coefficients(khat, np.array([1, 0, 0], complex), "amplitude")
    E0 = field(np.zeros((1, 3)), khat, W, vec)[0]
    P = power_z(khat, W, c)
    Pamp = amplitude_norm(khat, W, c)
    return 2 * np.pi * abs(E0[0])**2 / P, 2 * np.pi * abs(E0[0])**2 / Pamp


def power_law_beam_ratio(n):
    """(2 pi c/k^2) eps0|E(0)|^2/P at 2pi for c_p ∝ (k_z/k)^n (xhat . e_p)^* (closed form):
    [1/(n+2) + 1/(n+4)]^2 / [1/(2n+3) + 1/(2n+5)].  n = -1 power-optimal (4/3), n = 0
    amplitude-optimal (135/128), n = 1/2 Dias & Garcia de Abajo's 'diffraction-limited
    Gaussian' reference beam.  (An aplanatic lens is NOT a member of this family: its TM
    part lacks the cos(theta) of (I - khat khat) xhat; see aplanatic_per_power.)"""
    return (1 / (n + 2) + 1 / (n + 4))**2 / (1 / (2 * n + 3) + 1 / (2 * n + 5))


def power_law_beam_numeric(n, theta=np.pi / 2):
    khat, TH, PH, W = cap_quadrature(theta, 400, 256)
    c, vec = optimal_coefficients(khat, np.array([1, 0, 0], complex), "amplitude")
    w = np.abs(khat[:, 2])**n
    c, vec = c * w[:, None], vec * w[:, None]
    E0 = field(np.zeros((1, 3)), khat, W, vec)[0]
    return 2 * np.pi * abs(E0[0])**2 / power_z(khat, W, c)


def aplanatic_per_power(theta):
    """Aplanatic lens (sine condition), uniformly filled x-polarized pupil (Richards-Wolf):
    E_inf ∝ sqrt(cos th) (cos ph e_th - sin ph e_ph).  Closed form
    (2 pi c/k^2) eps0|E(0)|^2/P = I_a^2 / sin^2(theta),
    I_a = (2/3)(1 - cos^{3/2}) + (2/5)(1 - cos^{5/2})  ->  (16/15)^2 = 1.1378 at 2pi."""
    c = np.cos(theta)
    Ia = 2 / 3 * (1 - c**1.5) + 2 / 5 * (1 - c**2.5)
    return Ia**2 / np.sin(theta)**2


def aplanatic_numeric(theta):
    """Explicit-vector synthesis of the aplanatic-lens focus (independent of the closed form)."""
    khat, TH, PH, W = cap_quadrature(theta, 400, 256)
    eth = np.stack([np.cos(TH) * np.cos(PH), np.cos(TH) * np.sin(PH), -np.sin(TH)], 1)
    eph = np.stack([-np.sin(PH), np.cos(PH), 0 * PH], 1)
    vec = (np.cos(PH)[:, None] * eth - np.sin(PH)[:, None] * eph) * (np.cos(TH)**-0.5)[:, None]
    c = np.einsum("nj,npj->np", vec, np.conj(polarization_basis(khat)))
    E0 = field(np.zeros((1, 3)), khat, W, vec)[0]
    return 2 * np.pi * abs(E0[0])**2 / power_z(khat, W, c)


def main(make_figure=True):
    print("=" * 78)
    print("Sec. 3.1.2  focal-point intensity maxima, units k^2/(2 pi c)")
    print("=" * 78)
    hdr = f"{'sin th':>7} | {'power x':>10} {'power z':>10} | {'amp x':>10} {'amp z':>10} | {'max|quad-closed|':>16} {'offdiag':>8}"
    print(hdr)
    for s in [0.05, 0.3, 0.5, 0.8, 0.95, 1.0]:
        th = np.arcsin(s)
        cf = closed_forms(th)
        mc = check_matrix(th)
        err = max(abs(mc["power_diag"][0] - cf["power_x"]), abs(mc["power_diag"][2] - cf["power_z"]),
                  abs(mc["amplitude_diag"][0] - cf["amp_x"]), abs(mc["amplitude_diag"][2] - cf["amp_z"]))
        print(f"{s:7.2f} | {cf['power_x']:10.6f} {cf['power_z']:10.6f} | {cf['amp_x']:10.6f} {cf['amp_z']:10.6f} |"
              f" {err:16.2e} {max(mc['power_offdiag'], mc['amplitude_offdiag']):8.1e}")
    print("\nGap (power): x - z = cos(th) sin^2(th) >= 0;  (amplitude): s^2 - 3 s^4/4 >= 0")
    th = np.linspace(1e-3, np.pi / 2, 200)
    gap = bound_power_inplane(th) - bound_power_z(th) - np.cos(th) * np.sin(th)**2
    print(f"  max |(x - z) - cos sin^2| over theta grid: {np.abs(gap).max():.1e}")

    print("\nMonte-Carlo brute force (random directions + random polarization bases):")
    for s in [0.5, 0.8, 1.0]:
        thv = np.arcsin(s)
        bp = brute_force_rayleigh(thv, "power", seed=1)
        ba = brute_force_rayleigh(thv, "amplitude", seed=2)
        print(f"  sin th = {s:.1f}: power {bp:.4f} (closed {bound_power_inplane(thv):.4f});"
              f" amplitude {ba:.4f} (closed {bound_amp_inplane(thv):.4f})  [MC error ~1/sqrt(N)]")

    print("\nLimits:")
    t = 1e-3
    print(f"  paraxial: power/th^2 = {bound_power_inplane(t)/t**2:.6f}, amp/th^2 = {bound_amp_inplane(t)/t**2:.6f}  (both -> 1)")
    print(f"  2pi: power = {bound_power_inplane(np.pi/2):.6f} (4/3) -> U/P = k^2/(6 pi c);"
          f" amp = {bound_amp_inplane(np.pi/2):.6f} (3/4); ratio = {bound_power_inplane(np.pi/2)/bound_amp_inplane(np.pi/2):.6f} (16/9)")
    print(f"  2pi, power normalization: z-pol = {bound_power_z(np.pi/2):.6f} = in-plane (all polarizations equal)")
    print(f"  2pi, amplitude normalization: z-pol = {bound_amp_z(np.pi/2):.6f} != in-plane 0.75")
    print(f"  sin th = 0.8: power in-plane = {bound_power_inplane(np.arcsin(0.8)):.4f}  (cf. '0.66' in the FocusingFourPi table)")

    print("\nAmplitude-optimal beam measured per PHYSICAL power (new):")
    for s in [0.3, 0.8, 1.0]:
        thv = np.arcsin(s)
        true_ratio, amp_ratio = check_true_power_of_amp_optimum(thv)
        print(f"  sin th = {s:.1f}: per true power {true_ratio:.6f} (closed {amp_optimal_per_true_power(thv):.6f});"
              f" per amplitude norm {amp_ratio:.6f} (closed {bound_amp_inplane(thv):.6f})")
    print(f"  at 2pi: 135/128 = {135/128:.6f}; power-optimal advantage at equal physical power = "
          f"{bound_power_inplane(np.pi/2)/(135/128):.4f} (vs 16/9 = {16/9:.4f} across different normalizations)")

    print("\nReference beams c_p ∝ (k_z/k)^n (x.e_p)^* at 2pi (per physical power):")
    for n, name in [(-1, "power-optimal"), (-0.5, "family member n = -1/2"), (0, "amplitude-optimal"),
                    (0.5, "Dias2020 'Gaussian' reference"), (1, "c ∝ k_z/k")]:
        r_cf, r_num = power_law_beam_ratio(n), power_law_beam_numeric(n)
        print(f"  n = {n:+.1f} ({name:30s}): {r_cf:.6f} (numeric {r_num:.6f});  optimum / this = {(4/3)/r_cf:.4f}")
    for s_ in (0.8, 1.0):
        thv = np.arcsin(s_)
        a_cf, a_num = aplanatic_per_power(thv), aplanatic_numeric(thv)
        print(f"  aplanatic lens, uniform pupil, sin th = {s_:.1f}: {a_cf:.6f} (numeric {a_num:.6f});"
              f" optimum / this = {bound_power_inplane(thv)/a_cf:.4f}")

    khat, TH, PH, W = cap_quadrature(np.pi / 2, 300, 256)
    c, vec = optimal_coefficients(khat, np.array([1, 0, 0], complex), "power")
    vecH = np.einsum("np,npj->nj", c, np.cross(khat[:, None, :], polarization_basis(khat)))
    E0 = field(np.zeros((1, 3)), khat, W, vec)[0]
    H0 = field(np.zeros((1, 3)), khat, W, vecH)[0]
    print(f"\n2pi power-optimal focus: Z0|H(0)|/|E(0)| = {np.linalg.norm(H0)/np.linalg.norm(E0):.6f} (3/4); "
          f"H(0) = 0 at 4pi (odd in k_z)")

    if make_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        s = np.linspace(1e-3, 1, 400)
        th = np.arcsin(s)
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
        for a, div in zip(ax, [np.ones_like(s), s**2]):
            a.plot(s, bound_power_inplane(th) / div, "C0-", label="power norm, in-plane")
            a.plot(s, bound_power_z(th) / div, "C0--", label="power norm, z")
            a.plot(s, bound_amp_inplane(th) / div, "C3-", label="amplitude norm, in-plane")
            a.plot(s, bound_amp_z(th) / div, "C3--", label="amplitude norm, z")
            a.plot(s, amp_optimal_per_true_power(th) / div, "C2:", lw=2,
                   label="amp.-optimal beam, per physical power")
            a.set_xlabel(r"numerical aperture $\sin\theta$")
            a.set_xlim(0, 1)
        ax[0].set_ylabel(r"$(2\pi c/k^2)\,\varepsilon_0|E(0)|^2/P$")
        ax[1].set_ylabel(r"same, divided by $\sin^2\theta$")
        ax[0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        os.makedirs(FIGDIR, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIGDIR, f"focal_intensity_bounds.{ext}"), dpi=200)
        print(f"\nwrote {os.path.normpath(os.path.join(FIGDIR, 'focal_intensity_bounds.pdf'))}")


if __name__ == "__main__":
    main()
