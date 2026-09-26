"""Sec. 3.4: communication channels between two parallel square arrays (fig:CommChannels).

Tutorial claims checked (tutorial.tex line numbers as of 2026-09):
  1. [~1353, caption] "two parallel 16 lambda x 16 lambda source/receiver arrays separated by
     50 lambda" with the first 12 mode power fractions shown in panel (c):
        12.09, 11.66, 11.66, 11.24, 8.29, 8.28, 7.96, 7.96, 5.61, 2.51, 2.51, 2.41 %
     (read from figures/CommunicationChannels.pdf; identical to Miller, AOP 11, 679 (2019),
     Fig. 8, which uses 17 x 17 point sources/receivers on lambda centres).
     Paraxial heuristic N_H = A_T A_R / (lambda L)^2 = 26.2 for 16 lambda / 50 lambda.
  2. [~1364-1368] input vectors = RIGHT singular vectors, measurement vectors = LEFT singular
     vectors; T = sum_i lambda_i sigma_{l,i} sigma_{r,i}^dag; notation sentence and
     "excite the N corresponding right singular values".
  3. [~1370-1378] max received energy for unit input = lambda_1^2 at sigma_{r,1}; chain
     argument => best N orthogonal inputs capture sum_{i<=N} lambda_i^2 (Ky Fan).

FINDING: for the stated 16 lambda / 50 lambda geometry the SVD gives ~25 nearly equal channels
with mode 1 carrying only 3.2-4.1 % of sum|lambda|^2, not 12.09 %.  Miller's 12 percentages are
reproduced to 0.002 percentage points (rms) by the same 17 x 17 arrays at L = 100 lambda
(N_H = 6.6), so the figure's separation label (and our caption) should read 100 lambda.

Run:  python communication_modes.py   (about 6 s; writes figures/sec3_4_communication_modes.pdf/png)
"""
import os
import sys
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
LAM = 1.0
K = 2 * np.pi / LAM
FIGURE_PERCENT = np.array([12.09, 11.66, 11.66, 11.24, 8.29, 8.28, 7.96, 7.96, 5.61, 2.51, 2.51, 2.41])


def square_points(side, n, center=(0.0, 0.0), mode="nodes"):
    """n x n points on a side x side square.  'nodes': endpoints included (Miller's point
    arrays, spacing side/(n-1)); 'cells': midpoints of n x n cells (quadrature of a
    continuous aperture, spacing side/n)."""
    if mode == "nodes":
        x = np.linspace(-side / 2, side / 2, n)
    else:
        x = (np.arange(n) + 0.5) * side / n - side / 2
    X, Y = np.meshgrid(x + center[0], x + center[1], indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], 1)


def green_matrix(pts_r, pts_t, L):
    """Scalar free-space Green's function exp(ikr)/(4 pi r) between two parallel planes
    separated by L; rows = receiver points, columns = source points."""
    d2 = ((pts_r[:, None, :] - pts_t[None, :, :])**2).sum(-1) + L**2
    r = np.sqrt(d2)
    return np.exp(1j * K * r) / (4 * np.pi * r)


def power_fractions(T):
    s = np.linalg.svd(T, compute_uv=False)
    return s, 100 * s**2 / np.sum(s**2)


def miller_arrays(L, n=17, spacing=1.0):
    pts = square_points((n - 1) * spacing, n, mode="nodes")
    return green_matrix(pts, pts, L)


def continuous_apertures(side, L, dx):
    n = int(round(side / dx))
    pts = square_points(side, n, mode="cells")
    return green_matrix(pts, pts, L)


def fit_separation(n=17, spacing=1.0):
    """L that best reproduces the figure's 12 percentages (rms, percentage points)."""
    f = lambda L: np.sqrt(np.mean((power_fractions(miller_arrays(L, n, spacing))[1][:12] - FIGURE_PERCENT)**2))
    res = minimize_scalar(f, bounds=(40, 160), method="bounded", options=dict(xatol=1e-3))
    return res.x, res.fun


def svd_label_check(seed=0):
    """Asymmetric geometry (T not symmetric, not square): 16 lambda transmitter, 10 lambda
    receiver offset by (5, 3) lambda, L = 40 lambda.  Verifies the roles of right (input) and
    left (measurement) singular vectors.  (For identical coaxial arrays T = T^T and the Takagi
    form u_i = v_i^* makes a left/right swap hard to detect, hence the asymmetric geometry.)"""
    pt = square_points(16, 32, mode="cells")
    pr = square_points(10, 20, center=(5.0, 3.0), mode="cells")
    T = green_matrix(pr, pt, 40.0)
    Ul, s, Vh = np.linalg.svd(T, full_matrices=False)
    Vr = Vh.conj().T                                           # right singular vectors (inputs)
    N = 10
    recon = np.abs(T - (Ul * s) @ Vh).max() / s[0]
    # (i) T sigma_r,i = lambda_i sigma_l,i
    err_map = max(np.linalg.norm(T @ Vr[:, i] - s[i] * Ul[:, i]) / s[i] for i in range(N))
    # (ii) received energy with the first N right singular vectors vs Ky Fan sum
    E_right = np.sum(np.linalg.norm(T @ Vr[:, :N], axis=0)**2)
    kyfan = np.sum(s[:N]**2)
    # (iii) baseline: random orthonormal inputs
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(T.shape[1], N)) + 1j * rng.normal(size=(T.shape[1], N)))
    E_random = np.sum(np.linalg.norm(T @ Q, axis=0)**2)
    # (iv) crosstalk: measuring in left vectors u_j after exciting v_i gives lambda_i delta_ij
    C = Ul[:, :N].conj().T @ T @ Vr[:, :N]
    crosstalk = np.abs(C - np.diag(s[:N])).max() / s[0]
    return dict(shape=T.shape, recon=recon, err_map=err_map, E_right=E_right, kyfan=kyfan,
                E_random=E_random, crosstalk=crosstalk, s=s)


def main(make_figure=True):
    print("=" * 84)
    print("Sec. 3.4  communication channels: 16 lambda x 16 lambda arrays (fig:CommChannels)")
    print("=" * 84)
    NH50 = 16**2 * 16**2 / 50**2
    NH100 = 16**2 * 16**2 / 100**2
    print(f"Paraxial heuristic N_H = A_T A_R/(lambda L)^2: L = 50 -> {NH50:.1f};  L = 100 -> {NH100:.2f}")

    print("\n(a) continuous 16 lambda apertures, L = 50 lambda, midpoint sampling (convergence):")
    for dx in (0.5, 0.4):
        s, p = power_fractions(continuous_apertures(16, 50, dx))
        print(f"  dx = {dx:.2f} lambda ({int(16 / dx)}^2 pts): first 12 % = {np.round(p[:12], 2)}")
        print(f"      sum first 12 = {p[:12].sum():.1f} %, first 26 = {p[:26].sum():.1f} %;"
              f" #(lambda^2 > lambda_1^2/2) = {(p > p[0] / 2).sum()}, #(> lambda_1^2/10) = {(p > p[0] / 10).sum()}")

    print("\n(b) Miller's discrete arrays (17 x 17 points on lambda centres), first 12 fractions (%):")
    print(f"  figure (c)      : {FIGURE_PERCENT}")
    rows = {}
    for L in (50.0, 100.0):
        s, p = power_fractions(miller_arrays(L))
        rows[L] = p
        rms = np.sqrt(np.mean((p[:12] - FIGURE_PERCENT)**2))
        print(f"  L = {L:5.1f} lambda : {np.round(p[:12], 2)}  rms dev {rms:.3f};"
              f" #(>half of mode 1) = {(p > p[0] / 2).sum()}")
    Lfit, rms = fit_separation()
    print(f"  best-fit separation: L = {Lfit:.2f} lambda (rms {rms:.4f} percentage points)")

    print("\n(c) SVD role check on an asymmetric geometry (T is 400 x 1024, not symmetric):")
    c = svd_label_check()
    print(f"  reconstruction T = sum lambda_i sigma_l,i sigma_r,i^dag: rel err {c['recon']:.1e}")
    print(f"  T sigma_r,i = lambda_i sigma_l,i (i <= 10): max rel err {c['err_map']:.1e}")
    print(f"  crosstalk |sigma_l,j^dag T sigma_r,i - lambda_i delta_ij| / lambda_1 = {c['crosstalk']:.1e}")
    print(f"  energy received, 10 right singular vectors as inputs: {c['E_right']:.6e}"
          f" = Ky Fan sum {c['kyfan']:.6e};  10 random orthonormal inputs: {c['E_random']:.3e}")
    print("  (inputs live in C^1024 = source space = domain of T; left singular vectors live in"
          " C^400 = receiver space, so 'input = right, measurement = left' is the only consistent"
          " assignment)")

    if make_figure:
        make_plot(rows)
    return rows


def make_plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    m = np.arange(1, 41)
    for L, st in ((50.0, "o-"), (100.0, "s-")):
        ax[0].semilogy(m, rows[L][:40], st, ms=3, label=f"17x17 arrays, L = {L:.0f}$\\lambda$")
    ax[0].semilogy(np.arange(1, 13), FIGURE_PERCENT, "k*", ms=8, label="figure (c) / Miller 2019")
    ax[0].axvline(26.2, color="C0", ls=":", lw=1)
    ax[0].axvline(6.55, color="C1", ls=":", lw=1)
    ax[0].set_xlabel("mode number"); ax[0].set_ylabel("power fraction (%)")
    ax[0].set_ylim(1e-3, 30)
    ax[0].legend(fontsize=8); ax[0].set_title("coupling strengths (dotted: $N_H$)")
    x = np.arange(12)
    ax[1].bar(x - 0.27, rows[50.0][:12], 0.27, label="L = 50$\\lambda$ (caption)")
    ax[1].bar(x, rows[100.0][:12], 0.27, label="L = 100$\\lambda$")
    ax[1].bar(x + 0.27, FIGURE_PERCENT, 0.27, color="k", label="figure (c)")
    ax[1].set_xticks(x, [str(i + 1) for i in x])
    ax[1].set_xlabel("mode"); ax[1].set_ylabel("% of sum rule"); ax[1].legend(fontsize=8)
    ax[1].set_title("first 12 modes")
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"sec3_4_communication_modes.{ext}"), dpi=150)
    print("\nwrote figures/sec3_4_communication_modes.pdf/png")


if __name__ == "__main__":
    main()
