"""Sec. 4.2: time-reversal-symmetry (TRS) bounds on internal-field ratios.

Tutorial claims checked (tutorial.tex line numbers as of 2026-09):
  1. [~1461-1470, eq:TRTwoPort] For a lossless TRS two-port with |R1| = |R2| = R,
        (1 - R)/(1 + R) <= |E2|^2/|E1|^2 <= (1 + R)/(1 - R),
     starting from E1^* = T^* E2 + R1^* E1.  Checked (a) analytically by sweeping the phase
     of E1, (b) by random feasible fields for random symmetric unitary S, (c) with the
     generalized (Ma et al.) eigenproblem, and (d) on a physical lossless asymmetric
     multilayer whose interior fields are computed with a transfer matrix.
  2. [~1472-1480, eq:ETRConstraint] E^* = S^dag E; [~1487-1489] E_i = [I + S_r^T]^{-1} S_i^T E_r.
  3. [~1491-1496, eq:fTRRatio] f = E_r^T W' E_r / E_r^T V' E_r, bounded by the extremal
     eigenvalues of "W' x = lambda V' x".  FINDING: with W' = [I - iF^*] W [I + iF^T]
     (complex Hermitian), the complex eigenproblem reproduces the UNCONSTRAINED bound (for
     eta_1 = |E_1|^2/sum|E_j|^2 it is exactly [0, 1]).  Because E_r is real, only the real
     symmetric part enters; the correct pencil is (Re W', Re V') -- this is g_S(W), g_S(V)
     of Ma, Pestourie & Johnson, PRA 111, 063518 (2025), Eq. (2).
  4. [~1453] intuition paragraph: 90% power transmission => field amplitude fraction
     |t| = sqrt(0.9) = 0.949 (not 0.90); exact range of |E_B|/|E_A| is [0.72, 1.39].
  5. [~1500-1504, fig:TRRatioExamples] the "Sounas-Alu bound eta_j + eta_k <= 1": compare the
     actual Sounas-Alu multiport bound eta_j + eta_k - 2|S_jk| sqrt(eta_j eta_k) <= 1 - |S_jk|^2
     and the Ma bound on random 4-port S.

Run:  python tr_field_ratio_bounds.py   (about 5 s; writes figures/sec4_tr_field_ratio_bounds.pdf/png)
"""
import os
import sys
import numpy as np
import scipy.linalg as sla
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import tmm1d  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


# ----------------------------------------------------------------------------- S matrices
def haar_unitary(n, rng):
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r))).conj()


def random_trs_smatrix(n, rng):
    """Random lossless reciprocal (=> TRS) S: symmetric unitary, S = U U^T."""
    U = haar_unitary(n, rng)
    return U @ U.T


def F_matrix(S):
    """F_S = Im S (I + Re S)^{-1}  (tutorial: F_S^T = [I + S_r^T]^{-1} S_i^T)."""
    n = S.shape[0]
    return np.linalg.solve((np.eye(n) + S.real).T, S.imag.T).T


def trs_map(S):
    """M = I + i F_S^T, so that every TRS-feasible field vector is E = M x with x = E_r real."""
    return np.eye(S.shape[0]) + 1j * F_matrix(S).T


def constraint_residual(S, E):
    """|| E^* - S^dag E || (eq:ETRConstraint)."""
    return np.linalg.norm(E.conj() - S.conj().T @ E)


def g_S(X, S):
    """Ma et al. Eq. (2): g_S(X) = Re X + F Re X F^T + F Im X - Im X F^T."""
    F = F_matrix(S)
    return X.real + F @ X.real @ F.T + F @ X.imag - X.imag @ F.T


def primed(X, S):
    """Tutorial's X' = [I - i F^*] X [I + i F^T] (complex Hermitian for Hermitian X)."""
    M = trs_map(S)
    return M.conj().T @ X @ M


def ratio_bounds(S, W, V, use_real=True):
    """Extremal generalized eigenvalues of (W', V').  use_real=True: real symmetric pencil
    (Re W', Re V') = (g_S(W), g_S(V)), i.e., x restricted to real vectors (correct).
    use_real=False: complex Hermitian pencil as literally written in the tutorial."""
    Wp, Vp = primed(W, S), primed(V, S)
    if use_real:
        Wp, Vp = Wp.real, Vp.real
    lam = sla.eigh(Wp, Vp, eigvals_only=True)
    return lam.min(), lam.max()


def eta(E, j):
    return np.abs(E[j])**2 / np.sum(np.abs(E)**2, axis=0)


def sample_feasible(S, n_samples, rng):
    """Random TRS-feasible field vectors E = M x (x real Gaussian); shape (n, n_samples)."""
    return trs_map(S) @ rng.normal(size=(S.shape[0], n_samples))


def optimize_eta(S, j, sign=+1, n_starts=20, rng=None):
    """Direct nonlinear optimization of eta_j over real x (independent of any eigensolver)."""
    rng = np.random.default_rng(0) if rng is None else rng
    M = trs_map(S)

    def f(x):
        E = M @ x
        return -sign * np.abs(E[j])**2 / np.sum(np.abs(E)**2)
    best = min(minimize(f, rng.normal(size=S.shape[0]), method="BFGS").fun for _ in range(n_starts))
    return -sign * best


# ----------------------------------------------------------------------------- two-port
def two_port_bound(R):
    return (1 - R) / (1 + R), (1 + R) / (1 - R)


def two_port_phase_sweep(S, n_phi=4001):
    """Exact: E1 = exp(i phi) (|E1| = 1 wlog), E2 = (E1^* - R1^* E1)/T^* from
    E1^* = T^* E2 + R1^* E1.  Returns min, max of |E2|^2/|E1|^2 over phi."""
    phi = np.linspace(0, 2 * np.pi, n_phi)
    E1 = np.exp(1j * phi)
    E2 = (E1.conj() - S[0, 0].conj() * E1) / S[1, 0].conj()
    ratio = np.abs(E2)**2
    return ratio.min(), ratio.max()


def multilayer_two_port(rng, n_layers=10, n_freq=400, n_z=40):
    """Physical check: random lossless asymmetric dielectric stack (n in [1.3, 3.5]).
    Returns arrays (R, ratio, TR-identity residual) for interior points and frequencies."""
    n_idx = rng.uniform(1.3, 3.5, n_layers)
    d = rng.uniform(0.05, 0.4, n_layers)
    eps = n_idx**2 + 0j
    L = d.sum()
    z = np.linspace(0, L, n_z)
    Rs, ratios, resid = [], [], []
    for k0 in np.linspace(0.5, 12, n_freq):
        r1, t1, r2, t2, _ = tmm1d.rt_both_sides(eps, d, k0)
        E1 = tmm1d.interior_fields(eps, d, k0, z, "left")
        E2 = tmm1d.interior_fields(eps, d, k0, z, "right")
        # TR identity E1^* = R1^* E1 + T^* E2 (S referenced at the two faces)
        resid.append(np.abs(E1.conj() - (r1.conj() * E1 + t1.conj() * E2)).max())
        ok = np.abs(E1) > 1e-6                  # E1 = 0 forces E2 = 0 (0/0 ratio)
        Rs.append(np.full(ok.sum(), abs(r1)))
        ratios.append(np.abs(E2[ok])**2 / np.abs(E1[ok])**2)
    return np.concatenate(Rs), np.concatenate(ratios), np.array(resid)


# ----------------------------------------------------------------------------- multiport
def sounas_alu_ok(eta_j, eta_k, s):
    """Sounas-Alu 2017 Eq. (2)/Ma Eq. (9): eta_j + eta_k - 2 s sqrt(eta_j eta_k) <= 1 - s^2."""
    return eta_j + eta_k - 2 * s * np.sqrt(eta_j * eta_k) <= 1 - s**2 + 1e-12


def ma_region_boundary(S, j, k, n_alpha=721):
    """Boundary of the convex hull of feasible (eta_j, eta_k): for each direction
    w = (cos a, sin a) the maximizing eigenvector of the real pencil gives a boundary point."""
    n = S.shape[0]
    V = np.eye(n)
    M = trs_map(S)
    pts = []
    for a in np.linspace(0, 2 * np.pi, n_alpha):
        W = np.zeros((n, n))
        W[j, j], W[k, k] = np.cos(a), np.sin(a)
        lam, X = sla.eigh(primed(W, S).real, primed(V, S).real)
        E = M @ X[:, -1]
        pts.append([eta(E, j), eta(E, k)])
    return np.array(pts)


def main(make_figure=True):
    rng = np.random.default_rng(20260926)
    print("=" * 84)
    print("Sec. 4.2  time-reversal-symmetry field-ratio bounds")
    print("=" * 84)

    # ---------------- two-port --------------------------------------------------------
    print("\n[1] Two-port (1-R)/(1+R) <= |E2|^2/|E1|^2 <= (1+R)/(1-R)  [eq:TRTwoPort]")
    print(f"{'R':>6} | {'claimed lo':>10} {'claimed hi':>10} | {'phase-sweep lo':>14} {'hi':>9} |"
          f" {'Re-pencil lo':>12} {'hi':>9} | {'sampled lo':>10} {'hi':>9} | {'cplx pencil':>15}")
    for _ in range(5):
        S = random_trs_smatrix(2, rng)
        R = abs(S[0, 0])
        lo, hi = two_port_bound(R)
        plo, phi_ = two_port_phase_sweep(S)
        W, V = np.diag([0.0, 1.0]), np.diag([1.0, 0.0])
        mlo, mhi = ratio_bounds(S, W, V, True)
        Es = sample_feasible(S, 200000, rng)
        rs = np.abs(Es[1])**2 / np.abs(Es[0])**2
        clo, chi = ratio_bounds(S, W, V + 1e-14 * np.eye(2), False)
        print(f"{R:6.3f} | {lo:10.5f} {hi:10.5f} | {plo:14.5f} {phi_:9.5f} | {mlo:12.5f} {mhi:9.5f} |"
              f" {rs.min():10.5f} {rs.max():9.5f} | [{clo:.1e},{chi:.1e}]")
    print("  -> envelope confirmed exactly; the complex pencil gives [0, inf) (no bound).")

    print("\n[1d] Physical lossless asymmetric multilayer (10 random layers, n in [1.3,3.5]),"
          " 400 freqs x 40 interior points")
    Rm, ratio_m, resid = multilayer_two_port(rng)
    lo, hi = two_port_bound(Rm)
    viol = np.max(np.maximum(np.log(lo / ratio_m), np.log(ratio_m / hi)))
    print(f"  max TR-identity residual |E1^* - R1^* E1 - T^* E2| = {resid.max():.2e}")
    print(f"  R spans [{Rm.min():.3f}, {Rm.max():.3f}]; max log-violation of bound = {viol:.2e}"
          f" (<= 0 means inside); fraction of points within 2% of an edge:"
          f" {np.mean(np.minimum(abs(np.log(ratio_m/lo)), abs(np.log(ratio_m/hi))) < 0.02):.3f}")

    # ---------------- line ~1453 intuition ---------------------------------------------
    print("\n[4] Intuition paragraph (line ~1453): power transmission |T|^2 = 0.9")
    T = np.sqrt(0.9)
    R = np.sqrt(1 - 0.9)
    lo, hi = two_port_bound(R)
    print(f"  amplitude of the time-reversed component = |T| = {T:.4f}  (text: '90% of the"
          f" original amplitude')")
    print(f"  exact bound: |E_B|^2/|E_A|^2 in [{lo:.3f}, {hi:.3f}];  |E_B|/|E_A| in"
          f" [{np.sqrt(lo):.3f}, {np.sqrt(hi):.3f}]")

    # ---------------- n-port: complex vs real pencil ------------------------------------
    print("\n[3] n-port bound on eta_1 = |E_1|^2 / sum_j |E_j|^2 (W = e1 e1^T, V = I) [eq:fTRRatio]")
    print(f"{'n':>2} | {'Re-pencil min':>13} {'max':>8} | {'sampled min':>11} {'max':>8} |"
          f" {'BFGS min':>9} {'max':>8} | {'complex pencil':>16} | {'||g_S(W)-Re W\'||':>17}")
    for n in [3, 4, 4, 6, 8]:
        S = random_trs_smatrix(n, rng)
        W = np.zeros((n, n)); W[0, 0] = 1
        V = np.eye(n)
        rlo, rhi = ratio_bounds(S, W, V, True)
        clo, chi = ratio_bounds(S, W, V, False)
        E = sample_feasible(S, 200000, rng)
        et = eta(E, 0)
        omin, omax = optimize_eta(S, 0, -1, rng=rng), optimize_eta(S, 0, +1, rng=rng)
        gdiff = np.abs(g_S(W, S) - primed(W, S).real).max()
        res = max(constraint_residual(S, E[:, i]) for i in range(5))
        print(f"{n:2d} | {rlo:13.5f} {rhi:8.5f} | {et.min():11.5f} {et.max():8.5f} |"
              f" {omin:9.5f} {omax:8.5f} | [{clo:.1e}, {chi:.4f}] | {gdiff:17.1e}   (E*=S^dag E resid {res:.0e})")
    print("  -> the complex Hermitian pencil returns the trivial range [0, 1]: it optimizes over"
          " complex x, i.e. over ALL complex E (M invertible), discarding the TRS constraint.")
    print("     The real pencil (Re W', Re V') = (g_S(W), g_S(V)) is attained (BFGS = eigen)."
          )

    # vector-field version (S (x) I_3, three field components)
    n = 4
    S = random_trs_smatrix(n, rng)
    Sk = np.kron(S, np.eye(3))
    W = np.kron(np.diag([1.0, 0, 0, 0]), np.eye(3))
    V = np.eye(3 * n)
    rlo, rhi = ratio_bounds(Sk, W, V, True)
    E = sample_feasible(Sk, 100000, rng)
    et = np.sum(np.abs(E[:3])**2, 0) / np.sum(np.abs(E)**2, 0)
    print(f"  vector fields (S (x) I_3, n = 4): Re-pencil [{rlo:.4f}, {rhi:.4f}], sampled"
          f" [{et.min():.4f}, {et.max():.4f}]")

    # ---------------- Sounas-Alu vs Ma on (eta_j, eta_k) --------------------------------
    print("\n[5] 4-port (eta_1, eta_2) feasible region: Ma bound vs Sounas-Alu (SA) bound")
    n = 4
    S = random_trs_smatrix(n, rng)
    s12 = abs(S[0, 1])
    E = sample_feasible(S, 200000, rng)
    e1, e2 = eta(E, 0), eta(E, 1)
    Wd = np.zeros((n, n)); Wd[0, 0] = Wd[1, 1] = 1
    ma_sum = ratio_bounds(S, Wd, np.eye(n), True)[1]
    print(f"  |S_12| = {s12:.3f};  all samples satisfy SA: {np.all(sounas_alu_ok(e1, e2, s12))}")
    print(f"  max(eta_1 + eta_2): Ma bound {ma_sum:.4f}, sampled {np.max(e1 + e2):.4f},"
          f" SA/trivial bound 1 (SA along diagonal: min(1, 1+|S12|))")
    print(f"  max eta_1: Ma {ratio_bounds(S, np.diag([1.0, 0, 0, 0]), np.eye(4))[1]:.4f}; SA/trivial 1")

    if make_figure:
        make_plot(rng)
    return dict()


def make_plot(rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    # (a) two-port
    Rg = np.linspace(0, 0.999, 400)
    lo, hi = two_port_bound(Rg)
    Rs, rat = [], []
    for _ in range(400):
        S = random_trs_smatrix(2, rng)
        E = sample_feasible(S, 60, rng)
        Rs.append(np.full(60, abs(S[0, 0])))
        rat.append(np.abs(E[1])**2 / np.abs(E[0])**2)
    ax[0].semilogy(np.concatenate(Rs), np.concatenate(rat), ".", ms=1.5, color="0.6",
                   label="random symmetric unitary S, feasible E")
    Rm, ratio_m, _ = multilayer_two_port(np.random.default_rng(3), n_freq=250, n_z=25)
    ax[0].semilogy(Rm, ratio_m, "k.", ms=1.5, label="lossless multilayer (TMM fields)")
    ax[0].semilogy(Rg, lo, "r-", Rg, hi, "r-", lw=1.5)
    ax[0].set_ylim(1e-3, 1e3)
    ax[0].set_xlabel("reflectivity amplitude R")
    ax[0].set_ylabel(r"$|E_2|^2/|E_1|^2$")
    ax[0].set_title("(a) two-port: (1$\\mp$R)/(1$\\pm$R) envelope")
    ax[0].legend(fontsize=7, loc="upper left")
    # (b) four-port region
    S = random_trs_smatrix(4, np.random.default_rng(11))
    E = sample_feasible(S, 20000, rng)
    ax[1].plot(eta(E, 0), eta(E, 1), ".", ms=1, color="tab:blue", label="feasible samples")
    B = ma_region_boundary(S, 0, 1)
    ax[1].plot(B[:, 0], B[:, 1], "k-", lw=1.2, label="Ma et al. bound (real pencil)")
    s = abs(S[0, 1])
    a = np.linspace(0, 1, 400)
    up = s * a + np.sqrt((1 - s**2) * (1 - a**2))
    dn = s * a - np.sqrt((1 - s**2) * (1 - a**2))
    ax[1].plot(a**2, np.minimum(up**2, 1 - a**2), "c-", lw=1.5, label="Sounas-Alu (with $\\eta_j+\\eta_k\\leq 1$)")
    ok = dn > 0
    ax[1].plot(a[ok]**2, dn[ok]**2, "c-", lw=1.5)
    ax[1].plot([0, 1], [1, 0], "c--", lw=0.8)
    ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
    ax[1].set_xlabel(r"$\eta_1$"); ax[1].set_ylabel(r"$\eta_2$")
    ax[1].set_title(f"(b) random TRS 4-port, |S$_{{12}}$| = {s:.2f}")
    ax[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"sec4_tr_field_ratio_bounds.{ext}"), dpi=150)
    print("\nwrote figures/sec4_tr_field_ratio_bounds.pdf/png")


if __name__ == "__main__":
    main()
