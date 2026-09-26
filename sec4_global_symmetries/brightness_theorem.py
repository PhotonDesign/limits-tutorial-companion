"""Sec. 4.4: brightness theorem for waves (passive linear maps of coherency matrices).

Tutorial claims checked (tutorial.tex line numbers as of 2026-09):
  1. [~1626-1640] rho_out = S rho_in S^dag with S passive (S^dag S <= I) =>
        sum_{i<=N} lambda_i^down(rho_out) <= sum_{i<=N} lambda_i^down(rho_in)  for all N.
     Stronger termwise theorem (lab-log 2026-08-30, D. A. B. Miller):
        lambda_k(rho_out) <= lambda_k(rho_in) for every k  (s_k(M rho^1/2) <= s_1(M) s_k(rho^1/2)).
  2. [~1628-1631] "This is exactly the definition by which one vector is majorized by another"
     -- majorization requires equal totals; with loss the relation is WEAK (sub)majorization.
  3. [~1641] Hardy-Littlewood-Polya: "lambda_out = D lambda_in for some doubly-stochastic D ...
     every Schur-concave function of the spectrum, including the Shannon and von Neumann
     entropies, is monotonically non-decreasing".  Tested: (a) unitary S -> spectrum is
     exactly invariant (D = I; entropy constant, never increases); (b) lossy S -> the total
     decreases, so no doubly-stochastic D exists, and the normalized entropy can DECREASE.
     What does hold: lambda_out = D lambda_in with D DIAGONAL, 0 <= D_kk <= 1 (termwise).
  4. [~1649] concentrator: N incoherent unit-power inputs, one DL output channel ->
     average efficiency <= 1/N; N = 11 -> 9.09 % ("9%"); N = 5 -> 20 %.
  5. [~1657] TCMT: M < N resonances, N incoherent inputs -> average transmission through the
     resonances <= M/N (rank argument), checked on random TRS-consistent TCMT models.

Port/extension of overleaf/analysis/brightness_termwise_check.py (read-only original).

Run:  python brightness_theorem.py   (about 3 s; writes figures/sec4_brightness_theorem.pdf/png)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from tr_field_ratio_bounds import haar_unitary  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def eig_desc(A):
    return np.linalg.eigvalsh(A)[::-1]


def random_density(n, rng, log_range=6.0):
    lam = 10.0 ** rng.uniform(-log_range, 0.0, n)
    Q = haar_unitary(n, rng)
    return (Q * lam) @ Q.conj().T


def random_contraction(n_out, n_in, rng, smax=1.0):
    """Passive (sub-unitary) map: singular values uniform in [0, smax]."""
    U = haar_unitary(n_out, rng)[:, :min(n_out, n_in)]
    V = haar_unitary(n_in, rng)[:, :min(n_out, n_in)]
    s = rng.uniform(0, smax, min(n_out, n_in))
    return (U * s) @ V.conj().T


def termwise_trials(dims=range(2, 9), trials=2000, seed=20260830, rectangular=True):
    """Max over trials of max_k [lambda_k(out) - lambda_k(in)] and of the partial-sum excess.
    Rectangular maps: output dimension n_out drawn from [1, n+3]; spectra zero-padded."""
    rng = np.random.default_rng(seed)
    out = {}
    for n in dims:
        worst_term, worst_sum = -np.inf, -np.inf
        for _ in range(trials):
            rho = random_density(n, rng)
            n_out = rng.integers(1, n + 4) if rectangular else n
            M = random_contraction(n_out, n, rng)
            li = eig_desc(rho)
            lo = eig_desc(M @ rho @ M.conj().T)
            m = max(len(li), len(lo))
            li = np.pad(li, (0, m - len(li)))
            lo = np.pad(lo, (0, m - len(lo)))
            worst_term = max(worst_term, np.max(lo - li))
            worst_sum = max(worst_sum, np.max(np.cumsum(lo) - np.cumsum(li)))
        out[n] = (worst_term, worst_sum)
    return out


def shannon(p):
    p = np.asarray(p, float)
    p = p[p > 1e-15] / p.sum()
    return float(-np.sum(p * np.log(p))) + 0.0


def unitary_spectrum_invariance(n=6, trials=200, seed=1):
    rng = np.random.default_rng(seed)
    worst_spec, worst_entropy = 0.0, 0.0
    for _ in range(trials):
        rho = random_density(n, rng, 2.0)
        S = haar_unitary(n, rng)
        li, lo = eig_desc(rho), eig_desc(S @ rho @ S.conj().T)
        worst_spec = max(worst_spec, np.abs(lo - li).max() / li[0])
        worst_entropy = max(worst_entropy, abs(shannon(lo) - shannon(li)))
    return worst_spec, worst_entropy


def lossy_entropy_counterexample():
    """rho_in = diag(1/2, 1/2), M = diag(1, 0) (passive): normalized output spectrum (1, 0)."""
    rho = np.diag([0.5, 0.5])
    M = np.diag([1.0, 0.0])
    lo = eig_desc(M @ rho @ M.T)
    return shannon(eig_desc(rho)), shannon(lo), eig_desc(rho).sum(), lo.sum()


def lossy_entropy_random(n=4, trials=5000, seed=2):
    """Fraction of random passive lossy maps for which the normalized spectral entropy drops."""
    rng = np.random.default_rng(seed)
    drops, max_drop = 0, 0.0
    for _ in range(trials):
        rho = random_density(n, rng, 1.0)
        M = random_contraction(n, n, rng)
        d = shannon(eig_desc(M @ rho @ M.conj().T)) - shannon(eig_desc(rho))
        drops += d < -1e-12
        max_drop = max(max_drop, -d)
    return drops / trials, max_drop


def coherent_combiner_channel_entropy():
    """Unitary 50/50 combiner, coherent input (1,1)/sqrt2: channel powers (1/2,1/2) -> (1,0).
    Shannon entropy of CHANNEL POWERS (diagonal, not spectrum) drops log2 -> 0 with no loss."""
    c = np.array([1, 1]) / np.sqrt(2)
    S = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    return shannon(np.abs(c)**2), shannon(np.abs(S @ c)**2)


def miller_counterexample():
    lam_in = np.array([1.0, 0.3])
    lam_out = np.array([0.8, 0.4])
    return (np.cumsum(lam_out) <= np.cumsum(lam_in)).all(), np.prod(lam_out) / np.prod(lam_in)


# ------------------------------------------------------------------------ concentrator
def concentrator_efficiency(N, n_out=40, trials=2000, seed=3):
    """N incoherent unit-power inputs (rho_in = I_N); DL spot = one fixed output channel phi.
    Average efficiency (1/N) sum_k |phi^dag M e_k|^2 for random passive M, and for the
    optimal M (row phi^dag M of unit norm)."""
    rng = np.random.default_rng(seed)
    phi = np.zeros(n_out); phi[0] = 1
    best = 0.0
    for _ in range(trials):
        M = random_contraction(n_out, N, rng, smax=1.0)
        best = max(best, np.sum(np.abs(phi @ M)**2) / N)
    # optimal: put a unit-norm row into the DL channel, e.g. equal split over the N inputs
    Mopt = np.zeros((n_out, N), complex)
    Mopt[0] = 1 / np.sqrt(N)
    Q = haar_unitary(n_out - 1, rng)[:, :N - 1]              # rest of an isometry
    B = np.eye(N) - np.outer(Mopt[0].conj(), Mopt[0])
    w, v = np.linalg.eigh(B)
    Mopt[1:] = Q @ (np.sqrt(np.clip(w[1:], 0, None))[:, None] * v[:, 1:].conj().T)
    passive = np.linalg.norm(Mopt, 2) <= 1 + 1e-12
    eff_opt = np.sum(np.abs(Mopt[0])**2) / N
    lam1 = eig_desc(Mopt @ Mopt.conj().T)[0]
    return best, eff_opt, passive, lam1


# ------------------------------------------------------------------------ TCMT
def tcmt_model(n_in, n_out, M, rng):
    """Random reciprocal, TRS-consistent TCMT with direct pathway C = -I (no direct
    input->output transmission).  D real ((n_in+n_out) x M), Gamma = D^T D / 2,
    Omega real symmetric.  S(w) = C + D [i(Omega - w) + Gamma]^{-1} D^T."""
    P = n_in + n_out
    D = rng.normal(size=(P, M))
    A = rng.normal(size=(M, M))
    Omega = (A + A.T) / 2
    Gamma = D.T @ D / 2
    C = -np.eye(P)
    return C, D, Omega, Gamma


def tcmt_S(C, D, Omega, Gamma, w):
    Mres = Omega.shape[0]
    return C + D @ np.linalg.solve(1j * (Omega - w * np.eye(Mres)) + Gamma, D.T)


def tcmt_max_avg_transmission(n_in, n_out, M, trials=300, n_w=60, seed=4):
    rng = np.random.default_rng(seed)
    best, unit_err = 0.0, 0.0
    for _ in range(trials):
        C, D, Om, G = tcmt_model(n_in, n_out, M, rng)
        for w in np.linspace(-3, 3, n_w):
            S = tcmt_S(C, D, Om, G, w)
            unit_err = max(unit_err, np.abs(S.conj().T @ S - np.eye(len(S))).max())
            Toi = S[n_in:, :n_in]
            best = max(best, np.sum(np.abs(Toi)**2) / n_in)
    return best, unit_err


def tcmt_optimal_single_resonance(n_in=2, n_out=2):
    """One resonance, equal decay into inputs and outputs, on resonance: avg T = 1/n_in."""
    d = np.concatenate([np.full(n_in, 1 / np.sqrt(n_in)), np.full(n_out, 1 / np.sqrt(n_out))])[:, None]
    S = tcmt_S(-np.eye(n_in + n_out), d, np.zeros((1, 1)), d.T @ d / 2, 0.0)
    return np.sum(np.abs(S[n_in:, :n_in])**2) / n_in


def main(make_figure=True):
    print("=" * 84)
    print("Sec. 4.4  brightness theorem: rho_out = S rho_in S^dag, S passive")
    print("=" * 84)
    res = termwise_trials()
    print(" n | max_k [lam_k(out) - lam_k(in)] | max_N [partial-sum excess]   (2000 trials each,"
          " rectangular S)")
    for n, (wt, ws) in res.items():
        print(f"{n:2d} | {wt:+.3e}                     | {ws:+.3e}")
    print("  -> termwise theorem and the text's partial-sum (weak-majorization) statement hold.")

    ok_weak, det2 = miller_counterexample()
    print(f"\nMiller S5 example lam_in = (1, 0.3), lam_out = (0.8, 0.4): partial sums OK = {ok_weak},"
          f" but needs |det M|^2 = {det2:.4f} > 1 -> not passively reachable (termwise bound is"
          " strictly stronger).")

    ws, we = unitary_spectrum_invariance()
    print(f"\nUnitary S: max |lam_out - lam_in|/lam_1 = {ws:.1e}; max |Delta entropy| = {we:.1e}"
          " -> HLP 'doubly-stochastic mixing' is D = I (vacuous).")
    h_in, h_out, tr_in, tr_out = lossy_entropy_counterexample()
    print(f"Lossy S = diag(1,0), rho_in = diag(1/2,1/2): trace {tr_in:.2f} -> {tr_out:.2f} (no doubly-"
          f"stochastic D can map it), normalized entropy {h_in:.4f} -> {h_out:.4f} (DEcreases).")
    frac, mx = lossy_entropy_random()
    print(f"Random lossy 4x4 passive maps: normalized spectral entropy decreases in {100 * frac:.1f}%"
          f" of 5000 cases (max drop {mx:.3f} nats).")
    hc_in, hc_out = coherent_combiner_channel_entropy()
    print(f"(Channel-power Shannon entropy under a lossless 50/50 coherent combiner: {hc_in:.4f} ->"
          f" {hc_out:.4f}; also not monotone.)")

    print("\nConcentrator (line ~1649): max average DL efficiency for N incoherent inputs")
    for N in (1, 2, 5, 11):
        best, eff, passive, lam1 = concentrator_efficiency(N)
        print(f"  N = {N:2d}: bound 1/N = {100 / N:6.2f}%; optimal passive design {100 * eff:6.2f}%"
              f" (passive {passive}, lam_1(rho_out) = {lam1:.3f}); best of 2000 random {100 * best:6.2f}%")

    print("\nTCMT (line ~1657): N incoherent inputs, M resonances, avg transmission <= M/N")
    for (n_in, n_out, M) in ((2, 1, 1), (2, 2, 1), (3, 3, 2), (4, 4, 2), (2, 2, 2)):
        best, uerr = tcmt_max_avg_transmission(n_in, n_out, M)
        bound = min(M, n_out, n_in) / n_in
        print(f"  N = {n_in}, outputs = {n_out}, M = {M}: max avg T over random models = {best:.4f}"
              f"  (bound min(M, outputs)/N = {bound:.4f}); max |S^dag S - I| = {uerr:.1e}")
    print(f"  optimal single resonance, 2 in / 2 out, on resonance: avg T = {tcmt_optimal_single_resonance():.4f} (= 1/2)")

    if make_figure:
        make_plot()
    return res


def make_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ks, ratios = [], []
    for _ in range(400):
        rho = random_density(6, rng, 3.0)
        M = random_contraction(6, 6, rng)
        li, lo = eig_desc(rho), eig_desc(M @ rho @ M.conj().T)
        ks.append(np.arange(1, 7)); ratios.append(lo / li)
    ax[0].plot(np.concatenate(ks) + rng.uniform(-0.2, 0.2, 2400), np.concatenate(ratios), ".", ms=2)
    ax[0].axhline(1, color="r")
    ax[0].set_xlabel("eigenvalue index k"); ax[0].set_ylabel(r"$\lambda_k(\rho_{out})/\lambda_k(\rho_{in})$")
    ax[0].set_title("termwise theorem (n = 6, random passive S)")
    N = np.arange(1, 16)
    ax[1].plot(N, 100 / N, "b-o", ms=4, label="1/N bound")
    ax[1].plot(N, [100 * concentrator_efficiency(n, trials=200)[1] for n in N], "kx", label="explicit passive optimum")
    ax[1].axvline(11, ls=":", color="k"); ax[1].axhline(100 / 11, ls=":", color="k")
    ax[1].set_xlabel("# incoherent angles N"); ax[1].set_ylabel("max DL efficiency (%)")
    ax[1].legend(); ax[1].set_title("concentrator: N = 11 -> 9.09%")
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"sec4_brightness_theorem.{ext}"), dpi=150)
    print("\nwrote figures/sec4_brightness_theorem.pdf/png")


if __name__ == "__main__":
    main()
