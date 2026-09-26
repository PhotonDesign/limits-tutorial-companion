"""Sec. 3.2.4: space-time focusing in a complex (dispersive, multiple-scattering) medium.

Claim [eq:CMSTBound]: among currents s(x,t) on an aperture A, supported in a window [0,T]
and normalized by int_0^T int_A |s|^2 = 1, the maximum of |phi^T psi(x0,t0)|^2 is
    int_{t0-T}^{t0} dtau int_A |psi_0(x,tau)|^2,
where psi_0 is the field on A radiated by an impulsive dipole phi at (x0, 0) in the same
medium, and the optimum is s(x,t) ∝ psi_0(x, t0 - t) (the dipole response, time-reversed).
This reduces to the tutorial's int_0^T when t0 = T.

Model: a reciprocal, dispersive coupled-dipole cluster, alpha0_j(omega) = A_j w_j^2/(w_j^2 - omega^2
- i gamma omega) with real symmetric A_j (lossy Lorentz resonances), excited by electric
currents at N_A points of an open aperture.  All signals are band-limited to
omega in [0.5, 1.5] omega_c (smooth spectral window), periodic with a period much longer
than every time scale in the problem.  The retarded medium response h(x,x';tau) is
synthesized from the frequency-domain Green's matrices.  The space-time source-to-focus
map is built WITHOUT invoking reciprocity: we compute Gamma(x0, A; omega) directly.

Checks: (1) rank-one maximum = ||h^T phi||^2 over the window; (2) the time-reversed
dipole recipe (computed with the dipole at x0) attains it; (3) the maximum equals
int_{t0-T}^{t0} |psi_0|^2, and differs from int_0^T |psi_0|^2 when t0 != T.

Run: python spacetime_complex_media.py   (~10 s)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tutorial_tools.dipoles import DipoleMedium

WC = 2 * np.pi          # carrier (vacuum wavelength 1)


def build_responses(n_scat=10, n_ap=6, nf=256, seed=2):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-0.6, 0.6, size=(n_scat, 3))
    xs = xs[np.linalg.norm(xs, axis=1) > 0.2]
    x0 = np.zeros(3)
    apert = np.stack([rng.uniform(-1, 1, n_ap), rng.uniform(-1, 1, n_ap), np.full(n_ap, 1.5)], 1)
    A = []
    wres = rng.uniform(0.8, 1.2, len(xs)) * WC
    for j in range(len(xs)):
        R = rng.normal(size=(3, 3))
        A.append(((R + R.T) / 2 + 2 * np.eye(3)) * 0.4 * 6 * np.pi / WC**3)
    A = np.array(A)
    gamma = 0.05 * WC
    # frequency grid: period Tp = 2 pi / dw long compared with all delays
    dw = 2 * WC / nf
    w = dw * np.arange(1, nf + 1)
    win = np.exp(-((w - WC) / (0.25 * WC))**4)       # smooth band window
    G_0A = np.zeros((nf, 3, 3 * n_ap), complex)       # focus <- aperture currents
    G_A0 = np.zeros((nf, 3 * n_ap, 3), complex)       # aperture <- dipole at focus
    for n, wn in enumerate(w):
        alpha0 = A * (wres**2 / (wres**2 - wn**2 - 1j * gamma * wn))[:, None, None]
        med = DipoleMedium(xs, alpha0, k=wn)
        G_0A[n] = med.greens_E(x0[None, :], apert) * win[n]
        G_A0[n] = med.greens_E(apert, x0[None, :]) * win[n]
    return w, dw, G_0A, G_A0


def to_time(G, w, dw, taus):
    """h(tau) = 2 Re sum_n G(w_n) e^{-i w_n tau} dw / (2 pi)  (real, causal band-limited response)."""
    ph = np.exp(-1j * np.outer(taus, w))                # (Nt, Nf)
    return 2 * np.real(np.tensordot(ph, G, axes=(1, 0))) * dw / (2 * np.pi)


def run(T=2.0, t0_list=(2.0, 2.6), nt=160, verbose=True):
    w, dw, G_0A, G_A0 = build_responses()
    rng = np.random.default_rng(9)
    phi = rng.normal(size=3); phi /= np.linalg.norm(phi)       # real polarization
    t = np.linspace(0, T, nt, endpoint=False) + T / (2 * nt)    # source window samples
    dt = T / nt
    out = []
    for t0 in t0_list:
        h = to_time(G_0A, w, dw, t0 - t)                        # (nt, 3, 3NA): focus <- source at t
        Aop = np.einsum("a,jab->jb", phi, h).ravel() * np.sqrt(dt)   # linear functional on s
        bound = np.dot(Aop, Aop)                                # rank-one maximum
        # recipe: dipole phi fired at (x0, 0); its field on A at times t0 - t
        psi0 = to_time(G_A0, w, dw, t0 - t) @ phi               # (nt, 3NA)
        s = psi0.ravel() * np.sqrt(dt)
        achieved = np.dot(Aop, s)**2 / np.dot(s, s)
        # the bound as an integral of |psi0|^2 over [t0 - T, t0] vs over [0, T]
        tau_shift = t0 - t
        I_shift = np.sum(to_time(G_A0, w, dw, tau_shift) @ phi * (to_time(G_A0, w, dw, tau_shift) @ phi)) * dt
        tau_plain = t
        I_plain = np.sum((to_time(G_A0, w, dw, tau_plain) @ phi)**2) * dt
        out.append((t0, bound, achieved / bound, I_shift / bound, I_plain / bound))
        if verbose:
            print(f"  t0 = {t0:.2f} (T = {T:.2f}): bound {bound:.6e}; recipe/bound = {achieved/bound:.10f}; "
                  f"int_[t0-T,t0]|psi0|^2 / bound = {I_shift/bound:.10f}; int_[0,T]|psi0|^2 / bound = {I_plain/bound:.4f}")
    return out


def main():
    print("=" * 78)
    print("Sec. 3.2.4: space-time focusing in a dispersive multiple-scattering medium")
    print("=" * 78)
    run()


if __name__ == "__main__":
    main()
