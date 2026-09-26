"""Sec. 1.7 (Rayleigh quotients): Ky Fan's theorem and the two Hoffman-Wielandt-type trace
bounds, checked by direct numerical optimization over the Stiefel manifold
{X in C^{M x N}: X^dag X = I_N}, including the hypotheses the bounds need.

Tutorial claims checked (tutorial.tex line numbers):
  l.634      max_x x^dag A x / x^dag x = lambda_max(A)   (printed without the dagger)
  l.640      max over x orthogonal to x_1 = lambda_2(A)  (printed without the dagger)
  [eq:FanThm] l.645  max_{X^dag X = I_N} Tr(X^dag A X) = sum_{i<=N} lambda_i(A)
  [eq:HW1]    l.652  max_{X^dag X = I_N} Tr(X^dag A X B) <= sum lambda_i^down(A) lambda_i^down(B);
                     "one can relax the orthogonality constraint to X^dag X <= I" (l.655)
  [eq:HW2]    l.657  B >= 0:  min_{X^dag X = I_N} Tr(X^dag A X B) >= sum lambda_i^up(A) lambda_i^down(B)

Findings.  Fan and HW2 hold with equality (the eigenvector choice attains them).  HW1 holds
with equality when B >= 0 (or when X is square), but for N < M it FAILS for indefinite B,
and the relaxation to X^dag X <= I fails unless the N largest eigenvalues of A are >= 0.
Both extra hypotheses hold in the tutorial's application (density matrices, Sec. 4.4) but
are not stated where the theorem is given.

Method: maximize/minimize over X = Y (Y^dag Y)^{-1/2} (polar retraction, Y unconstrained,
complex) with L-BFGS from 10 random starts; the relaxed problem uses X = Q diag(sin t) W^dag
(Q isometry, W unitary), which spans exactly the contractions X^dag X <= I.  The relaxed
counterexample is also given explicitly: X = u_1 b_1^dag (top eigenvectors of A and B).

Run:  python rayleigh_quotients.py    (about 6 s; no figure)
"""
import os
import sys
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def herm(M, rng, eigs=None):
    Q, _ = np.linalg.qr(rng.normal(size=(M, M)) + 1j * rng.normal(size=(M, M)))
    lam = rng.normal(size=M) if eigs is None else np.asarray(eigs, float)
    return (Q * lam) @ Q.conj().T


def _unpack(v, M, N):
    return (v[:M * N] + 1j * v[M * N:]).reshape(M, N)


def _isometry(Y):
    w, V = np.linalg.eigh(Y.conj().T @ Y)
    return Y @ (V * w**-0.5) @ V.conj().T


def _contraction(v, M, N):
    """X = Q diag(sin t) W^dag with Q an M x N isometry and W an N x N unitary: every
    contraction X^dag X <= I has this form (SVD), and the boundary is reached at t = pi/2."""
    Q = _isometry(_unpack(v[:2 * M * N], M, N))
    W = _isometry(_unpack(v[2 * M * N:2 * M * N + 2 * N * N], N, N))
    t = v[2 * M * N + 2 * N * N:]
    return (Q * np.sin(t)) @ W.conj().T


def optimize_trace(A, B, sense="max", relaxed=False, starts=10, seed=0):
    """Extremize Tr(X^dag A X B) over isometries (or contractions if relaxed)."""
    M, N = A.shape[0], B.shape[0]
    rng = np.random.default_rng(seed)
    sgn = -1 if sense == "max" else 1
    nv = 2 * M * N + (2 * N * N + N if relaxed else 0)

    def f(v):
        X = _contraction(v, M, N) if relaxed else _isometry(_unpack(v, M, N))
        return sgn * np.real(np.trace(X.conj().T @ A @ X @ B))
    best = None
    for _ in range(starts):
        v0 = rng.normal(size=nv)
        r = minimize(f, v0, method="L-BFGS-B", options=dict(maxiter=3000, gtol=1e-12, ftol=1e-15))
        val = sgn * r.fun
        if best is None or (val > best if sense == "max" else val < best):
            best = val
    return best


def hw_rhs(A, B, kind):
    a = np.sort(np.linalg.eigvalsh(A))
    b = np.sort(np.linalg.eigvalsh(B))[::-1]
    N = len(b)
    return float(a[::-1][:N] @ b) if kind == "down" else float(a[:N] @ b)


def run(M=6, N=3, seed=0):
    rng = np.random.default_rng(seed)
    A = herm(M, rng)
    lam = np.sort(np.linalg.eigvalsh(A))[::-1]
    out = {}
    # Rayleigh quotient and deflation
    x1 = np.linalg.eigh(A)[1][:, -1]
    P = np.eye(M) - np.outer(x1, x1.conj())
    out["rq"] = (optimize_trace(A, np.eye(1), "max"), lam[0])
    out["rq2"] = (optimize_trace(P @ A @ P, np.eye(1), "max"), lam[1])
    # Fan
    out["fan"] = (optimize_trace(A, np.eye(N), "max"), lam[:N].sum())
    # HW with B >= 0
    Bp = herm(N, rng, eigs=rng.uniform(0.1, 2, N))
    out["hw1_psd"] = (optimize_trace(A, Bp, "max"), hw_rhs(A, Bp, "down"))
    out["hw2_psd"] = (optimize_trace(A, Bp, "min"), hw_rhs(A, Bp, "up"))
    # HW1 with indefinite B (N < M): counterexample
    Bi = herm(N, rng, eigs=[1.5, 0.4, -1.0])
    out["hw1_indef"] = (optimize_trace(A, Bi, "max"), hw_rhs(A, Bi, "down"))
    # HW1 with indefinite B but square X (N = M): holds (von Neumann / Ruhe)
    Bs = herm(M, rng)
    out["hw1_square"] = (optimize_trace(A, Bs, "max", starts=6), hw_rhs(A, Bs, "down"))
    # relaxation X^dag X <= I, A with fewer than N positive eigenvalues
    An = herm(M, rng, eigs=[2.0, -0.5, -1.0, -1.5, -2.0, -3.0])
    u1 = np.linalg.eigh(An)[1][:, -1]
    b1 = np.linalg.eigh(Bp)[1][:, -1]
    X = np.outer(u1, b1.conj())                              # X^dag X = b1 b1^dag <= I
    explicit = np.real(np.trace(X.conj().T @ An @ X @ Bp))
    out["hw1_relaxed"] = (optimize_trace(An, Bp, "max", relaxed=True), hw_rhs(An, Bp, "down"),
                          float(np.maximum(np.sort(np.linalg.eigvalsh(An))[::-1][:N], 0)
                                @ np.sort(np.linalg.eigvalsh(Bp))[::-1]), explicit)
    out["hw1_relaxed_psdA"] = (optimize_trace(A @ A, Bp, "max", relaxed=True), hw_rhs(A @ A, Bp, "down"))
    return out


def main():
    print("=" * 84)
    print("Sec. 1.7  Rayleigh quotients, Fan, Hoffman-Wielandt  (M = 6, N = 3, complex X)")
    print("=" * 84)
    r = run()
    fmt = lambda k, lab, rel: print(f"{lab:58s} numeric {r[k][0]:+.8f}   formula {r[k][1]:+.8f}   {rel}")
    fmt("rq", "max x^dag A x / x^dag x  (l.634)", "= lambda_max")
    fmt("rq2", "same, x orthogonal to x_1  (l.640)", "= lambda_2")
    fmt("fan", "[eq:FanThm] max Tr(X^dag A X)", "= sum of top N")
    fmt("hw1_psd", "[eq:HW1] max Tr(X^dag A X B), B > 0", "equality")
    fmt("hw2_psd", "[eq:HW2] min Tr(X^dag A X B), B > 0", "equality")
    fmt("hw1_indef", "[eq:HW1] B indefinite, N < M", "VIOLATED" if r["hw1_indef"][0] > r["hw1_indef"][1] + 1e-8 else "ok")
    fmt("hw1_square", "[eq:HW1] B indefinite, N = M (X unitary)", "holds")
    v, rhs, fixed, expl = r["hw1_relaxed"]
    print(f"{'[eq:HW1] relaxed X^dag X <= I, A with 1 positive eig':58s} numeric {v:+.8f}   formula {rhs:+.8f}"
          f"   VIOLATED")
    print(f"{'':58s} explicit X = u1 b1^dag gives {expl:+.8f} = sum max(lambda_i(A),0) lambda_i(B) = {fixed:+.8f}")
    fmt("hw1_relaxed_psdA", "[eq:HW1] relaxed, A >= 0", "holds")


if __name__ == "__main__":
    main()
