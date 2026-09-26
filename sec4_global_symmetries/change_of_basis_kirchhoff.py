"""Sec. 4.3: scattering-matrix change of basis and the Kirchhoff's-law example.

Tutorial claims checked (tutorial.tex line numbers as of 2026-09):
  1. [~1508-1512, eq:STransform] S_2 = U^T S_1 U for a unitary change of (incoming) basis U.
  2. [~1514-1522] lossless, reciprocal, and TRI (S^* S = I) are preserved by eq:STransform.
  3. [~1524-1530] singular values are invariant (S_2^dag S_2 = U^dag S_1^dag S_1 U).
  4. [~1542-1590, eq:S1Matrices, eq:S2Matrices] explicit matrices
        S_1 = [[0,1],[0,0]], A_1 = diag(1,0), E_1 = diag(0,1);
        U = [[1,i],[i,1]]/sqrt2 (circular basis);
        S_2 = [[i,1],[-1,i]]/2,  A_2 = E_2 = [[1,i],[-i,1]]/2  (diagonals 1/2 = Kirchhoff).
  5. [~1594] |S2_21| = |S2_12| ("reciprocal in amplitude") but S_2 != S_2^T.

Run:  python change_of_basis_kirchhoff.py   (< 1 s; no figure)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from tr_field_ratio_bounds import haar_unitary  # noqa: E402

I2 = np.eye(2)
S1 = np.array([[0, 1], [0, 0]], complex)
U_CIRC = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
S2_CLAIM = np.array([[1j, 1], [-1, 1j]]) / 2
A1_CLAIM, E1_CLAIM = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
A2_CLAIM = E2_CLAIM = np.array([[1, 1j], [-1j, 1]]) / 2


def transform(S, U):
    return U.T @ S @ U


def absorptivity(S):
    return np.eye(S.shape[0]) - S.conj().T @ S


def emissivity(S):
    return np.eye(S.shape[0]) - S @ S.conj().T


def kirchhoff_example():
    S2 = transform(S1, U_CIRC)
    return dict(S2=S2, A1=absorptivity(S1), E1=emissivity(S1),
                A2=absorptivity(S2), E2=emissivity(S2))


def symmetry_preservation(n=5, trials=200, seed=0):
    """Max violation of lossless / reciprocal / TRI / singular-value invariance after the
    transform, for random S_1 having each property and random unitary U."""
    rng = np.random.default_rng(seed)
    worst = dict(lossless=0.0, reciprocal=0.0, TRI=0.0, svals=0.0)
    for _ in range(trials):
        U = haar_unitary(n, rng)
        V = haar_unitary(n, rng)
        Sl = haar_unitary(n, rng)                              # lossless, not reciprocal
        A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        Sr = (A + A.T) / 2                                     # reciprocal, lossy/gainy
        St = V @ V.T                                           # TRI (symmetric unitary)
        I = np.eye(n)
        worst["lossless"] = max(worst["lossless"], np.abs(transform(Sl, U).conj().T @ transform(Sl, U) - I).max())
        S2r = transform(Sr, U)
        worst["reciprocal"] = max(worst["reciprocal"], np.abs(S2r - S2r.T).max())
        S2t = transform(St, U)
        worst["TRI"] = max(worst["TRI"], np.abs(S2t.conj() @ S2t - I).max())
        worst["svals"] = max(worst["svals"], np.abs(np.linalg.svd(Sr, compute_uv=False)
                                                     - np.linalg.svd(S2r, compute_uv=False)).max())
    return worst


def main():
    print("=" * 78)
    print("Sec. 4.3  change of basis S_2 = U^T S_1 U and the Kirchhoff example")
    print("=" * 78)
    ex = kirchhoff_example()
    for key, claim in [("A1", A1_CLAIM), ("E1", E1_CLAIM), ("S2", S2_CLAIM), ("A2", A2_CLAIM), ("E2", E2_CLAIM)]:
        print(f"  {key}: max |computed - claimed| = {np.abs(ex[key] - claim).max():.1e}")
    S2 = ex["S2"]
    print(f"  diag(A2) = {np.real(np.diag(ex['A2']))}, diag(E2) = {np.real(np.diag(ex['E2']))}  (claim 1/2, 1/2)")
    print(f"  |S2_12| = {abs(S2[0, 1]):.3f}, |S2_21| = {abs(S2[1, 0]):.3f};  S2 symmetric? {np.allclose(S2, S2.T)}")
    c1, c2 = U_CIRC[:, 0], U_CIRC[:, 1]
    print(f"  U columns: {np.round(c1, 3)}, {np.round(c2, 3)} -> y/x phase +90 deg and -90 deg"
          f" (opposite circular handedness): {np.isclose(c1[1] / c1[0], 1j) and np.isclose(c2[1] / c2[0], -1j)}")
    print(f"  singular values S1 {np.linalg.svd(S1, compute_uv=False)}, S2 {np.round(np.linalg.svd(S2, compute_uv=False), 12)}")
    w = symmetry_preservation()
    print("  random 5x5 tests (200 each), max violation after transform:",
          ", ".join(f"{k} {v:.1e}" for k, v in w.items()))
    return ex, w


if __name__ == "__main__":
    main()
