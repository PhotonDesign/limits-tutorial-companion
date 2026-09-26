"""Point-dipole (Foldy-Lax / coupled-dipole) model of a 3D "complex medium".

Units: eps0 = mu0 = c = Z0 = 1, omega = k, time dependence e^{-i omega t}.
Sources are electric dipoles p (current J = -i omega p delta) and magnetic dipoles m
(magnetic current M = -i omega mu0 m delta, with curl E = i omega mu0 H - M).

Free-space fields of a dipole at the origin, r = |x|, g = e^{ikr}/(4 pi r):
    E_p = k^2 G0 p,             H_p = k^2 (1 + i/(kr)) g  (rhat x p)
    E_m = -k^2 (1 + i/(kr)) g (rhat x m),    H_m = k^2 G0 m
    G0  = g [ (1 + i/kr - 1/(kr)^2) I + (-1 - 3i/kr + 3/(kr)^2) rhat rhat ].

Scatterers are electric point dipoles with 3x3 polarizability tensors, p_j = alpha_j E_loc(x_j),
alpha_j^{-1} = alpha0_j^{-1} - i k^3/(6 pi) I  (radiation reaction, from Im G0(0) = k/(6 pi) I).
Symmetries of the medium follow from the "bare" alpha0 (see derivations, Sec. 7):
    reciprocal  <=>  alpha0 = alpha0^T
    lossless    <=>  alpha0 = alpha0^dagger
    TRI         <=>  alpha0 real (time reversal: E -> E*, H -> -H*, p -> p*, m -> -m*)
"""
import numpy as np


def _geom(obs, src):
    d = obs[:, None, :] - src[None, :, :]
    r = np.linalg.norm(d, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rh = d / r[..., None]
    return r, rh


def G0_blocks(obs, src, k):
    """(No, Ns, 3, 3) array of G0(obs_i - src_j)."""
    r, rh = _geom(obs, src)
    kr = k * r
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.exp(1j * kr) / (4 * np.pi * r)
        A = g * (1 + 1j / kr - 1 / kr**2)
        B = g * (-1 - 3j / kr + 3 / kr**2)
    return A[..., None, None] * np.eye(3) + B[..., None, None] * rh[..., :, None] * rh[..., None, :]


def cross_blocks(obs, src, k):
    """(No, Ns, 3, 3) array C with C v = k^2 (1 + i/(kr)) g (rhat x v)."""
    r, rh = _geom(obs, src)
    kr = k * r
    f = k**2 * (1 + 1j / kr) * np.exp(1j * kr) / (4 * np.pi * r)
    X = np.zeros(rh.shape + (3,), complex)
    X[..., 0, 1] = -rh[..., 2]; X[..., 0, 2] = rh[..., 1]
    X[..., 1, 0] = rh[..., 2];  X[..., 1, 2] = -rh[..., 0]
    X[..., 2, 0] = -rh[..., 1]; X[..., 2, 1] = rh[..., 0]
    return f[..., None, None] * X


def dipole_fields(obs, src, p=None, m=None, k=1.0, chunk=2000):
    """E, H at obs (No,3) from electric dipoles p (Ns,3) and/or magnetic dipoles m at src."""
    E = np.zeros((obs.shape[0], 3), complex)
    H = np.zeros((obs.shape[0], 3), complex)
    for j0 in range(0, src.shape[0], chunk):
        sl = slice(j0, j0 + chunk)
        G = G0_blocks(obs, src[sl], k)
        C = cross_blocks(obs, src[sl], k)
        if p is not None:
            E += k**2 * np.einsum("ijab,jb->ia", G, p[sl])
            H += np.einsum("ijab,jb->ia", C, p[sl])
        if m is not None:
            E -= np.einsum("ijab,jb->ia", C, m[sl])
            H += k**2 * np.einsum("ijab,jb->ia", G, m[sl])
    return E, H


class DipoleMedium:
    """Collection of electric point scatterers embedded in vacuum."""

    def __init__(self, positions, alpha0, k=1.0):
        self.x = np.asarray(positions, float)
        self.k = k
        self.alpha0 = np.asarray(alpha0, complex)            # (Ns,3,3)
        n = self.x.shape[0]
        ainv = np.linalg.inv(self.alpha0) - 1j * k**3 / (6 * np.pi) * np.eye(3)
        Gss = G0_blocks(self.x, self.x + 0.0, k) if n > 1 else np.zeros((n, n, 3, 3))
        Gss = np.nan_to_num(Gss)                              # diagonal blocks are singular
        for i in range(n):
            Gss[i, i] = 0.0
        Minv = -k**2 * Gss
        for i in range(n):
            Minv[i, i] += ainv[i]
        self.Mmat = Minv.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
        self.T = np.linalg.inv(self.Mmat)                     # p = T E_inc (stacked)

    def transposed(self):
        return DipoleMedium(self.x, np.transpose(self.alpha0, (0, 2, 1)), self.k)

    def induced(self, src, p=None, m=None):
        Einc, _ = dipole_fields(self.x, src, p, m, self.k)
        return (self.T @ Einc.ravel()).reshape(-1, 3)

    def fields(self, obs, src, p=None, m=None):
        """Total E, H at obs due to source dipoles, including multiple scattering."""
        pind = self.induced(src, p, m)
        E0, H0 = dipole_fields(obs, src, p, m, self.k)
        E1, H1 = dipole_fields(obs, self.x, pind, None, self.k)
        return E0 + E1, H0 + H1

    def greens_E(self, obs, src):
        """Medium Green's matrix Gamma (3No x 3Ns): E(obs) = Gamma p(src) for electric dipoles
        at src, built WITHOUT invoking reciprocity: k^2 [G0(o,s) + k^2 G0(o,S) T G0(S,s)]."""
        k = self.k
        Gos = G0_blocks(obs, src, k).transpose(0, 2, 1, 3).reshape(3 * obs.shape[0], 3 * src.shape[0])
        GoS = G0_blocks(obs, self.x, k).transpose(0, 2, 1, 3).reshape(3 * obs.shape[0], 3 * self.x.shape[0])
        GSs = G0_blocks(self.x, src, k).transpose(0, 2, 1, 3).reshape(3 * self.x.shape[0], 3 * src.shape[0])
        return k**2 * (Gos + k**2 * GoS @ self.T @ GSs)


def random_medium(n, box, k, kind, seed=0, strength=0.6):
    """Random cluster of n electric point scatterers in a cube of side 'box' (centered).
    kind: 'lossless-reciprocal' (alpha0 real symmetric), 'lossy-reciprocal' (complex
    symmetric, Im > 0), 'tri-nonreciprocal' (real, non-symmetric), 'gyrotropic'
    (Hermitian, non-symmetric: lossless magneto-optic)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-box / 2, box / 2, size=(n, 3))
    a = strength * 6 * np.pi / k**3            # scale ~ radiation-limited polarizability
    alpha0 = np.zeros((n, 3, 3), complex)
    for j in range(n):
        R = rng.normal(size=(3, 3))
        S = (R + R.T) / 2 + 2 * np.eye(3)
        if kind == "lossless-reciprocal":
            A = S
        elif kind == "lossy-reciprocal":
            A = S + 1j * (0.8 * np.eye(3) + 0.1 * (R + R.T))
        elif kind == "tri-nonreciprocal":
            Q = rng.normal(size=(3, 3))
            A = S + 0.8 * (Q - Q.T)
        elif kind == "gyrotropic":
            Q = rng.normal(size=(3, 3))
            A = S + 0.8j * (Q - Q.T)
        else:
            raise ValueError(kind)
        alpha0[j] = a * A
    return x, alpha0
