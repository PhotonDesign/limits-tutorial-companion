"""Sec. 3.2: focusing in complex media -- reciprocity, not time reversal.

Model: a random 3D cluster of electric point scatterers (Foldy-Lax), in four flavors
(tutorial_tools/dipoles.py): lossless-reciprocal, lossy-reciprocal, TRI-but-nonreciprocal
(real non-symmetric alpha0), and gyrotropic (lossless, non-reciprocal, non-TRI).
The medium Green's matrix Gamma(x, x') is built directly from the multiple-scattering
T-matrix, without assuming reciprocity.

Part A -- the intensity bound and the optimal currents  [Eqs. CMFocusBound0, CMFocusBound,
          CMTRRecipe, and the generalized-reciprocity recipe of Sec. 3.2.5]
  Electric currents J on an aperture A (open near-field patch, or a closed far-field
  sphere), normalized by int_A |J|^2 dA, focus onto x0 with target polarization phi
  (random complex unit vector).
    * max |phi^dag E(x0)|^2 = || Gamma_{0A}^dag phi ||^2   (rank-one Rayleigh quotient)
    * recipe (Eq. CMTRRecipe): J ∝ conj( E on A radiated by a dipole phi^* at x0, same medium )
      -> efficiency 1 iff the medium is reciprocal (lossy or not); < 1 otherwise;
    * generalized recipe: same, but in the TRANSPOSED medium (alpha0 -> alpha0^T)
      -> efficiency 1 for every medium.
    * closed far-field sphere, lossless medium: the bound equals 2 P_rad (P_rad = power
      radiated by the dipole phi^* at x0, i.e. proportional to the LDOS); a near-field aperture
      exceeds it.

Part B -- partial time reversal on a closed surface  [Eqs. Msintconj, ConjFields]
  A real dipole p0 at x0 radiates in the medium.  On a closed sphere S that encloses all
  scatterers (some within 0.1 lambda of S -- no far-field assumption) we form the Love
  equivalent currents J_s = n x H, M_s = -n x E, negate them (perfectly cancelling currents),
  time-reverse them (J -> -J*, M -> M*), and radiate them alone in the medium.
    * sub-check: the equivalent currents reproduce the field outside S and zero inside;
    * inside S the TR field equals psi^* - Gamma s0^* = -2i Im E  for TRI media (including
      the TRI-but-nonreciprocal one: the construction does not need reciprocity);
    * it fails for lossy or gyrotropic (non-TRI) media;
    * outside S the TR field equals -E (not -2i Im E): Eq. ConjFields holds inside S.

Run: python complex_media_focusing.py   (~30 s)
"""
import os
import sys
import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tutorial_tools.dipoles import DipoleMedium, random_medium, G0_blocks

K = 2 * np.pi          # wavelength = 1
KINDS = ["lossless-reciprocal", "lossy-reciprocal", "tri-nonreciprocal", "gyrotropic"]


def sphere_nodes(R, n_th, n_ph, center=(0, 0, 0)):
    x, w = leggauss(n_th)
    th = np.arccos(x)
    ph = 2 * np.pi * (np.arange(n_ph) + 0.5) / n_ph
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    W = np.outer(w, np.full(n_ph, 2 * np.pi / n_ph)).ravel() * R**2
    TH, PH = TH.ravel(), PH.ravel()
    n = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], 1)
    return np.asarray(center) + R * n, n, W


def patch_nodes(z, half, n):
    x, w = leggauss(n)
    x = half * x; w = half * w
    X, Y = np.meshgrid(x, x, indexing="ij")
    W = np.outer(w, w).ravel()
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
    return pts, W


def make_medium(kind, seed=1, n=16, box=1.0, x0=np.zeros(3), dmin=0.12):
    """Random cluster avoiding a ball of radius dmin around the focal point x0."""
    x, a0 = random_medium(4 * n, box, K, kind, seed=seed)
    keep = np.linalg.norm(x - x0, axis=1) > dmin
    # also enforce a minimum mutual spacing to keep the model well conditioned
    sel = []
    for i in np.where(keep)[0]:
        if all(np.linalg.norm(x[i] - x[j]) > 0.12 for j in sel):
            sel.append(i)
        if len(sel) == n:
            break
    return DipoleMedium(x[sel], a0[sel], K)


def focusing_efficiencies(med, A_pts, A_w, x0, phi):
    Gam = med.greens_E(x0[None, :], A_pts)                 # 3 x 3N
    Aop = Gam * np.repeat(np.sqrt(A_w), 3)[None, :]
    lam = np.linalg.norm(Aop.conj().T @ phi)**2            # rank-one maximum
    # independent check with an iterative eigensolver on the full N x N operator Aop^dag phi phi^dag Aop
    from scipy.sparse.linalg import LinearOperator, eigsh
    n = Aop.shape[1]
    op = LinearOperator((n, n), dtype=complex,
                        matvec=lambda v: Aop.conj().T @ (phi * (phi.conj() @ (Aop @ v))))
    lam_eig = eigsh(op, k=1, which="LA", v0=np.ones(n, complex), tol=1e-12)[0][0]

    def eff(medium_for_recipe):
        Erec = medium_for_recipe.greens_E(A_pts, x0[None, :]) @ np.conj(phi)   # dipole phi* at x0
        u = np.repeat(np.sqrt(A_w), 3) * np.conj(Erec)     # u = sqrt(w) J,  J ∝ conj(E_rec)
        return abs(phi.conj() @ Aop @ u)**2 / np.vdot(u, u).real / lam
    return lam, lam_eig, eff(med), eff(med.transposed())


def radiated_power(med, x0, p):
    """P = (omega/2) Im(p^* . E_total(x0)) for a dipole p at x0 (includes radiation reaction)."""
    k = med.k
    # scattered part k^4 G0(x0,S) T G0(S,x0) p, plus free-space radiation reaction i k^3/(6 pi) p
    GoS = G0_blocks(x0[None, :], med.x, k).transpose(0, 2, 1, 3).reshape(3, -1)
    GSo = G0_blocks(med.x, x0[None, :], k).transpose(0, 2, 1, 3).reshape(-1, 3)
    E = k**4 * GoS @ med.T @ GSo @ p + 1j * k**3 / (6 * np.pi) * p
    return 0.5 * k * np.imag(np.vdot(p, E))


def part_A():
    print("-" * 78)
    print("Part A: focal-intensity bound and optimal currents")
    print("-" * 78)
    rng = np.random.default_rng(7)
    phi = rng.normal(size=3) + 1j * rng.normal(size=3)
    phi /= np.linalg.norm(phi)
    x0 = np.zeros(3)
    sph = sphere_nodes(6.0, 40, 80)
    apertures = {
        "open near-field patch (z = 0.9, 1.2 x 1.2)": patch_nodes(0.9, 0.6, 16),
        "closed sphere R = 6 (far field)": (sph[0], sph[2]),
    }
    print(f"  {'medium':22s} {'aperture':44s} {'recipe eff.':>11s} {'transposed':>11s} {'rank-1 check':>12s}")
    results = {}
    for kind in KINDS:
        med = make_medium(kind)
        for name, (pts, w) in apertures.items():
            lam, lam_eig, e1, e2 = focusing_efficiencies(med, pts, w, x0, phi)
            results[(kind, name)] = (lam, e1, e2)
            print(f"  {kind:22s} {name:44s} {e1:11.8f} {e2:11.8f} {abs(lam_eig/lam-1):12.1e}")
    print("\n  LDOS connection (lossless-reciprocal medium, closed spheres of radius R about x0):")
    med = make_medium("lossless-reciprocal")
    p = np.conj(phi)
    P = radiated_power(med, x0, p)
    for R in [0.06, 0.3, 3.0, 30.0]:   # R = 0.3 cuts through the cluster; any closed/open A is allowed
        if R < 1:
            # small sphere with no scatterers inside (dmin = 0.12 > R): near-field aperture
            pts, n, w = sphere_nodes(R, 40, 80, center=x0)
        else:
            pts, n, w = sphere_nodes(R, 60, 120, center=x0)
        lam, _, _, _ = focusing_efficiencies(med, pts, w, x0, phi)
        print(f"    R = {R:6.2f}: bound = {lam:.6e}   bound / (2 P_rad) = {lam/(2*P):.6f}")
    pts, w = patch_nodes(0.9, 0.6, 16)
    lam, _, _, _ = focusing_efficiencies(med, pts, w, x0, phi)
    print(f"    open near-field patch of Part A: bound / (2 P_rad) = {lam/(2*P):.4f}")
    print("    (-> 1 for a closed far-zone aperture, i.e. bound ∝ radiated power ∝ LDOS; a closed near-field")
    print("     aperture can give much more, while an open aperture captures less)")
    return results


def part_B(n_th=180, n_ph=360, R_S=1.0, box=1.13):
    print("-" * 78)
    print("Part B: partial time reversal on a closed surface S (radius %.2f lambda)" % R_S)
    print("-" * 78)
    rng = np.random.default_rng(11)
    x0 = np.array([0.1, -0.05, 0.08])
    p0 = np.array([0.3, -0.5, 0.8])                        # real dipole
    S_pts, S_n, S_w = sphere_nodes(R_S, n_th, n_ph)
    out = {}
    for kind in KINDS:
        base = make_medium(kind, x0=x0, box=box)
        # add one scatterer only 0.05 lambda inside S (near-field of the surface)
        near = np.array([[0.2, 0.3, 0.0]]); near = near / np.linalg.norm(near) * (R_S - 0.05)
        med = DipoleMedium(np.vstack([base.x, near]), np.concatenate([base.alpha0, base.alpha0[:1]]), K)
        dmin_S = (R_S - np.linalg.norm(med.x, axis=1)).min()
        E_S, H_S = med.fields(S_pts, x0[None, :], p=p0[None, :])
        J = np.cross(S_n, H_S); M = -np.cross(S_n, E_S)
        p_eq = (1j / K) * J * S_w[:, None]
        m_eq = (1j / K) * M * S_w[:, None]
        # evaluation points: inside S (away from scatterers and the source) and outside S
        cand = rng.uniform(-0.8, 0.8, size=(400, 3))
        dist = np.min(np.linalg.norm(cand[:, None, :] - np.vstack([med.x, x0])[None], axis=-1), axis=1)
        inside = cand[(np.linalg.norm(cand, axis=1) < 0.8 * R_S) & (dist > 0.1)][:25]
        outside = rng.normal(size=(15, 3)); outside = outside / np.linalg.norm(outside, axis=1)[:, None] * rng.uniform(1.3, 3, (15, 1))
        E_in, _ = med.fields(inside, x0[None, :], p=p0[None, :])
        E_out, _ = med.fields(outside, x0[None, :], p=p0[None, :])
        # sub-check: equivalence (extinction) theorem
        Eeq_in, _ = med.fields(inside, S_pts, p=p_eq, m=m_eq)
        Eeq_out, _ = med.fields(outside, S_pts, p=p_eq, m=m_eq)
        ext_in = np.abs(Eeq_in).max() / np.abs(E_in).max()
        ext_out = np.abs(Eeq_out - E_out).max() / np.abs(E_out).max()
        # perfectly cancelling currents -> time reverse -> radiate alone in the medium
        p_tr = np.conj(-p_eq)          # p -> p*
        m_tr = -np.conj(-m_eq)         # m -> -m*
        Etr_in, _ = med.fields(inside, S_pts, p=p_tr, m=m_tr)
        Etr_out, _ = med.fields(outside, S_pts, p=p_tr, m=m_tr)
        pred_in = -2j * np.imag(E_in)                         # psi* - Gamma s0*, s0 real
        err_in = np.abs(Etr_in - pred_in).max() / np.abs(pred_in).max()
        err_out = np.abs(Etr_out + E_out).max() / np.abs(E_out).max()
        err_out_imG = np.abs(Etr_out - (-2j * np.imag(E_out))).max() / np.abs(E_out).max()
        out[kind] = (err_in, err_out, ext_in, ext_out)
        print(f"  {kind:22s} min dist scatterer-to-S {dmin_S:5.3f} | extinction thm: inside {ext_in:.1e}, outside {ext_out:.1e}")
        print(f"  {'':22s} TR field vs -2i Im E (inside S): rel. err {err_in:.2e}   | outside S: vs -E {err_out:.1e}, vs -2i Im E {err_out_imG:.2f}")
    return out


def tr_maps(kinds=("tri-nonreciprocal", "gyrotropic"), n_th=120, n_ph=240, R_S=1.0, box=1.13, npix=61):
    """Slice z = z0 through the cluster: |psi_TR|, |2 Im E|, and their difference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tutorial_tools.dipoles import dipole_fields
    x0 = np.array([0.1, -0.05, 0.08])
    p0 = np.array([0.3, -0.5, 0.8])
    S_pts, S_n, S_w = sphere_nodes(R_S, n_th, n_ph)
    g = np.linspace(-0.85, 0.85, npix)
    X, Y = np.meshgrid(g, g, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, x0[2])], 1)
    fig, ax = plt.subplots(len(kinds), 3, figsize=(9, 3 * len(kinds)))
    for row, kind in enumerate(kinds):
        base = make_medium(kind, x0=x0, box=box)
        near = np.array([[0.2, 0.3, 0.0]]); near = near / np.linalg.norm(near) * (R_S - 0.05)
        med = DipoleMedium(np.vstack([base.x, near]), np.concatenate([base.alpha0, base.alpha0[:1]]), K)
        E_S, H_S = med.fields(S_pts, x0[None, :], p=p0[None, :])
        p_eq = (1j / K) * np.cross(S_n, H_S) * S_w[:, None]
        m_eq = (1j / K) * (-np.cross(S_n, E_S)) * S_w[:, None]
        pind_tr = med.induced(S_pts, np.conj(-p_eq), -np.conj(-m_eq))
        E0, _ = dipole_fields(pts, S_pts, np.conj(-p_eq), -np.conj(-m_eq), K, chunk=400)
        E1, _ = dipole_fields(pts, med.x, pind_tr, None, K)
        Etr = E0 + E1
        Eorig, _ = med.fields(pts, x0[None, :], p=p0[None, :])
        pred = -2j * np.imag(Eorig)
        dist = np.min(np.linalg.norm(pts[:, None, :] - np.vstack([med.x, x0])[None], axis=-1), axis=1)
        mask = (dist < 0.06) | (np.linalg.norm(pts, axis=1) > R_S)
        def img(F):
            F = F.copy(); F[mask] = np.nan
            return F.reshape(X.shape)
        a1 = np.linalg.norm(Etr, axis=1); a2 = np.linalg.norm(pred, axis=1)
        d = np.linalg.norm(Etr - pred, axis=1)
        vmax = np.nanpercentile(img(a2), 99)
        for col, (F, ttl) in enumerate([(a1, r"$|\psi_{\rm TR}|$"), (a2, r"$|2\,{\rm Im}\,\mathbf{E}|$"),
                                          (d, r"$|\psi_{\rm TR}+2i\,{\rm Im}\,\mathbf{E}|$")]):
            im = ax[row, col].imshow(img(F), extent=(g[0], g[-1], g[0], g[-1]), origin="lower",
                                     cmap="magma", vmin=0, vmax=vmax)
            ax[row, col].plot(med.x[:, 0], med.x[:, 1], "c.", ms=3)
            ax[row, col].plot(*x0[:2], "w+", ms=8)
            ax[row, col].set_title(ttl + ("" if col else f"   [{kind}]"), fontsize=8)
            ax[row, col].set_xticks([]); ax[row, col].set_yticks([])
            plt.colorbar(im, ax=ax[row, col], fraction=0.046)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "sec3_2_time_reversal_maps")
    for ext in ("pdf", "png"):
        fig.savefig(out + "." + ext, dpi=200)
    print("wrote figures/sec3_2_time_reversal_maps.*")


def main():
    print("=" * 78)
    print("Sec. 3.2: complex-media focusing with a 3D coupled-dipole medium (lambda = 1)")
    print("=" * 78)
    part_A()
    part_B()
    tr_maps()


if __name__ == "__main__":
    main()
