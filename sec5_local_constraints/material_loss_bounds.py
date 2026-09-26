"""Sec. 5.1: single-frequency cross-section bounds from the optical theorem.

Tutorial claims checked (tutorial.tex line numbers, equation labels):
  l.1683 [CSBoundML]  sigma_{abs,ext}/V <= k |chi|^2/Im chi,  sigma_scat/V <= (1/4) k |chi|^2/Im chi.
          Test: quasi-static spheroids, sigma_abs/V = k Im[chi/(1 + L chi)]; the bound is reached
          exactly at Re(1/chi) = -L; Drude-metal spheroids sweep out the envelope.
          With the radiative correction the bounds still hold.
  l.1708 [CSvswChannelBound] (sigma_{sca,ext})_{np} <= (2n+1) lambda^2/2pi,
          (sigma_abs)_{np} <= (1/4)(2n+1) lambda^2/2pi  -- checked with Mie coefficients.
  l.1713 [CSvswBoundFinite] sum_{n<=N}(2n+1) = N^2 + 2N; l.1718 [CSvswChannelBound2] n = 1 values.
  l.1739 [CSOTBound]  sigma_ext <= (lambda^2/2pi) sum_{n,p} (2n+1) rho_{np}/(rho_{np} + Im xi),
          with rho_{np} the eigenvalues of Im Gamma_0 on the bounding sphere (mie.rho_ImGamma0).
          (a) normalization validated by an independent plane-wave identity;
          (b) in the dipole limit rho = k^3 V/6pi it is saturated EXACTLY by a resonant
              radiatively-corrected ellipsoid;  (c) never exceeded by Mie spheres.
  l.1753 [DualBounds] absorption / scattering dual bounds: in the dipole limit they are
          saturated exactly by resonant radiatively-corrected ellipsoids (absorption when
          Im xi >= rho); never exceeded by Mie spheres.
  l.1742  large-loss / small-volume limit statement ("only the first term is retained").

Run:  python material_loss_bounds.py   (about 5 s; writes figures/sec5_material_loss_bounds.png/pdf)
"""
import os
import sys
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sec5_local_constraints.mie import cross_sections, rho_ImGamma0, planewave_ImGamma0_quadform, n_max
from sec5_local_constraints.extinction_sum_rules import DrudeLorentz, depolarization_prolate, HC_EVNM

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
HBARC = HC_EVNM / (2 * np.pi)                       # eV nm
DRUDE_AG = DrudeLorentz(wD=9.0, gD=0.05)            # simple Drude metal (eV)


# ---------------------------------------------------------------- quasi-static ellipsoids
def qs_cross_sections(chi, L, k, V, radiative=True):
    """Per-volume ext/abs/scat for a quasi-static ellipsoid (field along the axis with
    depolarization L), optionally radiatively corrected.  k, V in consistent units."""
    alpha = V * chi / (1 + L * chi)
    if radiative:
        alpha = alpha / (1 - 1j * k ** 3 * alpha / (6 * np.pi))
        sca = k ** 4 * abs(alpha) ** 2 / (6 * np.pi)
    else:
        sca = 0 * alpha.real
    ext = k * alpha.imag
    return ext / V, (ext - sca) / V, sca / V


def material_bound(chi, k):
    return k * abs(chi) ** 2 / chi.imag


def envelope_check(w_eV=np.linspace(1.0, 6.2, 400), Ls=np.linspace(0.005, 0.995, 600)):
    """Max over L of sigma_abs/V (no radiative correction) divided by the bound, and the
    same at the resonant L* = -Re(1/chi).  Returns arrays over frequency."""
    chi = DRUDE_AG.chi(w_eV)
    k = w_eV / HBARC
    b = material_bound(chi, k)
    s = np.array([qs_cross_sections(chi, L, k, 1.0, radiative=False)[1] for L in Ls])
    ratio_grid = s.max(0) / b
    Lstar = -np.real(1 / chi)
    ok = (Lstar > 0) & (Lstar < 1)
    ratio_star = qs_cross_sections(chi[ok], Lstar[ok], k[ok], 1.0, radiative=False)[1] / b[ok]
    return w_eV, ratio_grid, ok, ratio_star, s, b


def dipole_limit_checks(chi, kV_rho):
    """Dimensionless (k = 1) dipole-limit comparisons at fixed chi and rho = k^3 V/6pi.
    Returns max-over-L of radiatively corrected ext/abs/scat vs CSOTBound / DualBounds."""
    k = 1.0
    V = 6 * np.pi * kV_rho
    Imxi = chi.imag / abs(chi) ** 2
    rho = kV_rho
    C = 6 * np.pi / k ** 2                        # 3 lambda^2/2pi
    Lstar = -np.real(1 / chi)                     # resonance (assumed in (0,1))
    ext, ab, sc = qs_cross_sections(chi, Lstar, k, V)
    Ls = np.linspace(1e-4, 1 - 1e-4, 20001)
    e2, a2, s2 = qs_cross_sections(chi, Ls, k, V)
    ot_ext = C * rho / (rho + Imxi)
    dual_abs = dual_bound(np.array([rho]), np.array([C]), Imxi, "abs")
    dual_sca = dual_bound(np.array([rho]), np.array([C]), Imxi, "sca")
    return dict(ext_max=V * e2.max(), ext_res=V * ext, ot_ext=ot_ext,
                abs_max=V * a2.max(), dual_abs=dual_abs, sca_max=V * s2.max(), dual_sca=dual_sca,
                Imxi=Imxi, rho=rho, abs_channel=C / 4, mat_abs=k * V / Imxi)


# ---------------------------------------------------------------- spherical-domain bounds
def channel_weights(kR, nmax):
    n = np.arange(1, nmax + 1)
    rTE, rTM = rho_ImGamma0(kR, nmax)
    c = 2 * np.pi * (2 * n + 1)                   # (lambda^2/2pi)(2n+1) with k = 1
    return np.concatenate([c, c]), np.concatenate([rTE, rTM])


def ot_ext_bound(kR, Imxi, nmax=None, per_channel=False):
    """Eq. CSOTBound (k = 1)."""
    nmax = nmax or n_max(kR) + 10
    c, rho = channel_weights(kR, nmax)
    terms = c * rho / (rho + Imxi)
    return terms if per_channel else terms.sum()


def dual_bound(rho, c, Imxi, kind):
    """Eqs. DualBounds, minimized over the dual variable nu (k = 1 units)."""
    if kind == "abs":
        g = lambda nu: (nu ** 2 / 4) * np.sum(c * rho / (nu * rho + (nu - 1) * Imxi))
        lo = 1.0
    else:
        g = lambda nu: (nu ** 2 / 4) * np.sum(c * rho / ((nu - 1) * rho + nu * Imxi))
        lo = np.max(rho / (rho + Imxi))
    # log-parametrize nu - lo; a single-channel abs problem can have its infimum at nu -> 1
    f = lambda t: g(lo + np.exp(t))
    res = minimize_scalar(f, bounds=(-30, 8), method="bounded", options=dict(xatol=1e-12))
    return min(res.fun, f(-30.0))


def mie_bound_sweep(chi, kRs):
    """Mie sphere (m = sqrt(1+chi)) vs CSOTBound, DualBounds (abs, sca), per-channel
    CSOTBound, and unitary per-channel bounds.  Returns dict of max ratios."""
    m = np.sqrt(1 + chi + 0j)
    m = m if m.imag >= 0 else -m
    Imxi = chi.imag / abs(chi) ** 2
    out = dict(ext=[], abs=[], sca=[], chan_ot=[], chan_unit_sca=[], chan_unit_abs=[])
    for kR in kRs:
        nmax = n_max(kR) + 10
        cs = cross_sections(m, kR, nmax=nmax, per_channel=True)
        c, rho = channel_weights(kR, nmax)
        out["ext"].append(cs["ext"][0] / ot_ext_bound(kR, Imxi, nmax))
        out["abs"].append(cs["abs"][0] / dual_bound(rho, c, Imxi, "abs"))
        out["sca"].append(cs["sca"][0] / dual_bound(rho, c, Imxi, "sca"))
        ch_ext = np.concatenate([cs["ext_TE"][0], cs["ext_TM"][0]])
        out["chan_ot"].append(np.max(ch_ext / (c * rho / (rho + Imxi))))
        ch_sca = np.concatenate([cs["sca_TE"][0], cs["sca_TM"][0]])
        ch_abs = np.concatenate([cs["abs_TE"][0], cs["abs_TM"][0]])
        out["chan_unit_sca"].append(np.max(ch_sca / c))
        out["chan_unit_abs"].append(np.max(ch_abs / (c / 4)))
    return {k: np.array(v) for k, v in out.items()}


def unitary_channel_saturation():
    """Max over a sweep of lossless / lossy spheres of the per-channel cross-sections
    in units of (2n+1) lambda^2/2pi: scattering -> 1, absorption -> 1/4."""
    xs = np.linspace(0.2, 8, 1600)
    best_sca = 0.0
    for mr in (1.5, 2.0, 3.0, 4.0):
        cs = cross_sections(np.full_like(xs, mr, dtype=complex), xs, per_channel=True)
        n = np.arange(1, cs["sca_TM"].shape[-1] + 1)
        c = 2 * np.pi * (2 * n + 1)
        best_sca = max(best_sca, (cs["sca_TM"] / c).max(), (cs["sca_TE"] / c).max(),
                       (cs["ext_TM"] / c).max())
    # absorption: n = 1 TE (magnetic dipole) channel of a high-index lossy sphere near its
    # first Mie resonance; scan loss and size to find critical coupling (a_n = 1/2)
    best_abs = 0.0
    for mi in np.linspace(0.005, 0.4, 80):
        x = np.linspace(0.6, 1.4, 600)
        cs = cross_sections(np.full_like(x, 3.5 + 1j * mi, dtype=complex), x, per_channel=True)
        best_abs = max(best_abs, (cs["abs_TE"][:, 0] / (6 * np.pi)).max())
    return best_sca, best_abs


def main(make_figure=True):
    print("=" * 86)
    print("Sec. 5.1 material-loss, channel, and optical-theorem bounds")
    print("=" * 86)

    w, ratio_grid, ok, ratio_star, s, b = envelope_check()
    print("A. Quasi-static Drude spheroids (wp = 9 eV, gamma = 0.05 eV), sigma_abs/V = k Im[chi/(1+L chi)]:")
    print(f"   max over (w, L grid) of (sigma_abs/V) / (k|chi|^2/Im chi) = {ratio_grid.max():.6f}  (never above 1)")
    print(f"   at resonance L* = -Re(1/chi) in (0,1) ({ok.sum()} of {len(w)} frequencies, {w[ok].min():.2f}-{w[ok].max():.2f} eV):"
          f" ratio = 1 to {np.abs(ratio_star - 1).max():.1e}")
    # radiatively corrected, finite size
    worst = dict(ext=0, abs=0, sca=0)
    for bnm in (5.0, 20.0, 40.0):
        for asp in (1, 2, 3, 5, 8):
            V = 4 * np.pi * asp * bnm ** 3 / 3
            L = depolarization_prolate(asp * bnm, bnm)
            chi = DRUDE_AG.chi(w); k = w / HBARC
            e, a, sc = qs_cross_sections(chi, L, k, V, radiative=True)
            bb = material_bound(chi, k)
            worst["ext"] = max(worst["ext"], (e / bb).max())
            worst["abs"] = max(worst["abs"], (a / bb).max())
            worst["sca"] = max(worst["sca"], (sc / (bb / 4)).max())
    print(f"   with radiative correction (b = 5-40 nm, aspect 1-8): max ratios ext {worst['ext']:.4f}, abs {worst['abs']:.4f},"
          f" scat/(bound/4) {worst['sca']:.4f}  (all <= 1)")

    print("\nB. Dipole limit (k = 1): resonant radiatively-corrected ellipsoid vs CSOTBound / DualBounds, 3 lambda^2/2pi units:")
    print(f"   {'chi':>12s} {'rho':>7s} {'Im xi':>7s} | {'ext max':>9s} {'CSOTBound':>9s} | {'abs max':>9s} {'dual abs':>9s} | {'sca max':>9s} {'dual sca':>9s}")
    C = 6 * np.pi
    drows = []
    for chi, rho in ((-5 + 0.5j, 0.01), (-5 + 0.5j, 0.1), (-3 + 0.05j, 0.1), (-3 + 1.5j, 0.02), (-8 + 3j, 0.5)):
        d = dipole_limit_checks(chi, rho)
        drows.append(d)
        print(f"   {str(chi):>12s} {rho:7.3f} {d['Imxi']:7.4f} | {d['ext_max'] / C:9.6f} {d['ot_ext'] / C:9.6f} |"
              f" {d['abs_max'] / C:9.6f} {d['dual_abs'] / C:9.6f} | {d['sca_max'] / C:9.6f} {d['dual_sca'] / C:9.6f}")
    print("   -> CSOTBound and the scattering dual bound are saturated exactly; the absorption dual bound")
    print("      is saturated when Im xi >= rho and otherwise equals the channel limit 1/4 (not reachable by a single dipole).")

    print("\n   l.1758 'DualBounds simplify to their material-only ... counterparts': small-rho limit at Im xi = 0.1:")
    for rho in (1e-1, 1e-2, 1e-3, 1e-4):
        da = dual_bound(np.array([rho]), np.array([C]), 0.1, "abs")
        dsc = dual_bound(np.array([rho]), np.array([C]), 0.1, "sca")
        mat = C * rho / 0.1                                     # = k V |chi|^2/Im chi
        print(f"   rho = {rho:.0e}: dual abs / material abs bound = {da / mat:.4f};"
              f"  dual sca / material sca bound (1/4) = {dsc / (mat / 4):.2e}  (= 4 rho/(rho+Im xi)^2 * Im xi)")
    print("   -> absorption: yes; scattering: the dual bound scales as V^2 (tighter), never as the beta = 1/4 material bound.")

    print("\nC. Normalization of rho_{np} and plane-wave channel weights (independent identity, k = 1):")
    for kR in (0.5, 2.0, 5.0):
        nmax = n_max(kR) + 20
        n = np.arange(1, nmax + 1)
        rTE, rTM = rho_ImGamma0(kR, nmax)
        lhs = planewave_ImGamma0_quadform(kR)
        rhs = 2 * np.pi * np.sum((2 * n + 1) * (rTE ** 2 + rTM ** 2))
        c, rho = channel_weights(kR, nmax)
        print(f"   kR = {kR:3.1f}: psi^dag ImG0 psi = {lhs:.10e} vs (2pi) sum (2n+1)(rho_TE^2+rho_TM^2) = {rhs:.10e};"
              f"  sum c_n rho_n / kV = {np.sum(c * rho) / (4 * np.pi * kR ** 3 / 3):.10f}")

    print("\nD. Mie spheres vs spherical-domain bounds (max over kR in [0.05, 6] of sigma/bound, must be <= 1):")
    kRs = np.concatenate([np.geomspace(0.05, 1, 30), np.linspace(1.05, 6, 60)])
    for chi in (-4 + 0.3j, -2.2 + 0.1j, -8 + 1j, 3 + 0.5j, 11 + 0.05j):
        r = mie_bound_sweep(chi, kRs)
        print(f"   chi = {str(chi):>12s}: ext/CSOTBound {r['ext'].max():.4f}, abs/dual {r['abs'].max():.4f},"
              f" sca/dual {r['sca'].max():.4f}, per-channel ext/OT {r['chan_ot'].max():.6f},"
              f" chan sca/unit {r['chan_unit_sca'].max():.4f}, chan abs/(unit/4) {r['chan_unit_abs'].max():.4f}")

    bs, ba = unitary_channel_saturation()
    print("\nE. Per-channel unitary bounds [CSvswChannelBound, l.1708] with Mie coefficients (units (2n+1) lambda^2/2pi):")
    print(f"   max over lossless spheres of sigma_sca,np (and ext) = {bs:.6f}  (bound 1, reached at Mie resonances)")
    print(f"   max over lossy (m = 3.5 + i m'') n = 1 TE absorption = {ba:.6f}  (bound 1/4 = 3 lambda^2/8pi / (3 lambda^2/2pi))")
    N = np.arange(1, 8)
    print(f"   sum_(n<=N)(2n+1) - (N^2+2N) for N = 1..7: {[int(sum(2 * n + 1 for n in range(1, M + 1)) - (M * M + 2 * M)) for M in N]}"
          f";  N_TM = 3, N_TE = 0 -> {3 ** 2 + 2 * 3} lambda^2/2pi (Fig. SuperScat caption: 15)")

    print("\nF. l.1742 'large material loss ... only the first term in CSOTBound is retained':")
    for kR in (0.2, 1.0, 2.0, 4.0):
        Imxi = 1e4
        terms = ot_ext_bound(kR, Imxi, per_channel=True)
        V = 4 * np.pi * kR ** 3 / 3
        nmx = len(terms) // 2
        first = terms[nmx]                       # n = 1 TM term
        print(f"   kR = {kR:3.1f}, Im xi = 1e4: full sum / (kV/Im xi) = {terms.sum() / (V / Imxi):.6f};"
              f"  n=1 TM term alone / (kV/Im xi) = {first / (V / Imxi):.4f}")
    print("   -> at large loss every term becomes (2n+1) rho/Im xi and the SUM gives the material bound;")
    print("      the n = 1 term alone suffices only for small volume (kR << 1).")

    if make_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
        chi = DRUDE_AG.chi(w); k = w / HBARC
        for asp in (1, 2, 3, 5, 8):
            L = depolarization_prolate(asp, 1.0)
            a = qs_cross_sections(chi, L, k, 1.0, radiative=False)[1]
            ax[0].semilogy(w, a * 1e3, lw=1, label=f"aspect {asp}:1")
        ax[0].semilogy(w, material_bound(chi, k) * 1e3, "k", lw=2, label=r"$k|\chi|^2/\mathrm{Im}\,\chi$")
        ax[0].set_xlabel(r"$\hbar\omega$ (eV)"); ax[0].set_ylabel(r"$\sigma_{\rm abs}/V$ ($\mu$m$^{-1}$)")
        ax[0].set_title("Drude prolate spheroids (quasi-static)", fontsize=9)
        ax[0].legend(fontsize=7, frameon=False); ax[0].set_ylim(1e-1, 1e4)
        chi1 = -4 + 0.3j
        Imxi = chi1.imag / abs(chi1) ** 2
        kRs2 = np.geomspace(0.05, 6, 200)
        m = np.sqrt(1 + chi1)
        mie = np.array([cross_sections(m, x)["ext"][0] for x in kRs2])
        bnd = np.array([ot_ext_bound(x, Imxi) for x in kRs2])
        chan = np.array([2 * np.pi * (n_max(x) ** 2 + 2 * n_max(x)) for x in kRs2])
        matb = 4 * np.pi * kRs2 ** 3 / 3 / Imxi
        ax[1].loglog(kRs2, mie / (2 * np.pi), "C2", label=r"Mie sphere, $\chi=-4+0.3i$")
        ax[1].loglog(kRs2, bnd / (2 * np.pi), "k", lw=2, label="CSOTBound")
        ax[1].loglog(kRs2, matb / (2 * np.pi), "C0--", label=r"material bound $kV|\chi|^2/\mathrm{Im}\chi$")
        ax[1].set_xlabel(r"$kR$"); ax[1].set_ylabel(r"$\sigma_{\rm ext}/(\lambda^2/2\pi)$")
        ax[1].set_ylim(1e-4, 1e3); ax[1].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        os.makedirs(FIGDIR, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIGDIR, f"sec5_material_loss_bounds.{ext}"), dpi=200)
        print(f"\nwrote {os.path.normpath(os.path.join(FIGDIR, 'sec5_material_loss_bounds.pdf'))}")


if __name__ == "__main__":
    main()
