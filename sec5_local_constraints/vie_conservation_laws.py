"""Secs. 2.x / 5.1 / 5.3: global and local (D-matrix) conservation laws of the volume
integral equation, checked on a discretized 3D coupled-dipole (point-dipole VIE) model.

Model (tutorial Eq. IE / IE2, e^{-i omega t}, units eps0 = mu0 = c = 1, so omega = k):
    (Gamma_0 + xi) phi = -psi_inc,   xi = -chi^{-1},   phi = polarization density,
    Gamma_0[i,j] = v k^2 G(x_i - x_j)  (i != j),  Gamma_0[i,i] = (-1/3 + i k^3 v/6pi) I,
where G is the free-space electric dyadic, v the pixel volume.  Im Gamma_0 is exactly v
times the (positive-definite) kernel Im k^2 G sampled on the lattice.  "Re"/"Im" of a
matrix are its Hermitian / anti-Hermitian parts, as in the tutorial.

Checks (tutorial line numbers refer to tutorial.tex):
  1. Optical theorem [OpticalThm, OpticalThm2, TwoConstraints(a)], l.587/1666/1765:
        phi^dag (Im G0 + Im xi) phi = Im(psi^dag phi)            -> holds to 1e-15.
  2. Reactive-power law [ReacOpticalThm, TwoConstraints(b)], l.610/1766, AS PRINTED:
        phi^dag (Re G0 + Re xi) phi = +Re(psi^dag phi)            -> FAILS (O(1));
     with the sign fixed, = -Re(psi^dag phi)                      -> holds to 1e-15.
  3. Footnote l.1677: P_scat = (w/2) Im[psi^dag phi - phi^dag (Im xi) phi] AS PRINTED
     equals P_ext (Im of a real number is 0); the correct expression is
     (w/2)[Im(psi^dag phi) - phi^dag (Im xi) phi].  Demonstrated on phi_max, l.1677.
  4. P_scat = (w/2) phi^dag Im G0 phi equals the far-field Poynting flux (l.592), and
     P_ext = (w/2) Im(psi^dag phi) equals the forward-amplitude optical theorem (l.603).
     P_abs = (w/2) phi^dag Im xi phi = (w/2) psi_tot^dag Im chi psi_tot (l.596-597).
  5. Local constraints [ConsDMatrix, GenConstraint, GenConstraint2], l.617/1930/1935:
     phi^dag D (G0 + xi) phi = -phi^dag D psi for every pixel-local 3x3 D (9 per pixel),
     which sum to the global laws.
  6. Domain obliviousness (l.1938-1940): with a design domain V_d larger than the
     scatterer V_s, the IE residual is nonzero on V_d \\ V_s, but every *local* D
     constraint written on V_d still holds; a *nonlocal* D (coupling x in V_s to
     x' outside V_s) fails.
  7. Constraint count (l.1965): N_s * N_ns^2 = 9 N_s linearly independent local D's
     (lifted-matrix rank).
  8. Material-loss bounds [CSBoundML] and the full-optical-theorem extinction bound
     (l.1735-1737) hold for the solved scatterer; phi_opt saturates the constraint.

Run:  python vie_conservation_laws.py   (about 2 s; writes figures/sec5_vie_conservation.png/pdf)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


# ----------------------------------------------------------------------------- model
def ReH(A):
    """Hermitian part ("Re" of an operator in the tutorial)."""
    return 0.5 * (A + A.conj().T)


def ImH(A):
    """Anti-Hermitian part divided by i ("Im" of an operator in the tutorial)."""
    return (A - A.conj().T) / 2j


def greens_E(r, k):
    """Free-space electric dyadic k^2 G(r) (field at r from unit dipole at 0), eps0=1:
    e^{ikR}/(4 pi R) [ (k^2 + ik/R - 1/R^2) I + (-k^2 - 3ik/R + 3/R^2) rr ]."""
    R = np.linalg.norm(r)
    rr = np.outer(r, r) / R ** 2
    g = np.exp(1j * k * R) / (4 * np.pi * R)
    return g * ((k ** 2 + 1j * k / R - 1 / R ** 2) * np.eye(3)
                + (-k ** 2 - 3j * k / R + 3 / R ** 2) * rr)


def lattice(nx, ny, nz, d):
    g = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1)
    return d * g.reshape(-1, 3).astype(float)


def build_Gamma0(pos, v, k):
    """3N x 3N Green's-function matrix mapping polarization density to field."""
    N = len(pos)
    G = np.zeros((3 * N, 3 * N), complex)
    for i in range(N):
        G[3 * i:3 * i + 3, 3 * i:3 * i + 3] = (-1 / 3 + 1j * k ** 3 * v / (6 * np.pi)) * np.eye(3)
        for j in range(i + 1, N):
            B = v * greens_E(pos[i] - pos[j], k)
            G[3 * i:3 * i + 3, 3 * j:3 * j + 3] = B
            G[3 * j:3 * j + 3, 3 * i:3 * i + 3] = B       # reciprocal: G(r) = G(-r) = G^T
    return G


def random_passive_chi(rng, N, reciprocal=False, scale=3.0):
    """Per-pixel 3x3 susceptibilities chi = H + iK, H Hermitian (or real symmetric),
    K Hermitian PSD (passive).  Returns block-diagonal xi = -chi^{-1} and the chi blocks."""
    chis = []
    for _ in range(N):
        if reciprocal:
            A = rng.normal(size=(3, 3)); H = scale * (A + A.T) / 2
            B = rng.normal(size=(3, 3)); K = B @ B.T / 3 + 0.05 * np.eye(3)
        else:
            A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)); H = scale * ReH(A)
            B = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)); K = B @ B.conj().T / 3 + 0.05 * np.eye(3)
        chis.append(H + 1j * K)
    return blockdiag([-np.linalg.inv(c) for c in chis]), chis


def blockdiag(blocks):
    N = len(blocks)
    M = np.zeros((3 * N, 3 * N), complex)
    for i, b in enumerate(blocks):
        M[3 * i:3 * i + 3, 3 * i:3 * i + 3] = b
    return M


def plane_wave(pos, k, khat=(0, 0, 1), e=(1, 0, 0)):
    khat = np.asarray(khat, float); e = np.asarray(e, complex)
    return (np.exp(1j * k * pos @ khat)[:, None] * e[None, :]).reshape(-1)


def solve(G0, xi, psi):
    return np.linalg.solve(G0 + xi, -psi)


# ----------------------------------------------------------------------------- checks
def global_constraints(G0, xi, phi, psi):
    """Relative residuals of the global laws.  Normalization: |phi^dag (G0+xi) phi|."""
    scale = abs(phi.conj() @ (G0 + xi) @ phi)
    out = {}
    out["complex_IE_product"] = abs(phi.conj() @ (G0 + xi) @ phi + phi.conj() @ psi) / scale
    lhs_im = np.real(phi.conj() @ (ImH(G0) + ImH(xi)) @ phi)
    lhs_re = np.real(phi.conj() @ (ReH(G0) + ReH(xi)) @ phi)
    out["optical_theorem"] = abs(lhs_im - np.imag(psi.conj() @ phi)) / scale
    out["reactive_as_printed"] = abs(lhs_re - np.real(psi.conj() @ phi)) / scale
    out["reactive_corrected"] = abs(lhs_re + np.real(psi.conj() @ phi)) / scale
    out["Re_psi_phi_over_scale"] = abs(np.real(psi.conj() @ phi)) / scale
    return out


def local_constraints(G0, xi, phi, psi):
    """For every pixel i and every 3x3 unit matrix D = e_a e_b^T at pixel i:
    residual_iab = phi_ia^* [(G0 + xi) phi + psi]_ib.  Returns (N,3,3) complex array
    normalized by the global scale; plus local Im/Re power balances (N,)."""
    r = (G0 + xi) @ phi + psi
    N = len(phi) // 3
    P = phi.reshape(N, 3); Rr = r.reshape(N, 3)
    res = np.einsum("ia,ib->iab", P.conj(), Rr)
    scale = abs(phi.conj() @ (G0 + xi) @ phi)
    # local real-power balance (D = pixel identity), in "extinction = abs + scat" form
    Gphi = (G0 @ phi).reshape(N, 3)
    xphi = (xi @ phi).reshape(N, 3)
    ps = psi.reshape(N, 3)
    loc_scat = np.imag(np.sum(P.conj() * Gphi, 1))
    loc_abs = np.imag(np.sum(P.conj() * xphi, 1))
    loc_ext = np.imag(np.sum(ps.conj() * P, 1))
    loc_reac_lhs = np.real(np.sum(P.conj() * (Gphi + xphi), 1))
    loc_reac_rhs = -np.real(np.sum(ps.conj() * P, 1))
    return dict(res=res / scale, loc_scat=loc_scat, loc_abs=loc_abs, loc_ext=loc_ext,
                real_power_resid=np.abs(loc_scat + loc_abs - loc_ext) / scale,
                reactive_resid=np.abs(loc_reac_lhs - loc_reac_rhs) / scale,
                reactive_printed_resid=np.abs(loc_reac_lhs + loc_reac_rhs) / scale)


def far_field_powers(pos, v, k, phi, khat_inc=(0, 0, 1), e_inc=(1, 0, 0), nth=80, nph=160):
    """Scattered power from the far-field Poynting flux (Z0 = 1) and extinction from
    the forward-scattering amplitude (classic optical theorem), for unit |E0|."""
    p = v * phi.reshape(-1, 3)
    x, wq = np.polynomial.legendre.leggauss(nth)
    th = np.arccos(x)
    ph = 2 * np.pi * np.arange(nph) / nph
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    rhat = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1).reshape(-1, 3)
    W = (wq[:, None] * np.full(nph, 2 * np.pi / nph)[None, :]).reshape(-1)
    phase = np.exp(-1j * k * rhat @ pos.T)             # (Ndir, N)
    S = phase @ p                                      # sum_j p_j e^{-ik rhat.x_j}
    S_perp = S - rhat * np.sum(rhat * S, 1)[:, None]
    f = k ** 2 / (4 * np.pi) * S_perp                  # E_ff = f e^{ikr}/r
    P_scat_ff = 0.5 * np.sum(W * np.sum(abs(f) ** 2, 1))
    kh = np.asarray(khat_inc, float); e = np.asarray(e_inc, complex)
    Sf = np.exp(-1j * k * pos @ kh) @ p
    f_fwd = k ** 2 / (4 * np.pi) * (Sf - kh * (kh @ Sf))
    sigma_ext_fwd = 4 * np.pi / k * np.imag(e.conj() @ f_fwd)
    return P_scat_ff, 0.5 * sigma_ext_fwd              # P_ext = I_inc sigma_ext, I_inc = 1/2


def footnote_scat_check(chi, psi):
    """phi_max = i (|chi|^2/Im chi) psi (l.1677).  Returns (P_ext, P_scat corrected,
    P_scat as printed) in units of (w/2)."""
    a = abs(chi) ** 2 / chi.imag
    phi = 1j * a * psi
    Imxi = chi.imag / abs(chi) ** 2
    ext = np.imag(psi.conj() @ phi)
    scat_corr = np.imag(psi.conj() @ phi) - Imxi * np.real(phi.conj() @ phi)
    scat_printed = np.imag(psi.conj() @ phi - Imxi * (phi.conj() @ phi))
    return ext, scat_corr, scat_printed


def lifted_constraint_rank(G0, xi, psi):
    """Rank of the 9 N_s pixel-local constraint matrices in the lifted space
    X = [phi;1][phi;1]^dag: constraint (s,a,b) <-> e_{sa} r_{sb}^T with r = row of [G0+xi | psi]."""
    A = np.hstack([G0 + xi, psi[:, None]])
    n = A.shape[0]
    rows = []
    for s in range(n // 3):
        for a in range(3):
            u = np.zeros(n + 1); u[3 * s + a] = 1
            for b in range(3):
                rows.append(np.outer(u, A[3 * s + b]).reshape(-1))
    M = np.array(rows)
    sv = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(sv > sv[0] * 1e-10)), M.shape[0]


def domain_oblivious_check(k=2 * np.pi / 1.0, d=0.12, shape=(4, 4, 3), frac=0.5, seed=3):
    """Scatterer V_s = random subset of design lattice V_d; scalar design material chi_d.
    Returns dict of residual norms."""
    rng = np.random.default_rng(seed)
    posd = lattice(*shape, d)
    Nd = len(posd)
    v = d ** 3
    Gdd = build_Gamma0(posd, v, k)
    chi_d = 4.0 + 1.0j
    xi_d = -1 / chi_d
    inside = np.zeros(Nd, bool)
    inside[rng.choice(Nd, int(frac * Nd), replace=False)] = True
    idx = np.repeat(inside, 3)
    kh = np.array((0.3, 0.2, 1)) / np.linalg.norm((0.3, 0.2, 1))
    e = np.array((1, 0.5j, 0)); e = e - kh * (kh @ e)
    psi_d = plane_wave(posd, k, khat=kh, e=e)
    Gs = Gdd[np.ix_(idx, idx)]
    phi_s = solve(Gs, xi_d * np.eye(idx.sum()), psi_d[idx])
    phi_d = np.zeros(3 * Nd, complex); phi_d[idx] = phi_s
    r_d = (Gdd + xi_d * np.eye(3 * Nd)) @ phi_d + psi_d      # IE residual on V_d
    scale = abs(phi_d.conj() @ (Gdd + xi_d * np.eye(3 * Nd)) @ phi_d)
    P = phi_d.reshape(Nd, 3); Rr = r_d.reshape(Nd, 3)
    local = np.einsum("ia,ib->iab", P.conj(), Rr)             # all pixel-local D on V_d
    nonlocal_vals = np.einsum("ia,jb->ijab", P[inside].conj(), Rr[~inside])
    return dict(IE_resid_in_Vs=np.abs(r_d[idx]).max() / np.abs(psi_d).max(),
                IE_resid_outside_Vs=np.abs(r_d[~idx]).max() / np.abs(psi_d).max(),
                local_max=np.abs(local).max() / scale,
                global_resid=abs(phi_d.conj() @ r_d) / scale,
                nonlocal_max=np.abs(nonlocal_vals).max() / scale,
                n_design=Nd, n_scat=int(inside.sum()))


def material_bounds_check(k=2 * np.pi, d=0.1, shape=(3, 3, 3), chi=-6.0 + 0.8j):
    """Scalar homogeneous chi: sigma/V vs k |chi|^2/Im chi, and the full optical-theorem
    extinction bound psi^dag (Im G0 + Im xi)^{-1} psi.  Units: |E0| = 1, I_inc = 1/2."""
    pos = lattice(*shape, d); v = d ** 3; N = len(pos); V = N * v
    G0 = build_Gamma0(pos, v, k)
    xi = -1 / chi * np.eye(3 * N)
    psi = plane_wave(pos, k)
    phi = solve(G0, xi, psi)
    w = k
    P_ext = 0.5 * w * v * np.imag(psi.conj() @ phi)
    P_abs = 0.5 * w * v * np.real(phi.conj() @ ImH(xi) @ phi)
    P_scat = 0.5 * w * v * np.real(phi.conj() @ ImH(G0) @ phi)
    s = lambda P: P / 0.5 / V
    bound = k * abs(chi) ** 2 / chi.imag
    A = ImH(G0) + ImH(xi)
    ext_bound = 0.5 * w * v * np.real(psi.conj() @ np.linalg.solve(A, psi))
    phi_opt = 1j * np.linalg.solve(A, psi)
    opt_resid = abs(np.real(phi_opt.conj() @ A @ phi_opt) - np.imag(psi.conj() @ phi_opt))
    return dict(ext_V=s(P_ext), abs_V=s(P_abs), scat_V=s(P_scat), bound=bound,
                ext_OT_bound_V=s(ext_bound), phi_opt_constraint_resid=opt_resid,
                phi_norm_ineq=(np.real(phi.conj() @ phi), abs(chi) ** 2 / chi.imag * np.imag(psi.conj() @ phi)))


def run_general(seed=0, k=2 * np.pi, d=0.15, shape=(4, 4, 3), reciprocal=False):
    rng = np.random.default_rng(seed)
    pos = lattice(*shape, d) + 0.02 * rng.normal(size=(np.prod(shape), 3))
    v = d ** 3
    G0 = build_Gamma0(pos, v, k)
    xi, chis = random_passive_chi(rng, len(pos), reciprocal=reciprocal)
    kh = rng.normal(size=3); kh /= np.linalg.norm(kh)
    e = rng.normal(size=3) + 1j * rng.normal(size=3); e -= kh * (kh @ e); e /= np.linalg.norm(e)
    psi = plane_wave(pos, k, kh, e)
    phi = solve(G0, xi, psi)
    return dict(pos=pos, v=v, k=k, G0=G0, xi=xi, chis=chis, psi=psi, phi=phi, khat=kh, e=e)


def main(make_figure=True):
    print("=" * 84)
    print("Sec. 5.1/5.3 conservation laws on a 3D coupled-dipole VIE model (48 pixels, 144 unknowns)")
    print("=" * 84)
    m = run_general(seed=0)
    G0, xi, phi, psi = m["G0"], m["xi"], m["phi"], m["psi"]
    ev = np.linalg.eigvalsh(ImH(G0))
    evx = np.linalg.eigvalsh(ImH(xi))
    print(f"min eig Im Gamma0 = {ev.min():.2e} (max {ev.max():.2e});  min eig Im xi = {evx.min():.2e}   [both PSD]")
    print(f"Gamma0 symmetric: |G0 - G0^T|max = {np.abs(G0 - G0.T).max():.1e};  xi nonreciprocal: |xi - xi^T|max = {np.abs(xi - xi.T).max():.2f}")
    print(f"IE solve residual: {np.linalg.norm((G0 + xi) @ phi + psi) / np.linalg.norm(psi):.1e}")

    g = global_constraints(G0, xi, phi, psi)
    print("\nGlobal constraints (residual / |phi^dag (G0+xi) phi|):")
    print(f"  {'phi^dag(G0+xi)phi = -phi^dag psi (complex)':52s} {g['complex_IE_product']:.1e}")
    print(f"  {'Im: phi^dag(ImG0+Imxi)phi = +Im(psi^dag phi)  [l.587]':52s} {g['optical_theorem']:.1e}   HOLDS")
    print(f"  {'Re: phi^dag(ReG0+Rexi)phi = +Re(psi^dag phi)  [l.610,1766 as printed]':52s} {g['reactive_as_printed']:.3f}     FAILS  (= 2|Re psi^dag phi|/scale = {2 * g['Re_psi_phi_over_scale']:.3f})")
    print(f"  {'Re: phi^dag(ReG0+Rexi)phi = -Re(psi^dag phi)  [sign corrected]':52s} {g['reactive_corrected']:.1e}   HOLDS")
    mr = run_general(seed=5, reciprocal=True)
    gr = global_constraints(mr["G0"], mr["xi"], mr["phi"], mr["psi"])
    print(f"  (reciprocal random chi, seed 5: printed-reactive resid {gr['reactive_as_printed']:.3f}, corrected {gr['reactive_corrected']:.1e}, OT {gr['optical_theorem']:.1e})")

    loc = local_constraints(G0, xi, phi, psi)
    print("\nLocal (pixel-diagonal D) constraints, 48 pixels x 9 D's = 432 complex constraints:")
    print(f"  max |phi^dag D (G0+xi) phi + phi^dag D psi| / scale = {np.abs(loc['res']).max():.1e}")
    print(f"  local real power  Im[phi_i^dag(G0 phi)_i] + phi_i^dag Imxi phi_i = Im(psi_i^dag phi_i): max resid {loc['real_power_resid'].max():.1e}")
    print(f"  local reactive    Re[...] + phi_i^dag Rexi phi_i = -Re(psi_i^dag phi_i):          max resid {loc['reactive_resid'].max():.1e}")
    print(f"  local reactive with printed sign (+Re):                                          max resid {loc['reactive_printed_resid'].max():.2e}")
    print(f"  sum_i local = global: {abs(loc['loc_scat'].sum() - np.real(phi.conj() @ ImH(G0) @ phi)):.1e}")
    print(f"  note: 'local scattering' Im[phi_i^dag (G0 phi)_i] is not sign-definite: min {loc['loc_scat'].min():+.3e}, max {loc['loc_scat'].max():+.3e};")
    print(f"        'local extinction' min {loc['loc_ext'].min():+.3e}  (only the sums are positive)")

    k, v, pos = m["k"], m["v"], m["pos"]
    Pscat_ff, Pext_fwd = far_field_powers(pos, v, k, phi, m["khat"], m["e"])
    Pscat = 0.5 * k * v * np.real(phi.conj() @ ImH(G0) @ phi)
    Pext = 0.5 * k * v * np.imag(psi.conj() @ phi)
    Pabs = 0.5 * k * v * np.real(phi.conj() @ ImH(xi) @ phi)
    psi_tot = -xi @ phi
    ImChi = blockdiag([ImH(c) for c in m["chis"]])
    Pabs2 = 0.5 * k * v * np.real(psi_tot.conj() @ ImChi @ psi_tot)
    print("\nPower identities (omega = k, Z0 = 1, |E0| = 1):")
    print(f"  P_scat = (w/2) phi^dag ImG0 phi = {Pscat:.10f};  far-field Poynting flux = {Pscat_ff:.10f}  (rel diff {abs(Pscat / Pscat_ff - 1):.1e})")
    print(f"  P_ext  = (w/2) Im(psi^dag phi)  = {Pext:.10f};  forward-amplitude opt. thm = {Pext_fwd:.10f}  (rel diff {abs(Pext / Pext_fwd - 1):.1e})")
    print(f"  P_abs  = (w/2) phi^dag Imxi phi = {Pabs:.10f};  (w/2) psi_tot^dag Imchi psi_tot = {Pabs2:.10f}")
    print(f"  P_ext - P_abs - P_scat = {Pext - Pabs - Pscat:.1e}")
    fs = footnote_scat_check(-4 + 1.5j, psi)
    print("\nFootnote l.1677, phi = phi_max = i(|chi|^2/Im chi) psi, chi = -4+1.5i (units w/2):")
    print(f"  P_ext = {fs[0]:.6f};  P_ext - P_abs (correct: Im(psi^dag phi) - phi^dag Imxi phi) = {fs[1]:.1e}")
    print(f"  as printed, Im[psi^dag phi - phi^dag (Im xi) phi] = {fs[2]:.6f}  (= P_ext, not 0)  -> NOTATION ERROR")
    # also on the physical solution
    ps_printed = 0.5 * k * v * np.imag(psi.conj() @ phi - phi.conj() @ ImH(xi) @ phi)
    print(f"  on the solved scatterer: printed form gives {ps_printed:.6f} = P_ext, true P_scat = {Pscat:.6f}")

    do = domain_oblivious_check()
    print(f"\nDomain obliviousness (l.1938-1940): design lattice {do['n_design']} pixels, scatterer {do['n_scat']} pixels, chi_d = 4+i")
    print(f"  IE residual |(G0+xi)phi+psi| on V_s: {do['IE_resid_in_Vs']:.1e};  on V_d \\ V_s: {do['IE_resid_outside_Vs']:.2f}  (IE itself fails on V_d)")
    print(f"  all pixel-local D constraints on V_d: max resid {do['local_max']:.1e};  global on V_d: {do['global_resid']:.1e}   HOLD")
    print(f"  nonlocal D (x in V_s, x' in V_d \\ V_s): max |value| {do['nonlocal_max']:.2e}                     FAIL")

    mm = run_general(seed=1, shape=(3, 3, 2))
    rank, ncon = lifted_constraint_rank(mm["G0"], mm["xi"], mm["psi"])
    print(f"\nConstraint count (l.1965): N_s = 18 pixels, N_ns = 3 -> N_s N_ns^2 = {18 * 9}; lifted rank = {rank} of {ncon}")

    mb = material_bounds_check()
    print("\nMaterial-loss bounds [CSBoundML, l.1683] on a 27-pixel cube, chi = -6+0.8i (k|chi|^2/Im chi per volume):")
    print(f"  sigma_ext/V = {mb['ext_V']:.4f}, sigma_abs/V = {mb['abs_V']:.4f}, sigma_scat/V = {mb['scat_V']:.4f};"
          f"  bound = {mb['bound']:.4f} (ext, abs), {mb['bound'] / 4:.4f} (scat)")
    print(f"  full-OT extinction bound (l.1737): {mb['ext_OT_bound_V']:.4f} >= {mb['ext_V']:.4f};"
          f"  phi_opt constraint residual {mb['phi_opt_constraint_resid']:.1e}")
    print(f"  phi^dag phi = {mb['phi_norm_ineq'][0]:.4f} <= (|chi|^2/Im chi) Im(psi^dag phi) = {mb['phi_norm_ineq'][1]:.4f}  (l.1675)")

    if make_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
        i = np.arange(len(loc["reactive_resid"]))
        ax[0].semilogy(i, loc["real_power_resid"] + 1e-18, "C0o", ms=3, label="local real power (Im)")
        ax[0].semilogy(i, loc["reactive_resid"] + 1e-18, "C2s", ms=3, label=r"local reactive, $-\mathrm{Re}(\psi^\dagger\phi)$")
        ax[0].semilogy(i, loc["reactive_printed_resid"], "C3^", ms=3, label=r"local reactive, $+\mathrm{Re}$ (as printed)")
        ax[0].set_xlabel("pixel index"); ax[0].set_ylabel("|residual| / global scale")
        ax[0].legend(fontsize=7, frameon=False); ax[0].set_ylim(1e-19, 1)
        ax[0].set_title("pixel-local conservation laws", fontsize=9)
        ax[1].bar(np.arange(len(loc["loc_ext"])), loc["loc_ext"], color="C0", width=1.0, label="local extinction")
        ax[1].plot(np.arange(len(loc["loc_scat"])), loc["loc_scat"], "C3.", ms=4, label="local 'scattering'")
        ax[1].axhline(0, color="k", lw=0.5)
        ax[1].set_xlabel("pixel index"); ax[1].legend(fontsize=7, frameon=False)
        ax[1].set_title("local terms are not sign-definite", fontsize=9)
        fig.tight_layout()
        os.makedirs(FIGDIR, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIGDIR, f"sec5_vie_conservation.{ext}"), dpi=200)
        print(f"\nwrote {os.path.normpath(os.path.join(FIGDIR, 'sec5_vie_conservation.pdf'))}")


if __name__ == "__main__":
    main()
