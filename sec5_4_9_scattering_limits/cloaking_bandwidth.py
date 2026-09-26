"""Sec. 5.7 (sec:cloaking), bandwidth rules of thumb and bounds for cloaking, and their
numerical examples.

Tutorial claims (tutorial.tex line numbers):
  l.2432  tau ~= 1/Delta omega; Q_c = tau/Delta T; Delta T > Delta R/c; Delta T = 1/(Delta omega Q_c)
  l.2434  [eq:cloak_bound_1]  B < lambda0/(Delta R Q_c) < lambda0/(a Q_c)              (as printed)
  l.2437  example: Delta R ~ 1 m, Q_c = 5, lambda0 = 500 nm  ->  "B = 10^-5 %"
  l.2439  planar slab (Bode-Fano):  B <= (lambda0/d) / ((eps/eps0 - 1) |ln Gamma0|)
  l.2442  large impenetrable sphere: B < (lambda0/a) / |ln( sigma_scat/(2 pi a^2) )|
  l.2444  example: a = 1 m, ten-fold suppression, lambda0 = 500 nm  ->  "B = 2 x 10^-5 %"
  l.2451  caption: "The cloaking bandwidth decreases linearly with object size."

Checks:
  A. The chain of l.2432 gives Delta omega < c/(Delta R Q_c), i.e. B = Delta omega/omega0 <
     lambda0/(2 pi Delta R Q_c).  The printed bound (no 2 pi) corresponds to tau ~= 1/Delta f
     = 2 pi/Delta omega.  Both versions of the example are evaluated, plus the Gaussian
     intensity-FWHM convention tau = 4 ln2/Delta omega.
  B. Sphere example: the uncloaked large PEC sphere has sigma_sca -> 2 pi a^2 (extinction
     paradox; exact PEC Mie at ka = 50, 200), so "ten-fold suppression" means sigma = 0.2 pi a^2
     and B = (lambda0/a)/ln 10 = 2.2e-5 %  (1.7e-5 % if measured relative to pi a^2).
  C. Slab bound: in the thin-slab limit the slab is exactly a shunt capacitance
     C = eps0 (eps_r - 1) d loading a line of impedance eta0 (checked against the exact slab
     reflection coefficient), and the parallel-RC Bode-Fano limit then gives
     B <= (lambda0/d) / (2 (eps_r - 1) |ln Gamma0|): a factor 2 tighter than l.2439.  The cited
     derivation models the slab near a resonance (RLC), so this is reported as UNCLEAR.

Run:  python cloaking_bandwidth.py   (about 1 s)
"""
import numpy as np
from scipy.special import spherical_jn, spherical_yn


def rule_of_thumb(lam0, dR, Qc, tau_convention="1/domega"):
    """Max fractional bandwidth from Delta T = tau/Q_c > Delta R/c, tau = kappa/Delta omega."""
    kappa = {"1/domega": 1.0, "gaussian FWHM 4ln2/domega": 4 * np.log(2), "1/df = 2pi/domega": 2 * np.pi}[tau_convention]
    # kappa/(domega Qc) > dR/c  ->  domega < kappa c/(dR Qc)  ->  B < kappa lam0/(2 pi dR Qc)
    return kappa * lam0 / (2 * np.pi * dR * Qc)


def pec_sphere_sigma_sca(x):
    nmax = int(x + 4 * x ** (1 / 3) + 10)
    n = np.arange(1, nmax + 1)
    j, y = spherical_jn(n, x), spherical_yn(n, x)
    jp, yp = spherical_jn(n, x, derivative=True), spherical_yn(n, x, derivative=True)
    h, hp = j + 1j * y, jp + 1j * yp
    b = j / h                                        # psi/xi
    a = (j + x * jp) / (h + x * hp)                  # psi'/xi'
    return 2 * np.pi / x ** 2 * np.sum((2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2))   # units a^2 (k a = x)


def slab_reflection_exact(k0d, eps):
    n = np.sqrt(eps)
    r12 = (1 - n) / (1 + n)
    ph = np.exp(2j * n * k0d)
    return r12 * (1 - ph) / (1 - r12 ** 2 * ph)


def slab_reflection_shunt_c(k0d, eps):
    # shunt admittance Y = -i w (eps - 1) d (eta0 = 1): Gamma = (Z - 1)/(Z + 1), Z = 1/(1 + Y)
    Y = -1j * k0d * (eps - 1)
    Z = 1 / (1 + Y)
    return (Z - 1) / (Z + 1)


def main():
    print("=" * 88)
    print("Sec. 5.7  cloaking bandwidth rules of thumb and examples")
    print("=" * 88)
    lam0, dR, Qc = 500e-9, 1.0, 5.0
    print("A. Rule of thumb [eq:cloak_bound_1], Delta R = 1 m, Q_c = 5, lambda0 = 500 nm")
    print(f"   printed formula lambda0/(Delta R Q_c) = {lam0/(dR*Qc):.3e} = {100*lam0/(dR*Qc):.2e} %  (text: 10^-5 %)")
    for conv in ("1/domega", "gaussian FWHM 4ln2/domega", "1/df = 2pi/domega"):
        B = rule_of_thumb(lam0, dR, Qc, conv)
        print(f"   tau = {conv:27s}: B < {B:.3e} = {100*B:.2e} %")
    print("   -> with tau ~= 1/Delta omega as stated at l.2432 the bound is lambda0/(2 pi Delta R Q_c) = 1.6e-6 %;")
    print("      the printed bound and '10^-5 %' correspond to tau ~= 1/Delta f.")

    print("\nB. Sphere bound l.2442, a = 1 m, lambda0 = 500 nm, ten-fold suppression")
    for x in (50.0, 200.0):
        print(f"   exact PEC Mie, ka = {x:5.0f}: sigma_sca/(pi a^2) = {pec_sphere_sigma_sca(x)/np.pi:.4f}  (-> 2)")
    a = 1.0
    for ref, lab in ((2.0, "relative to 2 pi a^2 (large-sphere value)"), (1.0, "relative to pi a^2")):
        sig_norm = ref / 10
        B = (lam0 / a) / abs(np.log(0.5 * sig_norm))
        print(f"   {lab:42s}: |ln(sigma/2pi a^2)| = {abs(np.log(0.5*sig_norm)):.3f}, B < {B:.2e} = {100*B:.2e} %  (text: 2e-5 %)")

    print("\nC. Slab bound l.2439: thin slab = shunt C = eps0 (eps_r - 1) d on a line of impedance eta0")
    for eps in (2.25, 4.0, 12.0):
        k0d = np.array([1e-3, 1e-2, 3e-2])
        ex, sc = slab_reflection_exact(k0d, eps), slab_reflection_shunt_c(k0d, eps)
        print(f"   eps_r = {eps:5.2f}: |r_exact/r_shuntC - 1| at k0 d = 1e-3, 1e-2, 3e-2: "
              + ", ".join(f"{v:.1e}" for v in np.abs(ex / sc - 1)))
    print("   Bode-Fano (R = eta0, C = eps0 (eps_r-1) d): Delta omega |ln G0| <= pi/(RC) = pi c/((eps_r-1) d)")
    print("   => B <= (lambda0/d) / (2 (eps_r - 1) |ln G0|)   vs printed (lambda0/d)/((eps_r - 1)|ln G0|)")
    eps, d, G0 = 4.0, 50e-9, 0.1
    print(f"   e.g. eps_r = 4, d = 50 nm, lambda0 = 500 nm, G0 = 0.1: thin-slab {lam0/d/(2*(eps-1)*abs(np.log(G0))):.3f},"
          f" printed {lam0/d/((eps-1)*abs(np.log(G0))):.3f}")


if __name__ == "__main__":
    main()
