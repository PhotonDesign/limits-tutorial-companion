"""Sec. 3.1.2: direct check of the plane-wave power formula

    P = (1/2) Re int (E x H^*) . zhat dx dy = (2 pi^2 / Z0) sum_p int d^2k_par (k_z/k) |c_p|^2,

for an ARBITRARY complex spectrum c_p(k_par) (random amplitudes and phases for both
polarizations, independent at every k_par).  E and H are synthesized on a periodic
grid in the z = 0 plane by FFT, with H = (1/Z0) sum_p c_p (khat x e_p) e^{ik.x}, and the
Poynting flux is integrated numerically.  Cross-polarization terms (p != p') and
cross-wavevector terms must vanish for the formula to hold; any error in the
(k_z/k) factor or polarization algebra shows up at O(1).

Run: python power_normalization_check.py   (< 1 s)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tutorial_tools.angular_spectrum import polarization_basis


def run(sin_theta=0.95, n=128, dk=0.05, seed=0, k=1.0, Z0=1.0):
    rng = np.random.default_rng(seed)
    kx1 = dk * (np.arange(n) - n // 2)
    KX, KY = np.meshgrid(kx1, kx1, indexing="ij")
    KP = np.hypot(KX, KY)
    inside = KP < k * sin_theta
    KZ = np.sqrt(np.maximum(k**2 - KP**2, 0))
    khat = np.stack([KX, KY, KZ], -1)[inside] / k
    E_pol = polarization_basis(khat)                   # (N,2,3)
    H_pol = np.cross(khat[:, None, :], E_pol)          # khat x e_p
    c = (rng.normal(size=(khat.shape[0], 2)) + 1j * rng.normal(size=(khat.shape[0], 2)))
    # spectra on the k grid (vector valued)
    Ek = np.zeros((n, n, 3), complex)
    Hk = np.zeros((n, n, 3), complex)
    Ek[inside] = np.einsum("np,npj->nj", c, E_pol)
    Hk[inside] = np.einsum("np,npj->nj", c, H_pol) / Z0
    # E(x) = sum_k c e^{ik.x} dk^2 on the periodic grid x_m = m * (2pi/(n dk)).
    L = 2 * np.pi / dk
    dx = L / n
    def synth(F):
        F = np.fft.ifftshift(F, axes=(0, 1))
        return np.fft.ifft2(F, axes=(0, 1)) * n * n * dk**2
    E = synth(Ek)
    H = synth(Hk)
    S = 0.5 * np.real(np.cross(E, np.conj(H))[..., 2])
    P_direct = S.sum() * dx**2
    P_formula = 2 * np.pi**2 / Z0 * np.sum((khat[:, 2] * np.sum(np.abs(c)**2, 1))) * dk**2
    return P_direct, P_formula


def main():
    print("Plane-wave power formula, random complex spectra (both polarizations):")
    for s in [0.3, 0.8, 0.999]:
        for seed in range(2):
            Pd, Pf = run(sin_theta=s, seed=seed)
            print(f"  sin th = {s:5.3f}, seed {seed}: direct Poynting {Pd:.10e}  formula {Pf:.10e}  rel.err {abs(Pd/Pf-1):.1e}")


if __name__ == "__main__":
    main()
