"""Fast checks of Sec. 5.7 reflection bounds and spherical-wave sum rules
(sec5_4_9_scattering_limits/reflection_bounds.py, spherical_sum_rules.py)."""
import os
import sys
import numpy as np
from numpy.polynomial import Polynomial as Poly

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sec5_4_9_scattering_limits.reflection_bounds import (
    z_parallel_rc, rc_highfreq_coefficient, bode_fano_integral, gamma_from_z, ladder_study,
    rozanov_check, DEBYE, MAGNETIC, blaschke_terms, kk_reflection_phase, wrap)
from sec5_4_9_scattering_limits.spherical_sum_rules import (
    pec_allpass, pec_f, fvals, kk_check, sum_rule_ratio, MATERIALS)


def test_parallel_rc_impedance_and_sign():
    w = 0.7
    assert abs(z_parallel_rc(np.array([w]))[0] - 1 / (1 - 1j * w)) < 1e-14
    cg, ch = rc_highfreq_coefficient(w=(1e4,))
    assert abs(cg[0] - 2j) < 1e-3            # Gamma = -1 + 2i/(w RC): l.2407 prints -2i
    assert abs(ch[0] + 2) < 1e-3             # h coefficient -2/(RC) as printed


def test_bode_fano_bare_load_equality():
    I = bode_fano_integral(*gamma_from_z(Poly([1.0]), Poly([1.0, 1.0])))
    assert abs(I - np.pi) < 1e-8


def test_bode_fano_random_ladders():
    rows = ladder_study(n_networks=12, seed=11)
    for r in rows:
        assert abs(r["I"] - r["fano"]) < 1e-7            # Fano identity with UHP zeros
        assert r["I"] <= np.pi + 1e-9                     # [eq:Bode_Fano]
        assert r["band_product"] <= np.pi                 # [eq:Gamma_bound_BF]
        assert abs(r["coef"] - 2j) < 1e-3                 # network-independent expansion
        assert r["mag_diff"] < 1e-10                      # |Gamma| = |Gamma'|


def test_rozanov():
    for mat in (DEBYE, MAGNETIC):
        coef, I, B = rozanov_check(mat)
        assert abs(coef.real / mat["mu_s"] - 1) < 1e-3
        assert I <= B


def test_blaschke_term_converges_to_minus_d():
    parts, resid = blaschke_terms(Ms=(100, 10000))
    assert resid < 1e-12
    assert abs(parts[-1][1] + 1) < 1e-4                  # sum Im(1/w_m) -> -d/c, so a_1 = 0
    assert parts[-1][2] > 50 * parts[0][2]               # printed sum 1/Im w_m diverges


def test_kk_reflection_sign_convention():
    good = kk_reflection_phase([0.5, 1.3], +1, npts=400001)
    bad = kk_reflection_phase([0.5, 1.3], -1, npts=400001)
    assert max(abs(wrap(t - a)) for _, t, a in good) < 2e-3
    assert min(abs(abs(wrap(t - a)) - np.pi) for _, t, a in bad) < 2e-3


def test_pec_sphere_allpass_blaschke():
    for n in (1, 2, 4):
        for pol in ("TE", "TM"):
            r = pec_allpass(n, pol)
            assert r["maxdev"] < 1e-12                        # s_n e^{2ix} = B_n exactly -> h_n = 0
            assert abs(r["sum_im_inv"] + 1) < 1e-12           # correct a_1 = 2 + 2 sum Im(1/x_i) = 0
            assert 2 + 2 * r["sum_inv_im"] > 3                # printed l.2332 coefficient is not 0


def test_kk_scattering_needs_s_of_omega_prime():
    for f in (lambda x: pec_f(1, x, "TE"), lambda x: fvals(x, MATERIALS[0], "TM")):
        for x, re, rhs in kk_check(f, [0.5, 1.5], X=200.0, dx=4e-3):
            assert abs(re - rhs) < 2e-3
        # printed version (s_n(w) inside the integral) diverges at w' -> 0: cutoff-dependent
        b1 = kk_check(f, [1.5], X=200.0, dx=4e-3, typo=True)[0][2]
        b2 = kk_check(f, [1.5], X=200.0, dx=1e-3, typo=True)[0][2]
        assert abs(b1 - b2) > 0.05


def test_spherical_sum_rule_inequality():
    for mat in MATERIALS[:2]:
        r, rl, _ = sum_rule_ratio(mat, "TM", X=500.0)
        assert 0 < r <= 1 and abs(r - rl) < 1e-12
