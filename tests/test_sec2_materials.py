"""Fast regression tests for the Ch. 2 (bounds on constitutive material properties) validations."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sec2_material_bounds import max_refractive_index as mri
from sec2_material_bounds import dispersion_bandwidth_bounds as dbb
from sec2_material_bounds import sum_rules_and_estimates as sre


# ---------------------------------------------------------------- Sec. 2.1 maximum index
@pytest.mark.parametrize("wp,chip", [(2.0, 0.3), (5.0, 1.0), (15.0, 50.0)])
def test_lp_single_oscillator_optimum(wp, chip):
    v, wi, ci = mri.lp_max_chi(wp, chip)
    assert v == pytest.approx(mri.chi_max_closed(wp, chip), rel=2e-5)
    assert len(wi) <= 2 and wi.min() <= mri.w0_closed(wp, chip) <= wi.max()
    n = np.sqrt(1 + v)
    R = wp**2 * chip / (2 * n)
    assert (n**2 - 1)**2 / n / R == pytest.approx(1, rel=2e-5)          # Shim et al. Eq. 7, saturated
    assert ((n**2 - 1) / n)**2 / R == pytest.approx(1 / n, rel=2e-5)     # printed [eq:max_n]: loose by n


def test_silicon_example():
    p = mri.material_point("Si")
    assert p["n"] == pytest.approx(3.476, abs=2e-3)
    assert p["correct"] == pytest.approx(3.87, abs=0.02)
    assert p["printed"] == pytest.approx(7.25, abs=0.05)
    assert p["n"] < p["correct"] < p["printed"]


# ---------------------------------------------------------------- Sec. 2 / 2.2 dispersion and bandwidth
def test_derivative_bounds_random_media():
    A = dbb.test_A(n_media=120)
    assert A["q1"] >= 0                                       # [eq:derivative_bounds]
    assert A["q3"] >= -1e-10                                  # sharper L&L form
    assert A["ratio"] == pytest.approx(2.0, abs=1e-8)         # floor is 2, not 1
    assert A["drude_eq"] < 1e-12


@pytest.mark.parametrize("eps_m", [0.0, -1.0])
def test_min_slope_is_twice_printed(eps_m):
    s, drude_frac = dbb.lp_min_slope(eps_m)
    assert s == pytest.approx(2 * (1 - eps_m), rel=1e-9)
    assert drude_frac == pytest.approx(1.0, abs=1e-9)


def test_lossless_bandwidth_bound_is_drude():
    for B in (0.1, 0.8):
        v, dev, fd = dbb.minimax_lp(B, lossy=False)
        assert dev == pytest.approx(dbb.bound_lossless_tight(B), rel=1e-6)
        assert fd == pytest.approx(1.0, abs=1e-6)
        assert dev > dbb.bound_tutorial(B)                    # printed bound valid, not tight


def test_lossy_bandwidth_optimum():
    v, dev, _ = dbb.minimax_lp(0.4, lossy=True)
    assert 0.2 <= v < dev < 0.21                              # -> B/2 from above
    B = np.linspace(0.01, 1.9, 50)
    assert np.allclose(dbb.bound_lossy_tight(B), B / 2)      # Lind-Johansen form = B/2 exactly


def test_split_ring_mu_caveat():
    rows, muinf = dbb.srr_check()
    d = dict(rows)
    assert d[2.0] < 1 and d[0.5] > 1 and all(v >= muinf for _, v in rows)


def test_biisotropic_absorbed_power():
    err, agree = dbb.chiral_check(n=300)
    assert err < 1e-12 and agree == 1.0


# ---------------------------------------------------------------- sum rules and numbers
def test_sum_rules():
    ins, met = sre.media()
    for m in (ins, met):
        assert sre.f_sum(m) == pytest.approx(np.pi * m.wp2 / 2, rel=1e-10)
    assert abs(sre.re_integral(ins)) < 1e-10                  # footnote l.746: insulators only
    assert sre.re_integral(met) == pytest.approx(-np.pi * met.fD / (2 * met.gD), rel=1e-9)


def test_impulse_response_inertia():
    ins, met = sre.media()
    for m in (ins, met):
        assert abs(sre.polarization_current(m, -1e-3)) < 1e-6
        assert sre.polarization_current(m, 0.0) == pytest.approx(m.wp2 / 2, rel=1e-10)
        assert sre.polarization_current(m, 1e-4) == pytest.approx(m.wp2, rel=1e-3)


def test_loss_bound_and_trk():
    H, fs, target = sre.box_loss_bound()
    assert fs == pytest.approx(target, rel=1e-12)
    assert sre.loss_bound_random(n=50) < 1
    t, h = sre.trk(n_grid=2000)
    for (p, q), v in t.items():
        assert v == pytest.approx(0.5 if p == q else 0.0, abs=1e-4)


def test_quoted_numbers():
    n = sre.numbers()
    assert n["glass"] == pytest.approx(0.8647, abs=1e-4)
    assert n["bcs"] == pytest.approx(1.764, abs=1e-3)
    assert n["Um"] == pytest.approx(2.0)
