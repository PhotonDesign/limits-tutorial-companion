"""Fast regression tests for the Sec. 3.1 (free-space focusing) validations."""
import numpy as np
import pytest

from tutorial_tools.angular_spectrum import focal_matrix
from tutorial_tools import focusing as F
from tutorial_tools.vsw import vsw
import focal_intensity_bounds as fib
import power_normalization_check as pnc
import focal_plane_fields as fpf
import fourpi_spherical_waves as fps
import superoscillation as so
import spacetime_focusing as st


@pytest.mark.parametrize("s", [0.2, 0.6, 0.8, 0.97, 1.0])
def test_focal_matrix_matches_closed_forms(s):
    th = np.arcsin(s)
    Mp = focal_matrix(th, "power") / np.pi
    Ma = focal_matrix(th, "amplitude") / np.pi
    assert np.allclose(np.diag(Mp).real, [F.bound_power_inplane(th)] * 2 + [F.bound_power_z(th)], atol=1e-10)
    assert np.allclose(np.diag(Ma).real, [F.bound_amp_inplane(th)] * 2 + [F.bound_amp_z(th)], atol=1e-10)
    assert np.abs(Mp - np.diag(np.diag(Mp))).max() < 1e-12


def test_limits_and_ratios():
    assert F.bound_power_inplane(np.pi / 2) == pytest.approx(4 / 3)
    assert F.bound_power_z(np.pi / 2) == pytest.approx(4 / 3)
    assert F.bound_amp_inplane(np.pi / 2) / F.bound_amp_z(np.pi / 2) == pytest.approx(1.5)
    assert F.amp_optimal_per_true_power(np.pi / 2) == pytest.approx(135 / 128)
    t = 1e-3
    assert F.bound_power_inplane(t) / t**2 == pytest.approx(1, rel=1e-5)


def test_amp_optimum_true_power_numeric():
    r_true, r_amp = fib.check_true_power_of_amp_optimum(np.arcsin(0.8))
    assert r_true == pytest.approx(F.amp_optimal_per_true_power(np.arcsin(0.8)), rel=1e-10)
    assert r_amp == pytest.approx(F.bound_amp_inplane(np.arcsin(0.8)), rel=1e-10)


def test_power_formula():
    Pd, Pf = pnc.run(sin_theta=0.9, n=64, dk=0.08)
    assert Pd == pytest.approx(Pf, rel=1e-12)


def test_2pi_closed_form_fields():
    e_brute, e_closed = fpf.check_three_ways(n_pts=6)
    assert e_brute < 1e-10 and e_closed < 1e-10
    h1, h2 = fpf.check_half_of_4pi(n_pts=6)
    assert h1 < 1e-12 and h2 < 1e-12


def test_widths():
    f_y = fpf.intensity_fn(np.pi / 2, "power", "one", "y")
    f_x = fpf.intensity_fn(np.pi / 2, "power", "one", "x")
    f4 = lambda a: F.intensity_4pi_perp(2 * np.pi * np.asarray(a))
    assert 2 * F.half_max_radius(f_y, 1.5, n=301) == pytest.approx(0.4021, abs=2e-4)
    assert 2 * F.half_max_radius(f4, 1.5, n=301) == pytest.approx(0.4021, abs=2e-4)
    assert 2 * F.half_max_radius(f_x, 1.5, n=301) == pytest.approx(0.6337, abs=2e-4)


def test_vsw_4pi():
    assert fps.check_curl() < 1e-7
    N0 = vsw(1, 0, np.array([[0, 0, 1e-8]]))[1][0]
    assert np.linalg.norm(N0) == pytest.approx(2 / 3 * np.sqrt(3 / (8 * np.pi)), rel=1e-8)
    Pin = fps.incoming_power(radii=(5.0,))[0]
    assert Pin == pytest.approx(1 / 8, rel=1e-10)
    assert 0.25 * np.linalg.norm(N0)**2 / Pin == pytest.approx(1 / (3 * np.pi), rel=1e-8)


def test_superoscillation():
    assert so.three_point(0.03) * 2 / (45 / 0.03**4) == pytest.approx(1, abs=1e-3)
    assert so.three_point(np.pi) * 2 == pytest.approx(1, abs=1e-10)
    for g in (0.8, 0.3, 0.1):
        u = g * so.U0
        assert so.strehl_ring(u) == pytest.approx(so.strehl_ring_bruteforce(u), rel=1e-8)
    assert so.strehl_ring(1e-2) / (1e-8 / 192) == pytest.approx(1, abs=1e-3)


def test_spacetime():
    for x, t in [(0.7, 0.2), (-3.1, 1.9), (5.5, -4.0)]:
        a = st.psi1d_closed(np.array(x), np.array(t))
        b = st.psi1d_quad(x, t)
        assert abs(a - b) < 1e-10
    for U in (0.5, 3.0, 11.0):
        assert st.psi3d_closed(np.array([U]))[0] == pytest.approx(st.psi3d_quad(U), abs=1e-12)
    U = np.linspace(1e-3, 100, 100001)
    assert st.psi3d_closed(U).min() > 0


@pytest.mark.parametrize("s", [0.3, 0.8, 1.0])
def test_amplitude_closed_forms(s):
    rng = np.random.default_rng(2)
    rho, phi = rng.uniform(0.05, 3, 8), rng.uniform(0, 2 * np.pi, 8)
    Ex, Ey, Ez = F.optimal_field_xpol(rho, phi, 0 * rho, np.arcsin(s), "amplitude", n=600)
    Exc, Eyc = F.field_amp_closed_inplane(rho, phi, s)
    assert np.abs(Ex - Exc).max() < 1e-10 and np.abs(Ey - Eyc).max() < 1e-10
    if s == 1.0:
        assert np.abs(Ez - F.field_amp_2pi_Ez_closed(rho, phi)).max() < 1e-10


def test_spacetime_energy_weighted():
    from scipy.special import j1
    for x in (0.5, 2.0, 9.0):
        q = st.energy_weighted_2d_quad(x) / st.energy_weighted_2d_quad(1e-9)
        assert q == pytest.approx(2 * j1(x) / x, abs=1e-10)
    for U in (0.5, 3.0, 15.0):
        assert st.energy_weighted_3d_quad(U) == pytest.approx(st.energy_weighted_3d(np.array([U]))[0], abs=1e-10)
