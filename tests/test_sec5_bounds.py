"""Fast checks of the Sec. 5.1 cross-section bounds (sec5_local_constraints/material_loss_bounds.py, mie.py)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sec5_local_constraints.mie import cross_sections, rho_ImGamma0, planewave_ImGamma0_quadform
from sec5_local_constraints.material_loss_bounds import (
    envelope_check, dipole_limit_checks, mie_bound_sweep, unitary_channel_saturation, ot_ext_bound)


def test_mie_reference_value():
    x = 2 * np.pi * 0.525 / 0.6328                            # Bohren & Huffman test case
    assert abs(cross_sections(1.55 + 0j, x)["ext"][0] / (np.pi * x ** 2) - 3.10543) < 1e-5


def test_rho_normalization_identity():
    for kR in (0.5, 3.0):
        n = np.arange(1, 31)
        rTE, rTM = rho_ImGamma0(kR, 30)
        rhs = 2 * np.pi * np.sum((2 * n + 1) * (rTE ** 2 + rTM ** 2))
        assert abs(planewave_ImGamma0_quadform(kR) / rhs - 1) < 1e-10
    rTE, rTM = rho_ImGamma0(1e-2, 1)
    assert abs(rTM[0] / (2 / 9 * 1e-6) - 1) < 1e-4            # k^3 V / 6 pi


def test_material_bound_envelope():
    w, ratio_grid, ok, ratio_star, s, b = envelope_check()
    assert ratio_grid.max() <= 1 + 1e-12
    assert np.abs(ratio_star - 1).max() < 1e-12


def test_dipole_limit_saturation():
    for chi, rho in ((-5 + 0.5j, 0.01), (-3 + 0.05j, 0.1), (-3 + 1.5j, 0.02)):
        d = dipole_limit_checks(chi, rho)
        assert abs(d["ext_res"] / d["ot_ext"] - 1) < 1e-10     # CSOTBound saturated
        assert abs(d["sca_max"] / d["dual_sca"] - 1) < 1e-6    # scattering dual bound saturated
        assert d["abs_max"] <= d["dual_abs"] * (1 + 1e-9)
        if d["Imxi"] >= d["rho"]:
            assert abs(d["abs_max"] / d["dual_abs"] - 1) < 1e-6


def test_mie_spheres_obey_bounds():
    kRs = np.array([0.1, 0.32, 1.0, 2.5, 5.0])
    for chi in (-2.2 + 0.1j, 3 + 0.5j):
        r = mie_bound_sweep(chi, kRs)
        for key in ("ext", "abs", "sca", "chan_ot", "chan_unit_sca", "chan_unit_abs"):
            assert r[key].max() <= 1 + 1e-9, key


def test_unitary_channel_bounds_reached():
    bs, ba = unitary_channel_saturation()
    assert 0.9999 < bs <= 1 + 1e-12
    assert 0.249 < ba <= 0.25 + 1e-12


def test_large_loss_limit_is_full_sum():
    kR, Imxi = 2.0, 1e6
    terms = ot_ext_bound(kR, Imxi, per_channel=True)
    V = 4 * np.pi * kR ** 3 / 3
    assert abs(terms.sum() / (V / Imxi) - 1) < 1e-4
    assert terms[len(terms) // 2] / (V / Imxi) < 0.5          # n = 1 TM term alone is not enough
