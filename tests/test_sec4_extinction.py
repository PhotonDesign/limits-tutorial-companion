"""Tests for sec4_global_symmetries/extinction_symmetry_slab.py and extinction_symmetry_dipoles.py."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sec4_global_symmetries"))
import tmm1d  # noqa: E402
import extinction_symmetry_slab as es  # noqa: E402
import extinction_symmetry_dipoles as ed  # noqa: E402


def test_tmm_energy_and_vacuum():
    eps, d = es.graded_slab(50)
    r1, t1, r2, t2, kz0 = tmm1d.rt_both_sides(np.real(eps) + 0j, d, np.linspace(1, 20, 50))
    assert np.allclose(np.abs(r1)**2 + np.abs(t1)**2, 1)          # lossless version conserves
    assert np.allclose(t1, t2)                                      # reciprocity of t_face
    e_true, e_matlab = es.empty_slab_check(np.linspace(0.25, 5, 100))
    assert e_true < 1e-12 and e_matlab > 3.9                        # MATLAB convention fails


@pytest.mark.parametrize("theta,pol", [(0, "s"), (30, "s"), (30, "p"), (60, "p")])
def test_front_back_extinction_equal_abs_differs(theta, pol):
    eps, d = es.graded_slab()
    f = np.linspace(0.25, 5, 120)
    cs = es.cross_sections(eps, d, f, theta, pol)
    assert np.abs(cs["ext1"] - cs["ext2"]).max() < 1e-12
    assert np.abs(cs["abs1"] - cs["abs2"]).max() > 0.05
    assert np.all(cs["abs1"] > 0) and np.all(cs["sca1"] > 0)


def test_published_curve_is_face_referenced():
    eps, d = es.graded_slab()
    cs = es.cross_sections(eps, d, np.array([0.25, 0.457, 0.647]))
    # published panel (c): ~3.35, ~0.85, ~2.9  vs physical 2.48, 3.08, 1.70
    assert np.allclose(cs["ext_matlab1"], [3.35, 0.85, 2.92], atol=0.03)
    assert np.allclose(cs["ext1"], [2.476, 3.078, 1.695], atol=0.01)


def test_dipole_cluster_reciprocal_vs_gyrotropic():
    rec = ed.run_trials(True, n_trials=2, seed=3)
    gyr = ed.run_trials(False, n_trials=2, seed=3)
    for c1, c2, cT in rec:
        assert abs(c1[0] - c2[0]) < 1e-10 * c1[0]
        assert abs(c1[0] - c1[1] - c1[2]) < 1e-8 * c1[0]            # energy conservation
    assert max(abs(c1[0] - c2[0]) / c1[0] for c1, c2, _ in gyr) > 1e-2
    for c1, _, cT in gyr:
        assert abs(c1[0] - cT[0]) < 1e-10 * c1[0]                   # generalized reciprocity
