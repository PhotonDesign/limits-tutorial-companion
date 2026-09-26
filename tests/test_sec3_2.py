"""Fast regression tests for the Sec. 3.2 (complex media) validations."""
import numpy as np
import pytest

import complex_media_focusing as cmf

X0 = np.zeros(3)
PHI = np.array([0.3 + 0.1j, -0.5, 0.8j]) / np.linalg.norm([0.3 + 0.1j, -0.5, 0.8j])


@pytest.mark.parametrize("kind,reciprocal", [("lossless-reciprocal", True), ("lossy-reciprocal", True),
                                             ("tri-nonreciprocal", False), ("gyrotropic", False)])
def test_reciprocity_recipe(kind, reciprocal):
    med = cmf.make_medium(kind)
    pts, w = cmf.patch_nodes(0.9, 0.6, 10)
    lam, lam_eig, e1, e2 = cmf.focusing_efficiencies(med, pts, w, X0, PHI)
    assert lam_eig == pytest.approx(lam, rel=1e-8)
    assert e2 == pytest.approx(1, abs=1e-10)          # transposed-medium recipe always optimal
    if reciprocal:
        assert e1 == pytest.approx(1, abs=1e-10)
    else:
        assert e1 < 0.9


def test_time_reversal_identity():
    out = cmf.part_B(n_th=120, n_ph=240)
    assert out["lossless-reciprocal"][0] < 1e-4
    assert out["tri-nonreciprocal"][0] < 1e-3          # TRI suffices (no reciprocity needed)
    assert out["lossy-reciprocal"][0] > 0.1
    assert out["gyrotropic"][0] > 0.1
