"""Fast checks of the Sec. 5.2.3 T-matrix oscillator representation (sec5_local_constraints/tmatrix_oscillators.py)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sec5_local_constraints.tmatrix_oscillators import (
    single_lorentz, QuasiStaticT, GyrotropicT, sum_rules, reconstruct, XY)


def test_single_lorentz_sum_rule_is_pi_over_2():
    I1, I2, T0 = single_lorentz()
    assert abs(I1 - np.pi / 2) < 1e-9          # > 1 = printed bound w_p^2
    assert abs(I2 - np.pi / 2 * T0) < 1e-9


def test_quasistatic_cluster_sum_rules_and_reconstruction():
    qt = QuasiStaticT()
    J1, J2 = sum_rules(qt, qt.res)
    assert np.abs(J1 - np.pi / 2 * np.eye(qt.n)).max() < 1e-9
    assert np.abs(J2 - np.pi / 2 * qt.T0).max() < 1e-8 * np.abs(qt.T0).max()
    w = 1.3 + 0.25j
    assert np.abs(reconstruct(qt, w) - qt(w)).max() < 1e-10 * np.abs(qt(w)).max()


def test_gyrotropic_Y_term_has_no_i():
    gt = GyrotropicT()
    X, Y = XY(gt, 0.9)
    assert np.linalg.eigvalsh(X + Y).min() >= -1e-12 and np.linalg.eigvalsh(X - Y).min() >= -1e-12
    w = 1.1 + 0.3j
    T = gt(w)
    assert np.abs(reconstruct(gt, w) - T).max() < 1e-10 * np.abs(T).max()
    assert np.abs(reconstruct(gt, w, with_i=True) - T).max() > 0.1 * np.abs(T).max()
