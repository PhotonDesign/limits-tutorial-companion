"""Tests for sec3_4_communication_channels/communication_modes.py."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sec3_4_communication_channels"))
import communication_modes as cm  # noqa: E402


def test_figure_percentages_match_100_lambda_not_50():
    _, p100 = cm.power_fractions(cm.miller_arrays(100.0))
    _, p50 = cm.power_fractions(cm.miller_arrays(50.0))
    assert np.abs(p100[:12] - cm.FIGURE_PERCENT).max() < 0.006      # printed to 2 decimals
    assert p50[0] < 4.0 and np.abs(p50[:12] - cm.FIGURE_PERCENT).max() > 5


def test_channel_count_near_paraxial_estimate():
    _, p = cm.power_fractions(cm.continuous_apertures(16, 50, 0.5))
    n_strong = (p > p[0] / 2).sum()
    assert 20 <= n_strong <= 30                                     # N_H = 26.2


def test_singular_vector_roles():
    c = cm.svd_label_check()
    assert c["err_map"] < 1e-10 and c["crosstalk"] < 1e-10
    assert np.isclose(c["E_right"], c["kyfan"]) and c["E_random"] < c["kyfan"]
