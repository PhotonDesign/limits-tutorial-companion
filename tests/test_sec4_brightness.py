"""Tests for sec4_global_symmetries/brightness_theorem.py."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sec4_global_symmetries"))
import brightness_theorem as bt  # noqa: E402


def test_termwise_and_partial_sums():
    res = bt.termwise_trials(dims=range(2, 6), trials=300)
    for wt, ws in res.values():
        assert wt < 1e-12 and ws < 1e-12


def test_miller_example_needs_gain():
    ok_weak, det2 = bt.miller_counterexample()
    assert ok_weak and det2 > 1


def test_unitary_spectrum_invariant_and_lossy_entropy_drops():
    ws, we = bt.unitary_spectrum_invariance(trials=20)
    assert ws < 1e-12 and we < 1e-12
    h_in, h_out, tr_in, tr_out = bt.lossy_entropy_counterexample()
    assert h_out < h_in and tr_out < tr_in


def test_concentrator_one_over_N():
    for N in (5, 11):
        best, eff, passive, lam1 = bt.concentrator_efficiency(N, trials=100)
        assert passive and np.isclose(eff, 1 / N) and best <= 1 / N + 1e-12
    assert round(100 / 11) == 9


def test_tcmt_rank_bound():
    best, uerr = bt.tcmt_max_avg_transmission(3, 3, 1, trials=30, n_w=20)
    assert uerr < 1e-12 and best <= 1 / 3 + 1e-12
    assert np.isclose(bt.tcmt_optimal_single_resonance(), 0.5)
