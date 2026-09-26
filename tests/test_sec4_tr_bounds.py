"""Tests for sec4_global_symmetries/tr_field_ratio_bounds.py and change_of_basis_kirchhoff.py."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sec4_global_symmetries"))
import tr_field_ratio_bounds as tr  # noqa: E402
import change_of_basis_kirchhoff as cb  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(123)


def test_trs_parametrization_satisfies_constraint(rng):
    for n in (2, 4, 7):
        S = tr.random_trs_smatrix(n, rng)
        assert np.allclose(S, S.T) and np.allclose(S @ S.conj(), np.eye(n))
        E = tr.sample_feasible(S, 5, rng)
        for i in range(5):
            assert tr.constraint_residual(S, E[:, i]) < 1e-12


def test_g_S_equals_real_part_of_primed(rng):
    S = tr.random_trs_smatrix(5, rng)
    A = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    X = A + A.conj().T
    assert np.allclose(tr.g_S(X, S), tr.primed(X, S).real)


def test_two_port_envelope_exact(rng):
    for _ in range(5):
        S = tr.random_trs_smatrix(2, rng)
        lo, hi = tr.two_port_bound(abs(S[0, 0]))
        mlo, mhi = tr.ratio_bounds(S, np.diag([0.0, 1.0]), np.diag([1.0, 0.0]), True)
        plo, phi = tr.two_port_phase_sweep(S, 20001)
        assert np.isclose(mlo, lo, rtol=1e-9) and np.isclose(mhi, hi, rtol=1e-9)
        assert np.isclose(plo, lo, rtol=1e-4) and np.isclose(phi, hi, rtol=1e-4)


def test_physical_multilayer_inside_two_port_bound():
    R, ratio, resid = tr.multilayer_two_port(np.random.default_rng(1), n_freq=60, n_z=15)
    lo, hi = tr.two_port_bound(R)
    assert resid.max() < 1e-12
    assert np.all(ratio >= lo * (1 - 1e-8)) and np.all(ratio <= hi * (1 + 1e-8))


def test_complex_pencil_is_trivial_real_pencil_is_tight(rng):
    """The tutorial's complex Hermitian (W', V') gives [0, 1] for eta_1; Re pencil is attained."""
    n = 4
    S = tr.random_trs_smatrix(n, rng)
    W = np.zeros((n, n)); W[0, 0] = 1
    V = np.eye(n)
    clo, chi = tr.ratio_bounds(S, W, V, use_real=False)
    rlo, rhi = tr.ratio_bounds(S, W, V, use_real=True)
    assert abs(chi - 1) < 1e-10 and abs(clo) < 1e-10
    assert rhi < 0.99
    e = tr.eta(tr.sample_feasible(S, 50000, rng), 0)
    assert e.max() <= rhi + 1e-12
    assert abs(tr.optimize_eta(S, 0, +1, n_starts=5, rng=rng) - rhi) < 1e-6


def test_sounas_alu_satisfied_by_feasible_samples(rng):
    S = tr.random_trs_smatrix(4, rng)
    E = tr.sample_feasible(S, 20000, rng)
    assert np.all(tr.sounas_alu_ok(tr.eta(E, 0), tr.eta(E, 1), abs(S[0, 1])))


def test_intuition_amplitude_fraction():
    assert np.isclose(np.sqrt(0.9), 0.9487, atol=1e-4)
    lo, hi = tr.two_port_bound(np.sqrt(0.1))
    assert np.isclose(np.sqrt(lo), 0.721, atol=1e-3) and np.isclose(np.sqrt(hi), 1.387, atol=1e-3)


def test_kirchhoff_change_of_basis_matrices():
    ex = cb.kirchhoff_example()
    for key, claim in (("A1", cb.A1_CLAIM), ("E1", cb.E1_CLAIM), ("S2", cb.S2_CLAIM),
                       ("A2", cb.A2_CLAIM), ("E2", cb.E2_CLAIM)):
        assert np.allclose(ex[key], claim, atol=1e-14)
    w = cb.symmetry_preservation(n=4, trials=30)
    assert max(w.values()) < 1e-12
