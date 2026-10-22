"""Fast regression tests for the Secs. 2.1-2.2 Drude-Lorentz sandbox (sec2_material_bounds/drude_lorentz_sandbox.py)."""
import itertools
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sec2_material_bounds import drude_lorentz_sandbox as dl
from sec2_material_bounds.drude_lorentz_sandbox import Params


# ---------------------------------------------------------------- [eq:f-sum rule]
@pytest.mark.parametrize("w0,wp,g,fD", list(itertools.product((0.3, 1.0, 2.5), (0.5, 2.0), (1e-3, 0.05, 0.5),
                                                               (0.0, 0.4, 1.0))))
def test_f_sum_rule(w0, wp, g, fD):
    assert dl.fsum_ratio(Params(w0, wp, g, fD)) == pytest.approx(1.0, abs=1e-3)


def test_model_is_passive_and_derivative_correct():
    w = np.linspace(0.05, 3, 500)
    h = 1e-6
    for p in (Params(1.3, 1.1, 0.07, 0.3), Params(0.4, 2.0, 0.5, 0.0), Params(2.0, 0.7, 0.01, 1.0)):
        assert np.all(dl.eps(w, p).imag > 0)
        fd = (dl.eps(w + h, p) - dl.eps(w - h, p)) / (2 * h)
        assert np.allclose(dl.deps(w, p), fd, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------- [eq:max_n]
def test_max_n_never_exceeded_lossless():
    worst, _, count = dl.sweep_max_n()
    assert count > 300
    assert worst <= 1 + 1e-9


@pytest.mark.parametrize("w0,wp", [(1.05, 0.3), (1.5, 1.0), (3.0, 2.5)])
def test_single_lossless_oscillator_saturates_max_n(w0, wp):
    lhs, rhs, n, _ = dl.max_n_sides(Params(w0, wp, 0.0, 0.0))
    assert n > 1 and lhs == pytest.approx(rhs, rel=1e-10)
    lhs, rhs, n, _ = dl.max_n_sides(Params(w0, wp, 0.0, 0.05))      # Drude share: strictly below
    assert n > 1 and lhs < 0.99 * rhs


# ---------------------------------------------------------------- [eq:low_epsilon_bound]
def test_low_epsilon_bound_never_violated_lossless():
    worst, _ = dl.sweep_low_eps()
    assert worst >= 1 - 1e-12


def test_low_epsilon_bound_damped_models():
    """Not a tutorial claim (the bound is stated for lossless media): damped one-oscillator models
    stay above it on this grid too."""
    worst, _ = dl.sweep_low_eps(gammas=(0.01, 0.3), w0_grid=dl.GRID_W0[::5])
    assert worst >= 1


@pytest.mark.parametrize("B", [0.1, 0.4, 1.0, 1.6])
def test_drude_limit_closed_form(B):
    R, em = dl.model_low_eps(Params(0.0, 1.3, 0.0, 1.0), B)
    w1, w2 = dl.band(B)
    assert R == pytest.approx((w2**2 - w1**2) / (w2**2 + w1**2), rel=1e-9)
    assert R == pytest.approx(B / (1 + B**2 / 4), rel=1e-9)
    assert R > dl.bound_low_eps(B)


def test_closed_form_matches_numerical_minimization():
    for p, B in [(Params(0.0, 1.0, 0.0, 1.0), 0.4), (Params(0.5, 1.5, 0.0, 0.0), 1.0),
                 (Params(0.6, 1.5, 0.0, 0.5), 0.4), (Params(1.5, 1.0, 0.0, 0.0), 0.4)]:
        Rc, _ = dl.model_low_eps(p, B, closed_form=True)
        Rn, _ = dl.model_low_eps(p, B, closed_form=False)
        assert Rn == pytest.approx(Rc, rel=1e-6)
