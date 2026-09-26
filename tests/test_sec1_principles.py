"""Fast regression tests for the Ch. 1 (fundamental principles) validations."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sec1_fundamental_principles import kramers_kronig as kk
from sec1_fundamental_principles import passivity_herglotz as ph
from sec1_fundamental_principles import equivalence_principle as ep
from sec1_fundamental_principles import rayleigh_quotients as rq


# ---------------------------------------------------------------- Sec. 1.4 Kramers-Kronig
@pytest.mark.parametrize("seed", [0, 2])            # seed 2 contains a gain oscillator (l.396)
def test_kk_relations(seed):
    m = kk.random_insulator(seed, with_gain=(seed == 2))
    for w in (0.3, 1.1, 2.7):
        assert kk.kk_re(m, w) == pytest.approx(m.re(w), rel=1e-9)
        assert kk.kk_im(m, w) == pytest.approx(m.im(w), rel=1e-9)


def test_derivative_kk_positive():
    for w, fd, formula in kk.test_B():
        assert formula > 0
        assert fd == pytest.approx(formula, rel=1e-8)


def test_conductor_term_needs_eps0():
    rows, sigma0 = kk.test_C(ws=(0.5, 2.0))
    for r in rows:
        assert r["kkre"] == pytest.approx(r["re"], rel=1e-9)             # KKRe unmodified
        assert r["resid"] == pytest.approx(r["sig_eps0_w"], rel=1e-8)    # + sigma0/(eps0 w)
        assert r["resid"] / r["sig_w"] == pytest.approx(1 / kk.EPS0, rel=1e-8)


def test_asymptote():
    a, b = kk.test_D()
    assert a == pytest.approx(b, rel=1e-5)


# ---------------------------------------------------------------- Sec. 1.5 passivity
def test_bounded_real_and_poisson():
    A = ph.test_A()
    assert A["uhp_faces"] <= 1 + 1e-12 and A["re_axis"] <= 1 + 1e-12
    assert A["sym"] < 1e-12
    assert A["uhp_centre"] > 1.5                                  # l.437 caveat
    for z, err_faces, smax_c, err_centre in A["rows"]:
        assert err_faces < 1e-6                                   # [eq:SMatRep]
        assert err_centre > 1e-2


def test_matrix_herglotz():
    B = ph.test_B()
    assert B["min_eig"] >= 0 and B["min_eig_active"] < -1
    assert B["sym"] < 1e-12 and B["C"] < 1e-12 and B["D"] < 1e-9
    assert B["stieltjes"] < 1e-9


def test_herglotz_sum_rules():
    C = ph.test_C()
    for key in ("ins_p0", "ins_p1", "met_p0"):
        v, coeff, exact = C[key]
        assert v == pytest.approx(exact, rel=1e-9)
        assert coeff == pytest.approx(exact, rel=1e-6)
    assert C["met"]["a0"].imag == pytest.approx(80.0, rel=1e-6) and abs(C["met"]["a0"].real) < 1e-6
    assert abs(C["ins"]["a2"].real) < 1e-6 and abs(C["ins"]["bm2"].real) < 1e-6   # even orders imaginary
    growth = [v for _, v in C["met_p1_growth"]]
    assert growth[2] > 9 * growth[1] > 81 * growth[0]


# ---------------------------------------------------------------- Sec. 1.1 equivalence principle
def test_green_functions_follow_tutorial_conventions():
    res = ep.maxwell_residuals()
    assert res["J"] < 1e-7 and res["M"] < 1e-7
    pb, exact = ep.power_balance()
    for work, flux in pb.values():
        assert work == pytest.approx(exact, rel=1e-5) and flux == pytest.approx(exact, rel=1e-8)


def test_equivalence_principle():
    ei, eo = ep.test_ECinc(+1)
    assert ei < 1e-10 and eo < 1e-10
    ei, eo = ep.test_ECscat(+1)
    assert ei < 1e-10 and eo < 1e-10
    assert min(ep.test_ECinc(-1)) > 0.1 and min(ep.test_ECscat(-1)) > 0.1


# ---------------------------------------------------------------- Sec. 1.7 Rayleigh quotients
def test_fan_and_hoffman_wielandt():
    rng = np.random.default_rng(5)
    M, N = 5, 2
    A = rq.herm(M, rng)
    lam = np.sort(np.linalg.eigvalsh(A))[::-1]
    assert rq.optimize_trace(A, np.eye(N), "max", starts=3) == pytest.approx(lam[:N].sum(), abs=1e-7)
    Bp = rq.herm(N, rng, eigs=[1.3, 0.2])
    assert rq.optimize_trace(A, Bp, "max", starts=3) == pytest.approx(rq.hw_rhs(A, Bp, "down"), abs=1e-7)
    assert rq.optimize_trace(A, Bp, "min", starts=3) == pytest.approx(rq.hw_rhs(A, Bp, "up"), abs=1e-7)


def test_hw1_needs_psd_hypotheses():
    A = np.diag([1.0, 0.0, -1.0])
    B = np.diag([1.0, -1.0])                                   # indefinite, N < M
    X = np.eye(3)[:, [0, 2]]                                  # pairs b = -1 with lambda = -1
    assert np.trace(X.T @ A @ X @ B) == pytest.approx(2.0)
    assert rq.hw_rhs(A, B, "down") == pytest.approx(1.0)       # [eq:HW1] would say <= 1
    An = -np.eye(3)
    Bp = np.diag([1.0, 0.5])
    assert rq.hw_rhs(An, Bp, "down") < 0                       # X = 0 (allowed if X^dag X <= I) gives 0
