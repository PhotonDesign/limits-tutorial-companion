"""Fast checks of the Sec. 5.2 extinction sum rules (sec5_local_constraints/extinction_sum_rules.py)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sec5_local_constraints.extinction_sum_rules import (
    SILVERLIKE, DIELECTRIC, C0, depolarization_prolate, static_sum_rule_qs, highfreq_sum_rule_qs,
    mie_highfreq_sum_rule, mie_static_sum_rule, kk_check, plasma_contributions)


def test_depolarization_factor():
    assert abs(depolarization_prolate(1.0, 1.0) - 1 / 3) < 1e-14
    assert abs(depolarization_prolate(3.0, 1.0) - 0.10871) < 1e-5     # standard tabulated value


def test_quasistatic_sum_rules_SI():
    for mat in (SILVERLIKE, DIELECTRIC):
        V = 4 * np.pi * 30e-9 * (10e-9) ** 2 / 3
        L = depolarization_prolate(30, 10)
        s, t = static_sum_rule_qs(mat, V, L)
        assert abs(s / t - 1) < 1e-8                          # SRExt0
        val, corrected, printed = highfreq_sum_rule_qs(mat, V, L)
        assert abs(val / corrected - 1) < 1e-8                # pi wp^2 V/(2c)
        assert abs(val / printed * C0 - 1) < 1e-8             # printed SRExtInf off by 1/c


def test_mie_sum_rules():
    hf = mie_highfreq_sum_rule()
    assert abs(hf["total"] - 1) < 1e-3
    st = mie_static_sum_rule()
    assert abs(st["ext"] - 1) < 1e-3
    assert st["sca"] < 0.5                                    # l.1825: it is extinction, not scattering


def test_gendisp_needs_omega_prime():
    for w in (0.5, 0.9, 3.0):
        exact, kk, gd = kk_check(w)
        assert abs(kk - exact) < 1e-7
        assert abs(gd - exact) > 1e-3


def test_nuclear_plasma_weight():
    r, rp = plasma_contributions(14, 28.0855)
    assert 2e-4 < r < 3e-4 and 5e-4 < rp < 6e-4
