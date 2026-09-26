"""Fast checks of the Sec. 5.1/5.3 VIE conservation laws (sec5_local_constraints/vie_conservation_laws.py)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sec5_local_constraints.vie_conservation_laws import (
    run_general, global_constraints, local_constraints, far_field_powers, ImH,
    footnote_scat_check, domain_oblivious_check, lifted_constraint_rank, material_bounds_check)


def test_ImGamma0_psd_and_symmetric():
    m = run_general(seed=0)
    assert np.linalg.eigvalsh(ImH(m["G0"])).min() > -1e-12
    assert np.abs(m["G0"] - m["G0"].T).max() < 1e-14


def test_optical_theorem_holds_and_reactive_sign():
    for seed, rec in ((0, False), (5, True)):
        m = run_general(seed=seed, reciprocal=rec)
        g = global_constraints(m["G0"], m["xi"], m["phi"], m["psi"])
        assert g["optical_theorem"] < 1e-12
        assert g["reactive_corrected"] < 1e-12            # -Re(psi^dag phi) on the RHS
        assert g["reactive_as_printed"] > 1e-2            # tutorial l.610/1766 sign: fails


def test_local_constraints_and_power_identities():
    m = run_general(seed=0)
    G0, xi, phi, psi = m["G0"], m["xi"], m["phi"], m["psi"]
    loc = local_constraints(G0, xi, phi, psi)
    assert np.abs(loc["res"]).max() < 1e-12
    assert loc["real_power_resid"].max() < 1e-12 and loc["reactive_resid"].max() < 1e-12
    k, v = m["k"], m["v"]
    Pscat_ff, Pext_fwd = far_field_powers(m["pos"], v, k, phi, m["khat"], m["e"])
    Pscat = 0.5 * k * v * np.real(phi.conj() @ ImH(G0) @ phi)
    Pext = 0.5 * k * v * np.imag(psi.conj() @ phi)
    assert abs(Pscat / Pscat_ff - 1) < 1e-10 and abs(Pext / Pext_fwd - 1) < 1e-12


def test_footnote_scattering_expression():
    psi = np.ones(6, complex)
    ext, corr, printed = footnote_scat_check(-4 + 1.5j, psi)
    assert abs(corr) < 1e-12 * ext
    assert abs(printed - ext) < 1e-12 * ext              # printed form just returns P_ext


def test_domain_oblivious_local_vs_nonlocal():
    d = domain_oblivious_check()
    assert d["IE_resid_in_Vs"] < 1e-12 and d["IE_resid_outside_Vs"] > 0.1
    assert d["local_max"] < 1e-12 and d["global_resid"] < 1e-12
    assert d["nonlocal_max"] > 1e-3


def test_constraint_count_and_material_bounds():
    m = run_general(seed=1, shape=(2, 2, 2))
    rank, n = lifted_constraint_rank(m["G0"], m["xi"], m["psi"])
    assert rank == n == 8 * 9
    b = material_bounds_check()
    assert b["ext_V"] <= b["bound"] and b["abs_V"] <= b["bound"] and b["scat_V"] <= b["bound"] / 4
    assert b["ext_V"] <= b["ext_OT_bound_V"] and b["phi_opt_constraint_resid"] < 1e-10
