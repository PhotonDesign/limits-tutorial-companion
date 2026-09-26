"""Fast checks of Secs. 5.4-5.9 (sec5_4_9_scattering_limits/): polarizability and Q limits,
absorption/scattering tradeoff, cloaking bandwidth, sensing LOD, delay-bandwidth."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sec5_4_9_scattering_limits.polarizability_q_limits import (
    check_passivity, check_polar_corr, check_static, thal_ratio, check_lorentzian_bandwidth,
    plasmonic_q, GEOMS, mie_lossy_q, check_energy_balance)
from sec5_4_9_scattering_limits.absorption_scattering_tradeoff import (
    random_spheres, cross_sections_s, andersen_quantities, disk_scan)
from sec5_4_9_scattering_limits.cloaking_bandwidth import rule_of_thumb, pec_sphere_sigma_sca
from sec5_4_9_scattering_limits.sensing_lod import (
    lod_numeric, lod_printed, lod_corrected, amplitude)
from sec5_4_9_scattering_limits.delay_bandwidth import (
    tucker_max_lp, tucker_exact, tucker_printed, pulse_products, resonator_delays)
from sec5_local_constraints.mie import cross_sections


def test_sipe_kranendonk_and_max_alpha():
    mn, lossless_dev, amax = check_passivity(N=800)
    assert mn >= 1 - 1e-10 and lossless_dev < 1e-10 and amax <= 1 + 1e-12


def test_polar_corr_limit():
    errs = {(e, x): err for e, x, err in check_polar_corr()}
    for eps in (4.0, -5 + 1j, -2 + 0.5j):
        assert errs[(eps, 0.01)] < 1e-3
        assert errs[(eps, 0.01)] < errs[(eps, 0.1)] / 50        # ~ (ka)^2


def test_static_bound():
    amax, dmono, mie = check_static(N=2000)
    assert amax <= 1 and dmono >= 0
    assert all(v.real <= 1 + 1e-6 for _, v in mie)
    assert abs(mie[-1][1] - 1) < 1e-4


def test_lorentzian_bandwidth_l2064():
    dw, dw_th, amx, amx_th = check_lorentzian_bandwidth()
    assert abs(dw / dw_th - 1) < 1e-4 and abs(amx / amx_th - 1) < 1e-6


def test_thal_limit_lossless_drude():
    assert abs(thal_ratio(0.017, 1.0)[0] - 1) < 2e-4
    assert abs(thal_ratio(0.171, 1.0)[0] - 1.0054) < 1e-3
    assert abs(thal_ratio(0.017, 4.0)[0] - 2.0) < 1e-3          # (eps_inf + 2)/3, not 1


def test_plasmonic_q_geometry_independent():
    for einf in (1.0, 5.0):
        for name in ("sphere, vacuum", "prolate spheroid L = 0.1", "coated sphere core 2.1, f = 0.5 (mode 1)"):
            r = plasmonic_q(GEOMS[name], 0.01, einf)
            assert abs(r["im"] + 0.005) < 1e-10                 # Im w~ = -gamma/2
            assert abs(r["Q_pole"] / r["Q_gamma"] - 1) < 1e-9
            assert abs(r["Q_fwhm"] / r["Q_gamma"] - 1) < 2e-2
            assert abs(r["Q_ws"] / r["Q_gamma"] - 1) < 5e-3
    Q, Qa, Qc, wr = mie_lossy_q(0.01)
    assert abs(Q / Qa - 1) < 3e-3


def test_energy_balance_sphere_mode():
    rat, s, jump = check_energy_balance()
    assert abs(rat - 2) < 1e-7 and abs(s) < 1e-6 and abs(jump) < 1e-6


def test_cross_sections_in_s_form_and_disk():
    m, x = random_spheres(100)
    cs, css = cross_sections(m, x), cross_sections_s(m, x)
    for q in ("ext", "sca", "abs"):
        assert np.max(abs(cs[q] - css[q]) / np.maximum(abs(cs[q]), 1e-12)) < 1e-6
    smax, s0, ssc = disk_scan(npts=401)
    assert abs(smax - 1 / (8 * np.pi)) < 1e-12 and abs(ssc - smax) < 1e-12


def test_andersen_inequality():
    m, x = random_spheres(1500, seed=5)
    for lam in (1.0, 1e-6, 1e3):
        ext, sca, ab, D, S0 = andersen_quantities(m, x, lam)
        corrected = ext / (lam * np.sqrt(sca * D / np.pi))
        printed = ext / np.sqrt(sca * D / np.pi)
        assert corrected.max() <= 1 + 1e-10                         # lambda sqrt(sigma_s D/pi)
        assert abs(printed.max() / corrected.max() - lam) < 1e-6 * lam   # printed form scales with units
        assert (ab / (lam ** 2 * D / (4 * np.pi))).max() <= 1 + 1e-10


def test_cloaking_numbers():
    lam0 = 500e-9
    assert abs(rule_of_thumb(lam0, 1.0, 5.0, "1/domega") * 100 - 1.59e-6) < 1e-8     # tau = 1/dw
    assert abs(rule_of_thumb(lam0, 1.0, 5.0, "1/df = 2pi/domega") - lam0 / 5) < 1e-20  # printed form
    assert abs(pec_sphere_sigma_sca(200.0) / np.pi - 2) < 1e-2
    assert abs(100 * (lam0 / 1.0) / np.log(10) - 2.17e-5) < 1e-7                     # l.2444 example


def test_lod_factor_two():
    for lam0, S, QR, QNR, sig in ((1550.0, 100.0, 1e4, 1e4, 1e-4), (800.0, 300.0, 200.0, 50.0, 1e-5)):
        L, A0 = lod_numeric(lam0, S, QR, QNR, sig)
        assert abs(lod_corrected(lam0, S, QR, A0, sig) / L - 1) < 3e-3
        assert abs(lod_printed(lam0, S, QR, A0, sig) / L - 2) < 6e-3
    assert abs(1e4 * amplitude(1.0, 1.0, 1e4, 1e4) - 2500) < 1e-9


def test_tucker_dropped_term():
    for nmax, nmin, fb in ((3.5, 1.45, 0.2), (1.5, 1.45, 0.02)):
        lp = tucker_max_lp(nmax, nmin, fb)
        assert abs(lp - tucker_exact(nmax, nmin, fb)) < 1e-9
        assert lp > 1.2 * tucker_printed(nmax, nmin)       # violated although dw/wc << 2


def test_time_bandwidth_and_resonator():
    p = pulse_products(N=2 ** 20, T=2000.0)
    assert abs(p["Gaussian"][0] - 4 * np.log(2)) < 1e-2
    assert abs(p["one-sided exponential"][0] - np.log(2)) < 1e-2    # < 2
    assert abs(p["Gaussian"][1] - 0.5) < 1e-4
    rd = resonator_delays()
    assert abs(rd["one-port reflection delay"] - 2) < 1e-2
    assert abs(rd["two-port transmission delay"] - 1) < 1e-2
    assert abs(rd["|a|^2 FWHM"] - 2) < 1e-4
