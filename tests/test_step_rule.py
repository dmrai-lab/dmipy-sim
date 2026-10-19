"""The step law's rule (dmrai-lab/dmipy-sim#715): the physics step is sized for a stated tolerance on the walk's
diffusion spectrum up to the band it serves, and the stepped matrix method confirms the tolerance holds."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.physics import tolerance_sub_steps, resolve_sub_steps, STEP_TOLERANCE
from dmipy_sim.math import step_law as sl
from dmipy_sim.acquisition.waveforms import band_hz

D0 = 2e-9


@pytest.mark.parametrize("shape, make, n_modes", [
    ("sphere", lambda: d.Sphere(2.8e-6), (6, 6)),
    ("cylinder", lambda: d.Cylinder(5.2e-6, (0.0, 0.0, 1.0)), (6, 6)),
])
def test_the_rule_s_step_holds_the_tolerance_on_the_pore_s_slowest_mode(shape, make, n_modes):
    """At the step the rule gives for eps_D with a low band (the wall term binding), the stepped operator's rate
    error on the slowest mode is within eps_D, and not below a third of it (the rule is a bound, not a margin)."""
    g = make()
    eps = 1e-3
    ls = g.length_scales
    dt_p = sl.pore_step(ls.pore_rate, D0, eps, ls.wall_coefficient)
    n = tolerance_sub_steps(g, D0, 100 * dt_p, eps_D=eps, f_band=10.0)
    assert n == 100                                                      # the wall term set it
    sp = sl.SteppedPore(shape, 2 * ls.min_feature, D0, g, n_modes=n_modes)
    i1 = np.argsort(sp.lam)[1]
    err = sp.rate_error(sp.law_operator(dt_p, n_rad=40, n_th=12, n_a=24, n_b=48), dt_p)[i1]
    assert err <= eps * 1.05 and err > eps / 3


def test_the_band_term_binds_on_a_large_pore_and_is_the_pair_s_bulk_law():
    """A 20 um sphere at a 1 kHz band: the wall term allows 68 us, the band's bulk term 85 us; the pair's rate
    error on the band's mode at that step is eps_D, and the fixed step's reach is ten times shorter."""
    g = d.Sphere(10e-6)
    eps = 1e-3
    ls = g.length_scales
    assert sl.pore_step(ls.pore_rate, D0, eps, ls.wall_coefficient) == pytest.approx(68e-6, rel=0.05)
    dt_b = sl.band_step(1000.0, eps)
    assert dt_b == pytest.approx(84.5e-6, rel=0.02)
    x = np.sqrt(12 * np.pi * 1000.0 * dt_b)                             # k delta_0 of the band's mode
    rate_err = abs(np.log(sl.bulk_multiplier(x)) / (-x ** 2 / 6) - 1)
    assert rate_err == pytest.approx(eps, rel=0.05)
    assert sl.band_reach(eps, sl.FIXED_LENGTHS, sl.FIXED_WEIGHTS) < 0.2 < 1.7 < sl.band_reach(eps)
    assert tolerance_sub_steps(g, D0, 1e-4, f_band=1000.0) == 2 and tolerance_sub_steps(g, D0, 1e-4, f_band=10.0) == 2
    assert tolerance_sub_steps(g, D0, 1e-4, f_band=5000.0) == int(np.ceil(1e-4 / sl.band_step(5000.0, eps))) == 6


def test_a_flat_wall_has_no_wall_term_and_the_first_order_regime_caps_the_step():
    g = d.Box1D(4e-6)
    assert g.length_scales.wall_coefficient == 0.0 and sl.pore_step(g.length_scales.pore_rate, D0, 1e-3, 0.0) == np.inf
    n = tolerance_sub_steps(g, D0, 1e-3, f_band=10.0)
    assert np.sqrt(6 * D0 * 1e-3 / n) <= 2e-6 * (1 + 1e-9) < np.sqrt(6 * D0 * 1e-3 / (n - 1))


def test_the_tolerance_is_the_cost_lever():
    g = d.Sphere(2e-6)
    n = {eps: tolerance_sub_steps(g, D0, 1e-4, eps_D=eps, f_band=100.0) for eps in (1e-4, 1e-3, 5e-3)}
    assert n[1e-4] > n[1e-3] > n[5e-3] >= 1
    assert n[1e-4] == pytest.approx(10 * n[1e-3], rel=0.1)


def test_the_envelope_serves_a_geometry_that_declares_no_rate():
    """The ellipsoid declares only its tightest semi-axis: the envelope ``4 / h^2`` and 0.85 size its step, which is
    tighter than the sphere of that radius (whose rate is ``(2.08 / R)^2 = 4.33 / R^2`` at the same coefficient)."""
    e = d.Ellipsoid((2e-6, 3e-6, 4e-6))
    assert e.length_scales.pore_rate is None and e.length_scales.wall_coefficient is None
    assert tolerance_sub_steps(e, D0, 1e-4, f_band=100.0) >= tolerance_sub_steps(d.Sphere(2e-6), D0, 1e-4, f_band=100.0)


def test_the_band_of_a_waveform_is_where_its_dephasing_power_ends():
    pgse = d.pgse([[1, 0, 0]], 0.01, 0.04, bvalues=[3e9], n_t=1000)
    ogse = d.ogse([[1, 0, 0]], 50.0, 0.02, bvalues=[1e9], n_t=1000)
    narrow = d.pgse([[1, 0, 0]], 0.004, 0.04, bvalues=[1e9], n_t=1000)
    b_p, b_o, b_n = band_hz(pgse.G_eff, pgse.dt), band_hz(ogse.G_eff, ogse.dt), band_hz(narrow.G_eff, narrow.dt)
    assert 50 < b_p < 150 and 100 < b_o < 300 and b_n > b_p
    assert band_hz(np.zeros((1, 100, 3)), 1e-4) is None


def test_the_drivers_resolve_the_band_they_serve():
    """The fused walk resolves the waveform's band, a bare trajectory walk the save grid's Nyquist frequency, and
    the walk records what sized its step."""
    from dmipy_sim.engine.physics import make_step_fn
    g = d.Sphere(10e-6)
    seq = d.pgse([[1, 0, 0]], 0.01, 0.04, bvalues=[1e9], n_t=200)
    fb = band_hz(seq.G_eff, seq.dt)
    assert make_step_fn(g, D0, seq.dt, f_band=fb)[0].n_sub == resolve_sub_steps(g, D0, seq.dt, f_band=fb)
    w = d.simulate_trajectories(16, D0, g, 2e-3, 2e-4, seed=1, require_gpu=False)
    rule = w.work["step_rule"]
    assert rule["eps_D"] == STEP_TOLERANCE and rule["f_band_hz"] == pytest.approx(w.K / (2 * w.T)) and rule["override"] is None
    assert rule["sub_steps"] == w.sub_steps == max(rule["rules"].values()) and rule["dt_p"] == pytest.approx(w.dt_sim)
    assert set(rule["rules"]) >= {"step law", "collision lookup", "crossing", "surface local time"}
    w2 = d.simulate_trajectories(16, D0, g, 2e-3, 2e-4, seed=1, require_gpu=False, f_band=100.0)
    assert w2.sub_steps <= w.sub_steps and w2.work["step_rule"]["f_band_hz"] == 100.0
    from dmipy_sim.persistent_walk import merge_work
    assert merge_work([w.work, w.work])["step_rule"] == rule                   # a bundle's pools at one rule: the rule
    assert merge_work([w.work, w2.work])["step_rule"] == [rule, w2.work["step_rule"]]   # at different rules: each
