"""Adaptive stepping (engine/adaptive.py) against the fused producer on a strand pack: the finest class alone
is the fused walk (the raw positions to rounding, the accumulated bands to float32 accumulation rounding); with free steps and radius classes the guarantees hold (no tube entered or left,
the box kept, contact non-positive) and the statistics agree within the Monte-Carlo floor."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.engine.adaptive import simulate_trajectories_adaptive

D = 2e-9


def _pack(interior):
    cls = [np.array([[x, 0, -30e-6], [x, 0.5e-6, 0], [x, 0, 30e-6]]) for x in (-6e-6, 0, 6e-6)]
    return d.PackedCurvedCylinders(cls, [1.0e-6, 1.5e-6, 2.0e-6], interior=interior, box=(np.full(3, -10e-6), np.full(3, 10e-6)))


def test_the_finest_class_alone_is_the_fused_walk():
    g = _pack(False)
    fixed = _walk_geometry(300, D, g, 1e-3, 2.5e-4, seed=3, require_gpu=False, sub_steps=64, raw_subset=300, K=8)
    adaptive = simulate_trajectories_adaptive(300, D, g, 1e-3, 2.5e-4, seed=3, require_gpu=False, sub_steps=64,
                                              steps_per_round=64, n_classes=1, safety_sigma=1e9, walker_batch_size=300,
                                              raw_subset=300, K=8)
    assert adaptive.sub_steps == fixed.sub_steps == 64
    assert adaptive.block["round_s"] == pytest.approx(adaptive.block["h"])      # one round per block: the finest class's
    # the same noise sequence and the same wall rule; the gathered kernel compiles to a different op order, so
    # float32 rounding differs by ulps (1e-11 m on 1e-5 m coordinates), not by a step
    np.testing.assert_allclose(adaptive.raw_positions, fixed.raw_positions, rtol=0, atol=1e-10)
    np.testing.assert_allclose(adaptive.raw_boundary, fixed.raw_boundary, rtol=1e-5, atol=1e-10)
    np.testing.assert_array_equal(adaptive.compartment, fixed.compartment)
    # the accumulated coefficients sum the same increments in a different order in float32: measured at K = 8 over
    # four blocks, the position bands differ by 3.5e-6 of the largest band, the boundary bands by 1.1e-5 of theirs and
    # the boundary endpoint by 4.1e-6 of its largest (the bands' start and displacement columns by 1.4e-11 m)
    for name, tol in (("position_coeffs", 1e-5), ("boundary_bands", 3e-5), ("boundary_endpoint", 1e-5)):
        a, f = np.asarray(getattr(adaptive, name)), np.asarray(getattr(fixed, name))
        np.testing.assert_allclose(a, f, rtol=0, atol=tol * np.abs(f).max(), err_msg=name)


@pytest.mark.parametrize("interior", [False, True])
def test_the_guarantees_and_the_statistics(interior):
    g = _pack(interior)
    n = 6000
    a = simulate_trajectories_adaptive(n, D, g, 2e-3, 2.5e-4, seed=1, require_gpu=False, walker_batch_size=n, raw_subset=n)
    f = _walk_geometry(n, D, g, 2e-3, 2.5e-4, seed=2, require_gpu=False, raw_subset=n)
    P = a.raw_positions.reshape(-1, 3)
    if interior:
        assert g.inside_any(P).all()                                   # never left its tube
    else:
        assert not g.inside_any(P).any()                               # never entered one
    assert (np.abs(a.raw_positions) <= 10e-6 + 1e-9).all() and (a.raw_boundary <= 0).all() and a.illegal_crossings == 0
    # intra gains by class, not by distance; the free step needs a walker beyond 6 sigma of a round (16 finest steps at
    # the family's R/3 rule: 3.4 um on this pack), which a few percent of the exterior walkers of this box are
    assert a.stepping["kernel_steps_ratio"] > 1.2 and (interior or a.stepping["free_fraction"] > 0.01)
    # end-to-end displacement variance per axis and the accumulated contact, within the floor of n walkers
    da = a.raw_positions[:, -1] - a.raw_positions[:, 0]; df = f.raw_positions[:, -1] - f.raw_positions[:, 0]
    va, vf = (da ** 2).mean(0), (df ** 2).mean(0)
    assert np.abs(va / vf - 1.0).max() < 5 * np.sqrt(2.0 / n) * 1.5, va / vf
    ca, cf = a.raw_boundary.sum(1), f.raw_boundary.sum(1)
    se = np.sqrt(ca.var() / n + cf.var() / n)
    assert abs(ca.mean() - cf.mean()) < 4 * se, (ca.mean(), cf.mean(), se)


def test_a_geometry_without_wall_scales_is_refused():
    with pytest.raises(TypeError, match="wall_scales"):
        simulate_trajectories_adaptive(10, D, d.Sphere(3e-6), 1e-3, 5e-4, require_gpu=False)


def test_the_candidate_cache_is_the_full_gather():
    """A round stepped against the segments within its reach is the round stepped against the 27-cell gather:
    the same walls are met, so the same positions and contact (to rounding), from the same noise."""
    g = _pack(False)
    kw = dict(seed=5, require_gpu=False, walker_batch_size=2000, raw_subset=2000)
    full = simulate_trajectories_adaptive(2000, D, g, 2e-3, 2.5e-4, candidate_cache=False, **kw)
    cache = simulate_trajectories_adaptive(2000, D, g, 2e-3, 2.5e-4, candidate_cache=True, candidate_k_start=4, **kw)
    assert cache.stepping["candidate_cache"] and cache.stepping["candidate_k"] >= 4
    # the same segments decide every reflection; the two programs round differently at float32, and a grazing
    # step within an ulp of a wall may flip between hit and miss -- a handful of (walker, save) pairs, by less
    # than one step (0.13 um), never a wall crossed
    diff = np.abs(cache.raw_positions - full.raw_positions).max(-1)
    assert (diff > 1e-10).mean() < 2e-3 and diff.max() < 0.13e-6, ((diff > 1e-10).mean(), diff.max())
    dl = np.abs(cache.raw_boundary - full.raw_boundary)
    assert (dl > 1e-10).mean() < 2e-3 and not g.inside_any(cache.raw_positions.reshape(-1, 3)).any()


def test_the_walk_samples_the_field_it_carries():
    """The raw subset keeps the walk's own field reads at its read instants: with one round per block the reads are
    at the block ends, and equal the standalone basis evaluated at the stored positions (the first the start's read),
    the domain mean subtracted -- exactly so when the interval's segment list cannot change (straight strands, every
    strand within the cutoff throughout); a strand crossing the cutoff or a joint between the interval's gather and
    its end is the difference otherwise, at the size of that strand's field at the cutoff. With a read every round of
    four per block the reads at the block ends are those and the reads between them are the field along the way; the
    read series' quadratic variation is the reads' own."""
    from dmipy_sim.fields.strand_field import StrandFieldBasis
    cls = [np.array([[x, 0, -30e-6], [x, 0, 30e-6]]) for x in (-6e-6, 0, 6e-6)]
    g = d.PackedCurvedCylinders(cls, [1.0e-6, 1.5e-6, 2.0e-6], interior=False, box=(np.full(3, -10e-6), np.full(3, 10e-6)))
    sf = StrandFieldBasis(cls, [0.7e-6, 1.05e-6, 1.4e-6], [1.0e-6, 1.5e-6, 2.0e-6], cutoff_m=60e-6,
                          domain=(np.full(3, -10e-6), np.full(3, 10e-6)))
    w = simulate_trajectories_adaptive(400, D, g, 1e-3, 2.5e-4, seed=4, require_gpu=False, walker_batch_size=400,
                                       sub_steps=64, steps_per_round=64, field_basis=sf, raw_subset=400)
    assert w.raw_field.shape == (400, 5, 12) and w.field_basis is sf and w.has_field
    at = [sf.channels(w.raw_positions[:, k]) for k in range(5)]
    for k in range(5):
        np.testing.assert_allclose(w.raw_field[:, k], at[k], rtol=0, atol=2e-6)
    qv = (np.diff(np.asarray(w.raw_field, np.float64), axis=1) ** 2).sum(1)
    np.testing.assert_allclose(np.asarray(w.qv_field_increments, np.float64)[:, 0], qv, rtol=1e-4, atol=1e-6 * qv.max())
    # a read every round of four per block: the block ends among the reads, the rounds between them
    w2 = simulate_trajectories_adaptive(400, D, g, 1e-3, 2.5e-4, seed=4, require_gpu=False, walker_batch_size=400,
                                        sub_steps=64, steps_per_round=16, field_basis=sf, raw_subset=400,
                                        field_read_interval=2.5e-4 / 4)
    assert w2.raw_field.shape == (400, 17, 12) and w2.reads_per_block == 4
    for k in range(5):
        np.testing.assert_allclose(w2.raw_field[:, 4 * k], sf.channels(w2.raw_positions[:, k]), rtol=0, atol=2e-6)
    assert np.abs(w2.raw_field[:, 2] - w2.raw_field[:, 4]).max() > 1e-4         # the field along the way


def test_every_class_steps_within_the_band_term():
    """The classes double in step time from the smallest feature (the rule's pore term at the class radius), and
    none steps longer than the rule's band term, which does not grow with the radius (dmrai-lab/dmipy-sim#715):
    at a low band the doubling stands, at a high one the coarse classes are capped at the band's step, a power of
    two so each still divides the round; the walk's stepping record carries the tolerance and the band."""
    from dmipy_sim.engine.adaptive import adaptive_plan
    from dmipy_sim.engine.physics import STEP_TOLERANCE
    from dmipy_sim.math.step_law import band_step
    g = _pack(False)
    low = adaptive_plan(g, D, 2.5e-4, f_band=100.0)
    assert low.steps_c == (16, 8, 4, 2) and low.eps_D == STEP_TOLERANCE and low.f_band_hz == 100.0
    assert low.dt_band_s == pytest.approx(band_step(100.0, STEP_TOLERANCE))
    high = adaptive_plan(g, D, 2.5e-4, f_band=2e4)
    assert high.dt_band_s == pytest.approx(band_step(2e4, STEP_TOLERANCE)) and high.dt_band_s < high.dt_round
    assert all(high.dt_round / m <= high.dt_band_s * (1 + 1e-9) for m in high.steps_c)
    assert high.steps_c[0] == 16 and high.steps_c[-1] > 2 and all(m & (m - 1) == 0 for m in high.steps_c)
    assert high.n_min >= low.n_min
    w = simulate_trajectories_adaptive(64, D, g, 5e-4, 2.5e-4, seed=1, require_gpu=False, walker_batch_size=64, f_band=2e4)
    assert w.block["round_s"] == pytest.approx(high.dt_round)        # the round the certificate charges beside the block's
    st = w.stepping
    assert st["eps_D"] == STEP_TOLERANCE and st["f_band_hz"] == 2e4 and st["dt_band_s"] == pytest.approx(high.dt_band_s)
    assert tuple(st["steps_per_round_by_class"]) == high.steps_c


def test_a_refined_plan_shortens_every_step_by_the_refinement():
    """``step_refinement=r`` multiplies the finest sub-step count by ``r``: the rounds and every class's step shorten
    by ``r`` together, the band term with them, so a refined walk is the same walk at a uniformly finer step (what a
    two-walk refinement comparison needs). Anything but a positive integer is refused."""
    from dmipy_sim.engine.adaptive import adaptive_plan
    g = _pack(False)
    for f_band in (100.0, 2e4):
        base = adaptive_plan(g, D, 2.5e-4, f_band=f_band)
        fine = adaptive_plan(g, D, 2.5e-4, f_band=f_band, step_refinement=3)
        assert fine.n_min == 3 * base.n_min and fine.n_rounds == 3 * base.n_rounds
        assert fine.steps_c == base.steps_c and fine.dt_round == pytest.approx(base.dt_round / 3)
        assert fine.dt_band_s == pytest.approx(base.dt_band_s / 3)
        np.testing.assert_allclose(np.array(fine.step_l_c), np.array(base.step_l_c) / np.sqrt(3), rtol=1e-6)
    for bad in (0, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            adaptive_plan(g, D, 2.5e-4, step_refinement=bad)


def test_a_calibration_reference_walks_the_given_field_settings_without_a_pilot():
    """A diagnostic reference walk reads the field at the band and interval it is given, runs no field pilot, and
    records itself as unresolved and over no budget it could meet; it cannot be asked without both settings, nor
    alongside a production calibration or scope."""
    from dmipy_sim.fields.strand_field import StrandFieldBasis
    cls = [np.array([[x, 0, -30e-6], [x, 0, 30e-6]]) for x in (-6e-6, 0, 6e-6)]
    g = d.PackedCurvedCylinders(cls, [1.0e-6, 1.5e-6, 2.0e-6], interior=False, box=(np.full(3, -10e-6), np.full(3, 10e-6)))
    sf = StrandFieldBasis(cls, [0.7e-6, 1.05e-6, 1.4e-6], [1.0e-6, 1.5e-6, 2.0e-6], cutoff_m=60e-6,
                          domain=(np.full(3, -10e-6), np.full(3, 10e-6)))
    kw = dict(seed=4, require_gpu=False, walker_batch_size=200, sub_steps=64, steps_per_round=16, field_basis=sf)
    w = simulate_trajectories_adaptive(200, D, g, 1e-3, 2.5e-4, K_c3=2, field_read_interval=2.5e-4 / 4,
                                       calibration_reference=True, step_refinement=2, **kw)
    fr = w.block["field_read"]
    assert fr["calibration_reference"] and fr["calibration_resolved"] is False and fr["pilot_seconds"] == 0.0
    assert w.stepping["step_refinement"] == 2 and w.stepping["field_pilot_backend"] is None
    with pytest.raises(ValueError, match="explicit field source, band and read interval"):
        simulate_trajectories_adaptive(200, D, g, 1e-3, 2.5e-4, K_c3=2, calibration_reference=True, **kw)
    with pytest.raises(ValueError, match="cannot assert scoped production evidence"):
        simulate_trajectories_adaptive(200, D, g, 1e-3, 2.5e-4, K_c3=2, field_read_interval=2.5e-4 / 4,
                                       calibration_reference=True, query_scope={"q": 1}, **kw)
