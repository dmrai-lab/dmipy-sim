"""The work of a walk (dmrai-lab/dmipy-sim#623): wall encounters, crossings and budget exhaustions on
``PersistentWalk.work``, counted only when the geometry is asked to, never moving a walker.

Counting is a carried integer in the bounce loop and costs (+5 % on an analytic wall, measured on the L40S), so
``Geometry.count_walls`` is off by default and a counted walk must give the same trajectories as an uncounted one.
The count itself is checked against physics: a walker meets the wall of a convex pore on a fraction
``(S/V) l / 4`` of its steps of length ``l`` -- the surface-to-volume estimator the boundary local time is built
on, in discrete form. And ``exhausted_steps`` is the certificate that a bounce budget is sufficient: a budget no lane
exhausted gives the trajectories of any larger budget, by the loop's own rule (the leftover path is flown only when
the last iteration met nothing), while a budget that is exhausted cuts path and moves walkers.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.core import simulate_trajectories
from dmipy_sim.geometry._boundary import BounceLoop
from dmipy_sim.persistent_walk import PersistentWalk

D = 2e-9
R = 2e-6


def _walk(g, n=256, seed=11, **kw):
    return simulate_trajectories(n, D, g, 2e-3, 0.5e-3, seed=seed, tiers="all", walker_batch_size=n, **kw)


def _packed():
    c = np.array([[0.0, 0.0], [2.2 * R, 0.0], [0.0, 2.2 * R], [2.2 * R, 2.2 * R]])
    return d.PackedCylinders([R, 0.8 * R, R, 0.9 * R], c, 4.4 * R, permeability=2e-5)


def _mesh():
    from dmipy_sim.geometry import mesh_shapes
    V, F = mesh_shapes.icosphere(R, subdivisions=2)
    return d.Mesh(V, F, feature_radius=R)


def _label():
    from dmipy_sim.geometry.label_volume import LabelVolume
    lab = np.ones((30, 8, 8), np.uint8); lab[8:22] = 0
    return LabelVolume(lab, 0.5e-6)


@pytest.mark.parametrize("make", [lambda: d.Sphere(R), _packed, _mesh, _label], ids=["sphere", "packed", "mesh", "label"])
def test_counting_moves_no_walker_and_an_uncounted_walk_reports_none(make):
    plain = _walk(make())
    g = make(); g.count_walls = True
    counted = _walk(g)
    np.testing.assert_array_equal(plain.positions, counted.positions)
    np.testing.assert_array_equal(plain.boundary_local_time, counted.boundary_local_time)
    assert plain.work["n_hits"] is None and plain.work["exhausted_steps"] == 0      # exhaustion on every walk (#634)
    assert plain.work["walker_steps"] == counted.work["walker_steps"] == plain.positions.shape[0] * 4 * plain.sub_steps
    assert plain.work["bounce_budget"] == g.bounce_loop.budget
    assert counted.work["n_hits"] > 0 and counted.work["exhausted_steps"] == 0
    if getattr(g, "permeability", None) is None:
        assert counted.work["n_crossings"] == 0


def test_a_permeable_walk_counts_its_crossings():
    g = _packed(); g.count_walls = True
    w = _walk(g)
    assert 0 < w.work["n_crossings"] <= w.work["n_hits"]


def test_the_hit_rate_is_the_surface_to_volume_law():
    """A sphere of radius R meets its wall on (S/V) l / 4 = 3 l / (4 R) of its steps, to a few percent."""
    g = d.Sphere(R); g.count_walls = True
    w = _walk(g, n=4000, seed=3)
    l = np.sqrt(6.0 * D * w.dt_sim)
    rate = w.work["n_hits"] / w.work["walker_steps"]
    assert rate == pytest.approx(3.0 * l / (4.0 * R), rel=0.05)


def test_exhausted_steps_certify_the_budget():
    """No lane exhausted at a smaller budget -> the same trajectories as the derived budget; a budget that is
    exhausted moves walkers, and says so."""
    cls = d.Sphere
    original = cls.__dict__["bounce_loop"]

    def counted():
        g = d.Sphere(R); g.count_walls = True
        return g
    ref = _walk(counted())
    budget = ref.work["bounce_budget"]
    assert ref.work["exhausted_steps"] == 0
    try:
        cls.bounce_loop = property(lambda self, c=budget - 2: BounceLoop(c, count=self.count_walls))
        smaller = _walk(counted())
        assert smaller.work["exhausted_steps"] == 0
        np.testing.assert_array_equal(smaller.positions, ref.positions)
        cls.bounce_loop = property(lambda self: BounceLoop(1, count=self.count_walls))
        with pytest.warns(RuntimeWarning, match="exhausted the bounce budget"):
            one = _walk(counted())
        assert one.work["exhausted_steps"] > 0
        assert (one.positions != ref.positions).any()
    finally:
        cls.bounce_loop = original


def test_the_work_round_trips_through_the_walk_file(tmp_path):
    g = d.Sphere(R); g.count_walls = True
    w = _walk(g)
    w.save(tmp_path / "w.safetensors")
    back = PersistentWalk.load(tmp_path / "w.safetensors")
    assert back.work == w.work


def _myelinated():
    from dmipy_sim.geometry.myelin import MyelinatedCylinder
    return MyelinatedCylinder(inner_radius=R, outer_radius=1.4 * R, orientation=[0, 0, 1], D_intra=D, D_extra=D,
                              D_myelin=0.1e-9, water_fractions=(1.0, 0.0, 0.0))


def _packed_myelinated():
    from dmipy_sim.geometry.myelin import PackedMyelinatedCylinders
    return PackedMyelinatedCylinders(inner_radii=np.array([R, 0.8 * R]), g_ratios=np.array([0.7, 0.7]),
                                     centers=np.array([[0.0, 0.0], [3.0 * R, 0.0]]), cell_size=6.0 * R,
                                     D_intra=D, D_myelin=0.1e-9, D_extra=D)


@pytest.mark.parametrize("make", [_myelinated, _packed_myelinated], ids=["myelinated", "packed_myelinated"])
def test_the_myelinated_substrates_count_their_work_too(make):
    """The concentric kernels run the same loop and report the same record (#631): counting moves no walker, the
    uncounted walk reports None, the counted one the hits of its concentric walls. 2048 walkers: the packed-myelin
    walk fails to compile on jax 0.10.2 / CUDA at a hundred-odd walkers (#555) while two thousand run."""
    plain = _walk(make(), n=2048)
    g = make(); g.count_walls = True
    counted = _walk(g, n=2048)
    np.testing.assert_array_equal(plain.positions, counted.positions)
    np.testing.assert_array_equal(plain.boundary_local_time, counted.boundary_local_time)
    assert plain.work["n_hits"] is None
    assert counted.work["n_hits"] > 0 and counted.work["exhausted_steps"] == 0
    assert counted.work["n_crossings"] == 0                      # impermeable sheath by default


def test_a_binding_walk_counts_only_the_steps_that_happened():
    """With magnetisation transfer on, a bound walker's sub-step is not taken and adds no work: at the same
    sub-step (binding has its own sub-step rule, so it is pinned to the free walk's) the bound walk's hits are the
    free walk's less the share of steps spent bound, within the noise of two different walks."""
    g = _packed_myelinated(); g.count_walls = True
    free = simulate_trajectories(2048, D, g, 2e-3, 0.5e-3, seed=5, tiers="all", walker_batch_size=2048)
    bound = simulate_trajectories(2048, D, g, 2e-3, 0.5e-3, seed=5, tiers="all", walker_batch_size=2048,
                                  kappa_MT=5e-4, dwell_time=2e-4, equilibrate_binding=False,
                                  sub_steps=free.sub_steps)                                      # 2048: #555
    assert bound.has_binding and bound.bound_frac.mean() > 0 and bound.sub_steps == free.sub_steps
    taken = 1.0 - float(bound.bound_frac[:, 1:].mean())           # the share of sub-steps that happened
    assert 0 < bound.work["n_hits"] < free.work["n_hits"]
    assert bound.work["n_hits"] / free.work["n_hits"] == pytest.approx(taken, rel=0.15)


def test_the_bloch_walk_records_its_work_on_the_run(caplog):
    """The forward vector-Bloch engine has no walk object; with counting on it records the same work on its run
    and in the log, and the signal is the uncounted one to the bit."""
    import logging
    from dmipy_sim import pgse
    from dmipy_sim.engine.bloch import simulate_bloch
    seq = pgse([[1, 0, 0]], 2e-3, 4e-3, bvalues=[1e9], TE=8e-3, n_t=80, slew_rate=np.inf)
    plain = simulate_bloch(64, D, seq, d.Sphere(R), seed=3)
    g = d.Sphere(R); g.count_walls = True
    with caplog.at_level(logging.INFO, logger="dmipy_sim.engine.bloch"):
        counted = simulate_bloch(64, D, seq, g, seed=3)
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(counted))
    rec = [r for r in caplog.records if "bloch walk work" in r.getMessage()]
    assert rec and "'n_hits': " in rec[0].getMessage() and "'exhausted_steps': 0" in rec[0].getMessage()


def test_a_smaller_budget_is_certified_by_its_own_exhaustion_count():
    """`bounce_budget=` overrides the derived budget for one walk and is restored after it; a budget no step
    exhausted gives the derived budget's trajectories bit for bit (#634)."""
    g = d.Sphere(R)
    derived = g.bounce_loop.budget
    ref = _walk(g)
    small = _walk(g, bounce_budget=derived - 2)
    assert g.bounce_budget is None and g.bounce_loop.budget == derived            # restored
    assert small.work["bounce_budget"] == derived - 2 and small.work["exhausted_steps"] == 0
    np.testing.assert_array_equal(small.positions, ref.positions)
    with pytest.warns(RuntimeWarning, match="exhausted the bounce budget"):
        one = _walk(g, bounce_budget=1)
    assert one.work["exhausted_steps"] > 0 and (one.positions != ref.positions).any()
    with pytest.raises(ValueError, match="no bounce loop"):
        _walk(d.FreeDiffusion(), bounce_budget=3)


def test_the_override_reaches_the_packs_kernel_and_the_label_volume():
    for g in (_packed(), _label()):
        derived = g.bounce_loop.budget
        w = _walk(g, bounce_budget=max(1, derived - 1))
        assert w.work["bounce_budget"] == max(1, derived - 1)
        assert g.bounce_loop.budget == derived


def test_the_pack_refuses_an_exhausted_walk(tmp_path):
    from dmipy_sim.replay.bank import build_replay_pack
    g = d.Sphere(R)
    with pytest.warns(RuntimeWarning):
        w = _walk(g, bounce_budget=1)
    with pytest.raises(ValueError, match="exhausted its bounce budget"):
        build_replay_pack(w, id="x", license="CC0-1.0", citation="none", out_path=str(tmp_path / "x.rpk"))


def test_certify_bounce_budget_sizes_from_a_pilot():
    from dmipy_sim import certify_bounce_budget
    g = d.Sphere(R)
    rec = certify_bounce_budget(g, 256, D, 2e-3, 0.5e-3, seed=11, walker_batch_size=256)
    assert rec["derived"] == g.bounce_loop.budget
    assert rec["exhausted"][rec["derived"]] == 0 and rec["smallest_certified"] <= rec["derived"]
    assert rec["recommended"] == min(rec["derived"], rec["smallest_certified"] + 1)
    assert all(rec["exhausted"][b] == 0 for b in rec["exhausted"] if b >= rec["smallest_certified"])
    assert g.bounce_budget is None


def test_walk_spec_passes_the_budget_through_and_keeps_the_work():
    from dmipy_sim.spec import walk_spec
    g = d.Sphere(R)
    w = walk_spec(g.spec, 128, 2e-3, 0.5e-3, seed=11, diffusivity=D, bounce_budget=g.bounce_loop.budget - 1,
                  walker_batch_size=128)
    assert w.work["bounce_budget"] == g.bounce_loop.budget - 1 and w.work["exhausted_steps"] == 0
