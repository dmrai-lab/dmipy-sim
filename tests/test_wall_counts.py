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
    assert plain.work["n_hits"] is None and plain.work["exhausted_steps"] is None
    assert plain.work["walker_steps"] == counted.work["walker_steps"] == 256 * 4 * plain.sub_steps
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
