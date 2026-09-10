"""The bounce cap is the worst case a lane can meet, not a tuning constant.

A fixed-length scan costs every lane the same, so its length is set by the most adversarial
walker: a grazing ray nudged off the smallest object keeps its angle and makes chords of
`2 sqrt(2 nudge R)`, so a step of `R/6` needs `ceil(step/chord) + 1` reflections. Iterating a
finished lane is a no-op, so a walk under the cap and under twice the cap must agree to the bit;
if they ever differ, some lane hit the cap and it is too low.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.geometry import mesh_shapes
from dmipy_sim.geometry._boundary import bounce_budget

D = 2e-9


def _wf(n_t=120):
    return d.set_b(d.pgse([[1, 0, 0], [0, 0, 1]], 4e-3, 10e-3, gradient_strengths=0.1, n_t=n_t, slew_rate=np.inf), 1e9)


def test_analytic_caps_are_the_grazing_worst_case():
    for g in (d.Sphere(1e-6), d.Sphere(7e-6), d.Cylinder(2e-6, (0, 0, 1)), d.Ellipsoid((3e-6, 1e-6, 2e-6))):
        R = g.length_scales.min_feature
        chord = 2 * np.sqrt(2 * 1e-4 * R * R)
        assert g._MAX_BOUNCES == int(np.ceil((R / 6) / chord) + 1) == bounce_budget(R, 1e-4 * R, np.inf, R / 6)
    assert d.Sphere(1e-6)._MAX_BOUNCES == 7                      # scale-free at nudge = 1e-4 R
    assert bounce_budget(1e-6, 1e-10, 2e-8, 1.7e-7) == int(np.ceil(1.7e-7 / 2e-8) + 1)   # a gap rules


def test_mesh_cap_has_a_floor_and_grows_with_the_cell():
    V, F = mesh_shapes.icosphere(1e-6, subdivisions=2)
    m = d.Mesh(V, F, feature_radius=0.5e-6)
    assert m._MAX_BOUNCES >= 10
    assert d.Mesh(V, F, feature_radius=0.5e-6, max_bounces=4)._MAX_BOUNCES == 4
    wide = d.Mesh(V, F, feature_radius=0.5e-6, cell_size=3e-6)
    assert wide._MAX_BOUNCES == max(10, int(np.ceil(0.9 * wide.cell_size / (2 * 0.06 * 0.5e-6))) + 2) > 10


@pytest.mark.parametrize("make", [
    lambda: d.Sphere(1e-6), lambda: d.Cylinder(1e-6, (0, 0, 1)),
    lambda: d.Ellipsoid((1e-6, 1.5e-6, 0.8e-6)), lambda: d.Sphere(1e-6, permeability=2e-5),
], ids=["Sphere", "Cylinder", "Ellipsoid", "permeable Sphere"])
def test_no_lane_reaches_the_cap_at_the_dispatched_step(make):
    """Doubling the cap changes nothing: the budget covers every lane at the sub-step the
    dispatcher chooses."""
    g = make()
    cap = g._MAX_BOUNCES
    cls = type(g)
    original = cls.__dict__["_MAX_BOUNCES"]          # the class property; restore it afterwards
    sig = []
    for c in (cap, 2 * cap):
        cls._MAX_BOUNCES = property(lambda self, c=c: c)
        try:
            sig.append(np.asarray(d.simulate(3000, D, _wf(), make(), seed=0, require_gpu=False)))
        finally:
            cls._MAX_BOUNCES = original
    np.testing.assert_array_equal(sig[0], sig[1])


def test_no_mesh_lane_reaches_the_cap_at_the_dispatched_step():
    V, F = mesh_shapes.icosphere(1e-6, subdivisions=2)
    m = d.Mesh(V, F, feature_radius=0.5e-6)
    out = [np.asarray(d.simulate(2000, D, _wf(), d.Mesh(V, F, feature_radius=0.5e-6, max_bounces=c),
                                 seed=1, require_gpu=False)) for c in (m._MAX_BOUNCES, 2 * m._MAX_BOUNCES)]
    np.testing.assert_array_equal(out[0], out[1])


def test_the_cap_still_bounds_a_pathological_lane():
    """A cap of one reflection is the single-hit rule again: the leftover path is not flown."""
    from dmipy_sim.geometry._boundary import bounce_loop
    import jax.numpy as jnp

    def hit_once(r, dh, rem, decided):          # a ray between two parallel walls at x = 0 and x = 1
        to_wall = jnp.where(dh[0] > 0, 1.0 - r[0], r[0]) / jnp.maximum(jnp.abs(dh[0]), 1e-30)
        hit = to_wall < rem
        r_hit = r + jnp.minimum(to_wall, rem) * dh
        d_new = jnp.where(hit, dh * jnp.array([-1.0, 1.0, 1.0]), dh)
        rem_new = jnp.where(hit, rem - to_wall, 0.0)
        r_new = jnp.where(hit, r_hit, r + rem * dh)
        return r_new, d_new, rem_new, decided, jnp.float32(0.0), jnp.zeros((), bool)
    r0 = jnp.array([0.5, 0.0, 0.0], jnp.float32)
    dh = jnp.array([1.0, 0.0, 0.0], jnp.float32)
    r_many, _, _ = bounce_loop(hit_once, r0, dh, jnp.float32(10.3), 32)   # 10 reflections, 0.2 left over
    r_one, _, _ = bounce_loop(hit_once, r0, dh, jnp.float32(10.3), 1)
    assert 0.0 <= float(r_many[0]) <= 1.0
    assert float(r_one[0]) == pytest.approx(1.0)
