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


def _dispatched_steps(g, n, rng, key):
    """``n`` wall interactions' worth of starts and steps at the step the dispatcher chooses for ``g``: Gaussian
    steps at the resolved sub-step (their tails included), and a grazing set -- a start ``2 nudge`` inside the
    smallest object's wall with a tangential step of the rule's length ``R / 6``, the lane the cap is derived
    from (a sphere or a cylinder; an ellipsoid has no one radius and gets random starts)."""
    from dmipy_sim.engine.physics import resolve_sub_steps
    import jax
    dt = 1e-4; n_sub = resolve_sub_steps(g, D, dt)
    sigma = float(np.sqrt(2.0 * D * dt / n_sub))
    r0 = np.asarray(g.init_positions(n, key), np.float64)
    step = rng.normal(0.0, sigma, (n, 3))
    R = float(g.length_scales.min_feature)
    m = n // 5
    u = rng.normal(size=(m, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    graze = (R - 2e-4 * R) * u                                          # just inside the smallest sphere / cylinder radius
    if isinstance(g, d.Cylinder):
        graze[:, 2] = 0.0; graze[:, :2] *= (R - 2e-4 * R) / np.linalg.norm(graze[:, :2], axis=1, keepdims=True)
    t = rng.normal(size=(m, 3)); t -= (t * u).sum(1, keepdims=True) * u; t /= np.linalg.norm(t, axis=1, keepdims=True)
    if isinstance(g, d.Ellipsoid):
        graze = graze * 0.0 + r0[:m]                                  # an ellipsoid has no one radius: random starts
    return np.concatenate([r0, graze]).astype(np.float32), np.concatenate([step, (R / 6.0) * t]).astype(np.float32)


@pytest.mark.parametrize("make", [
    lambda: d.Sphere(1e-6), lambda: d.Cylinder(1e-6, (0, 0, 1)),
    lambda: d.Ellipsoid((1e-6, 1.5e-6, 0.8e-6)), lambda: d.Sphere(1e-6, permeability=2e-5),
], ids=["Sphere", "Cylinder", "Ellipsoid", "permeable Sphere"])
def test_no_lane_reaches_the_cap_at_the_dispatched_step(make):
    """Doubling the cap changes nothing: the budget covers every lane at the sub-step the dispatcher chooses.
    One wall interaction per lane -- the cap is a per-step property -- over 20k Gaussian steps and 4k grazing
    ones, under the cap and under twice the cap, to the bit."""
    import jax, jax.numpy as jnp
    g = make()
    cap = g._MAX_BOUNCES
    cls = type(g)
    original = cls.__dict__["_MAX_BOUNCES"]          # the class property; restore it afterwards
    r0, step = _dispatched_steps(g, 20_000, np.random.default_rng(0), jax.random.PRNGKey(0))
    keys = jax.random.split(jax.random.PRNGKey(1), r0.shape[0])
    out = []
    for c in (cap, 2 * cap, 1):
        cls._MAX_BOUNCES = property(lambda self, c=c: c)
        try:
            gg = make()
            if gg.permeability is not None:
                f = jax.jit(jax.vmap(lambda r, s, k: gg.permeate(r, s, jnp.float32(gg.permeability / D), jnp.float32(0.0), k)[0]))
                out.append(np.asarray(f(jnp.asarray(r0), jnp.asarray(step), keys)))
            else:
                out.append(np.asarray(jax.jit(jax.vmap(gg.reflect))(jnp.asarray(r0), jnp.asarray(step))))
        finally:
            cls._MAX_BOUNCES = original
    np.testing.assert_array_equal(out[0], out[1])
    if not isinstance(g, d.Ellipsoid):
        assert (out[2][20_000:] != out[1][20_000:]).any()               # a cap of 1 does cut the grazing lanes: the test has teeth


def _mesh_grazing_lane(m, V, F, rng):
    """The mesh's grazing lane, which ``_dispatched_steps`` cannot build (a mesh has no one radius): a start two
    nudges inside every facet's centroid, aimed along the facet for the dispatched sub-step's length, plus the
    same starts aimed at the facet's nearest edge midpoint. The lane the cap is derived from, on the wall itself."""
    from dmipy_sim.engine.physics import resolve_sub_steps
    dt = 1e-4; n_sub = resolve_sub_steps(m, D, dt)
    step_l = float(np.sqrt(6.0 * D * dt / n_sub))
    tri = V[F]; cen = tri.mean(1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True)
    n *= np.sign((n * cen).sum(1, keepdims=True))                        # outward, for a body around the origin
    start = cen - 2.0 * float(m._NUDGE) * n
    t = rng.normal(size=cen.shape); t -= (t * n).sum(1, keepdims=True) * n; t /= np.linalg.norm(t, axis=1, keepdims=True)
    mid = 0.5 * (tri[:, 0] + tri[:, 1]); to_edge = mid - start; to_edge /= np.linalg.norm(to_edge, axis=1, keepdims=True)
    return (np.concatenate([start, start]).astype(np.float32),
            (step_l * np.concatenate([t, to_edge])).astype(np.float32))


def test_no_mesh_lane_reaches_the_cap_at_the_dispatched_step():
    """Doubling the cap changes nothing on a mesh either: the Gaussian set at the dispatched sub-step, and the
    grazing lane on the facets, under the cap and under twice the cap, to the bit; and a cap of one does cut the
    grazing lane, so the test has teeth."""
    import jax, jax.numpy as jnp
    V, F = mesh_shapes.icosphere(1e-6, subdivisions=2)
    m = d.Mesh(V, F, feature_radius=0.5e-6)
    rng = np.random.default_rng(2)
    r0, step = _dispatched_steps(m, 5_000, rng, jax.random.PRNGKey(2))
    r0, step = r0[:5_000], step[:5_000]                                  # the Gaussian set; the mesh's own graze lane follows
    g0, gs = _mesh_grazing_lane(m, V, F, rng)
    r0, step = np.concatenate([r0, g0]), np.concatenate([step, gs])
    out = [np.asarray(jax.jit(jax.vmap(d.Mesh(V, F, feature_radius=0.5e-6, max_bounces=c).reflect))(jnp.asarray(r0), jnp.asarray(step)))
           for c in (m._MAX_BOUNCES, 2 * m._MAX_BOUNCES, 1)]
    np.testing.assert_array_equal(out[0], out[1])
    assert (out[2][5_000:] != out[1][5_000:]).any()                      # a cap of 1 does cut the grazing lane


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
