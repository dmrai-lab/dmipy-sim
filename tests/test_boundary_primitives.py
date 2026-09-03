"""Direct unit tests for the shared boundary primitives (`dmipy_sim/_boundary.py`).

These do NOT replace the per-geometry tests. Consolidation removed duplicate
IMPLEMENTATIONS, not duplicate physics: `test_sphere_misst_config1` and
`test_cylinder_misst_config1` check different analytic references, and a shared
`ray_sphere_t` does not make a sphere's restricted signal equal a cylinder's.

What consolidation did change is where a bug can now live. Every defect introduced while
doing it was in the WIRING, not the primitive:

  * PackedSpheres' sentinel offset was not minimum-imaged, while everything around it was;
  * PermeableShell spelled the side choice as a +/-1.0 float, so it matched no search for
    the boolean form the others used;
  * `reflect_with_log_weight` needed the discriminant for cos(alpha) = sqrt(disc)/R, and a
    replacement that discarded it raised NameError -- on the relaxivity path only, because
    plain `reflect` had become textually identical and did not use it.

None of those are visible to a unit test of the primitive, and all of them were caught by
per-geometry tests. So the per-geometry suite stays; this file adds the other half, which
did not exist before because there was no single implementation to point at: it pins each
invariant once, precisely, and makes a failure say "primitive" instead of "some geometry".
"""
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_sim._boundary import (keep_side_radial, keep_side_planar, keep_side_quadric,
                                 ray_sphere_t, ray_quadric_t, specular,
                                 transmit_probability, off_wall, step_off_wall)

R = jnp.float32(1.0e-6)
NUDGE = jnp.float32(1e-10)


# ── keep_side_* : the tie is the whole point ────────────────────────────────
@pytest.mark.parametrize("want_inside", [True, False])
def test_keep_side_radial_resolves_the_on_surface_tie(want_inside):
    """A walker exactly ON the surface belongs to neither side, so it must be moved.

    This is the float32 tie that made two equivalent spellings of "is it inside"
    disagree, and it is why the rule uses >= / <= rather than a != comparison.
    """
    q = jnp.array([float(R), 0.0, 0.0], jnp.float32)      # |q| == R exactly
    out, wrong = keep_side_radial(q, q, R, jnp.bool_(want_inside), NUDGE)
    assert bool(wrong), "landing exactly on the wall must count as the wrong side"
    d = float(jnp.linalg.norm(out))
    assert (d < float(R)) if want_inside else (d > float(R))


@pytest.mark.parametrize("want_inside", [True, False])
def test_keep_side_radial_is_a_noop_when_already_correct(want_inside):
    frac = 0.5 if want_inside else 2.0
    q = jnp.array([float(R) * frac, 0.0, 0.0], jnp.float32)
    out, wrong = keep_side_radial(q, q, R, jnp.bool_(want_inside), NUDGE)
    assert not bool(wrong)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(q))


def test_keep_side_radial_preserves_components_orthogonal_to_q():
    """Correction moves ALONG q, so a cylinder's axial coordinate must survive untouched."""
    q = jnp.array([float(R), 0.0], jnp.float32)                  # in-plane offset
    pos = jnp.array([float(R), 0.0, 7.5e-6], jnp.float32)        # ... plus an axial z
    out, wrong = keep_side_radial(pos, jnp.concatenate([q, jnp.zeros(1, jnp.float32)]),
                                  R, jnp.bool_(True), NUDGE)
    assert bool(wrong)
    assert float(out[2]) == pytest.approx(7.5e-6, rel=1e-6), "axial coordinate moved"


@pytest.mark.parametrize("want_below", [True, False])
def test_keep_side_planar_resolves_the_tie(want_below):
    wall = jnp.float32(2.5e-6)
    out, wrong = keep_side_planar(wall, wall, jnp.bool_(want_below), NUDGE)
    assert bool(wrong)
    assert (float(out) < float(wall)) if want_below else (float(out) > float(wall))


@pytest.mark.parametrize("want_inside", [True, False])
def test_keep_side_quadric_resolves_the_tie(want_inside):
    pos = jnp.array([1.0, 0.0, 0.0], jnp.float32)
    Q = jnp.float32(1.0)                                    # exactly on the quadric
    out, wrong = keep_side_quadric(pos, Q, jnp.bool_(want_inside), jnp.float32(1e-4))
    assert bool(wrong)
    n = float(jnp.linalg.norm(out))
    assert (n < 1.0) if want_inside else (n > 1.0)


# ── ray intersection ────────────────────────────────────────────────────────
def test_ray_sphere_t_roots_are_analytic():
    """Centred ray through a sphere: roots must be exactly -R and +R."""
    q = jnp.zeros(3, jnp.float32)
    d = jnp.array([1.0, 0.0, 0.0], jnp.float32)
    t_in, t_out, disc = ray_sphere_t(q, d, R)
    assert float(t_in) == pytest.approx(-float(R), rel=1e-5)
    assert float(t_out) == pytest.approx(float(R), rel=1e-5)
    assert float(disc) > 0


def test_ray_sphere_t_reports_a_miss_as_negative_disc():
    q = jnp.array([0.0, 5.0 * float(R), 0.0], jnp.float32)   # offset well past the surface
    d = jnp.array([1.0, 0.0, 0.0], jnp.float32)
    _, _, disc = ray_sphere_t(q, d, R)
    assert float(disc) < 0, "a ray that misses must be reported by disc, not by NaN"


def test_ray_sphere_t_batched_matches_scalar():
    """The batched (N-object) call is the same function, not a parallel implementation."""
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.normal(size=(8, 3)).astype(np.float32) * 1e-6)
    d = jnp.asarray((lambda v: v / np.linalg.norm(v))(rng.normal(size=3)).astype(np.float32))
    radii = jnp.asarray(np.full(8, float(R), np.float32))
    bt_in, bt_out, bdisc = ray_sphere_t(q, d, radii)
    for i in range(8):
        s_in, s_out, s_disc = ray_sphere_t(q[i], d, radii[i])
        assert float(bt_in[i]) == pytest.approx(float(s_in), rel=1e-5, abs=1e-12)
        assert float(bt_out[i]) == pytest.approx(float(s_out), rel=1e-5, abs=1e-12)
        assert float(bdisc[i]) == pytest.approx(float(s_disc), rel=1e-5, abs=1e-20)


def test_ray_quadric_t_reduces_to_ray_sphere_t_at_A_equals_1():
    """ray_sphere_t IS the A = 1 case -- if that ever stops holding, they have diverged."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = jnp.asarray((rng.normal(size=3) * 1e-6).astype(np.float32))
        d = jnp.asarray((lambda v: v / np.linalg.norm(v))(rng.normal(size=3)).astype(np.float32))
        s_in, s_out, s_disc = ray_sphere_t(q, d, R)
        B = jnp.dot(q, d)
        C = jnp.dot(q, q) - R * R
        g_in, g_out, g_disc = ray_quadric_t(jnp.float32(1.0), B, C)
        assert float(g_in) == pytest.approx(float(s_in), rel=1e-4, abs=1e-12)
        assert float(g_out) == pytest.approx(float(s_out), rel=1e-4, abs=1e-12)
        assert float(g_disc) == pytest.approx(float(s_disc), rel=1e-4, abs=1e-20)


# ── reflection & transmission ───────────────────────────────────────────────
def test_specular_flips_normal_component_and_keeps_tangential():
    n = jnp.array([0.0, 0.0, 1.0], jnp.float32)
    d = jnp.array([0.6, 0.0, -0.8], jnp.float32)
    out = specular(d, n)
    assert float(out[2]) == pytest.approx(0.8, rel=1e-6)     # normal component reversed
    assert float(out[0]) == pytest.approx(0.6, rel=1e-6)     # tangential preserved
    assert float(jnp.linalg.norm(out)) == pytest.approx(1.0, rel=1e-6)


def test_specular_is_an_involution():
    rng = np.random.default_rng(2)
    n = (lambda v: v / np.linalg.norm(v))(rng.normal(size=3)).astype(np.float32)
    d = (lambda v: v / np.linalg.norm(v))(rng.normal(size=3)).astype(np.float32)
    twice = specular(specular(jnp.asarray(d), jnp.asarray(n)), jnp.asarray(n))
    np.testing.assert_allclose(np.asarray(twice), d, atol=1e-6)


def test_transmit_probability_is_clamped_and_linear():
    kod = jnp.float32(1.0e4)
    assert float(transmit_probability(kod, jnp.float32(1e-8))) == pytest.approx(2e-4, rel=1e-5)
    assert float(transmit_probability(kod, jnp.float32(1.0))) == 1.0     # clamped
    assert float(transmit_probability(jnp.float32(0.0), jnp.float32(1.0))) == 0.0


# ── nudge off the wall ──────────────────────────────────────────────────────
@pytest.mark.parametrize("stay_inside", [True, False])
def test_off_wall_lands_exactly_one_nudge_clear_on_the_right_side(stay_inside):
    r_hit = jnp.array([float(R), 0.0, 0.0], jnp.float32)     # snapped to the surface
    n_out = jnp.array([1.0, 0.0, 0.0], jnp.float32)
    out = off_wall(r_hit, n_out, jnp.bool_(stay_inside), NUDGE)
    d = float(jnp.linalg.norm(out))
    expected = float(R) - float(NUDGE) if stay_inside else float(R) + float(NUDGE)
    assert d == pytest.approx(expected, rel=1e-6)


def test_step_off_wall_preserves_total_path_length():
    """The remaining distance is reduced by the nudge, so path length is conserved."""
    r_hit = jnp.array([float(R), 0.0, 0.0], jnp.float32)
    n_out = jnp.array([1.0, 0.0, 0.0], jnp.float32)
    d_refl = jnp.array([-1.0, 0.0, 0.0], jnp.float32)
    remaining = jnp.float32(3e-8)
    out = step_off_wall(r_hit, n_out, jnp.bool_(True), d_refl, remaining, NUDGE)
    travelled = float(jnp.linalg.norm(out - r_hit))
    assert travelled == pytest.approx(float(remaining), rel=1e-4)
