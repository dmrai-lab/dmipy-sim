"""The boundary local time counts the nudge a reflecting wall sets the walker off by (dmipy-sim-cuda#100).

After a reflection the walker is a ``nudge`` off the wall and the rest of the step is shortened by it, so it ends
``nudge - cos min(nudge, rem)`` further from the wall than the ideal specular reflection leaves it. That displacement is
the wall's push as much as ``2 d_perp`` is, and the local time is the sum of the pushes.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.geometry._boundary import nudge_push

F32 = jnp.float32


def test_nudge_push_rule():
    n = F32(1e-10)
    assert float(nudge_push(n, F32(1.0), F32(1e-7))) == pytest.approx(0.0, abs=1e-18)       # a head-on wall: nothing extra
    assert float(nudge_push(n, F32(0.0), F32(1e-7))) == pytest.approx(1e-10, rel=1e-6)      # grazing: all of it
    assert float(nudge_push(n, F32(0.5), F32(1e-7))) == pytest.approx(0.5e-10, rel=1e-6)
    assert float(nudge_push(n, F32(0.5), F32(0.4e-10))) == pytest.approx(0.8e-10, rel=1e-6)  # a step spent inside the nudge
    assert float(nudge_push(n, F32(0.7), F32(0.0))) == pytest.approx(1e-10, rel=1e-6)        # a rule that does not shorten the step


def _ray_reflect(r, dirn, rem, R, nudge, cylinder):
    """float64 specular reflection of a ray off |x| = R from inside, the walker set ``nudge`` in and the rest shortened."""
    r = np.asarray(r, float); dirn = np.asarray(dirn, float)
    if cylinder:
        r, dirn = r[:2], dirn[:2]
    b = r @ dirn; c = r @ r - R * R
    t = -b + np.sqrt(b * b - c)
    hit = r + t * dirn
    nrm = hit / np.linalg.norm(hit)
    d_perp = (rem - t) * abs(dirn @ nrm)
    dr = dirn - 2.0 * (dirn @ nrm) * nrm
    start = hit - nudge * nrm
    end = start + max(rem - t - nudge, 0.0) * dr
    return end, d_perp, abs(dirn @ nrm), rem - t


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
@pytest.mark.parametrize("angle", [0.2, 0.5, 0.9])
def test_local_time_is_the_walkers_push(kind, angle):
    """``-dlog_w / rho_over_D`` is the displacement of the reflected walker along the normal: the ideal reflection's
    ``2 d_perp`` plus what the nudge adds, so that (distance in from the wall with the nudge) - (without it) equals the
    push the rule counts."""
    R = 1e-6; nudge = 1e-4 * R
    geom = d.Sphere(R) if kind == "sphere" else d.Cylinder(R, (0.0, 0.0, 1.0))
    cyl = kind == "cylinder"
    r0 = np.array([0.995 * R, 0.0, 0.0])
    dirn = np.array([np.cos(angle), np.sin(angle), 0.0])
    step_l = 0.012 * R                                                           # a step short against the radius: the wall is locally flat
    r = jnp.asarray(r0, F32); step = jnp.asarray(dirn * step_l, F32)
    hit = geom._wall(r, step, F32(0.0), F32(1.0), jax.random.PRNGKey(0))
    end_ideal, d_perp, cos_a, rem_a = _ray_reflect(r0, dirn, step_l, R, 0.0, cyl)
    end_nudged, _, _, _ = _ray_reflect(r0, dirn, step_l, R, nudge, cyl)
    assert rem_a > nudge
    extra = np.linalg.norm(end_ideal) - np.linalg.norm(end_nudged)                  # further in with the nudge
    assert extra == pytest.approx(nudge * (1.0 - cos_a), rel=2e-2)                   # the rule's derivation
    got = -float(hit.dlog_w)
    assert got - 2.0 * d_perp == pytest.approx(extra, rel=5e-2)                    # the push beyond 2 d_perp is the nudge's
    # the walk itself: the walker the wall returns is where the nudged reference puts it
    r_new = np.asarray(hit.r, float)
    assert np.linalg.norm(r_new[:2] if cyl else r_new) == pytest.approx(np.linalg.norm(end_nudged), abs=2e-13)
