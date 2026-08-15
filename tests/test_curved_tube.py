"""Curved-fibre geometry: a tube swept along a polyline centreline.

The intra-axonal space of a constant-radius fibre following an arbitrary centreline is exactly the
Minkowski sum of that polyline with a ball -- cylindrical along each segment, a spherical patch at every
joint, smooth throughout. That is what makes it worth having over a chain of straight finite cylinders,
which leaves a kink, a gap or an overlap at every join, precisely where the wall decides the reflection.

The load-bearing test is the straight-centreline limit: with a straight polyline the geometry IS a
cylinder, and the cylinder's restricted signal is known analytically and already validated in this suite.
Anything the sweep gets wrong -- the distance metric, the joint patches, the reflection normal -- shows up
as a departure from it.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

from dmipy_sim import Cylinder, simulate, set_b
from dmipy_sim.curved_tube import CurvedTube, MultiShellCurvedTube
from dmipy_sim.waveforms import pgse

D = 2.0e-9
R = 5.0e-6
SEED = 12345


def _straight_centreline(half_length=30e-6, n=9):
    return np.stack([np.zeros(n), np.zeros(n), np.linspace(-half_length, half_length, n)], axis=1)


def _volume(g):
    v = g.volume
    return float(v() if callable(v) else v)


def test_straight_sweep_is_a_cylinder_geometrically():
    """Volume and confinement in the straight limit, before any diffusion is involved."""
    g = CurvedTube(_straight_centreline(20e-6), radius=R)
    assert _volume(g) == pytest.approx(np.pi * R ** 2 * 40e-6, rel=1e-6)

    r0 = np.asarray(g.init_positions(4000, jax.random.PRNGKey(SEED)))
    radial = np.linalg.norm(r0[:, :2], axis=1)
    assert radial.max() <= R, "a seed outside the tube wall"
    assert radial.max() > 0.95 * R, "seeds must fill the lumen, not hug the axis"
    assert np.abs(r0[:, 2]).max() <= 20e-6 + 1e-12


@pytest.mark.parametrize("b", [1.0e9, 2.5e9])
def test_straight_sweep_matches_the_analytic_cylinder(b):
    """Perpendicular restricted signal against `Cylinder`, the case with a known answer.

    Same walkers, same waveform, same seed -- the only difference is which geometry object reflects them,
    so any disagreement beyond Monte-Carlo noise is the sweep's own.
    """
    wf = set_b(pgse(delta=10e-3, DELTA=30e-3, G_magnitude=0.1, bvecs=[[1, 0, 0]], n_t=600), b)
    n = 20_000

    s_cyl = float(np.atleast_1d(simulate(n, D, wf, Cylinder(radius=R, orientation=(0, 0, 1)), seed=SEED))[0])
    s_swp = float(np.atleast_1d(simulate(n, D, wf, CurvedTube(_straight_centreline(), radius=R), seed=SEED))[0])

    # The two geometries share a seed but diverge at the first reflection, so the difference carries the
    # full Monte-Carlo variance rather than being paired. Checked for a hidden systematic before settling on
    # this bound: over b = 1-5e9 the residual looked like it trended with b (+1.9, +4.0, +6.3e-3), but
    # refining the timestep at fixed b turned +3.1e-3 into +2.5e-3 into -1.5e-3 -- it changes SIGN, and the
    # analytic cylinder is still moving over the same refinement. So the residual is discretisation and
    # noise, not a wall that sits at the wrong radius, and a tighter bound here would only be fitting noise.
    floor = 3.0 / np.sqrt(n)          # Monte-Carlo noise on an ensemble mean of unit-modulus phasors
    assert abs(s_swp - s_cyl) < floor, (
        f"straight sweep {s_swp:.4f} vs analytic cylinder {s_cyl:.4f} at b={b:.1e} "
        f"(tolerance {floor:.4f})")


def test_curvature_changes_the_perpendicular_signal():
    """A bent fibre is not a straight one: curvature tilts the local axis into the gradient direction, so
    some of the restricted direction becomes free. Guards against a 'curved' tube that quietly ignores its
    centreline -- which would pass the straight-limit test above perfectly."""
    wf = set_b(pgse(delta=10e-3, DELTA=30e-3, G_magnitude=0.1, bvecs=[[1, 0, 0]], n_t=600), 2.5e9)
    n = 20_000

    z = np.linspace(-30e-6, 30e-6, 21)
    straight = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    bent = np.stack([15e-6 * (z / 30e-6) ** 2, np.zeros_like(z), z], axis=1)   # parabolic bend in x

    s_straight = float(np.atleast_1d(simulate(n, D, wf, CurvedTube(straight, R), seed=SEED))[0])
    s_bent = float(np.atleast_1d(simulate(n, D, wf, CurvedTube(bent, R), seed=SEED))[0])

    assert s_bent < s_straight, (
        f"bending the fibre into the gradient must add attenuation: bent {s_bent:.4f} "
        f"vs straight {s_straight:.4f}")


def test_walkers_stay_inside_a_bent_tube():
    """Confinement on the shape the geometry exists for. Measured as distance to the centreline polyline,
    which is the wall's definition -- a walker leaking through a joint patch is exactly the artifact a
    chain of finite cylinders would produce."""
    z = np.linspace(-30e-6, 30e-6, 15)
    bent = np.stack([12e-6 * np.sin(z / 30e-6 * np.pi), np.zeros_like(z), z], axis=1)
    g = CurvedTube(bent, radius=R)

    wf = set_b(pgse(delta=10e-3, DELTA=30e-3, G_magnitude=0.1, bvecs=[[1, 0, 0]], n_t=400), 1e9)
    out = simulate(4000, D, wf, g, seed=SEED, return_positions=True)
    pos = np.asarray(out[1] if isinstance(out, tuple) else out)

    P = pos.reshape(-1, 3)
    seg_a, seg_b = bent[:-1], bent[1:]
    ab = seg_b - seg_a
    t = np.clip(((P[:, None, :] - seg_a) * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)
    closest = seg_a + t[..., None] * ab
    dist = np.linalg.norm(P[:, None, :] - closest, axis=-1).min(axis=1)

    assert dist.max() <= R * 1.02, f"walker {dist.max()/R:.3f}R from the centreline; the wall is at 1.0R"


def test_multishell_separates_lumen_from_sheath():
    """The myelinated form: an inner tube inside an outer one, seeded per shell."""
    g = MultiShellCurvedTube(_straight_centreline(20e-6), r_in=3e-6, r_out=5e-6)

    intra = np.asarray(g.init_positions(2000, jax.random.PRNGKey(SEED), shell="intra"))
    r_intra = np.linalg.norm(intra[:, :2], axis=1)
    assert r_intra.max() <= 3e-6, "an intra seed outside the inner wall"

    sheath = np.asarray(g.init_positions(2000, jax.random.PRNGKey(SEED + 1), shell="myelin"))
    r_sheath = np.linalg.norm(sheath[:, :2], axis=1)
    assert r_sheath.min() >= 3e-6 * 0.98, "a sheath seed inside the lumen"
    assert r_sheath.max() <= 5e-6, "a sheath seed outside the outer wall"
