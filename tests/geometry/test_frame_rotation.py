"""A rotated analytic geometry confines what its axis-aligned twin confines.

The frame change of a tilted cylinder, pack or myelinated axon is a 3x3 rotation of every position and step. On a
CUDA device a float32 ``@`` runs at TF32 precision (10 mantissa bits), so a walker a micron along the axis lands
nanometres off in its radial coordinate on every step -- past the nudge that keeps it on its side of the wall --
and the tilted geometry leaks: 48 % of a 0.72 um cylinder's walkers in 5000 steps of 35 nm, none for the same
cylinder along z. `rotate` in `_boundary` applies the frame change as exact products, and these walks pin it. They
are GPU tests: on the CPU the matmul is exact and the leak never was.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import simulate, set_b
from dmipy_sim.sequences import pgse

pytestmark = pytest.mark.gpu
if jax.devices()[0].platform != "gpu":
    pytest.skip("the frame-rotation leak is a TF32 (CUDA) effect; the CPU matmul is exact", allow_module_level=True)

R = 0.7182e-6                                   # DiSCo's thinnest strand
D0 = 0.6e-9
N, STEPS = 20_000, 5000
SIG = float(np.sqrt(2.0 * D0 * 1e-6))           # 35 nm per axis: 5000 of them wander 2.5 um along the axis


def _disc(n, r, rng):
    ph = rng.uniform(0, 2 * np.pi, n); rad = r * np.sqrt(rng.uniform(0, 1, n)) * 0.999
    return rad * np.cos(ph), rad * np.sin(ph)


def _walk(step_fn, r0):
    def body(r, k):
        return jax.vmap(step_fn)(r, jax.random.normal(k, (r.shape[0], 3), jnp.float32) * SIG), None
    return np.asarray(jax.lax.scan(body, jnp.asarray(r0, jnp.float32), jax.random.split(jax.random.PRNGKey(1), STEPS))[0])


def test_a_tilted_cylinder_keeps_every_walker():
    rng = np.random.default_rng(0); y, z = _disc(N, R, rng)
    g = d.Cylinder(R, (1.0, 0.0, 0.0))
    r0 = np.stack([np.zeros(N), y, z], 1)
    for name, fn in (("reflect", g.reflect), ("reflect_with_log_weight", lambda r, s: g.reflect_with_log_weight(r, s, jnp.float32(0.0))[0])):
        r = _walk(fn, r0)
        outside = np.hypot(r[:, 1], r[:, 2]) >= R
        assert outside.sum() == 0, (name, outside.mean())


def test_a_tilted_pack_keeps_its_interior_walkers():
    rng = np.random.default_rng(1); y, z = _disc(N, R, rng)
    g = d.PackedCylinders([R], [[0.0, 0.0]], 4.0e-6, orientation=(1.0, 0.0, 0.0))
    r = _walk(g.reflect, np.stack([np.zeros(N), y, z], 1))
    L = 4.0e-6
    yz = r[:, 1:] - L * np.floor(r[:, 1:] / L + 0.5)                 # back into the periodic cell
    outside = np.hypot(yz[:, 0], yz[:, 1]) >= R
    assert outside.sum() == 0, outside.mean()


def test_a_tilted_myelinated_axon_signals_as_the_upright_one():
    """The fused myelinated step rotates the frame in `physics`: an axon along x read across it must give what the
    axon along z gives across it, to the Monte Carlo floor of two independent runs."""
    n = 40_000; bvals = np.array([1.0e9, 3.0e9])
    def axon(axis):
        return d.MyelinatedCylinder(inner_radius=R, outer_radius=R / 0.7, orientation=axis, D_intra=D0, D_extra=D0,
                                    water_fractions=(1.0, 0.0, 0.0))
    def across(axis, grad):
        wf = set_b(pgse(np.tile(grad, (len(bvals), 1)), 10e-3, 30e-3, gradient_strengths=1.0, n_t=2000), bvals)
        return np.asarray(simulate(n_walkers=n, waveform=wf, geometry=axon(axis), seed=7))
    s_z = across((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)); s_x = across((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert np.all(np.abs(s_x - s_z) < 4.0 * np.sqrt(2.0 / n) + 0.005), (s_x, s_z)
