"""A geometry's ``orientation`` is a pose of the acquisition, never of the walk.

Every geometry is walked in its own frame (the cylinder kinds along +z) and `simulate` rotates the gradient into
that frame through ``_orient_R``, as the Mesh always did. So a tilted substrate is the upright one under a rotated
acquisition, exactly, and no walker or step is ever rotated -- which is what a 3x3 float32 ``@`` on a CUDA device
cannot do exactly: it runs at TF32 and the per-step frame change the analytic family used to apply leaked a tilted
0.72 um cylinder's walkers (48 % in 5000 steps of 35 nm).
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import simulate, set_b
from dmipy_sim.sequences import pgse

R, D0 = 0.7182e-6, 0.6e-9
AXIS = np.array([0.3, -0.5, 0.81]) / np.linalg.norm([0.3, -0.5, 0.81])


def _seq(grad):
    b = np.array([1.0e9, 3.0e9])
    return set_b(pgse(np.tile(grad, (len(b), 1)), 10e-3, 30e-3, gradient_strengths=1.0, n_t=1500), b)


@pytest.mark.parametrize("make", [
    lambda o: d.Cylinder(R, o),
    lambda o: d.PackedCylinders([R], [[0.0, 0.0]], 4.0e-6, orientation=o),
    lambda o: d.MyelinatedCylinder(inner_radius=R, outer_radius=R / 0.7, orientation=o, D_intra=D0, D_extra=D0,
                                   water_fractions=(1.0, 0.0, 0.0)),
], ids=["cylinder", "packed", "myelinated"])
def test_the_walk_is_in_the_substrate_frame_whatever_the_pose(make):
    g = make(AXIS); z = make((0.0, 0.0, 1.0))
    assert z._orient_R is None and np.allclose(g._orient_R @ [0, 0, 1], AXIS, atol=1e-6)
    kp, kz = jax.random.PRNGKey(0), jax.random.PRNGKey(0)
    np.testing.assert_array_equal(np.asarray(g.init_positions(500, kp)), np.asarray(z.init_positions(500, kz)))
    P = np.asarray(g.init_positions(500, kp))
    assert (np.abs(P[:, 2]) <= 1e-12).all()                                              # seeded in the x-y cross-section


@pytest.mark.parametrize("make, D", [
    (lambda o: d.Cylinder(R, o), D0),
    (lambda o: d.MyelinatedCylinder(inner_radius=R, outer_radius=R / 0.7, orientation=o, D_intra=D0, D_extra=D0,
                                    water_fractions=(1.0, 0.0, 0.0)), None),
], ids=["cylinder", "myelinated"])
def test_a_posed_substrate_is_the_upright_one_under_the_rotated_acquisition(make, D):
    """Same seed, same walk: the posed geometry with a lab gradient g gives what the upright one gives with the
    gradient rotated into the substrate frame, to float32 rounding."""
    g_lab = np.array([0.0, 1.0, 0.0])
    posed = make(AXIS); upright = make((0.0, 0.0, 1.0))
    g_sub = posed._orient_R.T @ g_lab
    kw = dict(n_walkers=4000, seed=3)
    if D is not None:
        kw["diffusivity"] = D
    s_posed = np.asarray(simulate(waveform=_seq(g_lab), geometry=posed, **kw))
    s_upright = np.asarray(simulate(waveform=_seq(g_sub), geometry=upright, **kw))
    np.testing.assert_allclose(s_posed, s_upright, rtol=2e-4, atol=2e-4)
