"""The curved-tube family's wall interaction and its step rule (dmipy-sim#76, the DiSCo sub-step probe).

An interior walker keeps to its own tube. The rule that mirrored the endpoint into whichever tube's axis was
nearest let a walker in one strand hop into a touching neighbour at a step-dependent rate: on DiSCo's intra pool
the displacement variance grew 2.3 % from R/12 to R/6 steps and 7.6 % at R/2, the contact channel 1.3 % and 5.5 %.
With the exit from the walker's own tube found along the step and the remainder mirrored there, the same walk
moves 0.02 % and 0.1 %.

An exterior walker in a gap narrower than its step bounces twice. With one bounce and the side guard the contact
channel in a 0.1 R slot between two thin strands read 2.6 % low at R/4 steps and 9 % low at R/2; with the second
bounce it is within the channel's 1 % noise at every step.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.geometry.curved_cylinder import STEP_FRACTION

R = 0.7182e-6                                   # DiSCo's thinnest strand
D0 = 0.6e-9


def _walk(g, r0, n_sub, T=2e-3, dt_save=2e-4):
    return d.simulate_trajectories(len(r0), D0, g, T, dt_save, seed=1, r0=r0, sub_steps=n_sub)


def _n_sub(frac, dt_save=2e-4):
    return int(np.ceil(dt_save / ((R / frac) ** 2 / (6 * D0))))


def test_an_interior_walker_never_changes_tube():
    """Two thin strands whose walls are 14 nm apart, a step of R / STEP_FRACTION (240 nm) across it: every walker ends in
    the tube it started in. The nearest-axis mirror moved walkers across."""
    gap = 0.02 * R
    cl1 = np.array([[-30e-6, 0, 0], [30e-6, 0, 0]]); cl2 = np.array([[-30e-6, 2 * R + gap, 0], [30e-6, 2 * R + gap, 0]])
    g = d.PackedCurvedCylinders([cl1, cl2], [R, R], interior=True)
    r0 = np.asarray(g.init_positions(20_000, jax.random.PRNGKey(0)))
    tube0 = np.asarray(jax.vmap(g.classify_position)(jnp.asarray(r0)))
    assert set(np.unique(tube0)) == {1, 2}
    w = _walk(g, r0, _n_sub(STEP_FRACTION))
    pos = np.asarray(w.positions)
    for k in range(pos.shape[1]):
        tube = np.asarray(jax.vmap(g.classify_position)(jnp.asarray(pos[:, k])))
        assert np.array_equal(tube, tube0), (k, (tube != tube0).sum(), (tube == 0).sum())


@pytest.mark.slow
def test_the_hairpin_keeps_its_walkers_at_the_rule():
    """The thinnest strand at DiSCo's sharpest joint (173 deg, a hairpin): at the rule's step no walker leaves it and the
    displacement variance matches a walk at R/12 to the Monte Carlo floor."""
    t = np.deg2rad(173.1); u = np.array([np.cos(t), np.sin(t), 0.0]); L = 15.3e-6
    cl = np.array([[-40e-6, 0, 0], [0, 0, 0], L * u, (L + 40e-6) * u])
    g = d.PackedCurvedCylinders([cl], [R], interior=True)
    r0 = np.asarray(g.init_positions(200_000, jax.random.PRNGKey(3)))
    r0 = r0[np.linalg.norm(r0, axis=1) < 6e-6][:20_000]
    msd = {}
    for frac in (12.0, STEP_FRACTION):
        pos = np.asarray(_walk(g, r0, _n_sub(frac)).positions)
        assert g.inside_any(pos[:, -1]).all(), frac
        msd[frac] = ((pos[:, -1] - pos[:, 0]) ** 2).mean(0)
    rel = np.abs(msd[STEP_FRACTION] / msd[12.0] - 1.0)
    assert np.all(rel < 4.0 * np.sqrt(2.0 / len(r0)) + 0.01), rel      # the floor of two 20k walks, and 1 %


@pytest.mark.slow
def test_the_contact_channel_in_a_narrow_gap_holds_at_the_rule():
    """Two thin strands 0.1 R apart, the exterior pool seeded in the slot: the contact channel at the rule's step is the
    R/12 channel to 3 % (one bounce and the side guard read 2.6 % low at R/4 and 9 % low at R/2)."""
    Ro = R / 0.7; gap = 0.1 * Ro
    cl1 = np.array([[-30e-6, 0, 0], [30e-6, 0, 0]]); cl2 = np.array([[-30e-6, 2 * Ro + gap, 0], [30e-6, 2 * Ro + gap, 0]])
    lo = np.array([-36e-6, -Ro - 3e-6, -Ro - 3e-6]); hi = np.array([36e-6, 3 * Ro + gap + 3e-6, Ro + 3e-6])
    g = d.PackedCurvedCylinders([cl1, cl2], [Ro, Ro], interior=False, box=(lo, hi))
    rng = np.random.default_rng(0); n = 60_000
    P = np.stack([rng.uniform(-5e-6, 5e-6, 30 * n), rng.uniform(0.8 * Ro, 1.2 * Ro + gap, 30 * n), rng.uniform(-0.5 * Ro, 0.5 * Ro, 30 * n)], 1)
    r0 = P[~g.inside_any(P)][:n]
    contact = {}
    for frac in (12.0, STEP_FRACTION):
        w = _walk(g, r0, int(np.ceil(2e-4 / ((Ro / frac) ** 2 / (6 * D0)))))
        assert not g.inside_any(np.asarray(w.positions)[:, -1]).any(), frac
        contact[frac] = float(np.asarray(w.boundary_local_time).sum(1).mean())
    assert abs(contact[STEP_FRACTION] / contact[12.0] - 1.0) < 0.03, contact
