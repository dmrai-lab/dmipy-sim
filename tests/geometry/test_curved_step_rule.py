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


def _two_tubes(gap):
    cl1 = np.array([[-30e-6, 0, 0], [30e-6, 0, 0]]); cl2 = np.array([[-30e-6, 2 * R + gap, 0], [30e-6, 2 * R + gap, 0]])
    return d.PackedCurvedCylinders([cl1, cl2], [R, R], interior=True)


def _impacts(g, gap):
    """(label, start, step): an interior walker of tube 1 placed ``off`` inside its wall facing tube 2, aimed at
    that wall at ``deg`` from its normal, for a step of ``dist``; and the mirror image starting in tube 2."""
    offsets = [("nudge", g.nudge_m), ("1e-3R", 1e-3 * R), ("0.1R", 0.1 * R)]
    dists = [("half_gap", 0.5 * gap), ("gap", gap), ("3gap", 3 * gap), ("10gap", 10 * gap), ("R/3", R / 3)]
    angles = [("head_on", 0.0), ("oblique", 45.0), ("grazing", 89.0), ("tangent", 90.0)]
    out = []
    for x in (-20e-6, 0.0, 20e-6):
        for oname, off in offsets:
            for dname, dist in dists:
                for aname, deg in angles:
                    th = np.deg2rad(deg)
                    for tube, y_wall, sign in ((1, R, 1.0), (2, R + gap, -1.0)):      # the facing walls
                        start = np.array([x, y_wall - sign * off, 0.0])
                        aim = np.array([np.sin(th), sign * np.cos(th), 0.0])         # towards the other tube
                        out.append((f"x{x * 1e6:+.0f}/{oname}/{dname}/{aname}/tube{tube}", tube, start, aim * dist))
    return out


def test_an_interior_walker_never_changes_tube():
    """Two thin strands whose walls are 14 nm apart: a walker fired from just inside one wall at the other, at every
    offset, distance and angle of the table, ends in the tube it started in and never further than it stepped.
    The nearest-axis mirror moved such walkers across."""
    gap = 0.02 * R
    g = _two_tubes(gap)
    cases = _impacts(g, gap)
    starts = jnp.asarray(np.stack([c[2] for c in cases]), jnp.float32)
    steps = jnp.asarray(np.stack([c[3] for c in cases]), jnp.float32)
    tube0 = np.array([c[1] for c in cases])
    assert np.array_equal(np.asarray(jax.vmap(g.classify_position)(starts)), tube0), "a start is not in its tube"
    r = np.asarray(jax.jit(jax.vmap(g.reflect))(starts, steps))
    tube = np.asarray(jax.vmap(g.classify_position)(jnp.asarray(r)))
    moved = tube != tube0
    if moved.any():
        rows = "\n".join(f"      {cases[i][0]:40} -> tube {tube[i]}" for i in np.flatnonzero(moved)[:12])
        pytest.fail(f"{moved.sum()}/{len(cases)} impacts changed tube:\n{rows}")
    gone = np.linalg.norm(r - np.asarray(starts), axis=1); asked = np.linalg.norm(np.asarray(steps), axis=1)
    assert not (gone > asked * (1 + 1e-4) + 1e-12).any(), "a reflection added distance"


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
