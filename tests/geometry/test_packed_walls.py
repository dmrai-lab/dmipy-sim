"""The packed geometries walk the same wall physics as every other geometry, at any packing.

`geometry.packed.packed_wall_kernel` is the one wall interaction behind `PackedCylinders` and
`PackedSpheres`. The fast tests are the adversarial impacts a pack adds to the single-object
table: a step spanning the gap between two objects, at every angle, up to ten gap widths. The
consequences on signals are the next two tests: a dense pack keeps every extra-axonal walker outside where the
single-hit rule let nine in ten through, and a pack of one permeable cylinder is `Cylinder(permeability=)`.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.geometry._boundary import bounce_budget
from dmipy_sim.geometry.packed import packed_candidate_count
from dmipy_sim.math.step_law import LONGEST_STEP
from dmipy_sim.engine._fused import simulate_fused
from dmipy_sim.engine.core import _walk_geometry
from tests.conftest import z_family

D = 2e-9
R = 1e-6
ANGLES = [0.0, 20.0, 45.0, 70.0, 89.0, 90.0, 135.0, 180.0]
SPANS = [0.5, 1.0, 3.0, 10.0]


def _two(kind, gap):
    """Two objects of radius R, `gap` apart along x, in a cell large enough to be irrelevant."""
    if kind == "cylinders":
        c = np.array([[0.0, 0.0], [2 * R + gap, 0.0]])
        return d.PackedCylinders([R, R], c, 20 * R), c
    c = np.array([[0.0, 0.0, 0.0], [2 * R + gap, 0.0, 0.0]])
    return d.PackedSpheres([R, R], c, 20 * R), c


def _gap_cases(gap, dim):
    mid = np.zeros(dim); mid[0] = R + gap / 2; mid[1] = 1e-3 * R
    out = []
    for f in SPANS:
        for deg in ANGLES:
            th = np.deg2rad(deg)
            step = np.zeros(dim); step[0], step[1] = np.cos(th), np.sin(th)
            out.append((f"{f}gap/{deg}deg", mid.copy(), step * f * gap))
    return out


@pytest.mark.parametrize("kind", ["cylinders", "spheres"])
def test_a_step_spanning_the_gap_between_two_objects_stays_in_the_gap(kind):
    """A walker in a gap of 0.05 R, steps up to ten gap widths at every angle: it must end
    outside both objects with its path length conserved. A single-hit rule ends it inside the
    neighbour; the kernel's budget is sized for exactly this zig-zag."""
    gap = 0.05 * R
    geom, centers = _two(kind, gap)
    dim = centers.shape[1]
    assert geom.bounce_loop.budget >= 2 + int(np.ceil((LONGEST_STEP * R / 6) / gap)), "budget must cover the longest step across the gap"
    cases = _gap_cases(gap, dim)
    starts = np.stack([np.r_[c[1], np.zeros(3 - dim)] for c in cases]).astype(np.float32)
    steps = np.stack([np.r_[c[2], np.zeros(3 - dim)] for c in cases]).astype(np.float32)
    out = np.asarray(jax.jit(jax.vmap(lambda p, s: geom.interact(p, s).r))(jnp.asarray(starts), jnp.asarray(steps)))
    dist = [np.linalg.norm(out[:, :dim] - c, axis=1) for c in centers]
    entered = (dist[0] <= R * (1 - 1e-6)) | (dist[1] <= R * (1 - 1e-6))
    rows = "\n".join(f"      {cases[i][0]:16} -> d0={dist[0][i] / R:.4f} R  d1={dist[1][i] / R:.4f} R"
                     for i in np.flatnonzero(entered)[:12])
    assert not entered.any(), f"{kind}: {entered.sum()}/{len(cases)} gap impacts ended inside an object:\n{rows}"
    moved = np.linalg.norm(out - starts, axis=1)
    asked = np.linalg.norm(steps, axis=1)
    assert not (moved > asked * (1 + 1e-4) + 1e-12).any(), f"{kind}: a reflection added path length"


def test_bounce_budget_and_candidate_count_follow_the_worst_case():
    nudge = 1e-4 * R
    chord = 2 * np.sqrt(2 * nudge * R)
    assert bounce_budget(R, nudge, np.inf, R / 6) == int(np.ceil((R / 6) / chord) + 1)
    assert bounce_budget(R, nudge, 0.01 * R, R / 6) == int(np.ceil((R / 6) / (0.01 * R)) + 1)
    assert bounce_budget(R, nudge, 1e-9, R / 6) == 32                     # capped
    assert bounce_budget(R, nudge, np.inf, 1e-9) == 2                      # never below two
    assert packed_candidate_count(100, R, R / 6, 2) == 8 and packed_candidate_count(4, R, R / 6, 2) == 4
    assert packed_candidate_count(100, R, 3 * R, 2) == int(np.ceil(np.pi * 4) + 2)
    assert 8 <= packed_candidate_count(100, R, R / 6, 3) <= 100
    # the geometry builds its kernel at the longest step the sub-step rule allows (the pair's long atom at R_min / 6)
    geom, _ = _two("cylinders", 0.02 * R)
    assert geom.bounce_loop.budget == bounce_budget(R, nudge, geom.min_gap, LONGEST_STEP * R / 6)


def _dense_pack(seed=1):
    radii = np.array([1.0, 1.3, 0.8, 1.1, 0.9, 1.2]) * 1e-6
    c, L, _ = d.pack_cylinders(radii, target_vf=0.45, seed=seed)
    return radii / 0.999, c, L                     # walls almost touching: min gap ~ 2 nm


N_DENSE = 2_000             # walkers whose zero leaks bound the leak rate below 0.23 % at 99 % confidence
BLOCK_DENSE = 5e-5          # s: one block of the dense pack's walk, 115 sub-steps of 72 nm


def _final_object(geom, pos):
    """The object (cylinder) id of each position: 0 outside every cylinder."""
    return np.asarray(jax.vmap(geom.classify_position)(jnp.asarray(pos, jnp.float32)))


def test_dense_pack_keeps_every_extra_axonal_walker_outside():
    """At outer packing 0.45 the single-hit rule ended 18033 of 20000 extra-axonal walkers inside
    a cylinder over one PGSE (they reflect off one wall into the neighbour, and the neighbour's
    sentinel then keeps them). None may end inside.

    Claim: an extra-axonal walker of the dense pack ends inside a cylinder with probability below 0.23 %: of N = 2,000
    walkers none inside miss a leak of that rate with probability ``(1 - 0.0023)^2000`` = 1 %. A leak is one step across a
    gap narrower than the step, not an accumulation, so the walk is one 50 us block (115 sub-steps of 72 nm; a walker
    travels 0.45 um of the 0.7 um between the cylinders) instead of the 20 ms PGSE (230 sub-steps per 100 us)."""
    radii, c, L = _dense_pack()
    pc = d.PackedCylinders(radii, c, L, pool="extra")
    r0 = np.asarray(pc.init_positions(N_DENSE, jax.random.PRNGKey(4)))
    assert (_final_object(pc, r0) == 0).all(), "a walker was seeded inside a cylinder"
    out = _walk_geometry(N_DENSE, D, pc, BLOCK_DENSE, BLOCK_DENSE, seed=4, r0=r0, raw_subset=N_DENSE, require_gpu=False)
    final = _final_object(pc, np.asarray(out.raw_positions)[:, -1])
    assert (final == 0).all(), f"{(final > 0).sum()} walkers ended inside a cylinder"


def test_a_pack_of_one_permeable_cylinder_is_the_cylinder():
    """Intra-seeded walkers leave through the membrane at the same rate whether the wall is the
    pack kernel's (a trial at every encounter) or `Cylinder.permeate`'s (one trial per step):
    on a single convex object at the R/25 step a second encounter is a grazing rarity.

    Claim: the fraction still inside after TE is the same to 0.03 (a 10 % difference of the exchanged third). The two
    walks start from the same walkers with the same seed, so their outcomes agree walker by walker except where a
    trial diverges: 2.0 % of the pairs differed at 6,000 walkers (the Cylinder and the pack left 0.6745 and 0.6732
    inside). The difference of the two fractions is then ``(b - c) / N`` for ``b`` and ``c`` the walkers that stayed in
    one walk and left in the other, a standard deviation ``sqrt((b + c)) / N`` -- a quarter of the independent walks'.
    The tolerance is ``z sqrt(b + c)`` and N = 1,500 resolves 0.03 for up to 5 % discordant pairs
    (``(2 z)^2 q / delta^2``)."""
    kappa, N = 2e-5, 1_500
    cyl = d.Cylinder(R, (0, 0, 1), permeability=kappa)
    pk = d.PackedCylinders([R], np.zeros((1, 2)), 40 * R, permeability=kappa)
    wf = d.set_b(d.pgse([[0, 0, 1]], 2e-3, 8e-3, gradient_strengths=0.01, n_t=100, slew_rate=np.inf), 1e6)
    r0 = cyl.init_positions(N, jax.random.PRNGKey(0))                # inside the lumen
    _, o_c, f_c = simulate_fused(N, D, wf, cyl, seed=6, r0=r0, return_compartments="final", require_gpu=False)
    _, o_p, f_p = simulate_fused(N, D, wf, pk, seed=6, r0=r0, return_compartments="final", require_gpu=False)
    assert (o_c == 1).all() and (o_p == 1).all()
    in_c, in_p = f_c == 1, f_p == 1
    assert in_c.mean() < 0.95, "no exchange happened"
    b, c = int((in_c & ~in_p).sum()), int((~in_c & in_p).sum())
    assert abs(b - c) <= z_family() * np.sqrt(b + c), (
        f"still inside after TE: Cylinder {in_c.mean():.4f} vs one-cylinder pack {in_p.mean():.4f} "
        f"({b} stayed only in the Cylinder, {c} only in the pack)")
    assert (b + c) / N < 0.05, "more discordant pairs than the walker count was sized for"
