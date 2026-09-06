"""The packed geometries walk the same wall physics as every other geometry, at any packing.

`geometry.packed.packed_wall_kernel` is the one wall interaction behind `PackedCylinders` and
`PackedSpheres`. The fast tests are the adversarial impacts a pack adds to the single-object
table: a step spanning the gap between two objects, at every angle, up to ten gap widths. The
slow tests are the consequences: a dense pack keeps every extra-axonal walker outside where the
single-hit rule let nine in ten through, membrane transmission does not depend on how many walls
a sub-step meets, and a pack of one permeable cylinder is `Cylinder(permeability=)`.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.geometry.packed import packed_bounce_budget, packed_candidate_count

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
    assert geom._wall.max_bounces >= 2 + int(np.ceil((R / 6) / gap)), "budget must cover R/6 across the gap"
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
    assert packed_bounce_budget(R, nudge, np.inf, R / 6) == int(np.ceil((R / 6) / chord) + 1)
    assert packed_bounce_budget(R, nudge, 0.01 * R, R / 6) == int(np.ceil((R / 6) / (0.01 * R)) + 1)
    assert packed_bounce_budget(R, nudge, 1e-9, R / 6) == 32                     # capped
    assert packed_bounce_budget(R, nudge, np.inf, 1e-9) == 2                      # never below two
    assert packed_candidate_count(100, R, R / 6, 2) == 8 and packed_candidate_count(4, R, R / 6, 2) == 4
    assert packed_candidate_count(100, R, 3 * R, 2) == int(np.ceil(np.pi * 4) + 2)
    assert 8 <= packed_candidate_count(100, R, R / 6, 3) <= 100
    # the geometry builds its kernel at the worst-case step the sub-step rule allows (R_min / 6)
    geom, _ = _two("cylinders", 0.02 * R)
    assert geom._wall.max_bounces == packed_bounce_budget(R, nudge, geom.min_gap, R / 6)


def _dense_pack(seed=1):
    radii = np.array([1.0, 1.3, 0.8, 1.1, 0.9, 1.2]) * 1e-6
    c, L, _ = d.pack_cylinders(radii, target_vf=0.45, seed=seed)
    return radii / 0.999, c, L                     # walls almost touching: min gap ~ 2 nm


@pytest.mark.slow
def test_dense_pack_keeps_every_extra_axonal_walker_outside():
    """At outer packing 0.45 the single-hit rule ended 18033 of 20000 extra-axonal walkers inside
    a cylinder over one PGSE (they reflect off one wall into the neighbour, and the neighbour's
    sentinel then keeps them). None may end inside."""
    radii, c, L = _dense_pack()
    pc = d.PackedCylinders(radii, c, L)
    wf = d.set_b(d.pgse(delta=5e-3, DELTA=15e-3, G_magnitude=0.1, bvecs=[[1, 0, 0]], n_t=200,
                        slew_rate=np.inf), 1.5e9)
    _, origin, final = d.simulate(20_000, D, wf, pc, seed=4, return_compartments="final", require_gpu=False)
    assert (origin == 0).all()
    assert (final == 0).all(), f"{(final > 0).sum()} walkers ended inside a cylinder"


@pytest.mark.slow
def test_membrane_transmission_does_not_depend_on_the_sub_step():
    """The adversarial permeable case: a pack whose gap is a fraction of the R/25 step. The
    fraction of extra-axonal walkers that have entered a cylinder after T must be the same at
    the dispatched sub-step and at a sub-step fine enough to meet one wall at a time, because
    every encounter is its own Powles trial."""
    # two cylinders (and their periodic images along x) a known 20 nm apart: half the R/25 step
    gap = 20e-9
    c = np.array([[0.0, 0.0], [2 * R + gap, 0.0]])
    pk = d.PackedCylinders([R, R], c, 2 * (2 * R + gap), permeability=2e-5)
    assert abs(pk.min_gap - gap) < 1e-12
    wf = d.set_b(d.pgse(delta=2e-3, DELTA=8e-3, G_magnitude=0.01, bvecs=[[1, 0, 0]], n_t=100,
                        slew_rate=np.inf), 1e6)
    N = 20_000
    from dmipy_sim.engine.physics import resolve_sub_steps
    n_auto = resolve_sub_steps(pk, D, wf.dt)
    step_auto = np.sqrt(6 * D * wf.dt / n_auto)
    assert step_auto > 1.5 * gap, "the dispatched step must span the gap for this to be adversarial"
    n_fine = int(np.ceil(n_auto * (step_auto / (0.5 * gap)) ** 2))
    _, _, f_auto = d.simulate(N, D, wf, pk, seed=5, return_compartments="final", require_gpu=False)
    _, _, f_fine = d.simulate(N, D, wf, pk, seed=5, return_compartments="final", sub_steps=n_fine,
                              require_gpu=False)
    p_auto, p_fine = (f_auto > 0).mean(), (f_fine > 0).mean()
    assert p_auto > 0.05, "no exchange happened"
    tol = 4.0 * np.sqrt(2 * p_fine * (1 - p_fine) / N)
    assert abs(p_auto - p_fine) < tol, f"entered fraction {p_auto:.4f} at {n_auto} sub-steps vs {p_fine:.4f} at {n_fine}"


@pytest.mark.slow
def test_a_pack_of_one_permeable_cylinder_is_the_cylinder():
    """Intra-seeded walkers leave through the membrane at the same rate whether the wall is the
    pack kernel's (a trial at every encounter) or `Cylinder.permeate`'s (one trial per step):
    on a single convex object at the R/25 step a second encounter is a grazing rarity."""
    kappa, N = 2e-5, 30_000
    cyl = d.Cylinder(R, (0, 0, 1), permeability=kappa)
    pk = d.PackedCylinders([R], np.zeros((1, 2)), 40 * R, permeability=kappa)
    wf = d.set_b(d.pgse(delta=2e-3, DELTA=8e-3, G_magnitude=0.01, bvecs=[[0, 0, 1]], n_t=100,
                        slew_rate=np.inf), 1e6)
    r0 = cyl.init_positions(N, jax.random.PRNGKey(0))                # inside the lumen
    _, o_c, f_c = d.simulate(N, D, wf, cyl, seed=6, r0=r0, return_compartments="final", require_gpu=False)
    _, o_p, f_p = d.simulate(N, D, wf, pk, seed=6, r0=r0, return_compartments="final", require_gpu=False)
    assert (o_c == 1).all() and (o_p == 1).all()
    p_c, p_p = (f_c == 1).mean(), (f_p == 1).mean()
    assert p_c < 0.95, "no exchange happened"
    tol = 4.0 * np.sqrt(2 * p_c * (1 - p_c) / N)
    assert abs(p_c - p_p) < tol, f"still inside after TE: Cylinder {p_c:.4f} vs one-cylinder pack {p_p:.4f}"
