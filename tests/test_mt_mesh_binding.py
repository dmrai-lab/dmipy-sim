"""Emergent MT through the MESH collision path must reach the same equilibrium as an analytic pore.

No MT test touched a `Mesh` before this one, and that gap hid a 33% error. MT's free->bound rate is not
imposed; it emerges from the boundary local time accumulated at wall encounters. `Mesh.reject_escape`
compared raw `_classify_arr` labels to catch leaks, and that classifier calls a point with an empty 27-cell
gather *exterior* -- so inside a pore wider than the gather, steps crossing the gather frontier with no wall
within reach of either end read as wall crossings and were discarded in BOTH directions. That sealed the
near-wall shell off from the bulk and starved the binding channel: the mesh sphere reached an equilibrium bound
fraction of 0.2296 against 0.3333, and the boundary local time summed over the ensemble was ALREADY accurate on the
broken path (0.9959 of (S/V)*D, 0.9990 after), because sealing traps walkers in the near-wall shell as much as it
excludes the interior and the two errors cancel for a linear log-weight. Binding saturates -- a bound walker freezes
and stops accumulating -- so only the saturating observable exposed it.

The defect is held on the guard itself, `Mesh._escaped`: it must fire for a step that crosses a wall and for no step that
meets none. The walk's consequence, the bound fraction `k_f/(k_f+k_r)` (exact in the fast-exchange limit) with
`kappa_MT` derived from the mesh's own S/V, takes 1.5e7 sub-steps of a mesh walk and a burn-in several times that to
resolve at the 6 % the broken path missed by 33 %, which is minutes of a GPU for a defect the guard shows in
a fraction of a second.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from dmipy_sim.geometry.mesh import Mesh
from dmipy_sim.engine.mt_walk import simulate_mt_trajectories
from dmipy_sim.engine.physics import collision_sub_steps

UM = 1e-6
R = 2e-6
D = 2e-9
K_F, K_R = 200.0, 400.0         # the binding rates, 1/s: the dwell time is 1 / K_R
BLOCK_H = 2e-5                  # s: the block the binding walk accumulates over
SUBDIVISIONS = 3


def _mesh_sphere(subdivisions=SUBDIVISIONS):
    """Icosphere built at unit scale then converted -- trimesh unitizes against an ABSOLUTE tolerance, so
    constructing directly at SI scale collapses the vertex normals."""
    ico = trimesh.creation.icosphere(subdivisions=subdivisions, radius=R / UM)
    V = np.asarray(ico.vertices, float) * UM
    F = np.asarray(ico.faces, np.int64)
    edges = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
    mesh = Mesh(V, F, periodic=False,
                voxel_min=V.min(0) - 0.5 * UM, voxel_max=V.max(0) + 0.5 * UM,
                feature_radius=0.5 * float(np.median(edges)))
    return mesh, V, F


def test_the_escape_guard_fires_on_wall_crossings_and_on_nothing_else():
    """A step deep in the pore, with no wall within the gather's reach of either end, is not an escape; a step across
    a wall, from one step inside it to one step outside, is.

    Claim: the guard is silent on every wall-free step and fires on every crossing one. The sealing defect flagged
    the wall-free steps that cross the gather frontier: 27 of 2,000 steps at the engine's own step on this mesh
    (1.4 %, whose raw labels differ), so 1,000 steps miss it with probability ``(1 - 0.014)^1000`` = 8e-7. The crossing
    steps are all 1,280 facets, once each, through their centres (never an edge)."""
    mesh, V, F = _mesh_sphere()
    ell = float(np.sqrt(6 * D * BLOCK_H / collision_sub_steps(mesh, D, BLOCK_H)))      # the walk's own step
    rng = np.random.default_rng(0)
    n = 1_000
    dirs = rng.normal(size=(n, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    deep = (dirs * (0.8 * R * rng.uniform(size=n) ** (1.0 / 3.0))[:, None]).astype(np.float32)   # within 0.8 R of the centre
    step = rng.normal(size=(n, 3)); step = (step / np.linalg.norm(step, axis=1)[:, None] * ell).astype(np.float32)
    escaped = jax.jit(jax.vmap(mesh._escaped))
    assert not np.asarray(escaped(jnp.asarray(deep), jnp.asarray(deep + step))).any(), "a wall-free step read as an escape"
    centres = V[F].mean(axis=1)
    out = centres / np.linalg.norm(centres, axis=1)[:, None]
    inside, outside = (centres - 0.5 * ell * out).astype(np.float32), (centres + 0.5 * ell * out).astype(np.float32)
    missed = ~np.asarray(escaped(jnp.asarray(inside), jnp.asarray(outside)))
    assert not missed.any(), f"{missed.sum()} of {len(F)} wall-crossing steps were not flagged as escapes"


def test_the_mt_driver_walks_the_pool_its_seeds_name():
    """`simulate_mt_trajectories` had no `r0`, so it always seeded `init_positions(n, key)` -- and that
    defaults to `intra=True`, INSIDE the surface.

    Every MT test before this one used a sphere, where inside IS the pool of interest, so the omission was
    invisible. It is not invisible in a fibre bundle: the extra-axonal pool's geometry is the OUTER surface
    and its walkers belong outside it, so the retired bundle builder's emergent branch (which passed `r0` on the
    plain path and dropped it on the MT path) re-simulated the intra pool with `D_extra` and labelled the
    result "extra". Wall contact then measured 0.54x the extra pool's analytic `(S/V)*D` -- not because the
    walk was wrong (MT local time tracked a plain walk to 0.97x) but because the walkers were in the wrong
    compartment.

    Asserted on the pool the walkers occupy rather than on a bound fraction: `intra=False` seeds are the
    same `mesh_contains` rejection sampling the intra seeds use, so this pins the plumbing without also
    depending on the binding physics. `kappa_MT=0` skips the bound-pool burn-in (which would advance the
    ensemble before the first saved step) and `block_h` is set so one step displaces ~0.035 um against
    R=2 um, i.e. containment cannot flip by diffusion alone.
    """
    from dmipy_sim.fields.susceptibility_field import mesh_contains
    import jax

    mesh, _, _ = _mesh_sphere(subdivisions=2)
    n, block_h, n_t = 500, 1e-7, 4
    V64 = np.asarray(mesh.vertices, np.float64)
    F64 = np.asarray(mesh.faces, np.int64)

    def frac_inside(p):
        return float(np.asarray(mesh_contains(V64, F64, np.asarray(p, np.float64))).mean())

    r0_out = np.asarray(mesh.init_positions(n, jax.random.PRNGKey(3), pool="extra"), np.float64)
    assert frac_inside(r0_out) < 0.01, "the outside seeds are not outside; test cannot conclude anything"

    def first_step_pool(r0):
        out = simulate_mt_trajectories(n, D, mesh, n_t * block_h, block_h, kappa_MT=0.0,
                                       dwell_time=1.0 / K_R, seed=11, r0=r0,
                                       sub_steps=collision_sub_steps(mesh, D, block_h),
                                       require_gpu=False, raw_subset=n)
        return frac_inside(np.asarray(out.raw_positions)[:, 0, :])

    # Self-guard: the default must still land inside, or an unrelated change could make this vacuous.
    default_in = first_step_pool(None)
    assert default_in > 0.95, (
        f"default seeding put only {default_in:.3f} inside -- `init_positions` no longer defaults to "
        f"intra=True, so this test no longer guards the bug it was written for")

    explicit_in = first_step_pool(r0_out)
    assert explicit_in < 0.02, (
        f"explicit outside seeds gave {explicit_in:.3f} inside the surface: `r0` is being ignored or "
        f"overwritten (default seeding gives {default_in:.3f})")
