"""Emergent MT through the MESH collision path must reach the same equilibrium as an analytic pore.

No MT test touched a `Mesh` before this one, and that gap hid a 33% error. MT's free->bound rate is not
imposed; it emerges from the boundary local time accumulated at wall encounters. `Mesh.reject_escape`
compared raw `_classify_arr` labels to catch leaks, and that classifier calls a point with an empty 27-cell
gather *exterior* -- so inside a pore wider than the gather, steps crossing the gather frontier with no wall
within reach of either end read as wall crossings and were discarded in BOTH directions. That sealed the
near-wall shell off from the bulk and starved the binding channel.

Why this test and not a surface-relaxivity one: the ensemble-summed local time was ALREADY accurate on the
broken path (0.9959 of (S/V)*D, 0.9990 after), because sealing traps walkers in the near-wall shell as much
as it excludes the interior and the two errors cancel for a linear log-weight. Binding saturates -- a bound
walker freezes and stops accumulating -- so only the saturating observable exposes it. A mesh
Brownstein-Tarr test would have passed throughout, which is why this is an MT test.

Held to `k_f/(k_f+k_r)`, exact in the fast-exchange limit, with `kappa_MT` derived from the mesh's OWN
measured S/V so the reference is theory rather than another simulation.
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from dmipy_sim.mesh import Mesh
from dmipy_sim.mt_walk import simulate_mt_trajectories
from dmipy_sim.physics import collision_sub_steps

UM = 1e-6
R = 2e-6
D = 2e-9
# A short dwell keeps this affordable: the bound-pool burn-in advances the ensemble in chunks of one dwell
# until occupancy plateaus, so it, not the saved walk, dominates the cost. k_f scales with it to hold
# f_b = 1/3. Runs in ~24 s on one GPU.
K_F, K_R = 200.0, 400.0
T_MAX, DT_SAVE, N_WALKERS = 0.010, 2e-5, 3000
SUBDIVISIONS = 3      # subdiv 2 halves the runtime but only shows -13% broken vs -33% here


def _mesh_sphere(subdivisions=SUBDIVISIONS):
    """Icosphere built at unit scale then converted -- trimesh unitizes against an ABSOLUTE tolerance, so
    constructing directly at SI scale collapses the vertex normals."""
    ico = trimesh.creation.icosphere(subdivisions=subdivisions, radius=R / UM)
    V = np.asarray(ico.vertices, float) * UM
    F = np.asarray(ico.faces, np.int64)
    scaled = trimesh.Trimesh(V, F, process=False)
    edges = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
    mesh = Mesh(V, F, periodic=False,
                voxel_min=V.min(0) - 0.5 * UM, voxel_max=V.max(0) + 0.5 * UM,
                feature_radius=0.5 * float(np.median(edges)))
    return mesh, float(scaled.area / abs(scaled.volume))


@pytest.mark.slow
def test_mesh_mt_reaches_the_analytic_bound_fraction():
    mesh, s_over_v = _mesh_sphere()
    f_b = K_F / (K_F + K_R)
    kappa_mt = K_F / s_over_v      # k_f = kappa*(S/V), from the mesh's own surface

    # Explicit sub_steps: the shipped MT rule keys off `feature_radius` (mesh resolution) and would ask for
    # thousands. The collision criterion is what governs here -- a step that outruns the 27-cell lookup
    # misses encounters outright, and a missed encounter contributes no local time, so the binding rate is
    # under-counted rather than merely noisy. Measured on this geometry: at n_sub=4 the bound fraction is
    # exactly 0.0000, at 8 it is 12% low, and it converges from 16 (this picks 30).
    n_sub = collision_sub_steps(mesh, D, DT_SAVE)

    out = simulate_mt_trajectories(N_WALKERS, D, mesh, T_MAX, DT_SAVE,
                                   kappa_MT=kappa_mt, dwell_time=1.0 / K_R,
                                   seed=5, sub_steps=n_sub, require_gpu=False)
    bound = float(np.asarray(out[4], np.float64).mean())

    rel = abs(bound - f_b) / f_b
    # Measured -1.47% on this exact configuration, and -32.92% on it before the fix, so the 6% bound sits
    # ~4x above the passing error and ~5x below the broken one.
    assert rel < 0.06, (
        f"mesh MT bound fraction {bound:.4f} vs analytic f_b {f_b:.4f} "
        f"(rel {rel:.3f}, n_sub={n_sub}, S/V={s_over_v:.5g})")


@pytest.mark.slow
def test_the_mt_driver_walks_the_pool_its_seeds_name():
    """`simulate_mt_trajectories` had no `r0`, so it always seeded `init_positions(n, key)` -- and that
    defaults to `intra=True`, INSIDE the surface.

    Every MT test before this one used a sphere, where inside IS the pool of interest, so the omission was
    invisible. It is not invisible in a fibre bundle: the extra-axonal pool's geometry is the OUTER surface
    and its walkers belong outside it, so `mesh_bundle_master`'s emergent branch (which passed `r0` on the
    plain path and dropped it on the MT path) re-simulated the intra pool with `D_extra` and labelled the
    result "extra". Wall contact then measured 0.54x the extra pool's analytic `(S/V)*D` -- not because the
    walk was wrong (MT local time tracked a plain walk to 0.97x) but because the walkers were in the wrong
    compartment.

    Asserted on the pool the walkers occupy rather than on a bound fraction: `intra=False` seeds are the
    same `mesh_contains` rejection sampling the intra seeds use, so this pins the plumbing without also
    depending on the binding physics. `kappa_MT=0` skips the bound-pool burn-in (which would advance the
    ensemble before the first saved step) and `dt_save` is set so one step displaces ~0.035 um against
    R=2 um, i.e. containment cannot flip by diffusion alone.
    """
    from dmipy_sim.susceptibility_field import mesh_contains
    import jax

    mesh, _ = _mesh_sphere(subdivisions=2)
    n, dt_save, n_t = 500, 1e-7, 4
    V64 = np.asarray(mesh.vertices, np.float64)
    F64 = np.asarray(mesh.faces, np.int64)

    def frac_inside(p):
        return float(np.asarray(mesh_contains(V64, F64, np.asarray(p, np.float64))).mean())

    r0_out = np.asarray(mesh.init_positions(n, jax.random.PRNGKey(3), intra=False), np.float64)
    assert frac_inside(r0_out) < 0.01, "the outside seeds are not outside; test cannot conclude anything"

    def first_step_pool(r0):
        out = simulate_mt_trajectories(n, D, mesh, n_t * dt_save, dt_save, kappa_MT=0.0,
                                       dwell_time=1.0 / K_R, seed=11, r0=r0,
                                       sub_steps=collision_sub_steps(mesh, D, dt_save),
                                       require_gpu=False)
        return frac_inside(np.asarray(out[0])[:, 0, :])

    # Self-guard: the default must still land inside, or an unrelated change could make this vacuous.
    default_in = first_step_pool(None)
    assert default_in > 0.95, (
        f"default seeding put only {default_in:.3f} inside -- `init_positions` no longer defaults to "
        f"intra=True, so this test no longer guards the bug it was written for")

    explicit_in = first_step_pool(r0_out)
    assert explicit_in < 0.02, (
        f"explicit outside seeds gave {explicit_in:.3f} inside the surface: `r0` is being ignored or "
        f"overwritten (default seeding gives {default_in:.3f})")
