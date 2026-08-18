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
