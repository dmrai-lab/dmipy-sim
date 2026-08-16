"""The acceleration grid must be a speed knob, not a physics knob.

`feature_radius` sets the cell size of the triangle-lookup grid. Nothing about the physics depends on it:
the same closed surface confines the same walkers whatever resolution we happen to index it at. So the
signal must not move when it changes, and no walker may leave an impermeable mesh at any setting.

This is the test disimpy runs (`test_mesh_diffusion` sweeps `n_sv` over `[1,1,1]`, `[1,5,20]`, `[10,10,10]`
-- including brute force with no acceleration at all -- checks the signal against MISST in every case, and
then asserts explicitly that no spins leaked). Cottaar's grid is sized from object count and MC/DC's from
step length; in neither does it enter the answer.

**These run with `reject_escape` disabled, deliberately.** That guard rejects a step whose start and end
classify differently, and it is a net, not a collision test: it cannot see an escape when both ends of the
step read the same, and rejecting a step freezes the walker for that interval, which is its own bias. What
has to be correct underneath is the collision response, so that is what is measured.

Walkers are seeded with `trimesh.sample.volume_mesh` and NOT with `Mesh.init_positions`, for two reasons.
The surface is identical at every subdivision (linear subdivision adds triangles, not curvature), so one
sample is valid for all three arms and the grid resolution becomes the only variable -- seeding per-arm
would compare different walker pools. And `init_positions` trusts the cell-gather classifier wherever the
gather is populated, which misplaces walkers OUTSIDE the surface at coarse settings (measured 6.3% on the
coarse arm here) -- a separate defect, tracked in dmrai-lab/dmipy-sim#50. Folding it in would have this
test fail for a reason that has nothing to do with collisions, which is what it is for.
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from dmipy_sim import set_b, simulate
from dmipy_sim.mesh import Mesh
from dmipy_sim.waveforms import pgse

UM = 1e-6
D = 1.0e-9
SEED = 4242
SUBDIVISIONS = (0, 1, 2)          # same surface, 384 -> 1536 -> 6144 triangles
N_WALKERS = 800
# One step is ~0.13 um here and the post-collision nudge is 1e-4 of that (~1e-5 um).
ESCAPE_TOL_UM = 1e-3


def _surface(subdivisions):
    """One closed cylinder, at unit scale.

    Built at unit scale and converted by the caller: trimesh unitizes against an absolute tolerance, so a
    cylinder constructed directly in metres comes back degenerate.
    """
    m = trimesh.creation.cylinder(radius=3.0, height=24.0, sections=96)
    for _ in range(subdivisions):
        m = m.subdivide()
    return m


@pytest.fixture(scope="module")
def walkers():
    """Start positions, sampled once inside the shared surface and reused by every arm."""
    base = _surface(0)
    pts = trimesh.sample.volume_mesh(base, 4 * N_WALKERS)[:N_WALKERS]
    assert len(pts) == N_WALKERS, "volume sampler under-filled"
    assert base.contains(pts).all()
    return pts


def _run(subdivisions, r0_um):
    tri = _surface(subdivisions)
    V = np.asarray(tri.vertices, float) * UM
    F = np.asarray(tri.faces, np.int64)
    e = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
    mesh = Mesh(V, F, periodic=False, voxel_min=V.min(0) - 2 * UM, voxel_max=V.max(0) + 2 * UM,
                feature_radius=0.5 * float(np.median(e)))
    mesh.reject_escape = False        # measure the collision response, not the safety net

    wf = set_b(pgse(delta=5e-3, DELTA=15e-3, G_magnitude=0.05, bvecs=[[1, 0, 0]], n_t=200), 5e8)
    r0 = np.ascontiguousarray(r0_um * UM, dtype=np.float32)
    out = simulate(len(r0), D, wf, mesh, seed=SEED, return_positions=True, require_gpu=False, r0=r0)
    arrs = [np.asarray(a) for a in out]
    signal = float(np.atleast_1d(arrs[0]).ravel()[0])
    pos = [a for a in arrs if a.ndim == 2 and a.shape == r0.shape][-1]

    # Escape means the walker LEFT, not that it is resting against the wall on the far side of a ray
    # test. A reflected walker is parked a nudge (1e-4 of a step) off the surface, so "outside" on its
    # own is a coin-flip for those; an escapee is a step or more clear of the wall. Judge by depth, with
    # a tolerance well above the nudge and well below one step.
    depth_um = -trimesh.proximity.signed_distance(tri, pos / UM)   # positive = outside
    return signal, int((depth_um > ESCAPE_TOL_UM).sum())


def test_an_impermeable_mesh_confines_its_walkers_at_every_grid_resolution(walkers):
    """No walker may leave a closed surface, whatever resolution the triangle index happens to use.

    Every walker starts verified-inside, so anything outside at the end crossed a wall it should have
    bounced off.
    """
    escaped = {s: _run(s, walkers)[1] for s in SUBDIVISIONS}
    assert escaped == {s: 0 for s in SUBDIVISIONS}, (
        f"walkers left an impermeable mesh: escapees by subdivision {escaped} of {N_WALKERS}; "
        f"the surface is closed and watertight at every one of them")


def test_the_signal_does_not_move_when_the_grid_is_refined(walkers):
    """Same geometry, same walkers, same waveform -- only the indexing resolution differs, so the answer
    must not. Refining a mesh is the natural response to a suspicious result; it must not change the
    physics."""
    signals = {s: _run(s, walkers)[0] for s in SUBDIVISIONS}
    lo, hi = min(signals.values()), max(signals.values())
    # Monte-Carlo scatter on 800 walkers is a few e-3; a physics-dependent grid moves it far more.
    assert hi - lo < 0.02, f"signal moved with grid resolution alone: {signals}"
