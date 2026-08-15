"""The acceleration grid must be a speed knob, not a physics knob.

`feature_radius` sets the cell size of the triangle-lookup grid. Nothing about the physics depends on it:
the same closed surface confines the same walkers whatever resolution we happen to index it at. So the
signal must not move when it changes, and no walker may leave an impermeable mesh at any setting.

This is the test disimpy runs (`test_mesh_diffusion` sweeps `n_sv` over `[1,1,1]`, `[1,5,20]`, `[10,10,10]`
-- including brute force with no acceleration at all -- checks the signal against MISST in every case, and
then asserts explicitly that no spins leaked). Cottaar's grid is sized from object count and MC/DC's from
step length; in neither does it enter the answer. Ours does, which is what #40 and #33 are.

**These run with `reject_escape` disabled, deliberately.** That guard rejects a step whose start and end
classify differently, and it is doing almost all of the confining: on the coarse mesh, retention falls from
93.4% to 45.0% when it is switched off. Testing with it on would certify the net rather than the
reflection, and the net is itself built on the classifier that #33 is about -- it cannot see an escape when
both sides of the step read "exterior", which is why its effectiveness collapses as the mesh is refined
(93.4% -> 35.9%). What has to be correct underneath is the collision test, so that is what is measured.

Both are `xfail(strict=True)`: they encode the requirement, they fail today, and they will fail *again*
when the fix lands -- as XPASS -- which is the prompt to delete the marker rather than quietly inherit it.
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


def _tube(subdivisions):
    """One closed cylinder, indexed at three grid resolutions.

    Built at unit scale and converted: trimesh unitizes against an absolute tolerance, so a cylinder
    constructed directly in metres comes back degenerate.
    """
    m = trimesh.creation.cylinder(radius=3.0, height=24.0, sections=96)
    for _ in range(subdivisions):
        m = m.subdivide()
    V = np.asarray(m.vertices, float) * UM
    F = np.asarray(m.faces, np.int64)
    e = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
    mesh = Mesh(V, F, periodic=False, voxel_min=V.min(0) - 2 * UM, voxel_max=V.max(0) + 2 * UM,
                feature_radius=0.5 * float(np.median(e)))
    mesh.reject_escape = False        # measure the collision test, not the safety net
    return mesh, m


def _run(subdivisions, n_walkers=800):
    mesh, tri = _tube(subdivisions)
    wf = set_b(pgse(delta=5e-3, DELTA=15e-3, G_magnitude=0.05, bvecs=[[1, 0, 0]], n_t=200), 5e8)
    out = simulate(n_walkers, D, wf, mesh, seed=SEED, return_positions=True, require_gpu=False)
    arrs = [np.asarray(a) for a in out]
    signal = float(np.atleast_1d(arrs[0]).ravel()[0])
    pos = [a for a in arrs if a.ndim == 2 and a.shape[-1] == 3][-1]
    return signal, float(tri.contains(pos / UM).mean())


@pytest.mark.xfail(strict=True, reason="dmrai-lab/dmipy-sim#41: candidate triangles are gathered around "
                                       "the step's start rather than along the swept segment, so the "
                                       "collision test misses crossings and walkers leave an impermeable "
                                       "mesh (measured 45% retained at the coarsest setting).")
def test_an_impermeable_mesh_confines_its_walkers_at_every_grid_resolution():
    """No walker may leave a closed surface, whatever resolution the triangle index happens to use."""
    retained = {s: _run(s)[1] for s in SUBDIVISIONS}
    worst = min(retained.values())
    assert worst > 0.98, (
        f"walkers escaped an impermeable mesh: retention by subdivision {retained}; "
        f"the surface is closed and watertight at every one of them")


@pytest.mark.xfail(strict=True, reason="dmrai-lab/dmipy-sim#41: retention depends on the grid resolution "
                                       "(93.4% -> 35.9% with the escape guard, 45.0% -> 9.8% without), so "
                                       "the signal follows it.")
def test_the_signal_does_not_move_when_the_grid_is_refined():
    """Same geometry, same walkers, same waveform -- only the indexing resolution differs, so the answer
    must not. Refining a mesh is the natural response to a suspicious result; it must not change the
    physics."""
    signals = {s: _run(s)[0] for s in SUBDIVISIONS}
    lo, hi = min(signals.values()), max(signals.values())
    # Monte-Carlo scatter on 800 walkers is a few e-3; a physics-dependent grid moves it far more.
    assert hi - lo < 0.02, f"signal moved with grid resolution alone: {signals}"
