"""The forward vector-Bloch walk must resolve the same collisions the scalar engine does.

`simulate_bloch`'s plain path accepted `sub_steps`, documented it, and dropped it: `_make_bloch_step_fn`
took exactly one displacement per waveform step. Analytic geometries did not care -- their reflect is
exact at any step length -- but a mesh cannot be, because a step longer than the collision-lookup cell
crosses triangles that were never gathered as candidates and the walker leaves. The path returned the
free-diffusion answer on a restricted mesh, silently, with no error and no warning (#69).

The reference here is `core.simulate` on the identical geometry, waveform and seed, plus an analytic
`Sphere` of the same radius. Both were already trusted: the mesh port was validated against them during
the confinement work. Agreement between the vector and scalar engines is the property worth pinning,
because it is what makes the vector engine's extra machinery (RF, relaxation, susceptibility) trustworthy
on a substrate rather than only on a pore.
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from dmipy_sim import FreeDiffusion, Sphere, pgse, set_b, simulate
from dmipy_sim.bloch import simulate_bloch
from dmipy_sim.mesh import Mesh
from dmipy_sim.physics import walk_sub_steps

UM = 1e-6
R = 2e-6
D = 2e-9
B = 2.0e9
N = 2000
FREE = float(np.exp(-B * D))
EXC = [{"t_s": 0.0, "flip_deg": 90.0, "axis_deg": 90.0}]     # 90_y -> Mx = cos(phi)
TOL = max(0.02, 1.0 / np.sqrt(N))


@pytest.fixture(scope="module")
def waveform():
    """Narrow-pulse PGSE in the square limit, so a 2 um pore is deep in the restricted regime.

    delta must be short against R^2/D = 2 ms or the "restricted" ensemble dephases DURING the pulse and
    stops being distinguishable from the free one: at delta = 4 ms the two differ by 0.02, at 0.5 ms by
    0.95. Reaching b = 2e9 that fast needs ~1.9 T/m, which no real gradient slews to -- irrelevant here,
    and `slew_rate=inf` is what makes the free limit exactly exp(-b*D).
    """
    return set_b(pgse(delta=0.5e-3, DELTA=30e-3, G_magnitude=0.05,
                      bvecs=[[1.0, 0.0, 0.0]], n_t=600, slew_rate=np.inf), B)


@pytest.fixture(scope="module")
def mesh_sphere():
    """Icosphere built at unit scale then converted: trimesh unitizes against an ABSOLUTE tolerance, so
    constructing directly at SI scale collapses the vertex normals."""
    ico = trimesh.creation.icosphere(subdivisions=2, radius=R / UM)
    V = np.asarray(ico.vertices, float) * UM
    F = np.asarray(ico.faces, np.int64)
    edges = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
    return Mesh(V, F, periodic=False,
                voxel_min=V.min(0) - 0.5 * UM, voxel_max=V.max(0) + 0.5 * UM,
                feature_radius=0.5 * float(np.median(edges)))


def _bloch(geom, wf, **kw):
    return float(np.real(simulate_bloch(N, D, wf, geom, EXC, seed=3, require_gpu=False, **kw)[0]))


def _scalar(geom, wf, **kw):
    return float(simulate(N, D, wf, geom, seed=3, require_gpu=False, **kw)[0])


@pytest.mark.slow
def test_free_diffusion_is_untouched(waveform):
    """The sub-step rule must not move a geometry that never collides: exp(-b*D) either way."""
    assert _bloch(FreeDiffusion(), waveform) == pytest.approx(FREE, abs=TOL)
    assert _bloch(FreeDiffusion(), waveform) == pytest.approx(
        _scalar(FreeDiffusion(), waveform), abs=1e-6)


@pytest.mark.slow
def test_an_analytic_pore_matches_the_scalar_engine(waveform):
    sph = Sphere(radius=R)
    e_vec, e_scal = _bloch(sph, waveform), _scalar(sph, waveform)
    assert e_scal > FREE + 0.5, "the analytic sphere is not restricted; b or R is wrong"
    assert e_vec == pytest.approx(e_scal, abs=TOL), (
        f"vector {e_vec:.5f} vs scalar {e_scal:.5f} on an analytic Sphere")


@pytest.mark.slow
def test_a_mesh_pore_matches_the_scalar_engine(mesh_sphere, waveform):
    """The #69 regression. Measured: 0.96317 against a scalar 0.96305, where it used to return 0.05052."""
    e_vec, e_scal = _bloch(mesh_sphere, waveform), _scalar(mesh_sphere, waveform)
    assert e_scal > FREE + 0.5, "the mesh sphere is not restricted in the SCALAR engine either"
    assert e_vec == pytest.approx(e_scal, abs=TOL), (
        f"vector {e_vec:.5f} vs scalar {e_scal:.5f} on a mesh Sphere (free limit {FREE:.5f}) -- "
        f"the Bloch walk is not resolving the mesh collisions")


@pytest.mark.slow
def test_one_displacement_per_waveform_step_loses_the_mesh(mesh_sphere, waveform):
    """The self-guard: reproduce the bug on demand, so the test above cannot pass vacuously.

    `sub_steps=1` is exactly what this path used to do unconditionally. Two things must happen: the
    signal collapses to the free limit (the walls are missed), and the runtime guard says so instead of
    letting it pass in silence -- which is the half of #69 that made it survive so long.
    """
    n_auto = walk_sub_steps(mesh_sphere, D, float(waveform.dt))
    assert n_auto > 1, (
        f"the auto-tune asks for {n_auto} sub-steps on this mesh, so sub_steps=1 is not a downgrade "
        f"and this test proves nothing")

    with pytest.warns(UserWarning, match="collision-lookup cell"):
        e_one = _bloch(mesh_sphere, waveform, sub_steps=1)

    assert e_one < FREE + 0.15, (
        f"at one displacement per waveform step the mesh still confined ({e_one:.5f} against a free "
        f"limit of {FREE:.5f}); the collision lookup is no longer the binding constraint and this "
        f"self-guard needs rewriting")
