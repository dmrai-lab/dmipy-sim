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

from dataclasses import replace
import numpy as np
from dmipy_sim import RFEvent
import pytest

trimesh = pytest.importorskip("trimesh")

from dmipy_sim import FreeDiffusion, Sphere, pgse, set_b, simulate
from dmipy_sim.engine.bloch import simulate_bloch
from dmipy_sim.geometry.mesh import Mesh
from dmipy_sim.engine.physics import walk_sub_steps

UM = 1e-6
R = 2e-6
D = 2e-9
B = 2.0e9
N = 2000
FREE = float(np.exp(-B * D))
EXC = [RFEvent(0.0, 90.0, axis_deg=90.0)]     # 90_y -> Mx = cos(phi)
TOL = max(0.02, 1.0 / np.sqrt(N))


@pytest.fixture(scope="module")
def waveform():
    """Narrow-pulse PGSE in the square limit, so a 2 um pore is deep in the restricted regime.

    delta must be short against R^2/D = 2 ms or the "restricted" ensemble dephases DURING the pulse and
    stops being distinguishable from the free one: at delta = 4 ms the two differ by 0.02, at 0.5 ms by
    0.95. Reaching b = 2e9 that fast needs ~1.9 T/m, which no real gradient slews to -- irrelevant here,
    and `slew_rate=inf` is what makes the free limit exactly exp(-b*D).
    """
    return set_b(pgse([[1.0, 0.0, 0.0]], 0.5e-3, 30e-3, gradient_strengths=0.05, n_t=600, slew_rate=np.inf), B)


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
    rf = EXC + [e for e in wf.rf if e.flip_deg == 180]        # the physical pair needs its 180
    return float(np.real(simulate_bloch(N, D, replace(wf, rf=rf), geom, **kw, seed=3, require_gpu=False)[0]))


def _scalar(geom, wf, **kw):
    # engine='fused' pins the comparison to the direct single-pass walk ("the validation
    # oracle", core.simulate's own description) that `_bloch` is checked against. Every
    # geometry here (FreeDiffusion, Sphere, the mesh sphere) declares `replay_parity`, so
    # `engine='auto'` (the default) now routes through the replay backend instead -- a
    # different numerical path that is not meant to agree bit-for-bit with the vector
    # engine's direct rotation, only within the walk's own Monte-Carlo floor.
    return float(simulate(N, D, wf, geom, seed=3, require_gpu=False, engine='fused', **kw)[0])


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
def test_one_displacement_per_waveform_step_still_resolves_the_mesh(mesh_sphere, waveform):
    """`sub_steps=1` no longer loses the mesh (#626; this test used to pin the opposite).

    It used to: a step longer than the collision-lookup cell crossed triangles that were never
    gathered as candidates, from a SINGLE gather at the step's start (#69). #624 gave every wall
    one loop (`BounceLoop`) that runs a geometry's collision rule to exhaustion, re-gathering
    candidates from each bounce's own position rather than only from the step's start -- so a step
    many cells long is still walked correctly, just coarsely, instead of silently missing the wall.
    The runtime warning (a step this long past the lookup cell is still an accuracy risk worth
    flagging) still fires; what it no longer certifies is "and therefore confinement breaks", so
    this checks that the walk STAYS confined at `sub_steps=1`, not that it collapses.

    Measured: 0.93149 at `sub_steps=1`, against 0.96317 fully resolved
    (`test_a_mesh_pore_matches_the_scalar_engine`) and a free limit of 0.01832 -- a few percent
    off the resolved answer, nowhere near free.
    """
    n_auto = walk_sub_steps(mesh_sphere, D, float(waveform.dt))
    assert n_auto > 1, (
        f"the auto-tune asks for {n_auto} sub-steps on this mesh, so sub_steps=1 is not a downgrade "
        f"and this test proves nothing")

    with pytest.warns(UserWarning, match="collision-lookup cell"):
        e_one = _bloch(mesh_sphere, waveform, sub_steps=1)

    assert e_one > FREE + 0.5, (
        f"at one displacement per waveform step the mesh signal ({e_one:.5f}) collapsed toward the "
        f"free limit ({FREE:.5f}); BounceLoop (#624) should keep the walk confined even at this "
        f"coarse an outer step")
