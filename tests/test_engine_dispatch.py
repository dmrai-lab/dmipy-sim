"""Engine-dispatch parity for ``simulate(engine=...)`` (Phase 5).

The replay engine is the primary signal path; the fused inline step-kernels are
kept as the validation oracle + fallback.  This module asserts the selector:

  (1) ``engine='replay'`` == ``engine='fused'`` to ``max(0.02, 1/sqrt(N))`` on
      each SUPPORTED geometry x effect (pure gradient / scalar T2 / surface
      relaxivity), for the geometries "auto" routes to replay.
  (2) ``engine='auto'`` (the default) reproduces the fused public result within
      tolerance on a representative set — i.e. the default did not silently
      shift the shipped physics.
  (3) ``engine='replay'`` raises ``NotImplementedError`` (naming the gap) on the
      unsupported paths (membrane permeability, PackedMyelinatedCylinders,
      return_compartments), and ``engine='auto'`` falls those back to fused.

SAME seed / N / waveform go through both engines.  The replay producer walks
with a (sub-stepped) scan and stores positions in float16, so parity is at the
MC-noise floor, not bit-identical.  Heavy MC -> auto-marked ``slow`` via
``tests/conftest.py::_SLOW_MC_MODULES``.
"""
import numpy as np
import numpy.testing as npt
import pytest

import dmipy_sim as d
from dmipy_sim import simulate
from dmipy_sim.geometries import pack_cylinders, pack_spheres

D = 2e-9          # m^2/s
SEED = 11
R = 5e-6          # m
T2 = 60e-3        # s
RHO = 1e-5        # m/s
N = 12_000


def _tol(n):
    return max(0.02, 1.0 / np.sqrt(n))


def _pgse(delta, DELTA, n_t, b, G_magnitude=0.2):
    return d.set_b(
        d.pgse(delta=delta, DELTA=DELTA, G_magnitude=G_magnitude,
               bvecs=[[1.0, 0.0, 0.0]], n_t=n_t, slew_rate=np.inf), b)


_WF = _pgse(delta=8e-3, DELTA=32e-3, n_t=400, b=1.0e9)
# Mesh MC is much heavier per step on CPU -> short TE / small n_t (still
# sub_steps == 1) and fewer walkers.
_WF_MESH = _pgse(delta=1.5e-3, DELTA=6e-3, n_t=140, b=0.5e9, G_magnitude=0.3)
_N_MESH = 3_000


def _packed_cyl():
    radii = np.full(4, 3e-6)
    centers, L, _ = pack_cylinders(radii, target_vf=0.20, seed=0)
    return d.PackedCylinders(radii=radii, centers=centers, L=L,
                             orientation=[0.0, 0.0, 1.0])


def _packed_sph():
    radii = np.full(4, 3e-6)
    centers, L, _ = pack_spheres(radii, target_vf=0.08, seed=0)
    return d.PackedSpheres(radii=radii, centers=centers, L=L)


def _mesh():
    trimesh = pytest.importorskip("trimesh")
    m = trimesh.creation.icosphere(subdivisions=2, radius=R)
    return d.Mesh(np.asarray(m.vertices, np.float64),
                  np.asarray(m.faces, np.int32))


def _mesh_rho():
    trimesh = pytest.importorskip("trimesh")
    m = trimesh.creation.icosphere(subdivisions=2, radius=R)
    return d.Mesh(np.asarray(m.vertices, np.float64),
                  np.asarray(m.faces, np.int32),
                  intra={"surface_relaxivity_t2": RHO})


# ── (base geom, surface-relaxivity geom or None, n_walkers, waveform) ─────────
def _supported_specs():
    return {
        "free":     (d.FreeDiffusion(), None, N, _WF),
        "box1d":    (d.Box1D(length=10e-6),
                     d.Box1D(length=10e-6, surface_relaxivity_t2=RHO), N, _WF),
        "sphere":   (d.Sphere(radius=R),
                     d.Sphere(radius=R, surface_relaxivity_t2=RHO), N, _WF),
        "cylinder": (d.Cylinder(radius=R, orientation=[0, 0, 1]),
                     d.Cylinder(radius=R, orientation=[0, 0, 1],
                                surface_relaxivity_t2=RHO), N, _WF),
        "ellipsoid": (d.Ellipsoid(semiaxes=(4e-6, 5e-6, 8e-6)), None, N, _WF),
        "packed_cyl": (_packed_cyl(), None, N, _WF),
        "packed_sph": (_packed_sph(), None, N, _WF),
        "mesh":     (_mesh(), _mesh_rho(), _N_MESH, _WF_MESH),
    }


_SUPPORTED = ["free", "box1d", "sphere", "cylinder", "ellipsoid",
              "packed_cyl", "packed_sph", "mesh"]


@pytest.fixture(scope="module")
def _case(request):
    name = request.param
    geom, geom_rho, n, wf = _supported_specs()[name]
    sf = np.asarray(simulate(n, D, wf, geom, seed=SEED, engine="fused",
                             require_gpu=False)).ravel()
    return dict(name=name, geom=geom, geom_rho=geom_rho, n=n, wf=wf, S_fused=sf)


@pytest.mark.parametrize("_case", _SUPPORTED, indirect=True)
def test_replay_matches_fused_pure_gradient(_case):
    """(1a) engine='replay' == engine='fused' — pure gradient (no relaxation)."""
    c = _case
    S_rep = np.asarray(simulate(c["n"], D, c["wf"], c["geom"], seed=SEED,
                                engine="replay", require_gpu=False)).ravel()
    npt.assert_allclose(
        S_rep, c["S_fused"], atol=_tol(c["n"]),
        err_msg=f"[{c['name']}] replay must match fused (pure gradient)")


@pytest.mark.parametrize("_case", _SUPPORTED, indirect=True)
def test_replay_matches_fused_scalar_T2(_case):
    """(1b) engine='replay' == engine='fused' — scalar T2."""
    c = _case
    S_f = np.asarray(simulate(c["n"], D, c["wf"], c["geom"], seed=SEED, T2=T2,
                              engine="fused", require_gpu=False)).ravel()
    S_r = np.asarray(simulate(c["n"], D, c["wf"], c["geom"], seed=SEED, T2=T2,
                              engine="replay", require_gpu=False)).ravel()
    npt.assert_allclose(
        S_r, S_f, atol=_tol(c["n"]),
        err_msg=f"[{c['name']}] replay must match fused (scalar T2)")


@pytest.mark.parametrize("_case", _SUPPORTED, indirect=True)
def test_replay_matches_fused_surface_relaxivity(_case):
    """(1c) engine='replay' == engine='fused' — surface relaxivity rho."""
    c = _case
    if c["geom_rho"] is None:
        pytest.skip(f"{c['name']} has no boundary — surface relaxivity N/A")
    S_f = np.asarray(simulate(c["n"], D, c["wf"], c["geom_rho"], seed=SEED,
                              engine="fused", require_gpu=False)).ravel()
    S_r = np.asarray(simulate(c["n"], D, c["wf"], c["geom_rho"], seed=SEED,
                              engine="replay", require_gpu=False)).ravel()
    npt.assert_allclose(
        S_r, S_f, atol=_tol(c["n"]),
        err_msg=f"[{c['name']}] replay must match fused (surface relaxivity)")


@pytest.mark.parametrize("_case", _SUPPORTED, indirect=True)
def test_auto_default_reproduces_fused(_case):
    """(2) engine='auto' (default) reproduces the fused public result — the
    shipped default did not silently shift the physics."""
    c = _case
    S_auto = np.asarray(simulate(c["n"], D, c["wf"], c["geom"], seed=SEED,
                                 require_gpu=False)).ravel()
    npt.assert_allclose(
        S_auto, c["S_fused"], atol=_tol(c["n"]),
        err_msg=f"[{c['name']}] auto default must reproduce fused")


def test_replay_raises_on_permeability():
    """(3) engine='replay' raises NotImplementedError for membrane permeability."""
    geom = d.Cylinder(radius=R, orientation=[0, 0, 1], permeability=2e-5)
    with pytest.raises(NotImplementedError, match="permeability"):
        simulate(2_000, D, _WF, geom, seed=SEED, engine="replay",
                 require_gpu=False)


def test_replay_raises_on_return_compartments():
    """(3) engine='replay' raises NotImplementedError for a single-pass internal."""
    geom = d.Cylinder(radius=R, orientation=[0, 0, 1])
    with pytest.raises(NotImplementedError, match="return_compartments"):
        simulate(2_000, D, _WF, geom, seed=SEED, engine="replay",
                 return_compartments="final", require_gpu=False)


def test_replay_raises_on_packed_myelin():
    """(3) engine='replay' raises NotImplementedError for PackedMyelinatedCylinders
    (fused single-reflection kernel is not position-parity with the replay walk)."""
    inner = np.full(3, 3e-6)
    g_ratios = np.full(3, 0.7)
    centers = np.array([[0.0, 0.0], [12e-6, 0.0], [-12e-6, 0.0]])
    geom = d.PackedMyelinatedCylinders(
        inner_radii=inner, g_ratios=g_ratios, centers=centers,
        cell_size=40e-6, N_max=3)
    with pytest.raises(NotImplementedError, match="PackedMyelinatedCylinders"):
        simulate(2_000, None, _WF, geom, seed=SEED, engine="replay",
                 require_gpu=False)


def test_auto_falls_back_to_fused_on_unsupported():
    """engine='auto' transparently uses fused for an unsupported path
    (permeability) — no error, a valid signal comes back."""
    geom = d.Cylinder(radius=R, orientation=[0, 0, 1], permeability=2e-5)
    S = np.asarray(simulate(3_000, D, _WF, geom, seed=SEED, require_gpu=False)).ravel()
    assert S.shape == (1,) and np.all(np.isfinite(S)) and 0.0 <= S[0] <= 1.0


def test_bad_engine_value_raises():
    with pytest.raises(ValueError, match="engine must be"):
        simulate(100, D, _WF, d.FreeDiffusion(), engine="nope", require_gpu=False)
