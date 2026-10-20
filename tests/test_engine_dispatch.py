"""The coefficient route of ``simulate`` against the fused oracle.

``simulate`` walks the substrate to coefficients and contracts them with the waveform; the fused
single-pass integrator (``engine._fused.simulate_fused``) is the oracle it is held to. This module asserts:

  (1) ``simulate`` == the oracle to ``max(0.02, 1/sqrt(N))`` on each SUPPORTED geometry x effect (pure
      gradient / scalar T2 / surface relaxivity).
  (2) A run the producer cannot walk (PackedMyelinatedCylinders) is the oracle's, exactly.

SAME seed / N / waveform go through both. The producer walks with a separate scan, so parity is at the
MC-noise floor, not bit-identical. The waveforms are coarse enough that ``physics.resolve_sub_steps`` asks
for more than one sub-step on every walled geometry (asserted in the fixture), so both exercise the shared
dispatch. Heavy MC -> auto-marked ``slow`` via ``tests/conftest.py::_SLOW_MC_MODULES``.
"""
import numpy as np
import numpy.testing as npt
import pytest

import dmipy_sim as d
from dmipy_sim import simulate
from dmipy_sim.engine._fused import simulate_fused
from dmipy_sim.geometry import pack_cylinders, pack_spheres

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
        d.pgse([[1.0, 0.0, 0.0]], delta, DELTA, gradient_strengths=G_magnitude, n_t=n_t, slew_rate=np.inf), b)


_WF = _pgse(delta=8e-3, DELTA=32e-3, n_t=200, b=1.0e9)
# Mesh MC is much heavier per step on CPU -> short TE / small n_t and fewer walkers; the feature
# radius sets a lookup cell fine enough that the collision criterion sub-steps too.
_WF_MESH = _pgse(delta=1.5e-3, DELTA=6e-3, n_t=140, b=0.5e9, G_magnitude=0.3)
_N_MESH = 3_000
_MESH_FEATURE = 1e-6


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
                  np.asarray(m.faces, np.int32), feature_radius=_MESH_FEATURE)


def _mesh_rho():
    trimesh = pytest.importorskip("trimesh")
    m = trimesh.creation.icosphere(subdivisions=2, radius=R)
    return d.Mesh(np.asarray(m.vertices, np.float64),
                  np.asarray(m.faces, np.int32), feature_radius=_MESH_FEATURE,
                  compartments={"intra": {"surface_relaxivity_t2": RHO}})


# ── (base geom, surface-relaxivity geom or None, n_walkers, waveform) ─────────
def _supported_specs():
    return {
        "free":     (d.FreeDiffusion(), None, N, _WF),
        "box1d":    (d.Box1D(length=6e-6),
                     d.Box1D(length=6e-6, surface_relaxivity_t2=RHO), N, _WF),
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
    if name != "free":
        from dmipy_sim.engine.physics import resolve_sub_steps
        n_sub = resolve_sub_steps(geom, D, float(wf.dt))
        assert n_sub > 1, (f"[{name}] the dispatch asks for {n_sub} sub-step; this parity must "
                           f"exercise sub-stepping")
    sf = np.asarray(simulate_fused(n, D, wf, geom, seed=SEED, require_gpu=False)).ravel()
    return dict(name=name, geom=geom, geom_rho=geom_rho, n=n, wf=wf, S_fused=sf)


@pytest.mark.parametrize("_case", _SUPPORTED, indirect=True)
def test_coefficient_route_matches_the_oracle_pure_gradient(_case):
    """(1a) pure gradient (no relaxation)."""
    c = _case
    S = np.asarray(simulate(c["n"], D, c["wf"], c["geom"], seed=SEED, require_gpu=False)).ravel()
    npt.assert_allclose(S, c["S_fused"], atol=_tol(c["n"]),
                        err_msg=f"[{c['name']}] the coefficient route must match the oracle (pure gradient)")


@pytest.mark.parametrize("_case", _SUPPORTED, indirect=True)
def test_coefficient_route_matches_the_oracle_scalar_T2(_case):
    """(1b) scalar T2."""
    c = _case
    S_f = np.asarray(simulate_fused(c["n"], D, c["wf"], c["geom"], seed=SEED, T2=T2, require_gpu=False)).ravel()
    S_r = np.asarray(simulate(c["n"], D, c["wf"], c["geom"], seed=SEED, T2=T2, require_gpu=False)).ravel()
    npt.assert_allclose(S_r, S_f, atol=_tol(c["n"]),
                        err_msg=f"[{c['name']}] the coefficient route must match the oracle (scalar T2)")


@pytest.mark.parametrize("_case", _SUPPORTED, indirect=True)
def test_coefficient_route_matches_the_oracle_surface_relaxivity(_case):
    """(1c) surface relaxivity rho."""
    c = _case
    if c["geom_rho"] is None:
        pytest.skip(f"{c['name']} has no boundary — surface relaxivity N/A")
    S_f = np.asarray(simulate_fused(c["n"], D, c["wf"], c["geom_rho"], seed=SEED, require_gpu=False)).ravel()
    S_r = np.asarray(simulate(c["n"], D, c["wf"], c["geom_rho"], seed=SEED, require_gpu=False)).ravel()
    npt.assert_allclose(S_r, S_f, atol=_tol(c["n"]),
                        err_msg=f"[{c['name']}] the coefficient route must match the oracle (surface relaxivity)")


def test_the_coefficient_route_serves_a_permeable_walk():
    """Permeability is NOT a gap, and this pins which half of the contract is which: kappa shapes the WALK,
    and the coefficients already contain every crossing that happened; the replay then applies the gradient
    off the bands, scalar T2/T1 off elapsed time and rho off the contact, none of which depend on kappa."""
    geom = d.Cylinder(radius=R, orientation=[0, 0, 1], permeability=2e-5)
    S = np.asarray(simulate(2_000, D, _WF, geom, seed=SEED, require_gpu=False)).ravel()
    assert S.shape == (1,) and np.all(np.isfinite(S)) and 0.0 <= S[0] <= 1.0 + 1e-6


def test_a_run_the_producer_cannot_walk_is_the_oracles():
    """PackedMyelinatedCylinders has a dedicated fused single-reflection kernel that is not position-parity with
    the multi-bounce walk (``core._replay_gap`` names it): ``simulate`` runs the oracle, bit for bit."""
    from dmipy_sim.engine.core import _replay_gap
    inner = np.full(3, 3e-6)
    g_ratios = np.full(3, 0.7)
    centers = np.array([[0.0, 0.0], [12e-6, 0.0], [-12e-6, 0.0]])
    geom = d.PackedMyelinatedCylinders(
        inner_radii=inner, g_ratios=g_ratios, centers=centers,
        cell_size=40e-6, N_max=3)
    assert "PackedMyelinatedCylinders" in _replay_gap(geom, None)
    np.testing.assert_array_equal(simulate(500, None, _WF, geom, seed=SEED, require_gpu=False),
                                  simulate_fused(500, None, _WF, geom, seed=SEED, require_gpu=False))
