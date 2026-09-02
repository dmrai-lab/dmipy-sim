"""Replay parity: walk-once + replay operators reproduce the fused ``simulate()``.

Phase-1 replay foundation.  For each geometry we walk the spins ONCE with
``simulate_trajectories(save_relaxation_data=True)`` and then apply
``replay`` (pure-gradient and with relaxation) post-hoc,
asserting the result equals the fused forward engine (``core.simulate``) — the
validation oracle — for

  (a) pure gradient PGSE (phase only),
  (b) scalar T2,
  (c) per-compartment T2 (``T2_per_comp=[T2, T2]`` ≡ scalar T2 via ``comp_traj``),
  (d) surface relaxivity (ρ replayed off ``dlog_boundary_unit`` vs a substrate
      with ``surface_relaxivity_t2`` baked into the walk).

SAME seed / N / waveform go through both paths.  The generic geometries walk via
a different (sub-stepped) scan than the fused engine, so parity is at the MC
noise floor: tolerance ``max(0.02, 1/sqrt(N))``.  The waveforms are sized so the
producer's sub-step auto-tune leaves ``sub_steps == 1`` (fused single-step is
already accurate), making (a)–(c) essentially bit-identical and isolating (d) to
the boundary-local-time statistics.

Heavy MC — auto-marked ``slow`` via ``tests/conftest.py::_SLOW_MC_MODULES``.
"""
import numpy as np
import numpy.testing as npt
import pytest

import dmipy_sim as d
from dmipy_sim import simulate, simulate_trajectories
from dmipy_sim.trajectories import replay

D = 2e-9          # m²/s
SEED = 7
R = 5e-6          # m (sphere/cylinder radius; mesh icosphere radius)
T2 = 60e-3        # s
RHO = 1e-5        # m/s (surface relaxivity)


def _tol(n):
    return max(0.02, 1.0 / np.sqrt(n))


def _pgse(delta, DELTA, n_t, b, G_magnitude=0.2):
    return d.set_b(
        d.pgse(delta=delta, DELTA=DELTA, G_magnitude=G_magnitude,
               bvecs=[[1.0, 0.0, 0.0]], n_t=n_t, slew_rate=np.inf), b)


# ── Geometry specs ──────────────────────────────────────────────────────────
# Each: (base geom, surface-relaxivity geom or None, n_walkers, waveform).
# Non-mesh geometries are cheap → fine dt (sub_steps==1) + moderate N.
_N_STD = 20_000
_WF_STD = _pgse(delta=8e-3, DELTA=32e-3, n_t=800, b=1.0e9)

# Mesh MC is far heavier per step on CPU → short TE / small n_t (still sub_steps==1
# so the fused engine stays accurate) and a small walker count.
_N_MESH = 3_000
_WF_MESH = _pgse(delta=1.5e-3, DELTA=6e-3, n_t=160, b=0.5e9, G_magnitude=0.3)


def _std_specs():
    specs = {
        "free": (d.FreeDiffusion(), None, _N_STD, _WF_STD),
        "box1d": (d.Box1D(length=10e-6),
                  d.Box1D(length=10e-6, surface_relaxivity_t2=RHO), _N_STD, _WF_STD),
        "sphere": (d.Sphere(radius=R),
                   d.Sphere(radius=R, surface_relaxivity_t2=RHO), _N_STD, _WF_STD),
        "cylinder": (d.Cylinder(radius=R, orientation=[0, 0, 1]),
                     d.Cylinder(radius=R, orientation=[0, 0, 1],
                                surface_relaxivity_t2=RHO), _N_STD, _WF_STD),
    }
    return specs


def _mesh_spec():
    trimesh = pytest.importorskip("trimesh")
    m = trimesh.creation.icosphere(subdivisions=2, radius=R)
    V = np.asarray(m.vertices, np.float64)
    F = np.asarray(m.faces, np.int32)
    return (d.Mesh(V, F),
            d.Mesh(V, F, intra={"surface_relaxivity_t2": RHO}),
            _N_MESH, _WF_MESH)


@pytest.fixture(scope="module")
def _replay_case(request):
    """One walk + the fused oracles for a geometry, computed once per param.

    Returns a dict with the saved walk (traj / dlog / comp), the waveform, and the
    fused-``simulate`` oracle signals S_a (no relaxation), S_b (scalar T2), and
    S_d (surface relaxivity, or None when the geometry has no boundary).
    """
    name = request.param
    if name == "mesh":
        geom, geom_rho, n, wf = _mesh_spec()
    else:
        geom, geom_rho, n, wf = _std_specs()[name]

    dt = float(wf.dt)
    n_t = wf.G.shape[1]
    T_max = dt * (n_t - 1)
    G = np.asarray(wf.G, np.float32)
    chi = np.ones(n_t)

    out = simulate_trajectories(n, D, geom, T_max, dt, seed=SEED,
                                save_relaxation_data=True, require_gpu=False)
    traj, dt_traj, sub_steps, dt_sim, dlog, comp = out

    S_a = np.asarray(simulate(n, D, wf, geom, seed=SEED, require_gpu=False)).ravel()
    S_b = np.asarray(simulate(n, D, wf, geom, seed=SEED, T2=T2, require_gpu=False)).ravel()
    S_d = (None if geom_rho is None
           else np.asarray(simulate(n, D, wf, geom_rho, seed=SEED,
                                    require_gpu=False)).ravel())

    return dict(name=name, n=n, wf=wf, G=G, chi=chi,
                traj=traj, dt_traj=dt_traj, sub_steps=sub_steps,
                dlog=dlog, comp=comp, S_a=S_a, S_b=S_b, S_d=S_d)


_ALL = ["free", "box1d", "sphere", "cylinder", "mesh"]


@pytest.mark.parametrize("_replay_case", _ALL, indirect=True)
def test_replay_pure_gradient_matches_simulate(_replay_case):
    """(a) Phase-only replay == fused simulate() (no relaxation)."""
    c = _replay_case
    S_rep = np.asarray(replay(
        c["traj"], c["dt_traj"], c["G"], c["wf"].dt)).ravel()
    npt.assert_allclose(
        S_rep, c["S_a"], atol=_tol(c["n"]),
        err_msg=f"[{c['name']}] pure-gradient replay must match simulate()")


@pytest.mark.parametrize("_replay_case", _ALL, indirect=True)
def test_replay_scalar_T2_matches_simulate(_replay_case):
    """(b) Scalar-T2 replay == fused simulate(T2=...)."""
    c = _replay_case
    S_rep = np.asarray(replay(
        c["traj"], c["dt_traj"], c["G"], c["wf"].dt,
        chi_perp=c["chi"], T2=T2)).ravel()
    npt.assert_allclose(
        S_rep, c["S_b"], atol=_tol(c["n"]),
        err_msg=f"[{c['name']}] scalar-T2 replay must match simulate(T2)")


@pytest.mark.parametrize("_replay_case", _ALL, indirect=True)
def test_replay_per_compartment_T2_matches_simulate(_replay_case):
    """(c) Per-compartment T2=[T2, T2] (via comp_traj) == fused simulate(T2=...).

    Exercises the ``T2_per_comp`` / ``comp_traj`` code path; with equal
    per-compartment values it must collapse onto the scalar-T2 oracle.
    """
    c = _replay_case
    S_rep = np.asarray(replay(
        c["traj"], c["dt_traj"], c["G"], c["wf"].dt,
        chi_perp=c["chi"], comp_traj=c["comp"], T2_per_comp=[T2, T2])).ravel()
    npt.assert_allclose(
        S_rep, c["S_b"], atol=_tol(c["n"]),
        err_msg=f"[{c['name']}] per-compartment-T2 replay must match simulate(T2)")


@pytest.mark.parametrize("_replay_case", _ALL, indirect=True)
def test_replay_surface_relaxivity_matches_simulate(_replay_case):
    """(d) Surface-relaxivity replay (ρ off dlog_boundary_unit) == fused
    simulate(surface_relaxivity_t2=ρ)."""
    c = _replay_case
    if c["S_d"] is None:
        pytest.skip(f"{c['name']} has no boundary — surface relaxivity N/A")
    S_rep = np.asarray(replay(
        c["traj"], c["dt_traj"], c["G"], c["wf"].dt, chi_perp=c["chi"],
        dlog_boundary_unit=c["dlog"], surface_relaxivity=RHO, D=D)).ravel()
    npt.assert_allclose(
        S_rep, c["S_d"], atol=_tol(c["n"]),
        err_msg=f"[{c['name']}] surface-relaxivity replay must match "
                f"simulate(surface_relaxivity_t2)")


def test_save_false_returns_4_tuple():
    """Without save_relaxation_data, simulate_trajectories returns a 4-tuple."""
    wf = _WF_STD
    dt = float(wf.dt); n_t = wf.G.shape[1]
    out = simulate_trajectories(2_000, D, d.Sphere(radius=R),
                                dt * (n_t - 1), dt, seed=SEED, require_gpu=False)
    assert len(out) == 4
    traj, dt_actual, sub_steps, dt_sim = out
    assert traj.shape == (2_000, n_t, 3)
    # f32 by default since #78: the walk is f32, the pack is f32, and the .rpk spec
    # permits only float32/float64 for `positions`. f16 is opt-in via storage_dtype.
    assert traj.dtype == np.float32


def test_save_true_returns_6_tuple_shapes_dtypes():
    """save_relaxation_data adds dlog_boundary_unit (storage_dtype) + comp_traj (int8)."""
    wf = _WF_STD
    dt = float(wf.dt); n_t = wf.G.shape[1]
    out = simulate_trajectories(2_000, D, d.Cylinder(radius=R, orientation=[0, 0, 1]),
                                dt * (n_t - 1), dt, seed=SEED,
                                save_relaxation_data=True, require_gpu=False)
    assert len(out) == 6
    traj, dt_actual, sub_steps, dt_sim, dlog, comp = out
    assert traj.shape == (2_000, n_t, 3) and traj.dtype == np.float32
    assert dlog.shape == (2_000, n_t) and dlog.dtype == np.float32
    assert comp.shape == (2_000, n_t)
    # Impermeable Cylinder: comp is discrete int8, all zeros; dlog <= 0.
    assert comp.dtype == np.int8
    assert np.all(np.asarray(comp) == 0)
    assert np.all(np.asarray(dlog, np.float64) <= 1e-6)
