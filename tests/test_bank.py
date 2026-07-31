"""Tests for dmipy_sim.bank — replay-pack (.rpk) generation, round-trip, replay parity,
and open-spec conformance. CPU-only, small and fast.

Phase-3 scope: compression codecs + .rpk creation. Staging/catalog/HF publishing
(Phase 4) are intentionally not present in the public bank module.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from dmipy_sim import simulate_trajectories
from dmipy_sim.geometries import FreeDiffusion
from dmipy_sim import bank
from dmipy_sim import compression as cx
from dmipy_sim.trajectories import replay as _replay
from dmipy_sim.constants import GAMMA

SPEC_CONFORMANCE = "/home/rutger/dmrai-ws/replay-pack-spec/examples/conformance_check.py"


@pytest.fixture(scope="module")
def master():
    """Free-diffusion master with a single-compartment relaxation channel + a
    (zero) boundary-local-time channel (so C2 can be exercised for conformance)."""
    res = simulate_trajectories(
        n_walkers=3000, diffusivity=2e-9, geometry=FreeDiffusion(),
        T_max=20e-3, dt_save=20e-3 / 48, seed=0, require_gpu=False,
        save_relaxation_data=True)
    m = bank.master_from_walk(res, D=2e-9, T2_per_comp=[0.05], T1_per_comp=[1.0],
                              w=np.ones(np.asarray(res[0]).shape[0]))
    return m


@pytest.fixture(scope="module")
def env():
    e = cx.default_envelope(); e["ogse_periods"] = [1, 2]; e["B0_list"] = []
    return e


def _pgse(Nt, dt, b, d):
    t = np.arange(Nt) * dt; T = Nt * dt
    prof = ((t < 0.2 * T).astype(float) - ((t >= 0.5 * T) & (t < 0.7 * T)).astype(float))
    q = GAMMA * dt * np.cumsum(prof); amp = np.sqrt(b / (dt * np.sum(q ** 2))) if b > 0 else 0.0
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    return (amp * prof[:, None] * d[None, :]).astype(np.float32)


def test_master_from_walk_shapes(master):
    tr = master["traj"]
    assert tr.ndim == 3 and tr.shape[2] == 3
    assert master["dt_traj"] > 0 and master["D_intra"] == 2e-9
    assert master["comp"].shape == tr.shape[:2]         # 6-tuple gave comp channel
    assert master["dlog_b"].shape == tr.shape[:2]       # ... and boundary local time


def test_build_pack_within_floor(master, env):
    pack = bank.build_replay_pack(master, id="test/free", method="lowrank", tol=3.0,
                                  license="CC-BY-4.0", citation="test", envelope=env,
                                  K=None)
    assert pack.fidelity["within_2x_floor"] or pack.fidelity["err_max"] <= 3 * pack.fidelity["floor_max"] + 5e-3
    assert pack.replay_envelope["gradient"] and pack.replay_envelope["bulk_relaxation"]
    assert pack.replay_envelope["field"] is False       # public: provider-driven, not stored
    assert pack.cx["walker_preserving"] is True


def test_write_read_roundtrip(master, env, tmp_path):
    out = tmp_path / "free.rpk"
    bank.build_replay_pack(master, id="test/free", method="lowrank", K=32,
                           license="CC-BY-4.0", citation="test", envelope=env, out_path=str(out))
    p = bank.read_rpk(str(out))
    assert p.method == "lowrank" and p.meta["id"] == "test/free"
    assert p.meta["rpk_schema_version"] == "1.2"
    # replay from pack == decode-then-engine on the same reconstructed walkers
    Nt = master["traj"].shape[1]; dt = master["dt_traj"]
    class WF:
        G = np.stack([_pgse(Nt, dt, b, [1, 0, 0]) for b in (1e9, 3e9)])
    S_pack = np.asarray(p.replay(WF))
    traj = np.asarray(p.reconstruct_walkers(), np.float32)
    S_ref = np.asarray(_replay(traj, dt, WF.G, dt))
    assert np.abs(S_pack - S_ref).max() < 1e-6


def test_pack_matches_raw_engine(master, env, tmp_path):
    out = tmp_path / "free.rpk"
    p = bank.build_replay_pack(master, id="test/free", method="lowrank", K=48,
                               license="CC-BY-4.0", citation="test", envelope=env, out_path=str(out))
    Nt = master["traj"].shape[1]; dt = master["dt_traj"]
    class WF:
        G = np.stack([_pgse(Nt, dt, b, d) for d in ([1, 0, 0], [0, 0, 1]) for b in (1e9, 2e9)])
    S_pack = np.asarray(p.replay(WF))
    S_raw = np.asarray(_replay(master["traj"].astype(np.float32), dt, WF.G, dt))
    assert np.abs(S_pack - S_raw).max() < 5e-3           # lossless to ~floor (1/sqrt(3000)=0.018)


def test_relaxation_channel_replays(master, env, tmp_path):
    out = tmp_path / "free.rpk"
    p = bank.build_replay_pack(master, id="test/free", method="lowrank", K=32,
                               license="CC-BY-4.0", citation="test", envelope=env, out_path=str(out))
    Nt = master["traj"].shape[1]; dt = master["dt_traj"]
    class WF:
        G = np.stack([_pgse(Nt, dt, 0.0, [1, 0, 0])])     # b=0 -> pure relaxation weight
    S = np.asarray(p.replay(WF, T2=master["T2_per_comp"], T1=master["T1_per_comp"]))
    # b0 signal is T2-weighted (< 1): exp(-TE/T2) with TE=20ms, T2=50ms ~ 0.67
    assert 0.5 < abs(S[0]) < 0.85


def test_distributional_is_gradient_only(master, env, tmp_path):
    out = tmp_path / "free_g.rpk"
    p = bank.build_replay_pack(master, id="test/free-g", method="gaussian", K=32,
                               license="CC-BY-4.0", citation="test", envelope=env, out_path=str(out))
    assert p.cx["walker_preserving"] is False
    assert p.replay_envelope["bulk_relaxation"] is False and p.replay_envelope["surface_relaxivity"] is False
    # no per-walker channels stored for a distributional pack
    assert "comp_rle_vals" not in p.arrays and "spin_weights" not in p.arrays
    # can still draw an arbitrary walker count for gradient replay
    Nt = master["traj"].shape[1]; dt = master["dt_traj"]
    class WF:
        G = np.stack([_pgse(Nt, dt, 1e9, [1, 0, 0])])
    S = np.asarray(p.replay(WF, n_walkers=5000))
    assert S.shape == (1,) and 0 <= abs(S[0]) <= 1.001


def test_mode_space_gradient_matches_dense(master, env, tmp_path):
    """Fast mode-space gradient replay == reconstruct-then-contract, to machine precision
    (it is an algebraic identity, not an approximation)."""
    out = tmp_path / "free.rpk"
    p = bank.build_replay_pack(master, id="test/free", method="lowrank", K=48,
                               license="CC-BY-4.0", citation="test", envelope=env, out_path=str(out))
    Nt = master["traj"].shape[1]; dt = master["dt_traj"]
    class WF:
        G = np.stack([_pgse(Nt, dt, b, d) for d in ([1, 0, 0], [0, 1, 0]) for b in (1e9, 2e9, 3e9)])
    # mode-space (no reconstruction)
    S_fast = np.asarray(p.replay_gradient(WF, complex_signal=True))
    # dense: reconstruct positions + contract
    traj = np.asarray(p.reconstruct_walkers(), np.float64)
    phi = GAMMA * dt * np.einsum("ntd,mtd->mn", traj, np.asarray(WF.G, np.float64))
    S_dense = np.exp(1j * phi).mean(1)
    assert np.abs(S_fast - S_dense).max() < 1e-9      # identity, machine precision
    # replay() with no relaxation args is a pure gradient replay; non-complex returns Re<exp(iφ)>
    assert np.abs(np.asarray(p.replay(WF)) - S_dense.real).max() < 1e-9


def test_fast_relaxation_matches_dense_engine(master, env, tmp_path):
    """Combined fast path (mode-space φ + relaxation log-weight from the compartment map)
    == the dense engine (reconstruct + replay), to precision."""
    from dmipy_sim.trajectories import replay as _replay_dense
    out = tmp_path / "free.rpk"
    p = bank.build_replay_pack(master, id="test/free", method="lowrank", K=48,
                               license="CC-BY-4.0", citation="test", envelope=env, out_path=str(out))
    Nt = master["traj"].shape[1]; dt = master["dt_traj"]
    class WF:
        G = np.stack([_pgse(Nt, dt, b, [1, 0, 0]) for b in (0.0, 1e9, 2e9)])
    S_fast = np.asarray(p.replay(WF, T2=master["T2_per_comp"], T1=master["T1_per_comp"],
                                 complex_signal=True))
    # dense reference: reconstruct + engine relaxation
    traj = np.asarray(p.reconstruct_walkers(), np.float32)
    phi, logw, _ = _replay_dense(
        traj, dt, np.asarray(WF.G, np.float32), dt, chi_perp=np.ones(Nt, np.float32),
        comp_traj=p._comp(), T2_per_comp=np.asarray(master["T2_per_comp"]),
        T1_per_comp=np.asarray(master["T1_per_comp"]), return_walker_signals=True)
    phi = np.asarray(phi); logw = np.asarray(logw)
    if logw.shape[-1] == 1:
        logw = np.broadcast_to(logw, phi.shape)
    nw = traj.shape[0]
    S_ref = (np.exp(logw) * np.exp(1j * phi) * (np.ones(nw) / nw)[None, :]).sum(1)
    assert np.abs(S_fast - S_ref).max() < 5e-4     # mode-space φ + run-sum logW vs engine


def test_surface_relaxivity_channel_present(master, env, tmp_path):
    """Opt-in surface-relaxivity gating stores a boundary_local_time channel (C2)."""
    out = tmp_path / "free_c2.rpk"
    p = bank.build_replay_pack(master, id="test/free-c2", method="lowrank", K=32,
                               license="CC-BY-4.0", citation="test", envelope=env,
                               surface_relaxivity=True, out_path=str(out))
    assert p.replay_envelope["surface_relaxivity"] is True
    assert p.boundary_local_time() is not None
    assert "boundary_local_time" in p.cx["channels"]


def test_replay_dispersed_deferred(master, env):
    p = bank.build_replay_pack(master, id="test/free", method="lowrank", K=16,
                               license="CC-BY-4.0", citation="test", envelope=env)
    with pytest.raises(NotImplementedError):
        p.replay_dispersed()


def test_replay_bloch_sequence_dispatch(master, env, tmp_path):
    """A BlochSequence dispatches replay() to the vector-Bloch engine (isinstance, NOT
    hasattr('rf_events') — a Waveform has that too) and matches a direct replay_bloch(...)
    on the reconstructed walkers, bit-for-bit."""
    from dmipy_sim.pulse_sequence import BlochSequence
    from dmipy_sim.trajectories import replay_bloch
    out = tmp_path / "free.rpk"
    p = bank.build_replay_pack(master, id="test/free", method="lowrank", K=48,
                               license="CC-BY-4.0", citation="test", envelope=env, out_path=str(out))
    Nt = master["traj"].shape[1]; dt = master["dt_traj"]; TE = dt * (Nt - 1)
    G = np.stack([_pgse(Nt, dt, b, [1, 0, 0]) for b in (0.0, 1e9, 2e9)])   # PHYSICAL lobes
    rf = [{'t_s': 0.0, 'flip_deg': 90.0, 'axis_deg': 0.0, 'duration_s': 0.0, 'offset_hz': 0.0},
          {'t_s': TE / 2.0, 'flip_deg': 180.0, 'axis_deg': 0.0, 'duration_s': 0.0, 'offset_hz': 0.0}]
    seq = BlochSequence(G=G, dt=dt, rf_events=rf, complex_signal=True, family="se")
    S = np.asarray(p.replay(seq, T2=master["T2_per_comp"], T1=master["T1_per_comp"]))
    assert S.shape == (3,) and np.iscomplexobj(S)          # Bloch path -> complex Mx+iMy
    # direct engine call, same reconstructed walkers + same per-comp relaxation
    traj = p.reconstruct_walkers()
    S_ref = np.asarray(replay_bloch(
        traj, dt, G, dt, rf, comp_traj=p._comp(),
        T2_per_comp=np.asarray(master["T2_per_comp"]),
        T1_per_comp=np.asarray(master["T1_per_comp"]),
        echo_steps=seq.echo_steps, weights=p.arrays.get("spin_weights")))
    assert np.abs(S - S_ref).max() < 1e-9


@pytest.mark.skipif(not os.path.exists(SPEC_CONFORMANCE),
                    reason="replay-pack-spec conformance checker not available")
def test_rpk_passes_spec_conformance(master, env, tmp_path):
    """A pack built by build_replay_pack validates against the OPEN spec's conformance
    checker as C0 (Gradient) + C1 (BulkRelax) + C2 (Surface) — the interop guarantee."""
    out = tmp_path / "conformance.rpk"
    p = bank.build_replay_pack(master, id="conformance/free", method="lowrank", K=32,
                               license="CC-BY-4.0", citation="dmipy-sim test",
                               envelope=env, surface_relaxivity=True, out_path=str(out))
    assert p.replay_envelope["bulk_relaxation"] and p.replay_envelope["surface_relaxivity"]
    r = subprocess.run([sys.executable, SPEC_CONFORMANCE, str(out)],
                       capture_output=True, text=True)
    report = r.stdout + r.stderr
    assert r.returncode == 0, f"conformance checker failed:\n{report}"
    assert "CONFORMANT" in report, report
    assert "C0-Gradient" in report and "C1-BulkRelax" in report and "C2-Surface" in report, report
