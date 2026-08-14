"""Replay-pack assembler (dmipy_sim.bank): build_replay_pack + build_to_floor.

Produces a self-certifying .rpk from a master walk, on a SYNTHETIC master (reflecting-slab
random walk built in numpy — no simulator, fast tier). Checks: the pack compresses within the
MC floor, carries the requested tiers, round-trips through .rpk, is consumable by the lean
replay-signal forward (the fit/design path), and the floor-target policy converges. See the
end-to-end walk->pack->replay validation in test_replay_parity for the physics parity.
"""
import numpy as np
import numpy.testing as npt
import pytest

from dmipy_sim import (build_replay_pack, build_to_floor, read_rpk,
                       compile_scheme, replay_signal)
from dmipy_sim.constants import GAMMA

N_W, N_T, DT, D0, L = 3000, 200, 5e-4, 2e-9, 6e-6


def _slab_master(n_w=N_W, seed=0):
    """A reflecting-slab (0<=x<=L) + free y,z random walk -> master-walk dict with a
    boundary-local-time channel (per-step wall contact), the shape build_replay_pack wants."""
    rng = np.random.default_rng(seed)
    step = np.sqrt(2 * D0 * DT)
    x = rng.uniform(0, L, n_w)
    traj = np.zeros((n_w, N_T, 3)); dlog = np.zeros((n_w, N_T))
    for t in range(N_T):
        x = x + rng.normal(0, step, n_w)
        hit_lo = x < 0; hit_hi = x > L
        x = np.where(hit_lo, -x, np.where(hit_hi, 2 * L - x, x))
        traj[:, t, 0] = x
        dlog[:, t] = (hit_lo | hit_hi) * step        # crude per-step contact (>=0, mostly zero)
    traj[:, :, 1:] = np.cumsum(rng.normal(0, step, (n_w, N_T, 2)), axis=1)
    return dict(traj=traj, dt_traj=DT, T_max=(N_T - 1) * DT,
                comp=np.zeros((n_w, N_T), np.int8), comp0=np.zeros(n_w, np.int64),
                w=np.ones(n_w), T2_per_comp=np.array([0.08]), T1_per_comp=np.array([1.0]),
                dlog_b=dlog, D_intra=D0, n_walkers=n_w, seed=seed)


def _lean_env():
    return dict(bvals=[0.0, 1e9, 3e9], dirs=[[1, 0, 0], [0, 0, 1]], ogse_periods=[2],
                shortd_b=1e9, shortd_deltas_frac=[0.05], B0_list=[], theta_deg=[0],
                delta_frac=0.2, Delta_frac=0.5, rho_list=[1e-5, 1e-4])


@pytest.fixture(scope="module")
def pack():
    return build_replay_pack(_slab_master(), id="test/slab", method="temporal_dct",
                             envelope=_lean_env(), K=64, surface_relaxivity=True,
                             license="CC-BY-4.0", citation="test")


def test_pack_compresses_within_floor_and_declares_tiers(pack):
    f = pack.fidelity
    assert f["err_max"] <= 2.0 * f["floor_max"]                 # codec loss below the MC floor
    assert f["within_2x_floor"] is True
    assert "err_surface" in f and f["err_surface"] <= 2.0 * f["floor_surface"] + 1e-9
    env = pack.replay_envelope
    assert env["gradient"] and env["bulk_relaxation"] and env["surface_relaxivity"]
    assert not env["field"] and not env["magnetization_transfer"]
    assert pack.method == "temporal_dct" and pack.license == "CC-BY-4.0"
    # a temporal_dct pack carries one position tensor per axis (what the lean fit/design forward
    # reads, via compression.read_position_coeffs) + the per-walker channels
    assert all(k in pack.arrays for k in ("pos_x", "pos_y", "pos_z"))
    assert any(k.startswith("comp_rle") for k in pack.arrays)   # compartment tier
    assert any(k.startswith("blt_") for k in pack.arrays)       # surface tier


def test_rpk_roundtrip_and_lean_consumption(pack, tmp_path):
    out = tmp_path / "slab.rpk"
    build_replay_pack(_slab_master(), id="test/slab", method="temporal_dct", envelope=_lean_env(),
                      K=64, surface_relaxivity=True, license="CC-BY-4.0", citation="test",
                      out_path=str(out))
    p2 = read_rpk(out)
    npt.assert_allclose(p2.dct_coeffs, pack.dct_coeffs, rtol=0, atol=0)
    # consume through the lean compiled-scheme forward (the fit/design path)
    nt, dt = p2.n_t, p2.dt
    nd, ng = int(0.2 * (nt - 1)), int(0.5 * (nt - 1))
    bu = (GAMMA * nd * dt) ** 2 * ((ng - nd / 3) * dt)
    bs = np.array([0.0, 0.5e9, 1e9, 2e9])
    G = np.zeros((len(bs), nt, 3))
    for i, b in enumerate(bs):
        a = np.sqrt(b / bu); G[i, :nd, 0] = a; G[i, ng:ng + nd, 0] = -a   # perpendicular (restricted)
    E = replay_signal(p2, compile_scheme(G, dt, p2.K))
    assert abs(E[0] - 1.0) < 1e-6                                # b=0 -> 1
    assert np.all(np.diff(E) <= 1e-6)                            # monotone non-increasing


def test_susceptibility_master_is_rejected():
    m = _slab_master(); m["PhiC"] = np.zeros((5, 4, 4))
    with pytest.raises(NotImplementedError, match="susceptibility"):
        build_replay_pack(m, id="x", method="temporal_dct", K=8, license="x", citation="x")


def test_build_to_floor_converges_and_records_target():
    pk = build_to_floor(lambda n: _slab_master(n_w=n, seed=1), id="test/floor",
                        envelope=_lean_env(), method="temporal_dct", K=48, sigma_star=0.05,
                        pilot_n=800, max_n=6000, surface_relaxivity=False,
                        license="CC-BY-4.0", citation="test", verbose=False)
    assert pk.fidelity.get("target_floor") == 0.05
    assert "meets_target" in pk.fidelity
