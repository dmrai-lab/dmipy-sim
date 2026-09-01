"""Replay-pack forward: the compiled-scheme replay reproduces the direct phase integral exactly, the
.rpk round-trips, the JAX twin matches and is differentiable in the waveform, and the surface knob
attenuates. Uses a synthetic pack (no simulator walk needed), so it is self-contained."""
import numpy as np
import numpy.testing as npt
from dmipy_sim.compression import read_position_coeffs, pack_position_arrays
import pytest

from scipy.fft import dct, idct
from dmipy_sim.replay import (read_rpk, write_rpk, compile_scheme, replay_signal,
                              replay_signal_jax, ReplayPack)
from dmipy_sim.constants import GAMMA

N_W, N_T, K = 400, 200, 48
DT = 5e-4


def _synth_pack(seed=0):
    "A synthetic pack: smooth random walker trajectories, stored as their DCT-II coefficients."
    rng = np.random.default_rng(seed)
    # low-frequency-ish trajectories (bounded ~micron scale), so K modes capture them
    traj = np.cumsum(rng.normal(0, 3e-7, size=(N_W, N_T, 3)), axis=1)
    traj -= traj.mean(1, keepdims=True)
    from dmipy_sim.compression import encode_bridge_dst
    arrays, cmeta, _ = encode_bridge_dst(traj, K)              # (N_W, K+2, 3) per axis
    arrays["spin_weights"] = np.ones(N_W, np.float32)
    meta = {"n_t": N_T, "dt": DT, "walk_params": {"n_t": N_T, "dt_traj": DT},
            "compression": {"method": cmeta["method"], "K": K}}
    return arrays, meta, traj


def _pgse(amp, delta, Delta, direction=(1., 0, 0)):
    g = np.zeros((N_T, 3)); nd = max(1, int(round(delta / DT))); ng = int(round(Delta / DT))
    u = np.asarray(direction, float); g[:nd] = amp * u; g[ng:ng + nd] = -amp * u
    return g


def test_replay_equals_direct_phase():
    "Compiled replay == the direct phi = gamma dt sum G.r over the (idct-reconstructed) trajectory."
    arrays, meta, traj = _synth_pack()
    pack = ReplayPack(arrays, meta)
    G = np.stack([_pgse(a, 10e-3, 30e-3) for a in (0.02, 0.05, 0.1)])   # (3, N_T, 3)
    W = compile_scheme(G, DT, K, GAMMA, n_t=N_T)
    E = replay_signal(pack, W)
    # direct reference: reconstruct the truncated trajectory and integrate the phase
    from dmipy_sim.compression import decode
    r = decode(arrays, {"method": "bridge_dst", "n_t": N_T})
    phi = GAMMA * DT * np.einsum("mtc,wtc->mw", G, r)                   # (n_meas, N_W)
    E_direct = np.abs(np.cos(phi).mean(1) + 1j * 0) if False else np.abs(np.exp(1j * phi).mean(1))
    npt.assert_allclose(E, E_direct, atol=1e-10)


def test_rpk_roundtrip(tmp_path):
    arrays, meta, _ = _synth_pack()
    p = tmp_path / "synth.rpk"
    write_rpk(p, arrays, meta)
    pk = read_rpk(p)
    assert pk.n_t == N_T and pk.K == K and pk.n_walkers == N_W
    npt.assert_allclose(pk.dt, DT)
    from dmipy_sim.compression import read_position_coeffs
    npt.assert_allclose(np.asarray(pk.position_coeffs), read_position_coeffs(arrays, dtype=np.float32))


def test_jax_twin_matches_and_differentiable():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    arrays, meta, _ = _synth_pack()
    G = np.stack([_pgse(a, 10e-3, 30e-3) for a in (0.03, 0.08)])
    W = compile_scheme(G, DT, K, GAMMA, n_t=N_T)
    E_np = replay_signal(arrays, W)
    C3 = read_position_coeffs(arrays, dtype=np.float32)
    E_jx = np.abs(np.asarray(replay_signal_jax(C3, arrays["spin_weights"], W)))
    npt.assert_allclose(E_jx, E_np, atol=2e-5)
    # differentiable in the compiled scheme (hence in the waveform): grad is finite
    def loss(Wj):
        return jnp.abs(replay_signal_jax(jnp.asarray(read_position_coeffs(arrays, dtype=np.float32)),
                                         jnp.asarray(arrays["spin_weights"]), Wj)).sum()
    g = jax.grad(loss)(jnp.asarray(W))
    assert np.all(np.isfinite(np.asarray(g))) and np.abs(np.asarray(g)).max() > 0


def test_surface_knob_attenuates():
    arrays, meta, _ = _synth_pack()
    N_W_ = arrays["pos_x"].shape[0]
    # synthetic boundary-local-time channel: real packs store dlog (<=0, an attenuation) at rho/D=1, so
    # the DC coefficient is negative; replay multiplies by rho/D>0 -> exp(<0) -> signal loss.
    blt = np.zeros((N_W_, 16), np.float32)
    blt[:, 0] = -np.abs(np.random.default_rng(1).normal(1.0, 0.2, N_W_))
    arrays = {**arrays, "blt_dct": blt}
    pack = ReplayPack(arrays, meta)
    G = _pgse(0.0, 10e-3, 30e-3)[None]                                  # b0
    W = compile_scheme(G, DT, K, GAMMA, n_t=N_T)
    E0 = replay_signal(pack, W)[0]
    Er = replay_signal(pack, W, rho_over_D=0.05)[0]                     # modest rho/D -> O(1) log-weight
    assert E0 == pytest.approx(1.0, abs=1e-9)                           # b0, no surface -> 1
    assert Er < 1.0                                                     # surface decays even b0
