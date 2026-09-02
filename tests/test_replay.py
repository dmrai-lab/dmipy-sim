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


def test_surface_knob_uses_the_real_c2_channel_and_refuses_without_one():
    """The rho knob must read the pack's ACTUAL C2 channel, not a key that no longer exists.

    The previous version of this test fabricated a `blt_dct` array and injected it, so it kept
    passing after C2 moved to the bridge form -- while `replay_signal` looked up the retired key,
    found nothing, and SILENTLY ignored rho_over_D. A caller asking for surface relaxivity got an
    unattenuated signal and no error. So: build the channel with the real encoder, and check both
    that the knob bites and that a pack without C2 refuses rather than skipping.
    """
    from dmipy_sim.compression import encode_boundary_bridge, decode_boundary_bridge, surface_logweight
    from dmipy_sim.replay import surface_logweight as replay_slw
    arrays, meta, _ = _synth_pack()
    n_w = arrays["pos_x"].shape[0]
    rng = np.random.default_rng(1)
    dlog = -np.abs(rng.normal(0, 1e-6, (n_w, N_T)))          # engine convention: <= 0
    a2, cm = encode_boundary_bridge(dlog, K=16)
    arrays = {**arrays, **a2}
    pack = ReplayPack(arrays, meta)

    G = _pgse(0.0, 10e-3, 30e-3)[None]                        # b0
    W = compile_scheme(G, DT, K, GAMMA, n_t=N_T)
    E0 = replay_signal(pack, W)[0]
    Er = replay_signal(pack, W, rho_over_D=5e3)[0]
    assert E0 == pytest.approx(1.0, abs=1e-9)
    assert Er < 0.99 * E0, f"rho knob did not bite: {Er} vs {E0}"

    # the endpoint path must equal summing the decoded per-save series
    npt.assert_allclose(replay_slw(arrays, 5e3, cm),
                        surface_logweight(decode_boundary_bridge(arrays, cm), 5e3), rtol=1e-6)

    # a pack with no C2 must RAISE when rho is requested, not silently return the bare signal
    bare = {k: v for k, v in arrays.items() if not k.startswith("blt_")}
    with pytest.raises(ValueError, match="no C2 channel"):
        replay_signal(bare, W, rho_over_D=5e3)


def test_blt_dct_attribute_is_retired_loudly():
    arrays, meta, _ = _synth_pack()
    with pytest.raises(AttributeError, match="retired"):
        ReplayPack(arrays, meta).blt_dct
