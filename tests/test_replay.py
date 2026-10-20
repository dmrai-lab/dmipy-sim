"""Replay-pack forward: the compiled-scheme replay reproduces the direct phase integral exactly, the
.rpk round-trips, the JAX twin matches and is differentiable in the waveform, and the surface knob
attenuates. Uses a synthetic pack (no simulator walk needed), so it is self-contained."""
import numpy as np
import numpy.testing as npt
from dmipy_sim.replay.compression import read_position_coeffs, pack_position_arrays
import pytest

from scipy.fft import dct, idct
from dmipy_sim.replay.replay import (read_rpk, write_rpk, compile_scheme, replay_signal,
                              replay_signal_jax, ReplayPack)
from dmipy_sim.constants import GAMMA
from dmipy_sim.geometry import FreeDiffusion
from dmipy_sim.spec import spec_of

N_W, N_T, K = 400, 200, 48
DT = 5e-4


def _synth_pack(seed=0):
    """A synthetic pack: random bridge coefficients ``[r(0), r(T) - r(0), beta_1 .. beta_K]`` per axis for a window
    of ``(N_T - 1)`` blocks of ``DT``, the bands decaying like the bridge of a diffusing path (variance ~ 1 / k^2)."""
    rng = np.random.default_rng(seed)
    T = (N_T - 1) * DT
    C = np.empty((N_W, K + 2, 3))
    C[:, 0, :] = rng.normal(0, 3e-6, (N_W, 3))
    C[:, 1, :] = rng.normal(0, 3e-6, (N_W, 3))
    C[:, 2:, :] = rng.normal(0, 1.0, (N_W, K, 3)) * 2e-6 / np.arange(1, K + 1)[None, :, None]
    arrays = pack_position_arrays(C, np.float32)                  # one tensor per spatial axis
    arrays["spin_weights"] = np.ones(N_W, np.float32)
    meta = {"walk_params": {"T_max": T, "block": {"h": DT, "m": 0, "n_blocks": N_T - 1},
                            "windows": {"n": 1, "T": T, "walks": [{"first": 0, "last": 0, "seed": 0}]}},
            "compression": {"method": "bridge_dst", "K": K},
            "substrate": spec_of(FreeDiffusion()).to_dict()}                # free walkers: nothing magnetic
    return arrays, meta, C


def _pgse(amp, delta, Delta, direction=(1., 0, 0)):
    g = np.zeros((N_T, 3)); nd = max(1, int(round(delta / DT))); ng = int(round(Delta / DT))
    u = np.asarray(direction, float); g[:nd] = amp * u; g[ng:ng + nd] = -amp * u
    return g


def test_replay_equals_direct_phase():
    """Compiled replay == the integral of the held waveform against the stored path (the chord plus the bands'
    series), by a midpoint rule fine enough to see it."""
    arrays, meta, _ = _synth_pack()
    pack = ReplayPack(arrays, meta)
    G = np.stack([_pgse(a, 10e-3, 30e-3) for a in (0.02, 0.05, 0.1)])   # (3, N_T, 3)
    T = (N_T - 1) * DT
    W = compile_scheme(G, DT, K, GAMMA, T=T)
    E = replay_signal(pack, W)
    C = read_position_coeffs(arrays, dtype=np.float64)
    sub = 64
    t = (np.arange((N_T - 1) * sub) + 0.5) * DT / sub                   # the held samples' pieces, finely
    k = np.arange(1, K + 1)
    r = (C[:, None, 0, :] + C[:, None, 1, :] * (t / T)[None, :, None]
         + np.einsum("wkc,kt->wtc", C[:, 2:, :], np.sqrt(2.0 / T) * np.sin(np.pi * np.outer(k, t) / T)))
    g = G[:, np.minimum((t / DT).astype(int), N_T - 1), :]
    phi = GAMMA * (DT / sub) * np.einsum("mtc,wtc->mw", g, r)
    npt.assert_allclose(E, np.abs(np.exp(1j * phi).mean(1)), atol=1e-6)


def test_rpk_roundtrip(tmp_path):
    arrays, meta, _ = _synth_pack()
    p = tmp_path / "synth.rpk"
    write_rpk(p, arrays, meta)
    pk = read_rpk(p)
    assert pk.n_t == N_T and pk.K == K and pk.n_walkers == N_W
    npt.assert_allclose(pk.dt, DT)
    from dmipy_sim.replay.compression import read_position_coeffs
    npt.assert_allclose(np.asarray(pk.position_coeffs), read_position_coeffs(arrays, dtype=np.float32))


def test_jax_twin_matches_and_differentiable():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    arrays, meta, _ = _synth_pack()
    G = np.stack([_pgse(a, 10e-3, 30e-3) for a in (0.03, 0.08)])
    W = compile_scheme(G, DT, K, GAMMA)
    E_np = replay_signal(arrays, W)
    C3 = read_position_coeffs(arrays, dtype=np.float32)
    E_jx = np.abs(np.asarray(replay_signal_jax(C3, arrays["spin_weights"], W)))
    npt.assert_allclose(E_jx, E_np, atol=1e-4)                         # float32 phases of ~1 rad
    # differentiable in the compiled scheme (hence in the waveform): grad is finite
    def loss(Wj):
        return jnp.abs(replay_signal_jax(jnp.asarray(read_position_coeffs(arrays, dtype=np.float32)),
                                         jnp.asarray(arrays["spin_weights"]), Wj)).sum()
    g = jax.grad(loss)(jnp.asarray(W))
    assert np.all(np.isfinite(np.asarray(g))) and np.abs(np.asarray(g)).max() > 0


def test_surface_knob_uses_the_real_c2_channel_and_refuses_without_one():
    """The rho2 knob must read the pack's ACTUAL C2 channel, built with the real encoder (not a
    fabricated array under some other key, which would let the knob miss silently): both that the
    knob bites on a pack that carries the channel, and that a pack without one refuses rather than
    skipping -- a caller asking for surface relaxivity must never get an unattenuated signal back
    with no error.
    """
    from dmipy_sim.replay.compression import (encode_boundary_bands, decode_boundary_bridge,
                                       surface_logweight_series as surface_logweight)
    from dmipy_sim.replay.replay import surface_logweight as replay_slw
    arrays, meta, _ = _synth_pack()
    n_w = arrays["pos_x"].shape[0]
    rng = np.random.default_rng(1)
    T = (N_T - 1) * DT
    dlog = -np.abs(rng.normal(0, 1e-6, (n_w, N_T)))          # engine convention: <= 0
    # the contact channel in its stored form: the exact endpoints and the continuous bands of the cumulative contact
    a2, cm = encode_boundary_bands(np.zeros(n_w), dlog.sum(1), rng.normal(0, 1e-7, (n_w, 16)), T=T)
    cm["K"] = 16
    arrays = {**arrays, **a2}
    pack = ReplayPack(arrays, meta)

    G = _pgse(0.0, 10e-3, 30e-3)[None]                        # b0
    W = compile_scheme(G, DT, K, GAMMA)
    E0 = replay_signal(pack, W)[0]
    Er = replay_signal(pack, W, rho2_over_D=5e3)[0]
    assert E0 == pytest.approx(1.0, abs=1e-9)
    assert Er < 0.99 * E0, f"rho2 knob did not bite: {Er} vs {E0}"

    # the endpoint path must equal summing the decoded per-save series
    npt.assert_allclose(replay_slw(arrays, 5e3, cm),
                        surface_logweight(decode_boundary_bridge(arrays, cm, n_t=N_T), 5e3), rtol=1e-6)

    # a pack with no C2 must RAISE when rho2 is requested, not silently return the bare signal
    bare = {k: v for k, v in arrays.items() if not k.startswith("blt_")}
    with pytest.raises(ValueError, match="no C2 channel"):
        replay_signal(bare, W, rho2_over_D=5e3)


