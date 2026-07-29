"""Tests for dmipy_sim.compression — the master-walk compression algorithm.

Reproduces, on a small CPU walk, the core claims:
  * every method encodes -> decodes to the right shape;
  * low-rank at full rank is exact; temporal_dct is a band-limiter;
  * on free (Gaussian) diffusion the distributional model reaches the noise floor;
  * RLE / per-channel codecs round-trip;
  * auto_select_modes returns a K that meets the fidelity tolerance;
  * mode-space gradient replay == decode-then-contract for every position codec.
All CPU-only, small and fast.
"""
import numpy as np
import pytest

from dmipy_sim import simulate_trajectories
from dmipy_sim.geometries import FreeDiffusion
from dmipy_sim import compression as cx


@pytest.fixture(scope="module")
def free_master():
    """A small free-diffusion master walk (Gaussian paths) + a single-compartment
    relaxation channel, as the compression campaign's Gaussian anchor."""
    tr, dt, sub, *_ = simulate_trajectories(
        n_walkers=3000, diffusivity=2e-9, geometry=FreeDiffusion(),
        T_max=20e-3, dt_save=20e-3 / 48, seed=0, require_gpu=False)
    tr = np.asarray(tr, np.float64)
    return dict(traj=tr, dt_traj=float(dt), T_max=20e-3,
                comp=np.zeros(tr.shape[:2], np.int8), comp0=np.zeros(tr.shape[0], np.int64),
                w=np.ones(tr.shape[0]), T2_per_comp=np.array([0.05]),
                T1_per_comp=np.array([1.0]), n_walkers=tr.shape[0], seed=0)


@pytest.mark.parametrize("method", cx.ALL_METHODS)
def test_encode_decode_shapes(free_master, method):
    X = free_master["traj"]
    arrays, meta, nbytes = cx.encode(X, method, K=16)
    nw = X.shape[0] if cx.is_walker_preserving(method) else None
    pos = cx.decode(arrays, meta, n_walkers=nw)
    assert pos.shape[1:] == X.shape[1:]
    assert nbytes > 0 and np.isfinite(pos).all()


def test_lowrank_exact_at_full_rank(free_master):
    X = free_master["traj"][:200]
    K = min(X.shape[0], 3 * X.shape[1])
    arrays, meta, _ = cx.encode(X, "lowrank", K=K)
    pos = cx.decode(arrays, meta)
    assert np.abs(pos - X).max() < 1e-3          # half-precision coeffs -> ~1e-3


def test_temporal_dct_is_bandlimited():
    # a path with a slow ramp + a fast oscillation; keeping few DCT bands must keep the
    # ramp and drop the oscillation.
    Nt = 128; t = np.linspace(0, 1, Nt)
    slow = t
    fast = 0.3 * np.sin(2 * np.pi * 30 * t)
    X = np.stack([slow + fast] * 3, axis=-1)[None]           # (1,Nt,3)
    arrays, meta, _ = cx.encode(X, "temporal_dct", K=6)
    rec = cx.decode(arrays, meta)[0, :, 0]
    assert np.abs(rec - slow).max() < 0.05                   # slow kept
    assert np.abs(rec - (slow + fast)).max() > 0.1           # fast dropped


def test_gaussian_reaches_floor_on_free(free_master):
    X = free_master["traj"]
    env = cx.default_envelope(); env["ogse_periods"] = [1, 2]; env["B0_list"] = []
    arrays, meta, _ = cx.encode(X, "gaussian", K=32)
    pos = cx.decode(arrays, meta, n_walkers=X.shape[0])
    fid = cx.measure_fidelity(free_master, pos, env)
    # Gaussian model on Gaussian diffusion: within a small multiple of the MC floor
    assert fid["err_max"] <= 4 * fid["floor_max"] + 5e-3


def test_rle_roundtrip():
    A = np.array([[0, 0, 0, 1, 1, 2, 2, 2],
                  [1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 1, 0, 1, 0, 1, 0, 1]], np.int8)
    v, l, c, nt = cx.rle_encode_rows(A)
    B = cx.rle_decode_rows(v, l, c, nt)
    assert np.array_equal(A, B)
    assert c.tolist() == [3, 1, 8]               # runs per row


def test_auto_select_modes(free_master):
    X = free_master["traj"]
    env = cx.default_envelope(); env["ogse_periods"] = [1, 2]; env["B0_list"] = []
    K, fid = cx.auto_select_modes(X, free_master, method="lowrank", env=env, tol=3.0,
                                  K_grid=(8, 16, 32, 64))
    assert K in (8, 16, 32, 64)
    assert "err_max" in fid and "per_family" in fid
    assert fid["err_max"] <= 3 * fid["floor_max"] + 5e-3


def test_bound_fraction_codec():
    """Quantized-RLE round-trip for the MT occupancy channel: ~binary with long runs."""
    rng = np.random.default_rng(0)
    # synthetic occupancy: mostly-0 with occasional bound dwells (runs of 1) + a few fractionals
    bf = np.zeros((200, 120), np.float32)
    for i in range(200):
        t = 0
        while t < 120:
            gap = rng.integers(5, 40); t += gap
            if t >= 120: break
            dur = rng.integers(3, 15)
            bf[i, t:t+dur] = 1.0
            if t + dur < 120: bf[i, t+dur] = rng.uniform(0, 1)   # fractional release save
            t += dur
    arr, meta = cx.encode_bound_fraction(bf, Q=256)
    dec = cx.decode_bound_fraction(arr, meta)
    assert dec.shape == bf.shape
    assert np.abs(dec - bf).max() <= 1.0 / 255 + 1e-6          # lossless to the quant step
    raw = bf.size * 2
    comp = sum(arr[k].nbytes for k in arr)
    assert comp < raw / 3                                       # meaningful compression


def test_boundary_local_time_codec_density_aware():
    """Surface channel codec is density-aware: sparse for isolated fibres (~15% nonzero),
    dense int8 for packed WM (~55%), each within the value-quant step and beating raw f16."""
    rng = np.random.default_rng(1)
    raw_bytes = 200 * 120 * 2                                   # dense float16 reference
    for dens, want_mode in [(0.15, "sparse"), (0.55, "dense")]:
        dlog = np.zeros((200, 120), np.float32)
        mask = rng.random((200, 120)) < dens
        dlog[mask] = -rng.random(mask.sum()).astype(np.float32) * 1e-6   # negative (<=0)
        arr, meta = cx.encode_boundary_local_time(dlog)
        assert meta["mode"] == want_mode, f"density {dens} -> {meta['mode']} (want {want_mode})"
        dec = cx.decode_boundary_local_time(arr, meta)
        assert dec.shape == dlog.shape
        step = meta["scale"] / meta["nlevels"]
        assert np.abs(dec - dlog).max() <= step + 1e-12         # lossless to the quant step
        comp = sum(arr[k].nbytes for k in arr)
        assert comp < raw_bytes                                 # both beat raw float16


def test_compartment_codec_integer_and_fractional():
    """Compartment codec: integer labels -> lossless; fractional occupancy (permeable,
    [0,1], near-binary) -> quantized-RLE, faithful (unlike a lossy integer cast)."""
    # integer (impermeable): exact round-trip
    ci = np.array([[0, 0, 0, 1, 1, 1, 1, 2, 2, 2]] * 50, np.int16)
    a, m = cx.encode_compartment(ci)
    assert m["fractional"] is False
    assert np.array_equal(cx.decode_compartment(a, m).astype(np.int16), ci)

    # fractional (permeable 2-compartment occupancy): near-binary with fractional crossings
    rng = np.random.default_rng(0)
    cf = np.zeros((80, 120), np.float32)
    for i in range(80):
        t = 0
        while t < 120:
            t += rng.integers(4, 30)
            if t >= 120: break
            dur = rng.integers(3, 20); cf[i, t:t+dur] = 1.0
            if t + dur < 120: cf[i, t+dur] = rng.uniform(0, 1)  # crossing save
            t += dur
    # integer RLE would int-cast the fractions -> lossy; the codec must route to quantized RLE
    a, m = cx.encode_compartment(cf, Q=256)
    assert m["fractional"] is True
    dec = cx.decode_compartment(a, m)
    assert dec.shape == cf.shape and dec.dtype == np.float32
    assert np.abs(dec - cf).max() <= m["scale"] / 255 + 1e-6      # lossless to the quant step
    assert sum(a[k].nbytes for k in a) < cf.size * 2              # beats dense float16


def test_mode_space_all_codecs_match_dense():
    """Mode-space gradient replay == decode-then-contract for every position codec
    (lowrank, temporal_dct exactly; gaussian/marginal exact at a fixed sampling seed)."""
    from dmipy_sim.constants import GAMMA
    rng = np.random.default_rng(0); Nw, Nt = 500, 60; dt = 5e-4
    X = np.cumsum(rng.normal(0, 1e-6, (Nw, Nt, 3)), axis=1)
    G = (rng.normal(size=(4, Nt, 3)) * 0.05)
    def dense(a, m):
        r = cx.decode(a, m, n_walkers=Nw, seed=0)
        phi = GAMMA * dt * np.einsum("ntd,mtd->mn", r, G)
        return np.exp(1j * phi).mean(1)
    for method in ("lowrank", "temporal_dct", "gaussian", "marginal"):
        a, m, _ = cx.encode(X, method, K=20)
        S_fast = cx.replay_gradient_lowrank(a, m, G, dt, n_walkers=Nw, seed=0)
        assert np.abs(S_fast - dense(a, m)).max() < 1e-9, method   # identity, machine precision
