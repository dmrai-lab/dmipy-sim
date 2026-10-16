"""The far crop as a shell-quantized truncated spectrum: smooth fields round-trip within the quantization's derived
bound, a short wavelength is dropped and nothing else, iso_local must be zero, an empty shell round-trips, and the
kept fraction at 10 um on a 2.5 um grid is the ball of radius pi / (2 h) in the Nyquist box."""
import numpy as np
import pytest

from dmipy_sim.fill.farspec import FIELD_CHANNELS, N_SHELLS, _layout, pack_far, save_far, unpack_far

H = 2.5e-6
N = (24, 20, 16)
XX, YY, ZZ, LOC = (FIELD_CHANNELS.index(c) for c in ("iso_P_xx", "iso_P_yy", "iso_P_zz", "iso_local"))


def _mode(n, m):
    return np.cos(2 * np.pi * m * np.arange(n) / n)


def _smooth(m_list, shape=N):
    """A 13-channel field of a few long modes plus a per-channel offset, with iso_local = 0 and iso_P_zz = -(xx + yy)."""
    nx, ny, nz = shape
    v = np.zeros(shape + (13,))
    for c in range(13):
        for (a, b, d) in m_list:
            v[..., c] += (c + 1) * 0.1 * _mode(nx, a)[:, None, None] * _mode(ny, b)[None, :, None] * _mode(nz, d)[None, None, :]
        v[..., c] += 0.5 * c
    v[..., LOC] = 0
    v[..., ZZ] = -(v[..., XX] + v[..., YY])
    return v.astype(np.float16)


def _bound(p):
    """The derived bound on |unpacked - exact| per stored channel: every kept coefficient's re and im are off by at
    most scale / (2 Q) (Q = 32767 or 127), each costs at most 2 / N of a node's value (the half spectrum counts a
    mode twice), so the field is off by (2 / N) sum_s n_s scale_s / Q_s."""
    shape = tuple(int(s) for s in p["shape"])
    _, shell, n16 = _layout(shape, float(p["spacing_m"]), float(p["k_cut"]), int(p["n_shells"]), float(p["k16"]))
    Q = np.where(np.arange(shell.size) < n16, 32767.0, 127.0)
    return 2.0 / np.prod(shape) * (p["scale"][shell].astype(np.float64) / Q[:, None]).sum(axis=0)


def test_smooth_field_round_trips_within_the_quantization_bound():
    v = _smooth([(1, 0, 0), (0, 1, 1), (1, 1, 1)])
    p = pack_far(v, H)
    assert p["mean"].shape == (11,) and p["q16"].dtype == np.int16 and p["q8"].dtype == np.int8 and p["scale"].shape == (N_SHELLS, 11)
    out = unpack_far(p)
    assert out.dtype == np.float16 and out.shape == v.shape
    b = _bound(p)
    err = np.abs(out.astype(np.float64) - v.astype(np.float64))
    stored = [FIELD_CHANNELS.index(c) for c in FIELD_CHANNELS if c not in ("iso_local", "iso_P_zz")]
    for j, ch in enumerate(stored):
        assert np.all(err[..., ch] <= b[j] + np.spacing(np.abs(v[..., ch]).max()).astype(np.float64) * 1.5), ch
    assert np.all(out[..., LOC] == 0)
    np.testing.assert_allclose(out[..., ZZ].astype(np.float64), -(out[..., XX].astype(np.float64) + out[..., YY].astype(np.float64)),
                               atol=0.1, rtol=2e-3)       # rebuilt in float32, then cast: differs by the cast of the sum


def test_a_short_wavelength_component_is_lost_alone():
    smooth = _smooth([(1, 1, 0)]).astype(np.float64)
    nx = N[0]
    short = 0.3 * np.cos(2 * np.pi * (nx // 2) * np.arange(nx) / nx)[:, None, None, None] * np.ones(N + (13,))   # 2 nodes = 5 um
    short[..., LOC] = 0; short[..., ZZ] = -(short[..., XX] + short[..., YY])
    out = unpack_far(pack_far((smooth + short).astype(np.float32), H)).astype(np.float64)
    np.testing.assert_allclose(out, smooth, rtol=3e-3, atol=5e-3)


def test_a_nonzero_iso_local_is_refused_by_name():
    v = _smooth([(1, 0, 0)]).astype(np.float32); v[3, 3, 3, LOC] = 1e-3
    with pytest.raises(ValueError, match="iso_local"):
        pack_far(v, H)


def test_an_empty_shell_has_scale_zero_and_round_trips():
    v = _smooth([(1, 0, 0)])            # one low mode: the high shells are empty of power
    p = pack_far(v, H)
    assert (p["scale"][-1] < 1e-3 * p["scale"].max()).all()
    z = pack_far(np.zeros(N + (13,), np.float16), H)
    assert (z["scale"] == 0).all() and not z["q16"].any() and not z["q8"].any()
    assert np.array_equal(unpack_far(z), np.zeros(N + (13,), np.float16))


def test_kept_fraction_at_10um_on_2p5um_grid():
    nx, ny, nz = 40, 40, 40
    index, _, _ = _layout((nx, ny, nz), H, 2 * np.pi / 10e-6, N_SHELLS, 2 * np.pi / 20e-6)
    frac = index.size / (nx * ny * (nz // 2 + 1))
    assert 0.05 < frac < 0.09
    print(f"kept fraction at 10 um on a 2.5 um grid: {frac:.4f}")


def test_npz_round_trip_and_determinism(tmp_path):
    p = pack_far(_smooth([(1, 0, 1)]), H)
    a, b = save_far(tmp_path / "a.npz", p), save_far(tmp_path / "b.npz", p)
    assert open(a, "rb").read() == open(b, "rb").read()
    assert np.array_equal(unpack_far(np.load(a)), unpack_far(p))
