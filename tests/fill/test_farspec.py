"""The far crop as a truncated spectrum: smooth fields round-trip to float16, a short wavelength is dropped and
nothing else, and the kept fraction at 10 um on a 2.5 um grid is the ball of radius pi / (2 h) in the Nyquist box."""
import numpy as np

from dmipy_sim.fill.farspec import pack_far, save_far, unpack_far

H = 2.5e-6
N = (24, 20, 16)


def _periodic_mode(n, m, axis_len, c0=0.5):
    """A mode of integer period count m on an axis of n nodes."""
    return np.cos(2 * np.pi * m * np.arange(n) / n)


def _smooth(m_list):
    nx, ny, nz = N
    v = np.zeros(N + (13,))
    for c in range(13):
        for (a, b, d) in m_list:
            v[..., c] += (c + 1) * 0.1 * _periodic_mode(nx, a, nx)[:, None, None] * _periodic_mode(ny, b, ny)[None, :, None] * _periodic_mode(nz, d, nz)[None, None, :]
        v[..., c] += 3.0 * c
    return v


def test_smooth_field_round_trips_to_float16():
    v = _smooth([(1, 0, 0), (0, 1, 1), (1, 1, 1)]).astype(np.float16)
    p = pack_far(v, H)
    assert p["mean"].dtype == np.float32 and p["coef"].dtype == np.complex64 and p["index"].dtype == np.int64
    out = unpack_far(p)
    assert out.dtype == np.float16 and out.shape == v.shape
    # the spectrum is complete, so only float32 rounding remains: at most one float16 step of the value (or of 0.25 near zero)
    err = np.abs(out.astype(np.float64) - v.astype(np.float64))
    assert np.all(err <= np.spacing(np.maximum(np.abs(v), np.float16(0.25))).astype(np.float64))
    assert np.mean(out == v) > 0.9


def test_a_short_wavelength_component_is_lost_alone():
    smooth = _smooth([(1, 1, 0)])
    nx = N[0]
    short = 0.3 * np.cos(2 * np.pi * (nx // 2) * np.arange(nx) / nx)[:, None, None, None] * np.ones(N + (13,))   # 2 nodes = 5 um
    out = unpack_far(pack_far((smooth + short).astype(np.float32), H)).astype(np.float64)
    np.testing.assert_allclose(out, smooth, rtol=1.5e-3, atol=2e-3)


def test_kept_fraction_at_10um_on_2p5um_grid():
    nx, ny, nz = 40, 40, 40
    p = pack_far(np.zeros((nx, ny, nz, 13), np.float16), H)
    # half spectrum of nz//2 + 1 planes; full-ball fraction (4/3 pi (1/2)^3 / 8) of the Nyquist box
    frac = p["index"].size / (nx * ny * (nz // 2 + 1))
    assert 0.05 < frac < 0.09
    print(f"kept fraction at 10 um on a 2.5 um grid: {frac:.4f}")


def test_npz_round_trip_and_determinism(tmp_path):
    p = pack_far(_smooth([(1, 0, 1)]).astype(np.float16), H)
    a, b = save_far(tmp_path / "a.npz", p), save_far(tmp_path / "b.npz", p)
    assert open(a, "rb").read() == open(b, "rb").read()
    loaded = np.load(a)
    assert np.array_equal(unpack_far(loaded), unpack_far(p))
