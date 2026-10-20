"""The host encoders' device route (``device="jax"``, :func:`compression._device_bridge`) forms the bridge of a walker
chunk on the device and gives the numpy route's coefficients to float32 rounding."""
import numpy as np

from dmipy_sim.replay import compression as cx


def _batch(b=400, n_t=300, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(size=(b, n_t, 3)).astype(np.float32) * np.float32(2e-7), axis=1)
    dlog = -(rng.exponential(1e-3, size=(b, n_t)).astype(np.float32) * (rng.uniform(size=(b, n_t)) < 0.2))
    return pos, dlog.astype(np.float32)


def test_position_coefficients_match_the_host_encoder():
    pos, _ = _batch()
    C_host = cx.read_position_coeffs(cx.encode_bridge_dst(pos, 16, 1e-4, device="numpy")[0], dtype=np.float32)
    C_dev = cx.read_position_coeffs(cx.encode_bridge_dst(pos, 16, 1e-4, device="jax")[0], dtype=np.float32)
    assert C_dev.shape == C_host.shape
    assert np.array_equal(C_dev[:, :1], C_host[:, :1])                          # the start is exact
    assert np.abs(C_dev[:, 1] - C_host[:, 1]).max() <= 1e-6 * np.abs(C_host[:, 1]).max()
    scale = np.abs(C_host[:, 2:]).max(axis=(0, 1))
    assert (np.abs(C_dev[:, 2:] - C_host[:, 2:]).max(axis=(0, 1)) <= 1e-6 * scale).all()


def test_boundary_coefficients_match_the_host_encoder():
    _, dlog = _batch()
    host, _ = cx.encode_boundary_bridge(dlog, 16, 1e-4, device="numpy")
    dev, _ = cx.encode_boundary_bridge(dlog, 16, 1e-4, device="jax")
    assert np.array_equal(dev["blt_start"], host["blt_start"])
    assert np.array_equal(dev["blt_endpoint"], host["blt_endpoint"])          # the endpoints are float64 on the host either way
    assert np.abs(dev["blt_bridge_dst"] - host["blt_bridge_dst"]).max() <= 1e-5 * np.abs(host["blt_bridge_dst"]).max()


def test_the_band_count_is_capped_by_the_saves():
    pos, dlog = _batch(n_t=12)
    arrays, meta, _ = cx.encode_bridge_dst(pos, 64, 1e-4, device="jax")
    assert meta["K"] == 10 and cx.read_position_coeffs(arrays).shape == (400, 12, 3)
    assert cx.encode_boundary_bridge(dlog, 64, 1e-4, device="jax")[0]["blt_bridge_dst"].shape == (400, 10)
