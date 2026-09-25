"""The bridge of a walker batch is formed on the device it is on; only the coefficients cross to the host
(dmrai-lab/dmipy-sim#446). They are the host encoders' coefficients to float32 rounding."""
import jax.numpy as jnp
import numpy as np

from dmipy_sim.replay import compression as cx


def _batch(b=400, n_t=300, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(size=(b, n_t, 3)).astype(np.float32) * np.float32(2e-7), axis=1)
    dlog = -(rng.exponential(1e-3, size=(b, n_t)).astype(np.float32) * (rng.uniform(size=(b, n_t)) < 0.2))
    return pos, dlog.astype(np.float32)


def test_position_coefficients_match_the_host_encoder():
    pos, _ = _batch()
    C_host = cx.read_position_coeffs(cx.encode_bridge_dst(pos, 16, device="numpy")[0], dtype=np.float32)
    C_dev, K = cx.bridge_coefficients_device(jnp.asarray(pos), 16)
    assert K == 16 and C_dev.shape == C_host.shape and C_dev.dtype == np.float32
    assert np.array_equal(C_dev[:, :2], C_host[:, :2])                          # the endpoints are exact
    scale = np.abs(C_host[:, 2:]).max(axis=(0, 1))
    assert (np.abs(C_dev[:, 2:] - C_host[:, 2:]).max(axis=(0, 1)) <= 1e-6 * scale).all()


def test_boundary_coefficients_match_the_host_encoder():
    _, dlog = _batch()
    arrays, _ = cx.encode_boundary_bridge(dlog, 16, device="numpy")
    a, e, bands = cx.boundary_coefficients_device(jnp.asarray(dlog), 16)
    assert np.array_equal(a, arrays["blt_start"])
    assert np.abs(e - arrays["blt_endpoint"]).max() <= 2e-6 * np.abs(arrays["blt_endpoint"]).max()
    assert np.abs(bands - arrays["blt_bridge_dst"]).max() <= 1e-5 * np.abs(arrays["blt_bridge_dst"]).max()


def test_the_band_count_is_capped_by_the_saves():
    pos, dlog = _batch(n_t=12)
    C, K = cx.bridge_coefficients_device(jnp.asarray(pos), 64)
    assert K == 10 and C.shape == (400, 12, 3)
    assert cx.boundary_coefficients_device(jnp.asarray(dlog), 64)[2].shape == (400, 10)
