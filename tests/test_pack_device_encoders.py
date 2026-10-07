"""The pack's bridge encoders on the device (dmipy-sim-cuda#33, the pack stage): the bridge and the cumulative local time
formed on the device per chunk give the numpy route's coefficients to float32 rounding, and the endpoints exactly."""
import numpy as np

from dmipy_sim.replay.compression import encode_bridge_dst, encode_boundary_bridge


def test_the_device_bridge_matches_the_numpy_route():
    rng = np.random.default_rng(3)
    X = np.cumsum(rng.normal(size=(300, 257, 3)).astype(np.float32) * 1e-7, axis=1) + 5e-6
    a_np, m_np, _ = encode_bridge_dst(X, 32, device="numpy")
    a_dv, m_dv, _ = encode_bridge_dst(X, 32, device="jax")
    assert m_np == m_dv
    coeffs = lambda a: np.stack([np.asarray(a[k], np.float64) for k in ("pos_x", "pos_y", "pos_z")], axis=-1)   # (n_w, K+2, 3)
    C_np = coeffs(a_np); C_dv = coeffs(a_dv)
    np.testing.assert_array_equal(C_np[:, :2], C_dv[:, :2])                 # the endpoints, exact
    scale = np.abs(C_np[:, 2:]).max()
    assert np.abs(C_np[:, 2:] - C_dv[:, 2:]).max() < 2e-6 * scale            # float32 bands against float64


def test_the_device_local_time_matches_the_numpy_route():
    rng = np.random.default_rng(4)
    dlog = -(rng.random((300, 257)) < 0.2).astype(np.float32) * rng.random((300, 257)).astype(np.float32) * 1e-3
    a_np, m_np = encode_boundary_bridge(dlog, 16, device="numpy")
    a_dv, m_dv = encode_boundary_bridge(dlog, 16, device="jax")
    assert m_np == m_dv
    np.testing.assert_allclose(a_dv["blt_endpoint"], a_np["blt_endpoint"], rtol=2e-6)
    np.testing.assert_array_equal(a_dv["blt_start"], a_np["blt_start"])
    scale = np.abs(a_np["blt_bridge_dst"].astype(np.float64)).max()
    assert np.abs(a_dv["blt_bridge_dst"].astype(np.float64) - a_np["blt_bridge_dst"].astype(np.float64)).max() < 1e-3 * scale   # f16 bands
