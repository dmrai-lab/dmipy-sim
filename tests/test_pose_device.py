"""The pose expansion's device kernels against their numpy oracles (dmrai-lab/dmipy-sim#449)."""
import numpy as np

from dmipy_sim.replay import so3
from dmipy_sim.replay.pose_device import field_factor


def _field(n_w=3000, seed=0, amp=4.0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-3.0, 3.0, n_w)
    M = rng.normal(size=(n_w, 3, 3)); A = amp * (M + np.swapaxes(M, 1, 2)) / 2.0
    return a, A


def test_the_field_factor_on_the_device_is_the_numpy_one_to_float32_rounding():
    a, A = _field()
    Lp = 12
    dirs, wq = so3.sphere_quadrature(Lp + 2, 2 * Lp + 2)
    Yw = so3.real_sh(Lp, dirs, full=True) * wq[:, None]
    ref = field_factor(a, A, dirs, Yw, device="numpy")
    dev = field_factor(a, A, dirs, Yw, device="jax", chunk_bytes=1 << 22)         # several chunks
    assert dev.shape == ref.shape
    scale = np.abs(ref).max()
    assert np.abs(dev - ref).max() <= 2e-5 * scale, np.abs(dev - ref).max() / scale


def test_the_numpy_route_is_chunk_independent():
    a, A = _field(n_w=700)
    Lp = 6
    dirs, wq = so3.sphere_quadrature(Lp + 2, 2 * Lp + 2)
    Yw = so3.real_sh(Lp, dirs, full=True) * wq[:, None]
    whole = field_factor(a, A, dirs, Yw, device="numpy")
    parts = field_factor(a, A, dirs, Yw, device="numpy", chunk_bytes=1 << 16)
    assert np.array_equal(whole, parts)


def test_the_field_products_on_the_device_are_the_numpy_ones_to_float32_rounding():
    from dmipy_sim.replay.pose_device import field_products
    rng = np.random.default_rng(1)
    n_w = 5000
    X = rng.normal(size=(n_w, 30)) / n_w
    F = rng.normal(size=(n_w, 49)) + 1j * rng.normal(size=(n_w, 49))
    ref = field_products(X, F.real, F.imag, device="numpy")
    dev = field_products(X, F.real, F.imag, device="jax", chunk_bytes=1 << 18)         # several chunks
    assert np.abs(dev - ref).max() <= 1e-6 * np.abs(ref).max()
