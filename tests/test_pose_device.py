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
    # a BLAS may block a product differently by its row count, so the chunks agree to rounding, not to the bit
    assert np.abs(whole - parts).max() <= 1e-12 * np.abs(whole).max()


def test_the_field_products_on_the_device_are_the_numpy_ones_to_float32_rounding():
    from dmipy_sim.replay.pose_device import field_products
    rng = np.random.default_rng(1)
    n_w = 5000
    X = rng.normal(size=(n_w, 30)) / n_w
    F = rng.normal(size=(n_w, 49)) + 1j * rng.normal(size=(n_w, 49))
    ref = field_products(X, F.real, F.imag, device="numpy")
    dev = field_products(X, F.real, F.imag, device="jax", chunk_bytes=1 << 18)         # several chunks
    assert np.abs(dev - ref).max() <= 1e-6 * np.abs(ref).max()


def _samples_case(n_w=2000, n_meas=3, n_R=50, seed=2):
    from dmipy_sim.replay import so3
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(n_w, n_meas, 3, 3)) * 1.5
    ew = rng.uniform(0.5, 1.5, n_w)
    R = so3.haar_rotations(n_R, seed)
    field = (0.7, 0.2, np.array([0.1, 0.3, 0.9]), rng.normal(size=n_w), rng.normal(size=(n_w, 6)), rng.normal(size=(n_w, 6)))
    return R, Q, ew, field


def test_pose_samples_on_the_device_are_the_numpy_ones_to_float32_rounding():
    from dmipy_sim.replay.pose_device import pose_samples
    R, Q, ew, field = _samples_case()
    for f in (None, field):
        ref = pose_samples(R, Q, ew, ew.sum(), field=f, device="numpy")
        dev = pose_samples(R, Q, ew, ew.sum(), field=f, device="jax", chunk_bytes=1 << 18)   # several chunks both ways
        assert np.abs(dev - ref).max() <= 3e-6, np.abs(dev - ref).max()


def test_pose_samples_numpy_route_is_chunk_independent():
    from dmipy_sim.replay.pose_device import pose_samples
    R, Q, ew, field = _samples_case(n_w=400, n_R=20)
    whole = pose_samples(R, Q, ew, ew.sum(), field=field, device="numpy")
    parts = pose_samples(R, Q, ew, ew.sum(), field=field, device="numpy", chunk_bytes=1 << 14)
    assert np.abs(whole - parts).max() <= 1e-13


def test_the_harmonics_on_the_device_are_the_numpy_ones_to_float32_rounding():
    from dmipy_sim.replay import so3
    from dmipy_sim.replay.pose_device import real_sh
    rng = np.random.default_rng(4)
    d = rng.normal(size=(3000, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    for L in (0, 3, 15, 28):
        ref = so3.real_sh(L, d, full=True)
        dev = real_sh(L, d, device="jax", chunk_bytes=1 << 16)
        assert dev.shape == ref.shape
        assert np.abs(dev - ref).max() <= 3e-5 * np.abs(ref).max(), (L, np.abs(dev - ref).max())


def test_the_bessel_values_on_the_device_are_the_numpy_ones_to_float32_rounding():
    from dmipy_sim.replay.replay import _spherical_jn_all
    from dmipy_sim.replay.pose_device import spherical_jn_all
    rng = np.random.default_rng(5)
    x = rng.uniform(0.0, 30.0, size=(700, 3)); x[0, 0] = 0.0; x[1, 1] = 1e-9
    for L in (2, 12, 30):
        ref = _spherical_jn_all(L, x)
        dev = spherical_jn_all(L, x, device="jax", chunk_bytes=1 << 14)
        assert dev.shape == ref.shape
        # a float32 recurrence over up to 60 orders at 30 rad: 1e-5 absolute, three orders under any response floor
        assert np.abs(dev - ref).max() <= 2e-5, (L, np.abs(dev - ref).max())


def test_the_field_bodies_on_the_device_are_the_numpy_ones_to_float32_rounding():
    from dmipy_sim.replay.pose_device import field_bodies
    rng = np.random.default_rng(6)
    n_w, nc, L, n_f = 3000, 3, 9, 25
    kappa = rng.uniform(0.0, 9.0, size=(n_w, nc))
    m = rng.normal(size=(n_w, nc, 3)); m_hat = m / np.linalg.norm(m, axis=2, keepdims=True)
    w = rng.uniform(0.5, 1.5, n_w) / n_w
    F = rng.normal(size=(n_w, n_f)) + 1j * rng.normal(size=(n_w, n_f))
    l_used = tuple(range(L + 1))
    ref = field_bodies(kappa, m_hat, w, F.real, F.imag, L, l_used, n_bessel=L + 24 + 9, device="numpy")
    dev = field_bodies(kappa, m_hat, w, F.real, F.imag, L, l_used, n_bessel=L + 24 + 9, device="jax", chunk_bytes=1 << 20)
    assert dev.shape == ref.shape == (nc * (L + 1) ** 2, n_f)
    assert np.abs(dev - ref).max() <= 1e-5 * np.abs(ref).max(), np.abs(dev - ref).max() / np.abs(ref).max()
