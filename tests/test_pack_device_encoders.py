"""The pack's device routes (dmipy-sim-cuda#33, the pack stage): the coded phases and the split-half floors read from a
pack's bridge coefficients on the device give the numpy route's to float32 rounding. The field series' DCT/quantisation
and the per-voxel census (dmipy-sim#685) follow the same pattern: a cached device matmul gives the numpy route's
coefficients to float32 rounding, and the census (a vectorised lookup, host-only) gives the retired per-walker loop's
assignment exactly."""
import numpy as np


def _coefficients(n_w, K, seed, *, start=5e-6):
    """Bridge coefficients ``[r(0), r(T) - r(0), beta_1 .. beta_K]`` per axis, ``(n_w, K + 2, 3)`` float64: a start
    about ``start``, a displacement and bands falling like the free bridge's (variance ~ 1 / k^2)."""
    rng = np.random.default_rng(seed)
    C = np.empty((n_w, K + 2, 3))
    C[:, 0, :] = start + rng.normal(0, 2e-6, (n_w, 3))
    C[:, 1, :] = rng.normal(0, 3e-7, (n_w, 3))
    C[:, 2:, :] = rng.normal(0, 1.0, (n_w, K, 3)) * 1e-7 / np.arange(1, K + 1)[None, :, None]
    return C


def test_the_device_coded_phases_match_the_numpy_route():
    """``coded_phases`` (``gamma sum C W``, the replay's own reading of a pack) on the device against the host's
    float64: the same phase of every walker under every waveform of the battery, to float32 rounding."""
    from dmipy_sim.replay.compression import coded_phases, acquisition_battery, default_envelope
    n_t, dt = 257, 1e-4
    C = _coefficients(300, 32, 3)
    G, _ = acquisition_battery(n_t, dt, default_envelope())
    phi_np = coded_phases(C, dt, G, n_t, device="numpy")
    phi_dv = coded_phases(C, dt, G, n_t, device="jax")
    assert phi_np.shape == phi_dv.shape == (300, G.shape[0])
    assert np.abs(phi_np - phi_dv).max() < 2e-6 * max(1.0, np.abs(phi_np).max())


def test_the_device_floor_matches_the_numpy_route():
    """``measure_floor_coded`` (the split-half floor of a walk over the battery, no path decoded) on either device."""
    from dmipy_sim.replay.compression import measure_floor_coded
    C = _coefficients(400, 16, 4)
    f_np = measure_floor_coded(C, 1e-4, 257, device="numpy")
    f_dv = measure_floor_coded(C, 1e-4, 257, device="jax")
    assert f_np["noise_floor"] == f_dv["noise_floor"]
    assert abs(f_np["floor_max"] - f_dv["floor_max"]) < 1e-5
    assert f_np["per_family"].keys() == f_dv["per_family"].keys()
    assert all(abs(f_np["per_family"][k] - f_dv["per_family"][k]) < 1e-5 for k in f_np["per_family"])


def test_the_device_field_series_encoder_matches_the_numpy_route():
    """``susc_path_encode_series``'s DCT (:func:`~dmipy_sim.replay.compression.dct_bands_series`, a cached device
    matmul reused across every walker chunk) and its quantisation (:func:`~dmipy_sim.replay.bank._quantise_susc_path`)
    on the device give the numpy route's dequantised coefficients to the container's own rounding."""
    from dmipy_sim.replay.bank import susc_path_encode_series
    from dmipy_sim.fields.hollow_cylinder import CHANNEL_NAMES
    rng = np.random.default_rng(11)
    n_w, n_t = 517, 65
    S = rng.normal(0, 1e-7, (n_w, n_t, 13)).astype(np.float32)
    S[:, :, 3] = 3 * S[:, :, 0] - S[:, :, 1] - S[:, :, 2]            # the trace identity: iso_P_zz is implied
    a_np, m_np = susc_path_encode_series(S, CHANNEL_NAMES, K=8, container=((None, 8),), layout="wtc", device="numpy", chunk=37)
    a_dv, m_dv = susc_path_encode_series(S, CHANNEL_NAMES, K=8, container=((None, 8),), layout="wtc", device="jax", chunk=37)
    assert {k: v for k, v in m_np.items() if k != "trace_residual"} == {k: v for k, v in m_dv.items() if k != "trace_residual"}
    assert m_np["bits"] == 8 and m_np["n_ch"] == 12                              # iso_P_zz dropped on both routes
    dq = lambda a: a["susc_path_dct"].astype(np.float64) * a["susc_path_scale"].astype(np.float64)[None]
    q_np, q_dv = dq(a_np), dq(a_dv)
    scale = np.abs(q_np).max()
    assert np.abs(q_np - q_dv).max() < 1e-2 * scale                              # float32 DCT + a device quantisation
    # bits=None skips the container: the device's float32 coefficients against the numpy route's float64 ones
    a_np32, _ = susc_path_encode_series(S, CHANNEL_NAMES, K=8, container=None, layout="wtc", device="numpy")
    a_dv32, _ = susc_path_encode_series(S, CHANNEL_NAMES, K=8, container=None, layout="wtc", device="jax")
    scale32 = np.abs(a_np32["susc_path_dct"].astype(np.float64)).max()
    assert np.abs(a_dv32["susc_path_dct"].astype(np.float64) - a_np32["susc_path_dct"].astype(np.float64)).max() < 1e-5 * scale32


def test_the_vectorised_census_matches_a_per_walker_lookup():
    """``_census_rows`` (the per-voxel certificate's binning, dmipy-sim#685) gives the same occupied-voxel table
    and row assignment as the retired per-walker Python dict lookup, for a scattered set of walkers."""
    from dmipy_sim.replay.bank import _census_rows
    rng = np.random.default_rng(12)
    n_w = 2000
    ijk_all = rng.integers(-3, 4, size=(n_w, 3))
    inside = rng.random(n_w) > 0.1
    ijk, row = _census_rows(ijk_all, inside)
    ref_ijk = np.unique(ijk_all[inside], axis=0)
    key = {tuple(v): i for i, v in enumerate(map(tuple, ref_ijk))}
    ref_row = np.full(n_w, -1, np.int64)
    for i in np.flatnonzero(inside):
        ref_row[i] = key[tuple(ijk_all[i])]
    np.testing.assert_array_equal(ijk, ref_ijk)
    np.testing.assert_array_equal(row, ref_row)
    # an empty selection names no voxel and assigns every walker -1
    empty_ijk, empty_row = _census_rows(ijk_all, np.zeros(n_w, bool))
    assert empty_ijk.shape == (0, 3) and (empty_row == -1).all()


def test_voxel_floor_coded_device_matches_numpy_and_uses_the_vectorised_census():
    """``voxel_floor_coded`` (the per-voxel certificate of a fill block, dmipy-sim#685) gives the same occupied
    voxels and walker counts on either device, and floors that agree to the coded phases' own float32 rounding."""
    from dmipy_sim.replay.bank import voxel_floor_coded
    from dmipy_sim.replay.compression import default_envelope
    from dmipy_sim.phantom import Grid
    n_w, n_t, K = 400, 65, 8
    C = _coefficients(n_w, K, 13)
    grid = Grid(shape=(2, 2, 2), voxel_size_m=(5e-6,) * 3, origin_m=(2.5e-6,) * 3)
    comp = np.zeros((n_w, n_t), np.int8)
    env = default_envelope()
    ijk_np, pools_np, n_np, floor_np, _ = voxel_floor_coded(C, 2e-5, n_t, grid, comp, env, device="numpy")
    ijk_jx, pools_jx, n_jx, floor_jx, _ = voxel_floor_coded(C, 2e-5, n_t, grid, comp, env, device="jax")
    np.testing.assert_array_equal(ijk_np, ijk_jx)
    assert pools_np == pools_jx
    np.testing.assert_array_equal(n_np, n_jx)
    ok = np.isfinite(floor_np)
    assert ok.any()
    assert np.abs(floor_np[ok] - floor_jx[ok]).max() < 1e-4 * max(1.0, float(np.nanmax(floor_np)))
