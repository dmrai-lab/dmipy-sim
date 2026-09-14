"""A block of a fill inherits the fill's codec certificate and measures only its own floor; the band transforms
run on the JAX device and agree with scipy.

A pack's fidelity battery replays the envelope on the raw and the decoded walk: ninety percent of a block's
pack time, re-proving the same codec on every block of one fill. `fidelity="inherited"` cites the certifying
pack, keeps its codec error, and reads this walk's split-half floor -- whole and per voxel -- from the stored
coefficients (RPK.md 9.4 rule 4)."""
import json

import numpy as np
import pytest

from dmipy_sim.io.strands import write_tck
from dmipy_sim.phantom import Grid
from dmipy_sim.replay import compression as cx
from dmipy_sim.replay.bank import build_replay_pack, voxel_fidelity_volumes
from dmipy_sim.spec import disco_spec, walk_spec, StratifiedByVoxel


@pytest.fixture(scope="module")
def walks(tmp_path_factory):
    """Two disjoint blocks of one small strand walk plan (the pack-merge fixture), K lossless on 5 saves."""
    tmp = tmp_path_factory.mktemp("inherit")
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "t.tck"), str(tmp / "d.txt")
    write_tck(tck, cls_, coordinate_unit_m=25e-6); np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    spec = disco_spec(tck, dia, side_m=20e-6)
    grid = Grid(shape=(2, 2, 2), voxel_size_m=(10e-6,) * 3, origin_m=(5e-6,) * 3)
    out = []
    for b in (0, 1):
        want = np.zeros(grid.shape, np.int64); want[b] = 12
        out.append(walk_spec(spec, T_max=8e-4, dt_save=2e-4, seed=7 + b, require_gpu=False, field=False,
                             seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want})))
    return grid, out


def test_device_bands_equal_scipy():
    """The DST-I bands on the JAX device (a full-precision matmul against the sine matrix) equal scipy's."""
    rng = np.random.default_rng(0)
    u = np.cumsum(rng.normal(0, 1e-7, (300, 257, 3)), axis=1); u -= u[:, :1] + (u[:, -1:] - u[:, :1]) * np.linspace(0, 1, 257)[None, :, None]
    b_np = cx.dst_bands(u[:, 1:-1], 64, device="numpy"); b_jx = cx.dst_bands(u[:, 1:-1], 64, device="jax")
    assert b_np.shape == (300, 64, 3) and np.abs(b_jx - b_np).max() < 1e-6 * np.abs(b_np).max()
    X = np.cumsum(rng.normal(0, 1e-7, (200, 129, 3)), axis=1) + 1e-3                # metres, at millimetre coordinates
    a_np, m_np, _ = cx.encode_bridge_dst(X, 32, device="numpy"); a_jx, m_jx, _ = cx.encode_bridge_dst(X, 32, device="jax")
    assert m_np == m_jx
    p_np = cx.decode_bridge_dst(a_np, m_np); p_jx = cx.decode_bridge_dst(a_jx, m_jx)
    assert np.abs(p_jx - p_np).max() < 1e-9                                       # nanometres apart, not an error
    with pytest.raises(ValueError, match="device"):
        cx.dst_bands(u[:, 1:-1], 8, device="cuda")
    dl = np.abs(rng.normal(0, 1e-3, (150, 129)))
    c_np, _ = cx.encode_boundary_bridge(dl, 8, np.float32, device="numpy"); c_jx, _ = cx.encode_boundary_bridge(dl, 8, np.float32, device="jax")
    assert np.abs(c_jx["blt_bridge_dst"] - c_np["blt_bridge_dst"]).max() < 1e-5 * np.abs(c_np["blt_bridge_dst"]).max()
    np.testing.assert_array_equal(c_jx["blt_endpoint"], c_np["blt_endpoint"])


def test_coded_phases_are_the_replays(walks):
    """Phases from the coefficients equal the dense phases when the bridge is lossless."""
    _, (w0, _) = walks
    X = np.asarray(w0.positions, np.float64); n_t = X.shape[1]; dt = float(w0.dt)
    arrays, meta, _ = cx.encode_bridge_dst(X, n_t - 2, device="numpy")
    C = cx.read_position_coeffs(arrays, dtype=np.float64)
    G, _ = cx.acquisition_battery(n_t, dt, cx.default_envelope())
    np.testing.assert_allclose(cx.coded_phases(C, dt, G, n_t, device="numpy"), cx._walker_phases(X, dt, G), atol=1e-6)   # the float32 container
    fl = cx.measure_floor_coded(C, dt, n_t)
    fid = cx.measure_fidelity(X, dt, X)
    assert abs(fl["floor_max"] - fid["floor_max"]) < 1e-7 and fl["per_family"].keys() == fid["per_family"].keys()


def test_a_block_inherits_the_certificate_and_measures_its_own_floor(walks, tmp_path):
    """The measured pack of block 0 certifies; block 1 inherits: same codec, the certifying error, its own floors
    (whole and per voxel, equal to a measured build of the same walk), no dense replay; a differing codec, a
    missing citation or an inherited certificate as the source are refused."""
    grid, (w0, w1) = walks
    K = w0.positions.shape[1] - 2
    cert = build_replay_pack(w0, id="fill/block-0", license="x", citation="x", K=K, voxel_grid=grid, blt_temporal_K=3,
                             device="numpy")
    assert cert.meta["fidelity"]["certified"] == "measured" and "inherited_from" not in cert.meta["fidelity"]
    shard = build_replay_pack(w1, id="fill/block-1", license="x", citation="x", voxel_grid=grid, device="numpy",
                              fidelity="inherited", fidelity_from=cert, out_path=str(tmp_path / "b1.rpk"))
    f = shard.meta["fidelity"]
    assert f["certified"] == "inherited" and f["inherited_from"]["id"] == "fill/block-0"
    assert f["err_max"] == cert.meta["fidelity"]["err_max"] and f["err_surface"] == cert.meta["fidelity"]["err_surface"]
    assert shard.K == cert.K and shard.meta["compression"]["channels"]["boundary_local_time"]["K"] == 3
    measured = build_replay_pack(w1, id="fill/block-1-measured", license="x", citation="x", K=K, voxel_grid=grid,
                                 blt_temporal_K=3, device="numpy")
    assert f["floor_max"] == max(v["floor_max"] for v in f["per_family"].values())    # this walk's own floor, the same split
    for fam in f["per_family"]:
        assert abs(f["per_family"][fam]["floor_max"] - measured.meta["fidelity"]["per_family"][fam]["floor_max"]) < 1e-7
    _, fl_s, n_s = voxel_fidelity_volumes(shard); _, fl_m, n_m = voxel_fidelity_volumes(measured)
    for name in fl_s:
        np.testing.assert_array_equal(n_s[name], n_m[name]); np.testing.assert_allclose(fl_s[name], fl_m[name], atol=1e-7)
    assert np.isnan(shard.arrays["voxel_certificate"][..., 2]).all()                  # the codec error is the certifying pack's
    for name in shard.arrays:                                                          # the same tensors as a measured build
        if name != "voxel_certificate":
            np.testing.assert_array_equal(shard.arrays[name], measured.arrays[name])
    import dmipy_sim as d
    seq = d.pgse([[1, 0, 0]], 2e-4, 4e-4, gradient_strengths=0.1, n_t=shard.n_t, slew_rate=np.inf)
    np.testing.assert_allclose(shard.replay(seq, tissue=False), measured.replay(seq, tissue=False), rtol=1e-12)
    from dmipy_sim.replay import read_rpk
    assert read_rpk(str(tmp_path / "b1.rpk")).meta["fidelity"]["certified"] == "inherited"
    with pytest.raises(ValueError, match="inherits a certificate only"):
        build_replay_pack(w1, id="x", license="x", citation="x", K=2, fidelity="inherited", fidelity_from=cert, device="numpy")
    with pytest.raises(ValueError, match="needs fidelity_from"):
        build_replay_pack(w1, id="x", license="x", citation="x", fidelity="inherited", device="numpy")
    with pytest.raises(ValueError, match="cannot certify another"):
        build_replay_pack(w1, id="x", license="x", citation="x", fidelity="inherited", fidelity_from=shard, device="numpy")
    with pytest.raises(ValueError, match="goes with"):
        build_replay_pack(w1, id="x", license="x", citation="x", K=K, fidelity_from=cert, device="numpy")
    # the citation survives as JSON (a fill distributes its certifying pack's meta, not the pack)
    again = build_replay_pack(w1, id="fill/block-1b", license="x", citation="x", fidelity="inherited", device="numpy",
                              fidelity_from=json.loads(json.dumps(cert.meta, default=float)))
    assert again.meta["fidelity"]["inherited_from"]["id"] == "fill/block-0"


def test_rounds_of_one_block_merge_and_recertify(walks, tmp_path):
    """A small host walks one block in rounds that share its voxels; `merge_packs(overlap="recertify")` joins them
    and re-reads the union's per-voxel floors from the coefficients: counts add, the replay of the union is the
    weight-averaged replay of the rounds, and the plain merge refuses shared voxels."""
    from dmipy_sim.replay.bank import merge_packs
    import dmipy_sim as d
    grid, (w0, w1) = walks
    K = w0.positions.shape[1] - 2
    cert = build_replay_pack(w0, id="fill/cert", license="x", citation="x", K=K, voxel_grid=grid, blt_temporal_K=3, device="numpy")
    spec = w1.spec
    want = np.zeros(grid.shape, np.int64); want[1] = 12
    rounds = []
    for r in (0, 1):
        w = walk_spec(spec, T_max=8e-4, dt_save=2e-4, seed=100 + r, require_gpu=False, field=False,
                      seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
        rounds.append(build_replay_pack(w, id=f"fill/block-1/round-{r}", license="x", citation="x", voxel_grid=grid, device="numpy",
                                        fidelity="inherited", fidelity_from=cert, out_path=str(tmp_path / f"r{r}.rpk")))
    with pytest.raises(ValueError, match="unless the merge recertifies"):
        merge_packs(rounds, id="fill/block-1")
    merged = merge_packs([str(tmp_path / "r0.rpk"), str(tmp_path / "r1.rpk")], id="fill/block-1", overlap="recertify", device="numpy",
                         out_path=str(tmp_path / "b1.rpk"))
    pv = merged.meta["fidelity"]["per_voxel"]
    assert pv["recertified"] and pv["shards"] == 2 and merged.n_walkers == rounds[0].n_walkers + rounds[1].n_walkers
    _, fl, n = voxel_fidelity_volumes(merged)
    _, fl0, n0 = voxel_fidelity_volumes(rounds[0]); _, fl1, n1 = voxel_fidelity_volumes(rounds[1])
    for name in n:
        np.testing.assert_array_equal(n[name], n0[name] + n1[name])
        assert np.isfinite(fl[name][n[name] > 1]).all() and (fl[name][n[name] == 0] == 0).all()
    assert np.isnan(merged.arrays["voxel_certificate"][..., 2]).all()
    seq = d.pgse([[1, 0, 0]], 2e-4, 4e-4, gradient_strengths=0.1, n_t=merged.n_t, slew_rate=np.inf)
    W = [float(np.asarray(pk.arrays["spin_weights"]).sum()) for pk in rounds]
    S = [pk.replay(seq, tissue=False)[0] for pk in rounds]
    np.testing.assert_allclose(merged.replay(seq, tissue=False)[0], (W[0] * S[0] + W[1] * S[1]) / (W[0] + W[1]), rtol=1e-6)   # float32 weights
