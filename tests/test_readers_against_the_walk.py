"""The readers against the walk they read (dmipy-sim#743): every reader takes the coefficient form, and what it
returns is what the walk's own block ends give.

The reference is built here from the raw subset of an order-0 walk (``block_order=0``, every walker kept): at
``m = 0`` the accumulated bands ARE ``sqrt(h) sinc(k / 2N)`` times the orthonormal DST-I of the bridge through the
block ends (the exactness relation of #741 / #745), so the walk's own block ends give the coefficients in float64,
and a replay contracts them with the waveform's projection. A pack stores the same coefficients in float32: the
columnar image and the pose expansion of the pack agree with that reference to float32 rounding. A two-window pack's
whole-window reads (``prefix`` on a window boundary, ``window(i)``) are the pack of the walk's own window, bit for bit
on the bands. A consolidation of shards that disagree on the block or the window table is refused by name.
"""
import dmipy_replay.replay.compression as _r_compression
import numpy as np
import pytest
from scipy.fft import dst

import dmipy_sim as d
from dmipy_replay.constants import GAMMA
from dmipy_sim.replay import ReplayPack, so3
from dmipy_sim.replay import compression as cx
from dmipy_sim.replay.bank import build_replay_pack

T_MAX, H, K = 2e-3, 1e-4, 12                 # 20 blocks; K below the N - 1 = 19 bands the block ends hold


def _reference_coeffs(raw, h, K):
    """``(n_w, K + 2, 3)`` float64: the start, the displacement and the ``K`` continuous bands of every walker's path
    from its block ends ``raw`` ``(n_w, N + 1, 3)`` by the ``m = 0`` exactness relation."""
    x = np.asarray(raw, np.float64)
    N = x.shape[1] - 1
    T = N * h
    tau = np.arange(N + 1) / N
    u = x - (x[:, :1] + (x[:, -1:] - x[:, :1]) * tau[None, :, None])
    k = np.arange(1, K + 1)
    factor = _r_compression.cosine_band_scale(K, T) * 2.0 * np.sin(np.pi * k / (2.0 * N)) / np.sqrt(2.0 / N)     # sqrt(h) sinc(k / 2N)
    beta = factor[None, :, None] * dst(u[:, 1:-1], type=1, norm="ortho", axis=1)[:, :K]
    return np.concatenate([x[:, :1], x[:, -1:] - x[:, :1], beta], axis=1)


def _reference_phases(C, seq, T, R=None):
    """``(n_w, n_meas)``: ``gamma C . W`` with ``W`` the waveform's projection on the bands (its gradient turned by
    ``R`` into the stored frame when given)."""
    G = np.asarray(seq.G_eff, np.float64)
    if R is not None:
        G = G @ np.asarray(R, np.float64)
    W = _r_compression.band_projection(G, float(seq.dt), T, C.shape[1] - 2)
    return GAMMA * np.einsum("wkc,mkc->wm", C, W)


def _seq():
    return d.set_b(d.pgse([[1, 0, 0], [0, 0, 1], [0.6, 0.8, 0.0]], 0.3e-3, 0.9e-3, gradient_strengths=0.2,
                          n_t=201, slew_rate=np.inf, TE=T_MAX), [2e9, 2e9, 2e9])


def _cell():
    """A periodic cell of two packed cylinders (both pools walked) and a 2 x 2 x 1 voxel grid over it."""
    from dmipy_replay.phantom.grid import Grid
    g = d.PackedCylinders([1.5e-6, 2e-6], [[-2.5e-6, 0.0], [2.5e-6, 0.0]], 10e-6)
    return g, Grid(shape=(2, 2, 1), voxel_size_m=(5e-6, 5e-6, 1e-5), origin_m=(-2.5e-6, -2.5e-6, 0.0))


@pytest.fixture(scope="module")
def walked(tmp_path_factory):
    """Two shards of one substrate -- the walkers starting in the cell's lower and upper half in x, disjoint voxel
    blocks of one fill -- each an order-0 walk with every walker's block ends kept, packed in the float containers
    (the pack is the walk's coefficients in float32) and consolidated into the columnar layout."""
    import warnings
    from dmipy_sim.fill.consolidate import consolidate
    g, grid = _cell()
    tmp = tmp_path_factory.mktemp("oracle"); (tmp / "shards").mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        starts = np.asarray(d.simulate_trajectories(600, 2e-9, g, 2 * H, H, seed=30, require_gpu=False, K=1).position_coeffs[:, 0, 0])
        walks, packs = [], []
        for block, sel in ((0, starts[:, 0] < 0), (1, starts[:, 0] >= 0)):
            r0 = np.ascontiguousarray(starts[sel], np.float32)
            w = d.simulate_trajectories(len(r0), 2e-9, g, T_MAX, H, seed=31 + block, r0=r0, require_gpu=False, K=K, block_order=0,
                                        raw_subset=len(r0))
            assert w.m == 0 and w.raw_positions.shape == (w.n_walkers, w.n_blocks + 1, 3)
            walks.append(w)
            packs.append(build_replay_pack(w, id=f"t/oracle-{block}", license="x", citation="x", K=K, voxel_grid=grid,
                                           position_container=None, blt_container=None,
                                           out_path=str(tmp / "shards" / f"block-{block:04d}.p1.rpk")))
    out = str(tmp / "layout")
    consolidate(str(tmp / "shards"), out, blocks=[0, 1], id="t/oracle")
    return walks, packs, ReplayPack.open(out), grid


def test_the_bands_are_the_block_ends_at_order_zero(walked):
    """The premise: the pack's coefficients are the reference's to the float32 the walk accumulated in."""
    walks, packs, _, _ = walked
    for w, pk in zip(walks, packs):
        ref = _reference_coeffs(w.raw_positions, w.h, K)
        got = np.asarray(pk.position_coeffs, np.float64)
        scale = np.abs(ref[:, 2:]).max()
        assert np.abs(got[:, 2:] - ref[:, 2:]).max() <= 2e-6 * scale
        np.testing.assert_allclose(got[:, :2], ref[:, :2], rtol=0, atol=1e-6 * np.abs(ref[:, 0]).max())


def test_the_columnar_image_is_the_walks_own_signal(walked):
    """``ColumnarPack.image`` of the consolidated layout against the per-voxel signal of the reference coefficients:
    float32 rounding."""
    walks, packs, layout, grid = walked
    seq = _seq()
    S, _floor, plan = layout.image(seq, tol=1e-12)
    assert plan["K"] == K and plan["rows"] == sum(w.n_walkers for w in walks)
    full = layout.view()
    w_rows = np.asarray(full.spin_weights, np.float64)
    r0_rows = np.asarray(full.r0, np.float64)
    C = np.concatenate([_reference_coeffs(w.raw_positions, w.h, K) for w in walks])
    r0_walk = C[:, 0, :]
    order = {tuple(np.round(r, 15)): i for i, r in enumerate(r0_walk)}
    idx = np.asarray([order[tuple(np.round(r, 15))] for r in r0_rows])        # the layout's rows, in the walk's order
    phi = _reference_phases(C[idx], seq, T_MAX)
    ijk, _ = grid.bin(r0_rows)
    v = np.ravel_multi_index(ijk.T, grid.shape)
    ref = np.full((int(np.prod(grid.shape)), phi.shape[1]), np.nan)
    for vox in np.unique(v):
        sel = v == vox
        ref[vox] = np.abs((w_rows[sel, None] * np.exp(1j * phi[sel])).sum(0) / w_rows[sel].sum())
    got = S.reshape(-1, phi.shape[1])
    held = np.isfinite(ref[:, 0])
    np.testing.assert_array_equal(np.isfinite(got[:, 0]), held)
    err = np.abs(got[held] - ref[held]).max()
    assert err <= 2e-5, err                                   # float32 coefficients against the float64 reference


def test_the_pose_expansion_is_the_walks_own_posed_signal(walked):
    """``pose_response`` (the closed form) of a shard against the reference coefficients replayed with the gradient
    turned at every rotation: float32 rounding."""
    walks, packs, _, _ = walked
    w, pk = walks[0], packs[0]
    seq = d.pgse([[1, 0, 0]], 0.3e-3, 0.9e-3, gradient_strengths=[0.2], n_t=201, slew_rate=np.inf, TE=T_MAX)
    pr = pk.pose_response(seq, method="closed")
    assert pr.n_samples == 0
    C = _reference_coeffs(w.raw_positions, w.h, K)
    wt = np.asarray(pk.spin_weights, np.float64)
    for R in so3.haar_rotations(8, 11):
        phi = _reference_phases(C, seq, T_MAX, R=pk.pose_rotation(R))
        ref = (wt[:, None] * np.exp(1j * phi)).sum(0) / wt.sum()
        np.testing.assert_allclose(pr.at(R), ref, rtol=0, atol=2e-5 + float(pr.misfit.max()))


@pytest.fixture(scope="module")
def two_windows():
    g = d.Cylinder(radius=3e-6, orientation=(0, 0, 1))
    w = d.simulate_trajectories(300, 2e-9, g, 4e-3, 1e-4, seed=9, require_gpu=False, K=12, window_T=2e-3, sub_steps=2)
    kw = dict(license="x", citation="x", K=12, position_container=None, blt_container=None)
    return w, build_replay_pack(w, id="t/two", **kw), kw


def _bands_equal(a, b):
    for key in ("pos_x", "pos_y", "pos_z"):
        np.testing.assert_array_equal(np.asarray(a.arrays[key]), np.asarray(b.arrays[key]), err_msg=key)


def test_a_whole_window_read_is_the_pack_of_the_walks_window(two_windows):
    """On a two-window pack, ``prefix`` on the window boundary and ``window(i)`` are the packs of the walk's own
    windows, bit for bit on the bands; a cut inside a later window is refused by name."""
    w, pk, kw = two_windows
    assert pk.n_windows == 2 and pk.T_window == pytest.approx(2e-3)
    _bands_equal(pk.prefix(2e-3), build_replay_pack(w.window(0), id="t/w0", **kw))
    _bands_equal(pk.window(1), build_replay_pack(w.window(1), id="t/w1", **kw))
    with pytest.raises(ValueError, match="cuts inside window 1"):
        pk.prefix(3e-3)


def test_shards_on_another_block_or_window_table_are_refused_by_name(tmp_path):
    """A consolidation of shards that disagree on ``walk_params.block`` or ``walk_params.windows`` is refused, naming
    the key."""
    from dmipy_sim.fill.consolidate import consolidate
    g, grid = _cell()
    kw = dict(license="x", citation="x", K=4, voxel_grid=grid)

    def shard(name, **walk):
        w = d.simulate_trajectories(60, 2e-9, g, 2e-3, **walk, seed=len(name), require_gpu=False, K=4, sub_steps=2)
        (tmp_path / name).mkdir(exist_ok=True)
        return w
    for case, other, key in (("block", dict(block_h=2e-4), "walk_params.block"),
                             ("windows", dict(block_h=1e-4, window_T=1e-3), "walk_params.windows")):
        root = tmp_path / case; root.mkdir()
        a = shard(f"{case}a", block_h=1e-4)
        b = shard(f"{case}b", **other)
        build_replay_pack(a, id="t/a", out_path=str(root / "block-0000.p1.rpk"), **kw)
        build_replay_pack(b, id="t/b", out_path=str(root / "block-0001.p1.rpk"), **kw)
        with pytest.raises(ValueError, match=r"differs in [^:]*" + key.replace(".", r"\.")):
            consolidate(str(root), str(tmp_path / f"{case}-out"), blocks=[0, 1], id="t/x")
