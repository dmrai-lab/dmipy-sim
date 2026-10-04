"""The shape-moment layout (dmipy-sim#505): a columnar pack contracted once against a waveform's shape, in tiles, replays
every amplitude and direction of that shape as the columnar image does, with the same split-half floor; a shape
is one direction with amplitude-invariant ramps, and a b-value maps to an amplitude through the recorded unit b."""
from __future__ import annotations
import json
import os

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.columnar import ColumnarPack
from dmipy_sim.replay.shape_moments import ShapeMoments, write_shape_moments, _profile
from dmipy_sim.replay.study import Acquisition, Protocol, Study
from dmipy_sim.fill.consolidate import consolidate
from dmipy_sim.spec import walk_spec, fill_field, StratifiedByVoxel
from tests.conftest import spec_without_source


@pytest.fixture(scope="module")
def layout(spec_grid):
    """Two blocks of the three-strand spec consolidated into columns, and their n_t."""
    spec, grid, tmp = spec_grid
    spec = spec_without_source(spec)      # consolidation mechanics, not the field tier
    os.makedirs(str(tmp / "sm_shards"), exist_ok=True)
    n_t = None
    for b in (0, 1):
        want = np.zeros(grid.shape, np.int64); want[b] = 12
        w = walk_spec(spec, T_max=8e-4, dt_save=2e-4, seed=11 + b, require_gpu=False,
                      seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
        pk = build_replay_pack(w, id=f"t/sm{b}", license="x", citation="x", K=3, voxel_grid=grid,
                               out_path=str(tmp / "sm_shards" / f"block-000{b}.p1.rpk"))
        n_t = pk.n_t
    consolidate(str(tmp / "sm_shards"), str(tmp / "sm_layout"), blocks=[0, 1], id="t/sm-columns")
    return ColumnarPack(str(tmp / "sm_layout")), grid, n_t, tmp


def _shape(n_t, delta=0.2e-3, Delta=0.5e-3, g=0.05, direction=(0.0, 0.0, 1.0)):
    return d.pgse([list(direction)], delta, Delta, gradient_strengths=g, n_t=n_t, slew_rate=np.inf)


def test_the_moments_replay_every_amplitude_and_direction_of_the_shape(layout):
    """Six directions at three amplitudes of one shape: the layout's image equals the columnar image of the
    same acquisitions (built as one sequence at those b-values) to float32 arithmetic, its floor the study's
    split-half floor, and NaN sits exactly where the columns have no rows."""
    col, grid, n_t, tmp = layout
    out = str(tmp / "sm_moments")
    manifest = write_shape_moments(col, {"a": _shape(n_t), "c": _shape(n_t, delta=0.2e-3, Delta=0.4e-3, g=0.02)}, out,
                                   tol=1e-9, chunk_rows=7)                  # two shapes on one grid: one contraction
    assert manifest["n_rows"] == col.n_rows and manifest["K"] == col.K and manifest["source"]["n_rows"] == col.n_rows
    sm = ShapeMoments(out)
    assert sm.shapes == ["a", "c"] and sm.moments("a").shape == (sm.n_tiles, sm.tile, 3) and sm.moments("a").dtype == np.float32
    assert sm.n_rows == col.n_rows and sm.n_tiles * sm.tile >= col.n_rows
    # the tiles hold every row once: the per-voxel weight sums are the columns' own, and a padding row weighs nothing
    full = col.view(); ijk, _ = grid.bin(full.r0); v = np.ravel_multi_index(ijk.T, grid.shape)
    np.testing.assert_allclose(sm.weights.sum(1), np.bincount(v, full.spin_weights, minlength=sm.n_vox), rtol=1e-6)
    assert (np.asarray(sm._column("w")) > 0).sum() == col.n_rows
    rng = np.random.default_rng(3)
    u = rng.normal(size=(6, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    dirs = np.concatenate([u, u, u, [[0, 0, 1]]]); b = np.r_[np.full(6, 2e8), np.full(6, 1e9), np.full(6, 3e9), 0.0]
    S, floor = sm.image("a", b, dirs)
    seq = d.pgse(dirs.tolist(), 0.2e-3, 0.5e-3, bvalues=b, n_t=n_t, slew_rate=np.inf)
    S_col, _, plan = col.image(seq, tol=1e-9, chunk_rows=5)
    assert S.shape == S_col.shape == tuple(grid.shape) + (19,)
    np.testing.assert_array_equal(np.isnan(S), np.isnan(S_col))
    err = np.nanmax(np.abs(S - S_col))
    assert err < 2e-6, err                    # float32 phases and complex64 sums against the float64 image: 2.4e-7 measured
    np.testing.assert_allclose(np.nan_to_num(S[..., -1]), np.nan_to_num(S_col[..., -1]) , atol=1e-6)   # b = 0 is 1
    _, floor_study, _ = col.image(Study(Protocol([Acquisition(seq)])), tol=1e-9, chunk_rows=5)
    np.testing.assert_array_equal(np.isnan(floor), np.isnan(floor_study[0]))
    assert np.nanmax(np.abs(floor - floor_study[0])) < 2e-6
    S_c, _ = sm.image("c", b[:6], dirs[:6])                                   # the second shape of the shared pass
    seq_c = d.pgse(dirs[:6].tolist(), 0.2e-3, 0.4e-3, bvalues=b[:6], n_t=n_t, slew_rate=np.inf)
    S_col_c, _, _ = col.image(seq_c, tol=1e-9, chunk_rows=5)
    assert np.nanmax(np.abs(S_c - S_col_c)) < 2e-6 and np.nanmax(np.abs(S_c - S[..., :6])) > 1e-3


def test_the_amplitude_of_a_b_value_is_the_shapes_own(layout):
    """``g = sqrt(b / b_unit)``: the recorded unit b makes the builder's own b at any amplitude."""
    col, grid, n_t, tmp = layout
    sm = ShapeMoments(str(tmp / "sm_moments"))
    seq = _shape(n_t, g=0.123)
    g = sm.amplitude("a", d.calc_b(seq))
    np.testing.assert_allclose(g, 0.123, rtol=1e-6)
    np.testing.assert_array_equal(sm.amplitude("a", [0.0]), [0.0])
    with pytest.raises(ValueError, match="negative"):
        sm.amplitude("a", [-1.0])
    m = json.load(open(os.path.join(str(tmp / "sm_moments"), "manifest.json")))
    assert m["shapes"]["a"]["build_spec"][0] == "pgse" and m["shapes"]["a"]["encoding"]["delta"] == [0.2e-3]
    assert m["code"]["package"] == "dmipy-sim" and m["source"]["manifest_sha256"]


def test_a_shape_is_one_direction_with_amplitude_invariant_ramps(layout):
    col, grid, n_t, tmp = layout
    s, u, g = _profile(_shape(n_t, g=0.07, direction=(0.6, 0.0, 0.8)))
    assert abs(s).max() == pytest.approx(1.0) and g == pytest.approx(0.07) and np.allclose(u, [0.6, 0, 0.8])
    with pytest.raises(ValueError, match="one-row"):
        _profile(d.pgse([[1, 0, 0], [0, 1, 0]], 0.2e-3, 0.5e-3, gradient_strengths=0.05, n_t=n_t, slew_rate=np.inf))
    with pytest.raises(ValueError, match="slew_rate=np.inf"):
        _profile(d.pgse([[0, 0, 1]], 1e-3, 2e-3, gradient_strengths=0.05, n_t=n_t, slew_rate=200.0))
    with pytest.raises(ValueError, match="one direction"):
        _profile(d.pte([0, 0, 1], 0.4e-3, bvalues=[1e9], n_t=n_t, slew_rate=np.inf))
    with pytest.raises(ValueError, match="at least one"):
        write_shape_moments(col, {}, str(tmp / "sm_none"))


def test_directions_must_match_and_be_unit(layout):
    col, grid, n_t, tmp = layout
    sm = ShapeMoments(str(tmp / "sm_moments"))
    with pytest.raises(ValueError, match="match"):
        sm.image("a", [1e9, 1e9], [[0, 0, 1]])
    with pytest.raises(ValueError, match="unit"):
        sm.image("a", [1e9], [[0, 0, 2]])
    with pytest.raises(KeyError):
        sm.moments("b")


def test_the_bands_are_judged_at_the_shapes_own_amplitude_and_the_device_rows_can_be_released(layout):
    """A shape built at a tenth of the amplitude needs no more bands than one built at the full amplitude (the
    truncation error scales with the amplitude squared, and the manifest records the amplitude judged at); after
    ``release`` the device holds the shared rows only."""
    col, grid, n_t, tmp = layout
    out = str(tmp / "sm_amp")
    lo = write_shape_moments(col, {"lo": _shape(n_t, g=0.005)}, out, tol=1e-9, chunk_rows=7)
    assert lo["shapes"]["lo"]["amplitude_built"] == pytest.approx(0.005) and lo["K"] <= col.K
    sm = ShapeMoments(str(tmp / "sm_moments"))
    sm.image("a", [1e9], [[0, 0, 1]]); sm.image("c", [1e9], [[0, 0, 1]])
    assert {k[1] for k in sm._device} == {"w", "tiles", "a", "c"}
    sm.release(keep=["c"])
    assert {k[1] for k in sm._device} == {"w", "tiles", "c"}


def test_the_torch_backend_is_the_jax_image(layout):
    """dmipy-sim#510: the same acquisitions on both backends agree to float32 arithmetic, NaN and floor alike; two
    torch calls are identical; a non-resident call leaves nothing on the device; the backend name is checked."""
    torch = pytest.importorskip("torch")
    col, grid, n_t, tmp = layout
    sm = ShapeMoments(str(tmp / "sm_moments"))
    rng = np.random.default_rng(5)
    u = rng.normal(size=(5, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    dirs = np.concatenate([u, u, [[0, 0, 1]]]); b = np.r_[np.full(5, 5e8), np.full(5, 2.5e9), 0.0]
    S_j, f_j = sm.image("a", b, dirs)
    torch.use_deterministic_algorithms(True)
    S_t, f_t = sm.image("a", b, dirs, backend="torch", device="cpu")
    np.testing.assert_array_equal(np.isnan(S_j), np.isnan(S_t))
    err = np.nanmax(np.abs(S_j - S_t)); assert err < 2e-6, err            # measured 7.6e-8 on the fixture (CPU)
    assert np.nanmax(np.abs(f_j - f_t)) < 2e-6
    S_t2, _ = sm.image("a", b, dirs, backend="torch", device="cpu")
    np.testing.assert_array_equal(np.nan_to_num(S_t), np.nan_to_num(S_t2))
    before = set(sm._device)
    S_t3, _ = sm.image("c", b[:3], dirs[:3], backend="torch", device="cpu", resident=False)
    assert set(sm._device) == before and np.isfinite(S_t3).any()
    with pytest.raises(ValueError, match="backend"):
        sm.image("a", b, dirs, backend="numpy")


def test_preload_keeps_the_padded_host_arrays_and_changes_no_number(layout):
    """After ``preload`` the device copies are built from the host cache; the image is the same and a non-resident
    call still leaves nothing on the device."""
    torch = pytest.importorskip("torch")
    col, grid, n_t, tmp = layout
    sm = ShapeMoments(str(tmp / "sm_moments"))
    b = np.array([1e9, 0.0]); dirs = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    before, _ = sm.image("a", b, dirs, backend="torch", device="cpu", resident=False)
    sm.preload(["a"])
    assert set(sm._host) == {"w", "tiles", "a"} and sm._host["a"].shape == (-(-sm.n_tiles // sm.TILES) * sm.TILES, sm.tile, 3)
    after, _ = sm.image("a", b, dirs, backend="torch", device="cpu", resident=False)
    np.testing.assert_array_equal(np.nan_to_num(before), np.nan_to_num(after))
    assert not any(k[1] == "a" for k in sm._device)
    sm.preload()
    assert set(sm._host) == {"w", "tiles", "a", "c"}


def test_the_weights_are_read_once_per_layout(layout):
    """The per-voxel weight sums are the layout's, not the image's: every image after the first reads neither the
    weight nor the tile column again for them (on DiSCo a float64 copy of 153 M rows per call)."""
    col, grid, n_t, tmp = layout
    sm = ShapeMoments(str(tmp / "sm_moments"))
    reads = []
    column = sm._column
    sm._column = lambda name: (reads.append(name), column(name))[1]
    b = np.array([1e9, 0.0]); dirs = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    first, _ = sm.image("a", b, dirs, resident=True)          # the device copy is kept: only the weights could read again
    n_first = len(reads)
    second, _ = sm.image("a", b, dirs, resident=True)
    assert "w" in reads[:n_first] and reads[n_first:] == [], reads
    np.testing.assert_array_equal(np.nan_to_num(first), np.nan_to_num(second))


@pytest.fixture(scope="module")
def layout_field(spec_grid):
    """Two blocks of the three-strand spec walked WITH the sheath's field (C3) and the contact and occupancy tiers,
    consolidated into columns; the merged pack beside it."""
    from dmipy_sim.replay.bank import merge_packs
    spec, grid, tmp = spec_grid
    os.makedirs(str(tmp / "sf_shards"), exist_ok=True)
    packs = []
    for b in (0, 1):
        want = np.zeros(grid.shape, np.int64); want[b] = 12
        w = walk_spec(spec, T_max=8e-4, dt_save=2e-4, seed=21 + b, require_gpu=False, field_res=0.5e-6,
                      seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
        w = fill_field(w, w.field_basis)                     # read at save resolution: no adaptive stepping here
        packs.append(build_replay_pack(w, id=f"t/sf{b}", license="x", citation="x", K=3, susc_path_K=4, voxel_grid=grid,
                                       out_path=str(tmp / "sf_shards" / f"block-000{b}.p1.rpk")))
    consolidate(str(tmp / "sf_shards"), str(tmp / "sf_layout"), blocks=[0, 1], id="t/sf-columns")
    return ColumnarPack(str(tmp / "sf_layout")), merge_packs(packs, id="t/sf-merged"), grid, packs[0].n_t, tmp


def _per_voxel(pack, grid, w, ew, E):
    ijk, _ = grid.bin(pack.r0); v = np.ravel_multi_index(ijk.T, grid.shape); keys, inv = np.unique(v, return_inverse=True)
    num = np.zeros((len(keys), E.shape[1]), complex); den = np.zeros(len(keys))
    np.add.at(num, inv, ew[:, None] * E); np.add.at(den, inv, w)
    return keys, np.abs(num / den[:, None])


def test_the_tiers_are_the_studys_terms_on_every_pathway(layout_field):
    """dmipy-sim#514: a spin-echo shape and a stimulated-echo shape in one layout with tiers; the image under a tissue
    (T2 and T1 per pool, rho2, chi) at 3 T equals the merged pack's ``walker_signals`` per voxel to float32 arithmetic
    -- the stimulated echo's storage time carries T1 and its pathway amplitude, its gate the field and the contact;
    the bare image is the pathway-scaled bare replay; a field asked of the bare layout is refused by name."""
    from dmipy_sim.spec.tissue import Tissue
    col, merged, grid, n_t, tmp = layout_field
    se = d.pgse([[0, 0, 1]], 0.2e-3, 0.5e-3, gradient_strengths=0.05, n_t=n_t, slew_rate=np.inf)
    ste = d.pgste([[0, 0, 1]], 0.2e-3, 0.4e-3, gradient_strengths=0.05, n_t=n_t, slew_rate=np.inf, ste_flip_angles=(90.0, 90.0, 90.0))
    out = str(tmp / "sf_moments")
    manifest = write_shape_moments(col, {"se": se, "ste": ste}, out, tol=1e-9, chunk_rows=7, tiers=True)
    tiers = manifest["tiers"]
    assert tiers["relaxation"] and len(tiers["groups"]) == 2 and all(g["contact"] and g["field"] for g in tiers["groups"].values())
    g_ste = next(g for g in tiers["groups"].values() if "ste" in g["shapes"])
    assert g_ste["pathway"] == pytest.approx(0.5, abs=1e-12) and g_ste["tau_t1"] > 0
    sm = ShapeMoments(out)
    t = Tissue(T2={"intra": 0.03, "extra": 0.08, "myelin": 0.01}, T1={"intra": 0.9, "extra": 1.4, "myelin": 0.3}, rho2=1e-5, chi_iso=-1e-7, chi_aniso=-1.5e-8)
    rng = np.random.default_rng(9); u = rng.normal(size=(4, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    for name, base, delta, Delta in (("se", se, 0.2e-3, 0.5e-3), ("ste", ste, 0.2e-3, 0.4e-3)):
        b = np.array([1e9, 1e9, 3e9, 0.0]); dirs = np.concatenate([u[:3], [[0, 0, 1]]])
        if name == "se":
            seq = d.pgse(dirs.tolist(), delta, Delta, bvalues=b, n_t=n_t, slew_rate=np.inf)
        else:
            seq = d.pgste(dirs.tolist(), delta, Delta, bvalues=b, n_t=n_t, slew_rate=np.inf, ste_flip_angles=(90.0, 90.0, 90.0))
        S, floor = sm.image(name, b, dirs, tissue=t, scanner=3.0)
        w, ew, E = merged.walker_signals(seq, tissue=t, scanner=3.0)
        keys, ref = _per_voxel(merged, grid, w, ew, E)
        got = S.reshape(-1, 4)[keys]
        err = np.abs(got - ref).max()
        assert err < 5e-6, (name, err)                                  # float32 phases and weights against float64: measured 1.6e-6 on the fixture (se)
        torch = pytest.importorskip("torch")
        St, _ = sm.image(name, b, dirs, tissue=t, scanner=3.0, backend="torch", device="cpu")   # the torch tiered kernel
        assert np.nanmax(np.abs(St - S)) < 5e-6
        S0, _ = sm.image(name, b, dirs)                                  # bare: the pathway amplitude alone
        w0, ew0, E0 = merged.walker_signals(seq)
        _, ref0 = _per_voxel(merged, grid, w0, ew0, E0)
        assert np.abs(S0.reshape(-1, 4)[keys] - ref0).max() < 5e-6
    bare = ShapeMoments(str(tmp / "sm_moments"))
    with pytest.raises(ValueError, match="tiers"):
        bare.image("a", [1e9], [[0, 0, 1]], tissue=t, scanner=3.0)


def test_the_field_direction_is_a_knob(layout_field):
    """The same tissue at 3 T with B0 along x: the image equals the per-voxel signal built from the pack's field
    channels contracted for x (``field_terms``), and differs from the z image; a non-unit direction is refused."""
    from dmipy_sim.spec.tissue import Tissue
    from dmipy_sim.replay.study import Acquisition, Primitives
    from dmipy_sim.replay.replay import _path_field_channels
    from dmipy_sim.fields.hollow_cylinder import field_terms
    col, merged, grid, n_t, tmp = layout_field
    sm = ShapeMoments(str(tmp / "sf_moments"))
    t = Tissue(T2={"intra": 0.03, "extra": 0.08, "myelin": 0.01}, rho2=1e-5, chi_iso=-1e-7, chi_aniso=-1.5e-8)
    b = np.array([1e9, 2e9]); dirs = np.array([[0.6, 0.0, 0.8], [0.0, 1.0, 0.0]])
    seq = d.pgse(dirs.tolist(), 0.2e-3, 0.5e-3, bvalues=b, n_t=n_t, slew_rate=np.inf)
    Sx, _ = sm.image("se", b, dirs, tissue=t, scanner=3.0, b0_direction=(1.0, 0.0, 0.0))
    Sz, _ = sm.image("se", b, dirs, tissue=t, scanner=3.0)
    prim = merged.walker_primitives(Acquisition(seq))
    P = merged._prepare(seq, tissue=None, scanner=None, orientation=None, compartment=None)
    Psi, names = _path_field_channels(P, seq)
    iso, aniso = field_terms(Psi, [1.0, 0.0, 0.0])
    prim_x = Primitives(w=prim.w, phi=prim.phi, field_iso=iso, field_aniso=aniso if aniso is not None else np.zeros_like(iso),
                        exposure_t2=prim.exposure_t2, exposure_t1=prim.exposure_t1, contact=prim.contact, D_walk=prim.D_walk,
                        voxel=prim.voxel, pathway=prim.pathway, by_pool=prim.by_pool)
    w, ew, E = prim_x.signals(t, 3.0)
    keys, ref = _per_voxel(merged, grid, w, ew, E)
    assert np.abs(Sx.reshape(-1, 2)[keys] - ref).max() < 5e-6
    assert np.nanmax(np.abs(Sx - Sz)) > 1e-6                            # the direction changes the field phase
    with pytest.raises(ValueError, match="unit"):
        sm.image("se", b, dirs, tissue=t, scanner=3.0, b0_direction=(0, 0, 2))


def test_the_files_a_layout_needs_include_its_tier_columns(layout_field):
    """ShapeMoments.files, what the Hub opener downloads: the manifest, w, tiles, every moment column, and with tiers
    the pool column and every group's contact and field columns; restricted to some shapes it keeps the tiers."""
    col, merged, grid, n_t, tmp = layout_field
    se = d.pgse([[0, 0, 1]], 0.2e-3, 0.5e-3, gradient_strengths=0.05, n_t=n_t, slew_rate=np.inf)
    ste = d.pgste([[0, 0, 1]], 0.2e-3, 0.4e-3, gradient_strengths=0.05, n_t=n_t, slew_rate=np.inf, ste_flip_angles=(90.0, 90.0, 90.0))
    out = tmp / "sf_files"
    m = write_shape_moments(col, {"se": se, "ste": ste}, str(out), tol=1e-9, chunk_rows=7, tiers=True)
    files = ShapeMoments.files(m)
    assert "manifest.json" in files and m["columns"]["w"] in files and m["columns"]["tiles"] in files
    assert m["tiers"]["pool_column"] in files
    for grp in m["tiers"]["groups"].values():
        assert all(f in files for f in grp["columns"].values())
    shapes = [s for s in m["columns"] if s not in ("w", "tiles")]
    assert all(m["columns"][s] in files for s in shapes)
    some = ShapeMoments.files(m, shapes[:1])
    assert m["columns"][shapes[0]] in some and m["tiers"]["pool_column"] in some
    assert len(shapes) == 1 or m["columns"][shapes[1]] not in some
    assert set(os.listdir(out)) >= set(files)                      # the written layout has every file the opener lists


def test_the_tier_maps_are_the_walkers_voxel_means(layout_field):
    """``tier_maps``: the per-voxel walk-weighted means of the pool membership, the contact term, the contact tier's
    factor and the field phase equal the same means taken on the host from the columns, on both backends; without a
    tissue there is no contact factor, without a scanner no phase; the direction changes the phase's spread."""
    from dmipy_sim.spec.tissue import Tissue
    col, merged, grid, n_t, tmp = layout_field
    se = d.pgse([[0, 0, 1]], 0.2e-3, 0.5e-3, gradient_strengths=0.05, n_t=n_t, slew_rate=np.inf)
    out = str(tmp / "sf_maps")
    write_shape_moments(col, {"se": se}, out, tol=1e-9, chunk_rows=7, tiers=True)
    sm = ShapeMoments(out)
    t = Tissue(T2={"intra": 0.03, "extra": 0.08, "myelin": 0.01}, T1={"intra": 0.9, "extra": 1.4, "myelin": 0.3}, rho2=1e-5, chi_iso=-1e-7, chi_aniso=-1.5e-8)
    g, grp = sm._tier_group("se")
    tiles = np.asarray(sm._column("tiles")); w = np.asarray(sm._column("w"), np.float64)
    pool = np.asarray(sm._column("pool")); c = np.asarray(sm._column(f"contact_{g}"), np.float64)

    def host_mean(x):
        num = np.bincount(tiles, (w * x[:len(tiles)]).sum(1), minlength=2 * sm.n_vox).reshape(-1, 2).sum(1)
        den = np.bincount(tiles, w.sum(1), minlength=2 * sm.n_vox).reshape(-1, 2).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan).reshape(grid.shape)

    u = np.array([0.6, 0.0, 0.8])
    _, rho_D, a_iso, a_aniso, _ = sm.terms("se", t, 3.0)
    iso, aniso = sm._field_terms(g, u); phase = a_iso * iso + a_aniso * aniso
    ref = dict(intra=host_mean(pool == 1), extra=host_mean(pool == 0), contact=host_mean(c), contact_weight=host_mean(np.exp(rho_D * c)),
               phase=host_mean(phase), phase_std=np.sqrt(np.maximum(host_mean(phase * phase) - host_mean(phase) ** 2, 0.0)))
    for backend, device in (("jax", None), ("torch", "cpu")):
        if backend == "torch":
            pytest.importorskip("torch")
        maps = sm.tier_maps("se", t, 3.0, backend=backend, device=device, b0_direction=u)
        for name in ("intra", "extra"):
            np.testing.assert_allclose(maps["pool"][name], ref[name], atol=1e-6, equal_nan=True)
        for name in ("contact", "contact_weight", "phase", "phase_std"):
            np.testing.assert_allclose(maps[name], ref[name], atol=1e-6, equal_nan=True), (backend, name)
        bare = sm.tier_maps("se", backend=backend, device=device)
        assert bare["contact_weight"] is None and bare["phase"] is None and bare["phase_std"] is None
        np.testing.assert_allclose(bare["contact"], ref["contact"], atol=1e-6, equal_nan=True)
        along_z = sm.tier_maps("se", t, 3.0, backend=backend, device=device)
        assert np.nanmax(np.abs(along_z["phase_std"] - maps["phase_std"])) > 0


def test_a_direction_the_host_has_not_contracted_is_contracted_on_the_device(layout_field):
    """The torch backend contracts the field channels for a new direction on its device: the per-row terms equal the
    host's contraction, the tiered image at that direction equals the host-fed image, the pair is cached once when
    resident and made once per call otherwise, and the host holds no copy of the terms."""
    torch = pytest.importorskip("torch")
    from dmipy_sim.spec.tissue import Tissue
    col, merged, grid, n_t, tmp = layout_field
    se = d.pgse([[0, 0, 1]], 0.2e-3, 0.5e-3, gradient_strengths=0.05, n_t=n_t, slew_rate=np.inf)
    out = str(tmp / "sf_device")
    write_shape_moments(col, {"se": se}, out, tol=1e-9, chunk_rows=7, tiers=True)
    t = Tissue(T2={"intra": 0.03, "extra": 0.08, "myelin": 0.01}, T1={"intra": 0.9, "extra": 1.4, "myelin": 0.3}, rho2=1e-5, chi_iso=-1e-7, chi_aniso=-1.5e-8)
    u = np.array([0.6, 0.0, 0.8]); b = np.array([1e9, 3e9, 0.0]); dirs = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    host = ShapeMoments(out); g, _ = host._tier_group("se")
    iso_h, aniso_h = host._field_terms(g, u)
    dev = ShapeMoments(out)
    iso_d, aniso_d = dev._field_terms(g, u, device="cpu")
    np.testing.assert_allclose(iso_d.numpy(), iso_h, atol=1e-6); np.testing.assert_allclose(aniso_d.numpy(), aniso_h, atol=1e-6)
    S_host, _ = host.image("se", b, dirs, tissue=t, scanner=3.0, b0_direction=u)                       # jax, the host terms
    S_dev, _ = dev.image("se", b, dirs, tissue=t, scanner=3.0, b0_direction=u, backend="torch", device="cpu")
    assert not any(k.startswith("field_") for k in dev._host), "the device route keeps no host copy of the terms"
    assert sum(k[1].startswith("field_iso_") for k in dev._device) == 1
    np.testing.assert_allclose(S_dev, S_host, atol=5e-6, equal_nan=True)
    S_again, _ = dev.image("se", b, dirs, tissue=t, scanner=3.0, b0_direction=u, backend="torch", device="cpu")
    np.testing.assert_array_equal(S_again, S_dev)
    S_once, _ = ShapeMoments(out).image("se", b, dirs, tissue=t, scanner=3.0, b0_direction=u, backend="torch", device="cpu", resident=False)
    np.testing.assert_allclose(S_once, S_dev, atol=1e-7, equal_nan=True)


def test_the_preloaded_field_terms_are_the_ones_the_image_uses(layout_field):
    """preload() names the field terms with the same direction key image() looks them up by, so a preloaded layout
    contracts nothing at the default direction (the pool's parent preloads, its worker inherits)."""
    col, merged, grid, n_t, tmp = layout_field
    sm = ShapeMoments(str(tmp / "sf_device"))
    sm.preload(["se"])
    g, _ = sm._tier_group("se")
    sm._set_b0_direction((0.0, 0.0, 1.0))
    assert all(name in sm._host for name in sm._tier_columns("se")), sorted(sm._host)
    calls = []
    original = sm._field_terms
    sm._field_terms = lambda *a, **k: calls.append(a) or original(*a, **k)
    from dmipy_sim.spec.tissue import Tissue
    t = Tissue(T2={"intra": 0.03, "extra": 0.08, "myelin": 0.01}, T1={"intra": 0.9, "extra": 1.4, "myelin": 0.3}, rho2=1e-5, chi_iso=-1e-7, chi_aniso=-1.5e-8)
    sm.image("se", np.array([1e9]), np.array([[0, 0, 1.0]]), tissue=t, scanner=3.0)
    assert calls == [], "the default direction's terms were contracted again"
