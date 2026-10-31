"""The time-segment layout (dmipy-sim#505, #770): partitions of the walk's time, the segment vectors contracted at the
source -- from the walk's float coefficients beside its pack, or from a quantised pack -- the widths derived per segment
and tier field, per-block rows consolidated into tiles; the image of every sequence piecewise constant on a stored
partition (a PGSE, a stimulated echo's three segments, a double diffusion encoding's four) is the sibling pack's replay
within the layout's rounding term, bare and with the tiers."""
from __future__ import annotations
import json
import os

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.replay.bank import build_replay_pack, merge_packs
from dmipy_sim.replay.compression import default_envelope
from dmipy_replay.replay.time_segments import FORMAT, Partition, Shape, read_manifest
from dmipy_sim.replay.time_segments import (
    TimeSegmentLayout,
    _walk_view,
    rounding_eps,
    segment_rows,
    write_time_segment_layout,
)
from dmipy_sim.spec import walk_spec, StratifiedByVoxel
from dmipy_replay.spec.tissue import Tissue

UM = 1e-6
D0 = 2e-9
G_MAX = 1.0                                                        # T/m: the envelope's gradient (the walk names no class)
N_T, TE = 201, 0.02                                                # a 0.1 ms grid over a 20 ms echo
TISSUE = Tissue(T2={"intra": 0.07, "extra": 0.06, "myelin": 0.015}, T1={"intra": 1.0, "extra": 1.2, "myelin": 0.4},
                rho2=2e-5, chi_iso=-1e-7, chi_aniso=-1e-8)


def _sequences():
    """A PGSE, a PGSTE and a DDE (two bipolar pairs about the spin echo's pulse) on a 0.1 ms grid, and the stimulated
    echo's mixing time."""
    se = d.pgse([[0, 0, 1]], 4e-3, 10e-3, gradient_strengths=0.1, TE=TE, n_t=N_T, slew_rate=np.inf)
    ste = d.pgste([[0, 0, 1]], 3e-3, 10e-3, gradient_strengths=0.1, n_t=N_T, slew_rate=np.inf, ste_flip_angles=(90.0, 90.0, 90.0))
    G = np.zeros((1, N_T, 3), np.float32)
    for i0, v in ((5, [0, 0, 0.1]), (35, [0, 0, -0.1]), (105, [0.1, 0, 0]), (135, [-0.1, 0, 0])):
        G[0, i0:i0 + 25, :] = v
    dde = se.with_gradient(G)
    lobes = Partition.of("lobes", ste).segments
    eg = np.asarray(ste.effective_gate)
    z = np.flatnonzero(eg == 0); z = z[(z * ste.dt > lobes[0][1]) & (z * ste.dt < lobes[1][0])]
    mixing = (z[0] * ste.dt, (z[-1] + 1) * ste.dt)
    return se, ste, dde, mixing


def _partitions():
    se, ste, dde, mixing = _sequences()
    lobes = Partition.of("lobes", ste).segments
    return {"pgse": Partition.of("pgse", se), "pgste": Partition.of("pgste", ste, segments=[lobes[0], mixing, lobes[1]]),
            "dde": Partition.of("dde", dde)}


@pytest.fixture(scope="module")
def myelin(tmp_path_factory):
    """A myelinated cylinder walked with its closed-form field, the contact and the pools (1200 walkers, 30 ms); its
    pack at a declared floor target on a 2 x 2 voxel grid; the segment rows from the walk and from the pack and their
    layouts."""
    from dmipy_sim.fields.hollow_cylinder import HollowCylinderField
    from dmipy_replay.phantom.grid import Grid
    tmp = tmp_path_factory.mktemp("segments")
    g = d.MyelinatedCylinder(2.0 * UM, 2.8 * UM, (0, 0, 1), D0, D0)
    w = d.simulate_trajectories(1200, D0, g, 0.03, seed=7, window_T=0.03, field_basis=HollowCylinderField.of(g), require_gpu=False)
    r0 = np.asarray(w.position_coeffs)[:, 0, 0, :]
    lo, hi = r0.min(0), r0.max(0)
    vs = np.array([(hi[0] - lo[0]) / 2 * 1.001, (hi[1] - lo[1]) / 2 * 1.001, 1e-5])
    grid = Grid(shape=(2, 2, 1), voxel_size_m=tuple(vs), origin_m=(lo[0] - 1e-9 + vs[0] / 2, lo[1] - 1e-9 + vs[1] / 2, 0.0))
    pk = build_replay_pack(w, id="t/segments", license="x", citation="x", sigma_star=0.02, planned_floor=0.02, voxel_grid=grid)
    parts = _partitions()
    rec_w = segment_rows(w, parts, str(tmp / "rows_walk"), grid=grid, pack=pk, g_max=G_MAX, tiers=True, background=True)
    rec_p = segment_rows(pk, parts, str(tmp / "rows_pack"), grid=grid, g_max=G_MAX, tiers=True, background=True)
    mw = write_time_segment_layout([str(tmp / "rows_walk")], str(tmp / "lay_walk"))
    mp = write_time_segment_layout([str(tmp / "rows_pack")], str(tmp / "lay_pack"))
    return dict(walk=w, pack=pk, grid=grid, parts=parts, rec_w=rec_w, rec_p=rec_p, mw=mw, mp=mp, tmp=tmp,
                lw=TimeSegmentLayout(str(tmp / "lay_walk")), lp=TimeSegmentLayout(str(tmp / "lay_pack")))


def _acquisitions():
    se, ste, dde, _ = _sequences()
    rng = np.random.default_rng(0)
    u = rng.normal(size=(6, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    b = np.r_[np.full(6, 1e9), np.full(6, 3e9), 0.0]; dirs = np.concatenate([u, u, [[0, 0, 1]]])
    G = np.asarray(dde.G)[0]
    return dict(pgse=d.pgse(dirs.tolist(), 4e-3, 10e-3, bvalues=b, TE=TE, n_t=N_T, slew_rate=np.inf),
                pgste=d.pgste(dirs.tolist(), 3e-3, 10e-3, bvalues=b, n_t=N_T, slew_rate=np.inf, ste_flip_angles=(90.0, 90.0, 90.0)),
                dde=d.ScannerSequence(G=np.stack([G * a for a in (0.3, 0.7, 1.0)] + [G * np.array([[-1.0, 1.0, 1.0]]) * 0.5]),
                                      dt=se.dt, rf=se.rf, readout=se.readout))


def _per_voxel(src, grid, seq, **kw):
    """``(n_vox, n_meas)``: the per-voxel ensemble magnitude of ``src``'s replay (walker by walker, start voxels)."""
    w, ew, E = src.walker_signals(seq, tol=None, **kw)
    ijk, _ = grid.bin(src.r0); v = np.ravel_multi_index(ijk.T, grid.shape)
    num = np.stack([np.bincount(v, (ew * E[:, k]).real, minlength=grid.n_voxels)
                    + 1j * np.bincount(v, (ew * E[:, k]).imag, minlength=grid.n_voxels) for k in range(E.shape[1])], 1)
    return np.abs(num / np.bincount(v, w, minlength=grid.n_voxels)[:, None])


def _term(sigma2, floor):
    """The rounding's term on a voxel's signal: ``3 sigma`` times its Monte-Carlo floor (a random phase of rms ``sigma``
    per walker moves the mean by ``sigma`` times its own spread) plus the bias ``sigma^2 / 2``."""
    return 3.0 * np.sqrt(sigma2) * floor + 0.5 * sigma2


def test_a_partition_is_read_from_its_sequence():
    """A PGSE is two lobes under the gate's +1 and -1, a stimulated echo's mixing time is a segment its gate stores, a
    DDE's two bipolar pairs are four lobes; ``b_matrix`` is the b-value of refocused levels (the builder's own b to its
    rectangular rule), and a PGSE's reach is ``min(G_max, sqrt(b_max / b_unit))``; a segment across a pulse is refused."""
    from dmipy_replay.acquisition.waveforms import calc_b
    parts = _partitions()
    se, ste, dde, mixing = _sequences()
    assert parts["pgse"].gate == (1, -1) and parts["pgste"].gate == (1, 0, -1) and parts["dde"].gate == (1, 1, -1, -1)
    assert parts["pgste"].segments[1] == pytest.approx(mixing)
    for name, seq in (("pgse", se), ("dde", dde)):
        L = parts[name].levels(seq)
        b = np.einsum("msc,st,mtc->m", L, parts[name].b_matrix(), L)
        np.testing.assert_allclose(b, calc_b(seq), rtol=2e-3)
    b_unit = float(calc_b(se)[0]) / 0.1 ** 2
    reach = parts["pgse"].reach(G_MAX, 3e9)
    np.testing.assert_allclose(reach, min(G_MAX, np.sqrt(3e9 / b_unit)), rtol=2e-3)
    assert parts["pgste"].reach(G_MAX, 3e9)[1] == 0.0
    lo = parts["pgse"].segments
    with pytest.raises(ValueError, match="changes sign"):
        Partition.of("across", se, segments=[lo[0], (lo[0][1], lo[1][1])])


def test_a_sequence_that_does_not_fit_is_refused_by_name():
    """Off the segments, not constant on one, unbalanced, on a stored segment, off the grid: each refused by name."""
    parts = _partitions()
    se, ste, dde, _ = _sequences()
    G = np.asarray(se.G).copy(); G[0, 2, 2] = 0.05
    with pytest.raises(ValueError, match="outside the segments"):
        parts["pgse"].levels(se.with_gradient(G))
    G = np.asarray(se.G).copy(); i = int(round(parts["pgse"].segments[0][0] / se.dt)); G[0, i, 2] *= 0.5
    with pytest.raises(ValueError, match="not constant"):
        parts["pgse"].levels(se.with_gradient(G))
    G = np.asarray(se.G).copy(); i = int(round(parts["pgse"].segments[1][0] / se.dt)); G[0, i:i + 40, 2] *= 0.5
    with pytest.raises(ValueError, match="net moment"):
        parts["pgse"].levels(se.with_gradient(G))
    coarse = d.pgse([[0, 0, 1]], 4e-3, 10e-3, gradient_strengths=0.1, TE=TE, n_t=61, slew_rate=np.inf)
    with pytest.raises(ValueError, match="grid"):
        parts["dde"].levels(coarse)
    p = Partition("stored", ((0.0, 1e-3), (1e-3, 2e-3), (2e-3, 3e-3)), (1, 0, -1), (Shape.constant(),) * 3)
    G = np.zeros((1, N_T, 3), np.float32); G[0, 0:10, 0] = 0.1; G[0, 10:20, 0] = 0.1; G[0, 20:30, 0] = 0.2
    with pytest.raises(ValueError, match="stores"):
        p.levels(se.with_gradient(G))


@pytest.mark.parametrize("source", ["pack", "walk"])
def test_the_rows_do_not_depend_on_the_walker_ranges_they_are_read_in(myelin, source, tmp_path):
    """``segment_rows`` read a few walkers at a time (``slice_bytes`` for about 90 of the 1200, so the pools' walkers
    fall in different ranges) writes every file byte for byte as the rows read in one range -- the segment vectors,
    the pool, the gated contact, the field channels, the background, the weights and the voxels, in (voxel, pool)
    order -- and the same ``rows.json`` but its creation time: the partitions' and groups' largest values (those of
    the stored float32 rows), the pool count and the pack's quantisation terms. From the pack (the quantised float32 production container) and from the
    walk (its float coefficients, decoded in float64)."""
    from dmipy_replay.replay.compression import position_shape
    from dmipy_sim.replay.compression import slice_walkers
    M = myelin
    src, kw = (M["pack"], {}) if source == "pack" else (M["walk"], dict(pack=M["pack"]))
    one = 1 << 40                                                       # every walker in one range
    view = M["pack"] if source == "pack" else _walk_view(M["walk"])
    few = 90 * (one // slice_walkers(view.arrays, one))                 # the bytes of 90 walkers' coefficients
    assert slice_walkers(view.arrays, few) == 90 and position_shape(view.arrays)[0] == 1200
    recs = {}
    for name, sb in (("whole", one), ("ranges", few)):
        recs[name] = segment_rows(src, M["parts"], str(tmp_path / name), grid=M["grid"], g_max=G_MAX, tiers=True,
                                  background=True, slice_bytes=sb, **kw)
    files = sorted(os.listdir(tmp_path / "whole"))
    assert files == sorted(os.listdir(tmp_path / "ranges")) and "pool.npy" in files and any(f.startswith("bg_") for f in files)
    assert not [f for f in files if f.startswith(".")]                  # the spilled columns are gone
    for f in files:
        if f != "rows.json":
            assert (tmp_path / "whole" / f).read_bytes() == (tmp_path / "ranges" / f).read_bytes(), f
    a, b = (json.loads((tmp_path / n / "rows.json").read_text()) for n in ("whole", "ranges"))
    for r in (a, b):
        r.pop("created")
    assert a == b
    top = lambda f: np.abs(np.load(tmp_path / "ranges" / f)).max(axis=0).tolist()     # the records are the stored rows'
    for name, p in a["partitions"].items():
        assert p["amax"] == top(p["column"])
    for gname, g in a["groups"].items():
        assert g["background_amax"] == top(f"bg_{gname}.npy") and g["field"]["amax"] == top(f"field_{gname}.npy")
        assert g["contact_amax"] == top(f"contact_{gname}.npy")
    g = next(iter(a["groups"].values()))
    assert g["pools"] >= 2 and g["contact"] and g["field"]["channels"] and g["background_amax"]
    if source == "pack":
        assert any(p["quantisation"] > 0 for p in a["partitions"].values()) and g["quantisation"]["contact"] > 0


def test_the_images_equal_the_sibling_packs_replay(myelin):
    """The acceptance of #770 on a walk with contact and field: a PGSE, a PGSTE (the stimulated echo's group) and a DDE,
    bare and with T2 / T1 / rho2 / the field at 7 T, from the layout written from the walk's float coefficients and from
    the one written from the quantised pack, against the pack's replay voxel by voxel -- within the rounding's term
    (the pack's quantisation added for the walk's layout), and the walk's layout against the walk's own float replay
    within its own rounding alone."""
    M = myelin
    view = _walk_view(M["walk"])
    for name, seq in _acquisitions().items():
        rw, rp = M["mw"]["partitions"][name]["rounding"], M["mp"]["partitions"][name]["rounding"]
        for kw in ({}, dict(tissue=TISSUE, scanner=7.0)):
            ref = _per_voxel(M["pack"], M["grid"], seq, **kw)
            exact = _per_voxel(view, M["grid"], seq, **kw)
            Sw, fw = M["lw"].image(seq, **kw); Sp, fp = M["lp"].image(seq, **kw)
            Sw, Sp = Sw.reshape(ref.shape), Sp.reshape(ref.shape)
            fl = float(np.nanmax(fw))
            assert np.abs(Sw - exact).max() < _term(rw["variance"], fl), (name, kw)
            assert np.abs(Sw - ref).max() < _term(rw["variance"] + rp["source_quantisation"], fl), (name, kw)
            assert np.abs(Sp - ref).max() < _term(rp["variance"], fl), (name, kw)
            assert np.abs(Sw - ref).max() < 0.25 * fl                     # well inside the Monte-Carlo floor


def test_the_widths_are_derived_per_segment_and_the_walk_needs_no_more_bits_than_the_pack(myelin):
    """The manifest records the rule per segment: ``step = sqrt(12 R / n) / reach`` with ``R`` the floor's budget
    (``eps^2``; from the pack ``eps^2`` less the pack's band quantisation contracted through the partition), each
    component's bits and scale against its largest value, its word and offset; a segment its gate stores holds none.
    For one floor the walk's widths are never wider than the pack's and its budget is the larger."""
    M = myelin
    eps = rounding_eps(0.02)
    for name in M["parts"]:
        pw, pp = M["mw"]["partitions"][name], M["mp"]["partitions"][name]
        assert pw["rounding"]["budget"] == pytest.approx(eps ** 2) and pw["rounding"]["source_quantisation"] == 0.0
        assert pp["rounding"]["budget"] == pytest.approx(eps ** 2 - pp["rounding"]["source_quantisation"])
        assert pp["rounding"]["source_quantisation"] > 0.0
        assert pw["rounding"]["variance"] <= pw["rounding"]["budget"] * (1 + 1e-9)
        n = sum(1 for s in pw["segments"] if s["q_max"] > 0)
        for sw, sp in zip(pw["segments"], pp["segments"]):
            assert set(sw) >= {"q_max", "floor_target", "bits", "offsets", "word", "scale", "step"}
            if sw["q_max"] == 0:
                assert sw["bits"] == [0, 0, 0] and sp["bits"] == [0, 0, 0]
                continue
            assert sw["step"] == pytest.approx(np.sqrt(12 * eps ** 2 / n) / sw["q_max"])
            assert all(bw <= bp for bw, bp in zip(sw["bits"], sp["bits"]))
            assert all(s <= sw["step"] * (1 + 1e-9) for s in sw["scale"])
        assert pw["n_words"] <= pp["n_words"]
    assert M["mw"]["source"]["kind"] == "walk" and M["mp"]["source"]["kind"] == "pack"
    assert M["mw"]["source"]["walk"]["starts_sha256"] and M["mw"]["source"]["pack"]["id"] == "t/segments"
    assert M["mw"]["certificate"]["terms"] == M["pack"].meta["fidelity"]["terms"]


def test_the_tier_columns_are_words(myelin):
    """Every tier and background column is uint32 words; a group's contact and field channels share one word array
    (the basis's twelve components, RPK.md 5.3, iso_P_zz formed by the reader; a component zero on every row not stored), its background
    another; the bytes per walker of the tiers are a fraction of the float32 columns' 52 + 4 + 12; the decoded contact
    and channels are the rows' within half a step."""
    M = myelin
    lay = M["lw"]
    for g, grp in M["mw"]["groups"].items():
        assert np.load(os.path.join(lay.path, grp["column"]), mmap_mode="r").dtype == np.uint32
        assert np.load(os.path.join(lay.path, grp["background"]["column"]), mmap_mode="r").dtype == np.uint32
        assert "iso_P_zz" not in grp["field"]["channels"] and len(grp["field"]["channels"]) == 12
        assert set(grp["field"]["stored"]) <= set(grp["field"]["channels"])
        assert 4 * (grp["tier_words"]["n_words"] + grp["background"]["n_words"]) < 0.5 * (52 + 4 + 12)
        cols = lay.tier_columns(g)
        rows = os.path.join(str(M["tmp"]), "rows_walk")
        vox = np.load(os.path.join(rows, "voxel.npy"))
        contact = np.load(os.path.join(rows, f"contact_{g}.npy"))
        step = next(f["scale"] for f in grp["tier_words"]["fields"] if f["name"] == "contact")
        got = np.sort(cols["contact"][np.asarray(lay._column("w")) > 0])
        np.testing.assert_allclose(got, np.sort(contact), atol=0.5 * step + 1e-12)
        assert len(vox) == len(contact) == M["walk"].n_walkers
    assert M["mw"]["bytes_per_walker"]["total"] == 4 + 1 + sum(4 * p["n_words"] for p in M["mw"]["partitions"].values()) + \
        sum(4 * (grp["tier_words"]["n_words"] + grp["background"]["n_words"]) for grp in M["mw"]["groups"].values())


def test_a_sequence_beyond_the_envelope_and_a_foreign_pack_are_refused(myelin):
    """A row above b_max or G_max is refused by name; a walk's rows need its own pack."""
    M = myelin
    with pytest.raises(ValueError, match="b_max"):
        M["lw"].image(d.pgse([[0, 0, 1]], 4e-3, 10e-3, bvalues=[5e9], TE=TE, n_t=N_T, slew_rate=np.inf))
    se, _, dde, _ = _sequences()
    with pytest.raises(ValueError, match="G_max"):
        M["lw"].image(dde.with_gradient(np.asarray(dde.G) * 12.0))
    with pytest.raises(ValueError, match="fits no stored partition"):
        M["lw"].image(d.pgse([[0, 0, 1]], 3e-3, 10e-3, bvalues=[1e9], TE=TE, n_t=N_T, slew_rate=np.inf))
    other = M["walk"].take(np.arange(M["walk"].n_walkers)[::-1])
    with pytest.raises(ValueError, match="not this walk's pack"):
        segment_rows(other, M["parts"], str(M["tmp"] / "foreign"), grid=M["grid"], pack=M["pack"], g_max=G_MAX)
    with pytest.raises(ValueError, match="sibling"):
        segment_rows(M["walk"], M["parts"], str(M["tmp"] / "nopack"), grid=M["grid"], g_max=G_MAX)


def test_a_layout_of_another_format_is_refused_by_name(myelin, tmp_path):
    """A ``time_segments/1`` manifest (one vector per shape) is not read: it is written again from its walk's rows."""
    import shutil
    old = str(tmp_path / "old")
    shutil.copytree(myelin["lw"].path, old)
    m = json.load(open(os.path.join(old, "manifest.json")))
    m["format"] = "time_segments/1"
    json.dump(m, open(os.path.join(old, "manifest.json"), "w"))
    with pytest.raises(ValueError, match=f"not a {FORMAT} layout"):
        TimeSegmentLayout(old)
    with pytest.raises(ValueError, match="time_segments/1"):
        read_manifest(old)


# ------------------------------------------------------------------ several blocks, the mechanics of the reader
@pytest.fixture(scope="module")
def blocks(spec_grid):
    """Two blocks of the three-strand spec walked with the sheath's field, the contact and the pools, each packed and
    its rows written from the walk; the layout of both blocks and the merged pack."""
    spec, grid, tmp = spec_grid
    packs, dirs = [], []
    for b in (0, 1):
        want = np.zeros(grid.shape, np.int64); want[b] = 12
        w = walk_spec(spec, T_max=8e-4, block_h=2e-4, seed=21 + b, require_gpu=False, adaptive_steps=True, K=3,
                      seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": want, "intra": want}))
        # certified up to ten times the sheath's chi, so that the layout's field words resolve its field at the
        # fixture's coarse floor (at the sheath's own the field rounds to zero on every row of 24 walkers)
        pk = build_replay_pack(w, id=f"t/sb{b}", license="x", citation="x", K=3, susc_path_K=4, voxel_grid=grid,
                               envelope=dict(default_envelope(), chi_max=2e-6))
        packs.append(pk)
        dirs.append(str(tmp / f"sb_rows{b}"))
        segment_rows(w, _strand_partitions(pk.n_t), dirs[-1], grid=grid, pack=pk, g_max=60.0, floor_target=0.3,
                     tiers=True, background=True)
    m = write_time_segment_layout(dirs, str(tmp / "sb_layout"))
    return TimeSegmentLayout(str(tmp / "sb_layout")), merge_packs(packs, id="t/sb-merged"), grid, packs[0].n_t, m, tmp


def _strand_partitions(n_t):
    se = d.pgse([[0, 0, 1]], 0.2e-3, 0.5e-3, gradient_strengths=0.05, n_t=n_t, slew_rate=np.inf)
    return {"se": se}


def _strand_protocol(n_t, n=4):
    rng = np.random.default_rng(5)
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    b = np.r_[np.full(n, 1e9), 0.0]; dirs = np.concatenate([u, [[0, 0, 1]]])
    return d.pgse(dirs.tolist(), 0.2e-3, 0.5e-3, bvalues=b, n_t=n_t, slew_rate=np.inf)


def test_the_blocks_consolidate_into_one_layout(blocks):
    """Two blocks' rows tiled into one layout: every row once (the per-voxel weight sums are the merged pack's, a padding
    row weighs nothing), each voxel's halves alternate, and the image is the merged pack's replay per voxel within the
    rounding's term, bare and with the tiers."""
    lay, merged, grid, n_t, m, tmp = blocks
    assert lay.n_rows == merged.n_walkers and (np.asarray(lay._column("w")) > 0).sum() == merged.n_walkers
    ijk, _ = grid.bin(merged.r0); v = np.ravel_multi_index(ijk.T, grid.shape)
    np.testing.assert_allclose(lay.weights.sum(1), np.bincount(v, merged.spin_weights, minlength=lay.n_vox), rtol=1e-6)
    assert [b["n_rows"] for b in m["source"]["blocks"]] == [len(np.load(str(tmp / f"sb_rows{i}" / "voxel.npy"))) for i in (0, 1)]
    seq = _strand_protocol(n_t)
    t = Tissue(T2={"intra": 0.03, "extra": 0.08, "myelin": 0.01}, T1={"intra": 0.9, "extra": 1.4, "myelin": 0.3},
               rho2=1e-5, chi_iso=-1e-7, chi_aniso=-1.5e-8)
    for kw in ({}, dict(tissue=t, scanner=3.0)):
        S, floor = lay.image(seq, **kw)
        ref = _per_voxel(merged, grid, seq, **kw)
        got = S.reshape(ref.shape)
        live = np.isfinite(got[:, 0])
        sig2 = m["partitions"]["se"]["rounding"]["variance"]
        assert np.abs(got[live] - ref[live]).max() < _term(sig2, float(np.nanmax(floor)))
    with pytest.raises(ValueError, match="gate"):
        lay.image(d.pgse([[0, 0, 1]], 0.2e-3, 0.5e-3, bvalues=[1e9], n_t=2 * n_t - 1, slew_rate=np.inf), tissue=t, scanner=3.0)


def test_a_pool_the_first_block_lacks_keeps_its_relaxation(spec_grid, tmp_path):
    """A first block of extra water alone and a second with the intra pool too: the layout's pool count is the
    largest any block's rows address, so each pool's T2 weights its own rows in the image."""
    spec, grid, _ = spec_grid
    packs, dirs = [], []
    for b, pools in ((0, ("extra",)), (1, ("extra", "intra"))):
        want = np.zeros(grid.shape, np.int64); want[b] = 24 // len(pools)            # as many walkers in each block
        per = {p: (want if p in pools else np.zeros(grid.shape, np.int64)) for p in ("extra", "intra")}
        w = walk_spec(spec, T_max=8e-4, block_h=2e-4, seed=31 + b, require_gpu=False, adaptive_steps=True, K=3,
                      seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel=per))
        pk = build_replay_pack(w, id=f"t/pl{b}", license="x", citation="x", K=3, susc_path_K=4, voxel_grid=grid)
        packs.append(pk); dirs.append(str(tmp_path / f"pl_rows{b}"))
        segment_rows(w, _strand_partitions(pk.n_t), dirs[-1], grid=grid, pack=pk, g_max=60.0, floor_target=0.3, tiers=True)
    m = write_time_segment_layout(dirs, str(tmp_path / "pl_layout"))
    assert m["tiers"]["pools"] == 2
    lay, merged = TimeSegmentLayout(str(tmp_path / "pl_layout")), merge_packs(packs, id="t/pl-merged")
    seq = _strand_protocol(packs[0].n_t)
    t = Tissue(T2={"intra": 0.0005, "extra": 0.08, "myelin": 0.01})
    S, floor = lay.image(seq, tissue=t)
    ref = _per_voxel(merged, grid, seq, tissue=t)
    got = S.reshape(ref.shape); live = np.isfinite(got[:, 0])
    sig2 = m["partitions"]["se"]["rounding"]["variance"]
    assert np.abs(got[live] - ref[live]).max() < _term(sig2, float(np.nanmax(floor)))


def test_blocks_that_disagree_are_refused(blocks, spec_grid):
    """Rows of two layouts (another partition) are not consolidated together."""
    lay, merged, grid, n_t, m, tmp = blocks
    rec = json.load(open(str(tmp / "sb_rows1" / "rows.json")))
    rec["partitions"]["se"]["segments"][0][0] += 1e-5
    bad = str(tmp / "sb_rows_bad"); os.makedirs(bad, exist_ok=True)
    import shutil
    for f in os.listdir(str(tmp / "sb_rows1")):
        shutil.copy(str(tmp / "sb_rows1" / f), bad)
    json.dump(rec, open(os.path.join(bad, "rows.json"), "w"))
    with pytest.raises(ValueError, match="partitions"):
        write_time_segment_layout([str(tmp / "sb_rows0"), bad], str(tmp / "sb_bad_layout"))


def test_the_torch_backend_is_the_jax_image(blocks):
    """dmipy-sim#510: the same sequence on both backends agrees to float32 arithmetic, bare and tiered, NaN and floor
    alike; two torch calls are identical; a non-resident call leaves nothing on the device; the backend is checked."""
    torch = pytest.importorskip("torch")
    lay, merged, grid, n_t, m, tmp = blocks
    seq = _strand_protocol(n_t, 5)
    t = Tissue(T2={"intra": 0.03, "extra": 0.08, "myelin": 0.01}, rho2=1e-5, chi_iso=-1e-7, chi_aniso=-1.5e-8)
    torch.use_deterministic_algorithms(True)
    for kw in ({}, dict(tissue=t, scanner=3.0, b0_direction=(0.6, 0.0, 0.8))):
        S_j, f_j = lay.image(seq, **kw)
        S_t, f_t = lay.image(seq, backend="torch", device="cpu", **kw)
        np.testing.assert_array_equal(np.isnan(S_j), np.isnan(S_t))
        assert np.nanmax(np.abs(S_j - S_t)) < 5e-6 and np.nanmax(np.abs(f_j - f_t)) < 5e-6
    S_t2, _ = lay.image(seq, backend="torch", device="cpu")
    np.testing.assert_array_equal(np.nan_to_num(S_t2), np.nan_to_num(lay.image(seq, backend="torch", device="cpu")[0]))
    before = set(lay._device)
    lay.image(seq, backend="torch", device="cpu", resident=False)
    assert set(lay._device) == before
    with pytest.raises(ValueError, match="backend"):
        lay.image(seq, backend="no-such-backend")


def test_the_field_direction_is_a_knob_contracted_in_the_kernel(blocks):
    """The field direction changes the field phase; it enters the kernel as twelve contraction weights, so a layout
    holds no per-direction state and a direction asked again is the same image to the bit."""
    lay, merged, grid, n_t, m, tmp = blocks
    seq = _strand_protocol(n_t, 2)
    t = Tissue(T2={"intra": 0.03, "extra": 0.08, "myelin": 0.01}, rho2=1e-5, chi_iso=-1e-7, chi_aniso=-1.5e-8)
    Sz, _ = lay.image(seq, tissue=t, scanner=3.0)
    Sx, _ = lay.image(seq, tissue=t, scanner=3.0, b0_direction=(1.0, 0.0, 0.0))
    assert np.nanmax(np.abs(Sx - Sz)) > 1e-7
    with pytest.raises(ValueError, match="unit"):
        lay.image(seq, tissue=t, scanner=3.0, b0_direction=(0, 0, 2))
    rng = np.random.default_rng(4)
    dirs = rng.normal(size=(7, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    n_dev = len(lay._device)
    for u in dirs:
        lay.image(seq, tissue=t, scanner=3.0, b0_direction=tuple(u))
    assert len(lay._device) == n_dev and not lay._host
    S_again, _ = lay.image(seq, tissue=t, scanner=3.0, b0_direction=tuple(dirs[0]))
    S_fresh, _ = TimeSegmentLayout(lay.path).image(seq, tissue=t, scanner=3.0, b0_direction=tuple(dirs[0]))
    np.testing.assert_array_equal(S_again, S_fresh)


def _backends():
    out = ["jax"]
    try:
        import torch  # noqa: F401
        out.append("torch")
    except ImportError:
        pass
    return out


@pytest.mark.parametrize("backend", _backends())
def test_the_words_are_decoded_in_the_kernel_to_the_float32_decode(blocks, backend):
    """The kernel's decode of the packed words (a shift up and an arithmetic shift down, times the float32 scale) is
    the float32 decode of :func:`unpack_words` bit for bit, for a partition's words, the contact and the background;
    the field channels the kernel contracts are decoded in float64 (exact: a code of at most 24 bits times a float32
    scale) and summed in channel order with the direction's weights, bit for bit the host's same products in the same
    order."""
    from dmipy_replay.replay.field import iso_weights, q_of_H
    from dmipy_replay.replay.time_segments import _decode, _field_terms_of_words, _layout_of, unpack_words
    from dmipy_sim.replay.time_segments import _device_field_spec, _x64
    lay, merged, grid, n_t, m, tmp = blocks
    g = m["partitions"]["se"]["group"]
    cols = TimeSegmentLayout.columns(m)
    tw = np.asarray(lay._column(f"t_{g}"))
    coded = [(np.asarray(lay._column("p_se")), _layout_of(m["partitions"]["se"])), (tw, _layout_of(m["groups"][g]["tier_words"]))]
    if f"bg_{g}" in cols:
        coded.append((np.asarray(lay._column(f"bg_{g}")), _layout_of(m["groups"][g]["background"])))
    if backend == "torch":
        import torch
        dev = lambda a: torch.as_tensor(a.view(np.int32))
        host = lambda a: a.numpy()
    else:
        import jax.numpy as jnp
        dev = lambda a: jnp.asarray(a.view(np.int32))
        host = np.asarray
    for col, (layout, scale) in coded:
        sc = scale if backend == "jax" else __import__("torch").as_tensor(scale)
        ref = unpack_words(col, scale, layout)
        for f, v in enumerate(_decode(dev(col), sc, layout)):
            if v is not None:
                assert host(v).dtype == np.float32
                np.testing.assert_array_equal(host(v), ref[..., f])
    u = np.array([0.6, -0.3, 0.74]); u /= np.linalg.norm(u)
    spec = lay._field_spec(g, u, 1.0, 1.0)
    assert spec is not None and len(spec[0]) == len(m["groups"][g]["field"]["stored"])
    with _x64(backend):
        iso, aniso = _field_terms_of_words(dev(tw), _device_field_spec(spec, backend, "cpu"), backend == "torch")
        iso = host(iso)
        aniso = None if aniso is None else host(aniso)
    layout, scale, names = lay._tier_code(g)
    ch = m["groups"][g]["field"]["channels"]
    wts = dict(zip(ch[:6], iso_weights(u))); wts.update(zip(ch[6:12], q_of_H(u)))
    t32 = tw.view(np.int32)
    ref = {}
    for nm in [c for c in ch if c in names]:
        w_, off, b = layout[names.index(nm)]
        k = (t32[..., w_] << (32 - off - b)) >> (32 - b)
        term = k.astype(np.float64) * np.float64(scale[names.index(nm)]) * wts[nm]
        key = "iso" if nm in ch[:6] else "aniso"
        ref[key] = term if key not in ref else ref[key] + term
    assert iso.dtype == np.float64
    np.testing.assert_array_equal(iso, ref["iso"])
    if "aniso" in ref:
        np.testing.assert_array_equal(aniso, ref["aniso"])


@pytest.mark.parametrize("backend", _backends())
def test_a_streamed_image_is_the_resident_image_to_the_bit(blocks, backend):
    """``resident=False`` reads and uploads each chunk of every column on its own and keeps nothing; its image is the
    resident one to the bit -- bare, with every tier at an oblique field, and with a machine's delivery and its own
    gradient -- and so are the tier maps."""
    from dmipy_replay.acquisition.scanners import ScannerLimits
    lay, merged, grid, n_t, m, tmp = blocks
    seq = _strand_protocol(n_t, 3)
    t = Tissue(T2={"intra": 0.03, "extra": 0.08, "myelin": 0.01}, rho2=1e-5, chi_iso=-1e-7, chi_aniso=-1.5e-8)
    u = (0.6, 0.0, 0.8)
    fresh = TimeSegmentLayout(lay.path)
    fresh.TILES = 1                                                   # many chunks, the last one padded
    kw = dict(backend=backend, device="cpu") if backend == "torch" else dict(backend=backend)
    for extra in ({}, dict(tissue=t, scanner=7.0, b0_direction=u)):
        S_r, f_r = fresh.image(seq, resident=True, **kw, **extra)
        S_s, f_s = fresh.image(seq, resident=False, **kw, **extra)
        np.testing.assert_array_equal(S_r, S_s); np.testing.assert_array_equal(f_r, f_s)
    swoop = ScannerLimits.of("hyperfine_swoop_64mT")
    dl = fresh.delivery(seq, swoop, grid.centred_at((0.05, 0.0, 0.0)))
    S_r, _ = fresh.image(seq, resident=True, delivered=dl, **kw)
    S_s, _ = fresh.image(seq, resident=False, delivered=dl, **kw)
    np.testing.assert_array_equal(S_r, S_s)
    n_dev = len(fresh._device)
    fresh.image(seq, resident=False, tissue=t, scanner=3.0, **kw)
    assert len(fresh._device) == n_dev
    a = fresh.tier_maps("se", t, 7.0, b0_direction=u, resident=True, **kw)
    b = fresh.tier_maps("se", t, 7.0, b0_direction=u, resident=False, **kw)
    for k in ("contact", "contact_weight", "phase", "phase_std"):
        np.testing.assert_array_equal(a[k], b[k])


def test_the_field_phase_at_an_oblique_field_is_the_packs(myelin):
    """With the field along an oblique axis and an anisotropic susceptibility alone, an isotropic one alone and both,
    the layout's image is the pack's replay within the same distance as without any field (the rounding's): the
    stored channels meet the direction's twelve weights in the basis's own order (RPK.md 5.3, ``iso_P_zz`` folded into
    the isotropic weights, never formed)."""
    from dataclasses import replace
    from dmipy_replay.acquisition.scanners import ScannerLimits
    M = myelin
    u = np.array([1.0, 0.3, 0.5]); u /= np.linalg.norm(u)
    sc = replace(ScannerLimits.of("siemens_magnetom_terra_7T"), b0_axis=tuple(u))
    seq = _acquisitions()["pgse"]

    def gap(t):
        ref = _per_voxel(M["pack"], M["grid"], seq, tissue=t, scanner=sc)
        S, _ = M["lw"].image(seq, tissue=t, scanner=float(sc.field_T), b0_direction=tuple(u))
        return float(np.abs(S.reshape(ref.shape) - ref).max())
    base = gap(replace(TISSUE, chi_iso=0.0, chi_aniso=0.0))
    for t in (replace(TISSUE, chi_iso=0.0, chi_aniso=-2e-7), replace(TISSUE, chi_iso=-2e-7, chi_aniso=0.0), replace(TISSUE, chi_iso=-2e-7, chi_aniso=-2e-7)):
        assert gap(t) < base * 1.05 + 1e-5, (t.chi_iso, t.chi_aniso)


def test_the_tier_maps_and_the_files(blocks):
    """``tier_maps`` are the walk-weighted means of the decoded tier columns per voxel; ``files`` lists every column a
    layout needs (restricted to some partitions it keeps the tiers); the weights are read once per layout."""
    lay, merged, grid, n_t, m, tmp = blocks
    t = Tissue(T2={"intra": 0.03, "extra": 0.08, "myelin": 0.01}, rho2=1e-5, chi_iso=-1e-7, chi_aniso=-1.5e-8)
    g = m["partitions"]["se"]["group"]
    tiles = np.asarray(lay._column("tiles")); w = np.asarray(lay._column("w"), np.float64)
    c = lay.tier_columns(g)["contact"].astype(np.float64)
    num = np.bincount(tiles, (w * c).sum(1), minlength=2 * lay.n_vox).reshape(-1, 2).sum(1)
    den = np.bincount(tiles, w.sum(1), minlength=2 * lay.n_vox).reshape(-1, 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = np.where(den > 0, num / den, np.nan).reshape(grid.shape)
    maps = lay.tier_maps("se", t, 3.0)
    np.testing.assert_allclose(maps["contact"], ref, atol=1e-6, equal_nan=True)
    assert maps["phase"] is not None and lay.tier_maps("se")["phase"] is None
    files = TimeSegmentLayout.files(m)
    assert "manifest.json" in files and "w.npy" in files and "tiles.npy" in files and "pool.npy" in files
    assert all(os.path.exists(os.path.join(lay.path, f)) for f in files)
    assert m["partitions"]["se"]["column"] in files and m["groups"][g]["column"] in files
    fresh = TimeSegmentLayout(lay.path); reads = []
    column = fresh._column
    fresh._column = lambda name: (reads.append(name), column(name))[1]
    seq = _strand_protocol(n_t, 1)
    fresh.image(seq); n_first = len(reads)
    fresh.image(seq)
    assert "w" in reads[:n_first] and reads[n_first:] == []


def test_from_a_pack_the_tiers_carry_the_packs_own_quantisation(myelin):
    """The layout's certificate is a sum of terms: from a pack it carries the pack's own quantisation of the gated
    contact and field (:func:`tier_quantisation_terms`) beside the tiers' rounding, at the same widths as from the walk,
    and ``err_max`` covers both; from the walk those terms are zero. The stated contact and field terms bound the pack's
    per-walker departure from the walk's float contraction (and are not vacuous for the contact); a floor target below
    the pack's own term is refused by name."""
    from dmipy_sim.replay.time_segments import tier_quantisation_terms, _gate_sequence
    from dmipy_sim.replay.study import Acquisition
    from dmipy_replay.replay.replay import _path_field_channels
    from dmipy_replay.replay.field import field_terms
    M = myelin
    cw, cp = M["mw"]["certificate"], M["mp"]["certificate"]
    for g in M["mp"]["groups"]:
        assert cw["tiers"][g]["source"] == dict(contact=0.0, field=0.0)
        assert cp["tiers"][g]["source"]["contact"] > 0.0 and cp["tiers"][g]["source"]["field"] > 0.0
        for fw, fp in zip(M["mw"]["groups"][g]["tier_words"]["fields"], M["mp"]["groups"][g]["tier_words"]["fields"]):
            assert fw["name"] == fp["name"] and fw["step"] == pytest.approx(fp["step"])      # the same rule from either source
    assert cw["err_max"] == pytest.approx(cw["err_rounding"]) and cp["err_max"] > cp["err_rounding"]
    view = _walk_view(M["walk"])
    env = M["rec_p"]["source"]["envelope"]
    for name in ("pgse", "pgste"):
        gate = _gate_sequence(M["parts"][name].template)
        q_c, q_f = tier_quantisation_terms(M["pack"], gate, env)
        c_p = M["pack"].walker_primitives(Acquisition(gate)).contact; c_w = view.walker_primitives(Acquisition(gate)).contact
        rms_c = env["rho2_over_D_max"] * np.sqrt(np.mean((c_p - c_w) ** 2))
        assert 0.05 * np.sqrt(q_c) < rms_c <= np.sqrt(q_c), (name, rms_c, np.sqrt(q_c))
        f = []
        for src in (M["pack"], view):
            Psi, _ = _path_field_channels(src._prepare(gate, tissue=None, scanner=None, orientation=None, compartment=None), gate)
            f.append(np.asarray(field_terms(np.asarray(Psi), [0.0, 0.0, 1.0])[0]))
        assert env["B0_chi_max_T"] * np.sqrt(np.mean((f[0] - f[1]) ** 2)) <= np.sqrt(q_f), name
    tight = str(M["tmp"] / "rows_pack_tight")
    segment_rows(M["pack"], M["parts"], tight, grid=M["grid"], g_max=G_MAX, tiers=True, floor_target=1e-5)
    with pytest.raises(ValueError, match="pack's own"):
        write_time_segment_layout([tight], str(M["tmp"] / "lay_tight"))
