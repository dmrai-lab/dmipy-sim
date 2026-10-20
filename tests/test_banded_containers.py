"""The integer band containers of the bridge channels and the static C1 column (registry 0.6).

A bridge's sine coefficients fall as 1/k, so a per-band scale with 16 bits on the first bands and 8 beyond keeps
the reconstruction at the nanometre level while the pack shrinks 3.4x at K = 128; the C2 bridge takes the same
container; a walker that never changes pool stores one label. Every reader dequantises in one place, so the
coefficient-space replay of #242 and the dense oracles read either container alike.
"""
from __future__ import annotations

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences as _seqmod
from dmipy_sim.replay import compression as _cx
from dmipy_sim.replay.bank import build_replay_pack, merge_packs
from dmipy_sim.spec.tissue import Tissue

D0 = 2e-9


def test_the_band_container_round_trips_to_nanometres_at_a_third_of_the_bytes():
    rng = np.random.default_rng(0); n_w, n_t = 400, 253
    X = np.cumsum(rng.normal(0, np.sqrt(2 * D0 * 4e-4), (n_w, n_t, 3)), axis=1) + 1e-3      # 100 ms at 1 mm
    a_f, m_f, nb_f = _cx.encode_bridge_dst(X, 128, 1e-4)
    a_b, m_b, nb_b = _cx.encode_bridge_dst(X, 128, 1e-4, container=_cx.TWO_LEVEL_CONTAINER)
    assert nb_f / nb_b > 3.0 and m_b["container"] == [{"bands": [0, 16], "bits": 16}, {"bands": [16, 128], "bits": 8}]
    assert a_b["pos_x_b0"].dtype == np.int16 and a_b["pos_x_b1"].dtype == np.int8 and a_b["pos_x_ends"].dtype == np.float32
    Xf, Xb = _cx.decode(a_f, m_f, n_t=n_t), _cx.decode(a_b, m_b, n_t=n_t)
    assert np.abs(Xb - Xf).max() < 1e-7 and np.sqrt(((Xb - Xf) ** 2).mean()) < 2e-8               # 100 nm max, 20 nm rms
    np.testing.assert_array_equal(_cx.read_position_coeffs(a_b)[:, :2], _cx.read_position_coeffs(a_f)[:, :2])   # the endpoints exact
    with pytest.raises(ValueError, match="covers bands"):
        _cx.encode_bridge_dst(X, 32, 1e-4, container=((16, 16),))
    with pytest.raises(ValueError, match="bits must be 3..16"):
        _cx.encode_bridge_dst(X, 32, 1e-4, container=((None, 2),))


def test_the_c2_container_and_the_static_c1_column():
    rng = np.random.default_rng(1); n_w, n_t = 300, 121
    ell = -np.abs(rng.normal(0, 1e-6, (n_w, n_t))) * (rng.uniform(size=(n_w, n_t)) < 0.3); ell[:, 0] = 0.0
    a_f, m_f = _cx.encode_boundary_bridge(ell, 64, 1e-4, dtype=np.float32)    # continuous bands ~1e-6: float16 subnormal
    a_b, m_b = _cx.encode_boundary_bridge(ell, 64, 1e-4, container=((None, 8),))
    assert "blt_bridge_dst" not in a_b and a_b["blt_b0"].dtype == np.int8 and _cx.has_c2(a_b) and _cx.c2_bands_K(a_b, {}) == 64
    chi = np.ones(n_t); chi[0] = 0.0; chi[80:] = 0.0
    lw_f = _cx.surface_logweight_bridge(a_f, m_f, 3.0, chi, 1e-4); lw_b = _cx.surface_logweight_bridge(a_b, m_b, 3.0, chi, 1e-4)
    assert np.abs(lw_b - lw_f).max() < 5e-3 * np.abs(lw_f).max()
    np.testing.assert_allclose(_cx.decode_boundary_bridge(a_b, m_b, n_t=n_t).sum(1), np.asarray(a_b["blt_endpoint"]), rtol=1e-5)
    lab = np.repeat(rng.integers(0, 3, (n_w, 1)), n_t, axis=1)
    a1, m1 = _cx.encode_occupancy({"comp": lab})
    assert list(a1) == ["comp_static"] and a1["comp_static"].dtype == np.int8 and m1["columns"][0]["kind"] == "static"
    np.testing.assert_array_equal(_cx.decode_occupancy(a1, m1)["comp"], lab[:, 0])            # a static column: one label per walker
    np.testing.assert_allclose(_cx.relaxation_logweight_runs(a1, m1["columns"][0], [0.08, 0.03, 0.01], [1.0, 1.2, 0.3], 4e-4, chi, chi),
                               _cx.relaxation_logweight(lab, [0.08, 0.03, 0.01], [1.0, 1.2, 0.3], 4e-4, chi, chi), rtol=1e-12)
    crossing = lab.copy(); crossing[0, 60:] = 2
    a2, m2 = _cx.encode_occupancy({"comp": crossing})
    assert m2["columns"][0]["kind"] == "label" and "comp_rle_vals" in a2                   # a crossing keeps the runs


@pytest.fixture(scope="module")
def packs(tmp_path_factory):
    walk = d.simulate_trajectories(300, D0, d.Cylinder(2e-6, (0, 0, 1)), 0.01, 5e-4, seed=0, require_gpu=False)
    tmp = tmp_path_factory.mktemp("bands")
    f = build_replay_pack(walk, id="t/float", K=8, license="x", citation="x", blt_temporal_K=8)
    b = build_replay_pack(walk, id="t/bands", K=8, license="x", citation="x", blt_temporal_K=8,
                          position_container=((4, 16), (None, 8)), blt_container="bands", out_path=str(tmp / "bands.rpk"))
    return f, b, tmp


def test_a_pack_in_the_integer_containers_replays_every_knob_like_the_float_one(packs):
    f, b, tmp = packs
    assert "pos_x" not in b.arrays and "pos_x_ends" in b.arrays and _cx.has_c2(b.arrays) and "comp_static" in b.arrays
    assert b.has_relaxation and b.has_surface and b.n_walkers == f.n_walkers and b.K == f.K
    assert b.meta["compression"]["container"] == [{"bands": [0, 4], "bits": 16}, {"bands": [4, 8], "bits": 8}]
    fb, ff = b.meta["fidelity"], f.meta["fidelity"]          # a container costs the certificate its quantisation only
    assert fb["err_truncation"] == ff["err_truncation"] and fb["floor_max"] == ff["floor_max"]
    assert all(0.0 < x["err_quantisation"] <= 0.2 * x["floor_max"] for x in (fb, ff))   # f: the derived default
    Xb, Xf = b.positions(), f.positions()
    assert np.abs(Xb - Xf).max() < 1e-7
    seq = _seqmod.pgse([[1, 0, 0], [0, 0, 1]], 2e-3, 6e-3, bvalues=[1e9, 1e9], TE=9.5e-3, n_t=4 * b.n_t + 1, slew_rate=np.inf)
    for kw in (dict(), dict(tissue=Tissue(T2={"extra": 0.08, "intra": 0.03}, T1={"extra": 1.0, "intra": 1.2})), dict(tissue=Tissue(rho2=1e-5, D=D0))):
        np.testing.assert_allclose(b.replay(seq, **kw), f.replay(seq, **kw), rtol=2e-3)
    from dmipy_sim.replay import read_rpk
    back = read_rpk(str(tmp / "bands.rpk"))
    np.testing.assert_allclose(back.replay(seq), b.replay(seq), rtol=1e-12)
    assert back.arrays["pos_x_b1"].dtype == np.int8
    m = merge_packs([b, back], id="t/merged")                                              # two blocks, each with its scales
    assert m.n_walkers == 2 * b.n_walkers and m.arrays["pos_band_scale"].shape[0] == 2 and m.arrays["band_block"][-1] == 1
    np.testing.assert_allclose(m.positions()[b.n_walkers:], Xb, rtol=0, atol=1e-15)


def test_a_prefix_of_a_banded_pack_replays(packs):
    """ReplayPack.prefix re-encodes from the dequantised coefficients: a banded pack cuts like a float one."""
    f, b, tmp = packs
    TE = 5e-3
    pb, pf = b.prefix(TE), f.prefix(TE)
    seq = _seqmod.pgse([[1, 0, 0]], 1e-3, 3e-3, bvalues=[1e9], TE=TE, n_t=4 * pb.n_t + 1, slew_rate=np.inf)
    np.testing.assert_allclose(pb.replay(seq), pf.replay(seq), rtol=2e-3)



# ------------------------------------------------------------------ the curve and the packed depths

@pytest.mark.parametrize("bits", [3, 4, 5, 6, 7, 9, 12, 13, 15])
def test_a_packed_range_round_trips_the_quantised_values_exactly(bits):
    rng = np.random.default_rng(bits); n_w, m = 257, 24
    lim = 2 ** (bits - 1) - 1
    q = rng.integers(-lim, lim + 1, (n_w, m, 3))
    t = _cx._store_range(q, bits)
    assert t.dtype == np.uint8 and t.shape == (n_w, -(-m * bits // 8), 3)
    np.testing.assert_array_equal(_cx._load_range(t, bits, m), q)
    np.testing.assert_array_equal(_cx._load_range(_cx._store_range(q[..., 0], bits), bits, m), q[..., 0])
    # a row is its values side by side, MSB first, offset-binary
    row = np.unpackbits(_cx._store_range(q[:1, :, 0], bits)[0])[: m * bits].reshape(m, bits)
    np.testing.assert_array_equal(row @ (1 << np.arange(bits - 1, -1, -1)), q[0, :, 0] + 2 ** (bits - 1))


@pytest.mark.parametrize("bits", [3, 5, 7, 11])
def test_quantise_bands_reconstructs_the_integers_times_the_scale_at_every_depth(bits):
    rng = np.random.default_rng(1); B = rng.normal(size=(100, 20, 3)) / np.arange(1, 21)[None, :, None]
    cont = ((8, bits), (None, 8))
    arrays, meta = _cx.quantise_bands(B, cont, "pos", "pos_band_scale")
    got = _cx.dequantise_bands(arrays, meta["container"], "pos", "pos_band_scale")
    lim = 2 ** (bits - 1) - 1; scale = np.abs(B).max(axis=0) / np.where(np.arange(20) < 8, lim, 127)[:, None]
    assert np.abs(got - B).max() <= 0.5 * scale.max() * (1 + 1e-6)
    assert got.shape == B.shape


def test_the_curve_is_the_rule_grouped_into_whole_bytes():
    c = _cx.band_curve(256, 13)
    ranges = [(r["bands"][0], r["bands"][1], r["bits"]) for r in c]
    assert ranges[0][0] == 0 and ranges[-1][1] == 256 and all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    for k0, k1, b in ranges[:-1]:
        assert (k1 - k0) * b % 8 == 0
    bits = np.empty(256, int)
    for k0, k1, b in ranges:
        bits[k0:k1] = b
    ideal = np.clip(np.rint(13 - np.log2(np.arange(1, 257))), 3, 16)
    assert np.all(bits >= np.minimum(ideal, bits)) and bits[0] == 13 and bits[-1] == 5
    assert abs(bits.mean() - ideal.mean()) < 0.5                                    # whole-byte grouping moves the mean by under half a bit
    assert _cx.band_curve(16, 16, b_max=8) == [{"bands": [0, 16], "bits": 8}]


def test_a_pack_with_a_five_bit_range_reads_back_and_replays(tmp_path):
    walk = d.simulate_trajectories(300, D0, d.Cylinder(2e-6, (0, 0, 1)), 0.01, 5e-4, seed=0, require_gpu=False)
    f = build_replay_pack(walk, id="t/f", K=16, license="x", citation="x", blt_temporal_K=16, blt_container="bands",
                          position_container=((4, 16), (None, 8)))
    p = build_replay_pack(walk, id="t/p", K=16, license="x", citation="x", blt_temporal_K=16, blt_container="bands",
                          position_container=((8, 12), (None, 5)), out_path=str(tmp_path / "p.rpk"))
    cont = p.meta["compression"]["container"]
    assert any(r["bits"] == 5 for r in cont) and any(k.endswith("u5") for k in p.arrays)
    from dmipy_sim.replay.bank import read_rpk
    back = read_rpk(str(tmp_path / "p.rpk"))
    np.testing.assert_array_equal(_cx.read_position_coeffs(back.arrays), _cx.read_position_coeffs(p.arrays))
    seq = _seqmod.pgse([[1, 0, 0], [0, 0, 1]], 2e-3, 6e-3, bvalues=[1e9, 1e9], TE=9.5e-3, n_t=4 * p.n_t + 1, slew_rate=np.inf)
    np.testing.assert_allclose(back.replay(seq), f.replay(seq), rtol=5e-3)
    assert p.meta["rpk_schema_version"] == "0.7"


def test_the_two_level_container_still_reads_as_before():
    rng = np.random.default_rng(3); X = np.cumsum(rng.normal(0, 1e-6, (50, 61, 3)), axis=1)
    a, m, _ = _cx.encode_bridge_dst(X, 24, 1e-4, container=_cx.TWO_LEVEL_CONTAINER)
    assert sorted(k for k in a if k.startswith("pos_x_b")) == ["pos_x_b0", "pos_x_b1"]
    sc = np.asarray(a["pos_band_scale"], np.float64)[0, 0]                        # (K,) the x scales
    by_hand = np.concatenate([a["pos_x_ends"], a["pos_x_b0"] * sc[:16], a["pos_x_b1"] * sc[16:]], axis=1)
    np.testing.assert_array_equal(_cx.read_position_coeffs(a)[:, :, 0], by_hand)


# ------------------------------------------------------------- C2 on a curve, and the C3 path channel's container

def test_the_contact_channel_takes_a_curve_and_replays(tmp_path):
    walk = d.simulate_trajectories(300, D0, d.Cylinder(2e-6, (0, 0, 1)), 0.01, 5e-4, seed=0, require_gpu=False)
    f = build_replay_pack(walk, id="t/f", K=16, license="x", citation="x", blt_temporal_K=16, blt_container="bands")
    c = _cx.band_curve(16, 12)
    p = build_replay_pack(walk, id="t/c", K=16, license="x", citation="x", blt_temporal_K=16, blt_container=c,
                          position_container=c, out_path=str(tmp_path / "c.rpk"))
    cont = p.meta["compression"]["channels"]["boundary_local_time"]["container"]
    assert cont == c and any(k.startswith("blt_b") and "u" in k for k in p.arrays)
    from dmipy_sim.replay.bank import read_rpk
    back = read_rpk(str(tmp_path / "c.rpk"))
    np.testing.assert_array_equal(_cx.bridge_bands(back.arrays, cont and p.meta["compression"]["channels"]["boundary_local_time"]),
                                  _cx.bridge_bands(p.arrays, p.meta["compression"]["channels"]["boundary_local_time"]))
    seq = _seqmod.pgse([[1, 0, 0]], 2e-3, 6e-3, bvalues=[1e9], TE=9.5e-3, n_t=4 * p.n_t + 1, slew_rate=np.inf)
    kw = dict(tissue=Tissue(rho2=1e-5, D=D0))
    np.testing.assert_allclose(back.replay(seq, **kw), f.replay(seq, **kw), rtol=5e-3)


@pytest.mark.parametrize("bits", [3, 5, 7, 11])
def test_a_path_container_round_trips_the_quantised_values(bits):
    from dmipy_sim.replay.bank import _quantise_susc_path, susc_path_coeffs
    rng = np.random.default_rng(bits); n_w, n_ch, K = 64, 12, 10
    C = rng.normal(size=(n_w, n_ch, K)) / np.arange(1, K + 1)
    meta = dict(channel="susc_path_dct", K=K, n_t=40, n_ch=n_ch, channels=[f"c{i}" for i in range(n_ch)], iso_P_zz="stored")
    a, m = _quantise_susc_path(C, dict(meta), [{"modes": [0, 4], "bits": 12}, {"modes": [4, K], "bits": bits}])
    assert m["container"] == [{"modes": [0, 4], "bits": 12}, {"modes": [4, K], "bits": bits}] and m["bits"] is None
    got, _ = susc_path_coeffs(a, m)
    scale = np.asarray(a["susc_path_scale"], np.float64)
    assert got.shape == C.shape and np.abs(got - C).max() <= 0.5 * scale.max() * (1 + 1e-6)
    np.testing.assert_allclose(got[:, :, :4], np.rint(C[:, :, :4] / scale[:, :4]) * scale[:, :4], rtol=1e-6, atol=1e-15)


def test_a_one_range_container_is_the_single_depth_form():
    from dmipy_sim.replay.bank import _quantise_susc_path, path_codec
    assert path_codec([{"modes": [0, 8], "bits": 8}]) == 8 and path_codec(((None, 16),)) == 16 and path_codec(8) == 8
    assert isinstance(path_codec([{"modes": [0, 4], "bits": 8}, {"modes": [4, 8], "bits": 6}]), tuple)
    C = np.random.default_rng(0).normal(size=(10, 12, 8))
    meta = dict(channel="susc_path_dct", K=8, n_t=40, n_ch=12, channels=[f"c{i}" for i in range(12)], iso_P_zz="stored")
    a, m = _quantise_susc_path(C, dict(meta), path_codec([{"modes": [0, 8], "bits": 8}]))
    assert list(a) == ["susc_path_dct", "susc_path_scale"] and a["susc_path_dct"].dtype == np.int8 and m["bits"] == 8 and "container" not in m


def test_a_pack_with_a_field_stores_the_path_channel_in_a_container_and_replays(tmp_path):
    from tests.test_field_sample_every import _walk
    from tests.replay_frames import field_along
    from dmipy_sim.io.strands import write_tck, concat_centerlines
    from dmipy_sim.spec import disco_spec
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp_path / "t.tck"), str(tmp_path / "d.txt")
    write_tck(tck, *concat_centerlines(cls_), coordinate_unit_m=25e-6); np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    spec = disco_spec(tck, dia, side_m=20e-6)
    w = _walk(spec, 1)
    kw = dict(license="x", citation="x", K=8, susc_path_K=8, device="numpy")
    ref = build_replay_pack(w, id="t/p8", susc_path_bits=16, **kw)
    cont = [{"modes": [0, 2], "bits": 14}, {"modes": [2, 8], "bits": 6}]
    pk = build_replay_pack(w, id="t/pc", susc_path_bits=cont, out_path=str(tmp_path / "pc.rpk"), **kw)
    pm = pk.meta["compression"]["channels"]["susceptibility_path"]
    assert pm["container"] == cont and "susc_path_dct" not in pk.arrays and "susc_path_dct_b1u6" in pk.arrays
    assert np.isfinite(pk.meta["fidelity"]["err_susc_path"])
    from dmipy_sim.replay.bank import read_rpk
    back = read_rpk(str(tmp_path / "pc.rpk"))
    seq = d.gre(4e-4, gradient_directions=[[1, 0, 0]], bvalues=[0.0], delta=1e-4, Delta=2e-4, n_t=pk.n_t, slew_rate=np.inf)
    seq_x, R = field_along(seq, (1, 0, 0))
    ten = Tissue(chi_iso=-0.1e-6, chi_aniso=-0.1e-6)
    E_ref = ref.walker_signals(seq_x, orientation=R, scanner=7.0, tissue=ten)[2]
    E_c = back.walker_signals(seq_x, orientation=R, scanner=7.0, tissue=ten)[2]
    assert np.abs(np.angle(E_c[:, 0] / E_ref[:, 0])).max() < 0.1


# ------------------------------------------------------------------ the derived default

def test_the_derived_container_is_one_common_step_and_gains_a_bit_per_quartered_floor():
    K, n_t, dt = 64, 201, 5e-4
    scales = 1e-4 * np.sqrt(dt) / np.arange(1, K + 1)                            # bands in m sqrt(s) (RPK.md 9.5.3)
    c = {f: _cx.derive_container(scales, K, f, n_t=n_t, dt=dt) for f in (2e-2, 5e-3, 1.25e-3)}
    bits = {f: np.concatenate([[r["bits"]] * (r["bands"][1] - r["bands"][0]) for r in v]) for f, v in c.items()}
    for f, v in c.items():
        assert v[0]["bands"][0] == 0 and v[-1]["bands"][1] == K
        assert all((r["bands"][1] - r["bands"][0]) * r["bits"] % 8 == 0 for r in v[:-1])
    d1, d2 = bits[2e-2].mean(), bits[5e-3].mean()
    assert 0.5 < d2 - d1 < 1.5                                                     # the bias term: a bit per quartering of the floor
    # a band below the step is not dropped, it keeps the minimum depth
    assert bits[2e-2].min() == 3 and bits[1.25e-3].max() <= 16
    # the step is common up to the byte grouping
    f = 5e-3; b = bits[f]; ok = (b > 3) & (b < 16); st = scales / 2.0 ** (b - 1)
    assert st[ok].max() / st[ok].min() < 5.0        # byte grouping moves a band by up to a few bits


def test_the_default_pack_is_derived_for_its_floor_and_a_block_inherits_the_certifying_ranges(tmp_path):
    walk = d.simulate_trajectories(400, D0, d.Cylinder(2e-6, (0, 0, 1)), 0.01, 5e-4, seed=0, require_gpu=False,
                                   raw_subset=400).saves()                     # a fill's block: a walk of saves
    p = build_replay_pack(walk, id="t/d", K=16, license="x", citation="x", blt_temporal_K=16, sigma_star=5e-3)
    q = build_replay_pack(walk, id="t/d2", K=16, license="x", citation="x", blt_temporal_K=16, sigma_star=1e-3)
    assert p.meta["compression"]["container"] != _cx.band_curve(16, 10)
    assert sum(r["bits"] * (r["bands"][1] - r["bands"][0]) for r in q.meta["compression"]["container"]) > \
        sum(r["bits"] * (r["bands"][1] - r["bands"][0]) for r in p.meta["compression"]["container"])      # a lower floor: more bits
    blk = build_replay_pack(walk, id="t/blk", K=16, license="x", citation="x", blt_temporal_K=16, fidelity="inherited", fidelity_from=p)
    assert blk.meta["compression"]["container"] == p.meta["compression"]["container"]
    # measured against the walk's saves, the error is what K = 16 of the saves' 20 bands drops plus the container's
    # quantisation: within the floor of 400 walkers (0.084 against 0.117 measured)
    assert p.meta["fidelity"]["within_2x_floor"] and p.meta["fidelity"]["err_max"] < p.meta["fidelity"]["floor_max"]
