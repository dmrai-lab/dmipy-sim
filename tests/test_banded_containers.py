"""The integer band containers of the bridge channels and the static C1 column (registry 0.6).

A bridge's sine coefficients fall as 1/k, so a per-band scale with 16 bits on the first bands and 8 beyond keeps
the reconstruction at the nanometre level while the pack shrinks 3.4x at K = 128; the C2 bridge takes the same
container; a walker that never changes pool stores one label. Every reader dequantises in one place, so the
coefficient-space replay of #242 and the dense oracles read either container alike.
"""
from __future__ import annotations
import dmipy_replay.replay.compression as _r_compression
import dmipy_replay.replay.field as _r_field

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences as _seqmod
from dmipy_sim.replay import compression as _cx
from dmipy_sim.replay.bank import build_replay_pack, merge_packs
from dmipy_replay.spec.tissue import Tissue
from tests.conftest import dense_relaxation_logweight

D0 = 2e-9


def test_the_band_container_round_trips_to_nanometres_at_a_third_of_the_bytes():
    rng = np.random.default_rng(0); n_w, K = 400, 128
    C = np.zeros((n_w, K + 2, 3)); C[:, 0] = 1e-3; C[:, 1] = rng.normal(0, 1e-5, (n_w, 3))
    C[:, 2:] = rng.normal(0, 1, (n_w, K, 3)) * (1e-7 / np.arange(1, K + 1))[None, :, None]       # bands falling as 1/k
    a_f = _cx.pack_position_arrays(C, dtype=np.float32)
    a_b = _cx.pack_position_arrays(C, container=_cx.TWO_LEVEL_CONTAINER)
    nb_f, nb_b = (sum(np.asarray(v).nbytes for v in a.values()) for a in (a_f, a_b))
    assert nb_f / nb_b > 3.0 and a_b["pos_x_b0"].dtype == np.int16 and a_b["pos_x_b1"].dtype == np.int8 and a_b["pos_x_ends"].dtype == np.float32
    Cf, Cb = _r_compression.read_position_coeffs(a_f), _r_compression.read_position_coeffs(a_b)
    np.testing.assert_array_equal(Cb[:, :2], Cf[:, :2])                                           # the endpoints exact
    assert np.abs(Cb[:, 2:] - C[:, 2:]).max() < 1e-9                                               # bands to the quantisation step
    with pytest.raises(ValueError, match="covers bands"):
        _cx.pack_position_arrays(C[:, :34], container=((16, 16),))
    with pytest.raises(ValueError, match="bits must be 3..16"):
        _cx.pack_position_arrays(C[:, :34], container=((None, 2),))


def test_the_c2_container_and_the_static_c1_column():
    rng = np.random.default_rng(1); n_w, K, T = 300, 64, 0.012
    end = -np.abs(rng.normal(0, 1e-5, n_w)); bands = rng.normal(0, 1, (n_w, K)) * (1e-7 / np.arange(1, K + 1))[None, :]
    a_f, m_f = _cx.encode_boundary_bands(np.zeros(n_w, np.float32), end, bands, T=T, dtype=np.float32)
    a_b, m_b = _cx.encode_boundary_bands(np.zeros(n_w, np.float32), end, bands, T=T, container=((None, 8),))
    assert "blt_bridge_dst" not in a_b and a_b["blt_b0"].dtype == np.int8 and _r_compression.has_c2(a_b) and _r_compression.c2_bands_K(a_b, {}) == 64
    n_t = 121; chi = np.ones(n_t); chi[0] = 0.0; chi[80:] = 0.0
    lw_f = _r_compression.surface_logweight_bridge(a_f, m_f, 3.0, chi, 1e-4); lw_b = _r_compression.surface_logweight_bridge(a_b, m_b, 3.0, chi, 1e-4)
    assert np.abs(lw_b - lw_f).max() < 5e-3 * np.abs(lw_f).max()
    B = _r_compression.local_time_at(a_b, m_b, [0.0, T])                                                 # the bridge pinned at both ends
    np.testing.assert_allclose(B[:, 1] - B[:, 0], np.asarray(a_b["blt_endpoint"]), rtol=1e-5, atol=1e-10)
    lab = rng.integers(0, 3, n_w)
    a1, m1 = _cx.encode_static_occupancy(lab)
    assert list(a1) == ["comp_static"] and a1["comp_static"].dtype == np.int8 and m1["columns"][0]["kind"] == "static"
    np.testing.assert_array_equal(_r_compression.occupancy_events(a1, m1["columns"][0])[0], lab)             # a static column: one label per walker
    act = np.ones(n_t); act[-1] = 0.0                                                         # the readout, held over nothing
    gate = _r_compression.CumulativeGate(chi, 1e-4); on = _r_compression.CumulativeGate(act, 1e-4)
    got = _r_compression.relaxation_logweight_runs(a1, m1["columns"][0], [0.08, 0.03, 0.01], [1.0, 1.2, 0.3], gate, on, t1=(n_t - 1) * 1e-4)
    dense = dense_relaxation_logweight(lab, [0.08, 0.03, 0.01], [1.0, 1.2, 0.3], 1e-4, chi, act)
    np.testing.assert_allclose(got, dense, rtol=1e-12)
    from dmipy_replay.replay.events import EventList
    ev = EventList([0, 1] + [1] * (n_w - 1), [60 * 1e-4], [2])                                    # walker 0 crosses into pool 2
    a2, m2 = _cx.encode_events_occupancy({"comp": (lab, ev)})
    assert m2["columns"][0]["kind"] == "events" and "comp_events_t" in a2                           # a crossing keeps the events


@pytest.fixture(scope="module")
def packs(tmp_path_factory):
    walk = d.simulate_trajectories(300, D0, d.Cylinder(2e-6, (0, 0, 1)), 0.01, 5e-4, seed=0, require_gpu=False)
    tmp = tmp_path_factory.mktemp("bands")
    f = build_replay_pack(walk, id="t/float", K=8, license="x", citation="x", blt_temporal_K=8)
    b = build_replay_pack(walk, id="t/bands", K=8, license="x", citation="x", blt_temporal_K=8,
                          position_container=((4, 16), (None, 8)), blt_container="derived", out_path=str(tmp / "bands.rpk"))
    return f, b, tmp


def test_a_pack_in_the_integer_containers_replays_every_knob_like_the_float_one(packs):
    f, b, tmp = packs
    assert "pos_x" not in b.arrays and "pos_x_ends" in b.arrays and _r_compression.has_c2(b.arrays) and "comp_static" in b.arrays
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
    np.testing.assert_array_equal(_r_compression._load_range(t, bits, m), q)
    np.testing.assert_array_equal(_r_compression._load_range(_cx._store_range(q[..., 0], bits), bits, m), q[..., 0])
    # a row is its values side by side, MSB first, offset-binary
    row = np.unpackbits(_cx._store_range(q[:1, :, 0], bits)[0])[: m * bits].reshape(m, bits)
    np.testing.assert_array_equal(row @ (1 << np.arange(bits - 1, -1, -1)), q[0, :, 0] + 2 ** (bits - 1))


@pytest.mark.parametrize("bits", [3, 5, 7, 11])
def test_quantise_bands_reconstructs_the_integers_times_the_scale_at_every_depth(bits):
    rng = np.random.default_rng(1); B = rng.normal(size=(100, 20, 3)) / np.arange(1, 21)[None, :, None]
    cont = ((8, bits), (None, 8))
    arrays, meta = _cx.quantise_bands(B, cont, "pos", "pos_band_scale")
    got = _r_compression.dequantise_bands(arrays, meta["container"], "pos", "pos_band_scale")
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
    f = build_replay_pack(walk, id="t/f", K=16, license="x", citation="x", blt_temporal_K=16, blt_container="derived",
                          position_container=((4, 16), (None, 8)))
    p = build_replay_pack(walk, id="t/p", K=16, license="x", citation="x", blt_temporal_K=16, blt_container="derived",
                          position_container=((8, 12), (None, 5)), out_path=str(tmp_path / "p.rpk"))
    cont = p.meta["compression"]["container"]
    assert any(r["bits"] == 5 for r in cont) and any(k.endswith("u5") for k in p.arrays)
    from dmipy_sim.replay.bank import read_rpk
    back = read_rpk(str(tmp_path / "p.rpk"))
    np.testing.assert_array_equal(_r_compression.read_position_coeffs(back.arrays), _r_compression.read_position_coeffs(p.arrays))
    seq = _seqmod.pgse([[1, 0, 0], [0, 0, 1]], 2e-3, 6e-3, bvalues=[1e9, 1e9], TE=9.5e-3, n_t=4 * p.n_t + 1, slew_rate=np.inf)
    np.testing.assert_allclose(back.replay(seq), f.replay(seq), rtol=5e-3)
    assert p.meta["rpk_schema_version"] == "0.7"


def test_the_two_level_container_still_reads_as_before():
    rng = np.random.default_rng(3); C = rng.normal(0, 1e-6, (50, 26, 3))
    a = _cx.pack_position_arrays(C, container=_cx.TWO_LEVEL_CONTAINER)
    assert sorted(k for k in a if k.startswith("pos_x_b")) == ["pos_x_b0", "pos_x_b1"]
    sc = np.asarray(a["pos_band_scale"], np.float64)[0, 0]                        # (K,) the x scales
    by_hand = np.concatenate([a["pos_x_ends"], a["pos_x_b0"] * sc[:16], a["pos_x_b1"] * sc[16:]], axis=1)
    np.testing.assert_array_equal(_r_compression.read_position_coeffs(a)[:, :, 0], by_hand)


# ------------------------------------------------------------- C2 on a curve, and the C3 path channel's container

def test_the_contact_channel_takes_a_curve_and_replays(tmp_path):
    walk = d.simulate_trajectories(300, D0, d.Cylinder(2e-6, (0, 0, 1)), 0.01, 5e-4, seed=0, require_gpu=False)
    f = build_replay_pack(walk, id="t/f", K=16, license="x", citation="x", blt_temporal_K=16, blt_container="derived")
    c = _cx.band_curve(16, 12)
    p = build_replay_pack(walk, id="t/c", K=16, license="x", citation="x", blt_temporal_K=16, blt_container=c,
                          position_container=c, out_path=str(tmp_path / "c.rpk"))
    cont = p.meta["compression"]["channels"]["boundary_local_time"]["container"]
    assert cont == c and any(k.startswith("blt_b") and "u" in k for k in p.arrays)
    from dmipy_sim.replay.bank import read_rpk
    back = read_rpk(str(tmp_path / "c.rpk"))
    np.testing.assert_array_equal(_r_compression.bridge_bands(back.arrays, cont and p.meta["compression"]["channels"]["boundary_local_time"]),
                                  _r_compression.bridge_bands(p.arrays, p.meta["compression"]["channels"]["boundary_local_time"]))
    seq = _seqmod.pgse([[1, 0, 0]], 2e-3, 6e-3, bvalues=[1e9], TE=9.5e-3, n_t=4 * p.n_t + 1, slew_rate=np.inf)
    kw = dict(tissue=Tissue(rho2=1e-5, D=D0))
    np.testing.assert_allclose(back.replay(seq, **kw), f.replay(seq, **kw), rtol=5e-3)


@pytest.mark.parametrize("bits", [3, 5, 7, 11])
def test_a_path_container_round_trips_the_quantised_values(bits):
    from dmipy_sim.replay.bank import _quantise_susc_path
    from dmipy_replay.replay.field import susc_path_coeffs
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
    meta = dict(channel="susc_field_basis", K=8, T=4e-3, n_ch=12, channels=[f"c{i}" for i in range(12)], iso_P_zz="stored")
    a, m = _quantise_susc_path(C, dict(meta), path_codec([{"modes": [0, 8], "bits": 8}]))
    assert list(a) == ["susc_path_dct", "susc_path_scale"] and a["susc_path_dct"].dtype == np.int8 and m["bits"] == 8 and "container" not in m


def test_a_pack_with_a_field_stores_the_path_channel_in_a_container_and_replays(tmp_path):
    from tests.test_bank import _susc_walk
    from tests.replay_frames import field_along
    w = _susc_walk(200, seed=3, K_c3=8)
    kw = dict(license="x", citation="x", K=8, susc_path_K=8, device="numpy")
    ref = build_replay_pack(w, id="t/p8", field_container=((None, 16),), **kw)
    cont = [{"modes": [0, 2], "bits": 14}, {"modes": [2, 8], "bits": 6}]
    pk = build_replay_pack(w, id="t/pc", field_container=cont, out_path=str(tmp_path / "pc.rpk"), **kw)
    pm = pk.meta["compression"]["channels"]["susc_field_basis"]
    assert pm["container"] == cont and "susc_path_dct" not in pk.arrays and "susc_path_dct_b1u6" in pk.arrays
    assert np.isfinite(pk.meta["fidelity"]["terms"]["field_read"])
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


def test_the_default_pack_is_derived_for_its_floor(tmp_path):
    walk = d.simulate_trajectories(400, D0, d.Cylinder(2e-6, (0, 0, 1)), 0.01, 5e-4, seed=0, require_gpu=False, raw_subset=400)
    p = build_replay_pack(walk, id="t/d", K=16, license="x", citation="x", blt_temporal_K=16, sigma_star=5e-3, fidelity="measured")
    q = build_replay_pack(walk, id="t/d2", K=16, license="x", citation="x", blt_temporal_K=16, sigma_star=1e-3)
    assert p.meta["compression"]["container"] != _cx.band_curve(16, 10)
    assert sum(r["bits"] * (r["bands"][1] - r["bands"][0]) for r in q.meta["compression"]["container"]) > \
        sum(r["bits"] * (r["bands"][1] - r["bands"][0]) for r in p.meta["compression"]["container"])      # a lower floor: more bits
    # measured against the walk's block ends, the error is what K = 16 of the walk's bands drops plus the container's
    # quantisation: within the floor of 400 walkers
    assert p.meta["fidelity"]["within_2x_floor"] and p.meta["fidelity"]["err_max"] < p.meta["fidelity"]["floor_max"]


# ------------------------------------------------------------------ the field channel's derived container

def _spec_unpack(row, bits, m):
    """The registry's rule, bit by bit: a row's values side by side, MSB first, offset-binary (``q + 2**(bits-1)``),
    the last byte zero-padded."""
    s = "".join(f"{int(b):08b}" for b in row)
    return [int(s[i * bits:(i + 1) * bits], 2) - (1 << (bits - 1)) for i in range(m)]


@pytest.mark.parametrize("bits", list(range(3, 17)))
def test_a_field_range_stores_the_registrys_bit_layout_at_every_depth(bits):
    from dmipy_sim.replay.bank import _quantise_susc_path
    from dmipy_replay.replay.field import susc_path_coeffs
    rng = np.random.default_rng(bits); n_w, n_ch, K, K0 = 12, 4, 12, 8
    C = rng.normal(size=(n_w, n_ch, K))
    meta = dict(channel="susc_field_basis", K=K, T=4e-3, n_ch=n_ch, channels=[f"c{i}" for i in range(n_ch)], iso_P_zz="stored")
    a, m = _quantise_susc_path(C, dict(meta), [{"modes": [0, K0], "bits": bits}, {"modes": [K0, K], "bits": 3}])
    name = _r_compression.range_name("susc_path_dct", 0, bits)
    t = np.asarray(a[name])
    scale = np.asarray(a["susc_path_scale"], np.float64)
    q_want = np.rint(C / scale[None]).astype(int)
    if bits in (8, 16):
        assert t.dtype == (np.int8 if bits == 8 else np.int16) and t.shape == (n_w, n_ch, K0)
        np.testing.assert_array_equal(t, q_want[:, :, :K0])
    else:
        assert t.dtype == np.uint8 and t.shape == (n_w, n_ch, -(-K0 * bits // 8)) and name.endswith(f"u{bits}")
        for w in range(n_w):
            for c in range(n_ch):
                assert _spec_unpack(t[w, c], bits, K0) == list(q_want[w, c, :K0])
    got, _ = susc_path_coeffs(a, m)
    np.testing.assert_allclose(got, q_want * scale[None])


def test_the_field_container_is_one_common_step_in_the_contracted_field():
    """Bits follow the contracted amplitude of the modes (one step across them), gain a bit per quartered floor, stay in
    3..16, and the realised phase noise of the stored steps is within the budget the floor allows."""
    from dmipy_replay.constants import GAMMA
    from dmipy_sim.replay import bank
    from dmipy_replay.replay.field import CHANNEL_NAMES
    rng = np.random.default_rng(1); n_w, K, T = 4000, 32, 0.1
    names = list(CHANNEL_NAMES)
    amp = 1e-2 / (1.0 + 0.1 * np.arange(K))                                   # a red spectrum, in field-basis units
    C = rng.normal(size=(n_w, K, 12)) * amp[None, :, None]
    kw = dict(B0=7.0, chi_iso=1e-7, chi_aniso=1e-8)
    depth = {}
    for f in (2e-2, 5e-3, 1.25e-3):
        fc = bank.DerivedFieldContainer(floor=f, **kw)
        a, m = bank.susc_path_encode_modes(C, names, T=T, container=fc)
        cont = m["container"]
        assert cont[0]["modes"][0] == 0 and cont[-1]["modes"][1] == K and all(3 <= r["bits"] <= 16 for r in cont)
        assert all((r["modes"][1] - r["modes"][0]) * r["bits"] % 8 == 0 for r in cont[:-1])
        depth[f] = np.concatenate([[r["bits"]] * (r["modes"][1] - r["modes"][0]) for r in cont])
        assert depth[f][0] >= depth[f][-1]                                       # the redder modes hold more bits
        # the realised noise of the stored steps against the unit-modulus bound the derivation holds to the budget
        eps = bank.field_quantisation(a, m, _cx.default_envelope(), T=T, h=T / 200, **kw)
        assert 0.0 < eps <= _cx.container_eps(f) * 1.0001
    assert 0.5 < depth[5e-3].mean() - depth[2e-2].mean() < 1.5 and 0.5 < depth[1.25e-3].mean() - depth[5e-3].mean() < 1.5
    # the measured phase error of the stored modes under the gradient echo (all the weight on mode 0) and a spin echo
    fc = bank.DerivedFieldContainer(floor=5e-3, **kw)
    a, m = bank.susc_path_encode_modes(C, names, T=T, container=fc)
    Cq, _ = _r_field.susc_path_coeffs(a, m)
    ref = np.transpose(C, (0, 2, 1))
    h = T / 200; n_t = 201
    for gate in (np.ones(n_t), bank._cpmg_gate(n_t, 1)):
        g = _r_compression.cosine_projection(gate.reshape(1, -1, 1), h, T, K)[0, :, 0]
        for dvec in ([0, 0, 1], [1, 0, 0], [1, 1, 1]):
            phi = lambda X: GAMMA * _r_field.susc_path_field(X, np.asarray(dvec, float), B0=7.0, chi_iso=1e-7, chi_aniso=1e-8, has_aniso=True) @ g
            assert np.std(phi(Cq) - phi(ref)) <= bank.field_quantisation(a, m, _cx.default_envelope(), T=T, h=h, **kw) * 1.05


def test_the_default_field_container_is_derived_and_the_quantisation_term_carries_it(tmp_path):
    from tests.test_bank import _susc_walk, _env7
    from dmipy_sim.replay.bank import read_rpk
    env = _env7(theta_deg=[0, 90])
    w = _susc_walk(200, seed=3, K_c3=8, envelope=env)
    kw = dict(license="x", citation="x", K=8, susc_path_K=8, device="numpy", envelope=env, fidelity="measured")
    pk = build_replay_pack(w, id="t/pd", out_path=str(tmp_path / "pd.rpk"), **kw)
    fl = build_replay_pack(w, id="t/pf", field_container=None, **kw)
    pm = pk.meta["compression"]["channels"]["susc_field_basis"]
    assert pm["container"] and "susc_path_dct" not in pk.arrays and pm["bits"] is None
    f, f0 = pk.meta["fidelity"], fl.meta["fidelity"]
    assert f0["err_quantisation_field"] == 0.0                                   # the float container adds none
    assert f["err_quantisation"] >= f["err_quantisation_field"] > 0.0 and f["err_quantisation"] > f0["err_quantisation"]
    # the Bloch route certifies the stored container within its floor
    assert f["err_susc_bloch"] <= f["floor_susc_bloch"]
    back = read_rpk(str(tmp_path / "pd.rpk"))
    np.testing.assert_array_equal(back.arrays["susc_path_scale"], pk.arrays["susc_path_scale"])


@pytest.fixture(scope="module")
def straight_cylinder_walk():
    """A myelinated cylinder along z walked with its closed-form field: the z-components ``iso_P_xz``, ``iso_P_yz``,
    ``aniso_G_xz``, ``aniso_G_yz`` are zero in every walker."""
    from dmipy_sim.fields.hollow_cylinder import HollowCylinderField
    from dmipy_sim.replay import compression as cx
    g = d.MyelinatedCylinder(2e-6, 3e-6, (0, 0, 1), D0, D0)
    env = dict(cx.default_envelope(), B0_list=[7.0], theta_deg=[90.0], max_refocus_pulses=4)
    w = d.simulate_trajectories(300, D0, g, 0.01, seed=1, require_gpu=False, K=16, window_T=0.01, field_basis=HollowCylinderField.of(g),
                                envelope=env)
    return w, env


def test_the_zero_components_are_not_stored_and_the_trace_component_is_never_a_channel(straight_cylinder_walk):
    from dmipy_sim.replay import bank
    w, env = straight_cylinder_walk
    C = np.asarray(w.field_coeffs, np.float64)[:, 0]
    names = list(w.field_channel_names)
    assert len(names) == 12 and "iso_P_zz" not in names
    zero = ["iso_P_xz", "iso_P_yz", "aniso_G_xz", "aniso_G_yz"]
    assert all(np.abs(C[..., names.index(n)]).max() == 0.0 for n in zero)
    kw = dict(license="x", citation="x", envelope=env, device="numpy")
    full = build_replay_pack(w, id="t/f", field_container=None, **kw)
    pk = build_replay_pack(w, id="t/z", **kw)
    pm = pk.meta["compression"]["channels"]["susc_field_basis"]
    assert "iso_P_zz" not in pm and "iso_P_zz" not in pm["channels"]
    assert set(pm["zero_components"]) == set(zero) and pm["n_ch"] == 12 - len(zero) == len(pm["channels"])
    assert not set(zero) & set(pm["channels"]) and pk.arrays["susc_path_scale"].shape == (pm["n_ch"], pm["K"])
    # the reader returns the full canonical twelve, the omitted components zero
    Cq, nq = _r_field.susc_path_coeffs(pk.arrays, pm)
    assert nq == names and Cq.shape == (300, 12, pm["K"])
    assert all(np.all(Cq[:, nq.index(n)] == 0.0) for n in zero)
    assert np.abs(Cq - np.transpose(C, (0, 2, 1))).max() < 0.05 * np.abs(C).max()      # the container's own error
    # the replay of the field is the float pack's, to the container's quantisation
    from dmipy_sim import RFEvent, ScannerSequence
    nt = pk.n_t; dt = pk.dt; TE = (nt - 1) * dt
    ten = Tissue(chi_iso=-1e-7, chi_aniso=-1e-7)
    for rf in ([RFEvent(0.0, 90)], [RFEvent(0.0, 90), RFEvent(TE / 2, 180)]):
        seq = ScannerSequence(G=np.zeros((1, nt, 3)), dt=dt, rf=rf)
        E = pk.walker_signals(seq, scanner=7.0, tissue=ten)[2][:, 0]; E0 = full.walker_signals(seq, scanner=7.0, tissue=ten)[2][:, 0]
        assert np.abs(np.angle(E / E0)).max() < 0.1
    nb = lambda p: sum(v.nbytes for k, v in p.arrays.items() if k.startswith("susc_path_dct")) / p.n_walkers
    assert nb(pk) < nb(full) * 0.5


def test_a_shard_takes_the_certifying_packs_layout_or_is_refused(straight_cylinder_walk):
    from dmipy_sim.replay import bank
    w, env = straight_cylinder_walk
    kw = dict(license="x", citation="x", envelope=env, device="numpy")
    pk = build_replay_pack(w, id="t/z", **kw)
    pm = pk.meta["compression"]["channels"]["susc_field_basis"]
    again = build_replay_pack(w, id="t/z2", field_container=pm, **kw)
    pm2 = again.meta["compression"]["channels"]["susc_field_basis"]
    assert pm2["channels"] == pm["channels"] and pm2["container"] == pm["container"] and sorted(again.arrays) == sorted(pk.arrays)
    C = np.asarray(w.field_coeffs, np.float64)[:, 0]
    names = list(w.field_channel_names)
    # a component the layout drops, nonzero in these modes: refused by name
    C2 = C.copy(); C2[..., names.index("iso_P_xz")] = 1e-2 * np.abs(C).max()
    with pytest.raises(ValueError, match="iso_P_xz"):
        bank.susc_path_encode_modes(C2, names, T=w.T, container=bank.FieldCodec.of(pm))
    # a field that offers iso_P_zz is refused by name, whatever the layout
    with pytest.raises(ValueError, match="iso_P_zz"):
        bank.susc_path_encode_modes(np.concatenate([C, C[..., :1]], axis=-1), names + ["iso_P_zz"], T=w.T, container=bank.FieldCodec.of(pm))


def test_the_contact_container_is_held_to_the_contacts_own_floor():
    """The contact's container is derived against the tighter of the pack's floor and the surface reading's own
    split-half floor (``floor_surface``): more bits than the pack's floor alone asks for, and a quantisation, read
    back walk-free, well inside that floor."""
    from dmipy_sim.replay import bank
    walk = d.simulate_trajectories(2000, D0, d.Cylinder(2e-6, (0, 0, 1)), 0.01, 5e-4, seed=0, require_gpu=False)
    p = build_replay_pack(walk, id="t/c", K=16, license="x", citation="x", blt_temporal_K=16, sigma_star=5e-2)
    f = p.meta["fidelity"]
    assert f["floor_surface"] > 0 and f["err_surface_quantisation"] <= 0.5 * f["floor_surface"] and f["err_surface"] <= f["floor_surface"]
    c2 = np.asarray(walk.boundary_endpoint)[:, 0]; bands = np.asarray(walk.boundary_bands)[:, 0, :16]
    env = _cx.default_envelope()
    assert bank._contact_floor(c2, bands, env, T=walk.T, h=walk.h, D=D0) == pytest.approx(f["floor_surface"], rel=0.2)
    # the container follows that floor: a tighter one asks for more bits than the pack's floor alone
    bits = lambda q: sum(r["bits"] * (r["bands"][1] - r["bands"][0]) for r in q.meta["compression"]["channels"]["boundary_local_time"]["container"])
    real = bank._contact_floor
    bank._contact_floor = lambda *a, **k: 1e-6
    try:
        tight = build_replay_pack(walk, id="t/c2", K=16, license="x", citation="x", blt_temporal_K=16, sigma_star=5e-2)
    finally:
        bank._contact_floor = real
    assert bits(tight) > bits(p)
