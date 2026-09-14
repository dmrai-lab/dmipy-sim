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

D0 = 2e-9


def test_the_band_container_round_trips_to_nanometres_at_a_third_of_the_bytes():
    rng = np.random.default_rng(0); n_w, n_t = 400, 253
    X = np.cumsum(rng.normal(0, np.sqrt(2 * D0 * 4e-4), (n_w, n_t, 3)), axis=1) + 1e-3      # 100 ms at 1 mm
    a_f, m_f, nb_f = _cx.encode_bridge_dst(X, 128)
    a_b, m_b, nb_b = _cx.encode_bridge_dst(X, 128, container=_cx.BAND_CONTAINER)
    assert nb_f / nb_b > 3.0 and m_b["container"] == [{"bands": [0, 16], "bits": 16}, {"bands": [16, 128], "bits": 8}]
    assert a_b["pos_x_b0"].dtype == np.int16 and a_b["pos_x_b1"].dtype == np.int8 and a_b["pos_x_ends"].dtype == np.float32
    Xf, Xb = _cx.decode(a_f, m_f), _cx.decode(a_b, m_b)
    assert np.abs(Xb - Xf).max() < 1e-7 and np.sqrt(((Xb - Xf) ** 2).mean()) < 2e-8               # 100 nm max, 20 nm rms
    np.testing.assert_array_equal(_cx.read_position_coeffs(a_b)[:, :2], _cx.read_position_coeffs(a_f)[:, :2])   # the endpoints exact
    with pytest.raises(ValueError, match="covers bands"):
        _cx.encode_bridge_dst(X, 32, container=((16, 16),))
    with pytest.raises(ValueError, match="8 or 16"):
        _cx.encode_bridge_dst(X, 32, container=((None, 4),))


def test_the_c2_container_and_the_static_c1_column():
    rng = np.random.default_rng(1); n_w, n_t = 300, 121
    ell = -np.abs(rng.normal(0, 1e-6, (n_w, n_t))) * (rng.uniform(size=(n_w, n_t)) < 0.3); ell[:, 0] = 0.0
    a_f, m_f = _cx.encode_boundary_bridge(ell, K=64, dtype=np.float16)
    a_b, m_b = _cx.encode_boundary_bridge(ell, K=64, container=((None, 8),))
    assert "blt_bridge_dst" not in a_b and a_b["blt_b0"].dtype == np.int8 and _cx.has_c2(a_b) and _cx.c2_bands_K(a_b, {}) == 64
    chi = np.ones(n_t); chi[0] = 0.0; chi[80:] = 0.0
    lw_f = _cx.surface_logweight_bridge(a_f, m_f, 3.0, chi); lw_b = _cx.surface_logweight_bridge(a_b, m_b, 3.0, chi)
    assert np.abs(lw_b - lw_f).max() < 5e-3 * np.abs(lw_f).max()
    np.testing.assert_allclose(_cx.decode_boundary_bridge(a_b, m_b).sum(1), np.asarray(a_b["blt_endpoint"]), rtol=1e-5)
    lab = np.repeat(rng.integers(0, 3, (n_w, 1)), n_t, axis=1)
    a1, m1 = _cx.encode_occupancy({"comp": lab})
    assert list(a1) == ["comp_static"] and a1["comp_static"].dtype == np.int8 and m1["columns"][0]["kind"] == "static"
    np.testing.assert_array_equal(_cx.decode_occupancy(a1, m1)["comp"], lab)
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
    assert "pos_x" not in b.arrays and "pos_x_ends" in b.arrays and "blt_b0" in b.arrays and "comp_static" in b.arrays
    assert b.has_relaxation and b.has_surface and b.n_walkers == f.n_walkers and b.K == f.K
    assert b.meta["compression"]["container"] == [{"bands": [0, 4], "bits": 16}, {"bands": [4, 8], "bits": 8}]
    fb, ff = b.meta["fidelity"], f.meta["fidelity"]                                          # the container costs the certificate nothing
    assert abs(fb["err_max"] - ff["err_max"]) < 1e-3 * ff["err_max"] and fb["floor_max"] == ff["floor_max"]
    Xb, Xf = _cx.decode(b.arrays, b.meta["compression"]), _cx.decode(f.arrays, f.meta["compression"])
    assert np.abs(Xb - Xf).max() < 1e-7
    seq = _seqmod.pgse([[1, 0, 0], [0, 0, 1]], 2e-3, 6e-3, bvalues=[1e9, 1e9], TE=9.5e-3, n_t=4 * b.n_t + 1, slew_rate=np.inf)
    for kw in (dict(tissue=False), dict(tissue=False, T2=[0.08, 0.03], T1=[1.0, 1.2]), dict(tissue=False, rho=1e-5, D=D0)):
        np.testing.assert_allclose(b.replay(seq, **kw), f.replay(seq, **kw), rtol=2e-3)
    from dmipy_sim.replay import read_rpk
    back = read_rpk(str(tmp / "bands.rpk"))
    np.testing.assert_allclose(back.replay(seq, tissue=False), b.replay(seq, tissue=False), rtol=1e-12)
    assert back.arrays["pos_x_b1"].dtype == np.int8
    m = merge_packs([b, back], id="t/merged")                                              # two blocks, each with its scales
    assert m.n_walkers == 2 * b.n_walkers and m.arrays["pos_band_scale"].shape[0] == 2 and m.arrays["band_block"][-1] == 1
    np.testing.assert_allclose(_cx.decode(m.arrays, m.meta["compression"])[b.n_walkers:], Xb, rtol=0, atol=1e-15)


def test_a_prefix_of_a_banded_pack_replays(packs):
    """ReplayPack.prefix re-encodes from the dequantised coefficients: a banded pack cuts like a float one."""
    f, b, tmp = packs
    TE = 5e-3
    pb, pf = b.prefix(TE), f.prefix(TE)
    seq = _seqmod.pgse([[1, 0, 0]], 1e-3, 3e-3, bvalues=[1e9], TE=TE, n_t=4 * pb.n_t + 1, slew_rate=np.inf)
    np.testing.assert_allclose(pb.replay(seq, tissue=False), pf.replay(seq, tissue=False), rtol=2e-3)

