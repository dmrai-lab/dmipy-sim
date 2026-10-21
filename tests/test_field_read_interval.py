"""The strand field along the walk is read every ``field_read_interval`` seconds: ``walk_spec(field_read_interval=)``
gives the interval the adaptive producer reads the field at (snapped to a whole number of the pool's rounds per
block, never finer than a round), or each pool's pilot derives it; the walk and the pack record it, and the replay
gates the stored modes on it.

The derived interval is the rule of :func:`dmipy_sim.engine.accumulate.field_read_rule` on the pilot's measured read
series (its quadratic variation per pool, RPK.md 9.5.5) -- the read that keeps ``v_read`` under the floor's square --
and the term the pack's certificate charges is ``(gamma B0 chi)^2 dt_f^2 / 12 x (sum_c sqrt(QV_c))^2`` of the walk's own read series
(``fidelity.terms.field_read``). The walk's reads steer nothing: a coarser read keeps the positions and the contact
identical and changes only the field's modes."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.accumulate import field_read_rule, field_read_term, field_read_B0_chi_max, field_read_steps
from dmipy_sim.io.strands import write_tck, concat_centerlines
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec import disco_spec, walk_spec
from dmipy_sim.spec.tissue import Tissue
from tests.replay_frames import field_along

BLOCK = 1e-4                       # the walk's block: two rounds of the extra pool, four of the intra pool
FINE = BLOCK / 4                   # a given read finer than the derived one: every round of each pool


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("readint")
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "t.tck"), str(tmp / "d.txt")
    write_tck(tck, *concat_centerlines(cls_), coordinate_unit_m=25e-6); np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    return disco_spec(tck, dia, side_m=20e-6)


def _walk(spec, interval=None, seed=3):
    return walk_spec(spec, 60, 8e-4, BLOCK, seed=seed, n_probe=20_000, require_gpu=False, adaptive_steps=True, K=4,
                     field_cutoff_max_m=25e-6, **({} if interval is None else dict(field_read_interval=interval)))


@pytest.fixture(scope="module")
def walks(spec):
    """The default (derived) read and a given finer one."""
    return _walk(spec), _walk(spec, FINE)


def test_the_field_is_read_at_its_interval_and_the_walk_is_unchanged(walks):
    """A finer given interval: the same walk (positions, contact), the field's modes differ, the walk records the
    read it made (``block["field_read"]``, ``field_read_interval``); the bundle states its coarsest pool's read, the
    given interval snapped to that pool's round (never finer than a round)."""
    wd, wc = walks
    np.testing.assert_array_equal(wc.position_coeffs, wd.position_coeffs)
    np.testing.assert_array_equal(wc.boundary_bands, wd.boundary_bands)
    np.testing.assert_array_equal(wc.boundary_endpoint, wd.boundary_endpoint)
    fd, fc = wd.block["field_read"], wc.block["field_read"]
    assert fd["rule"] == "pilot" and fc["rule"] == "given" and fc["dt_f_rule"] == pytest.approx(FINE)
    assert wc.field_read_interval == pytest.approx(BLOCK / 2) == fc["dt_f"]               # the extra pool's round
    assert fc["at_step"] and fc["n_read"] == 1                                            # the given read is below its round
    assert wc.field_read_interval < wd.field_read_interval                               # the derived read is coarser here
    assert wd.field_coeffs.shape == wc.field_coeffs.shape == (60, 1, wd.K_c3, 13)
    a, b = np.asarray(wd.field_coeffs, np.float64), np.asarray(wc.field_coeffs, np.float64)
    assert np.abs(a - b).max() > 1e-5 * np.abs(a).max()                           # a different quadrature of the same field
    assert np.abs(a - b).max() < 0.1 * np.abs(a).max()                            # of the same function


def test_the_default_interval_is_the_rule_on_the_pilots_read_series(spec, walks):
    """Each pool's pilot reads every round and measures its read series' quadratic variation; the pool's interval is
    :func:`field_read_rule` of it at the envelope's field strength and the spec's chi (``v_read`` under the floor's
    square), snapped to the pool's rounds (never finer than a round); the bundle records the coarsest pool's read and
    every pool's own, and the trapezoid's constant measured on the pilot at every coarser read the block allows."""
    wd, _ = walks
    fr = wd.block["field_read"]
    assert set(fr["pools"]) == {"intra", "extra"} and fr["pilot_walkers"] > 0
    for name, p in fr["pools"].items():
        qv = np.asarray(p["qv_increments"])
        assert qv.shape == (13,) and (qv >= 0).all() and qv.sum() > 0              # the walkers move through the field
        assert p["dt_f_rule"] == pytest.approx(field_read_rule(qv, fr["B0"], fr["chi"], fr["floor"], p["constants"]))
        assert np.all(np.asarray(fr["constants"]) >= np.asarray(p["constants"]))          # the bundle charges the largest
    assert len(fr["constants"]) == 13 and min(fr["constants"]) >= 1.0 / 12.0 - 1e-15
    assert fr["measured_constants"]                                               # measured at the coarser reads
    n_read, dt_f = field_read_steps(fr["dt_f_rule"], fr["step_s"], int(round(wd.block["h"] / fr["step_s"])))
    assert (n_read, wd.field_read_interval) == (fr["n_read"], pytest.approx(dt_f)) and dt_f == pytest.approx(fr["dt_f"])
    assert fr["at_step"] == (fr["dt_f_rule"] < fr["step_s"])


def test_the_pack_records_the_read_and_the_replay_gates_on_it(walks, tmp_path):
    """The pack's meta carries the read (the path channel's descriptor, the walk's parameters, the envelope), the
    certificate charges the rule's term, the replay's field phase equals the gated contraction of the stored modes,
    and a prefix keeps the read."""
    wd, wc = walks
    pks = {}
    for name, w in (("derived", wd), ("fine", wc)):
        pks[name] = _float_pack(w, id=f"t/read-{name}", license="x", citation="x", K=4, susc_path_K=4, device="numpy",
                                out_path=str(tmp_path / f"{name}.rpk"))
        pk = pks[name]; fr = w.block["field_read"]
        pm = pk.meta["compression"]["channels"]["susc_field_basis"]
        assert pm["K"] == 4 and pm["n_ch"] == 12 and "n_t" not in pm and "dt" not in pm
        assert pm["read_interval"] == pytest.approx(w.field_read_interval) and pm["read"]["dt_f"] == pytest.approx(fr["dt_f"])
        assert pk.meta["walk_params"]["field_read_interval"] == pytest.approx(w.field_read_interval)
        env = pk.meta["replay_envelope"]["field_read"]
        assert env["interval_s"] == pytest.approx(w.field_read_interval) and env["at_step"] == fr["at_step"]
        qv = pk.meta["walk_params"]["qv"]["field_increments"]                      # the walk's own read series
        np.testing.assert_allclose(qv, np.asarray(w.qv_field_increments, np.float64)[:, 0].mean(0), rtol=1e-6)
        at = env["term_at"]
        assert env["B0_chi_max_T"] == pytest.approx(field_read_B0_chi_max(w.field_read_interval, qv, at["floor"], fr["constants"]))
        term = field_read_term(w.field_read_interval, qv, at["B0"], at["chi"], fr["constants"])
        assert pk.fidelity["terms"]["field_read"] == pytest.approx(term, rel=1e-4) and term > 0
        assert pm["max_refocus_pulses"] == 2 and np.isfinite(pk.fidelity["terms"]["quadrature"]["field"])
    assert pks["derived"].fidelity["terms"]["field_read"] >= pks["fine"].fidelity["terms"]["field_read"]
    pk = pks["derived"]
    pm = pk.meta["compression"]["channels"]["susc_field_basis"]
    seq = d.gre(4e-4, gradient_directions=[[1, 0, 0]], bvalues=[0.0], delta=1e-4, Delta=2e-4, n_t=pk.n_t, slew_rate=np.inf)
    seq_x, R = field_along(seq, (1, 0, 0))
    w_, ew, E = pk.walker_signals(seq_x, orientation=R, scanner=7.0, tissue=Tissue(chi_iso=-0.1e-6, chi_aniso=-0.1e-6))
    w0, ew0, E0 = pk.walker_signals(seq)
    phase = np.angle(E[:, 0] / E0[:, 0])                             # the field's phase per walker
    from dmipy_sim.replay.bank import path_field_integral
    from dmipy_sim.fields.hollow_cylinder import contract
    Psi, _ = path_field_integral(pk.arrays, pm, seq, pk.T_window, n_w=pk.n_walkers)
    expect = contract(Psi, (1, 0, 0), B0=7.0, chi_iso=-0.1e-6, chi_aniso=-0.1e-6)
    np.testing.assert_allclose(np.angle(np.exp(1j * (phase - expect))), 0.0, atol=1e-6)
    child = pk.prefix(4e-4, K=4)
    assert child.meta["compression"]["channels"]["susc_field_basis"]["read"] == pm["read"]
    assert child.meta["replay_envelope"]["field_read"] == pk.meta["replay_envelope"]["field_read"]
    # half the window, half the reads: the read series' QV and the term at the same interval halve
    np.testing.assert_allclose(child.meta["walk_params"]["qv"]["field_increments"],
                               0.5 * np.asarray(pk.meta["walk_params"]["qv"]["field_increments"]), rtol=1e-6)
    assert child.fidelity["terms"]["field_read"] == pytest.approx(0.5 * pk.fidelity["terms"]["field_read"], rel=1e-4)


def test_the_read_interval_needs_the_adaptive_producer(spec):
    """A strand field is read by the adaptive producer only: a fixed-step walk of a strand substrate is refused, with
    or without an interval."""
    from dmipy_sim.spec.substrate import SpecError
    for kw in ({}, dict(field_read_interval=BLOCK)):
        with pytest.raises(SpecError, match="adaptive"):
            walk_spec(spec, 30, 8e-4, BLOCK, seed=3, n_probe=20_000, require_gpu=False, **kw)


def test_shards_with_the_path_channel_merge(spec, tmp_path):
    """Each pack quantises its path channel with its own scale table; a merge stacks the tables with a block axis
    and the decoder applies each walker's own, so the merged coefficients are the shards' and the field replay is
    the weight-averaged replay of the shards."""
    from dmipy_sim.replay.bank import merge_packs, susc_path_coeffs
    pks = [_float_pack(_walk(spec, BLOCK, seed=s_), id=f"t/shard-{s_}", license="x", citation="x", K=4, susc_path_K=4, device="numpy",
                       out_path=str(tmp_path / f"s{s_}.rpk"))
           for s_ in (3, 4)]
    m = merge_packs([str(tmp_path / "s3.rpk"), str(tmp_path / "s4.rpk")], id="t/merged")
    pm = m.meta["compression"]["channels"]["susc_field_basis"]
    assert m.arrays["susc_path_scale"].shape[0] == 2 and "band_block" in m.arrays
    C_m, names = susc_path_coeffs(m.arrays, pm)
    C_s = np.concatenate([susc_path_coeffs(pk.arrays, pk.meta["compression"]["channels"]["susc_field_basis"])[0] for pk in pks])
    np.testing.assert_allclose(C_m, C_s, rtol=1e-12, atol=0)
    seq = d.gre(4e-4, gradient_directions=[[1, 0, 0]], bvalues=[0.0], delta=1e-4, Delta=2e-4, n_t=m.n_t, slew_rate=np.inf)
    kw = dict(scanner=7.0, tissue=Tissue(chi_iso=-0.1e-6, chi_aniso=-0.1e-6))
    W = [float(np.asarray(pk.arrays["spin_weights"]).sum()) if "spin_weights" in pk.arrays else pk.n_walkers for pk in pks]
    S = [pk.replay(seq, **kw)[0] for pk in pks]
    np.testing.assert_allclose(m.replay(seq, **kw)[0], (W[0] * S[0] + W[1] * S[1]) / (W[0] + W[1]), rtol=1e-6)


def _float_pack(*a, **k):
    """``build_replay_pack`` in the float containers: this module compares packs and replays to the bit, which the
    derived integer default does not promise."""
    k.setdefault("position_container", None); k.setdefault("blt_container", None); k.setdefault("field_container", None)
    return build_replay_pack(*a, **k)
