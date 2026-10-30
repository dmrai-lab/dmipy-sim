"""Plan before read (dmipy-sim#813): a replay decodes only the position bands and field modes its acquisition needs, states
the truncation error that costs, and gives the same signal bit for bit as a full decode cut at the same band.

The planner is :mod:`dmipy_sim.replay.planning`, the rule the columnar layout's planner reads too. Tests: the plan of a pack
against the columnar planner on a shared case; the prefix replay against the full decode cut at the plan; the error bound on
several waveforms (the real canonical-pores pack, when it is on this machine, adds OGSE 400 Hz); the decoded cache's prefix
entries; the field modes; and ``hf://`` range reads against a local HTTP server that honours ``Range``."""
import dmipy_replay.replay.compression as _r_compression
import contextlib
import http.server
import os
import threading

import numpy as np
import numpy.testing as npt
import pytest

import dmipy_sim as d
from dmipy_sim import build_replay_pack
from dmipy_replay.constants import GAMMA
from dmipy_sim.replay import compression as cx
from dmipy_sim.replay.columnar import ColumnarPack
from dmipy_replay.replay.planning import PLAN_TOL, Planned, ReadPlan, band_tail, fewest, floor_of
from dmipy_sim.replay.replay import ReplayPack
from dmipy_replay.replay.replay import _signal_factor
from dmipy_replay.sequences.builders import gre, ogse, pgse
from dmipy_replay.spec.tissue import Tissue

T_MAX, DT, N_W, K_BANDS = 0.02, 5e-4, 4000, 40
N_T = int(round(T_MAX / DT)) + 1
PORES = os.path.expanduser("~/dmrai-ws/release-work/pores-v2/packs/cylinder/d05.00um.rpk")


@pytest.fixture(scope="module")
def pack():
    """A cylinder of 3 um radius walked for 20 ms: 4000 walkers, 40 quantised bands, the contact bands."""
    w = d.simulate_trajectories(N_W, 2e-9, d.Cylinder(3e-6, (0.0, 0.0, 1.0)), T_MAX, DT, seed=1, require_gpu=False,
                                K=K_BANDS, window_T=T_MAX)
    return build_replay_pack(w, id="test/plan-before-read", license="x", citation="x")


def cache_kinds(pk):
    """The kinds of decode a pack's cache holds (the first element of each entry's key)."""
    c = pk.arrays.decoded
    return [k[0] for k in list(c._entries) + list(c._prefixes)]


def _fresh(pk):
    """A pack of the same tensors with an empty decoded cache of its own."""
    return ReplayPack(pk.arrays, pk.meta)


def _battery():
    """Waveforms that need different bands: a PGSE and OGSE of rising frequency, on the pack grid."""
    axis = [[1.0, 0.0, 0.0]]
    rows = {"pgse": pgse(axis, 0.004, 0.012, bvalues=[1e9], TE=T_MAX, n_t=N_T, slew_rate=200.0)}
    for f in (100.0, 200.0, 500.0):
        g = min(0.3, 200.0 / (2 * np.pi * f))
        rows[f"ogse{f:.0f}"] = ogse(axis, f, 0.5 * T_MAX, shape="cosine", gradient_strengths=[g], TE=T_MAX, n_t=N_T, slew_rate=200.0)
    return rows


def _cut_phase(pk, seq, K):
    """The phase of a full decode cut at ``K`` bands, contracted by hand: the reference a prefix replay must equal bit for bit."""
    P = pk._prepare(seq, tissue=None, scanner=None, orientation=None, compartment=None)
    (seg, t0, T_s), = P["windows"]
    C = _r_compression.read_position_coeffs(seg.arrays, dtype=np.float64, cached=False)[:, :K + 2]
    Wp = _r_compression.band_projection(P["G_eff_wf"], P["dt_wf"], T_s, K, t0=t0)
    return C.reshape(P["n_w"], -1) @ (GAMMA * Wp).reshape(Wp.shape[0], -1).T


# ------------------------------------------------------------------ the plan
def test_a_plan_states_its_counts_error_source_and_bytes(pack):
    seq = _battery()["ogse200"]
    plan = pack.plan(seq)
    assert isinstance(plan, ReadPlan)
    assert 0 < plan.K < pack.K and plan.K_stored == pack.K and plan.bands_chosen == "planned"
    assert plan.band_error <= plan.limit and plan.tol == PLAN_TOL and plan.floor == floor_of(pack.meta, pack.n_walkers)
    assert plan.band_variance.startswith("free bridge")           # no measured record in the header: the walk-free closed form
    assert plan.modes == 0 and plan.modes_stored == 0 and plan.mode_error == 0.0   # no field channel
    assert 0 < plan.bytes_read < plan.bytes_stored
    assert plan.error == plan.band_error + plan.mode_error
    assert str(plan).startswith(f"K = {plan.K}/{pack.K} bands")
    # it is the fewest: one band fewer exceeds the limit
    var = np.repeat(_r_compression.band_variances(pack.K, pack.T_window, pack.diffusivity)[:, None], 3, axis=1)
    tail = band_tail(np.asarray(seq.G_eff), seq.dt, [(pack.T_window, 0.0)], var).max(axis=0)
    assert tail[plan.K] <= plan.limit < tail[plan.K - 1]


def test_a_tighter_tolerance_plans_more_bands(pack):
    for name, seq in _battery().items():
        Ks = [pack.plan(seq, tol=tol).K for tol in (1.0, 0.25, 0.05, 0.01, 0.001)]
        assert Ks == sorted(Ks), name
    assert pack.plan(_battery()["ogse100"], tol=1e-6).K > pack.plan(_battery()["ogse100"]).K


def test_the_plan_is_the_columnar_planners_on_a_shared_case(pack):
    """One header record, two layouts: the pack's plan and the columnar layout's ``bands_for`` are one rule."""
    var = np.repeat(_r_compression.band_variances(pack.K, pack.T_window, pack.diffusivity)[:, None], 3, axis=1) * (1.0 + np.arange(3))[None]
    meta = dict(pack.meta, columnar=dict(band_variance=var.T.tolist()))
    stamped = ReplayPack(pack.arrays, meta)
    col = ColumnarPack.__new__(ColumnarPack)
    col.meta, col.K, col.floor = meta, pack.K, floor_of(pack.meta, pack.n_walkers)
    for seq in _battery().values():
        plan = stamped.plan(seq)
        assert plan.band_variance.startswith("stored: header columnar.band_variance")
        col.band_groups = list(range(pack.K + 1))                         # every count a candidate
        K_c, err_c = col.bands_for(seq, tol=PLAN_TOL)
        assert (plan.K, plan.band_error) == (K_c, err_c)
        col.band_groups = [0, 8, 16, 24, 32, 40]                            # groups: the first group at or beyond the fewest
        K_g, err_g = col.bands_for(seq, tol=PLAN_TOL)
        assert K_g == next(g for g in col.band_groups if g >= plan.K) and err_g <= plan.limit
    # the stored record is the plan's source and changes it: a larger variance asks for more bands
    seq = _battery()["ogse200"]
    assert stamped.plan(seq).K >= pack.plan(seq).K


def test_a_pack_with_no_variance_and_no_diffusivity_is_refused_by_name(pack):
    wp = {k: v for k, v in pack.meta["walk_params"].items() if k != "diffusivity"}
    bare = ReplayPack(pack.arrays, dict(pack.meta, walk_params=wp))
    with pytest.raises(ValueError, match="stores no per-band variance and records no diffusivity"):
        bare.plan(_battery()["pgse"])
    assert bare.plan(_battery()["pgse"], K=pack.K).K == pack.K            # a full read states no error to compute
    with pytest.raises(ValueError, match="outside 0 .. 40"):
        pack.plan(_battery()["pgse"], K=pack.K + 1)


def test_explicit_counts_override_the_plan_and_carry_their_error(pack):
    seq = _battery()["pgse"]
    p = pack.plan(seq, K=5)
    assert p.K == 5 and p.bands_chosen == "explicit" and p.band_error > pack.plan(seq).band_error
    full = pack.plan(seq, tol=None)
    assert full.K == pack.K and full.bands_chosen == "full" and full.band_error == 0.0 and full.bytes_read == full.bytes_stored


# ------------------------------------------------------------------ the replay reads the plan
def test_a_prefix_replay_is_bitwise_the_full_decode_cut_at_the_plan(pack):
    for name, seq in _battery().items():
        pk = _fresh(pack)
        plan = pk.plan(seq)
        _, ew, phi = pk.walker_phases(seq)
        assert isinstance(phi, Planned) and phi.plan == plan
        npt.assert_array_equal(np.asarray(phi), _cut_phase(pk, seq, plan.K))
        # the same cut from a pack whose full decode is already held: a view of it, the same bits
        warm = _fresh(pack)
        _r_compression.read_position_coeffs(warm.arrays, dtype=np.float64)
        npt.assert_array_equal(np.asarray(warm.walker_phases(seq)[2]), np.asarray(phi))
        S = pk.replay(seq, complex_signal=True)
        assert S.plan == plan
        P = pk._prepare(seq, tissue=None, scanner=None, orientation=None, compartment=None)
        E = _signal_factor(_cut_phase(pk, seq, plan.K), P["voxel"])
        npt.assert_array_equal(np.asarray(S), (P["pathway"] * P["ew"][:, None] * E).sum(0) / P["norm"])
    # an explicit K equals the plan's when they name the same count
    pk = _fresh(pack)
    seq = _battery()["ogse100"]
    npt.assert_array_equal(np.asarray(pk.replay(seq, K=pk.plan(seq).K)), np.asarray(pk.replay(seq)))


def test_tol_none_is_the_full_decode_of_the_pack(pack):
    pk = _fresh(pack)
    seq = _battery()["ogse200"]
    P = pk._prepare(seq, tissue=None, scanner=None, orientation=None, compartment=None)
    (seg, t0, T_s), = P["windows"]
    C = _r_compression.read_position_coeffs(seg.arrays, dtype=np.float64, cached=False)
    Wp = _r_compression.band_projection(P["G_eff_wf"], P["dt_wf"], T_s, pk.K, t0=t0)
    want = C.reshape(P["n_w"], -1) @ (GAMMA * Wp).reshape(Wp.shape[0], -1).T
    got = pk.walker_phases(seq, tol=None)[2]
    npt.assert_array_equal(np.asarray(got), want)
    assert got.plan.K == pk.K and got.plan.bands_chosen == "full"


def test_the_planned_error_bounds_the_signal_it_costs(pack):
    """The replay at the plan against the replay of every band: within the planned error on each waveform."""
    pk = _fresh(pack)
    for name, seq in _battery().items():
        S, full = pk.replay(seq), pk.replay(seq, tol=None)
        assert S.plan.K <= full.plan.K
        assert np.abs(S - full).max() <= S.plan.error, name
        assert S.plan.error <= S.plan.limit


@pytest.mark.skipif(not os.path.exists(PORES), reason="the canonical-pores pack is not on this machine")
def test_the_planned_error_bounds_the_signal_on_a_pores_pack_including_ogse_400():
    from dmipy_sim.replay.replay import read_rpk
    pk = read_rpk(PORES)
    n_t, T = 1001, 0.1
    axis = [[1.0, 0.0, 0.0]]
    rows = {"pgse 10/40": pgse(axis, 0.01, 0.04, bvalues=[3e9], TE=T, n_t=n_t, slew_rate=200.0)}
    for f in (50.0, 100.0, 200.0, 400.0, 800.0):
        g = min(0.3, 200.0 / (2 * np.pi * f))
        rows[f"ogse {f:.0f}"] = ogse(axis, f, 0.4 * T, shape="cosine", gradient_strengths=[g], TE=T, n_t=n_t, slew_rate=200.0)
    for name, seq in rows.items():
        S, full = pk.replay(seq), pk.replay(seq, tol=None)
        assert np.abs(S - full).max() <= S.plan.error + 1e-12, name
    assert pk.plan(rows["ogse 400"]).K == 82                                # the issue's worst row of the battery
    assert pk.plan(rows["pgse 10/40"]).K < 40
    assert pk.decoded_bytes < 0.4 * pk.n_walkers * (pk.K + 2) * 3 * 8 * 3   # not three full decodes


def test_a_zero_gradient_row_decodes_no_position(pack):
    pk = _fresh(pack)
    seq = gre(0.01, n_t=int(round(0.01 / DT)) + 1)
    S = pk.replay(seq)
    assert S.plan.K == 0 and S.plan.band_error == 0.0
    assert pk.decoded_bytes == 0
    S2 = pk.replay(seq, tissue=Tissue(rho2=1e-6))                          # the contact is read, still no band of the positions
    assert S2.plan.K == 0 and pk.decoded_bytes > 0 and not any(k.startswith("position") for k in cache_kinds(pk))


# ------------------------------------------------------------------ the decoded cache
def test_prefixes_are_extended_not_decoded_again(pack, monkeypatch):
    calls = []
    real = _r_compression._read_position_coeffs
    monkeypatch.setattr(_r_compression, "_read_position_coeffs", lambda a, ax, dt, lo=0, hi=None: (calls.append((lo, hi)), real(a, ax, dt, lo, hi))[1])
    pk = _fresh(pack)
    a = _r_compression.read_position_coeffs(pk.arrays, dtype=np.float64, bands=10)
    b = _r_compression.read_position_coeffs(pk.arrays, dtype=np.float64, bands=25)          # only bands 10 .. 24 are decoded
    c = _r_compression.read_position_coeffs(pk.arrays, dtype=np.float64, bands=17)         # a view of the held prefix
    assert calls == [(0, 10), (10, 25)]
    npt.assert_array_equal(b[:, :12], a)
    assert c.base is not None and not c.flags.writeable and c.shape[1] == 19
    assert pk.decoded_bytes == b.nbytes                                        # the longer prefix replaced the shorter
    full = _r_compression.read_position_coeffs(pk.arrays, dtype=np.float64)
    npt.assert_array_equal(full[:, :27], b)
    n_calls = len(calls)
    d_ = _r_compression.read_position_coeffs(pk.arrays, dtype=np.float64, bands=30)        # the full decode is held: a view of it
    assert len(calls) == n_calls and d_.base is full
    # every count is the leading entries of the full decode, whatever range it ends in
    ref = _r_compression.read_position_coeffs(_fresh(pack).arrays, dtype=np.float64, cached=False)
    for K in (0, 1, 7, 8, 9, 15, 16, 17, 39, 40):
        npt.assert_array_equal(_r_compression.read_position_coeffs(_fresh(pack).arrays, dtype=np.float64, bands=K), ref[:, :K + 2], err_msg=str(K))
    with pytest.raises(ValueError, match="outside the 40 bands"):
        _r_compression.read_position_coeffs(pk.arrays, bands=41)


def test_a_plan_reads_a_replay_with_a_zero_budget_the_same(pack):
    cached, bare = _fresh(pack), _fresh(pack)
    bare.decode_budget = 0
    seq = _battery()["ogse200"]
    npt.assert_array_equal(np.asarray(cached.replay(seq)), np.asarray(bare.replay(seq)))
    assert bare.decoded_bytes == 0


def test_a_protocol_carries_one_plan_per_sequence(pack):
    from dmipy_replay.acquisition.scanner_sequence import Protocol
    rows = list(_battery().values())[:2]
    prot = Protocol(tuple(rows))
    S = _fresh(pack).replay(prot)
    assert len(S.plan) == 2 and [p.K for p in S.plan] == [pack.plan(s).K for s in rows]


# ------------------------------------------------------------------ the field modes
def test_field_modes_are_planned_and_a_prefix_is_bitwise_the_truncated_read(field_pack):
    pk = _fresh(field_pack)
    tissue = Tissue(chi_iso=-1e-7, chi_aniso=-5e-8)
    seq = d.pgse([[1, 0, 0]], 1.2e-4, 3.6e-4, bvalues=[2e8], TE=8e-4, n_t=int(round(8e-4 / pk.dt)) + 1, slew_rate=np.inf)
    plan = pk.plan(seq, tissue=tissue, scanner=3.0)
    assert plan.modes_stored == 4 and 0 <= plan.modes <= 4
    assert plan.mode_variance.startswith("measured on all 90 walkers")
    from dmipy_replay.replay.field import susc_path_coeffs
    from dmipy_replay.replay.replay import _path_field_channels
    pm = pk.meta["compression"]["channels"]["susc_field_basis"]
    C_all, _ = susc_path_coeffs(_fresh(field_pack).arrays, pm)
    gate = np.asarray(seq.effective_gate, np.float64).reshape(-1)[:-1]
    for M in (1, 2, 3):
        ref = _fresh(field_pack)
        C_cut, _ = susc_path_coeffs(ref.arrays, pm, modes=M)
        npt.assert_array_equal(C_cut, C_all[:, :, :M])                         # the leading modes of the full decode, bit for bit
        P = ref._prepare(seq, tissue=tissue, scanner=3.0, orientation=None, compartment=None,
                         read=dict(tol=PLAN_TOL, K=pk.K, modes=M))
        assert P["plan"].modes == M and P["plan"].modes_chosen == "explicit"
        Psi, _ = _path_field_channels(P, seq)
        g = _r_compression.cosine_projection(gate.reshape(1, -1, 1), float(seq.dt), float(pm["T"]), M)[0, :, 0]
        npt.assert_array_equal(Psi, GAMMA * np.einsum("k,wck->wc", g, C_all[:, :, :M]))
        got = _fresh(field_pack).walker_phases(seq, tissue=tissue, scanner=3.0, K=pk.K, modes=M)[2]
        assert got.plan.modes == M
    p0 = pk.plan(seq, tissue=tissue, scanner=3.0, K=pk.K, modes=0)
    assert p0.modes == 0 and p0.mode_error > 0.0
    phi0 = pk.walker_phases(seq, tissue=tissue, scanner=3.0, K=pk.K, modes=0)[2]
    npt.assert_array_equal(np.asarray(phi0), np.asarray(pk.walker_phases(seq, K=pk.K)[2]))     # no mode read: the gradient alone
    with pytest.raises(ValueError, match="outside 1 .. 4"):
        susc_path_coeffs(pk.arrays, pm, modes=0)


def test_primitives_carry_the_plan_and_the_field_at_the_certified_scale(pack):
    pk = _fresh(pack)
    seq = _battery()["ogse200"]
    prim = pk.walker_primitives(seq)
    assert prim.plan == pk.plan(seq)
    npt.assert_array_equal(prim.phi, np.asarray(pk.walker_phases(seq)[2]))
    assert prim.signal().plan == prim.plan


# ------------------------------------------------------------------ the Hub, by range
class _Ranges(http.server.BaseHTTPRequestHandler):
    """A GET that honours ``Range`` over one in-memory payload, recording every range asked (``type(self).asked``)."""
    data = b""
    asked = []

    def do_GET(self):
        body = type(self).data
        rng = self.headers.get("Range")
        start_s, _, end_s = rng.split("=", 1)[1].partition("-")
        start, end = int(start_s), min(int(end_s), len(body) - 1)
        type(self).asked.append((start, end))
        chunk = body[start:end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def log_message(self, *a):
        pass


@contextlib.contextmanager
def _serve(data):
    handler = type("H", (_Ranges,), {"data": data, "asked": []})
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/pack.rpk", handler
    finally:
        httpd.shutdown()
        t.join()


def _tensor_spans(blob):
    """Name -> inclusive byte span of every tensor in a .rpk byte string."""
    import json
    n = int.from_bytes(blob[:8], "little")
    head = json.loads(blob[8:8 + n])
    head.pop("__metadata__", None)
    return {k: (8 + n + v["data_offsets"][0], 8 + n + v["data_offsets"][1] - 1) for k, v in head.items()}


def test_a_remote_pack_fetches_only_the_planned_ranges_and_replays_bit_for_bit(pack, tmp_path):
    from dmipy_sim.replay.remote import RemotePack
    local = tmp_path / "pack.rpk"
    pack.save(str(local))
    data = local.read_bytes()
    spans = _tensor_spans(data)
    seq = _battery()["pgse"]
    tissue = Tissue(rho2=1e-6)
    with _serve(data) as (url, handler):
        remote = RemotePack("hf://owner/plan/pack.rpk", url=url, revision="rev0")
        header_bytes = remote.bytes_fetched
        assert header_bytes < 0.05 * len(data)
        # the plan comes from the header alone: no tensor moved
        plan = remote.plan(seq)
        assert plan == ReplayPack.load(str(local)).plan(seq) and remote.bytes_fetched == header_bytes
        S = remote.replay(seq)
        want = ReplayPack.load(str(local)).replay(seq)
        npt.assert_array_equal(np.asarray(S), np.asarray(want))
        assert S.plan == plan
        # nothing beyond the plan's band ranges was asked for: no request touches a band-range tensor above K
        starts = [int(r["bands"][0]) for r in pack.meta["compression"]["container"]]
        import re
        beyond = [spans[k] for k in spans if re.fullmatch(r"pos_[xyz]_b\d+(u\d+)?", k) and
                  starts[int(re.match(r"pos_[xyz]_b(\d+)", k).group(1))] >= plan.K]
        assert beyond and not any(s <= b and a <= e for a, b in handler.asked for s, e in beyond)
        assert not any(s <= b and a <= e for a, b in handler.asked for k, (s, e) in spans.items() if k.startswith("blt_"))   # no contact asked for
        assert 0 < remote.bytes_fetched < 0.6 * len(data)
        # a replay with a relaxivity adds the contact tensors; a higher band adds ranges, never the ones held
        before = remote.bytes_fetched
        S2 = remote.replay(_battery()["ogse500"], tissue=tissue)
        npt.assert_array_equal(np.asarray(S2), np.asarray(ReplayPack.load(str(local)).replay(_battery()["ogse500"], tissue=tissue)))
        assert remote.bytes_fetched > before
        again = remote.bytes_fetched
        remote.replay(_battery()["ogse500"], tissue=tissue)
        assert remote.bytes_fetched == again                                   # held: a repeat moves nothing


def test_a_remote_pack_with_field_modes_fetches_the_mode_ranges_it_plans(field_pack, tmp_path):
    from dmipy_sim.replay.remote import RemotePack
    local = tmp_path / "field.rpk"
    field_pack.save(str(local))
    data = local.read_bytes()
    tissue = Tissue(chi_iso=-1e-7, chi_aniso=-5e-8)
    seq = d.pgse([[1, 0, 0]], 1.2e-4, 3.6e-4, bvalues=[2e8], TE=8e-4, n_t=int(round(8e-4 / field_pack.dt)) + 1, slew_rate=np.inf)
    with _serve(data) as (url, handler):
        remote = RemotePack("hf://owner/plan/field.rpk", url=url, revision="rev0")
        want = ReplayPack.load(str(local))
        plan = remote.plan(seq, tissue=tissue, scanner=3.0)                    # a pilot of the modes is fetched for the variance
        assert plan == want.plan(seq, tissue=tissue, scanner=3.0)
        S = remote.replay(seq, tissue=tissue, scanner=3.0)
        npt.assert_array_equal(np.asarray(S), np.asarray(want.replay(seq, tissue=tissue, scanner=3.0)))
        S0 = remote.replay(seq)                                                # no field term: no mode tensor is read for it
        npt.assert_array_equal(np.asarray(S0), np.asarray(want.replay(seq)))


def test_a_prefix_without_rpk_is_the_columnar_layout_and_a_rpk_is_the_remote_reader(monkeypatch):
    import dmipy_sim.replay.remote as remote
    with pytest.raises(ValueError, match="a pack on the Hub is a .rpk file"):
        remote.RemotePack("hf://owner/name/prefix", url="http://unused")
    seen = []
    monkeypatch.setattr(remote, "RemotePack", lambda uri, **kw: seen.append(uri) or "remote")
    assert ReplayPack.open("hf://owner/name/packs/x.rpk") == "remote" and seen == ["hf://owner/name/packs/x.rpk"]
