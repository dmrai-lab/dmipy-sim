"""``pack_plan``: the pre-walk plan of a pack per tier -- band count and the bytes at the raw, bit-packed, coded and
stored stages, every figure labelled ``derived``, ``measured`` or ``proxy`` -- compared with a built pack's
``plan_of_pack`` by ``compare_plans``, and checked itself by ``pack_plan_problems``.

``inputs_of(path)`` reads a pack file's own header and returns the keyword arguments for ``pack_plan``; the pack set
is the two committed fixtures and the built fixed-width pack of the module fixture (600 walkers). Planted faults
check that each problem sentence is produced.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_replay.replay import container
from dmipy_replay.replay.container import parse_header
from dmipy_replay.replay.envelope import Envelope
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.replay import bank, plan
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.container import write_rpk
from dmipy_sim.replay.plan import compare_plans, pack_plan, pack_plan_problems, plan_of_pack
from tests._envelope import battery_reach, stated

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
CYLINDER = DATA / "cylinder_k12.rpk"
STRANDS = DATA / "strands_field_k4.rpk"

#: the two committed fixtures, plus the built fixed-width pack of the module fixture appended by ``packs``
PACKS = (CYLINDER, STRANDS)

#: the keys of a tier's dictionary of a pre-walk plan
TIER_KEYS = {"why", "banded", "bands", "channels", "codec", "bits", "eps", "events",
             "raw", "bit_packed", "coded", "stored"}

#: the expected figures of ``pack_plan(**inputs_of(path), codec=None)`` per fixture tier and for the total:
#: (raw lo, raw hi, bit_packed lo, bit_packed hi), ``None`` for a stage the tier does not carry
EXPECTED = {
    CYLINDER.name: {"positions": (67200, 67200, 15744, 38544),
                    "contact": (22400, 22400, 5248, 12848),
                    "occupancy": (400, 400, None, None),
                    "total": (90000, 90000, 21392, 51792)},
    STRANDS.name: {"positions": (6480, 6480, 2748, 4368),
                   "contact": (2160, 2160, 916, 1456),
                   "field": (17280, 8847360, 2352, 4521984),
                   "occupancy": (90, 90, None, None),
                   "weights": (360, 360, None, None),
                   "total": (26370, 8856450, 6466, 4528258)},
}

#: the strands fixture planned with a stated field band count
EXPECTED_FIELD_BANDS_4 = {"field": (17280, 17280, 2352, 8832)}


def _header_of(path):
    """The ``(header, meta)`` of the pack file at ``path``, read from its header only."""
    with open(path, "rb") as f:
        first = f.read(8)
        n = int.from_bytes(first, "little")
        head = first + f.read(n)
    _, hdr = parse_header(head)
    return hdr, json.loads((hdr.get("__metadata__") or {})["rpk"])


def inputs_of(path):
    """The keyword arguments for ``pack_plan`` read from the pack file at ``path``'s own header: its substrate
    metadata as ``spec``, the replay envelope's maxima, the stored band and window, one fixed tolerance, the
    walker count, whether the pack carries ``spin_weights`` and a voxel certificate, and the codec with its chunk
    width."""
    hdr, meta = _header_of(path)
    env = Envelope.from_header(meta)
    win = meta["walk_params"]["windows"]
    built = plan_of_pack(path)
    coded = any(t["codec"] is not None for t in built["tiers"].values())
    chunk_walkers = 4096
    if coded:
        for r in (meta.get("compression") or {}).get("container") or []:
            if r.get("codec") is not None and "chunk_walkers" in r:
                chunk_walkers = int(r["chunk_walkers"])
                break
    n_voxels = int(hdr["voxel_ijk"]["shape"][0]) if "voxel_ijk" in hdr else None
    return {
        "spec": meta["substrate"],
        "envelope": {"band_hz": float(meta["compression"]["temporal_bandwidth_hz"]), "window_s": float(win["T"]),
                     "n_windows": int(win["n"]), "G_max": env.G_max, "slew_max": None, "B0_max": env.B0_max,
                     "chi_max": env.chi_max, "rho2_max": env.rho2_max,
                     "max_refocusing_pulses": env.max_refocusing_pulses},
        "tolerance": 5e-3,
        "n_walkers": int(meta["walk_params"]["n_walkers"]),
        "codec": "bitshuffle-zlib" if coded else None,
        "chunk_walkers": chunk_walkers,
        "weights": "weights" in built["tiers"],
        "n_voxels": n_voxels,
        "field_bands": None,
    }


def _figures_of(node, path=""):
    """Every figure (a dict with a ``lo`` key) under ``node``, as ``(path, figure)`` pairs."""
    return plan._figures_of(node, path)


@pytest.fixture(scope="module")
def walk():
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    return _walk_geometry(600, 2e-9, g, 20e-3, 2.5e-4, seed=7, require_gpu=False)


@pytest.fixture(scope="module")
def built_pack(walk, tmp_path_factory):
    d_ = tmp_path_factory.mktemp("pack_plan")
    path = d_ / "fixed.rpk"
    build_replay_pack(walk, id="test/codec", license="x", citation="x", K=16, out_path=str(path),
                      **stated(of=walk, G_max=battery_reach(walk.T_window)))
    return path


@pytest.fixture(scope="module")
def packs(built_pack):
    return list(PACKS) + [built_pack]


@pytest.fixture(scope="module")
def coded_pack(walk, tmp_path_factory):
    d_ = tmp_path_factory.mktemp("pack_plan_coded")
    path = d_ / "coded.rpk"
    build_replay_pack(walk, id="test/codec", license="x", citation="x", K=16, out_path=str(path),
                      codec="bitshuffle-zlib", chunk_walkers=256,
                      **stated(of=walk, G_max=battery_reach(walk.T_window)))
    return path


def test_the_plan_of_a_packs_own_inputs_agrees_with_plan_of_pack(packs):
    for path in packs:
        kw = inputs_of(path)
        kw["codec"] = None
        pl = pack_plan(**kw)
        built = plan_of_pack(path)
        assert compare_plans(pl, built) == []
        assert list(pl["tiers"]) == list(built["tiers"])
        assert pl["pack"] == {"n_walkers": built["pack"]["n_walkers"], "windows": built["pack"]["windows"]}

    for path in (CYLINDER, STRANDS):
        kw = inputs_of(path)
        kw["codec"] = None
        pl = pack_plan(**kw)
        built = plan_of_pack(path)
        expected = EXPECTED[path.name]
        for tier, (rlo, rhi, blo, bhi) in expected.items():
            fig = pl["total"] if tier == "total" else pl["tiers"][tier]
            assert (fig["raw"]["lo"], fig["raw"]["hi"]) == (rlo, rhi), (path.name, tier)
            if blo is None:
                assert fig["bit_packed"]["lo"] is None
            else:
                assert (fig["bit_packed"]["lo"], fig["bit_packed"]["hi"]) == (blo, bhi), (path.name, tier)
        for tier, t in pl["tiers"].items():
            bt = built["tiers"][tier]
            if tier in ("positions", "contact", "occupancy", "weights"):
                assert t["raw"]["value"] == bt["raw"]["value"], (path.name, tier)
            if bt["bit_packed"]["value"] is not None:
                assert t["bit_packed"]["lo"] <= bt["bit_packed"]["value"] <= t["bit_packed"]["hi"], (path.name, tier)

    kw = inputs_of(STRANDS)
    kw["field_bands"] = 4
    kw["codec"] = None
    pl = pack_plan(**kw)
    built = plan_of_pack(STRANDS)
    for tier, (rlo, rhi, blo, bhi) in EXPECTED_FIELD_BANDS_4.items():
        t = pl["tiers"][tier]
        assert (t["raw"]["lo"], t["raw"]["hi"]) == (rlo, rhi)
        assert (t["bit_packed"]["lo"], t["bit_packed"]["hi"]) == (blo, bhi)
    assert pl["tiers"]["field"]["raw"]["value"] == built["tiers"]["field"]["raw"]["value"] == 17280
    assert pl["tiers"]["field"]["bit_packed"]["lo"] <= built["tiers"]["field"]["bit_packed"]["value"] \
        <= pl["tiers"]["field"]["bit_packed"]["hi"]
    kw = inputs_of(STRANDS)
    kw["codec"] = None
    bands = pack_plan(**kw)["tiers"]["field"]["bands"]
    assert (bands["lo"], bands["hi"]) == (4, 2048)
    assert bands["lo"] <= built["tiers"]["field"]["bands"] <= bands["hi"]


def test_every_figure_has_one_of_the_three_provenances(packs):
    for path in packs:
        pl = pack_plan(**inputs_of(path))
        assert pack_plan_problems(pl) == []
        figs = _figures_of(pl["tiers"]) + _figures_of(pl["total"])
        assert figs
        for _, fig in figs:
            assert set(fig) == {"value", "lo", "hi", "provenance", "source"}
            assert fig["provenance"] in plan.PROVENANCE
        for t in pl["tiers"].values():
            assert t["eps"] is None
        assert pl["tiers"]["positions"]["raw"]["provenance"] == "derived"
    strands = pack_plan(**inputs_of(STRANDS))
    figs = dict(_figures_of(strands["tiers"]))
    proxies = [p for p, f in figs.items() if f["provenance"] == "proxy"]
    derived = [p for p, f in figs.items() if f["provenance"] == "derived" and f["value"] is not None]
    assert any(p.endswith(".bits") for p in proxies)
    assert any(p.endswith("field.bands") for p in proxies)
    assert any(p.endswith("field.bit_packed") for p in proxies)
    assert "positions.raw" in derived


def _drop_spec_key(key):
    def mutate(kw):
        kw["spec"] = {k: v for k, v in kw["spec"].items() if k != key}
    return mutate


def _one_pool_spec(weights=False, mt=False):
    def mutate(kw):
        spec = json.loads(json.dumps(kw["spec"]))
        spec["pools"] = spec["pools"][:1]
        if mt:
            spec["walls"][0]["mt_reactivity"]["inside"] = 1.0
        kw["spec"] = spec
        kw["weights"] = weights
    return mutate


def refusal_problems(table):
    """One sentence per case of ``table`` -- ``(name, mutation)`` applied to the strands inputs -- that does not
    raise a ``ValueError`` whose message names the input."""
    problems = []
    for name, mutate in table:
        kw = inputs_of(STRANDS)
        mutate(kw)
        try:
            pack_plan(**kw)
        except ValueError as e:
            if name not in str(e):
                problems.append(f"{name}: did not refuse by name")
        except Exception as e:
            problems.append(f"{name}: did not refuse by name ({type(e).__name__}: {e})")
        else:
            problems.append(f"{name}: did not refuse by name")
    return problems


def refusal_table():
    """The ``(name, mutation)`` cases of test 3, applied to the strands inputs."""
    return [
        ("tolerance", lambda kw: kw.__setitem__("tolerance", None)),
        ("tolerance", lambda kw: kw.__setitem__("tolerance", -1.0)),
        ("n_walkers", lambda kw: kw.__setitem__("n_walkers", 0)),
        ("n_walkers", lambda kw: kw.__setitem__("n_walkers", float("inf"))),
        ("n_walkers", lambda kw: kw.__setitem__("n_walkers", float("nan"))),
        ("band_hz", lambda kw: kw["envelope"].pop("band_hz")),
        ("window_s", lambda kw: kw["envelope"].pop("window_s")),
        ("n_windows", lambda kw: kw["envelope"].pop("n_windows")),
        ("B0_max", lambda kw: kw["envelope"].__setitem__("B0_max", None)),
        ("chi_max", lambda kw: kw["envelope"].__setitem__("chi_max", None)),
        ("max_refocusing_pulses", lambda kw: kw["envelope"].__setitem__("max_refocusing_pulses", None)),
        ("pools", _drop_spec_key("pools")),
        ("walls", _drop_spec_key("walls")),
        ("weights", _one_pool_spec(weights=True)),
        ("mt", _one_pool_spec(mt=True)),
        ("codec", lambda kw: kw.__setitem__("codec", "lz4")),
        ("chunk_walkers", lambda kw: kw.__setitem__("chunk_walkers", 3)),
        ("field_bands", lambda kw: kw.__setitem__("field_bands", 0)),
        ("n_voxels", lambda kw: kw.__setitem__("n_voxels", 0)),
        ("shares", lambda kw: kw["envelope"].__setitem__("shares", {})),
    ]


def test_missing_inputs_are_refused_by_name():
    assert refusal_problems(refusal_table()) == []

    kw = inputs_of(STRANDS)
    spec = json.loads(json.dumps(kw["spec"]))
    spec["pools"][2]["susceptibility"]["director"] = "unknown"
    kw["spec"] = spec
    pack_plan(**kw)


def test_the_plan_is_json_and_round_trips(packs):
    for path in packs:
        pl = pack_plan(**inputs_of(path))
        s = json.dumps(pl, allow_nan=False)
        assert json.loads(s) == pl
        assert pl["schema"] == "pack_plan/1"
        assert set(pl) == {"schema", "inputs", "pack", "tiers", "total"}
        for t in pl["tiers"].values():
            assert set(t) == TIER_KEYS
        _assert_plain(pl)


def _assert_plain(node):
    if isinstance(node, dict):
        for v in node.values():
            _assert_plain(v)
    elif isinstance(node, list):
        for v in node:
            _assert_plain(v)
    else:
        assert not isinstance(node, np.generic)
        assert node is None or isinstance(node, (bool, int, float, str))


def _dummy_fig(v, lo, hi=None):
    return plan._fig(v, lo, lo if hi is None else hi, provenance="derived", source="dummy")


def test_a_provider_adds_a_tier_and_a_bad_one_is_refused():
    def dummy(inputs):
        assert isinstance(inputs, dict) and "spec" in inputs
        return {"extra": {"why": "dummy", "raw": _dummy_fig(10, 10), "bit_packed": _dummy_fig(None, None),
                          "coded": _dummy_fig(None, None), "stored": _dummy_fig(10, 10)}}

    kw = inputs_of(CYLINDER)
    base = pack_plan(**kw)
    pl = pack_plan(**kw, providers=(dummy,))
    assert list(pl["tiers"])[-1] == "extra"
    assert list(pl["tiers"])[:-1] == list(base["tiers"])
    assert pl["inputs"]["providers"] == ["dummy"]
    assert pl["total"]["raw"]["value"] == base["total"]["raw"]["value"] + 10
    assert pack_plan_problems(pl) == []

    def dup(inputs):
        return {"positions": {"why": "dup", "raw": _dummy_fig(1, 1), "bit_packed": _dummy_fig(1, 1),
                              "coded": _dummy_fig(1, 1), "stored": _dummy_fig(1, 1)}}

    def guess(inputs):
        return {"extra": {"why": "guess", "raw": plan._fig(1, 1, 1, provenance="guess", source="guess"),
                          "bit_packed": _dummy_fig(1, 1), "coded": _dummy_fig(1, 1), "stored": _dummy_fig(1, 1)}}

    def numpy_value(inputs):
        return {"extra": {"why": "numpy", "raw": {"value": np.int64(3), "lo": 3, "hi": 3,
                                                   "provenance": "derived", "source": "numpy"},
                          "bit_packed": _dummy_fig(1, 1), "coded": _dummy_fig(1, 1), "stored": _dummy_fig(1, 1)}}

    def numpy_float(inputs):
        return {"extra": {"why": "numpy float", "raw": {"value": np.float64(3.0), "lo": 3.0, "hi": 3.0,
                                                        "provenance": "derived", "source": "numpy float"},
                          "bit_packed": _dummy_fig(1, 1), "coded": _dummy_fig(1, 1), "stored": _dummy_fig(1, 1)}}

    def no_stored(inputs):
        return {"extra": {"why": "no stored", "raw": _dummy_fig(1, 1), "bit_packed": _dummy_fig(1, 1),
                          "coded": _dummy_fig(1, 1)}}

    for bad, tier in ((dup, "positions"), (guess, "extra"), (numpy_value, "extra"), (numpy_float, "extra"),
                      (no_stored, "extra")):
        with pytest.raises(ValueError) as e:
            pack_plan(**kw, providers=(bad,))
        assert bad.__name__ in str(e.value) and tier in str(e.value), (bad.__name__, str(e.value))


def test_windows_count_per_window_tensors_and_shared_tensors_once(packs, tmp_path):
    fixed = packs[-1]
    arrays, hdr = container.read_safetensors(fixed)
    meta = json.loads(hdr["rpk"])
    two_arrays = dict(arrays)
    two_arrays.update({f"s1/{k}": v for k, v in arrays.items() if not bank._is_shared_key(k)})
    two_meta = json.loads(json.dumps(meta))
    two_meta["walk_params"]["windows"] = {"n": 2, "T": meta["walk_params"]["windows"]["T"], "walks": []}
    two = tmp_path / "two_windows.rpk"
    write_rpk(two, two_arrays, two_meta)

    kw = inputs_of(two)
    assert kw["envelope"]["n_windows"] == 2
    pl = pack_plan(**kw)
    built_two = plan_of_pack(two)
    built_one = plan_of_pack(fixed)
    assert compare_plans(pl, built_two) == []
    assert pl["pack"]["windows"] == 2
    assert pl["tiers"]["occupancy"]["raw"]["value"] == built_one["tiers"]["occupancy"]["raw"]["value"] \
        == built_two["tiers"]["occupancy"]["raw"]["value"]
    assert pl["tiers"]["positions"]["raw"]["value"] == 2 * built_one["tiers"]["positions"]["raw"]["value"] \
        == built_two["tiers"]["positions"]["raw"]["value"]
    assert pack_plan_problems(pl) == []


def test_a_coded_pack_is_planned_sound_and_a_wrong_figure_is_still_reported(coded_pack, monkeypatch):
    kw = inputs_of(coded_pack)
    assert kw["codec"] == "bitshuffle-zlib" and kw["chunk_walkers"] == 256
    built = plan_of_pack(coded_pack)
    pl = pack_plan(**kw)
    assert compare_plans(pl, built) == []
    assert pack_plan_problems(pl) == []
    for t in ("positions", "contact"):
        assert built["tiers"][t]["codec"] == "bitshuffle-zlib" == pl["tiers"][t]["codec"]
        for stage in ("coded", "stored"):
            fig = pl["tiers"][t][stage]
            assert fig["value"] is None
            assert fig["provenance"] == "proxy"
            assert fig["source"] == ("bound:lane_widened_block_padded" if stage == "coded" else "coded")
            assert fig["lo"] <= built["tiers"][t]["stored"]["value"] <= fig["hi"]
        # the raw and bit-packed stages are still pinned and compared on a coded pack
        assert pl["tiers"][t]["raw"]["value"] == built["tiers"][t]["raw"]["value"]
        assert pl["tiers"][t]["bit_packed"]["lo"] <= built["tiers"][t]["bit_packed"]["value"] \
            <= pl["tiers"][t]["bit_packed"]["hi"]

    # a plant: a wrong raw on the coded pack is still reported
    real_mode_count = plan.mode_count
    with monkeypatch.context() as m:
        m.setattr(plan, "mode_count", lambda T, f: real_mode_count(T, f) + 1)
        problems = compare_plans(pack_plan(**kw), built)
    assert any("tier positions: raw" in s and "differs from the planned" in s for s in problems), problems

    # a plant: a wrong bit-packed range on the coded pack is still reported (its stored rows are 48792 bytes,
    # above the hi a clip to 8 bits plans: 600*3*24 + 24*600 + 12*16)
    with monkeypatch.context() as m:
        m.setattr(plan, "BITS_RANGE", (3, 8))
        problems = compare_plans(pack_plan(**kw), built)
    assert any("tier positions: bit_packed" in s and "is outside the planned" in s for s in problems), problems


def test_each_planted_fault_is_found(packs, tmp_path, monkeypatch):
    cylinder, strands = CYLINDER, STRANDS
    kw_c = inputs_of(cylinder)
    kw_c["codec"] = None
    kw_s = inputs_of(strands)
    kw_s["codec"] = None
    assert compare_plans(pack_plan(**kw_c), plan_of_pack(cylinder)) == []
    assert compare_plans(pack_plan(**kw_s), plan_of_pack(strands)) == []

    # A, K one too many
    real_mode_count = plan.mode_count
    with monkeypatch.context() as m:
        m.setattr(plan, "mode_count", lambda T, f: real_mode_count(T, f) + 1)
        problems = compare_plans(pack_plan(**kw_c), plan_of_pack(cylinder))
    assert any("tier positions: raw" in s and "differs from the planned" in s for s in problems), problems

    # B, a clip too narrow
    with monkeypatch.context() as m:
        m.setattr(plan, "BITS_RANGE", (3, 12))
        problems = compare_plans(pack_plan(**kw_c), plan_of_pack(cylinder))
    assert any("tier positions: bit_packed" in s and "is outside the planned" in s for s in problems), problems

    # C, a shared tensor counted per window
    fixed = packs[-1]
    arrays, hdr = container.read_safetensors(fixed)
    meta = json.loads(hdr["rpk"])
    two_arrays = dict(arrays)
    two_arrays.update({f"s1/{k}": v for k, v in arrays.items() if not bank._is_shared_key(k)})
    two_meta = json.loads(json.dumps(meta))
    two_meta["walk_params"]["windows"] = {"n": 2, "T": meta["walk_params"]["windows"]["T"], "walks": []}
    two = tmp_path / "two_windows.rpk"
    write_rpk(two, two_arrays, two_meta)
    kw_two = inputs_of(two)
    with monkeypatch.context() as m:
        m.setattr(bank, "_SHARED_KEYS", ())
        problems = compare_plans(pack_plan(**kw_two), plan_of_pack(two))
    assert any("tier occupancy: raw" in s for s in problems), problems

    # D, a spec that disagrees with the pack
    kw = json.loads(json.dumps(kw_c))
    kw["spec"]["walls"][0]["permeability"]["in_to_out"] = 1e-6
    problems = compare_plans(pack_plan(**kw), plan_of_pack(cylinder))
    assert any("tier occupancy: raw 400 is outside the planned" in s for s in problems), problems

    # E, the wrong channel set
    kw = json.loads(json.dumps(kw_s))
    for pool in kw["spec"]["pools"]:
        if isinstance(pool.get("susceptibility"), dict):
            pool["susceptibility"]["chi_aniso"] = None
    problems = compare_plans(pack_plan(**kw), plan_of_pack(strands))
    assert any("tier field: channels 12 differs from the planned 6" in s for s in problems), problems

    # F, a fourth label
    def exact_fig(value=None, lo=None, hi=None, *, provenance, source):
        return {"value": value, "lo": lo, "hi": hi,
                "provenance": "exact" if provenance == "derived" else provenance, "source": source}

    with monkeypatch.context() as m:
        m.setattr(plan, "_fig", exact_fig)
        problems = pack_plan_problems(pack_plan(**kw_s))
    assert any("carries provenance 'exact'" in s for s in problems), problems

    # G, a proxy with a made-up centre
    real_fig = plan._fig

    def centred_fig(value=None, lo=None, hi=None, *, provenance, source):
        if provenance == "proxy" and lo is not None:
            value = lo
        return real_fig(value, lo, hi, provenance=provenance, source=source)

    with monkeypatch.context() as m:
        m.setattr(plan, "_fig", centred_fig)
        problems = pack_plan_problems(pack_plan(**kw_s))
    assert any("is proxy and states a central value" in s for s in problems), problems

    # H, not JSON
    def numpy_fig(value=None, lo=None, hi=None, *, provenance, source):
        return {"value": None if value is None else np.int64(value),
                "lo": None if lo is None else np.int64(lo),
                "hi": None if hi is None else np.int64(hi),
                "provenance": provenance, "source": source}

    with monkeypatch.context() as m:
        m.setattr(plan, "_fig", numpy_fig)
        problems = pack_plan_problems(pack_plan(**kw_s))
    assert any("the plan is not JSON" in s for s in problems), problems

    # I, the refusals removed: every case of the full table must then be reported
    with monkeypatch.context() as m:
        m.setattr(plan, "_refuse", lambda *a, **k: None)
        problems = refusal_problems(refusal_table())
    reported = {p.split(":")[0] for p in problems}
    assert reported == {name for name, _ in refusal_table()}, problems
    assert all("did not refuse by name" in s for s in problems), problems

    # J, a provider validated by nobody
    def guess(inputs):
        return {"extra": {"why": "guess", "raw": plan._fig(1, 1, 1, provenance="guess", source="guess"),
                          "bit_packed": _dummy_fig(1, 1), "coded": _dummy_fig(1, 1), "stored": _dummy_fig(1, 1)}}

    with monkeypatch.context() as m:
        m.setattr(plan, "_validate_provider_tier", lambda *a, **k: None)
        pl = pack_plan(**kw_s, providers=(guess,))
        problems = pack_plan_problems(pl)
    assert any("carries provenance 'guess'" in s for s in problems), problems

    # a stand-in source on a derived figure
    def stand_in_fig(value=None, lo=None, hi=None, *, provenance, source):
        return real_fig(value, lo, hi, provenance=provenance,
                        source="stand-in:x" if source.startswith("dmipy_replay.replay.storage") else source)

    with monkeypatch.context() as m:
        m.setattr(plan, "_fig", stand_in_fig)
        problems = pack_plan_problems(pack_plan(**kw_s))
    assert any("stand-in" in s for s in problems), problems
