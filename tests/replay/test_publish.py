"""Publishing a pack (dmipy_sim.replay.publish) against the fake hub: the file, its manifest row and the rendered card
go up in one commit; a second publish replaces the row; a pack without its contract is refused; the sha256 the hub holds
is checked on publish and on ``ReplayPack.load("hf://...")``; a local load is untouched."""
import json
import os
import shutil

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.fill import FakeHub, sha256_of
from dmipy_sim.replay import ReplayPack
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay import publish as pub

ENV = dict(bvals=[0.0, 1e9], dirs=[[1, 0, 0]], ogse_periods=[2], shortd_b=1e9, shortd_deltas_frac=[0.05],
           B0_list=[], theta_deg=[0], delta_frac=0.2, Delta_frac=0.5, rho_list=[1e-5])
REPO = "owner/packs"


def _pack(id="test/cyl", **kw):
    walk = d.simulate_trajectories(200, 2e-9, d.Cylinder(2e-6, (0, 0, 1)), 4e-3, 5e-4, seed=0, require_gpu=False)
    return build_replay_pack(walk, id=id, K=4, envelope=ENV, **{"license": "CC-BY-4.0", "citation": "a test pack", **kw})


@pytest.fixture(scope="module")
def pack():
    return _pack()


@pytest.fixture
def hub(tmp_path):
    return FakeHub(str(tmp_path / "hub"))


def _manifest(hub):
    return json.load(open(os.path.join(hub.root, "manifest.json")))


def _hub_module(monkeypatch):
    """The ``huggingface_hub`` module, or a stub of the two names the load path reads (``hf_hub_download`` and the
    not-found errors) when the optional dependency is not installed, so the path is tested without it."""
    import sys
    import types
    try:
        import huggingface_hub
    except ImportError:
        huggingface_hub = types.ModuleType("huggingface_hub"); errors = types.ModuleType("huggingface_hub.errors")
        for name in ("EntryNotFoundError", "RepositoryNotFoundError", "RevisionNotFoundError", "HfHubHTTPError"):
            setattr(errors, name, type(name, (Exception,), {}))
        huggingface_hub.errors = errors; huggingface_hub.hf_hub_download = None
        monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
        monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return huggingface_hub


def _serve_from(hub, monkeypatch):
    """``hf_hub_download`` reading the fake hub's directory, so ``ReplayPack.load("hf://...")`` runs without the network."""
    huggingface_hub = _hub_module(monkeypatch)
    import huggingface_hub.errors
    EntryNotFoundError = huggingface_hub.errors.EntryNotFoundError

    def fake_download(repo_id, filename, *, repo_type=None, **kw):
        assert repo_id == REPO and repo_type == "dataset"
        src = os.path.join(hub.root, filename)
        if not os.path.isfile(src):
            raise EntryNotFoundError(f"{filename} is not in {repo_id}")
        dst = os.path.join(hub.root, "..", "cache", filename); os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        return dst
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)


def test_publish_puts_the_file_the_manifest_row_and_the_card_up_in_one_commit(pack, hub):
    uri = pub.publish(pack, REPO, hub=hub)
    assert uri == f"hf://{REPO}/packs/test-cyl.rpk"
    assert hub.exists("packs/test-cyl.rpk") and hub.exists("manifest.json") and hub.exists("README.md")
    assert len(hub.log) == 1 and set(hub.log[0]["adds"]) == {"packs/test-cyl.rpk", "manifest.json", "README.md"}
    man = _manifest(hub)
    assert man["schema"] == "substratecommons/1" and len(man["packs"]) == 1
    row = man["packs"][0]
    assert row["id"] == "test/cyl" and row["path"] == "packs/test-cyl.rpk"
    assert row["sha256"] == sha256_of(os.path.join(hub.root, "packs/test-cyl.rpk"))
    assert row["bytes"] == os.path.getsize(os.path.join(hub.root, "packs/test-cyl.rpk"))
    assert row["n_walkers"] == 200 and row["K"] == 4 and row["method"] == "bridge_dst"
    assert row["T_max"] == pytest.approx(4e-3) and row["dt_traj"] == pytest.approx(5e-4)
    assert row["channels"][0] == "positions" and "boundary_local_time" in row["channels"]
    assert row["floor_max"] == pack.fidelity["floor_max"] and row["err_max"] == pack.fidelity["err_max"]
    assert row["within_2x_floor"] is True and row["license"] == "CC-BY-4.0" and row["citation"] == "a test pack"
    assert row["substrate"] == "analytic/cylinder" and len(row["commit"]) == 40
    assert row["segments"]["n"] == 1                                      # every pack declares its table (RPK.md 4.3)
    assert man["substrate"]["id"] == "analytic/cylinder" and [p["name"] for p in man["substrate"]["pools"]] == ["extra", "intra"]
    readme = open(os.path.join(hub.root, "README.md")).read()
    assert readme.startswith("---\nlicense: cc-by-4.0\npretty_name: packs\n---\n")
    assert "`analytic/cylinder`" in readme and "| intra | — | 1 |" in readme
    assert "| `test/cyl` | 4 ms | 200 | 4 | positions, boundary_local_time, compartment |" in readme
    assert f"| {row['commit'][:8]} | `packs/test-cyl.rpk` |" in readme
    assert pack.source is None                               # the temporary file it was written to is not its source


def test_a_file_publishes_the_same_as_the_object_and_the_method_is_the_function(pack, hub, tmp_path):
    pack.save(str(tmp_path / "cyl.rpk"))
    assert pub.publish(str(tmp_path / "cyl.rpk"), REPO, hub=hub, path="mine/cyl.rpk", message="the pack") == f"hf://{REPO}/mine/cyl.rpk"
    assert hub.log[0]["message"] == "the pack"
    assert pack.publish(REPO, hub=hub) == f"hf://{REPO}/packs/test-cyl.rpk"
    rows = _manifest(hub)["packs"]
    assert [r["path"] for r in rows] == ["mine/cyl.rpk", "packs/test-cyl.rpk"] and rows[0]["sha256"] == rows[1]["sha256"]


def test_publishing_twice_replaces_the_row(pack, hub):
    pub.publish(pack, REPO, hub=hub)
    pub.publish(pack, REPO, hub=hub)
    man = _manifest(hub)
    assert len(man["packs"]) == 1 and len(hub.log) == 2
    assert open(os.path.join(hub.root, "README.md")).read().count("`test/cyl`") == 1
    other = _pack(id="test/other")
    pub.publish(other, REPO, hub=hub)
    assert [r["id"] for r in _manifest(hub)["packs"]] == ["test/cyl", "test/other"]
    assert _manifest(hub)["substrate"]["id"] == "analytic/cylinder"


def test_a_pack_without_its_contract_is_refused(hub, tmp_path):
    p = _pack(id="test/nolic")
    del p.meta["license"]
    with pytest.raises(ValueError, match="no license"):
        pub.publish(p, REPO, hub=hub)
    p.meta["license"] = "CC-BY-4.0"; p.meta["fidelity"] = {}; p.meta["citation"] = ""
    with pytest.raises(ValueError, match="no citation, fidelity"):
        pub.publish(p, REPO, hub=hub)
    p.meta["id"] = None; p.save(str(tmp_path / "noid.rpk"))
    with pytest.raises(ValueError, match="no id"):
        pub.publish(str(tmp_path / "noid.rpk"), REPO, hub=hub)
    assert not hub.log and not hub.exists("manifest.json")


def test_segments_are_carried_into_the_row_and_the_card(pack, hub):
    p = ReplayPack(pack.arrays, pack.meta)
    p.meta["walk_params"] = dict(pack.meta["walk_params"], segments=dict(n=4, n_t=3, T=1e-3, seeds=[1, 2, 3, 4]))
    pub.publish(p, REPO, hub=hub)
    row = _manifest(hub)["packs"][0]
    assert row["segments"] == dict(n=4, n_t=3, T=1e-3, seeds=[1, 2, 3, 4])
    assert "| 4 ms (4 × 1 ms) |" in open(os.path.join(hub.root, "README.md")).read()


class _CorruptingHub(FakeHub):
    """A hub that flips a byte of every ``.rpk`` it stores: what the verification after the commit must catch."""

    def _commit(self, adds, deletes, message, expect):
        super()._commit(adds, deletes, message, expect)
        for r in adds:
            if r.endswith(".rpk"):
                b = bytearray(open(self._p(r), "rb").read()); b[-1] ^= 0xFF; open(self._p(r), "wb").write(b)


def test_publish_verifies_what_the_hub_holds(pack, tmp_path):
    hub = _CorruptingHub(str(tmp_path / "bad"))
    with pytest.raises(RuntimeError, match="the hub holds sha256"):
        pub.publish(pack, REPO, hub=hub)


def test_load_from_the_hub_checks_the_manifest_sha256(pack, hub, monkeypatch):
    uri = pub.publish(pack, REPO, hub=hub)
    _serve_from(hub, monkeypatch)
    same = ReplayPack.load(uri)
    assert same.n_walkers == pack.n_walkers and same.id == "test/cyl" and same.digest == pack.digest
    stored = os.path.join(hub.root, "packs/test-cyl.rpk")
    b = bytearray(open(stored, "rb").read()); b[-1] ^= 0xFF; open(stored, "wb").write(b)
    with pytest.raises(ValueError, match="sha256"):
        ReplayPack.load(uri)


def test_load_from_a_hub_without_a_manifest_and_of_a_non_pack(pack, hub, monkeypatch):
    pub.publish(pack, REPO, hub=hub)
    os.remove(os.path.join(hub.root, "manifest.json"))
    _serve_from(hub, monkeypatch)
    assert ReplayPack.load(f"hf://{REPO}/packs/test-cyl.rpk").n_walkers == 200
    with pytest.raises(ValueError, match="takes a pack"):
        ReplayPack.load(f"hf://{REPO}/README.md")


def test_parse_uri():
    assert pub.parse_uri("hf://owner/name/packs/x.rpk") == ("owner/name", "packs/x.rpk")
    assert pub.parse_uri("hf://owner/name/a/b/c.rpk") == ("owner/name", "a/b/c.rpk")
    assert pub.is_hub_uri("hf://o/n/x.rpk") and not pub.is_hub_uri("/o/n/x.rpk") and not pub.is_hub_uri(None)
    for bad in ("owner/name/x.rpk", "hf://owner/name", "hf://owner//x.rpk", "hf://owner"):
        with pytest.raises(ValueError):
            pub.parse_uri(bad)


def test_a_local_load_never_touches_the_hub(pack, tmp_path, monkeypatch):
    monkeypatch.setattr(_hub_module(monkeypatch), "hf_hub_download", lambda *a, **k: pytest.fail("the hub was asked for a local file"))
    p = tmp_path / "cyl.rpk"
    pack.save(str(p))
    assert ReplayPack.load(str(p)).digest == pack.digest and ReplayPack.load(p).source == str(p)


def test_the_cli_prints_the_uri(pack, hub, tmp_path, capsys, monkeypatch):
    pack.save(str(tmp_path / "cyl.rpk"))
    monkeypatch.setattr(pub, "_publish_file", lambda local, meta, repo, **kw: pub.uri_of(repo, kw["path"] or "packs/x.rpk"))
    pub.main([str(tmp_path / "cyl.rpk"), "--repo", REPO, "--path", "packs/y.rpk"])
    assert capsys.readouterr().out.strip() == f"hf://{REPO}/packs/y.rpk"


def test_a_phantom_substrate_cites_a_published_pack_by_its_uri(monkeypatch):
    """A PackSubstrate given an hf:// URI loads the pack through ReplayPack.load, so a phantom cites published packs."""
    from dmipy_sim.phantom import PackSubstrate
    from dmipy_sim.replay.replay import ReplayPack
    seen = {}
    fake = ReplayPack({"pos_x": np.zeros((2, 4), np.float32)}, {"id": "o/p", "walk_params": {"n_t": 3, "segments": {"n": 1, "n_t": 3, "T": 1e-3, "walks": []}}})
    monkeypatch.setattr(ReplayPack, "load", classmethod(lambda cls, uri: seen.setdefault("uri", uri) and fake))
    sub = PackSubstrate("hf://SubstrateCommons/gm-spheres/packs/100ms_K48.rpk", m0=1.0)
    assert sub.uri.startswith("hf://") and sub.name == "100ms_K48"
    assert sub.pack is fake and seen["uri"] == "hf://SubstrateCommons/gm-spheres/packs/100ms_K48.rpk"
