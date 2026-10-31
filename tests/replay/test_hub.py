"""The hub interface (:mod:`dmipy_sim.replay.hub`) over its fake: a 429 and a 5xx on a commit are retried, a commit that
must not block is not, a read that fails is retried and a missing file is not."""
import pytest

from dmipy_sim.replay import hub as hubmod
from dmipy_sim.replay.hub import FakeHub


@pytest.fixture
def hub(tmp_path):
    h = FakeHub(str(tmp_path / "hub"))
    h.commit({"manifest.json": b"{}"}, [], "seed the repository")
    return h


def test_a_429_and_an_outage_are_retried_and_tries_1_is_not(hub, monkeypatch):
    monkeypatch.setattr(hubmod, "RATE_LIMIT_WAIT_S", 0.01); monkeypatch.setattr(hubmod, "OUTAGE_WAIT_S", 0.01)
    hub.fail_429.append("pack"); hub.fail_500 += ["pack"] * 3
    hub.commit({"a.txt": b"a"}, [], "pack a")
    assert hub.exists("a.txt") and not hub.fail_429 and not hub.fail_500
    hub.fail_500.append("quick"); n = len(hub.log)
    with pytest.raises(Exception, match="500"):
        hub.commit({"b.txt": b"b"}, [], "quick b", tries=1)
    assert len(hub.log) == n and not hub.exists("b.txt")


def test_a_read_that_fails_is_retried_and_a_missing_file_is_not(hub, monkeypatch):
    waits = []
    monkeypatch.setattr(hubmod.time, "sleep", lambda s: waits.append(s))
    hub.fail_read += ["exists", "exists", "files", "get"]
    assert hub.exists("manifest.json") and hub.files() and hub.get("manifest.json")
    assert waits == [30, 60, 30, 30]
    with pytest.raises(FileNotFoundError):
        hub.get("nothing/here.json")
    assert waits == [30, 60, 30, 30]
    hub.fail_read += ["exists"] * 8                                      # beyond READ_TRIES: the error surfaces
    with pytest.raises(Exception, match="the hub failed"):
        hub.exists("manifest.json")


def test_a_stale_parent_is_raised_at_once(hub):
    with pytest.raises(hubmod.StaleParent):
        hub.commit({"c.txt": b"c"}, [], "c", parent="fake-0")
