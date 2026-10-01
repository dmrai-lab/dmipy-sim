"""Window-ranged reads of a replay pack from the Hugging Face Hub (dmipy-sim#525 item 2):
``ReplayPack.load(uri, windows=range(k))`` fetches only the pack's leading windows by HTTP byte range and caches
them locally; a one-window read replays bit for bit as the whole file does, and an acquisition that reaches a
window the pack was not loaded with is refused by name.

The byte-range selection is exercised against a local HTTP server that honours ``Range`` (``http.server`` does
not out of the box, so a 40-line handler stands in for the Hub's own resolve URL -- :func:`fetch_windows`'s
``url=`` override is for exactly this). The one test that touches the real Hub is marked ``slow`` and skips
itself without a login."""
import contextlib
import http.server
import json
import os
import tempfile
import threading

import numpy as np
import numpy.testing as npt
import pytest

import dmipy_sim as d
from dmipy_sim.replay.replay import ReplayPack, read_rpk
from dmipy_sim.replay import publish as pub
from tests.test_bank import _slab_master, _lean_env

N_T, DT, N_W = 61, 5e-4, 300          # 30 ms of walk, three windows of 10 ms each
REPO, PATH = "owner/windows", "packs/windows-fixture.rpk"
URI = pub.uri_of(REPO, PATH)


def _fixture_master():
    m = _slab_master(n_w=N_W, seed=0)
    for k in ("traj", "comp", "dlog_b"):
        m[k] = np.asarray(m[k])[:, :N_T]
    m["T_max"] = (N_T - 1) * DT
    return m


def _fixture_pack():
    """A tiny, deterministic 3-window pack (positions + C1 + C2): bridge_dst at K=19 over 21-save, 10 ms
    windows -- small enough to serve from memory, with real ``sN/`` tensors to select by byte range."""
    return d.build_replay_pack(_fixture_master(), id="test/windows-fixture", K=19, envelope=_lean_env(),
                               segment_T=0.01, license="CC-BY-4.0", citation="dmipy-sim window-read test fixture")


def _seq(TE):
    return d.pgse([[1, 0, 0]], 0.002, 0.006, gradient_strengths=0.2, n_t=TE_N_T(TE), slew_rate=np.inf, TE=TE)


def TE_N_T(TE):
    return int(round(TE / DT)) + 1


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """The 40 lines ``http.server.SimpleHTTPRequestHandler`` does not give you: a GET that honours ``Range``
    and answers 206 with exactly the requested bytes, serving one in-memory payload (``type(self).data``)."""
    data = b""

    def do_GET(self):
        body = type(self).data
        rng = self.headers.get("Range")
        if rng:
            spec = rng.split("=", 1)[1]
            start_s, _, end_s = spec.partition("-")
            start, end = int(start_s), (int(end_s) if end_s else len(body) - 1)
            end = min(end, len(body) - 1)
            chunk = body[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


@contextlib.contextmanager
def _serve(data):
    handler = type("Handler", (_RangeHandler,), {"data": data})
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/pack.rpk"
    finally:
        httpd.shutdown()
        t.join()


@pytest.fixture(scope="module")
def pack():
    return _fixture_pack()


@pytest.fixture
def served(pack, tmp_path):
    """The fixture pack served over a local Range-honouring HTTP server, and the Hub cache directory
    redirected under ``tmp_path`` so the test never touches the real cache."""
    local = tmp_path / "src.rpk"
    pack.save(str(local))
    data = local.read_bytes()
    with _serve(data) as url:
        yield url, data


def test_a_one_window_read_fetches_only_its_bytes_and_matches_the_whole_file(pack, served, tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hfcache"))
    url, data = served
    assert pack.n_segments == 3
    local1 = pub.fetch_windows(URI, range(1), url=url, revision="rev0")
    p1 = read_rpk(local1)
    assert p1.meta["windows_present"] == 1
    assert p1.meta["provenance"]["hub_window_read"]["revision"] == "rev0"
    fetched1 = p1.meta["provenance"]["hub_window_read"]["bytes_fetched"]
    assert 0 < fetched1 < len(data)                                        # strictly a part of the whole file
    # every tensor a one-window read can hold (shared + segment 0's) is byte-identical to the whole file's
    seg0_names = set(pack._segment_arrays(0))
    assert set(p1.arrays) == seg0_names
    for k in seg0_names:
        npt.assert_array_equal(p1.arrays[k], pack.arrays[k], err_msg=k)
    # a one-window acquisition replays bit for bit as the full file does (RPK.md 4.3)
    seq = _seq(0.01)
    npt.assert_array_equal(p1.replay(seq, complex_signal=True), pack.replay(seq, complex_signal=True))

    # a larger k fetches only the missing windows and rewrites the same local file
    local2 = pub.fetch_windows(URI, range(2), url=url, revision="rev0")
    assert local2 == local1
    p2 = read_rpk(local2)
    assert p2.meta["windows_present"] == 2
    fetched2 = p2.meta["provenance"]["hub_window_read"]["bytes_fetched"]
    assert fetched2 < fetched1                                            # window 1 alone, not window 0 again
    for k in p2.arrays:
        npt.assert_array_equal(p2.arrays[k], pack.arrays[k], err_msg=k)

    # a second range(1) call, already satisfied, fetches nothing new
    local3 = pub.fetch_windows(URI, range(1), url=url, revision="rev0")
    p3 = read_rpk(local3)
    assert p3.meta["windows_present"] == 2                                 # the file already holds more


def test_an_acquisition_beyond_the_loaded_windows_is_refused_by_name(pack, served, tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hfcache"))
    url, _ = served
    local = pub.fetch_windows(URI, range(1), url=url, revision="rev0")
    p1 = read_rpk(local)
    assert p1.windows_present == 1 and p1.n_segments == 3
    with pytest.raises(ValueError, match=r"load windows=range\(2\)"):
        p1.replay(_seq(0.015), complex_signal=True)                        # reaches into window 1


def test_window_prefix_refuses_a_non_prefix_selection():
    assert pub.window_prefix(range(3)) == 3
    assert pub.window_prefix([0, 1]) == 2
    for bad in (range(1, 3), [1, 2], [0, 2], [], range(0)):
        with pytest.raises(ValueError, match="prefix"):
            pub.window_prefix(bad)


def test_replay_pack_load_takes_windows_through_a_hub_uri_only(pack, served, tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hfcache"))
    url, _ = served
    p1 = read_rpk(pub.fetch_windows(URI, range(1), url=url, revision="rev0"))
    reloaded = ReplayPack.load(URI, windows=range(1), url=url, revision="rev0")
    assert reloaded.windows_present == 1
    npt.assert_array_equal(reloaded.replay(_seq(0.01), complex_signal=True), p1.replay(_seq(0.01), complex_signal=True))
    with pytest.raises(ValueError, match="hf://"):
        ReplayPack.load(str(tmp_path / "x.rpk"), windows=range(1))


def test_pack_substrate_passes_windows_through_to_the_load(monkeypatch):
    from dmipy_sim.phantom import PackSubstrate
    seen = {}
    fake = ReplayPack({"pos_x": np.zeros((2, 4), np.float32)},
                      {"id": "o/p", "windows_present": 2,
                       "walk_params": {"n_t": 3, "segments": {"n": 2, "n_t": 3, "T": 1e-3, "walks": []}}})

    def fake_load(cls, uri, *, windows=None, **kw):
        seen["uri"], seen["windows"] = uri, windows
        return fake

    monkeypatch.setattr(ReplayPack, "load", classmethod(fake_load))
    sub = PackSubstrate("hf://o/windows/packs/x.rpk", m0=1.0, windows=range(2))
    assert sub.pack is fake and seen["uri"] == "hf://o/windows/packs/x.rpk" and seen["windows"] == range(2)
    m = sub.to_meta()
    assert m["windows"] == 2
    back = PackSubstrate.from_meta(dict(m, id="o/windows/packs/x.rpk"), pack="hf://o/windows/packs/x.rpk")
    assert back._windows == 2
    with pytest.raises(ValueError, match="windows="):
        PackSubstrate("/local/path.rpk", m0=1.0, windows=range(1))


# ------------------------------- the real Hub (slow, skipped without a login) -------------------------------
FIXTURE_REPO = "SubstrateCommons/parity-fixtures"
FIXTURE_PATH = "test-fixtures/windows-fixture.rpk"       # outside packs/: ours, not one of the curated reference packs


@pytest.mark.slow
def test_a_windowed_read_from_the_real_hub():
    """Uploads the tiny fixture pack to the public dataset once -- idempotent, skipped when a file of the same
    sha256 is already there -- and reads its first window back by HTTP byte range, through the public entry
    point, with no ``url=``/``revision=`` override: the real Hub resolver end to end."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.whoami()
    except Exception as e:
        pytest.skip(f"no hub login / no network: {e}")
    from dmipy_sim.fill.hub import sha256_of
    pack = _fixture_pack()
    with tempfile.TemporaryDirectory(prefix="dmipy-windows-fixture-") as tmp:
        local = os.path.join(tmp, "windows-fixture.rpk")
        pack.save(local)
        sha = sha256_of(local)
        held = None
        try:
            info = api.get_paths_info(FIXTURE_REPO, [FIXTURE_PATH], repo_type="dataset")
            if info and getattr(info[0], "lfs", None) is not None:
                held = info[0].lfs.sha256
        except Exception:
            held = None
        if held != sha:
            api.upload_file(path_or_fileobj=local, path_in_repo=FIXTURE_PATH, repo_id=FIXTURE_REPO,
                            repo_type="dataset", commit_message="add the windows=range(k) hub-read test fixture (dmipy-sim#525)")
    uri = pub.uri_of(FIXTURE_REPO, FIXTURE_PATH)
    loaded = ReplayPack.load(uri, windows=range(1))
    assert loaded.windows_present == 1 and loaded.n_segments == 3
    assert loaded.n_walkers == pack.n_walkers
    npt.assert_array_equal(loaded.arrays["pos_x"], pack.arrays["pos_x"])
    npt.assert_array_equal(loaded.replay(_seq(0.01), complex_signal=True), pack.replay(_seq(0.01), complex_signal=True))
