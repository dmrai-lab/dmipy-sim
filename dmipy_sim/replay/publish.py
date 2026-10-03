"""The one path from a pack on disk to a pack on a hub, and back: :func:`publish` puts a ``.rpk`` file in a
dataset repository together with the dataset's ``manifest.json`` and its ``README.md`` (the card), in one
commit, and verifies what the hub holds; :func:`fetch` brings ``hf://owner/name/path.rpk`` into the local cache
and checks it against that manifest.

The hub is whatever the caller names: ``repo`` is ``owner/name`` and the commits go through a
:class:`~dmipy_sim.fill.hub.Hub`-like object (``commit``, ``exists``, ``get_live``), the real one by default,
a :class:`~dmipy_sim.fill.hub.FakeHub` in the tests. The pack's own header is the record: a pack carries its
``id``, ``license``, ``citation`` and ``fidelity`` and one without any of them is refused, since those are the
contract a consumer reads. The manifest (``{"schema": "substratecommons/1", "substrate": ..., "packs": [row, ...]}``)
holds one row per published file, keyed on the path, and the card is rendered from the manifest alone, never
edited by hand.

::

    python -m dmipy_sim.replay.publish PACK.rpk --repo OWNER/NAME [--path packs/x.rpk] [--message ...]
"""
import argparse
import json
import logging
import os
import sys
import tempfile

import numpy as np

from ..fill.hub import is_not_found, sha256_of

SCHEMA = "substratecommons/1"
log = logging.getLogger("dmipy_sim.replay")
MANIFEST = "manifest.json"
README = "README.md"
CONTRACT = ("id", "license", "citation", "fidelity")     # what a published pack must carry in its header

__all__ = ["publish", "fetch", "fetch_windows", "parse_uri", "is_hub_uri", "manifest_row", "render_readme",
           "header_of", "window_prefix"]


# ------------------------------- URIs -------------------------------
def is_hub_uri(x):
    """Whether ``x`` names a file on the hub (``hf://owner/name/path``)."""
    return isinstance(x, str) and x.startswith("hf://")


def parse_uri(uri):
    """``hf://owner/name/path/to/file`` -> ``("owner/name", "path/to/file")``; anything else is refused."""
    if not is_hub_uri(uri):
        raise ValueError(f"not a hub URI: {uri!r} (expected hf://owner/name/path/to/file.rpk)")
    parts = uri[len("hf://"):].split("/", 2)
    if len(parts) < 3 or not all(parts):
        raise ValueError(f"a hub URI names a repository and a file, hf://owner/name/path/to/file.rpk: {uri!r}")
    owner, name, path = parts
    return f"{owner}/{name}", path


def uri_of(repo, path):
    return f"hf://{repo}/{path}"


# ------------------------------- the pack's header -------------------------------
def header_of(path):
    """The ``rpk`` JSON header of a ``.rpk`` file, read without its arrays."""
    from safetensors import safe_open
    with safe_open(str(path), framework="numpy") as f:
        hdr = f.metadata() or {}
    return json.loads(hdr.get("rpk") or hdr.get("json") or "{}")


def _check_contract(meta):
    missing = [k for k in CONTRACT if not meta.get(k)]
    if missing:
        raise ValueError(f"a published pack carries its {', '.join(CONTRACT)}; this one has no "
                         f"{', '.join(missing)} (build it with build_replay_pack(..., id=, license=, citation=))")


def _commit_of(meta):
    """The code commit recorded by the producer: ``provenance.code.commit`` (every pack built after dmipy-sim#541,
    content-derived, so it stays in the file), else ``provenance.run.code.commit`` or the pack step's (a pack
    built before #541, read as it is -- its run record, including its code, was in the file)."""
    c = ((meta.get("provenance") or {}).get("code") or {}).get("commit")
    if c:
        return c
    run = (meta.get("provenance") or {}).get("run") or {}
    for node in (run, run.get("pack") or {}, run.get("walk") or {}):
        c = (node.get("code") or {}).get("commit")
        if c:
            return c
    return None


def _substrate_summary(sub):
    """What the card says about a substrate: id, box, boundary, the pools with their D and water fraction."""
    if not sub:
        return None
    dom = sub.get("domain") or {}
    return dict(id=sub.get("id"), box_min=dom.get("box_min"), box_max=dom.get("box_max"), boundary=dom.get("boundary"),
                pools=[dict(name=p.get("name"), D=p.get("D"), water_fraction=p.get("water_fraction"))
                       for p in (sub.get("pools") or [])])


def manifest_row(meta, path, sha256, nbytes, *, local=None):
    """One manifest row for the pack whose header is ``meta``, stored at ``path`` with these bytes.

    ``local`` -- the file itself -- lets a pack stored in more than one window (RPK.md 4.3) carry
    ``segment_bytes``, the bytes of ONE window (the shared tensors -- weights, a static label, the field grid,
    the voxel tables -- counted once, not once per window): what a ``windows=range(k)`` read of it costs, so a
    consumer can plan the read before it fetches anything (:func:`fetch_windows`)."""
    wp = meta.get("walk_params") or {}
    cx = meta.get("compression") or {}
    fid = meta.get("fidelity") or {}
    row = dict(id=meta.get("id"), path=path, sha256=sha256, bytes=int(nbytes),
               n_walkers=wp.get("n_walkers"), T_max=wp.get("T_max"), dt_traj=wp.get("dt_traj"),
               K=cx.get("K"), method=cx.get("method"), temporal_bandwidth_hz=cx.get("temporal_bandwidth_hz"),
               channels=["positions", *sorted(cx.get("channels") or {})],
               floor_max=fid.get("floor_max"), err_max=fid.get("err_max"), within_2x_floor=fid.get("within_2x_floor"),
               commit=_commit_of(meta), license=meta.get("license"), citation=meta.get("citation"),
               substrate=(meta.get("substrate") or {}).get("id"))
    if wp.get("segments"):
        row["segments"] = wp["segments"]
        n = int(wp["segments"].get("n") or 1)
        if n > 1 and local is not None:
            sb = _segment_bytes(local, n)
            if sb is not None:
                row["segment_bytes"] = sb
    return row


def _local_header(local):
    """A ``.rpk`` file's safetensors header (tensor name -> ``{dtype, shape, data_offsets}`` plus
    ``__metadata__``), read from its first bytes alone."""
    with open(local, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n))


def _segment_bytes(local, n):
    """The bytes of one window of a segmented pack's file (the shared tensors counted once, not once per
    window), or ``None`` when the file carries no per-window tensors (``sN/`` keys) to measure."""
    header = _local_header(local)
    names = [k for k in header if k != "__metadata__"]
    per_segment = {k[3:] for k in names if k.startswith("s1/")}
    if not per_segment:
        return None
    def size(k):
        a, b = header[k]["data_offsets"]
        return b - a
    shared_bytes = sum(size(k) for k in names if "/" not in k and k not in per_segment)
    data_total = sum(size(k) for k in names)
    return int(round((data_total - shared_bytes) / n))


# ------------------------------- the card -------------------------------
def _fmt(x, spec):
    return "—" if x is None else format(x, spec)


def _size(nbytes):
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if nbytes >= div:
            return f"{nbytes / div:.1f} {unit}"
    return f"{nbytes} B"


def _duration(row):
    T = row.get("T_max")
    s = "—" if T is None else f"{T * 1e3:g} ms"
    seg = row.get("segments")
    if seg and int(seg.get("n") or 1) > 1:                       # a walk in several windows: how many of what length
        s += f" ({seg.get('n')} × {_fmt(None if seg.get('T') is None else seg['T'] * 1e3, 'g')} ms"
        if row.get("segment_bytes"):                             # what one windows=range(k) read costs
            s += f", {_size(row['segment_bytes'])}/window"
        s += ")"
    return s


def render_readme(manifest, repo):
    """The dataset card from its manifest: HF front matter, the substrate, one table row per pack."""
    packs = manifest.get("packs") or []
    licenses = sorted({r["license"].lower() for r in packs if r.get("license")})
    lines = ["---"]
    if len(licenses) == 1:
        lines.append(f"license: {licenses[0]}")
    elif licenses:
        lines.append("license:"); lines += [f"  - {l}" for l in licenses]
    lines += [f"pretty_name: {repo.split('/')[-1]}", "---", "", f"# {repo}", "",
              "Replay packs (`.rpk`) built with `dmipy_sim`: one converged Monte-Carlo walk each, stored so any acquisition "
              "can be replayed on it. Load one with `ReplayPack.load(\"hf://" + repo + "/<path>\")`; the manifest "
              "(`manifest.json`) holds the sha256 every load is checked against. This file is rendered from the "
              "manifest by `dmipy_sim.replay.publish`.", ""]
    sub = manifest.get("substrate")
    if sub:
        lines += ["## Substrate", "", f"`{sub.get('id')}`", ""]
        if sub.get("box_min") and sub.get("box_max"):
            side = [f"{(hi - lo) * 1e6:g}" for lo, hi in zip(sub["box_min"], sub["box_max"])]
            lines.append(f"* box: {' × '.join(side)} µm")
        if sub.get("boundary"):
            lines.append(f"* boundary: {', '.join(map(str, sub['boundary']))}")
        if sub.get("pools"):
            lines += ["", "| pool | D (m²/s) | water fraction |", "|---|---|---|"]
            lines += [f"| {p.get('name')} | {_fmt(p.get('D'), 'g')} | {_fmt(p.get('water_fraction'), 'g')} |" for p in sub["pools"]]
        lines.append("")
    lines += ["## Packs", "", "| id | duration | walkers | bands | channels | floor | size | commit | path |", "|---|---|---|---|---|---|---|---|---|"]
    for r in packs:
        lines.append(f"| `{r.get('id')}` | {_duration(r)} | {_fmt(r.get('n_walkers'), 'd')} | {_fmt(r.get('K'), 'd')} | "
                     f"{', '.join(r.get('channels') or [])} | {_fmt(r.get('floor_max'), '.3g')} | {_size(r.get('bytes') or 0)} | "
                     f"{(r.get('commit') or '—')[:8]} | `{r.get('path')}` |")
    cites = sorted({r["citation"] for r in packs if r.get("citation")})
    if cites:
        lines += ["", "## Citation", ""] + [f"* {c}" for c in cites]
    return "\n".join(lines) + "\n"


# ------------------------------- publish -------------------------------
def _load_manifest(hub, revision=None):
    """The dataset's manifest as it is now, or as it was at ``revision`` (a head a commit will declare as its parent)."""
    if hub.exists(MANIFEST):
        with open(hub.get_live(MANIFEST) if revision is None else hub.get_at(MANIFEST, revision)) as f:
            return json.load(f)
    return {"schema": SCHEMA, "packs": []}


MANIFEST_TRIES = 8           # a commit refused because another publisher moved the head is prepared again from the new head


def _hub_sha256(hub, path):
    """The sha256 of the bytes the hub holds at ``path``: what the hub reports for an LFS file, else the file read back."""
    api = getattr(hub, "api", None)
    if api is not None:
        info = api.get_paths_info(hub.repo, [path], repo_type="dataset")
        if info and getattr(info[0], "lfs", None) is not None:
            return info[0].lfs.sha256
    return sha256_of(hub.get_live(path))


def publish(pack_or_path, repo, *, path=None, hub=None, message=None):
    """Put a pack in the dataset ``repo`` (``owner/name``) and return its URI ``hf://owner/name/<path>``.

    ``pack_or_path`` is a ``.rpk`` file or a :class:`~dmipy_sim.replay.ReplayPack` (written to a temporary
    file first: the bytes uploaded are the bytes hashed). ``path`` defaults to ``packs/<id>.rpk`` with the id's
    slashes replaced. The file, the manifest row (replacing the row at the same path) and the regenerated card go
    up in ONE commit through ``hub`` (default :class:`~dmipy_sim.fill.hub.Hub`); afterwards the sha256 the hub
    holds is read back and must equal the manifest's."""
    from .replay import ReplayPack
    if isinstance(pack_or_path, ReplayPack):
        _check_contract(pack_or_path.meta)
        source = pack_or_path.source
        with tempfile.TemporaryDirectory(prefix="dmipy-publish-") as tmp:
            local = os.path.join(tmp, "pack.rpk")
            pack_or_path.save(local)
            pack_or_path.source = source                          # the temporary file is not where the pack lives
            return _publish_file(local, pack_or_path.meta, repo, path=path, hub=hub, message=message)
    local = os.fspath(pack_or_path)
    return _publish_file(local, header_of(local), repo, path=path, hub=hub, message=message)


def _publish_file(local, meta, repo, *, path, hub, message):
    _check_contract(meta)
    if hub is None:
        from ..fill.hub import Hub
        hub = Hub(repo)
    path = path or f"packs/{str(meta['id']).replace('/', '-')}.rpk"
    sha, nbytes = sha256_of(local), os.path.getsize(local)
    row = manifest_row(meta, path, sha, nbytes, local=local)
    from ..fill.hub import StaleParent
    for attempt in range(MANIFEST_TRIES):
        # the manifest is read at the head and the commit declares that head as its parent: two publishers of one
        # dataset (a build and a sync loop) each read, add a row and write, and without the parent the later
        # write silently dropped the earlier one's row (nine packs of the canonical pores had none)
        head = hub.head()
        manifest = _load_manifest(hub, head)
        manifest["schema"] = manifest.get("schema") or SCHEMA
        manifest["packs"] = [r for r in manifest.get("packs") or [] if r.get("path") != path] + [row]
        if not manifest.get("substrate"):
            manifest["substrate"] = _substrate_summary(meta.get("substrate"))
        readme = render_readme(manifest, repo)
        try:
            hub.commit({path: local, MANIFEST: json.dumps(manifest, indent=1).encode(), README: readme.encode()}, [],
                       message or f"publish {meta['id']} ({path})", parent=head)
            break
        except StaleParent:
            log.info("publish %s: the manifest moved past %s; prepared again (%d/%d)", path, head[:8], attempt + 2, MANIFEST_TRIES)
    else:
        raise RuntimeError(f"{uri_of(repo, path)}: the manifest moved under every one of {MANIFEST_TRIES} attempts")
    held = _hub_sha256(hub, path)
    if held != sha:
        raise RuntimeError(f"{uri_of(repo, path)}: the hub holds sha256 {held}, the manifest says {sha}")
    return uri_of(repo, path)


# ------------------------------- load -------------------------------
def fetch(uri):
    """``hf://owner/name/path.rpk`` -> the file's local cache path, its sha256 checked against the dataset's
    manifest when the manifest has a row for it (a mismatch raises). Refuses a URI that is not a ``.rpk``."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(f"loading {uri} needs huggingface_hub: pip install \"dmipy-sim[bank]\"") from e
    repo, path = parse_uri(uri)
    if not path.endswith(".rpk"):
        raise ValueError(f"{uri}: ReplayPack.load takes a pack (.rpk); the manifest and the card are not packs")
    local = hf_hub_download(repo, path, repo_type="dataset")
    try:
        with open(hf_hub_download(repo, MANIFEST, repo_type="dataset")) as f:
            manifest = json.load(f)
    except Exception as e:                                        # no manifest: nothing to check against
        if not is_not_found(e):
            raise
        return local
    rows = [r for r in manifest.get("packs") or [] if r.get("path") == path]
    if rows and rows[0].get("sha256"):
        got = sha256_of(local)
        if got != rows[0]["sha256"]:
            raise ValueError(f"{uri}: the file's sha256 is {got}, the manifest says {rows[0]['sha256']}; refusing to load it")
    return local


# ------------------------------- windowed load (RPK.md 4.3) -------------------------------
_ST_DTYPES = {"BOOL": "bool", "U8": "uint8", "I8": "int8", "I16": "int16", "U16": "uint16",
              "I32": "int32", "U32": "uint32", "I64": "int64", "U64": "uint64",
              "F16": "float16", "F32": "float32", "F64": "float64"}


def window_prefix(windows):
    """``windows`` (a ``range`` or a sequence of window indices) as the plain count ``k`` it must be: a prefix
    ``0, 1, ..., k-1``. The walk is continuous (RPK.md 4.3), so a pack cannot serve window 2 without window 1,
    and any other selection is refused by name rather than silently read as a prefix."""
    seq = list(windows)
    if not seq or seq != list(range(len(seq))):
        raise ValueError(f"windows must be a prefix range(k) (0, 1, ..., k-1), the walk being continuous; got {windows!r}")
    return len(seq)


def _auth_headers():
    """``Authorization: Bearer <token>`` for a logged-in hub read, or no header for an anonymous one."""
    try:
        from huggingface_hub import get_token
        tok = get_token()
    except Exception:
        tok = None
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _range_get(url, start, end, session, headers):
    """Bytes ``[start, end]`` (inclusive) of ``url`` via one HTTP Range request; a server that ignores the header
    and answers 200 with the whole body is sliced to the same bytes rather than trusted as the range."""
    r = session.get(url, headers={**headers, "Range": f"bytes={start}-{end}"}, timeout=60)
    r.raise_for_status()
    body = r.content
    return body if r.status_code == 206 else body[start:end + 1]


def _fetch_header(url, session, headers):
    """A pack's safetensors header at ``url``, from two Range requests: the 8-byte little-endian length, then
    that many bytes of JSON (tensor name -> ``{dtype, shape, data_offsets}``, plus ``__metadata__``)."""
    n = int.from_bytes(_range_get(url, 0, 7, session, headers), "little")
    header = json.loads(_range_get(url, 8, 8 + n - 1, session, headers))
    return n, header


def _coalesce(ranges):
    """Sorted ``(start, end)`` inclusive byte ranges with every adjacent or overlapping pair merged into one."""
    out = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _window_tensor_names(names, i, per_segment):
    """The tensor names of window ``i``: segment 0's are the unprefixed per-segment names, segment ``i > 0``'s
    its ``s{i}/`` ones. The shared tensors (weights, a static label, the field grid, the voxel tables) are not a
    window's -- they are fetched once, regardless of ``i``."""
    if i == 0:
        return sorted(per_segment)
    p = f"s{int(i)}/"
    return [n for n in names if n.startswith(p)]


def _read_cached_windows(local_path):
    """``(k, arrays)`` already written at ``local_path`` by an earlier windowed fetch -- ``(None, None)`` when
    nothing is cached there yet."""
    if not os.path.isfile(local_path):
        return None, None
    from .replay import read_rpk
    pack = read_rpk(local_path)
    k = pack.meta.get("windows_present")
    return (None if k is None else int(k)), pack.arrays


def _windows_cache_path(repo, path, revision):
    """Where a windowed read of ``hf://repo/path`` at ``revision`` is cached locally: a file beside the hub's
    own cache, keyed on the repository, the file and the pinned revision (not on ``k`` -- a later, larger
    ``k`` rewrites this same file rather than starting a new one, RPK.md 8.3)."""
    from huggingface_hub.constants import HF_HUB_CACHE
    owner, name = repo.split("/", 1)
    d = os.path.join(HF_HUB_CACHE, f"datasets--{owner}--{name}--windows", revision)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, path.replace("/", "__"))


def _fetch_windows_to(url, k, local_path, *, revision, session=None):
    """The implementation behind :func:`fetch_windows`, taking a plain resolvable ``url`` rather than resolving
    one itself -- a Hub resolve URL in production, a local test server's in :mod:`tests`, so the byte-range
    selection runs the same code path with no Hub involved."""
    import requests
    cached_k, cached_arrays = _read_cached_windows(local_path)
    if cached_k is not None and cached_k >= k:
        return local_path
    sess = session or requests.Session()
    headers = _auth_headers()
    header_len, header = _fetch_header(url, sess, headers)
    data_base = 8 + header_len
    names = [n for n in header if n != "__metadata__"]
    meta = json.loads((header.get("__metadata__") or {}).get("rpk") or "{}")
    n_total = int(((meta.get("walk_params") or {}).get("segments") or {}).get("n") or 1)
    if k > n_total:
        raise ValueError(f"{url}: this pack stores {n_total} window(s); windows=range({k}) asks for more than it has")
    per_segment = {n[3:] for n in names if n.startswith("s1/")}
    want_from = 0 if cached_k is None else cached_k
    wanted = [n for n in names if "/" not in n and n not in per_segment] if want_from == 0 else []
    for i in range(want_from, k):
        wanted += _window_tensor_names(names, i, per_segment)
    byte_ranges = {n: (header[n]["data_offsets"][0] + data_base, header[n]["data_offsets"][1] + data_base - 1)
                   for n in wanted}
    merged = _coalesce(byte_ranges.values())
    fetched = {r: _range_get(url, r[0], r[1], sess, headers) for r in merged}
    new_arrays = {}
    for n, (s, e) in byte_ranges.items():
        ms, me = next(r for r in merged if r[0] <= s and e <= r[1])
        blob = fetched[(ms, me)][s - ms: e - ms + 1]
        dtype = header[n]["dtype"]
        np_dtype = _ST_DTYPES.get(dtype)
        if np_dtype is None:
            raise NotImplementedError(f"{n}: safetensors dtype {dtype!r} has no numpy mapping for a windowed read")
        new_arrays[n] = np.frombuffer(blob, dtype=np_dtype).reshape(header[n]["shape"])
    arrays = dict(cached_arrays or {}, **new_arrays)
    bytes_fetched = sum(e - s + 1 for s, e in merged)
    out_meta = dict(meta, windows_present=k)
    prov = dict(out_meta.get("provenance") or {})
    # a sha256 of the whole pack cannot be checked against a part of it: what is checkable is recorded instead --
    # the pinned revision this read came from and the bytes it fetched to get here
    prov["hub_window_read"] = dict(revision=revision, windows_present=k, bytes_fetched=bytes_fetched)
    out_meta["provenance"] = prov
    from .replay import write_rpk
    write_rpk(local_path, arrays, out_meta)
    return local_path


def fetch_windows(uri, windows, *, url=None, revision=None):
    """``hf://owner/name/path.rpk``'s leading windows (RPK.md 4.3) -> the local cache path of a ``.rpk`` holding
    just them. ``windows`` is a prefix ``range(k)`` of window indices (:func:`window_prefix`).

    The pack's header is fetched with one HTTP Range request at the dataset's pinned revision; the shared
    tensors and windows ``0..k-1`` are selected from it and their byte ranges fetched (adjacent ranges
    coalesced into one request each), then written as a local safetensors file in the hub cache, keyed on
    ``(repo, revision, path)``. A later call for a larger ``k`` reads what that file already holds and fetches
    only the newly-needed windows, rewriting it in place.

    The whole-file sha256 check (:func:`fetch`) does not apply here -- a part of a file has no sha256 of the
    whole to be checked against -- so the written pack instead records the revision and the bytes this read
    fetched, in ``provenance.hub_window_read``.

    ``url=`` / ``revision=`` override the Hub's own URL and revision resolution: how a test drives this past a
    plain HTTP server, with no Hub call at all."""
    repo, path = parse_uri(uri)
    if not path.endswith(".rpk"):
        raise ValueError(f"{uri}: ReplayPack.load takes a pack (.rpk); the manifest and the card are not packs")
    k = window_prefix(windows)
    if revision is None:
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(repo).sha
    resolve_url = url
    if resolve_url is None:
        from huggingface_hub import hf_hub_url
        resolve_url = hf_hub_url(repo, path, repo_type="dataset", revision=revision)
    local = _windows_cache_path(repo, path, revision)
    return _fetch_windows_to(resolve_url, k, local, revision=revision)


# ------------------------------- CLI -------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m dmipy_sim.replay.publish", description="Publish a replay pack to a dataset repository.")
    p.add_argument("pack", help="the .rpk file")
    p.add_argument("--repo", required=True, help="the dataset, owner/name")
    p.add_argument("--path", default=None, help="the path in the repository (default packs/<id>.rpk)")
    p.add_argument("--message", default=None, help="the commit message")
    a = p.parse_args(argv)
    print(publish(a.pack, a.repo, path=a.path, message=a.message))


if __name__ == "__main__":                                        # pragma: no cover
    sys.exit(main())
