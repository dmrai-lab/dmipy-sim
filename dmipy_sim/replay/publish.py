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
import os
import sys
import tempfile

from ..fill.hub import is_not_found, sha256_of

SCHEMA = "substratecommons/1"
MANIFEST = "manifest.json"
README = "README.md"
CONTRACT = ("id", "license", "citation", "fidelity")     # what a published pack must carry in its header

__all__ = ["publish", "fetch", "parse_uri", "is_hub_uri", "manifest_row", "render_readme", "header_of"]


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
    """The code commit recorded by the producer: ``provenance.run.code.commit`` or the pack step's."""
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


def manifest_row(meta, path, sha256, nbytes):
    """One manifest row for the pack whose header is ``meta``, stored at ``path`` with these bytes."""
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
    return row


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
    if seg:
        s += f" ({seg.get('n')} × {_fmt(None if seg.get('T') is None else seg['T'] * 1e3, 'g')} ms)"
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
def _load_manifest(hub):
    if hub.exists(MANIFEST):
        with open(hub.get_live(MANIFEST)) as f:
            return json.load(f)
    return {"schema": SCHEMA, "packs": []}


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
    row = manifest_row(meta, path, sha, nbytes)
    manifest = _load_manifest(hub)
    manifest["schema"] = manifest.get("schema") or SCHEMA
    manifest["packs"] = [r for r in manifest.get("packs") or [] if r.get("path") != path] + [row]
    if not manifest.get("substrate"):
        manifest["substrate"] = _substrate_summary(meta.get("substrate"))
    readme = render_readme(manifest, repo)
    hub.commit({path: local, MANIFEST: json.dumps(manifest, indent=1).encode(), README: readme.encode()}, [],
               message or f"publish {meta['id']} ({path})")
    held = _hub_sha256(hub, path)
    if held != sha:
        raise RuntimeError(f"{uri_of(repo, path)}: the hub holds sha256 {held}, the manifest says {sha}")
    return uri_of(repo, path)


# ------------------------------- load -------------------------------
def fetch(uri):
    """``hf://owner/name/path.rpk`` -> the file's local cache path, its sha256 checked against the dataset's
    manifest when the manifest has a row for it (a mismatch raises). Refuses a URI that is not a ``.rpk``."""
    from huggingface_hub import hf_hub_download
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
