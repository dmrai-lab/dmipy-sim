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
import subprocess
import sys
import tempfile

import numpy as np

from ..fill.hub import is_not_found, sha256_of

SCHEMA = "substratecommons/1"
log = logging.getLogger("dmipy_sim.replay")
MANIFEST = "manifest.json"
README = "README.md"
CONTRACT = ("id", "license", "citation", "fidelity")     # what a published pack must carry in its header

__all__ = ["publish", "manifest_row", "render_readme",
           "header_of", "code_commit"]






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


def _built_commit_of(meta):
    """The commit that BUILT this pack -- sealed in its own header, never corrected after the fact (dmipy-sim#499):
    ``provenance.code.commit`` (every pack built after dmipy-sim#541, content-derived, so it stays in the file),
    else ``provenance.run.code.commit`` or the pack step's (a pack built before #541, read as it is -- its run
    record, including its code, was in the file).

    This is a different fact from :func:`code_commit`, the commit of the code PUBLISHING the pack: a branch can be
    squashed or rebased after the walk ran, so the commit that built the bytes and the commit a consumer can fetch
    today are not always the same one, and a manifest row names each (``built_commit`` / ``published_commit``)."""
    c = ((meta.get("provenance") or {}).get("code") or {}).get("commit")
    if c:
        return c
    run = (meta.get("provenance") or {}).get("run") or {}
    for node in (run, run.get("pack") or {}, run.get("walk") or {}):
        c = (node.get("code") or {}).get("commit")
        if c:
            return c
    return None


def code_commit(*, require_clean=True):
    """The dmipy-sim commit the code PUBLISHING a pack is at right now -- what a manifest row's
    ``published_commit`` names (dmipy-sim#499), read fresh at call time and never copied from a pack's own
    provenance (:func:`_built_commit_of`): a pack built on a feature branch and published after that branch was
    squashed into main is published AT the squash commit, not at the extinct branch commit its bytes were built
    at, and calling this at publish time is what makes that true without a later correction.

    A dirty tree is refused (when ``require_clean``) because the commit does not describe the code that ran, and
    an unpushed HEAD is refused because a manifest row that names a commit nobody can fetch is not provenance.
    """
    from .. import __file__ as pkg
    root = os.path.dirname(os.path.dirname(os.path.abspath(pkg)))

    def git(*args):
        r = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=60)
        if r.returncode:
            raise ValueError(f"publish: git {' '.join(args)} failed in {root}: {r.stderr.strip()}")
        return r.stdout.strip()

    head = git("rev-parse", "HEAD")
    if not require_clean:
        return head
    dirty = git("status", "--porcelain")
    if dirty:
        raise ValueError(f"publish: the dmipy-sim tree at {root} is dirty, so {head[:8]} does not describe "
                         f"the code that ran:\n{dirty}")
    if not git("branch", "-r", "--contains", head):
        raise ValueError(f"publish: {head[:8]} is on no remote branch; a manifest row that names a commit "
                         f"nobody can fetch is not provenance -- push the branch first")
    return head


def _commit_cell(built, published):
    """One markdown cell naming a pack's built and published commits (dmipy-sim#499): a single, unlabelled hash
    never appears twice meaning two different things. The two are the same fact when a pack is published right
    after it is built, so the common case prints one short hash; they differ once a branch is squashed or
    rebased after the walk ran, and then each is named."""
    b, p = built or None, published or None
    if not b and not p:
        return "—"
    if b and p and b == p:
        return b[:8]
    parts = []
    if b:
        parts.append(f"built {b[:8]}")
    if p:
        parts.append(f"published {p[:8]}")
    return ", ".join(parts)


def _substrate_summary(sub):
    """What the card says about a substrate: id, box, boundary, the pools with their D and water fraction."""
    if not sub:
        return None
    dom = sub.get("domain") or {}
    return dict(id=sub.get("id"), box_min=dom.get("box_min"), box_max=dom.get("box_max"), boundary=dom.get("boundary"),
                pools=[dict(name=p.get("name"), D=p.get("D"), water_fraction=p.get("water_fraction"))
                       for p in (sub.get("pools") or [])])


def manifest_row(meta, path, sha256, nbytes, *, local=None, published_commit=None):
    """One manifest row for the pack whose header is ``meta``, stored at ``path`` with these bytes.

    ``local`` -- the file itself -- lets a pack stored in more than one window (RPK.md 4.3) carry
    ``window_bytes``, the bytes of ONE window (the shared tensors -- weights, a static label, the field grid,
    the voxel tables -- counted once, not once per window): what a ``windows=range(k)`` read of it costs, so a
    consumer can plan the read before it fetches anything (:func:`fetch_windows`).

    A row names two commits, never one unlabelled ``commit`` (dmipy-sim#499): ``built_commit``, read from the
    pack's own sealed provenance (:func:`_built_commit_of`) and never corrected after the fact, and
    ``published_commit``, the commit of the code that wrote THIS row (the caller's, from :func:`code_commit` at
    publish time -- :func:`_publish_file` fills it in when the caller does not pass one). A re-publish of the
    same bytes from a later commit updates ``published_commit`` without touching the sealed ``built_commit``."""
    wp = meta.get("walk_params") or {}
    cx = meta.get("compression") or {}
    fid = meta.get("fidelity") or {}
    row = dict(id=meta.get("id"), path=path, sha256=sha256, bytes=int(nbytes),
               n_walkers=wp.get("n_walkers"), T_max=wp.get("T_max"),
               block={k: (wp.get("block") or {}).get(k) for k in ("h", "m")},
               K=cx.get("K"), method=cx.get("method"), temporal_bandwidth_hz=cx.get("temporal_bandwidth_hz"),
               channels=["positions", *sorted(cx.get("channels") or {})],
               floor_max=fid.get("floor_max"), err_max=fid.get("err_max"), within_2x_floor=fid.get("within_2x_floor"),
               built_commit=_built_commit_of(meta), published_commit=published_commit,
               license=meta.get("license"), citation=meta.get("citation"),
               substrate=(meta.get("substrate") or {}).get("id"))
    if meta.get("pack_license"):                             # the terms the FILE is under; `license` is the source's
        row["pack_license"] = meta["pack_license"]
        if meta.get("pack_license_url"):
            row["pack_license_url"] = meta["pack_license_url"]
    if wp.get("windows"):
        row["windows"] = wp["windows"]
        n = int(wp["windows"].get("n") or 1)
        if n > 1 and local is not None:
            sb = _window_bytes(local, n)
            if sb is not None:
                row["window_bytes"] = sb
    return row


def _local_header(local):
    """A ``.rpk`` file's safetensors header (tensor name -> ``{dtype, shape, data_offsets}`` plus
    ``__metadata__``), read from its first bytes alone."""
    with open(local, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n))


def _window_bytes(local, n):
    """The bytes of one window of a windowed pack's file (the shared tensors counted once, not once per
    window), or ``None`` when the file carries no per-window tensors (``sN/`` keys) to measure."""
    header = _local_header(local)
    names = [k for k in header if k != "__metadata__"]
    per_window = {k[3:] for k in names if k.startswith("s1/")}
    if not per_window:
        return None
    def size(k):
        a, b = header[k]["data_offsets"]
        return b - a
    shared_bytes = sum(size(k) for k in names if "/" not in k and k not in per_window)
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
    seg = row.get("windows")
    if seg and int(seg.get("n") or 1) > 1:                       # a walk in several windows: how many of what length
        s += f" ({seg.get('n')} × {_fmt(None if seg.get('T') is None else seg['T'] * 1e3, 'g')} ms"
        if row.get("window_bytes"):                             # what one windows=range(k) read costs
            s += f", {_size(row['window_bytes'])}/window"
        s += ")"
    return s


def _hub_license_tags(expr):
    """The Hub's licence tags of an SPDX expression: each identifier the Hub knows, lower-cased, and ``other`` for
    one it does not (a commercial option, ``LicenseRef-...``)."""
    tags = []
    for part in str(expr).replace("(", " ").replace(")", " ").split():
        if part.upper() in ("OR", "AND", "WITH"):
            continue
        tags.append("other" if part.lower().startswith("licenseref-") else part.lower())
    return list(dict.fromkeys(tags))


def render_readme(manifest, repo):
    """The dataset card from its manifest: HF front matter, the substrate, one table row per pack, and the terms
    the packs are distributed under when they have terms of their own (``pack_license``, with ``license``, the
    source substrate's, stated beside it)."""
    packs = manifest.get("packs") or []
    pack_terms = sorted({r["pack_license"] for r in packs if r.get("pack_license")})
    licenses = sorted({r["license"].lower() for r in packs if r.get("license")})
    tags = [t for e in pack_terms for t in _hub_license_tags(e)] if pack_terms else licenses
    tags = list(dict.fromkeys(tags))
    lines = ["---"]
    if len(tags) == 1:
        lines.append(f"license: {tags[0]}")
    elif tags:
        lines.append("license:"); lines += [f"  - {l}" for l in tags]
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
                     f"{_commit_cell(r.get('built_commit'), r.get('published_commit'))} | `{r.get('path')}` |")
    cites = sorted({r["citation"] for r in packs if r.get("citation")})
    if cites:
        lines += ["", "## Citation", ""] + [f"* {c}" for c in cites]
    if pack_terms:
        urls = sorted({r["pack_license_url"] for r in packs if r.get("pack_license_url")})
        lines += ["", "## Licence", "",
                  "The packs are distributed under " + " and ".join(f"`{e}`" for e in pack_terms)
                  + (f" (the full terms: {', '.join(urls)})" if urls else "") + ". The source geometry of each pack keeps its own "
                  "licence, recorded in the pack's `license` field" + (f": {', '.join(licenses)}" if licenses else "") + "; the format never relicenses a source."]
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


def publish(pack_or_path, repo, *, path=None, hub=None, message=None, published_commit=None):
    """Put a pack in the dataset ``repo`` (``owner/name``) and return its URI ``hf://owner/name/<path>``.

    ``pack_or_path`` is a ``.rpk`` file or a :class:`~dmipy_sim.replay.ReplayPack` (written to a temporary
    file first: the bytes uploaded are the bytes hashed). ``path`` defaults to ``packs/<id>.rpk`` with the id's
    slashes replaced. The file, the manifest row (replacing the row at the same path) and the regenerated card go
    up in ONE commit through ``hub`` (default :class:`~dmipy_sim.fill.hub.Hub`); afterwards the sha256 the hub
    holds is read back and must equal the manifest's.

    ``published_commit`` names the manifest row's own commit (dmipy-sim#499): the commit of the code THIS call is
    running as, as distinct from the pack's sealed ``built_commit``. A caller that already knows it (the
    reference-pack protocol's stricter, clean-and-pushed :func:`~dmipy_sim.replay.reference.code_commit`) passes
    it; left out, this reads one itself with :func:`code_commit`'s lenient form (no clean-tree or pushed-branch
    requirement -- a plain ``publish()`` call is not that protocol)."""
    from .replay import ReplayPack
    if isinstance(pack_or_path, ReplayPack):
        _check_contract(pack_or_path.meta)
        source = pack_or_path.source
        with tempfile.TemporaryDirectory(prefix="dmipy-publish-") as tmp:
            local = os.path.join(tmp, "pack.rpk")
            pack_or_path.save(local)
            pack_or_path.source = source                          # the temporary file is not where the pack lives
            return _publish_file(local, pack_or_path.meta, repo, path=path, hub=hub, message=message,
                                 published_commit=published_commit)
    local = os.fspath(pack_or_path)
    return _publish_file(local, header_of(local), repo, path=path, hub=hub, message=message,
                         published_commit=published_commit)


def _publish_file(local, meta, repo, *, path, hub, message, published_commit=None):
    _check_contract(meta)
    if hub is None:
        from ..fill.hub import Hub
        hub = Hub(repo)
    path = path or f"packs/{str(meta['id']).replace('/', '-')}.rpk"
    if published_commit is None:
        try:
            published_commit = code_commit(require_clean=False)
        except Exception as e:                  # an installed, non-git copy of dmipy-sim still publishes a pack
            log.warning("publish %s: could not read the running code's commit for published_commit (%s)", path, e)
            published_commit = None
    sha, nbytes = sha256_of(local), os.path.getsize(local)
    row = manifest_row(meta, path, sha, nbytes, local=local, published_commit=published_commit)
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
