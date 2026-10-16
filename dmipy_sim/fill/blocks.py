"""Blocks by the excursion, their kit tarballs and the shards a block is dealt into (dmrai-lab/tessera#49 steps 5-6).

A BLOCK is the unit a node downloads (its kit, one uncompressed tar); a SHARD is the unit a node walks (a slice of the
block's counts). The plan grid is cut into cubes of ``edge_voxels`` (the excursion rule: an edge of a few times the
walk's 16 sigma), :func:`plan_blocks` writes every block's tar and the plan file ``plan.p<pass>.json`` that lists them,
and :func:`shard_counts` deals a block's per-voxel counts into ``n`` shards exactly.

    python -m dmipy_sim.fill.blocks --repo OWNER/NAME --pass 1 --edge 20 --shard-walkers 160000 --out DIR [--publish]

The plan file::

    {"repo", "variant", "pass", "scale", "block_edge_voxels", "shard_walkers", "code": {"dmipy_sim"},
     "blocks": [{"block", "i", "j", "k", "seed", "walkers": {pool: n}, "voxels", "n_shards",
                 "kit": {"path", "sha256", "bytes"}}]}

A block's tar holds the block kit's files at their kit-relative paths with ``far_spec.npz``
(:mod:`dmipy_sim.fill.farspec`) in place of ``far_full.u16``; ``kit.json`` lists every member with its sha256."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from .farspec import pack_far, save_far
from .kit import _sha256, write_block_kit

log = logging.getLogger("dmipy_sim.fill")
SEED_BASE = 1000


def pass_scale(rc, pass_n):
    """The share of every block's counts that pass ``pass_n`` of the recipe's plan fills."""
    for P in rc.passes:
        if (P.get("pass") or 1) == pass_n:
            return float(P["scale"])
    raise ValueError(f"pass {pass_n}: the plan has passes {[P.get('pass') for P in rc.passes]}")


def block_table(rc, edge_voxels, *, pass_n, shard_walkers):
    """The plan grid cut into cubes of ``edge_voxels`` voxels (the last of an axis shorter), in C order of the cube
    index: rows ``{block, i, j, k, seed, walkers, voxels, n_shards}`` with half-open voxel ranges, ``seed = 1000 +
    block``, ``walkers`` the per-pool totals of ``rc.counts(row, pass_scale=scale)`` and ``n_shards = ceil(total /
    shard_walkers)``. A cube the plan gives no walker is left out."""
    scale = pass_scale(rc, pass_n)
    shape = rc.grid().shape; e = int(edge_voxels)
    cuts = [[(a, min(a + e, n)) for a in range(0, n, e)] for n in shape]
    rows = []
    for ci in cuts[0]:
        for cj in cuts[1]:
            for ck in cuts[2]:
                row = dict(block=len(rows), i=list(ci), j=list(cj), k=list(ck), seed=SEED_BASE + len(rows),
                           voxels=int((ci[1] - ci[0]) * (cj[1] - cj[0]) * (ck[1] - ck[0])))
                want, _, _ = rc.counts(row, pass_scale=scale)
                walkers = {n: int(v.sum()) for n, v in want.items() if v.sum() > 0}
                total = sum(walkers.values())
                if total == 0:
                    continue
                row["walkers"] = walkers
                row["n_shards"] = int(math.ceil(total / int(shard_walkers)))
                rows.append(row)
    for b, row in enumerate(rows):                           # dense block numbers over the rows kept
        row["block"] = b; row["seed"] = SEED_BASE + b
    return rows


def shard_counts(want, i, n):
    """Shard ``i`` of ``n`` of a block's per-voxel counts ``want`` (``{pool: int array}``): per voxel and pool
    ``floor(want (i + 1) / n) - floor(want i / n)``, so the ``n`` shards sum exactly to ``want``. A pool with no
    walker in the shard is absent."""
    out = {}
    for pool, w in want.items():
        w = np.asarray(w, np.int64)
        s = (w * (i + 1)) // n - (w * i) // n
        if s.sum() > 0:
            out[pool] = s
    return out


def write_block_tar(rc, row, out_tar, *, wavelength_min_m=10e-6):
    """Block ``row``'s kit as one uncompressed tar at ``out_tar``: :func:`~dmipy_sim.fill.kit.write_block_kit`'s files
    with the far crop stored as ``far_spec.npz`` (:func:`~dmipy_sim.fill.farspec.pack_far`) instead of
    ``far_full.u16`` and ``kit.json``'s file table updated to match. Members are sorted, with zeroed owner and time,
    so the same block always gives the same bytes. Returns ``(path, sha256, bytes)``."""
    with tempfile.TemporaryDirectory(prefix=f"block-{row['block']:04d}-") as tmp:
        d = Path(tmp)
        write_block_kit(rc, row, d)
        kit = json.loads((d / "kit.json").read_text())
        dense = d / "far_full.u16"
        if dense.exists():
            meta = json.loads((d / "far_full.json").read_text())
            crop = np.fromfile(dense, np.uint16).view(np.float16).reshape(meta["shape"])
            save_far(d / "far_spec.npz", pack_far(crop, meta["spacing_m"], wavelength_min_m))
            dense.unlink()
            del kit["files"]["far_full.u16"]
            kit["files"]["far_spec.npz"] = _sha256(d / "far_spec.npz")
            kit["files"] = dict(sorted(kit["files"].items()))
            (d / "kit.json").write_text(json.dumps(kit, indent=1))
        out_tar = Path(out_tar); out_tar.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out_tar, "w", format=tarfile.GNU_FORMAT) as tf:
            for p in sorted(q for q in d.rglob("*") if q.is_file()):
                ti = tarfile.TarInfo(str(p.relative_to(d))); ti.size = p.stat().st_size; ti.mode = 0o644
                with open(p, "rb") as f:
                    tf.addfile(ti, f)
    return str(out_tar), _sha256(out_tar), out_tar.stat().st_size


def _code_commit():
    from .kit import _dmipy_sim_version
    try:
        from ..replay.publish import code_commit
        return code_commit(require_clean=False)
    except Exception:
        return _dmipy_sim_version()


def plan_blocks(rc, *, edge_voxels=20, shard_walkers=160000, pass_n, out_dir, wavelength_min_m=10e-6, repo=None):
    """The block plan of pass ``pass_n``: :func:`block_table`'s rows, each block's tar written to
    ``out_dir/kit/<variant>/block-NNNN.p<pass>.tar`` and its ``kit`` entry filled, and the plan file written to
    ``out_dir/plan.p<pass>.json`` (module docstring). ``repo`` is the Hub dataset the plan names as the home of
    its kits and shards (the recipe's own unless given: a rehearsal keeps its shards apart from the dataset whose
    recipe it reads). Returns the plan dict."""
    out = Path(out_dir); rows = block_table(rc, edge_voxels, pass_n=pass_n, shard_walkers=shard_walkers)
    for row in rows:
        rel = f"kit/{rc.variant}/block-{row['block']:04d}.p{pass_n}.tar"
        _, sha, nbytes = write_block_tar(rc, row, out / rel, wavelength_min_m=wavelength_min_m)
        row["kit"] = dict(path=rel, sha256=sha, bytes=int(nbytes))
        log.info("block %d: %d shards, kit %.1f MB", row["block"], row["n_shards"], nbytes / 1e6)
    plan = dict(repo=repo or getattr(rc.hub, "repo", None), variant=rc.variant, **{"pass": pass_n}, scale=pass_scale(rc, pass_n),
                block_edge_voxels=int(edge_voxels), shard_walkers=int(shard_walkers), code=dict(dmipy_sim=_code_commit()), blocks=rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"plan.p{pass_n}.json").write_text(json.dumps(plan, indent=1))
    return plan


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dmipy_sim.fill.blocks", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True); ap.add_argument("--variant", default=None)
    ap.add_argument("--pass", dest="pass_n", type=int, required=True)
    ap.add_argument("--edge", type=int, default=20); ap.add_argument("--shard-walkers", type=int, default=160000)
    ap.add_argument("--out", required=True); ap.add_argument("--wavelength-um", type=float, default=10.0)
    ap.add_argument("--publish", action="store_true", help="upload the plan file and every tar in one commit")
    ap.add_argument("--publish-to", default=None, metavar="OWNER/NAME",
                    help="the dataset the plan names and --publish writes to (default: --repo); a rehearsal reads the "
                         "recipe from the real dataset and keeps its kits and shards elsewhere")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from .hub import Hub
    from .recipe import Recipe
    hub = Hub(a.repo); rc = Recipe(hub, a.variant)
    target = a.publish_to or a.repo
    plan = plan_blocks(rc, edge_voxels=a.edge, shard_walkers=a.shard_walkers, pass_n=a.pass_n, out_dir=a.out,
                       wavelength_min_m=a.wavelength_um * 1e-6, repo=target)
    if a.publish:
        adds = {f"blocks/{rc.variant}/plan.p{a.pass_n}.json": os.path.join(a.out, f"plan.p{a.pass_n}.json")}
        adds.update({b["kit"]["path"]: os.path.join(a.out, b["kit"]["path"]) for b in plan["blocks"]})
        (hub if target == a.repo else Hub(target)).commit(adds, [], f"{rc.variant} blocks: plan p{a.pass_n}, {len(plan['blocks'])} kit tarballs")
    return plan


if __name__ == "__main__":
    main()
