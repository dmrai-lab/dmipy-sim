"""What every generator of the substrate spectrum (dmrai-lab/dmipy-sim#697, tessera#43) shares: the edges and
walk lengths the policy suite is built at, the halo rule that refuses a combination before any geometry is
drawn, the file index (sha256 per file) every entry ships, and the tiling arithmetic (a tile of 1 mm, its own
RNG stream, never two equal) a streaming generator partitions its domain into.

A spectrum entry is one directory: a substrate spec (geometry files + ``spec.json``), one ``manifest.json`` +
``plan/`` per walkable (edge, walk length) pair (the halo rule may refuse some), and a ``spectrum.json`` index
of every file's sha256. ``dmipy_sim.spectrum`` writes these; a fill (:mod:`dmipy_sim.fill`) reads them exactly
as it reads any other recipe -- the spectrum is generated, not a new kind of thing.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

__all__ = ["EDGES_M", "WALK_LENGTHS_S", "HALO_FREE_DIFFUSIVITY_M2_S", "HALO_MARGIN_CELLS", "TILE_EDGE_M",
           "SpectrumHaloError", "halo_m", "check_halo", "sha256_of", "write_index", "verify_index", "tile_seed",
           "derive_seed", "n_tiles_per_axis", "git_commit", "write_manifest", "pilot_n_and_floor", "pool_census"]

#: the four edges of the spectrum (#697): one voxel, DiSCo's own two sizes, and the "100x" tiled case.
EDGES_M = (25e-6, 250e-6, 1e-3, 5e-3)

#: the two walk lengths of the policy suite (#697 / tessera#43): 20 ms and 100 ms.
WALK_LENGTHS_S = (0.020, 0.100)

#: m^2/s, the free-water diffusivity the COMMON halo policy is checked against -- not a pool's own diffusivity
#: (a class may walk slower or faster water), but the one number that makes "25 um at 100 ms" a stated, citable
#: refusal: a textbook self-diffusion coefficient of water near body temperature (Holz et al., Phys. Chem. Chem.
#: Phys. 2 (2000) 4740, D(37 C) ~= 3.0e-9 m^2/s; 2.0e-9 m^2/s is the commonly budgeted in-vivo free-water value
#: used throughout the dMRI Monte-Carlo literature, e.g. Fieremans & Lee, NeuroImage 182 (2018) 39). At
#: :data:`WALK_LENGTHS_S`, this gives halos of 71.6 and 160.0 um, rounding to the 72 / 160 um of the issue text.
HALO_FREE_DIFFUSIVITY_M2_S = 2.0e-9

#: the halo rule's margin cells (dmipy_sim.fill.kit._reach_margin: ``8 sqrt(2 D T)``).
HALO_MARGIN_CELLS = 8.0

#: m, the tile a streaming generator fills independently (own RNG stream) and stitches: #697's "1 mm" unit.
TILE_EDGE_M = 1e-3


class SpectrumHaloError(ValueError):
    """Refused by name: the halo a walk of this length needs does not fit the edge asked for."""


def halo_m(T_max_s, *, D=HALO_FREE_DIFFUSIVITY_M2_S):
    """The excursion halo (m) of a walk of ``T_max_s`` seconds at diffusivity ``D``: ``8 sqrt(2 D T)``, the same
    rule :func:`dmipy_sim.fill.kit._reach_margin` applies to a block's own download margin."""
    return float(HALO_MARGIN_CELLS) * float(np.sqrt(2.0 * float(D) * float(T_max_s)))


def check_halo(edge_m, T_max_s, *, D=HALO_FREE_DIFFUSIVITY_M2_S, what=None):
    """Refuse by name when the halo of a ``T_max_s`` walk does not fit inside ``edge_m``: a walker started
    anywhere in the box could reach past its wall before the walk ends, and a tiled substrate has no
    neighbour beyond the edge to receive it. Returns the halo (m) when it fits."""
    h = halo_m(T_max_s, D=D)
    if h > float(edge_m):
        raise SpectrumHaloError(
            f"{what or 'this substrate'}: the halo of an 8 sqrt(2 D T) walk at T = {T_max_s * 1e3:.0f} ms is "
            f"{h * 1e6:.1f} um, which does not fit the {edge_m * 1e6:.1f} um edge; refused (#697's halo rule)")
    return h


def n_tiles_per_axis(edge_m, *, tile_edge_m=TILE_EDGE_M):
    """How many tiles of ``tile_edge_m`` side a cube of ``edge_m`` side is streamed as, per axis: 1 below the
    tile edge (the whole domain is one tile), else the exact ratio -- :data:`EDGES_M` divide it evenly."""
    if edge_m <= tile_edge_m * (1 + 1e-9):
        return 1
    n = edge_m / tile_edge_m
    if abs(n - round(n)) > 1e-6:
        raise ValueError(f"edge {edge_m:g} m is not a whole number of {tile_edge_m:g} m tiles ({n:g})")
    return int(round(n))


def derive_seed(seed, *indices):
    """A deterministic per-tile seed from the generator's own ``seed`` and its tile indices (ints or names,
    e.g. ``derive_seed(seed, "pilot")``): a 63-bit hash (not the identity and not a counter), so tile ``(0, 0)``
    and tile ``(0, 1)`` draw unrelated streams and no two tiles of a pack are equal -- the SAME indices always
    re-derive the SAME seed (reproducible, no state carried between tiles, which is what lets a tile be
    generated on its own in a stream)."""
    key = (int(seed),) + tuple(indices)
    return int(hashlib.sha256(repr(key).encode()).hexdigest(), 16) % (2 ** 63 - 1)


def tile_seed(seed, ix, iy):
    """The RNG seed of tile column ``(ix, iy)`` of a generator run with ``seed`` (:func:`derive_seed`)."""
    return derive_seed(seed, "tile", ix, iy)


def sha256_of(path):
    """The sha256 of a file's bytes, read in chunks (never the whole file in one allocation)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def write_index(out_dir, *, extra=None):
    """Write ``spectrum.json`` under ``out_dir``: the sha256 of every OTHER file under it (walked, sorted,
    posix-relative paths), plus ``extra`` fields the caller wants recorded (``id``, ``class``, ``edge_m``,
    ``seed``, sizes, timings). Re-walks the directory rather than taking a file list, so nothing written is
    missed and nothing stale lingers."""
    files = {}
    for root, _dirs, names in os.walk(out_dir):
        for name in sorted(names):
            if name == "spectrum.json":
                continue
            p = os.path.join(root, name)
            rel = os.path.relpath(p, out_dir).replace(os.sep, "/")
            files[rel] = dict(sha256=sha256_of(p), bytes=os.path.getsize(p))
    index = dict(extra or {}, files=files)
    path = os.path.join(out_dir, "spectrum.json")
    with open(path, "w") as fh:
        json.dump(index, fh, indent=1, sort_keys=True)
    return path


def verify_index(index_path):
    """Whether every file :func:`write_index` listed still hashes to what it recorded. Raises naming the first
    mismatch or missing file; returns the number of files checked."""
    out_dir = os.path.dirname(os.path.abspath(index_path))
    index = json.load(open(index_path))
    for rel, meta in index["files"].items():
        p = os.path.join(out_dir, rel)
        if not os.path.exists(p):
            raise ValueError(f"{index_path}: indexes {rel!r}, which does not exist")
        got = sha256_of(p)
        if got != meta["sha256"]:
            raise ValueError(f"{index_path}: {rel!r} hashes to {got}, index says {meta['sha256']}")
        if os.path.getsize(p) != meta["bytes"]:
            raise ValueError(f"{index_path}: {rel!r} is {os.path.getsize(p)} bytes, index says {meta['bytes']}")
    return len(index["files"])


def git_commit():
    """The worktree's own commit, or ``None`` outside a git checkout (recorded in every manifest's ``code``)."""
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def pilot_n_and_floor(target_floor):
    """``(n_pilot, floor_pilot)`` of the textbook Monte-Carlo floor ``1/sqrt(n)`` at a stated precision: the one
    pilot-walk rule every generator in this package uses (:func:`dmipy_sim.spectrum.strands.strands`'s own
    derivation). ``n_pilot = ceil(1 / target_floor**2)``, so ``floor_pilot`` hits ``target_floor`` exactly."""
    n_pilot = int(np.ceil(1.0 / float(target_floor) ** 2))
    return n_pilot, 1.0 / np.sqrt(n_pilot)


def pool_census(preds, grid, *, census_draws=200, seed=0):
    """The per-voxel volume fraction of every named membership test in ``preds`` (``{name: points -> bool}``)
    on ``grid``, by rejection sampling (:func:`dmipy_sim.spec.seeding.fill_per_voxel`'s own rule, at the cost of
    ``census_draws`` draws per voxel) -- the generic census a geometry with no closed-form clipping primitive
    (a packed cylinder/sphere, a label volume) still gets exactly: ``{name: (n_voxels,) float}``, grid-shaped
    arrays the caller reshapes. A predicate is a geometry's own exact classifier
    (:meth:`~dmipy_sim.geometry.base.Geometry.classify_positions_exact`, thresholded per pool) or a spec's
    membership test (:class:`~dmipy_sim.spec.walk._PoolTests`, for a multi-surface bundle) -- whichever the
    caller's substrate kind has. ``fill_per_voxel`` needs a positive ``want`` to run at all (it is written to
    stratify real seeds, not just measure a fraction), so this calls it with ``want = census_draws`` everywhere
    and reads back only its ``f`` (the kept points are discarded)."""
    from ..spec.seeding import fill_per_voxel
    want = np.full(grid.n_voxels, int(census_draws), np.int64)
    out = {}
    for name, pred in preds.items():
        _, _, f, _, _ = fill_per_voxel(pred, grid, want, trials_max=max(5000, census_draws * 10),
                                       census_draws=census_draws, seed=seed)
        out[name] = f
    return out


def write_manifest(variant_dir, *, id_, license_, citation, substrate, grid, T_max_s, scanner, counts_rel,
                   blocks_rel, total_walkers, n_blocks, dt_save_divisor=200.0):
    """A recipe-shaped ``manifest.json`` for a substrate kind :mod:`dmipy_sim.fill` does not read yet
    (``substrate["kind"]`` is not ``"disco"``): the SAME outer shape (``code``, ``substrate``, ``grid``,
    ``walk``, ``pack``, ``plan``) dmipy_sim.spectrum.strands.strands's own manifest uses, so a reader
    already expecting that shape finds the same fields, but ``substrate`` carries whatever this generator's
    kind needs (a ``spec_file`` for a packed-cylinder/sphere or label-volume entry, since none of those has a
    :mod:`dmipy_sim.fill.recipe.Recipe` reader of its own -- extending ``Recipe`` to the new kinds is a
    follow-up, not this generator's job, which is to write the data in the shape the recipe is DEFINED to
    have). No field basis: these classes carry no susceptibility source (dmipy-sim#697's PR2 scope)."""
    from ..replay.bank import mode_count
    man = dict(
        id=id_, license=license_, citation=citation,
        code=dict(repo="dmrai-lab/dmipy-sim", commit=git_commit()),
        substrate=substrate,
        variants=dict(field=dict(field=False)), default_variant="field",
        grid=dict(shape=list(grid.shape), voxel_size_m=list(grid.voxel_size_m), origin_m=list(grid.origin_m),
                  attach="substrate"),
        walk=dict(T_max_s=float(T_max_s), dt_save_s=float(T_max_s) / dt_save_divisor, scanner=scanner,
                  floor_fraction=0.1, adaptive_steps=False, walker_batch_size=4096, census_draws=200),
        pack=dict(K=mode_count(T_max_s, scanner), K_path=mode_count(T_max_s, scanner), position_container="bands",
                  blt_container="bands", blt_K=mode_count(T_max_s, scanner)),
        plan=dict(file=counts_rel, blocks=blocks_rel, walkers=int(total_walkers), n_blocks=int(n_blocks)),
    )
    path = os.path.join(variant_dir, "manifest.json")
    with open(path, "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    return path
