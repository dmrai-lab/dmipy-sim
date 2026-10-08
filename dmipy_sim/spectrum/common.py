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
           "derive_seed", "n_tiles_per_axis"]

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
