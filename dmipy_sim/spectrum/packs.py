"""The analytic-pack generators of the substrate spectrum (dmrai-lab/dmipy-sim#697): random periodic packings
of cylinders or spheres at a stated volume fraction, tiled the same way :mod:`dmipy_sim.spectrum.strands`
tiles -- independent tiles, their own RNG stream, streamed one at a time -- and read back through
:func:`dmipy_sim.spec.producers.packed_request`'s own geometries (:class:`~dmipy_sim.geometry.PackedCylinders`,
:class:`~dmipy_sim.geometry.PackedSpheres`).

**Why the object count is fixed, not the radius.** A strand's AREAL density (strands per unit cross-section)
makes its object count scale with ``edge**2``; a packed cylinder's cross-sectional packing is the same.  A
packed SPHERE's is a VOLUME fraction, so at a fixed micron-scale radius its object count scales with
``edge**3``: a 1 mm tile at DiSCo's own ~1 um pore scale would need tens of millions of spheres, and
``pack_spheres``' random sequential addition is O(n) per placement against every placed sphere so far -- O(n^2)
overall, intractable there. This generator instead fixes the OBJECT COUNT per tile
(:data:`N_OBJECTS_PER_TILE`, chosen so RSA stays fast at every edge) and SOLVES the gamma radius law's scale
from the stated target fraction and the tile's own edge -- the pore size tracks the edge rather than a fixed
physical scale. This is a stated control-the-cost choice (#697: "substrates whose size is controlled"), not a
claim that any real material's pore size varies with the sample size.

**Tiling.** A cylinder's packing is 2-D (its axis is unconfined -- :class:`PackedCylinders`' domain is
periodic in the cross-section and open along the axis, so nothing bounds or needs tiling along it); it tiles
in columns exactly as a strand does. A sphere's packing is 3-D and tiles in a cube of tiles. Each tile is
packed independently at ``L =`` its own edge (never derived from the target fraction, so tiles place edge to
edge exactly); a tile's own placement confines every CENTRE to its own sub-box, so two tiles' objects never
overlap, but an object's RADIUS may cross into a neighbour's -- the same accepted simplification
dmipy-sim#699 took for a strand's lateral confinement, stated here for the same reason (no cross-tile overlap
check is attempted).
"""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

from .common import (EDGES_M, SpectrumHaloError, TILE_EDGE_M, check_halo, derive_seed, n_tiles_per_axis,
                     pilot_n_and_floor, write_index, write_manifest)

log = logging.getLogger("dmipy_sim.spectrum")

__all__ = ["cylinders", "spheres", "N_OBJECTS_PER_TILE", "GAMMA_SHAPE"]

#: objects per tile: fixed so random sequential addition (O(n^2)) stays fast at every edge (#697).
N_OBJECTS_PER_TILE = 400

#: the gamma radius law's shape parameter (polydispersity), the same at every edge and every tile.
GAMMA_SHAPE = 4.0

#: RSA_LIMIT-safe stated volume fractions (dmipy_sim.spec.producers.RSA_LIMIT: cylinder 0.60, sphere 0.40).
TARGET_FRACTION = {"cylinder": 0.5, "sphere": 0.3}

_default_scanner_class = "connectom"
VOXEL_SIZE_M = 25e-6


def _gamma_scale_for(kind, tile_edge_m, n_objects, target_fraction, shape=GAMMA_SHAPE):
    """The gamma law's scale that makes ``n_objects`` radii, in expectation, cover ``target_fraction`` of a
    tile of ``tile_edge_m`` (area for a cylinder's cross-section, volume for a sphere) -- the moment equations
    for a gamma(``shape``, ``scale``) variate's second (cylinder) or third (sphere) raw moment."""
    if kind == "cylinder":
        er2_over_scale2 = shape * (shape + 1.0)
        scale2 = target_fraction * tile_edge_m ** 2 / (n_objects * np.pi * er2_over_scale2)
        return float(np.sqrt(scale2))
    er3_over_scale3 = shape * (shape + 1.0) * (shape + 2.0)
    scale3 = target_fraction * tile_edge_m ** 3 / (n_objects * (4.0 / 3.0) * np.pi * er3_over_scale3)
    return float(scale3 ** (1.0 / 3.0))


def _place_tile(kind, edge_m, tile_edge_m, idx, seed, *, n_objects, target_fraction, shape):
    """One tile's radii and GLOBAL-frame centres: packed independently at ``L = tile_edge_m`` (never derived
    from the fraction, so tiles place edge to edge exactly), then shifted from its own
    ``[-tile_edge_m/2, tile_edge_m/2]`` frame into the domain's ``[-edge_m/2, edge_m/2]`` one at tile index
    ``idx`` (a tuple of 2 ints for a cylinder's cross-section, 3 for a sphere)."""
    from ..geometry.packing import pack_cylinders, pack_spheres
    dim = len(idx)
    tseed = derive_seed(seed, "tile", *idx)
    scale = _gamma_scale_for(kind, tile_edge_m, n_objects, target_fraction, shape)
    rng = np.random.default_rng(tseed)
    radii = np.maximum(rng.gamma(shape, scale, int(n_objects)), scale * 1e-2)
    pack_fn = pack_cylinders if kind == "cylinder" else pack_spheres
    local_centers, L, achieved = pack_fn(radii, L=tile_edge_m, seed=tseed % (2 ** 32))
    offset = -edge_m / 2.0 + (np.asarray(idx, float) + 0.5) * tile_edge_m
    global_centers = local_centers + offset[:dim]
    return radii, global_centers, float(achieved)


def _occupied_by_objects(kind, radii, centers, grid):
    """Which voxels of ``grid`` any object (a cylinder's cross-section circle, unrestricted along its axis; a
    sphere) touches, localized per object -- its own voxel-index bounding box, grown by its radius -- not a
    pass over every voxel: O(n_objects x its own local box), the shape that stays bounded as the edge (and so
    the voxel count) grows, unlike a Monte-Carlo census against the geometry's classifier (tried first, O(grid
    voxels x census draws); see the call site)."""
    dim = centers.shape[1]
    corner = np.asarray(grid.corner_m, float)[:dim]
    step = np.asarray(grid.step_m, float)[:dim]
    shape_full = np.asarray(grid.shape)
    occ = np.zeros(grid.shape, bool)
    for r, c in zip(radii, centers):
        lo = np.floor((c - r - corner) / step).astype(np.int64)
        hi = np.ceil((c + r - corner) / step).astype(np.int64)
        lo = np.clip(lo, 0, shape_full[:dim] - 1)
        hi = np.clip(hi, 0, shape_full[:dim] - 1)
        if np.any(hi < lo):
            continue
        ranges = [np.arange(lo[a], hi[a] + 1) for a in range(dim)]
        mesh = np.meshgrid(*ranges, indexing="ij")
        idx = np.stack([m.ravel() for m in mesh], axis=1)
        pts = corner + (idx.astype(float) + 0.5) * step
        inside = np.sum((pts - c) ** 2, axis=1) <= r * r
        if not inside.any():
            continue
        hit = idx[inside]
        if dim == 2:
            occ[hit[:, 0], hit[:, 1], :] = True
        else:
            occ[hit[:, 0], hit[:, 1], hit[:, 2]] = True
    return occ.reshape(-1)


def _pack(kind, edge_m, seed, *, out_dir, target_fraction, n_objects_per_tile, shape, T_max_s, scanner):
    from ..spec.build import spec_of
    from ..geometry.packed import PackedCylinders, PackedSpheres
    from ..phantom import Grid

    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    sub_dir = os.path.join(out_dir, "substrate")
    os.makedirs(sub_dir, exist_ok=True)

    n_side = n_tiles_per_axis(edge_m)
    tile_edge = edge_m / n_side
    dim = 2 if kind == "cylinder" else 3
    idx_ranges = [range(n_side)] * dim

    radii_all, centers_all, achieved_fracs = [], [], []
    import itertools
    for idx in itertools.product(*idx_ranges):
        r, c, ach = _place_tile(kind, edge_m, tile_edge, idx, seed, n_objects=n_objects_per_tile,
                               target_fraction=target_fraction, shape=shape)
        radii_all.append(r); centers_all.append(c); achieved_fracs.append(ach)
    radii = np.concatenate(radii_all)
    centers = np.concatenate(centers_all, axis=0)
    n_objects = len(radii)

    Geom = PackedCylinders if kind == "cylinder" else PackedSpheres
    # PackedCylinders/PackedSpheres.__init__ computes min_gap EAGERLY as a full pairwise distance matrix
    # (dmipy_sim.geometry.packing.periodic_min_gap): O(n_objects^2), ~60 GB and an OOM kill at the 5 mm
    # sphere case's 50,000 combined objects (measured). The spec this entry cites is built from one
    # REPRESENTATIVE tile's own objects (the same statistics, bounded at n_objects_per_tile, safe to
    # construct and walk) rather than the full tiled count; the census/plan below still cover the WHOLE
    # domain from the raw per-tile arrays directly, which need no Geometry object at all.
    spec_radii, spec_centers, spec_L = (radii, centers, float(edge_m)) if n_side == 1 else \
        (radii_all[0], centers_all[0], float(tile_edge))
    geom = Geom(spec_radii, spec_centers, spec_L)
    spec = spec_of(geom, id=f"spectrum/{kind}s/edge_{edge_m:g}m_seed_{seed}")
    spec_path = os.path.join(sub_dir, "spec.json")
    spec.save(spec_path)

    if kind == "cylinder":
        shape3 = (int(round(edge_m / VOXEL_SIZE_M)),) * 2 + (1,)
        voxel_size = (VOXEL_SIZE_M, VOXEL_SIZE_M, edge_m)
    else:
        shape3 = (int(round(edge_m / VOXEL_SIZE_M)),) * 3
        voxel_size = (VOXEL_SIZE_M,) * 3
    origin = tuple(-edge_m / 2.0 + 0.5 * np.asarray(voxel_size))
    grid = Grid(shape=shape3, voxel_size_m=voxel_size, origin_m=origin)

    n_pilot, floor_pilot = pilot_n_and_floor(0.02)
    # a voxel's two pools only need a SEEN/not-seen decision here (the count below is flat at n_pilot wherever
    # census > 0), so this reads it directly off the objects -- which voxels an object's own ball touches --
    # rather than through a Monte-Carlo rejection census: O(n_objects x its own local box of voxels), never
    # O(n_voxels), which is what keeps this tractable from 25 um to 5 mm (a global rejection census against the
    # geometry's own `classify_positions_exact`, tried first, measured ~5 min at 1 mm already -- O(n_voxels) is
    # the wrong shape for a cost that must stay bounded as the edge grows).
    intra_occ = _occupied_by_objects(kind, radii, centers, grid)
    count_intra = np.where(intra_occ, n_pilot, 0).astype(np.int64)
    # the extra pool is seen almost everywhere at these volume fractions (0.3-0.5): the complement, true unless
    # an object happens to fill a voxel completely (not reached at this radius/voxel ratio).
    count_extra = np.full(grid.n_voxels, n_pilot, np.int64)
    counts_path = os.path.join(sub_dir, "counts.npz")
    np.savez(counts_path, count_extra=count_extra.reshape(grid.shape), count_intra=count_intra.reshape(grid.shape))
    total_walkers = int(count_extra.sum() + count_intra.sum())

    blocks = [dict(block=0, i=[0, grid.shape[0]], j=[0, grid.shape[1]], k=[0, grid.shape[2]],
                  seed=int(seed), voxels=int(grid.n_voxels),
                  walkers=dict(extra=int(count_extra.sum()), intra=int(count_intra.sum())))]
    blocks_path = os.path.join(sub_dir, "blocks.json")
    with open(blocks_path, "w") as fh:
        json.dump(dict(grid=dict(shape=list(grid.shape)), target_walkers_per_block=total_walkers, blocks=blocks),
                 fh, indent=1)

    license_ = (f"synthetic geometry, generated for dmrai-lab/dmipy-sim#697: a random periodic {kind} packing "
               f"at a stated volume fraction, no external data")
    citation = f"dmrai-lab/dmipy-sim#697, dmrai-lab/tessera#43"
    variants, refused = [], []
    for T_max in T_max_s:
        try:
            check_halo(edge_m, T_max, what=f"{kind}s edge={edge_m * 1e6:.0f} um")
        except SpectrumHaloError as e:
            log.info("spectrum.packs: %s", e)
            refused.append(float(T_max))
            continue
        variant_dir = os.path.join(out_dir, f"T{int(round(T_max * 1e3)):03d}ms")
        os.makedirs(variant_dir, exist_ok=True)
        write_manifest(variant_dir, id_=f"spectrum/{kind}s/edge_{edge_m:g}m_seed_{seed}/T{T_max}",
                      license_=license_, citation=citation,
                      substrate=dict(kind=f"packed_{kind}", spec_file="../substrate/spec.json",
                                    n_objects=int(n_objects), target_fraction=float(target_fraction),
                                    side_m=float(edge_m), spec_side_m=float(spec_L),
                                    spec_is_one_representative_tile=bool(n_side > 1)),
                      grid=grid, T_max_s=T_max, scanner=scanner, counts_rel="../substrate/counts.npz",
                      blocks_rel="../substrate/blocks.json", total_walkers=total_walkers, n_blocks=len(blocks))
        variants.append(float(T_max))

    index_path = write_index(out_dir, extra=dict(
        id=f"spectrum/{kind}s/edge_{edge_m:g}m_seed_{seed}", generator=f"dmipy_sim.spectrum.packs.{kind}s",
        class_=f"packed_{kind}", edge_m=float(edge_m), seed=int(seed), n_objects=int(n_objects),
        n_tiles_per_axis=int(n_side), achieved_fraction_mean=float(np.mean(achieved_fracs)),
        variants_t_max_s=variants, refused_t_max_s=refused, pilot=dict(n_pilot=n_pilot, floor_pilot=floor_pilot),
        seconds=time.time() - t_start))

    return dict(out_dir=out_dir, n_objects=n_objects, n_tiles_per_axis=n_side, refused=refused, variants=variants,
               achieved_fraction_mean=float(np.mean(achieved_fracs)), index=index_path,
               seconds=time.time() - t_start)


def cylinders(edge_m, seed, *, out_dir, target_fraction=None, n_objects_per_tile=N_OBJECTS_PER_TILE,
             shape=GAMMA_SHAPE, T_max_s=(0.020, 0.100), scanner=_default_scanner_class):
    """Generate one packed-cylinder entry of the substrate spectrum (dmrai-lab/dmipy-sim#697): a random
    periodic packing of parallel cylinders at ``target_fraction`` (default :data:`TARGET_FRACTION`, 0.5, under
    the RSA saturation of 0.60) in a square cross-section of ``edge_m`` side, the axis unconfined. See the
    module docstring for the tiling and object-count rule."""
    return _pack("cylinder", edge_m, seed, out_dir=out_dir,
                target_fraction=float(target_fraction if target_fraction is not None else TARGET_FRACTION["cylinder"]),
                n_objects_per_tile=n_objects_per_tile, shape=shape, T_max_s=T_max_s, scanner=scanner)


def spheres(edge_m, seed, *, out_dir, target_fraction=None, n_objects_per_tile=N_OBJECTS_PER_TILE,
           shape=GAMMA_SHAPE, T_max_s=(0.020, 0.100), scanner=_default_scanner_class):
    """Generate one packed-sphere entry of the substrate spectrum (dmrai-lab/dmipy-sim#697): a random periodic
    packing of spheres at ``target_fraction`` (default :data:`TARGET_FRACTION`, 0.3, under the RSA saturation
    of 0.40) in a cube of ``edge_m`` side. See the module docstring for the tiling and object-count rule."""
    return _pack("sphere", edge_m, seed, out_dir=out_dir,
                target_fraction=float(target_fraction if target_fraction is not None else TARGET_FRACTION["sphere"]),
                n_objects_per_tile=n_objects_per_tile, shape=shape, T_max_s=T_max_s, scanner=scanner)
