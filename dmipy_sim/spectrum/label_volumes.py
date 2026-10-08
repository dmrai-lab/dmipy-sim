"""The label-volume generator of the substrate spectrum (dmrai-lab/dmipy-sim#697): a random sphere packing
(:mod:`dmipy_sim.spectrum.packs`'s own tiling and object-count rule, reused rather than re-derived) rasterised
onto a voxel grid as a segmented image -- a porous medium of SOLID grains (label 1, no water, not seeded --
:func:`dmipy_sim.spec.producers.label_volume_spec`'s own convention, the one Imperial-rocks/Ling-sand-packs
family already uses) and a pore space (label 0, "free", the one walked pool), written as a single-file NRRD
(:func:`dmipy_sim.io.label_volume.write_nrrd`) and read back the same way any other label volume is: by
:func:`~dmipy_sim.spec.producers.label_volume_spec`, nothing constructed directly."""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

from .common import SpectrumHaloError, check_halo, derive_seed, n_tiles_per_axis, pilot_n_and_floor, write_index, write_manifest
from .packs import GAMMA_SHAPE, N_OBJECTS_PER_TILE, _occupied_by_objects, _place_tile

log = logging.getLogger("dmipy_sim.spectrum")

__all__ = ["label_volume", "TARGET_FRACTION", "VOXEL_SIZE_M"]

#: under spheres' own RSA_LIMIT (0.40): the grain (solid) fraction of the segmented image.
TARGET_FRACTION = 0.3

#: voxels per axis, at every edge (DiSCo's own 40^3 convention, dmipy_sim.spectrum.strands): FIXED, not the
#: other classes' absolute :data:`~dmipy_sim.spectrum.packs.VOXEL_SIZE_M` -- a label volume's wall IS the grid
#: (the Manhattan surface of the segmentation), so the smallest edge (25 um) needs internal voxel structure to
#: have a wall at all, which an absolute 25 um voxel (one voxel, no face between any two) does not give it.
N_VOXELS_PER_AXIS = 40


def label_volume(edge_m, seed, *, out_dir, target_fraction=TARGET_FRACTION, n_objects_per_tile=N_OBJECTS_PER_TILE,
                 shape=GAMMA_SHAPE, T_max_s=(0.020, 0.100), scanner="connectom"):
    """Generate one label-volume entry of the substrate spectrum (dmrai-lab/dmipy-sim#697): a random sphere
    packing at ``target_fraction`` (default 0.3, under the RSA saturation of 0.40) rasterised onto a
    ``VOXEL_SIZE_M``-voxel grid of ``edge_m`` side as a segmented image (grain = solid, free = pore space),
    tiled exactly as :func:`dmipy_sim.spectrum.packs.spheres` tiles -- see its module docstring for why the
    object count is fixed and the radius solved from it, and for the tile's own accepted simplification (a
    grain confined by its centre to its own tile, not its radius)."""
    from ..spec.producers import label_volume_spec
    from ..io.label_volume import write_nrrd
    from ..phantom import Grid

    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    sub_dir = os.path.join(out_dir, "substrate")
    os.makedirs(sub_dir, exist_ok=True)

    n_side = n_tiles_per_axis(edge_m)
    tile_edge = edge_m / n_side
    import itertools
    radii_all, centers_all, achieved_fracs = [], [], []
    for idx in itertools.product(range(n_side), range(n_side), range(n_side)):
        r, c, ach = _place_tile("sphere", edge_m, tile_edge, idx, seed, n_objects=n_objects_per_tile,
                               target_fraction=target_fraction, shape=shape)
        radii_all.append(r); centers_all.append(c); achieved_fracs.append(ach)
    radii = np.concatenate(radii_all)
    centers = np.concatenate(centers_all, axis=0)

    shape3 = (N_VOXELS_PER_AXIS,) * 3
    voxel_size_scalar = edge_m / N_VOXELS_PER_AXIS
    voxel_size = (voxel_size_scalar,) * 3
    origin = tuple(-edge_m / 2.0 + 0.5 * np.asarray(voxel_size))
    grid = Grid(shape=shape3, voxel_size_m=voxel_size, origin_m=origin)

    grain = _occupied_by_objects("sphere", radii, centers, grid).reshape(grid.shape)
    labels = grain.astype(np.uint8)  # 0 = free (pore), 1 = grain (solid)
    nrrd_path = os.path.join(sub_dir, "labels.nrrd")
    write_nrrd(nrrd_path, labels, voxel_size, origin=grid.corner_m)

    spec = label_volume_spec(nrrd_path, pools={0: "free", 1: "grain"}, voxel_size=voxel_size, D=2.0e-9,
                            rho2=0.0, id=f"spectrum/label_volume/edge_{edge_m:g}m_seed_{seed}")
    spec_path = os.path.join(sub_dir, "spec.json")
    spec.save(spec_path)

    n_pilot, floor_pilot = pilot_n_and_floor(0.02)
    count_free = np.where(~grain, n_pilot, 0).astype(np.int64)
    counts_path = os.path.join(sub_dir, "counts.npz")
    np.savez(counts_path, count_free=count_free)
    total_walkers = int(count_free.sum())

    blocks = [dict(block=0, i=[0, grid.shape[0]], j=[0, grid.shape[1]], k=[0, grid.shape[2]], seed=int(seed),
                  voxels=int(grid.n_voxels), walkers=dict(free=total_walkers))]
    blocks_path = os.path.join(sub_dir, "blocks.json")
    with open(blocks_path, "w") as fh:
        json.dump(dict(grid=dict(shape=list(grid.shape)), target_walkers_per_block=total_walkers, blocks=blocks),
                 fh, indent=1)

    license_ = ("synthetic geometry, generated for dmrai-lab/dmipy-sim#697: a random sphere packing "
               "rasterised as a segmented image, no external data")
    citation = "dmrai-lab/dmipy-sim#697, dmrai-lab/tessera#43"
    variants, refused = [], []
    for T_max in T_max_s:
        try:
            check_halo(edge_m, T_max, what=f"label_volume edge={edge_m * 1e6:.0f} um")
        except SpectrumHaloError as e:
            log.info("spectrum.label_volumes: %s", e)
            refused.append(float(T_max))
            continue
        variant_dir = os.path.join(out_dir, f"T{int(round(T_max * 1e3)):03d}ms")
        os.makedirs(variant_dir, exist_ok=True)
        write_manifest(variant_dir, id_=f"spectrum/label_volume/edge_{edge_m:g}m_seed_{seed}/T{T_max}",
                      license_=license_, citation=citation,
                      substrate=dict(kind="label_volume", spec_file="../substrate/spec.json",
                                    image_file="../substrate/labels.nrrd", n_objects=int(len(radii)),
                                    target_fraction=float(target_fraction), side_m=float(edge_m)),
                      grid=grid, T_max_s=T_max, scanner=scanner, counts_rel="../substrate/counts.npz",
                      blocks_rel="../substrate/blocks.json", total_walkers=total_walkers, n_blocks=len(blocks))
        variants.append(float(T_max))

    index_path = write_index(out_dir, extra=dict(
        id=f"spectrum/label_volume/edge_{edge_m:g}m_seed_{seed}", generator="dmipy_sim.spectrum.label_volumes.label_volume",
        class_="label_volume", edge_m=float(edge_m), seed=int(seed), n_objects=int(len(radii)),
        n_tiles_per_axis=int(n_side), achieved_fraction_mean=float(np.mean(achieved_fracs)),
        variants_t_max_s=variants, refused_t_max_s=refused, pilot=dict(n_pilot=n_pilot, floor_pilot=floor_pilot),
        seconds=time.time() - t_start))

    return dict(out_dir=out_dir, n_objects=int(len(radii)), n_tiles_per_axis=n_side, refused=refused,
               variants=variants, achieved_fraction_mean=float(np.mean(achieved_fracs)), index=index_path,
               seconds=time.time() - t_start)
