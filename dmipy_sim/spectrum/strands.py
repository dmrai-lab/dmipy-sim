"""The strands generator of the substrate spectrum (dmrai-lab/dmipy-sim#697): random curved strands at DiSCo's
own radius and density statistics (:mod:`dmipy_sim.spectrum.disco_stats`), streamed tile by tile so a 5 mm pack
is never held whole in memory, built into the exact same shape a DiSCo fill reads
(:func:`dmipy_sim.spec.producers.disco_spec` + :mod:`dmipy_sim.fill.recipe`).

**Tiling.** The domain is a cube of edge ``edge_m``. It is partitioned into columns of
:data:`~dmipy_sim.spectrum.common.TILE_EDGE_M` (1 mm) lateral size (one column when ``edge_m <= 1 mm``, else a
``n x n`` grid of them); each strand belongs to exactly one column and is generated end to end along z inside
it, so a column is a self-contained streaming unit: its own :func:`~dmipy_sim.spectrum.common.tile_seed`, its
own Poisson strand count (DiSCo's own areal density times the column's footprint), each strand's centerline and
radius written straight to the track/diameter files and folded into the running per-voxel census before the
column's arrays are dropped. No two columns draw the same seed and no two columns of one generator run are
equal (:data:`DISCO_AREAL_DENSITY_PER_M2` makes that a statistical statement, not a structural one -- two
columns of the same size ARE drawn from the same distribution, which is the point).

**Counts.** A strand's radius is drawn once per strand (axolemma / inner, DiSCo's own statistics) and carried
for its whole length, as :func:`~dmipy_sim.spec.producers.disco_spec` requires. Per-voxel walker counts come
from one pilot: a tiny 25 um realisation of the SAME statistics is walked on the JAX engine
(:func:`~dmipy_sim.spec.walk.walk_spec`) at ``n_pilot = ceil(1 / target_floor**2)`` walkers per pool -- the
textbook Monte-Carlo floor ``1/sqrt(n)`` this codebase already cites throughout (``scanners.py``'s
``save_interval`` derivation, ``replay.bank.storage_band_hz``'s certificate) -- which both proves the geometry
and its seeding are walkable and gives ``n_pilot`` its floor by construction (``floor_pilot = 1/sqrt(n_pilot)``).
Because the strand statistics are translation-invariant across columns, that SAME ``n_pilot`` is the target for
every occupied voxel of every larger grid: the per-voxel CENSUS (the pool's local volume fraction, read off the
generated geometry directly -- :func:`dmipy_sim.spec.seeding._clip_segments_to_voxels`'s analytic clipped-volume
sum, no rejection sampling needed) decides only which voxels are "seen" (count > 0) and which hold none of a
pool, never how many walkers a seen voxel gets (:func:`dmipy_sim.spec.seeding.plan_seeding`, fed a pilot floor
and count broadcast over every census-positive voxel of the target grid)."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time

import numpy as np

from .common import (EDGES_M, HALO_FREE_DIFFUSIVITY_M2_S, SpectrumHaloError, TILE_EDGE_M, check_halo, derive_seed,
                     n_tiles_per_axis, tile_seed, write_index)
from .disco_stats import DISCO_AREAL_DENSITY_PER_M2, DISCO_INNER_RADIUS_STATS_M, DISCO_POINTS_PER_MM
from ._tck_stream import StreamingTckWriter

log = logging.getLogger("dmipy_sim.spectrum")

__all__ = ["strands", "PILOT_TARGET_FLOOR", "PILOT_EDGE_M", "FAR_GRID_NODE_BUDGET", "FAR_NEAR_M", "FAR_BLEND_M",
          "FAR_CUTOFF_M"]

#: m, the pilot's own edge: the spectrum's smallest size, one voxel.
PILOT_EDGE_M = 25e-6

#: the pilot's stated Monte-Carlo precision target (2 %): :func:`dmipy_sim.spec.seeding.plan_seeding`'s
#: ``target_floor``, giving ``n_pilot = ceil(1 / target_floor**2) = 2500`` walkers per pool per voxel.
PILOT_TARGET_FLOOR = 0.02

#: nodes, the far grid's SIZE/TIME budget (#697: "substrates whose size is controlled, not collected"). The
#: closed-form basis's per-node cost is set by :data:`FAR_NEAR_M` alone (fixed, independent of edge_m), so
#: capping the node COUNT caps both the far grid's bytes and its build time the same way at every edge -- the
#: resolution that pays for it: 1 mm gets ~5 um spacing (DiSCo's own released far grid is 2.5 um / 64 M nodes,
#: "hours on a GH200"; this generator's 8 M-node budget is deliberately 8x coarser and faster), 5 mm ~25 um.
FAR_GRID_NODE_BUDGET = 8_000_000

#: m, the far grid's spacing floor: below the node budget's own bound, never coarser than this -- DiSCo's own
#: released far grid's spacing, so a small edge (where the budget never binds) gets DiSCo-grade resolution.
FAR_SPACING_FLOOR_M = 2.5e-6

#: m, the DiSCo recipe's own voxel (its released grid is 40^3 of these over 1 mm): the spectrum's one grid
#: resolution, so a 5 mm plan is the 1 mm plan's grid tiled 5x on every axis.
VOXEL_SIZE_M = 25e-6

_DEFAULT_SCANNER = "connectom"


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _draw_radius(rng, stats):
    """One inner (axolemma) radius from a normal truncated to ``[min, max]`` at DiSCo's own mean/std."""
    for _ in range(100):
        r = rng.normal(stats["mean"], stats["std"])
        if stats["min"] <= r <= stats["max"]:
            return float(r)
    return float(np.clip(stats["mean"], stats["min"], stats["max"]))


def _generate_column(seed, ix, iy, tile_edge_m, edge_m, *, radii_stats, density, points_per_mm, wiggle_fraction=0.02):
    """Every strand of column ``(ix, iy)``: a generator of ``(centerline_m (k, 3), radius_m)``, one per strand,
    its own RNG stream (:func:`~dmipy_sim.spectrum.common.tile_seed`). A strand runs the FULL z-extent
    ``edge_m`` (the column, not a sub-tile, is the streaming unit in z -- nothing stitches across a z-face
    because nothing is ever cut there), laterally a reflected random walk about an ``(x0, y0)`` drawn uniform in
    the column's own footprint, so it never leaves its home column (the generator's one simplifying rule: a
    strand may cross a Z face by construction, since it is generated whole, but is confined in X/Y to the
    column that drew it)."""
    rng = np.random.default_rng(tile_seed(seed, ix, iy))
    lo = np.array([ix * tile_edge_m, iy * tile_edge_m])
    area = tile_edge_m ** 2
    n_strands = int(rng.poisson(density * area))
    n_points = max(2, int(round(points_per_mm * edge_m * 1e3)))
    z = np.linspace(0.0, edge_m, n_points)
    dz = edge_m / max(n_points - 1, 1)
    for _ in range(n_strands):
        r_in = _draw_radius(rng, radii_stats)
        margin = r_in * 1.5
        half = tile_edge_m / 2.0 - margin
        half = max(half, tile_edge_m * 1e-3)
        x0, y0 = lo + tile_edge_m / 2.0 + rng.uniform(-half, half, 2)
        step = wiggle_fraction * tile_edge_m
        dx = np.concatenate([[0.0], rng.normal(0.0, step, n_points - 1)])
        dy = np.concatenate([[0.0], rng.normal(0.0, step, n_points - 1)])
        x = np.clip(x0 + np.cumsum(dx), lo[0] + margin, lo[0] + tile_edge_m - margin)
        y = np.clip(y0 + np.cumsum(dy), lo[1] + margin, lo[1] + tile_edge_m - margin)
        centerline = np.stack([x, y, z], axis=1)
        yield centerline, r_in


def _census_from_segments(A, B, r, grid):
    """The analytic per-voxel volume fraction of the swept tube ``A -> B`` radius ``r`` on ``grid``: the clipped
    cylindrical volume per voxel (:func:`dmipy_sim.spec.seeding._clip_segments_to_voxels`) over the voxel volume
    -- a deterministic geometric sum, no Monte-Carlo draws (the same primitive
    :func:`dmipy_sim.spec.seeding.fill_swept_by_voxel` clips before it samples points inside the result)."""
    from ..spec.seeding import _clip_segments_to_voxels
    if len(A) == 0:
        return np.zeros(grid.n_voxels)
    seg, vox, t0, t1 = _clip_segments_to_voxels(A, B, r, grid)
    L = np.linalg.norm(B - A, axis=1)
    w = np.pi * r[seg] ** 2 * L[seg] * (t1 - t0)
    return np.bincount(vox, weights=w, minlength=grid.n_voxels) / float(np.prod(grid.voxel_size_m))


def _pilot_walk(seed, radii_stats, density, points_per_mm, *, out_dir, n_pilot, scanner=_DEFAULT_SCANNER):
    """Build and walk the pilot (:data:`PILOT_EDGE_M`, one column, seed :func:`~.common.derive_seed`'s
    ``"pilot"`` branch -- never one of the main generation's own tile seeds): writes ``pilot/strands.tck`` +
    ``pilot/strands_diameters.txt`` under ``out_dir``, walks it a short, safe ``T_max`` (well inside its own
    halo budget: :data:`PILOT_EDGE_M` is too small for either policy walk length, dmipy-sim#697's own halo
    rule), and returns ``(n_pilot, floor_pilot, pilot_dir)``. This is the "pilot shard walked on the JAX engine
    at the smallest edge" dmipy-sim#697 asks for: it PROVES the statistics are walkable (a real
    :func:`~dmipy_sim.spec.walk.walk_spec` call, JAX, CPU) rather than assuming it; the floor itself is the
    closed form ``1/sqrt(n_pilot)`` by construction (``n_pilot`` is chosen to hit the stated target exactly),
    not re-measured from the walk's own signal."""
    from ..spec import disco_spec, walk_spec
    pilot_dir = os.path.join(out_dir, "pilot")
    os.makedirs(pilot_dir, exist_ok=True)
    tracks = os.path.join(pilot_dir, "strands.tck")
    diam = os.path.join(pilot_dir, "strands_diameters.txt")
    pseed = derive_seed(seed, "pilot")
    n_written = 0
    with StreamingTckWriter(tracks, coordinate_unit_m=25e-6) as w, open(diam, "w") as df:
        for centerline, r_in in _generate_column(pseed, 0, 0, PILOT_EDGE_M, PILOT_EDGE_M, radii_stats=radii_stats,
                                                 density=density, points_per_mm=points_per_mm):
            w.add(centerline)
            df.write(f"{2.0 * r_in / 1e-3:.9f}\n")
            n_written += 1
    if n_written == 0:
        raise RuntimeError(f"the {PILOT_EDGE_M * 1e6:.0f} um pilot drew zero strands at seed {pseed}; "
                           "pick a different generator seed")
    spec = disco_spec(tracks, diam, coordinate_unit_m=25e-6, diameter_unit_m=1e-3, side_m=PILOT_EDGE_M, field=False)
    T_safe = min((PILOT_EDGE_M / 16.0) ** 2 / (2.0 * HALO_FREE_DIFFUSIVITY_M2_S), 1e-3)
    dt_save = T_safe / 20.0
    t0 = time.time()
    walk_spec(spec, n_pilot * 2, T_safe, dt_save, seed=pseed, require_gpu=False, n_probe=max(20_000, n_pilot * 4))
    log.info("spectrum.strands: pilot walked %d walkers/pool in %.1f s (T_max=%.2g s)", n_pilot, time.time() - t0, T_safe)
    floor_pilot = 1.0 / np.sqrt(n_pilot)
    return n_pilot, float(floor_pilot), pilot_dir


def _build_far_grid(spec, out_path, *, node_budget=FAR_GRID_NODE_BUDGET, spacing_floor_m=FAR_SPACING_FLOOR_M):
    """The substrate's far field grid (:meth:`~dmipy_sim.fields.strand_field.StrandFieldBasis.build_far_grid`):
    spacing ``h = max(spacing_floor_m, edge / node_budget**(1/3))`` -- DiSCo-grade (2.5 um) resolution UNLESS
    the node budget would be exceeded at that spacing, in which case the edge alone is coarsened (#697:
    "substrates whose size is controlled"). ``near_m`` / ``blend_m`` / ``cutoff_m`` are then DERIVED from ``h``
    and the largest sheath radius so the switch always starts beyond its required gate
    (``near_m = 2 R_out_max + 4.5h``, ``blend_m = 2h``: the basis's own ``near - blend >= r_min`` check with a
    margin of ``0.5h``, as tight as is safe) -- fixing them independently of ``h`` (as
    ``tests/fill/conftest.py``'s tiny fixed-spacing template does) breaks the moment a coarser-than-fixture
    ``h`` makes ``2h`` comparable to the sheath radius, which is exactly what a budget-coarsened 5 mm spacing
    does. Because the gather cost per node grows with ``near_m**3`` and ``near_m`` must grow with ``h`` once
    ``h`` is no longer tiny next to the sheath radius, coarsening ``h`` for a bigger edge does NOT keep the
    far grid's wall-clock constant (measured: #697's PR body states the per-edge numbers) -- only its byte
    size is bounded by ``node_budget``; nor does it keep a node's candidate count under the basis's own
    ``segments_max`` (the retry loop below raises it knowingly when a node needs more, never silently).
    Returns ``(path, n_nodes, spacing_m, near_m, seconds)``."""
    from ..spec.walk import _PoolTests
    from ..fields.strand_field import StrandFieldBasis
    pt = _PoolTests(spec)
    src0 = spec.field_source_pools[0].id
    ob, ib = pt.boundary(pt.inside_w[src0]), pt.boundary(pt.outside_w[src0])
    lo, hi = np.asarray(pt.lo), np.asarray(pt.hi)
    edge = float((hi - lo).max())
    h = max(float(spacing_floor_m), edge / float(node_budget) ** (1.0 / 3.0))
    R_out_max = float(np.asarray(ob.radii).max())
    near_m = 2.0 * R_out_max + 4.5 * h               # minimum (2 R + 4h) + half a cell's margin
    blend_m = 2.0 * h
    cutoff_m = 1.3 * near_m
    basis = StrandFieldBasis(ob.centerlines, ib.radii, ob.radii, cutoff_m=cutoff_m, domain=(pt.lo, pt.hi))
    t0 = time.time()
    # the candidate cap per node (StrandFieldBasis's own segments_max, default 4096) is sized for DiSCo's own
    # density; at the spectrum's denser/coarser-spaced 5 mm case a node's near_m sphere holds more segments than
    # that, and the basis refuses by name rather than silently dropping any -- doubling and retrying is the
    # "raise it knowingly" its own docstring asks for, bounded so a real bug still raises.
    segments_max = basis.segments_max
    for _ in range(6):
        try:
            grid = basis.build_far_grid(h, near_m, blend_m=blend_m)
            break
        except ValueError as e:
            if "segments_max" not in str(e) or segments_max >= 1 << 20:
                raise
            segments_max *= 2
            log.info("spectrum.strands: a node exceeded segments_max=%d; retrying at %d", basis.segments_max, segments_max)
            basis = StrandFieldBasis(ob.centerlines, ib.radii, ob.radii, cutoff_m=cutoff_m, domain=(pt.lo, pt.hi),
                                     segments_max=segments_max)
    dt = time.time() - t0
    grid.save(out_path)
    n_nodes = int(np.prod(grid.shape))
    log.info("spectrum.strands: far grid %s nodes at %.2f um spacing (near %.1f um), %.1f s", f"{n_nodes:,}",
            h * 1e6, near_m * 1e6, dt)
    return out_path, n_nodes, h, near_m, dt


def _manifest_for(edge_m, T_max_s, *, variant_dir, substrate_rel, far_grid_rel, grid, counts_rel, blocks_rel,
                  n_tiles, id_, license_, citation, scanner=_DEFAULT_SCANNER, total_walkers, n_blocks):
    from ..replay.bank import mode_count
    commit = _git_commit()
    man = dict(
        id=id_, license=license_, citation=citation,
        code=dict(repo="dmrai-lab/dmipy-sim", commit=commit),
        substrate=dict(kind="disco", tracks=substrate_rel["tracks"], diameters=substrate_rel["diameters"],
                       coordinate_unit_m=25e-6, diameter_unit_m=1e-3, side_m=float(edge_m),
                       far_grid=far_grid_rel),
        variants=dict(field=dict(field=far_grid_rel is not None)), default_variant="field",
        grid=dict(shape=list(grid.shape), voxel_size_m=list(grid.voxel_size_m), origin_m=list(grid.origin_m),
                  attach="substrate"),
        walk=dict(T_max_s=float(T_max_s), dt_save_s=float(T_max_s) / 200.0, scanner=scanner, floor_fraction=0.1,
                  adaptive_steps=True, walker_batch_size=4096, census_draws=200),
        pack=dict(K=mode_count(T_max_s, scanner), K_path=mode_count(T_max_s, scanner), position_container="bands",
                  blt_container="bands", blt_K=mode_count(T_max_s, scanner)),
        plan=dict(file=counts_rel, blocks=blocks_rel, walkers=int(total_walkers), n_blocks=int(n_blocks)),
    )
    path = os.path.join(variant_dir, "manifest.json")
    with open(path, "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    return path


def strands(edge_m, seed, *, out_dir, radii_stats=None, density=None, points_per_mm=DISCO_POINTS_PER_MM,
           T_max_s=(0.020, 0.100), target_floor=PILOT_TARGET_FLOOR, far_grid_node_budget=FAR_GRID_NODE_BUDGET,
           scanner=_DEFAULT_SCANNER):
    """Generate one strands entry of the substrate spectrum (dmrai-lab/dmipy-sim#697): random curved strands at
    DiSCo's own radius and density statistics (:mod:`dmipy_sim.spectrum.disco_stats`, overridable for a
    different class of statistics) in a cube of ``edge_m`` side, tiled in 1 mm columns streamed one at a time.

    Writes under ``out_dir``: ``substrate/strands.tck`` + ``strands_diameters.txt`` (the generated geometry,
    DiSCo's own track/diameter file formats) and ``substrate/spec.json`` (the built
    :class:`~dmipy_sim.spec.SubstrateSpec`, field off -- a spec is cheap and always written; the field-on
    variant lives in each walk length's own manifest, which carries the far grid), then for every ``T_max`` in
    ``T_max_s`` whose halo (:func:`~dmipy_sim.spectrum.common.check_halo`) fits ``edge_m``: a
    ``T<ms>ms/manifest.json`` + ``T<ms>ms/plan/{counts.npz,blocks.json}`` and, once per entry (not per walk
    length -- the field doesn't care how long a walk is), ``substrate/far.npy`` + ``.json``
    (:func:`_build_far_grid`). A walk length the halo refuses is skipped and named in the return value's
    ``refused`` list, never silently (dmipy-sim#697's own example: 25 um refuses both 20 and 100 ms under the
    common 2.0e-9 m^2/s halo policy, :data:`~dmipy_sim.spectrum.common.HALO_FREE_DIFFUSIVITY_M2_S`). Finishes
    with ``spectrum.json`` (sha256 of every file).

    Returns a dict: ``out_dir``, ``n_strands``, ``n_tiles_per_axis``, ``refused`` (list of ``T_max_s`` refused),
    ``variants`` (list of the ``T_max_s`` written), ``far_grid`` (``None`` or its ``(path, n_nodes, spacing_m,
    near_m, seconds)``), ``pilot`` (``n_pilot``, ``floor_pilot``), ``seconds`` (wall time of the whole call)."""
    t_start = time.time()
    radii_stats = dict(radii_stats or DISCO_INNER_RADIUS_STATS_M)
    density = float(density if density is not None else DISCO_AREAL_DENSITY_PER_M2)
    os.makedirs(out_dir, exist_ok=True)
    sub_dir = os.path.join(out_dir, "substrate")
    os.makedirs(sub_dir, exist_ok=True)

    n_pilot = int(np.ceil(1.0 / float(target_floor) ** 2))
    n_pilot, floor_pilot, _pilot_dir = _pilot_walk(seed, radii_stats, density, points_per_mm, out_dir=out_dir,
                                                  n_pilot=n_pilot, scanner=scanner)

    n_side = n_tiles_per_axis(edge_m)
    tile_edge = edge_m / n_side
    tracks = os.path.join(sub_dir, "strands.tck")
    diam = os.path.join(sub_dir, "strands_diameters.txt")
    shape = (int(round(edge_m / VOXEL_SIZE_M)),) * 3
    from ..phantom import Grid
    grid = Grid(shape=shape, voxel_size_m=(VOXEL_SIZE_M,) * 3, origin_m=(VOXEL_SIZE_M / 2.0,) * 3)
    f_inner = np.zeros(grid.n_voxels)
    f_outer = np.zeros(grid.n_voxels)
    n_strands = 0
    blocks = []
    with StreamingTckWriter(tracks, coordinate_unit_m=25e-6) as w, open(diam, "w") as df:
        for ix in range(n_side):
            for iy in range(n_side):
                A, B, Rin, Rout = [], [], [], []
                for centerline, r_in in _generate_column(seed, ix, iy, tile_edge, edge_m, radii_stats=radii_stats,
                                                         density=density, points_per_mm=points_per_mm):
                    w.add(centerline)
                    df.write(f"{2.0 * r_in / 1e-3:.9f}\n")
                    n_strands += 1
                    A.append(centerline[:-1]); B.append(centerline[1:])
                    Rin.append(np.full(len(centerline) - 1, r_in))
                    Rout.append(np.full(len(centerline) - 1, r_in / 0.7))
                if A:
                    A = np.concatenate(A); B = np.concatenate(B)
                    Rin = np.concatenate(Rin); Rout = np.concatenate(Rout)
                    f_inner += _census_from_segments(A, B, Rin, grid)
                    f_outer += _census_from_segments(A, B, Rout, grid)
                vox_per_tile = int(round(tile_edge / VOXEL_SIZE_M))
                blocks.append(dict(block=len(blocks), i=[ix * vox_per_tile, (ix + 1) * vox_per_tile],
                                  j=[iy * vox_per_tile, (iy + 1) * vox_per_tile], k=[0, shape[2]],
                                  seed=tile_seed(seed, ix, iy)))
    if n_strands == 0:
        raise RuntimeError(f"edge {edge_m:g} m, seed {seed}: zero strands drawn; pick a different seed")

    f_inner = np.clip(f_inner, 0.0, 1.0)
    f_outer = np.clip(f_outer, 0.0, 1.0)
    f_extra = np.clip(1.0 - f_outer, 0.0, 1.0)
    seen_intra = f_inner > 0
    seen_extra = f_extra > 0
    count_intra = np.where(seen_intra, n_pilot, 0).astype(np.int64)
    count_extra = np.where(seen_extra, n_pilot, 0).astype(np.int64)
    counts_path = os.path.join(sub_dir, "counts.npz")
    np.savez(counts_path, count_intra=count_intra.reshape(grid.shape), count_extra=count_extra.reshape(grid.shape))
    total_walkers = int(count_intra.sum() + count_extra.sum())
    for b in blocks:
        vox = np.zeros(grid.shape, bool)
        vox[b["i"][0]:b["i"][1], b["j"][0]:b["j"][1], b["k"][0]:b["k"][1]] = True
        b["voxels"] = int(vox.sum())
        b["walkers"] = dict(intra=int(count_intra.reshape(grid.shape)[vox].sum()),
                           extra=int(count_extra.reshape(grid.shape)[vox].sum()))
    blocks_path = os.path.join(sub_dir, "blocks.json")
    with open(blocks_path, "w") as fh:
        json.dump(dict(grid=dict(shape=list(grid.shape)), target_walkers_per_block=total_walkers // max(len(blocks), 1),
                      blocks=blocks), fh, indent=1)

    from ..spec import disco_spec
    spec = disco_spec(tracks, diam, coordinate_unit_m=25e-6, diameter_unit_m=1e-3, side_m=edge_m, field=False)
    spec.save(os.path.join(sub_dir, "spec.json"))

    far_info = None
    refused, variants = [], []
    id_base = f"spectrum/strands/edge_{edge_m:g}m_seed_{seed}"
    license_ = ("synthetic geometry, generated for dmrai-lab/dmipy-sim#697: not DiSCo's own data, only its "
              "released radius/density STATISTICS (CC BY 4.0, Rafael-Patino et al. 2021) are reused")
    citation = ("Rafael-Patino et al., Data in Brief 38 (2021) 107429, doi:10.1016/j.dib.2021.107429 (the "
              "statistics this generator matches); dmrai-lab/dmipy-sim#697, dmrai-lab/tessera#43")
    for T_max in T_max_s:
        try:
            check_halo(edge_m, T_max, what=f"strands edge={edge_m * 1e6:.0f} um")
        except SpectrumHaloError as e:
            log.info("spectrum.strands: %s", e)
            refused.append(float(T_max))
            continue
        if far_info is None:
            spec_field = disco_spec(tracks, diam, coordinate_unit_m=25e-6, diameter_unit_m=1e-3, side_m=edge_m, field=True)
            far_path = os.path.join(sub_dir, "far.npy")
            far_info = _build_far_grid(spec_field, far_path, node_budget=far_grid_node_budget)
        variant_dir = os.path.join(out_dir, f"T{int(round(T_max * 1e3)):03d}ms")
        os.makedirs(variant_dir, exist_ok=True)
        _manifest_for(edge_m, T_max, variant_dir=variant_dir,
                     substrate_rel=dict(tracks="../substrate/strands.tck", diameters="../substrate/strands_diameters.txt"),
                     far_grid_rel="../substrate/far.npy", grid=grid, counts_rel="../substrate/counts.npz",
                     blocks_rel="../substrate/blocks.json", n_tiles=n_side, id_=f"{id_base}/T{T_max}",
                     license_=license_, citation=citation, scanner=scanner, total_walkers=total_walkers,
                     n_blocks=len(blocks))
        variants.append(float(T_max))

    index_path = write_index(out_dir, extra=dict(
        id=id_base, generator="dmipy_sim.spectrum.strands", class_="strands", edge_m=float(edge_m), seed=int(seed),
        n_strands=int(n_strands), n_tiles_per_axis=int(n_side), target_floor=float(target_floor), n_pilot=int(n_pilot),
        floor_pilot=float(floor_pilot), variants_t_max_s=variants, refused_t_max_s=refused,
        far_grid=(None if far_info is None else dict(n_nodes=far_info[1], spacing_m=far_info[2], near_m=far_info[3],
                                                    seconds=far_info[4])),
        seconds=time.time() - t_start))

    return dict(out_dir=out_dir, n_strands=n_strands, n_tiles_per_axis=n_side, refused=refused, variants=variants,
               far_grid=far_info, pilot=dict(n_pilot=n_pilot, floor_pilot=floor_pilot), index=index_path,
               seconds=time.time() - t_start)
