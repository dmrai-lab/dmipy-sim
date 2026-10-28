"""``walk_spec``: the producer that reads a spec and nothing else.

A spec with a single geometry (any analytic family, one mesh surface) is walked by
``simulate_trajectories`` on ``geometry_from_spec``. A multi-surface mesh spec -- a bundle of fibres,
each an inner and an outer closed surface -- is walked pool by pool: the intra pool inside the inner
surfaces, the extra pool outside the outer ones with the domain's faces as walls, a stuck myelin
pool (``D = 0``) frozen where it was seeded; seeding follows the spec's rule (uniform density by
measured volume, weights by water fraction or thinning) and the pools are joined into one
:class:`~dmipy_sim.coefficient_walk.CoefficientWalk` whose compartment channel carries the spec's ids. The field
basis of a field-source pool is computed on the domain grid (or the strands' closed form) and READ by the walk.
"""
import logging

import numpy as np

from ..run import Run, current
import jax.numpy as jnp

from .substrate import SubstrateSpec, SpecError, susceptibility_field_of
from .build import geometry_from_spec


def walk_spec(spec, n_walkers=None, T_max=None, block_h=None, *, scanner="connectom", diffusivity=None,
              seed=0, n_probe=200_000, field_res=None, field_budget=None, field_cutoff_m=25e-6,
              field_cutoff_tol=0.02, field_cutoff_max_m=50e-6, require_gpu=None, walker_batch_size=50_000, tiers="all",
              seeding=None, adaptive_steps=False, field_read_interval=None, K_c3=None, field_far=None, field_gather_every=4,
              run_dir=None, spool=False, context=None, bounce_budget=None, backend="jax", K=None, block_order=None,
              window_T=None, raw_subset=0, envelope=None, field_calibration=None, calibration_floor=None, query_scope=None,
              field_target_floor=None):
    """Walk ``spec`` and return a :class:`~dmipy_sim.coefficient_walk.CoefficientWalk` carrying the spec.

    The walk accumulates the scanner class's band (#643): ``K`` per storage window (``window_T``) the class's
    certified band times the margin (:func:`~dmipy_sim.replay.bank.mode_count`) unless given, the step law resolving
    the same band; its block is the band's own (``block_order``, ``block_h`` as in
    :func:`~dmipy_sim.engine.core.simulate_trajectories`), with adaptive steps too
    (:func:`~dmipy_sim.engine.adaptive.simulate_trajectories_adaptive`: a block a whole number of rounds).

    ``context`` is a :class:`WalkContext` of the spec (its pool tests, boundaries, walking geometries and the
    strand-field basis with its far grid), kept by a producer across the walks of one spec so that none of it is
    built or compiled again; it carries the far grid, so ``field_far`` is not passed with it.

    ``field_far`` is a :class:`~dmipy_sim.fields.strand_field.FarGrid` (or its ``.npy`` path) built for the strand
    substrate: the closed form is then summed over the few strands within its ``near_m`` of a point and the grid read
    beyond, the cutoff being the grid's (no doubling; the grid records what it summed to). A one-off per substrate
    (``StrandFieldBasis.build_far_grid``), the lever that makes a dense substrate's field affordable in the walk.

    ``diffusivity`` is the SINGLE-surface path's reference D: that walk has one diffusivity, the intra pool's when
    the spec declares one, else the seeded pool's, and this argument when the pool declares none or in place of it.
    A multi-surface spec carries a D per pool and walks each pool at its own, so the argument is refused there --
    the D belongs in the pools (:meth:`~dmipy_sim.spec.SubstrateSpec.replace`, or the producer's own argument).

    ``field_gather_every`` is how many blocks the adaptive walk reuses a walker's list of strands in reach (gathered
    with the margin the walker can travel in between; 4 by default): with a far grid the reach is the switch's
    15 um and the gather is the field's main cost, so 16 halves the field's share of the walk.

    ``field_read_interval`` is the interval (s) between the walk's reads of the field, DERIVED per pool from its
    pilot unless given (:func:`~dmipy_sim.engine.accumulate.field_read_of_pilot`: the read series' measured quadratic
    variation; a read is never finer than the walk's step or round), and ``K_c3`` the field channel's band (modes per window), derived on a
    pilot unless given (:func:`~dmipy_sim.replay.bank.derive_field_band`; a fill passes the certifying pack's).

    ``field_target_floor`` is the numerical error budget the derived field band of a gridded field is held to (the
    ``sigma_star`` of the pack the walk is built into; :func:`~dmipy_sim.engine.core.simulate_trajectories`): it
    goes with a gridded field source whose band is derived (``K_c3`` None), and is the budget of the fixed-step
    walk's pilot as ``calibration_floor`` is of the adaptive strand walk's.

    ``field_calibration`` maps every diffusing seeded pool name to its validated
    :class:`~dmipy_sim.engine.calibration.FieldCalibration`. Reuse checks physical, numerical,
    query and backend scope before walking; records never cross pools. ``query_scope`` states
    the scanner-capability/tissue identity, absolute complex-signal target and confidence.
    ``calibration_floor`` sets the numerical field budget independently of walker count.
    These options require adaptive strand fields; see ``docs/field-calibration.md``.

    ``adaptive_steps`` walks every pool whose geometry offers ``wall_scales`` (the curved tubes) with the adaptive
    producer (:func:`dmipy_sim.engine.adaptive.simulate_trajectories_adaptive`): free steps away from every
    wall, the R/6 step of the walker's own nearest tube near one, the same channels accumulated per round; the walk
    records how it stepped (``CoefficientWalk.stepping``).

    ``seeding`` replaces the spec's uniform rule with :class:`~dmipy_sim.spec.seeding.StratifiedByVoxel`: the
    same number of walkers of every seeded pool in each voxel of a grid the pool occupies, weighted by the
    pool's measured volume fraction in that voxel over the count -- what a pack meant for ``Phantom.partition``
    needs, its per-voxel floor set by the certificate rather than by Poisson chance. ``n_walkers`` is then
    what the grid and the counts give, and is not passed.

    The spec decides whether there is a susceptibility field to walk, through its pools alone
    (:func:`~dmipy_sim.spec.substrate.susceptibility_field_of`: some pool magnetic), whatever its surface kinds;
    every walk of such a spec READS its basis in the walk (:func:`field_source_kind`), certified before the walk
    starts, and a magnetic pool whose surfaces no field route computes is refused before anything is walked. A
    strand substrate (sheathed swept polylines) takes the per-segment closed form
    (:class:`~dmipy_sim.fields.strand_field.StrandFieldBasis`): every segment within ``field_cutoff_m`` of a point
    contributes its exact finite-segment field (a strand's own term, within its gate, is its nearest segment's
    cylinder), and the cutoff doubles until the channels at a sample of the walk's start positions change by less
    than ``field_cutoff_tol`` (relative rms) when it doubles again, up to ``field_cutoff_max_m`` -- the certificate
    the basis records, with the change the last doubling still made when the bound stopped it (``converged:
    False``). With ``field_far`` the part beyond the grid's ``near_m`` is read from that grid instead of summed, and
    the cutoff is the grid's. A strand source needs ``adaptive_steps=True`` (the reads are the adaptive producer's)
    and is refused otherwise. Every other field-source substrate rasterises the field basis on the domain grid at the
    node spacing the field source's thinnest shell sets (:func:`~dmipy_sim.fields.susceptibility_field.field_resolution`
    of ``spec.validity.thinnest_shell``: :data:`~dmipy_sim.fields.susceptibility_field.FIELD_NODES_ACROSS` nodes
    across it, measured; ``field_res`` in metres overrides it, and a spec that records no shell needs it) within
    ``field_budget`` nodes (default the host's, :func:`~dmipy_sim.fields.susceptibility_field.field_node_budget`:
    half the memory ceiling at the build's measured bytes per node); a basis beyond the budget is refused, never
    coarsened; the grid is read inside the step (:func:`~dmipy_sim.engine.core.simulate_trajectories` with
    ``field_basis=``). A frozen pool (``D = 0``) holds its seat: its field modes are the channels there (the ``k =
    0`` mode alone).

    ``run_dir`` is where the walk's record goes (:mod:`dmipy_sim.run`; the default root otherwise); with ``spool``
    every finished walker batch is written into it at once, and a call with the same arguments and the same
    ``run_dir`` resumes: the batches found there are read back, the rest walked -- a killed walk costs the batch
    in progress, not the walk.
    """
    with Run("walk_spec", params=dict(spec=getattr(spec, "id", None), n_walkers=n_walkers, T_max=T_max, block_h=block_h,
                                      adaptive_steps=adaptive_steps, seed=seed), run_dir=run_dir) as run:
        import logging
        from ..engine.core import simulate_trajectories
        from ..replay.bank import class_band
        spec = SubstrateSpec.from_dict(spec) if isinstance(spec, dict) else spec
        spec.validate()
        if T_max is None:
            raise TypeError("walk_spec needs T_max (seconds)")
        kind = field_source_kind(spec)
        if any(value is not None for value in (field_calibration, calibration_floor, query_scope)):
            if not adaptive_steps or kind != "strands":
                raise SpecError("scoped field calibration requires an adaptive strand-field walk")
            from collections.abc import Mapping
            from ..engine.calibration import FieldCalibration
            if field_calibration is not None:
                if not isinstance(field_calibration, Mapping):
                    raise TypeError("walk_spec field_calibration maps pool names to FieldCalibration records")
                expected = {p.name for p in spec.pools if p.id in spec.seeding.pools and p.D}
                if set(field_calibration) != expected:
                    raise ValueError(f"field calibration must cover exactly the diffusing seeded pools: {sorted(expected)}")
                if not all(isinstance(c, FieldCalibration) for c in field_calibration.values()):
                    raise TypeError("each pool needs a validated FieldCalibration")
                bands = {c.record["band"]["K"] for c in field_calibration.values()}
                if len(bands) != 1:
                    raise ValueError("pool calibrations need a common field band; regenerate their evidence at that band")
        if field_target_floor is not None:
            if kind != "grid" or adaptive_steps:
                raise SpecError("field_target_floor is the numerical budget of a gridded field's band in the fixed-step walk "
                                "(the adaptive strand walk takes calibration_floor)")
            if K_c3 is not None:
                raise SpecError("field_target_floor is the budget of a field band the walk derives; K_c3 was given")
        if kind == "strands" and not adaptive_steps:
            raise SpecError(f"spec {spec.id!r} has a strand field source, which the adaptive producer reads along the walk: "
                            "pass adaptive_steps=True (a fixed-step walk of a strand substrate carries no field)")
        if context is not None:
            if not isinstance(context, WalkContext):
                raise TypeError(f"context must be a WalkContext, got {type(context).__name__}")
            if field_far is not None:
                raise TypeError("the context carries the far grid: give context= or field_far=, not both")
            context.check(spec)
        if seeding is not None:
            from .seeding import DrawnSeeds, StratifiedByVoxel
            if not isinstance(seeding, (StratifiedByVoxel, DrawnSeeds)):
                raise TypeError(f"seeding must be a StratifiedByVoxel or the DrawnSeeds of one, got {type(seeding).__name__}")
            if n_walkers is not None:
                raise TypeError("n_walkers is what a stratified seeding produces: give seeding= or n_walkers=, not both")
            if not _needs_bundle_walk(spec):
                raise SpecError("stratified seeding is for the pool-by-pool walk of a multi-surface spec")
            n_walkers = (int(seeding.n_walkers) if isinstance(seeding, DrawnSeeds)
                         else int(sum(seeding.count_for(p.name).sum() for p in spec.pools if p.id in spec.seeding.pools)))
        elif n_walkers is None:
            raise TypeError("walk_spec needs n_walkers, or a seeding= that sets the count per voxel")
        if diffusivity is not None and _needs_bundle_walk(spec):
            # one number has nothing to mean for a walk that steps each pool at that pool's own D
            stepped = [spec.pool(i).name for i in spec.seeding.pools if spec.pool(i).D]     # a D = 0 pool is frozen
            stepped = stepped or [spec.pool(i).name for i in spec.seeding.pools]
            raise SpecError(
                f"walk_spec: diffusivity={diffusivity!r} was given for a multi-surface spec, which walks each pool "
                f"at its own D (" + ", ".join(f"{p.name}={p.D!r}" for p in spec.pools) + "). Set the D of the pools "
                f"in the spec -- the producer's own argument, or SubstrateSpec.replace, e.g. "
                f"spec.replace(D={{{', '.join(f'{n!r}: {diffusivity!r}' for n in stepped)}}}) -- rather than beside "
                f"it; a number here reaches the walk's resolution and nothing else.")
        if K is None:                                        # the class's band per storage window (#643)
            from ..engine.accumulate import window_count
            from ..replay.bank import mode_count
            K = mode_count(window_count(T_max, window_T)[1], scanner)
        walk_kw = dict(K=K, block_order=block_order, window_T=window_T, raw_subset=raw_subset, envelope=envelope, bounce_budget=bounce_budget,
                       backend=backend, require_gpu=require_gpu, walker_batch_size=walker_batch_size, f_band=class_band(scanner))
        if not _needs_bundle_walk(spec):
            g = geometry_from_spec(spec)
            gap = g.trajectory_walk_gap()
            if gap is not None:
                raise SpecError(f"walk_spec cannot walk spec {getattr(spec, 'id', None)!r}: its "
                                f"geometry ({type(g).__name__}) has no persistent walk -- {gap}")
            D = diffusivity
            if D is None:                                   # the walk's reference diffusivity: the intra pool's when
                names = {p.name: p for p in spec.pools}     # there is one (multi-pool kernels carry per-pool D), else
                pool = names.get("intra") if names.get("intra") is not None and names["intra"].D is not None \
                    else spec.pool(spec.seeding.pools[0])   # the seeded pool's
                D = pool.D
            if D is None:
                raise SpecError("the spec's seeded pool has no D and no diffusivity= was given")
            fb = None
            if kind == "grid":
                fb = _single_geometry_field_grid(spec, g, field_res, field_budget)
            if adaptive_steps:
                raise SpecError(f"adaptive_steps=True: adaptive stepping is for a bundle of curved tubes, and this spec "
                                f"walks one {type(g).__name__} (fused steps)")
            w = simulate_trajectories(int(n_walkers), float(D), g, T_max=T_max, block_h=block_h, seed=seed, tiers=tiers,
                                      field_basis=fb, **({} if fb is None else dict(K_c3=K_c3, field_read_interval=field_read_interval,
                                                                                   field_target_floor=field_target_floor)),
                                      **walk_kw)
            import dataclasses
            return dataclasses.replace(w, geometry=g, spec=spec, scanner=str(scanner))
        return _walk_bundle(spec, int(n_walkers), float(T_max), (None if block_h is None else float(block_h)), seed, n_probe, field_res,
                            field_budget=field_budget, scanner=scanner, field_cutoff_m=field_cutoff_m, field_cutoff_tol=field_cutoff_tol, seeding=seeding,
                            field_cutoff_max_m=field_cutoff_max_m, adaptive_steps=adaptive_steps, field_read_interval=field_read_interval, K_c3=K_c3,
                            field_far=field_far, field_gather_every=int(field_gather_every), context=context, spool=bool(spool), walk_kw=walk_kw,
                            field_calibration=field_calibration, calibration_floor=calibration_floor, query_scope=query_scope,
                            field_target_floor=field_target_floor)


#: The surface kinds a field raster is made from (:func:`field_grid_of_spec`): their membership tests are exact.
RASTERISED_KINDS = ("mesh", "sphere_union", "swept_polyline")


def field_source_kind(spec):
    """How ``spec``'s susceptibility field is computed for a walk: ``None`` when the substrate has none (no pool is
    magnetic, :func:`~dmipy_sim.spec.substrate.susceptibility_field_of`); ``"strands"`` when the magnetic pool lies
    between two swept-polyline walls (the per-segment closed form,
    :class:`~dmipy_sim.fields.strand_field.StrandFieldBasis`); ``"grid"`` when it is a shell between two cylinder
    walls (the analytic raster, :func:`~dmipy_sim.fields.susceptibility_field.field_grid_of`) or is bounded by
    surfaces of :data:`RASTERISED_KINDS` (the spec's own raster, :func:`field_grid_of_spec`). Any other magnetic
    pool is refused, naming its surfaces: a field the walk cannot compute is never left out of the pack. So is a
    second magnetic pool, since a walk computes the field of one."""
    if susceptibility_field_of(spec) == "absent":
        return None
    src = spec.field_source_pools
    if len(src) > 1:
        raise SpecError(f"spec {spec.id!r}: the pools {[p.name for p in src]} are all magnetic, and a walk computes the "
                        "field of one magnetic pool")
    pid = src[0].id
    outer = [w for w in spec.walls if w.inside_pool == pid]; inner = [w for w in spec.walls if w.outside_pool == pid]
    k_out, k_in = {w.surface.kind for w in outer}, {w.surface.kind for w in inner}
    if outer and inner and k_out == k_in == {"swept_polyline"}:
        return "strands"
    if outer and inner and k_out == k_in == {"cylinder"}:
        return "grid"
    if outer and (k_out | k_in) <= set(RASTERISED_KINDS) and len(k_out) == 1 and len(k_in) <= 1:
        return "grid"
    raise SpecError(
        f"spec {spec.id!r}: pool {src[0].name!r} is magnetic (a susceptibility block) and is "
        + (f"inside {sorted(k_out)} walls" if outer else "inside no wall")
        + (f" and outside {sorted(k_in)} walls" if inner else "")
        + ": no field route computes the field of that geometry -- a walk computes it for a shell between two "
          "swept polylines or two cylinders, or for a pool bounded by "
        + "/".join(RASTERISED_KINDS) + " surfaces. Remove the susceptibility block (the substrate then has no field) "
          "or describe the magnetic pool with surfaces a field route computes")


def _single_geometry_field_grid(spec, g, field_res, field_budget):
    """The :class:`~dmipy_sim.fields.susceptibility_field.FieldGrid` a single-geometry walk of ``spec`` samples: the
    analytic raster of a myelinated cylinder or packed cell (:func:`~dmipy_sim.fields.susceptibility_field.field_grid_of`;
    the cell's grid periodic), else the spec's own raster (:func:`field_grid_of_spec`)."""
    from ..geometry.myelin import MyelinatedCylinder, PackedMyelinatedCylinders
    from ..fields.susceptibility_field import field_grid_of
    if isinstance(g, (MyelinatedCylinder, PackedMyelinatedCylinders)):
        return field_grid_of(g, res=field_res)
    return field_grid_of_spec(spec, field_res=field_res, field_budget=field_budget)


def field_grid_of_spec(spec, *, field_res=None, field_budget=None, context=None, n_check=400_000, seed=0):
    """The rasterised field basis of ``spec``'s field source, a :class:`~dmipy_sim.fields.susceptibility_field.FieldGrid`:
    the one raster a walk reads (:func:`walk_spec`; a mesh/sphere-union/myelinated-cylinder source, or the grid
    cross-check of a strand source). The shell pool's occupancy comes from the
    spec's own membership tests on the domain grid, the directors from its inner surface, at the node spacing the
    thinnest shell sets (:func:`~dmipy_sim.fields.susceptibility_field.field_resolution` of
    ``spec.validity.thinnest_shell``; ``field_res`` in metres overrides it, and a spec that records no shell needs
    it) within ``field_budget`` nodes (default the host's,
    :func:`~dmipy_sim.fields.susceptibility_field.field_node_budget`); a basis beyond the budget is refused
    naming the shell, never coarsened. The raster is checked before it is returned: the shell fraction it holds
    against the fraction of ``n_check`` uniform points the spec's exact membership tests put in the shell, recorded
    in the grid's ``certificate`` and refused past one percent of the shell (a raster that has lost its sheath's
    volume has lost its field). The raster spans the domain box exactly (its nodes the box's side over a whole
    number of spacings) and its field is the box's periodic convolution, so an axis the domain declares periodic is
    one period of the field and the grid states it ``periodic``: a walker that has left the cell reads its image's
    field, never the grid's clamped edge. ``context`` is a :class:`WalkContext` of the spec whose tests are reused."""
    from ..fields.susceptibility_field import (FieldGrid, FIELD_NODES_ACROSS, field_node_budget, field_resolution,
                                               mesh_field_basis, predicate_field_basis)
    if field_source_kind(spec) is None:
        raise SpecError(f"spec {spec.id!r} has no magnetic pool (no pool carries a susceptibility block); there is no "
                        "field to rasterise")
    bounding = {w.surface.kind for w in spec.walls if spec.field_source_pools[0].id in (w.inside_pool, w.outside_pool)}
    if not bounding <= set(RASTERISED_KINDS):
        raise SpecError(f"spec {spec.id!r}: the magnetic pool is bounded by {sorted(bounding)} surfaces, and the spec's "
                        f"own raster is made from {list(RASTERISED_KINDS)} surfaces (a shell between two cylinders is "
                        "rasterised from its geometry, fields.susceptibility_field.field_grid_of)")
    g = context.tests if context is not None else _PoolTests(spec)
    src = spec.field_source_pools[0].id
    outer_b = g.boundary(g.inside_w[src]) if g.inside_w[src] else None
    inner_b = g.boundary(g.outside_w[src]) if g.outside_w[src] else None
    if outer_b is None:
        raise SpecError(f"field-source pool {g.pools[src].name!r} is bounded by no wall; its occupancy cannot be rasterised")
    lo, hi = g.lo, g.hi
    strands = outer_b.kind == "swept_polyline" and inner_b is not None and inner_b.kind == "swept_polyline"
    shell = spec.validity.thinnest_shell
    if field_res is None:
        if shell is None:
            raise SpecError(f"the field basis's node spacing follows the field source's thinnest shell, and spec "
                            f"{spec.id!r} records none (validity.thinnest_shell); pass field_res= (metres) or record it")
        field_res = field_resolution(shell)
        spacing = f"{field_res * 1e6:.3f} um, {FIELD_NODES_ACROSS:g} nodes across its thinnest shell of {shell * 1e6:.3f} um"
    else:
        spacing = f"the given field_res of {field_res * 1e6:.3f} um"
    budget = float(field_budget) if field_budget is not None else field_node_budget()
    n_vox = int(np.prod(np.ceil((hi - lo) / float(field_res))))
    if n_vox > budget:
        raise SpecError(f"the field basis of this substrate would be {n_vox:.2e} nodes at {spacing} over its "
                        f"{np.round((hi - lo) * 1e6, 1).tolist()} um domain, beyond the voxel budget of {budget:.1e} nodes "
                        + ("(field_budget=)" if field_budget is not None else
                           "(half this host's memory ceiling at the build's measured bytes per node)")
                        + ". Raise field_budget= on a host with the memory"
                        + (", or walk a strand source at all: its default is the per-segment closed form, which "
                           "needs no grid" if strands else "")
                        + "; a coarser field_res= is a choice to under-resolve the sheath, never the default")
    if inner_b is not None and inner_b.kind == "mesh" and outer_b.kind == "mesh":
        basis, origin, _ = mesh_field_basis(inner_b.bodies, outer_b.bodies, lo, hi, res=field_res, include_aniso=True)
    else:
        ref_b = inner_b if inner_b is not None else outer_b
        director = ref_b.director if ref_b.kind == "swept_polyline" else None
        basis, origin, _ = predicate_field_basis(inner_b.contains if inner_b is not None else None, outer_b.contains,
                                                 lo, hi, res=field_res, include_aniso=True, director=director)
    frac_raster = float(basis["shell_fraction"])                  # the raster against the membership it was made from
    q = lo + np.random.default_rng(int(seed)).random((int(n_check), 3)) * (hi - lo)
    member = outer_b.contains(q) & ~(inner_b.contains(q) if inner_b is not None else np.zeros(len(q), bool))
    frac = float(member.mean()); se = float(np.sqrt(max(frac * (1.0 - frac), 1e-12) / len(q)))
    cert = dict(res_m=float(field_res), nodes_across_thinnest_shell=(None if shell is None else float(shell / field_res)),
                shape=[int(s) for s in basis["shape"]], shell_fraction_raster=frac_raster, shell_fraction_sampled=frac,
                shell_fraction_se=se, n_sampled=int(n_check))
    if abs(frac_raster - frac) > max(0.01 * frac, 4.0 * se):
        raise SpecError(f"the field raster at {spacing} holds {frac_raster:.4f} of the domain as shell where the spec's "
                        f"membership puts {frac:.4f} +- {se:.4f}: the grid does not resolve the shell")
    side = np.asarray(basis["shape"], float) * np.asarray(basis["voxel_size"], float)
    if not np.allclose(np.asarray(origin, float), lo, rtol=0.0, atol=1e-9 * float(np.max(hi - lo))) or \
            not np.allclose(side, hi - lo, rtol=1e-9, atol=0.0):
        raise SpecError(f"the field raster spans {np.asarray(origin).tolist()} + {side.tolist()}, not the domain box "
                        f"{lo.tolist()} .. {hi.tolist()}: its periodic axes would not be one period")
    return FieldGrid(basis, np.asarray(origin, float), cert, periodic=tuple(bool(p) for p in g.periodic))


def _needs_bundle_walk(spec):
    """A spec whose walls are closed surfaces (meshes, sphere unions, strand packs) with more than one wall or
    more than one seeded pool has no single Geometry: it is walked pool by pool."""
    def bundle_kind(w):
        k = w.surface.kind
        return k in ("mesh", "sphere_union") or (k == "swept_polyline" and bool(w.surface.instances))
    return bool(spec.walls) and all(bundle_kind(w) for w in spec.walls) \
        and (len(spec.walls) > 1 or len(spec.seeding.pools) > 1)


class _Boundary:
    """The closed surfaces of a set of walls as ONE boundary: membership on the host and the Geometry that walks
    the pool inside (``"intra"``) or outside (``"extra"``) it. One surface family per boundary."""

    def __init__(self, walls, backend="jax"):
        kinds = {w.surface.kind for w in walls}
        if len(kinds) != 1:
            raise SpecError(f"walls {[w.name for w in walls]} mix surface kinds {sorted(kinds)}; one pool's boundary "
                            f"must be one kind")
        self.kind = kinds.pop()
        self.walls = walls
        #: which arithmetic a ``swept_polyline`` boundary's `PackedCurvedCylinders` runs its host-side
        #: membership on (`PackedCurvedCylinders.backend`); a mesh or sphere-union boundary is numpy already
        self.backend = backend
        self._geom = {}
        if self.kind == "mesh":
            from ..geometry.mesh import load_ply
            from ..fields.susceptibility_field import MeshBodies
            from .build import mesh_surface_file
            self.bodies = MeshBodies([load_ply(mesh_surface_file(w.surface), scale=(w.surface.scale or 1.0))
                                      for w in walls])
            self.V, self.F = self.bodies.V, self.bodies.F
        elif self.kind == "sphere_union":
            from .build import sphere_union_arrays
            parts = [sphere_union_arrays(w.surface) for w in walls]
            self.centers = np.concatenate([p[0] for p in parts]); self.radii = np.concatenate([p[1] for p in parts])
        else:
            from .build import polyline_arrays
            parts = [polyline_arrays(w.surface) for w in walls]
            self.centerlines = [c for p in parts for c in p[0]]; self.radii = np.concatenate([p[1] for p in parts])

    def segments(self):
        """The swept polylines as segments ``(A, B, r)``."""
        if self.kind != "swept_polyline":
            raise SpecError("segments is the swept-polyline decomposition")
        A = np.vstack([c[:-1] for c in self.centerlines]); B = np.vstack([c[1:] for c in self.centerlines])
        r = np.concatenate([np.full(len(c) - 1, rr) for c, rr in zip(self.centerlines, self.radii)])
        return A, B, r

    def sample_inside(self, n, rng):
        """``n`` points uniform by volume inside the swept polylines (by segment volume, then a disc, then the
        segment's length): the exact intra draw, no rejection. A point sits at least the family's representable
        nudge inside its wall (:func:`~dmipy_sim.geometry._boundary.representable_nudge`), so a float32
        classification at the strands' coordinates reads it as inside."""
        from ..geometry._boundary import representable_nudge
        A, B, r = self.segments()
        margin = representable_nudge(1e-4 * float(r.min()), float(np.abs(np.concatenate([A, B])).max() + r.max()))
        L = np.linalg.norm(B - A, axis=1); w = np.pi * r ** 2 * L
        k = rng.choice(len(w), size=int(n), p=w / w.sum())
        t = rng.uniform(0.0, 1.0, int(n))[:, None]
        C = A[k] + (B[k] - A[k]) * t
        T = (B[k] - A[k]) / np.maximum(L[k], 1e-30)[:, None]
        ref = np.tile([0.0, 0.0, 1.0], (int(n), 1)); ref[np.abs((T * ref).sum(1)) > 0.9] = [1.0, 0.0, 0.0]
        e1 = np.cross(T, ref); e1 /= np.linalg.norm(e1, axis=1, keepdims=True); e2 = np.cross(T, e1)
        rr = np.maximum(r[k] - margin, 0.0) * np.sqrt(rng.uniform(0.0, 1.0, int(n))); th = rng.uniform(0.0, 2 * np.pi, int(n))
        return C + rr[:, None] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)

    def volume(self):
        """The volume of the swept polylines (segments summed; overlaps counted twice)."""
        if self.kind != "swept_polyline":
            raise SpecError("volume is the swept-polyline volume")
        return float(sum(np.pi * rr ** 2 * np.linalg.norm(np.diff(c, axis=0), axis=1).sum()
                         for c, rr in zip(self.centerlines, self.radii)))

    def director(self, pts):
        """The radial director at ``pts`` from the geometry itself, or ``None`` when this surface family has none
        to give (a mesh, a sphere union: the field basis then takes the mask gradient, dmipy-sim#213)."""
        if self.kind != "swept_polyline":
            return None
        return self.geometry("extra", None, None, None, False, None).radial_directors(pts)

    def contains(self, pts):
        pts = np.asarray(pts, float)
        if self.kind == "mesh":
            return self.bodies.contains(pts)
        if self.kind == "sphere_union":
            from ..io.caterpillar import points_inside_union
            return points_inside_union(self.centers, self.radii, pts)
        return np.asarray(self.geometry("extra", None, None, None, False, None).inside_any(pts), bool)

    def geometry(self, pool, lo, hi, periodic, reflect, feature):
        key = (pool, reflect)
        if key not in self._geom:
            if self.kind == "mesh":
                from ..geometry.mesh import Mesh
                g = Mesh(self.V, self.F, periodic=periodic, voxel_min=lo, voxel_max=hi, feature_radius=feature, pool=pool,
                         box_reflect=reflect)
            elif self.kind == "sphere_union":
                from ..geometry.sphere_union import SphereUnion
                g = SphereUnion(self.centers, self.radii, pool=pool, feature_radius=feature,
                                box=((lo, hi) if reflect else None))
            else:
                from ..geometry.curved_cylinder import PackedCurvedCylinders
                g = PackedCurvedCylinders(self.centerlines, self.radii, interior=(pool == "intra"),
                                      box=((lo, hi) if reflect else None), backend=self.backend)
            self._geom[key] = g
        return self._geom[key]


def _strand_field(outer_b, inner_b, lo, hi, traj, cutoff_m, tol, seed, segments_max=4096, cutoff_max=50e-6):
    """The per-segment field basis of a strand substrate (its sheath between ``inner_b`` and ``outer_b``), the
    cutoff doubled until the channels at a sample of the walk's start positions move by less than ``tol``."""
    from ..fields.strand_field import StrandFieldBasis
    if len(outer_b.centerlines) != len(inner_b.centerlines):
        raise SpecError("the sheath's inner and outer walls list different numbers of strands")
    sf = StrandFieldBasis(outer_b.centerlines, inner_b.radii, outer_b.radii, cutoff_m=cutoff_m, domain=(lo, hi),
                          segments_max=segments_max)
    rng = np.random.default_rng(int(seed) + 7)
    sample = traj[rng.choice(traj.shape[0], size=min(traj.shape[0], 2000), replace=False), 0]
    extent = float(np.max(np.asarray(hi) - np.asarray(lo)))
    log = logging.getLogger("dmipy_sim")
    while True:
        err = sf.cutoff_error(sample)
        log.info("walk_spec: strand field cutoff %.0f um -> %.0f um changes the channels by iso %.4f, aniso %.4f (tol %.3f)",
                 sf.cutoff_m * 1e6, 2 * sf.cutoff_m * 1e6, err["iso"], err["aniso"], tol)
        converged = max(err.values()) <= tol
        if converged or 2.0 * sf.cutoff_m > cutoff_max or 2.0 * sf.cutoff_m >= extent:   # the bound, or the whole domain
            break
        sf = sf.with_cutoff(2.0 * sf.cutoff_m)
    if not converged:
        log.warning("walk_spec: the strand field's cutoff stopped at %.0f um (bound %.0f um) with the next doubling still "
                    "changing the channels by iso %.3f, aniso %.3f: the pack records it (converged: False)",
                    sf.cutoff_m * 1e6, cutoff_max * 1e6, err["iso"], err["aniso"])
    return sf.with_cutoff(sf.cutoff_m, certificate=dict(cutoff_error=err, tol=float(tol), n_sample=int(len(sample)),
                                                       cutoff_max_m=float(cutoff_max), converged=bool(converged)))


class _PoolTests:
    """A spec's pools as membership tests: the walls each pool is inside and outside, their boundaries built once,
    the domain box, the water fractions -- what seeding and walking share."""

    def __init__(self, spec, backend="jax"):
        self.pools = {p.id: p for p in spec.pools}
        for w in spec.walls:
            if w.permeability.in_to_out > 0 or w.permeability.out_to_in > 0:
                raise SpecError(f"wall {w.name!r} is permeable; a permeable multi-surface walk is not implemented")
        self.inside_w = {p: [w for w in spec.walls if w.inside_pool == p] for p in self.pools}
        self.outside_w = {p: [w for w in spec.walls if w.outside_pool == p] for p in self.pools}
        self._bounds = {}
        self.lo, self.hi = np.asarray(spec.domain.box_min, float), np.asarray(spec.domain.box_max, float)
        self.periodic = [b == "periodic" for b in spec.domain.boundary]
        self.reflect = "reflect" in spec.domain.boundary
        self.seeded = list(spec.seeding.pools)
        self.wf = {pid: self.pools[pid].water_fraction for pid in self.pools}
        #: ``"numpy"`` for a draw with no JAX, or asked for explicitly (`PackedCurvedCylinders.backend`);
        #: a mesh / sphere-union boundary ignores it, already being plain numpy
        self.backend = backend

    def boundary(self, walls):
        key = tuple(w.name for w in walls)
        if key not in self._bounds:
            self._bounds[key] = _Boundary(walls, backend=self.backend)
        return self._bounds[key]

    def member(self, pid):
        def pred(pts):
            m = np.ones(len(pts), bool)
            if self.inside_w[pid]:
                m &= self.boundary(self.inside_w[pid]).contains(pts)
            if self.outside_w[pid]:
                m &= ~self.boundary(self.outside_w[pid]).contains(pts)
            return m
        return pred

    def by_volume(self, pid):
        """Whether the pool is the inside of swept polylines only: seeds are then drawn by segment volume, exactly."""
        return bool(self.inside_w[pid]) and not self.outside_w[pid] and all(w.surface.kind == "swept_polyline" for w in self.inside_w[pid])

    def sampler(self, pid):
        """``(n, rng) -> (points, accepted)``: the pool's own draw -- inside its swept polylines by segment volume
        (exact, no rejection), else uniform in the box and rejected by membership."""
        pred = self.member(pid); lo, hi = self.lo, self.hi
        if self.by_volume(pid):
            b = self.boundary(self.inside_w[pid])          # strands may leave the box: a draw outside it is rejected
            return lambda n, rng: (lambda P: (P, np.all((P >= lo) & (P <= hi), axis=1)))(b.sample_inside(n, rng))
        return lambda n, rng: (lambda P: (P, pred(P)))(rng.uniform(lo, hi, (n, 3)))


class WalkContext:
    """What a walk of ``spec`` builds before its first step, kept by a producer that walks block after block of one
    spec (the DiSCo fill: 1399 blocks, nothing but the seeds changing): the pool membership tests and boundaries,
    each pool's walking geometry once it is built (its segment tables on the device, its kernels compiled), and
    the strand-field basis with its far grid (the grid on the device once). Keyed by what determines it -- the
    spec's dict and the far grid's sha256 -- so a producer keeps one across blocks and builds another only when
    the key changes; :func:`walk_spec` and :func:`draw_seeds` refuse a context of another key."""

    def __init__(self, spec, *, field_far=None):
        from ..fields.strand_field import FarGrid
        self.spec = spec
        self.far = None if field_far is None else (field_far if isinstance(field_far, FarGrid) else FarGrid.load(field_far))
        self.key = self.key_of(spec, self.far)
        self.tests = _PoolTests(spec)
        self._basis = None

    @staticmethod
    def key_of(spec, far):
        import hashlib, json
        h = hashlib.sha256(json.dumps(spec.to_dict(), sort_keys=True, default=str).encode()).hexdigest()
        return (h, None if far is None else far.meta.get("sha256"))

    def check(self, spec, far=None):
        """Raise unless this context was built for ``spec`` (and for ``far`` when one is named)."""
        if self.key[0] != self.key_of(spec, None)[0]:
            raise ValueError("the walk context was built for another spec")
        if far is not None and self.key[1] != self.key_of(spec, far)[1]:
            raise ValueError("the walk context carries another far grid")

    def field_basis(self, *, starts=None, cutoff_m=25e-6, tol=0.02, seed=0, cutoff_max=50e-6):
        """The strand-field basis of the spec's field source, built once and cached for every later call (it is the
        substrate's, not a block's -- the fill's whole point in keeping this context): with a far grid, directly at
        its cutoff (the particle-mesh split, no doubling); without one, the per-segment closed form of
        :func:`_strand_field`, whose cutoff is doubled against ``starts`` (the FIRST call's own walk start
        positions) until it stops changing by more than ``tol``. ``None`` when the source is not a pair of
        swept-polyline walls, OR when it is and there is no far grid and no ``starts`` yet -- a bare query before
        any walk is answered, not refused; only a caller that actually needs the basis and gives no ``starts``
        (:func:`_walk_bundle`, which always does) would have nothing to build it from, and there is nothing to
        refuse here that walking itself would not also need. Every call after the first ignores ``starts`` --
        the cached basis answers for every block."""
        if self._basis is not None:
            return self._basis
        if field_source_kind(self.spec) != "strands":
            return None
        from ..fields.strand_field import StrandFieldBasis
        g = self.tests; src0 = self.spec.field_source_pools[0].id
        ob, ib = g.boundary(g.inside_w[src0]), g.boundary(g.outside_w[src0])
        if len(ob.centerlines) != len(ib.centerlines):
            raise SpecError("the sheath's inner and outer walls list different numbers of strands")
        if self.far is not None:
            self._basis = StrandFieldBasis(ob.centerlines, ib.radii, ob.radii, cutoff_m=self.far.cutoff_m, domain=(g.lo, g.hi),
                                           certificate=dict(cutoff_m=float(self.far.cutoff_m), far_grid=self.far.meta,
                                                            note="the cutoff the far grid summed to; not doubled here")).with_far(self.far)
        elif starts is not None:
            self._basis = _strand_field(ob, ib, g.lo, g.hi, starts, cutoff_m, tol, seed, cutoff_max=cutoff_max)
        return self._basis


def _explicit_seeds(spec, pid, n, member=None):
    """``(positions, weights)`` for a pool whose spec seeds it from a CITED list of start positions.

    The list is read from ``spec.seeding.positions`` -- resolved and sha256-checked like any other file a spec
    cites -- and consumed by the rule the citation states. ``read: "cyclic"`` is MC/DC's
    ``DynamicsSimulation::initWalkerPosition``: walker ``i`` starts at line ``i mod n``, so a run of more
    walkers than the file has lines repeats it. Every position is checked to be inside the pool it seeds; one
    that is not is refused rather than walked from the wrong side.
    """
    from .build import resolve_surface_file, _sha256
    pos = spec.seeding.positions
    path = resolve_surface_file(pos["file"])
    if pos.get("sha256") and _sha256(path) != pos["sha256"]:
        raise SpecError(f"{path} does not match the sha256 seeding.positions cites")
    fmt = (pos.get("format") or "xyz").lower()
    if fmt != "xyz":
        raise SpecError(f"seeding.positions.format {fmt!r} is not a format this reads; 'xyz' is three numbers "
                        f"per line")
    A = np.loadtxt(path, dtype=np.float64) * float(pos.get("scale") or 1.0)
    if A.ndim != 2 or A.shape[1] != 3:
        raise SpecError(f"{path}: {A.shape} is not (n, 3) start positions")
    read = (pos.get("read") or "cyclic").lower()
    if read != "cyclic":
        raise SpecError(f"seeding.positions.read {read!r} is not a rule this knows; 'cyclic' is i mod n")
    if member is not None:
        # chunked: a membership test is a device kernel compiled for its batch shape, and the file may hold
        # more rows than one batch
        inside = np.concatenate([np.asarray(member(A[i:i + 512]), bool).ravel()
                                 for i in range(0, len(A), 512)])
        if not inside.all():
            bad = np.flatnonzero(~inside)
            raise SpecError(
                f"{path}: {bad.size} of {len(A)} cited start positions are not inside pool {pid} -- lines "
                f"{(bad[:8] + 1).tolist()}{' ...' if bad.size > 8 else ''}, e.g. "
                f"{np.round(A[bad[0]] * 1e6, 4).tolist()} um. A walker started on the wrong side of a wall "
                f"walks a different substrate, so the list is refused rather than used")
    r0 = A[np.arange(int(n)) % len(A)]
    return r0, np.ones(len(r0))


def draw_seeds(spec, seeding, seed, *, context=None, backend="jax"):
    """The stratified seeds of ``spec`` drawn on the CPU: every seeded pool's start positions and weights on
    ``seeding``'s grid, as a :class:`~dmipy_sim.spec.seeding.DrawnSeeds` that :func:`walk_spec` takes in place
    of the :class:`~dmipy_sim.spec.seeding.StratifiedByVoxel` they were drawn from, with the same result to the
    bit; ``context`` is a :class:`WalkContext` of the spec whose pool tests the draw uses. A pool the seeding
    wants nowhere is drawn empty, and the walk leaves it out (a round of a pass may hold none of a sparse pool);
    a pool wanted somewhere that no draw lands in is refused. What a producer draws for its next block while the device walks this one (dmipy-sim#258): the draw of
    a DiSCo block is 20 s of CPU the walk otherwise waits for. Pool ``pid`` is drawn from ``seed + 13 pid``.
    ``backend="numpy"`` draws a strand substrate's extra pool (the one membership test this draw runs,
    :class:`~dmipy_sim.geometry.curved_cylinder.PackedCurvedCylinders`'s `inside_any`) off its numpy classifier
    rather than JAX's -- for a node without it, or asked for explicitly; the seeds are identical either way
    (:meth:`~dmipy_sim.geometry.curved_cylinder.PackedCurvedCylinders.classify_positions_exact_np` agrees with
    the JAX one element for element). No effect on a mesh or sphere-union substrate, already plain numpy here.
    Given a ``context``, its pool tests were already built at the context's own backend and ``backend`` must
    agree (a context built once and drawn from repeatedly carries one backend, like it carries one spec)."""
    from .seeding import DrawnSeeds, StratifiedByVoxel, fill_per_voxel, fill_swept_by_voxel
    from ..run import Run
    if not isinstance(seeding, StratifiedByVoxel):
        raise TypeError(f"draw_seeds draws a StratifiedByVoxel, got {type(seeding).__name__}")
    log = logging.getLogger("dmipy_sim")
    if context is not None:
        context.check(spec)
        if backend != context.tests.backend:
            raise ValueError(f"draw_seeds(backend={backend!r}) does not match the context's "
                             f"backend={context.tests.backend!r}")
    g = context.tests if context is not None else _PoolTests(spec, backend=backend); grid = seeding.grid; positions, weights = {}, {}
    with Run("draw_seeds", params=dict(seed=int(seed), n_voxels=int(grid.n_voxels))) as run:
        for pid in g.seeded:
            name = g.pools[pid].name; s = int(seed) + 13 * pid
            want = seeding.count_for(name)
            log.info("walk_spec: seeding pool %s per voxel (%d wanted)", name, int(want.sum()))
            run.phase(f"seeding {name}", wanted=int(want.sum()))
            if g.by_volume(pid):                           # inside swept polylines: drawn per voxel from the segments
                A_, B_, r_ = g.boundary(g.inside_w[pid]).segments()   # that meet it, the census their clipped volume
                P, v, f, n_drawn = fill_swept_by_voxel(A_, B_, r_, grid, want, seed=s, census_draws=int(seeding.census_draws))
            else:                                          # drawn in each wanted voxel, kept by membership
                P, v, f, trials, n_drawn = fill_per_voxel(g.member(pid), grid, want, trials_max=int(seeding.trials_per_voxel_max),
                                                          census_draws=int(seeding.census_draws), seed=s)
            inb = np.all((P >= g.lo) & (P <= g.hi), axis=1)  # strands may leave the box; the grid may reach beyond it
            P, v = P[inb], v[inb]
            if len(P) == 0 and int(want.sum()) > 0:
                raise SpecError(f"pool {name!r}: no seed landed in it")
            log.info("walk_spec: %d seeds in %d voxels from %d draws", len(P), int(np.bincount(v, minlength=grid.n_voxels).astype(bool).sum()), n_drawn)
            n_have = np.bincount(v, minlength=grid.n_voxels)
            positions[name] = P
            weights[name] = f[v] * g.wf[pid] / np.maximum(n_have[v], 1)  # f_pool,v x water fraction / n_pool,v: volume-correct per voxel
    return DrawnSeeds(positions=positions, weights=weights, grid=grid, seed=int(seed), drawn_from=seeding)


def _frozen_channels(field, p0):
    """The field basis's channels ``(n, n_ch)`` at the frozen walkers' seats ``p0``."""
    p0 = jnp.asarray(np.asarray(p0, np.float32))
    if hasattr(field, "within_device"):
        seg, keep, _ = field.within_device()(p0)
        return np.asarray(field.channels_at_device()(p0, seg, keep)) - field.mean[None, :]
    return np.asarray(field.channels_at_device()(p0))


def _join_pool_walks(parts, weights_of, seed, spec, field, works, scanner):
    """The pools' coefficient walks of a bundle as one :class:`~dmipy_sim.coefficient_walk.CoefficientWalk` in a
    random walker order (any prefix a fair subsample): ``parts`` is ``(pool id, walk or seats)`` per pool, a frozen
    pool (``D = 0``) its seats ``(n, 3)``, held there over every window (its displacement and bands zero, no
    contact, the field at its seat: the ``k = 0`` mode alone, ``sqrt(T) b(seat)``); the compartment is the pool id
    per walker (static: a bundle's walls are impermeable)."""
    import dataclasses
    from ..coefficient_walk import merge_work
    walks = [w for _, w in parts if not isinstance(w, np.ndarray)]
    w0 = walks[0]
    for w in walks[1:]:
        if (w.K, w.m, w.n_windows, w.n_blocks) != (w0.K, w0.m, w0.n_windows, w0.n_blocks) or abs(w.h - w0.h) > 1e-12 * w0.h:
            raise SpecError("the pools of a bundle are walked on one band and block; two walks disagree")
    n_win, Kc = int(w0.n_windows), int(w0.position_coeffs.shape[2])
    T_w = float(w0.T_window)
    surface = all(w.boundary_bands is not None for w in walks)
    sampled = field is not None and all(w.field_coeffs is not None for w in walks)
    w_read = w0
    if sampled:
        K_f, n_ch = int(w0.K_c3), int(w0.field_coeffs.shape[3])
        for w in walks[1:]:
            if int(w.K_c3) != K_f:
                raise SpecError("the pools of a bundle read the field at one band; two walks disagree")
        # every pool reads at its own interval (its pilot's, snapped to a whole number of ITS steps per block): the pack
        # records the coarsest read, and its certificate charges the term at it on every pool's read series
        w_read = max(walks, key=lambda w: float(w.field_read_interval))
    pc, be, bb, ids, wts, fc, qf, qd, qp, qb = [], [], [], [], [], [], [], [], [], []
    for pid, w in parts:
        if isinstance(w, np.ndarray):                                  # a frozen shell: held at its seat
            n = len(w)
            c = np.zeros((n, n_win, Kc, 3), np.float32); c[:, :, 0, :] = w[:, None, :]
            pc.append(c); qp.append(np.zeros((n, n_win, 3), np.float32))
            if surface:
                be.append(np.zeros((n, n_win), np.float32)); bb.append(np.zeros((n, n_win, w0.boundary_bands.shape[2]), np.float32))
                qb.append(np.zeros((n, n_win), np.float32))
            if sampled:
                modes = np.zeros((n, n_win, K_f, n_ch), np.float32)
                modes[:, :, 0, :] = (np.sqrt(T_w) * _frozen_channels(field, w))[:, None, :]
                fc.append(modes); qf.append(np.zeros((n, n_win, n_ch), np.float32))
                qd.append(np.zeros((n, n_win, n_ch), np.float32))            # a frozen walker reads one value
        else:
            n = w.n_walkers
            pc.append(np.asarray(w.position_coeffs, np.float32)); qp.append(np.asarray(w.qv_positions, np.float32))
            if surface:
                be.append(np.asarray(w.boundary_endpoint)); bb.append(np.asarray(w.boundary_bands)); qb.append(np.asarray(w.qv_boundary))
            if sampled:
                fc.append(np.asarray(w.field_coeffs, np.float32)); qf.append(np.asarray(w.qv_field, np.float32))
                qd.append(np.asarray(w.qv_field_increments, np.float32))
        ids.append(np.full(n, pid, np.int8))
        wts.append(np.asarray(weights_of[pid], float))
    n_tot = sum(len(i) for i in ids)
    order = np.random.default_rng(int(seed) + 991).permutation(n_tot)          # any prefix is a fair subsample
    cat = lambda xs: np.concatenate(xs, axis=0)[order]
    wts = cat(wts)
    by_name = {p.name: p for p in spec.pools}
    D_ref = by_name["intra"].D if ("intra" in by_name and by_name["intra"].D) else float(w0.diffusivity)
    block = dict(w0.block)
    # A joined walk is not the first pool's calibration. Preserve each pool's
    # provenance separately, including its own read and band measurements.
    block.pop("calibration_scope", None)
    calibration_pools = {
        spec.pool(pid).name: dict(calibration_scope=w.block["calibration_scope"],
                                  field_read=w.block["field_read"], field_band=w.block["field_band"])
        for pid, w in parts if not isinstance(w, np.ndarray) and "calibration_scope" in w.block}
    if calibration_pools:
        block["calibration_pools"] = calibration_pools
    if sampled:
        # the coarsest pool's read; every pool's own record under ``pools``, and the largest trapezoid constant per
        # channel any pool's pilot charged
        reads = [w.block["field_read"] for w in walks]
        name_of_pool = {p.id: p.name for p in spec.pools}
        pools = {}
        for pid, w in parts:
            if not isinstance(w, np.ndarray):
                r = w.block["field_read"]
                own = (r.get("pools") or {}).get(name_of_pool[pid], {})
                pools[name_of_pool[pid]] = dict(own, dt_f=float(w.field_read_interval), n_read=int(r["n_read"]), rule=r["rule"],
                                                constants=r.get("constants"))
        block["field_read"] = dict(w_read.block["field_read"], pools=pools)
        consts = [r.get("constants") for r in reads if r.get("constants") is not None]
        if consts:
            block["field_read"]["constants"] = [float(x) for x in np.max(np.asarray(consts, np.float64), axis=0)]
    for key in ("k0_residual_m", "k0_residual_relative"):
        block[key] = float(max(w.block.get(key, 0.0) for w in walks))
    name_of = {p.id: p.name for p in spec.pools}
    block["quadratic_variation"] = {name_of[pid]: w.block.get("quadratic_variation") for pid, w in parts
                                    if not isinstance(w, np.ndarray)}
    stepping = {name_of[pid]: w.stepping for pid, w in parts if not isinstance(w, np.ndarray) and w.stepping is not None} or None
    return dataclasses.replace(
        w0, position_coeffs=cat(pc), boundary_endpoint=(cat(be) if surface else None),
        boundary_bands=(cat(bb) if surface else None), compartment=cat(ids), comp_events=None, bound_start=None, bound_events=None,
        field_basis=(field if sampled else None), field_coeffs=(cat(fc) if sampled else None), qv_field=(cat(qf) if sampled else None),
        qv_field_increments=(cat(qd) if sampled else None),
        field_read_interval=(w_read.field_read_interval if sampled else w0.field_read_interval),
        raw_positions=None, raw_boundary=None, raw_compartment=None, raw_bound=None, raw_field=None, qv_positions=cat(qp),
        qv_boundary=(cat(qb) if surface else None), seed=int(seed), diffusivity=D_ref, geometry=None, spec=spec,
        weights=(None if np.allclose(wts, 1.0) else wts), block=block, work=merge_work(works), scanner=str(scanner),
        stepping=(dict(rule="adaptive", pools=stepping) if stepping else None),
        illegal_crossings=int(sum(w.illegal_crossings for w in walks)))


def _walk_bundle(spec, n_walkers, T_max, block_h, seed, n_probe, field_res, *, field_budget=None, scanner=None,
                 field_cutoff_m=25e-6, field_cutoff_tol=0.02, seeding=None, field_cutoff_max_m=50e-6, adaptive_steps=False,
                 field_read_interval=None, K_c3=None, field_far=None, field_gather_every=4, context=None, spool=False, walk_kw=None,
                 field_calibration=None, calibration_floor=None, query_scope=None, field_target_floor=None):
    """Walk a multi-surface spec pool by pool: every seeded pool is defined by the walls it is inside and the walls it
    is outside; a pool with D > 0 walks the interior of its inside-walls (intra, glia) or the exterior of its
    outside-walls (extra); a shell pool at D = 0 (myelin) is frozen where it was seeded; the field basis is
    rasterised from the same membership tests (or the strands' closed form), and READ by every pool's walk at one
    band and interval (the first walked pool's pilot sets them for the rest)."""
    from ..engine.core import simulate_trajectories
    walk_kw = dict(walk_kw or {})
    log = logging.getLogger("dmipy_sim")
    current().phase("context", reused=(context is not None))
    ctx = context if context is not None else WalkContext(spec, field_far=field_far)
    g = ctx.tests
    pools, inside_w, outside_w, boundary, member, sampler = g.pools, g.inside_w, g.outside_w, g.boundary, g.member, g.sampler
    lo, hi, periodic, reflect, seeded, wf = g.lo, g.hi, g.periodic, g.reflect, g.seeded, g.wf
    if seeding is None:
        probe = np.random.default_rng(int(seed) + 99).uniform(lo, hi, (int(n_probe), 3))
        frac = {pid: float(member(pid)(probe).mean()) for pid in seeded}
        if spec.seeding.rule == "explicit":
            return _explicit_seeds(spec, pid, n, member(pid))
        if spec.seeding.weights == "thin":                 # seed by volume x water fraction, every walker weight 1
            mass = {pid: frac[pid] * wf[pid] for pid in seeded}
        else:                                              # seed by volume, carry the water fraction as a weight
            mass = {pid: frac[pid] for pid in seeded}
        tot = sum(mass.values())
        if tot <= 0:
            raise SpecError("no probe point landed in any seeded pool; the domain box does not cover the substrate")
        counts = {pid: max(1, int(round(n_walkers * mass[pid] / tot))) for pid in seeded}

        def seeds(pid, s):
            out, need, draw = [], counts[pid], sampler(pid)
            rng = np.random.default_rng(s)
            while need > 0:
                pts, ok = draw(max(4 * need, 1024), rng)
                keep = pts[ok][:need]
                out.append(keep); need -= len(keep)
            return np.concatenate(out), np.full(len(np.concatenate(out)), 1.0 if spec.seeding.weights == "thin" else wf[pid])
    else:
        from .seeding import DrawnSeeds
        drawn = seeding if isinstance(seeding, DrawnSeeds) else draw_seeds(spec, seeding, seed, context=ctx)

        def seeds(pid, s):
            return drawn.positions[pools[pid].name], drawn.weights[pools[pid].name]
    feature = float(spec.validity.smallest_feature)
    parts, walked = [], None                               # (pid, walk or seats)
    weights_of = {}; seeds_of = {}
    for pid in seeded:
        seeds_of[pid], weights_of[pid] = seeds(pid, seed + 13 * pid)
        if len(seeds_of[pid]) == 0:
            if seeding is None:
                raise SpecError(f"pool {pools[pid].name!r}: no seed landed in it")
            log.info("walk_spec: pool %s has no seed in this walk; not walked", pools[pid].name)
    seeded = tuple(pid for pid in seeded if len(seeds_of[pid]))  # a seeding may hold none of a pool: a round of a pass
    if not seeded:
        raise SpecError("the seeding holds no seed of any pool; nothing to walk")
    # the field basis is built before the walk and read by it: a strand field certified on the start positions (read
    # by the adaptive producer), a gridded one rasterised from the membership tests (read at every step)
    current().phase("field basis")
    kind = field_source_kind(spec)                         # None when no pool is magnetic
    sf = fg = None
    if kind == "strands":
        starts = np.concatenate([np.asarray(seeds_of[pid], np.float32) for pid in seeded])
        # the basis is the substrate's, not this block's -- ctx.field_basis caches it (built from THIS block's
        # starts the first time, with or without a far grid; ignored on every call after)
        sf = ctx.field_basis(starts=starts[:, None, :], cutoff_m=field_cutoff_m, tol=field_cutoff_tol, seed=seed,
                             cutoff_max=field_cutoff_max_m)
    elif kind == "grid":
        if adaptive_steps:
            raise SpecError("adaptive_steps reads a strand field in the walk; this spec's field source is gridded, which "
                            "the fixed-step walk reads at every step (adaptive_steps=False)")
        fg = field_grid_of_spec(spec, field_res=field_res, field_budget=field_budget, context=ctx)
    works = []
    field_kw = dict(K_c3=K_c3, field_read_interval=field_read_interval)
    fixed_kw = dict(field_kw, field_target_floor=field_target_floor)       # the fixed-step walk's pilot also takes the band's budget
    for pid in seeded:
        pool = pools[pid]
        shell = bool(inside_w[pid]) and bool(outside_w[pid])
        r0 = seeds_of[pid]
        n = len(r0)
        if pool.D in (None, 0.0):
            if not shell:
                raise SpecError(f"pool {pool.name!r} needs D to be walked")
            parts.append((pid, r0.astype(np.float32)))             # frozen shell
            continue
        if shell:
            raise SpecError(f"pool {pool.name!r} diffuses between two surfaces; a diffusing shell pool is not "
                            f"implemented (set D = 0 for a stuck pool)")
        geo = (boundary(inside_w[pid]).geometry("intra", lo, hi, periodic, reflect, feature) if inside_w[pid]
               else boundary(outside_w[pid]).geometry("extra", lo, hi, periodic, reflect, feature))
        log.info("walk_spec: walking pool %s, %d walkers%s", pool.name, n, " (adaptive steps)" if adaptive_steps else "")
        current().phase(f"walk {pool.name}", n_walkers=int(n), adaptive_steps=bool(adaptive_steps))
        if adaptive_steps:
            if not hasattr(geo, "wall_scales"):
                raise SpecError(f"pool {pool.name!r}: its {type(geo).__name__} offers no wall_scales; adaptive stepping "
                                f"is for the curved tubes")
            from ..engine.adaptive import simulate_trajectories_adaptive
            w = simulate_trajectories_adaptive(n, float(pool.D), geo, T_max, block_h, seed=seed + 13 * pid, r0=r0, spec=spec,
                                               field_basis=sf, field_reuse_intervals=int(field_gather_every),
                                               spool=(pool.name if spool else None),
                                               **({} if sf is None else dict(field_kw,
                                                    field_calibration=(None if field_calibration is None else field_calibration[pool.name]),
                                                    calibration_floor=calibration_floor, query_scope=query_scope)),
                                               **{k: v for k, v in walk_kw.items() if k != "bounce_budget"})
        else:
            w = simulate_trajectories(n, float(pool.D), geo, T_max=T_max, block_h=block_h, seed=seed + 13 * pid, r0=r0,
                                      field_basis=fg, **({} if fg is None else fixed_kw), **walk_kw)
        if w.field_coeffs is not None and field_kw.get("K_c3") is None:
            # the first walked pool's pilot sets the band for the pools after it (one channel, one band); every pool's
            # own pilot sets its read
            field_kw = dict(field_kw, K_c3=int(w.K_c3))
            fixed_kw = dict(field_kw, field_target_floor=None)
        walked = w
        works.append(w.work)
        parts.append((pid, w))
    if walked is None:
        raise SpecError("no seeded pool diffuses; nothing to walk")
    current().phase("finalize")
    walk = _join_pool_walks(parts, weights_of, seed, spec, sf if sf is not None else fg, works, scanner)
    object.__setattr__(walk, "run", current())
    return walk
