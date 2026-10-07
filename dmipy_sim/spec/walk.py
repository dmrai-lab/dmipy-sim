"""``walk_spec``: the producer that reads a spec and nothing else.

A spec with a single geometry (any analytic family, one mesh surface) is walked by
``simulate_trajectories`` on ``geometry_from_spec``. A multi-surface mesh spec -- a bundle of fibres,
each an inner and an outer closed surface -- is walked pool by pool: the intra pool inside the inner
surfaces, the extra pool outside the outer ones with the domain's faces as walls, a stuck myelin
pool (``D = 0``) frozen where it was seeded; seeding follows the spec's rule (uniform density by
measured volume, weights by water fraction or thinning) and the pools are concatenated into one
``PersistentWalk`` whose compartment channel carries the spec's ids. The field basis of a
field-source pool is computed on the domain grid and rides on the walk. This is what
the bespoke bundle builders used to decide in code.
"""
import logging

import numpy as np

from ..run import Run, current
import jax.numpy as jnp

from .substrate import SubstrateSpec, SpecError, susceptibility_field_of
from .build import geometry_from_spec


def walk_spec(spec, n_walkers=None, T_max=None, dt_save=None, *, scanner="connectom", floor_fraction=0.1, diffusivity=None,
              seed=0, n_probe=200_000, field_res=None, field_budget=None, field_cutoff_m=25e-6,
              field_cutoff_tol=0.02, field_cutoff_max_m=50e-6, require_gpu=None, walker_batch_size=50_000, tiers="all",
              seeding=None, adaptive_steps=False, field_sample_every=1, field_far=None, field_gather_every=4, run_dir=None,
              spool=False, context=None, defer_field=False, bounce_budget=None, backend="jax"):
    """Walk ``spec`` and return a :class:`~dmipy_sim.persistent_walk.PersistentWalk` carrying the spec.

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

    ``field_gather_every`` is how many save intervals the walk reuses a walker's list of strands in reach (gathered
    with the margin the walker can travel in between; 4 by default): with a far grid the reach is the switch's
    15 um and the gather is the field's main cost, so 16 halves the field's share of the walk.

    ``field_sample_every`` reads the field along the walk at every that-many-th save interval only: the path
    channel keeps a few modes over the walk and lives on its own grid, recorded on the walk and in the pack; the
    positions keep the save grid the envelope asks for.

    ``adaptive_steps`` walks every pool whose geometry offers ``wall_scales`` (the curved tubes) with the adaptive
    producer (:func:`dmipy_sim.engine.adaptive.simulate_trajectories_adaptive`): free steps away from every
    wall, the R/6 step of the walker's own nearest tube near one, the same channels; the walk records how it
    stepped (``PersistentWalk.stepping``).

    ``seeding`` replaces the spec's uniform rule with :class:`~dmipy_sim.spec.seeding.StratifiedByVoxel`: the
    same number of walkers of every seeded pool in each voxel of a grid the pool occupies, weighted by the
    pool's measured volume fraction in that voxel over the count -- what a pack meant for ``Phantom.partition``
    needs, its per-voxel floor set by the certificate rather than by Poisson chance. ``n_walkers`` is then
    what the grid and the counts give, and is not passed.

    The spec decides whether there is a susceptibility field to walk, through its pools alone
    (:func:`~dmipy_sim.spec.substrate.susceptibility_field_of`: some pool magnetic), whatever its surface kinds;
    every walk of such a spec carries its basis, certified before the walk starts, and a magnetic pool whose
    surfaces no field route computes is refused before anything is walked (:func:`field_source_kind`). A strand
    substrate (sheathed swept polylines) takes the per-segment closed form (:class:`~dmipy_sim.fields.strand_field.StrandFieldBasis`): every segment within ``field_cutoff_m`` of a
    point contributes its exact finite-segment field (a strand's own term, within its gate, is its nearest
    segment's cylinder), and the cutoff doubles until the channels at a sample of the walk's start positions
    change by less than ``field_cutoff_tol`` (relative rms) when it doubles again, up to
    ``field_cutoff_max_m`` -- the certificate the basis records, with the change the last doubling still made
    when the bound stopped it (``converged: False``). With ``field_far`` the part beyond the grid's ``near_m``
    is read from that grid instead of summed, and the cutoff is the grid's. Every other field-source substrate
    rasterises the field basis on the domain grid at the node spacing the field source's thinnest shell sets
    (:func:`~dmipy_sim.fields.susceptibility_field.field_resolution` of ``spec.validity.thinnest_shell``:
    :data:`~dmipy_sim.fields.susceptibility_field.FIELD_NODES_ACROSS` nodes across it, measured; ``field_res`` in
    metres overrides it, and a spec that records no shell needs it) within ``field_budget`` nodes (default the
    host's, :func:`~dmipy_sim.fields.susceptibility_field.field_node_budget`: half the memory ceiling at the
    build's measured bytes per node); a basis beyond the budget is refused, never coarsened.

    The basis is SAMPLED along the walk at the sub-step (``PersistentWalk.field_samples``: the mean over every save
    interval of the channels at the walker's sub-step positions, :func:`field_source_kind`): a gridded source (a
    myelinated cylinder or packed cell, rasterised by :func:`~dmipy_sim.fields.susceptibility_field.field_grid_of`,
    periodic for a cell; a mesh or sphere-union sheath by :func:`field_grid_of_spec`) by every walk, the grid read
    inside the step (:func:`~dmipy_sim.engine.core.simulate_trajectories` with ``field_basis=``); a strand source
    with ``adaptive_steps=True`` (:func:`~dmipy_sim.engine.adaptive.simulate_trajectories_adaptive`, at
    ``field_gather_every``). A strand source walked without adaptive steps carries the basis and no
    ``field_samples``, and :func:`~dmipy_sim.replay.bank.build_replay_pack` refuses to pack it until the samples
    exist. A frozen pool (``D = 0``) holds its seat: its samples are the channels there, at every save.
    ``defer_field=True`` records that obligation on the walk (``PersistentWalk.field_deferred``) instead
    of building the basis now, for a producer that means to fill it later from a basis of its own
    (:func:`fill_field`, e.g. the CACTUS route: one :class:`~dmipy_sim.fields.susceptibility_field.FieldGrid`
    per substrate, sampled on every block's kept walk). Deferring is refused for a spec with no field source
    (there is nothing to defer) and unless ``run_dir`` is also given: the quadrature ``fill_field`` takes needs
    the raw walk kept, and ``run_dir`` is this producer's persistence (above).

    ``run_dir`` is where the walk's record goes (:mod:`dmipy_sim.run`; the default root otherwise); with ``spool``
    every finished walker batch is written into it at once, and a call with the same arguments and the same
    ``run_dir`` resumes: the batches found there are read back, the rest walked -- a killed walk costs the batch
    in progress, not the walk.
    """
    with Run("walk_spec", params=dict(spec=getattr(spec, "id", None), n_walkers=n_walkers, T_max=T_max, dt_save=dt_save,
                                      defer_field=defer_field, adaptive_steps=adaptive_steps, seed=seed), run_dir=run_dir) as run:
        import logging
        from ..engine.core import simulate_trajectories
        from ..persistent_walk import PersistentWalk, merge_work
        from ..acquisition.scanners import save_interval
        spec = SubstrateSpec.from_dict(spec) if isinstance(spec, dict) else spec
        spec.validate()
        if T_max is None:
            raise TypeError("walk_spec needs T_max (seconds)")
        has_field = susceptibility_field_of(spec) == "present"      # the one decision: some pool is magnetic
        kind = None if defer_field else field_source_kind(spec)
        in_walk = kind == "grid" or (kind == "strands" and adaptive_steps)
        if int(field_sample_every) != 1 and not in_walk:
            raise ValueError("field_sample_every is the save grid of a field sampled in the walk: a gridded source is, a "
                             "strand source with adaptive_steps=True")
        if defer_field:
            if not has_field:
                raise SpecError(f"defer_field=True was given for spec {spec.id!r}, which declares no susceptibility "
                                "source (no pool sets a susceptibility): there is no field tier to defer")
            if adaptive_steps:
                raise TypeError("defer_field=True defers the field tier to fill_field; it cannot be combined with "
                                "adaptive_steps=True, which samples the field in the walk")
            if run_dir is None:
                raise TypeError("defer_field=True needs the raw walk kept for fill_field to read later: give run_dir= "
                                "(walk_spec persists the walk's record there), or sample the field in the walk "
                                "instead of deferring it")
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
                f"it; a number here reaches the save-grid rule and nothing else.")
        if dt_save is None:
            Ds = [p.D for p in spec.pools if p.D] + ([float(diffusivity)] if diffusivity else [])
            dt_save = save_interval(T_max, n_walkers, scanner, D=(max(Ds) if Ds else 2e-9), floor_fraction=floor_fraction,
                                    field=has_field)
            logging.getLogger("dmipy_sim").info("walk_spec: dt_save=%.3g s derived for %s over T_max=%.3g s with %d walkers "
                                                "(n_t=%d)", dt_save, scanner, T_max, n_walkers, int(round(T_max / dt_save)) + 1)
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
            w = simulate_trajectories(int(n_walkers), float(D), g, T_max=T_max, dt_save=dt_save, seed=seed,
                                      require_gpu=require_gpu, walker_batch_size=walker_batch_size, tiers=tiers,
                                      field_basis=fb, field_sample_every=(int(field_sample_every) if fb is not None else 1),
                                      bounce_budget=bounce_budget, backend=backend)
            return PersistentWalk(w.positions, w.dt, w.sub_steps, w.dt_sim, w.boundary_local_time, w.compartment,
                                  w.bound_frac, w.illegal_crossings, w.seed, w.diffusivity, geometry=g, spec=spec, run=w.run,
                                  field_basis=w.field_basis, field_samples=w.field_samples,
                                  field_sample_every=w.field_sample_every, field_deferred=bool(defer_field), work=w.work, scanner=str(scanner))
        return _walk_bundle(spec, int(n_walkers), float(T_max), float(dt_save), seed, n_probe, field_res,
                            require_gpu, walker_batch_size, field_budget=field_budget, scanner=scanner, field_cutoff_m=field_cutoff_m, field_cutoff_tol=field_cutoff_tol, seeding=seeding,
                            field_cutoff_max_m=field_cutoff_max_m, adaptive_steps=adaptive_steps, field_sample_every=int(field_sample_every), field_far=field_far, field_gather_every=int(field_gather_every), context=context, spool=bool(spool), defer_field=bool(defer_field), bounce_budget=bounce_budget, backend=backend)


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


def fill_field(walk, basis):
    """``walk`` with ``basis``'s channels read at the walk's own saved positions -- the deferred field tier's
    own step (:func:`walk_spec` with ``defer_field=True``), or any basis given after the walk was kept (the
    CACTUS route: one :class:`~dmipy_sim.fields.susceptibility_field.FieldGrid` per substrate, sampled on every
    block's kept walk at pack time).

    ``basis`` answers ``channels(points)`` (a :class:`~dmipy_sim.fields.susceptibility_field.FieldGrid` or a
    :class:`~dmipy_sim.fields.strand_field.StrandFieldBasis`; a loaded
    :class:`~dmipy_sim.fields.strand_field.StrandFieldRecord` evaluates nothing, so it cannot fill a walk).
    This reads it at every saved position -- ``field_sample_every=1``, since the walk's own save grid is all
    there is to read: a QUADRATURE of the true sub-step interval mean an in-walk sampling would have taken, at
    the save grid's resolution rather than the physics step's. The returned walk carries ``field_basis=basis``,
    the samples, and ``field_fill`` -- the record that the tier was read this way, with ``basis``'s own
    certificate -- which :func:`~dmipy_sim.replay.bank.build_replay_pack` carries into the susceptibility
    channel's metadata as ``sampling: "read_at_save_resolution"``.
    """
    import dataclasses
    if walk.field_samples is not None:
        raise ValueError("fill_field: the walk already carries field samples (sampled in the walk at the sub-step); "
                         "there is nothing to fill")
    pts = np.asarray(walk.positions, np.float64).reshape(-1, 3)
    ch = np.asarray(basis.channels(pts), np.float64)
    samples = ch.reshape(walk.n_walkers, walk.n_t, ch.shape[-1]).astype(np.float32)
    cert = getattr(basis, "certificate", None)
    if cert is None:
        cert = getattr(basis, "meta", {}) or {}
    record = dict(read="save_resolution", note="the susceptibility tier was read along the kept walk's saved "
                 "positions from a basis given after the walk, not sampled in the walk", basis_certificate=cert)
    return dataclasses.replace(walk, field_basis=basis, field_samples=samples, field_sample_every=1, field_fill=record)


def field_grid_of_spec(spec, *, field_res=None, field_budget=None, context=None, n_check=400_000, seed=0):
    """The rasterised field basis of ``spec``'s field source, a :class:`~dmipy_sim.fields.susceptibility_field.FieldGrid`:
    the one raster a walk records (:func:`walk_spec`), or a basis given to a kept walk's :func:`fill_field` (the
    deferred route: a mesh/sphere-union/myelinated-cylinder source, or the grid cross-check of a strand source).
    The shell pool's occupancy comes from the
    spec's own membership tests on the domain grid, the directors from its inner surface, at the node spacing the
    thinnest shell sets (:func:`~dmipy_sim.fields.susceptibility_field.field_resolution` of
    ``spec.validity.thinnest_shell``; ``field_res`` in metres overrides it, and a spec that records no shell needs
    it) within ``field_budget`` nodes (default the host's,
    :func:`~dmipy_sim.fields.susceptibility_field.field_node_budget`); a basis beyond the budget is refused
    naming the shell, never coarsened. The raster is checked before it is returned: the shell fraction it holds
    against the fraction of ``n_check`` uniform points the spec's exact membership tests put in the shell, recorded
    in the grid's ``certificate`` and refused past one percent of the shell (a raster that has lost its sheath's
    volume has lost its field). ``context`` is a :class:`WalkContext` of the spec whose tests are reused."""
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
    return FieldGrid(basis, np.asarray(origin, float), cert)


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

    def __init__(self, walls):
        kinds = {w.surface.kind for w in walls}
        if len(kinds) != 1:
            raise SpecError(f"walls {[w.name for w in walls]} mix surface kinds {sorted(kinds)}; one pool's boundary "
                            f"must be one kind")
        self.kind = kinds.pop()
        self.walls = walls
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
                                      box=((lo, hi) if reflect else None))
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

    def __init__(self, spec):
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

    def boundary(self, walls):
        key = tuple(w.name for w in walls)
        if key not in self._bounds:
            self._bounds[key] = _Boundary(walls)
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

    def field_basis(self):
        """The strand-field basis of the spec's field source with the far grid (the particle-mesh split), built once;
        ``None`` when the context has no far grid or the source is not a pair of swept-polyline walls."""
        if self._basis is None and self.far is not None and field_source_kind(self.spec) == "strands":
            from ..fields.strand_field import StrandFieldBasis
            g = self.tests; src0 = self.spec.field_source_pools[0].id
            ob, ib = g.boundary(g.inside_w[src0]), g.boundary(g.outside_w[src0])
            if len(ob.centerlines) != len(ib.centerlines):
                raise SpecError("the sheath's inner and outer walls list different numbers of strands")
            self._basis = StrandFieldBasis(ob.centerlines, ib.radii, ob.radii, cutoff_m=self.far.cutoff_m, domain=(g.lo, g.hi),
                                           certificate=dict(cutoff_m=float(self.far.cutoff_m), far_grid=self.far.meta,
                                                            note="the cutoff the far grid summed to; not doubled here")).with_far(self.far)
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


def draw_seeds(spec, seeding, seed, *, context=None):
    """The stratified seeds of ``spec`` drawn on the CPU: every seeded pool's start positions and weights on
    ``seeding``'s grid, as a :class:`~dmipy_sim.spec.seeding.DrawnSeeds` that :func:`walk_spec` takes in place
    of the :class:`~dmipy_sim.spec.seeding.StratifiedByVoxel` they were drawn from, with the same result to the
    bit; ``context`` is a :class:`WalkContext` of the spec whose pool tests the draw uses. A pool the seeding
    wants nowhere is drawn empty, and the walk leaves it out (a round of a pass may hold none of a sparse pool);
    a pool wanted somewhere that no draw lands in is refused. What a producer draws for its next block while the device walks this one (dmipy-sim#258): the draw of
    a DiSCo block is 20 s of CPU the walk otherwise waits for. Pool ``pid`` is drawn from ``seed + 13 pid``."""
    from .seeding import DrawnSeeds, StratifiedByVoxel, fill_per_voxel, fill_swept_by_voxel
    from ..run import Run
    if not isinstance(seeding, StratifiedByVoxel):
        raise TypeError(f"draw_seeds draws a StratifiedByVoxel, got {type(seeding).__name__}")
    log = logging.getLogger("dmipy_sim")
    if context is not None:
        context.check(spec)
    g = context.tests if context is not None else _PoolTests(spec); grid = seeding.grid; positions, weights = {}, {}
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


def _walk_bundle(spec, n_walkers, T_max, dt_save, seed, n_probe, field_res, require_gpu, batch, field_budget=None, scanner=None, bounce_budget=None, backend="jax",
                 field_cutoff_m=25e-6, field_cutoff_tol=0.02, seeding=None, field_cutoff_max_m=50e-6, adaptive_steps=False, field_sample_every=1, field_far=None, field_gather_every=4, context=None, spool=False, defer_field=False):
    """Walk a multi-surface spec pool by pool: every seeded pool is defined by the walls it is inside and the walls it
    is outside; a pool with D > 0 walks the interior of its inside-walls (intra, glia) or the exterior of its
    outside-walls (extra); a shell pool at D = 0 (myelin) is frozen where it was seeded; the field basis is
    rasterised from the same membership tests."""
    from ..engine.core import simulate_trajectories
    from ..persistent_walk import PersistentWalk, merge_work
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
    parts, n_t, walked = [], None, None                    # (pid, positions or seeds, local time or None)
    weights_of = {}; stepping = []; seeds_of = {}
    for pid in seeded:
        seeds_of[pid], weights_of[pid] = seeds(pid, seed + 13 * pid)
        if len(seeds_of[pid]) == 0:
            if seeding is None:
                raise SpecError(f"pool {pools[pid].name!r}: no seed landed in it")
            log.info("walk_spec: pool %s has no seed in this walk; not walked", pools[pid].name)
    seeded = tuple(pid for pid in seeded if len(seeds_of[pid]))  # a seeding may hold none of a pool: a round of a pass
    if not seeded:
        raise SpecError("the seeding holds no seed of any pool; nothing to walk")
    # the field basis is built before the walk and sampled by it: a strand field certified on the start positions
    # (sampled with adaptive steps), a gridded one rasterised from the membership tests (sampled at every sub-step);
    # defer_field skips this -- the obligation is recorded on the walk, a basis given later fills it
    current().phase("field basis")
    kind = None if defer_field else field_source_kind(spec)                # None when no pool is magnetic
    sf = fg = None
    if kind == "strands":
        src0 = spec.field_source_pools[0].id
        ob, ib = boundary(inside_w[src0]), boundary(outside_w[src0])
        starts = np.concatenate([np.asarray(seeds_of[pid], np.float32) for pid in seeded])
        if ctx.far is not None:                                          # the split: the grid's cutoff, no doubling
            sf = ctx.field_basis()
        else:
            sf = _strand_field(ob, ib, lo, hi, starts[:, None, :], field_cutoff_m, field_cutoff_tol, seed, cutoff_max=field_cutoff_max_m)
    elif kind == "grid":
        if adaptive_steps:
            raise SpecError("adaptive_steps samples a strand field in the walk; this spec's field source is gridded, which "
                            "the fixed-step walk samples at every sub-step (adaptive_steps=False)")
        fg = field_grid_of_spec(spec, field_res=field_res, field_budget=field_budget, context=ctx)
    works = []
    field_samples = []
    for pid in seeded:
        pool = pools[pid]
        shell = bool(inside_w[pid]) and bool(outside_w[pid])
        r0 = seeds_of[pid]
        n = len(r0)
        if pool.D in (None, 0.0):
            if not shell:
                raise SpecError(f"pool {pool.name!r} needs D to be walked")
            parts.append((pid, r0.astype(np.float32), None))      # frozen shell
            continue
        if shell:
            raise SpecError(f"pool {pool.name!r} diffuses between two surfaces; a diffusing shell pool is not "
                            f"implemented (set D = 0 for a stuck pool)")
        g = (boundary(inside_w[pid]).geometry("intra", lo, hi, periodic, reflect, feature) if inside_w[pid]
             else boundary(outside_w[pid]).geometry("extra", lo, hi, periodic, reflect, feature))
        log.info("walk_spec: walking pool %s, %d walkers%s", pool.name, n, " (adaptive steps)" if adaptive_steps else "")
        current().phase(f"walk {pool.name}", n_walkers=int(n), adaptive_steps=bool(adaptive_steps))
        if adaptive_steps:
            if not hasattr(g, "wall_scales"):
                raise SpecError(f"pool {pool.name!r}: its {type(g).__name__} offers no wall_scales; adaptive stepping "
                                f"is for the curved tubes")
            from ..engine.adaptive import simulate_trajectories_adaptive
            w = simulate_trajectories_adaptive(n, float(pool.D), g, T_max, dt_save, seed=seed + 13 * pid, r0=r0, spec=spec,
                                               require_gpu=require_gpu, walker_batch_size=batch, field_basis=sf, field_sample_every=int(field_sample_every), field_reuse_intervals=int(field_gather_every),
                                               spool=(pool.name if spool else None), backend=backend)
            stepping.append((pool.name, w.stepping))
            if w.field_samples is not None:
                field_samples.append((pid, w.field_samples))
        else:
            w = simulate_trajectories(n, float(pool.D), g, T_max=T_max, dt_save=dt_save, seed=seed + 13 * pid, r0=r0,
                                      require_gpu=require_gpu, walker_batch_size=batch, field_basis=fg,
                                      field_sample_every=(int(field_sample_every) if fg is not None else 1),
                                      bounce_budget=bounce_budget, backend=backend)
            if w.field_samples is not None:
                field_samples.append((pid, w.field_samples))
        n_t, walked = w.n_t, w
        works.append(w.work)
        parts.append((pid, np.asarray(w.positions, np.float32),
                      None if w.boundary_local_time is None else np.asarray(w.boundary_local_time, np.float32)))
    if walked is None:
        raise SpecError("no seeded pool diffuses; nothing to walk")
    current().phase("finalize")
    surface = all(dl is not None for pid, pos, dl in parts if pos.ndim == 3)      # every walked pool records contact
    sampled = {pid: arr for pid, arr in field_samples}
    n_tf = len(range(0, n_t, int(field_sample_every)))                     # the field's own save grid
    sizes = [len(pos) for _, pos, _ in parts]
    n_tot = sum(sizes)
    order = np.random.default_rng(int(seed) + 991).permutation(n_tot)      # any prefix is a fair subsample
    dest_of = np.argsort(order)         # order's inverse: concat-row g belongs at output row dest_of[g], exactly
    # what `concatenate(...)[order]` would have put there -- scattered straight into the pre-sized arrays below,
    # one write per array instead of a concatenate and a second, cache-hostile gather copy (7+ GB moved twice over
    # DiSCo's walk, and the dominant host cost of the walk stage at scale: dmipy-sim-cuda#33)
    traj = np.empty((n_tot, n_t, 3), np.float32)
    dlog = np.empty((n_tot, n_t), np.float32) if surface else None
    ids = np.empty(n_tot, np.int8)
    wts = np.empty(n_tot, float)
    samples = np.empty((n_tot, n_tf, 13), np.float32) if sampled else None
    start = 0
    for pid, pos, dl in parts:
        n = len(pos); dest = dest_of[start:start + n]; start += n
        if samples is not None:
            if pid in sampled:
                samples[dest] = sampled[pid]
            else:                                                            # a frozen shell: its start's channels, constant
                p0 = jnp.asarray(np.asarray(pos if pos.ndim == 2 else pos[:, 0], np.float32))
                if sf is not None:
                    seg, keep, _ = sf.within_device()(p0)
                    c0 = np.asarray(sf.channels_at_device()(p0, seg, keep)) - sf.mean[None, :]
                else:
                    c0 = np.asarray(fg.channels_at_device()(p0))
                samples[dest] = c0[:, None, :].astype(np.float32)             # broadcast over the field's save grid
        if pos.ndim == 2:                                                        # a frozen shell: no path, no contact
            traj[dest] = pos[:, None, :]                                         # broadcast over every save
            if dlog is not None:
                dlog[dest] = 0.0
        else:
            traj[dest] = pos
            if dlog is not None:
                dlog[dest] = dl if dl is not None else 0.0
        ids[dest] = pid
        wts[dest] = np.asarray(weights_of[pid], float)
    comp = np.repeat(ids[:, None], n_t, axis=1)
    by_name = {p.name: p for p in spec.pools}
    D_ref = by_name["intra"].D if ("intra" in by_name and by_name["intra"].D) else float(walked.diffusivity)
    walk = PersistentWalk(traj, float(walked.dt), int(walked.sub_steps), float(walked.dt_sim), boundary_local_time=dlog,
                          compartment=comp, seed=int(seed), diffusivity=D_ref, spec=spec,
                          weights=(None if np.allclose(wts, 1.0) else wts), field_basis=(sf if sf is not None else fg),
                          stepping=(dict(rule="adaptive", pools=dict(stepping)) if stepping else None),
                          field_samples=samples, field_sample_every=(int(field_sample_every) if samples is not None else 1),
                          work=merge_work(works), scanner=str(scanner),
                          field_deferred=bool(defer_field))
    object.__setattr__(walk, "run", current())
    return walk
