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

from .substrate import SubstrateSpec, SpecError
from .build import geometry_from_spec


def walk_spec(spec, n_walkers=None, T_max=None, dt_save=None, *, scanner="connectom", floor_fraction=0.1, diffusivity=None,
              seed=0, n_probe=200_000, field=True, field_res=None, field_budget=None, field_cutoff_m=25e-6,
              field_cutoff_tol=0.02, field_cutoff_max_m=50e-6, require_gpu=None, walker_batch_size=50_000, tiers="all",
              seeding=None, adaptive_steps=False, field_sample_every=1, field_far=None, field_gather_every=4, run_dir=None,
              spool=False, context=None):
    """Walk ``spec`` and return a :class:`~dmipy_sim.persistent_walk.PersistentWalk` carrying the spec.

    ``context`` is a :class:`WalkContext` of the spec (its pool tests, boundaries, walking geometries and the
    strand-field basis with its far grid), kept by a producer across the walks of one spec so that none of it is
    built or compiled again; it carries the far grid, so ``field_far`` is not passed with it.

    ``field_far`` is a :class:`~dmipy_sim.fields.strand_field.FarGrid` (or its ``.npy`` path) built for the strand
    substrate: the closed form is then summed over the few strands within its ``near_m`` of a point and the grid read
    beyond, the cutoff being the grid's (no doubling; the grid records what it summed to). A one-off per substrate
    (``StrandFieldBasis.build_far_grid``), the lever that makes a dense substrate's field affordable in the walk.

    ``field_gather_every`` is how many save intervals the walk reuses a walker's list of strands in reach (gathered
    with the margin the walker can travel in between; 4 by default): with a far grid the reach is the switch's
    15 um and the gather is the field's main cost, so 16 halves the field's share of the walk.

        ``field_sample_every`` reads the strand field along the walk at every that-many-th save only (the adaptive
    producer samples it in the walk): the path channel keeps a few modes over the walk and lives on its own grid,
    recorded on the walk and in the pack; the positions keep the save grid the envelope asks for.

    ``adaptive_steps`` walks every pool whose geometry offers ``wall_scales`` (the curved tubes) with the adaptive
    producer (:func:`dmipy_sim.engine.adaptive.simulate_trajectories_adaptive`): free steps away from every
    wall, the R/6 step of the walker's own nearest tube near one, the same channels; the walk records how it
    stepped (``PersistentWalk.stepping``).

    ``seeding`` replaces the spec's uniform rule with :class:`~dmipy_sim.spec.seeding.StratifiedByVoxel`: the
    same number of walkers of every seeded pool in each voxel of a grid the pool occupies, weighted by the
    pool's measured volume fraction in that voxel over the count -- what a pack meant for ``Phantom.partition``
    needs, its per-voxel floor set by the certificate rather than by Poisson chance. ``n_walkers`` is then
    what the grid and the counts give, and is not passed.

    ``field`` is the susceptibility field basis the walk carries for the pack's C3 tier: ``True`` builds it,
    ``False`` leaves it out, ``"grid"`` forces the rasterised k-space route. A strand substrate (sheathed swept
    polylines) takes the per-segment closed form by default
    (:class:`~dmipy_sim.fields.strand_field.StrandFieldBasis`): every strand within ``field_cutoff_m`` of a
    point contributes its nearest segment's hollow-cylinder field, and the cutoff doubles until the channels
    at a sample of the walk's start positions change by less than ``field_cutoff_tol`` (relative rms) when
    it doubles again, up to ``field_cutoff_max_m`` -- the certificate the basis records, with the change the
    last doubling still made when the bound stopped it (``converged: False``): over a domain that is mostly
    sparse (DiSCo) the 1/r^2 fields of distant bundles are a share of the field's variance that a cutoff sum
    reaches only as 1/cutoff, which a far-field grid, not a larger cutoff, will settle. Every other substrate, and a strand substrate
    with ``field="grid"`` (the cross-check of the closed form on a small strand voxel), rasterises the field basis on
    the domain grid at the node spacing the field source's thinnest shell sets
    (:func:`~dmipy_sim.fields.susceptibility_field.field_resolution` of ``spec.validity.thinnest_shell``:
    :data:`~dmipy_sim.fields.susceptibility_field.FIELD_NODES_ACROSS` nodes across it, measured; ``field_res`` in
    metres overrides it, and a spec that records no shell needs it) within ``field_budget`` nodes (default the
    host's, :func:`~dmipy_sim.fields.susceptibility_field.field_node_budget`: half the memory ceiling at the
    build's measured bytes per node); a basis beyond the budget is refused, never coarsened.
    
    ``run_dir`` is where the walk's record goes (:mod:`dmipy_sim.run`; the default root otherwise); with ``spool``
    every finished walker batch is written into it at once, and a call with the same arguments and the same
    ``run_dir`` resumes: the batches found there are read back, the rest walked -- a killed walk costs the batch
    in progress, not the walk.
    """
    with Run("walk_spec", params=dict(spec=getattr(spec, "id", None), n_walkers=n_walkers, T_max=T_max, dt_save=dt_save, field=field,
                                      adaptive_steps=adaptive_steps, seed=seed), run_dir=run_dir) as run:
        import logging
        from ..engine.core import simulate_trajectories
        from ..persistent_walk import PersistentWalk
        from ..acquisition.scanners import save_interval
        spec = SubstrateSpec.from_dict(spec) if isinstance(spec, dict) else spec
        spec.validate()
        if T_max is None:
            raise TypeError("walk_spec needs T_max (seconds)")
        if int(field_sample_every) != 1 and not adaptive_steps:
            raise ValueError("field_sample_every reads the field in the walk, which the adaptive producer does: pass adaptive_steps=True")
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
        if dt_save is None:
            Ds = [p.D for p in spec.pools if p.D] + ([float(diffusivity)] if diffusivity else [])
            dt_save = save_interval(T_max, n_walkers, scanner, D=(max(Ds) if Ds else 2e-9), floor_fraction=floor_fraction,
                                    field=bool(field and spec.field_source_pools))
            logging.getLogger("dmipy_sim").info("walk_spec: dt_save=%.3g s derived for %s over T_max=%.3g s with %d walkers "
                                                "(n_t=%d)", dt_save, scanner, T_max, n_walkers, int(round(T_max / dt_save)) + 1)
        if not _needs_bundle_walk(spec):
            g = geometry_from_spec(spec)
            D = diffusivity
            if D is None:                                   # the walk's reference diffusivity: the intra pool's when
                names = {p.name: p for p in spec.pools}     # there is one (multi-pool kernels carry per-pool D), else
                pool = names.get("intra") if names.get("intra") is not None and names["intra"].D is not None \
                    else spec.pool(spec.seeding.pools[0])   # the seeded pool's
                D = pool.D
            if D is None:
                raise SpecError("the spec's seeded pool has no D and no diffusivity= was given")
            w = simulate_trajectories(int(n_walkers), float(D), g, T_max=T_max, dt_save=dt_save, seed=seed,
                                      require_gpu=require_gpu, walker_batch_size=walker_batch_size, tiers=tiers)
            return PersistentWalk(w.positions, w.dt, w.sub_steps, w.dt_sim, w.boundary_local_time, w.compartment,
                                  w.bound_frac, w.illegal_crossings, w.seed, w.diffusivity, geometry=g, spec=spec, run=w.run)
        return _walk_bundle(spec, int(n_walkers), float(T_max), float(dt_save), seed, n_probe, field, field_res,
                            require_gpu, walker_batch_size, field_budget=field_budget, field_cutoff_m=field_cutoff_m, field_cutoff_tol=field_cutoff_tol, seeding=seeding,
                            field_cutoff_max_m=field_cutoff_max_m, adaptive_steps=adaptive_steps, field_sample_every=int(field_sample_every), field_far=field_far, field_gather_every=int(field_gather_every), context=context, spool=bool(spool))


def field_grid_of_spec(spec, *, field_res=None, field_budget=None, context=None, n_check=400_000, seed=0):
    """The rasterised field basis of ``spec``'s field source, a :class:`~dmipy_sim.fields.susceptibility_field.FieldGrid`:
    the one raster a walk records (:func:`walk_spec`) and a re-pack of a kept walk samples
    (:func:`~dmipy_sim.replay.bank.build_replay_pack` with ``field=``). The shell pool's occupancy comes from the
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
    if not spec.field_source_pools:
        raise SpecError(f"spec {spec.id!r} has no field-source pool (a pool with a susceptibility); there is no field to rasterise")
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
                        + ". Raise field_budget= on a host with the memory, walk it with field=False"
                        + (", or field=True evaluates a strand substrate's per-segment closed form instead" if strands else "")
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
            self.bodies = MeshBodies([load_ply(w.surface.file, scale=(w.surface.scale or 1.0)) for w in walls])
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
        if self._basis is None and self.far is not None and self.spec.field_source_pools:
            from ..fields.strand_field import StrandFieldBasis
            g = self.tests; src0 = self.spec.field_source_pools[0].id
            ob = g.boundary(g.inside_w[src0]) if g.inside_w[src0] else None; ib = g.boundary(g.outside_w[src0]) if g.outside_w[src0] else None
            if ob is not None and ib is not None and ob.kind == "swept_polyline" and ib.kind == "swept_polyline":
                if len(ob.centerlines) != len(ib.centerlines):
                    raise SpecError("the sheath's inner and outer walls list different numbers of strands")
                self._basis = StrandFieldBasis(ob.centerlines, ib.radii, ob.radii, cutoff_m=self.far.cutoff_m, domain=(g.lo, g.hi),
                                               certificate=dict(cutoff_m=float(self.far.cutoff_m), far_grid=self.far.meta,
                                                                note="the cutoff the far grid summed to; not doubled here")).with_far(self.far)
        return self._basis


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


def _walk_bundle(spec, n_walkers, T_max, dt_save, seed, n_probe, field, field_res, require_gpu, batch, field_budget=None,
                 field_cutoff_m=25e-6, field_cutoff_tol=0.02, seeding=None, field_cutoff_max_m=50e-6, adaptive_steps=False, field_sample_every=1, field_far=None, field_gather_every=4, context=None, spool=False):
    """Walk a multi-surface spec pool by pool: every seeded pool is defined by the walls it is inside and the walls it
    is outside; a pool with D > 0 walks the interior of its inside-walls (intra, glia) or the exterior of its
    outside-walls (extra); a shell pool at D = 0 (myelin) is frozen where it was seeded; the field basis is
    rasterised from the same membership tests."""
    from ..engine.core import simulate_trajectories
    from ..persistent_walk import PersistentWalk
    log = logging.getLogger("dmipy_sim")
    ctx = context if context is not None else WalkContext(spec, field_far=field_far)
    g = ctx.tests
    pools, inside_w, outside_w, boundary, member, sampler = g.pools, g.inside_w, g.outside_w, g.boundary, g.member, g.sampler
    lo, hi, periodic, reflect, seeded, wf = g.lo, g.hi, g.periodic, g.reflect, g.seeded, g.wf
    if seeding is None:
        probe = np.random.default_rng(int(seed) + 99).uniform(lo, hi, (int(n_probe), 3))
        frac = {pid: float(member(pid)(probe).mean()) for pid in seeded}
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
    # a strand field is certified on the start positions and, with adaptive steps, sampled by the walk itself
    sf = None
    if field and field != "grid" and spec.field_source_pools:
        src0 = spec.field_source_pools[0].id
        ob = boundary(inside_w[src0]) if inside_w[src0] else None; ib = boundary(outside_w[src0]) if outside_w[src0] else None
        if ob is not None and ib is not None and ob.kind == "swept_polyline" and ib.kind == "swept_polyline":
            starts = np.concatenate([np.asarray(seeds_of[pid], np.float32) for pid in seeded])
            if ctx.far is not None:                                      # the split: the grid's cutoff, no doubling
                sf = ctx.field_basis()
            else:
                sf = _strand_field(ob, ib, lo, hi, starts[:, None, :], field_cutoff_m, field_cutoff_tol, seed, cutoff_max=field_cutoff_max_m)
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
                                               spool=(pool.name if spool else None))
            stepping.append((pool.name, w.stepping))
            if w.field_samples is not None:
                field_samples.append((pid, w.field_samples))
        else:
            w = simulate_trajectories(n, float(pool.D), g, T_max=T_max, dt_save=dt_save, seed=seed + 13 * pid, r0=r0,
                                      require_gpu=require_gpu, walker_batch_size=batch)
        n_t, walked = w.n_t, w
        parts.append((pid, np.asarray(w.positions, np.float32),
                      None if w.boundary_local_time is None else np.asarray(w.boundary_local_time, np.float32)))
    if walked is None:
        raise SpecError("no seeded pool diffuses; nothing to walk")
    traj, dlog, ids, wts, fs = [], [], [], [], []
    surface = all(dl is not None for pid, pos, dl in parts if pos.ndim == 3)      # every walked pool records contact
    sampled = {pid: arr for pid, arr in field_samples}
    for pid, pos, dl in parts:
        if sampled:
            if pid in sampled:
                fs.append(sampled[pid])
            else:                                                            # a frozen shell: its start's channels, constant
                p0 = jnp.asarray(np.asarray(pos if pos.ndim == 2 else pos[:, 0], np.float32))
                seg, keep, _ = sf.within_device()(p0)
                c0 = np.asarray(sf.channels_at_device()(p0, seg, keep)) - sf.mean[None, :]
                fs.append(np.repeat(c0[:, None, :].astype(np.float32), n_t, axis=1))
        if pos.ndim == 2:                                                        # a frozen shell: no path, no contact
            pos = np.repeat(pos[:, None, :], n_t, axis=1); dl = np.zeros((len(pos), n_t), np.float32)
        elif dl is None:
            dl = np.zeros((len(pos), n_t), np.float32)                           # a placeholder: dropped below
        wt = np.asarray(weights_of[pid], float)
        traj.append(pos); dlog.append(dl); ids.append(np.full(len(pos), pid, np.int8)); wts.append(wt)
    traj = np.concatenate(traj); dlog = np.concatenate(dlog); ids = np.concatenate(ids); wts = np.concatenate(wts)
    order = np.random.default_rng(int(seed) + 991).permutation(len(ids))   # any prefix is a fair subsample
    traj, dlog, ids, wts = traj[order], dlog[order], ids[order], wts[order]
    samples = np.concatenate(fs)[order] if fs else None
    if not surface:
        dlog = None                                                              # the geometry records no surface time
    comp = np.repeat(ids[:, None], n_t, axis=1)
    fg = None
    if field and spec.field_source_pools:
        src = spec.field_source_pools[0].id
        outer_b = boundary(inside_w[src]) if inside_w[src] else None
        inner_b = boundary(outside_w[src]) if outside_w[src] else None
        strands = (outer_b is not None and outer_b.kind == "swept_polyline" and inner_b is not None
                   and inner_b.kind == "swept_polyline")
        if strands and field != "grid":
            fg = sf                                                          # built and certified on the start positions
        else:
            fg = field_grid_of_spec(spec, field_res=field_res, field_budget=field_budget, context=ctx)
    by_name = {p.name: p for p in spec.pools}
    D_ref = by_name["intra"].D if ("intra" in by_name and by_name["intra"].D) else float(walked.diffusivity)
    walk = PersistentWalk(traj, float(walked.dt), int(walked.sub_steps), float(walked.dt_sim), boundary_local_time=dlog,
                          compartment=comp, seed=int(seed), diffusivity=D_ref, spec=spec,
                          weights=(None if np.allclose(wts, 1.0) else wts), field_basis=fg,
                          stepping=(dict(rule="adaptive", pools=dict(stepping)) if stepping else None),
                          field_samples=samples, field_sample_every=(int(field_sample_every) if samples is not None else 1))
    object.__setattr__(walk, "run", current())
    return walk
