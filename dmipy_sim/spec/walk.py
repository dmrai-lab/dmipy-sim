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

from .substrate import SubstrateSpec, SpecError
from .build import geometry_from_spec


def walk_spec(spec, n_walkers=None, T_max=None, dt_save=None, *, scanner="connectom", floor_fraction=0.1, diffusivity=None,
              seed=0, n_probe=200_000, field=True, field_res=0.2e-6, field_budget=5e7, field_cutoff_m=25e-6,
              field_cutoff_tol=0.02, field_cutoff_max_m=50e-6, require_gpu=None, walker_batch_size=50_000, tiers="all",
              seeding=None, adaptive_steps=False):
    """Walk ``spec`` and return a :class:`~dmipy_sim.persistent_walk.PersistentWalk` carrying the spec.

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
    with ``field="grid"``, rasterises within ``field_budget`` voxels (13 float32 channels each) at
    ``field_res``, the cross-check of the closed form on a small strand voxel.
    """
    import logging
    from ..engine.core import simulate_trajectories
    from ..persistent_walk import PersistentWalk
    from ..acquisition.scanners import save_interval
    spec = SubstrateSpec.from_dict(spec) if isinstance(spec, dict) else spec
    spec.validate()
    if T_max is None:
        raise TypeError("walk_spec needs T_max (seconds)")
    if seeding is not None:
        from .seeding import StratifiedByVoxel
        if not isinstance(seeding, StratifiedByVoxel):
            raise TypeError(f"seeding must be a StratifiedByVoxel, got {type(seeding).__name__}")
        if n_walkers is not None:
            raise TypeError("n_walkers is what a stratified seeding produces: give seeding= or n_walkers=, not both")
        if not _needs_bundle_walk(spec):
            raise SpecError("stratified seeding is for the pool-by-pool walk of a multi-surface spec")
        n_walkers = int(sum(seeding.count_for(p.name).sum() for p in spec.pools if p.id in spec.seeding.pools))
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
                              w.bound_frac, w.illegal_crossings, w.seed, w.diffusivity, geometry=g, spec=spec)
    return _walk_bundle(spec, int(n_walkers), float(T_max), float(dt_save), seed, n_probe, field, field_res,
                        require_gpu, walker_batch_size, field_budget=float(field_budget), field_cutoff_m=field_cutoff_m, field_cutoff_tol=field_cutoff_tol, seeding=seeding,
                        field_cutoff_max_m=field_cutoff_max_m, adaptive_steps=adaptive_steps)


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
            Vs, Fs, off = [], [], 0
            for w in walls:
                V, F = load_ply(w.surface.file, scale=(w.surface.scale or 1.0))
                Vs.append(np.asarray(V, float)); Fs.append(np.asarray(F, np.int64) + off); off += len(V)
            self.V, self.F = np.concatenate(Vs), np.concatenate(Fs)
        elif self.kind == "sphere_union":
            from .build import sphere_union_arrays
            parts = [sphere_union_arrays(w.surface) for w in walls]
            self.centers = np.concatenate([p[0] for p in parts]); self.radii = np.concatenate([p[1] for p in parts])
        else:
            from .build import polyline_arrays
            parts = [polyline_arrays(w.surface) for w in walls]
            self.centerlines = [c for p in parts for c in p[0]]; self.radii = np.concatenate([p[1] for p in parts])

    def sample_inside(self, n, rng):
        """``n`` points uniform by volume inside the swept polylines (by segment volume, then a disc, then the
        segment's length): the exact intra draw, no rejection."""
        if self.kind != "swept_polyline":
            raise SpecError("sample_inside is the swept-polyline draw; other surfaces are seeded by rejection")
        A = np.vstack([c[:-1] for c in self.centerlines]); B = np.vstack([c[1:] for c in self.centerlines])
        r = np.concatenate([np.full(len(c) - 1, rr) for c, rr in zip(self.centerlines, self.radii)])
        L = np.linalg.norm(B - A, axis=1); w = np.pi * r ** 2 * L
        k = rng.choice(len(w), size=int(n), p=w / w.sum())
        t = rng.uniform(0.0, 1.0, int(n))[:, None]
        C = A[k] + (B[k] - A[k]) * t
        T = (B[k] - A[k]) / np.maximum(L[k], 1e-30)[:, None]
        ref = np.tile([0.0, 0.0, 1.0], (int(n), 1)); ref[np.abs((T * ref).sum(1)) > 0.9] = [1.0, 0.0, 0.0]
        e1 = np.cross(T, ref); e1 /= np.linalg.norm(e1, axis=1, keepdims=True); e2 = np.cross(T, e1)
        rr = r[k] * np.sqrt(rng.uniform(0.0, 1.0, int(n))); th = rng.uniform(0.0, 2 * np.pi, int(n))
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
            from ..fields.susceptibility_field import mesh_contains
            return np.asarray(mesh_contains(self.V, self.F, pts), bool)
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


def _strand_field(outer_b, inner_b, lo, hi, traj, cutoff_m, tol, seed, strands_max=1024, cutoff_max=50e-6):
    """The per-segment field basis of a strand substrate (its sheath between ``inner_b`` and ``outer_b``), the
    cutoff doubled until the channels at a sample of the walk's start positions move by less than ``tol``."""
    from ..fields.strand_field import StrandFieldBasis
    if len(outer_b.centerlines) != len(inner_b.centerlines):
        raise SpecError("the sheath's inner and outer walls list different numbers of strands")
    sf = StrandFieldBasis(outer_b.centerlines, inner_b.radii, outer_b.radii, cutoff_m=cutoff_m, domain=(lo, hi),
                          strands_max=strands_max)
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


def _walk_bundle(spec, n_walkers, T_max, dt_save, seed, n_probe, field, field_res, require_gpu, batch, field_budget=5e7,
                 field_cutoff_m=25e-6, field_cutoff_tol=0.02, seeding=None, field_cutoff_max_m=50e-6, adaptive_steps=False):
    """Walk a multi-surface spec pool by pool: every seeded pool is defined by the walls it is inside and the walls it
    is outside; a pool with D > 0 walks the interior of its inside-walls (intra, glia) or the exterior of its
    outside-walls (extra); a shell pool at D = 0 (myelin) is frozen where it was seeded; the field basis is
    rasterised from the same membership tests."""
    from ..engine.core import simulate_trajectories
    from ..fields.susceptibility_field import FieldGrid, mesh_field_basis, predicate_field_basis
    from ..persistent_walk import PersistentWalk
    log = logging.getLogger("dmipy_sim")
    pools = {p.id: p for p in spec.pools}
    for w in spec.walls:
        if w.permeability.in_to_out > 0 or w.permeability.out_to_in > 0:
            raise SpecError(f"wall {w.name!r} is permeable; a permeable multi-surface walk is not implemented")
    inside_w = {p: [w for w in spec.walls if w.inside_pool == p] for p in pools}
    outside_w = {p: [w for w in spec.walls if w.outside_pool == p] for p in pools}
    bounds = {}

    def boundary(walls):
        key = tuple(w.name for w in walls)
        if key not in bounds:
            bounds[key] = _Boundary(walls)
        return bounds[key]

    def member(pid):
        def pred(pts):
            m = np.ones(len(pts), bool)
            if inside_w[pid]:
                m &= boundary(inside_w[pid]).contains(pts)
            if outside_w[pid]:
                m &= ~boundary(outside_w[pid]).contains(pts)
            return m
        return pred

    lo, hi = np.asarray(spec.domain.box_min, float), np.asarray(spec.domain.box_max, float)
    periodic = [b == "periodic" for b in spec.domain.boundary]
    reflect = "reflect" in spec.domain.boundary
    seeded = list(spec.seeding.pools)
    wf = {pid: pools[pid].water_fraction for pid in pools}

    def sampler(pid):
        """``(n, rng) -> (points, accepted)``: the pool's own draw -- inside its swept polylines by segment volume
        (exact, no rejection), else uniform in the box and rejected by membership."""
        pred = member(pid)
        if inside_w[pid] and not outside_w[pid] and all(w.surface.kind == "swept_polyline" for w in inside_w[pid]):
            b = boundary(inside_w[pid])                    # strands may leave the box: a draw outside it is rejected
            return lambda n, rng: (lambda P: (P, np.all((P >= lo) & (P <= hi), axis=1)))(b.sample_inside(n, rng))
        return lambda n, rng: (lambda P: (P, pred(P)))(rng.uniform(lo, hi, (n, 3)))

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
        from .seeding import fill_per_voxel
        grid = seeding.grid
        V_box = float(np.prod(hi - lo)); V_vox = float(np.prod(grid.voxel_size_m))

        def bin_index(P):
            ijk, inside = grid.bin(P)
            flat = np.ravel_multi_index(tuple(np.clip(ijk, 0, np.asarray(grid.shape) - 1).T), grid.shape)
            return np.where(inside, flat, -1)

        def seeds(pid, s):
            want = seeding.count_for(pools[pid].name)
            by_volume = inside_w[pid] and not outside_w[pid] and all(w.surface.kind == "swept_polyline" for w in inside_w[pid])
            log.info("walk_spec: seeding pool %s per voxel (%d wanted)", pools[pid].name, int(want.sum()))
            P, v, f, trials, n_drawn = fill_per_voxel(sampler(pid), bin_index, grid.n_voxels, want,
                                                      trials_max=int(seeding.trials_per_voxel_max), seed=s)
            if by_volume:                                  # the census of a volume-uniform draw over the WHOLE pool:
                V_pool = boundary(inside_w[pid]).volume()  # the pool's volume in a voxel over the voxel's volume
                f = trials / max(n_drawn, 1) * V_pool / V_vox
            n_have = np.bincount(v, minlength=grid.n_voxels)
            w = f[v] * wf[pid] / np.maximum(n_have[v], 1)  # f_pool,v x water fraction / n_pool,v: volume-correct per voxel
            return P, w
    feature = float(spec.validity.smallest_feature)
    parts, n_t, walked = [], None, None                    # (pid, positions or seeds, local time or None)
    weights_of = {}; stepping = []
    for pid in seeded:
        pool = pools[pid]
        shell = bool(inside_w[pid]) and bool(outside_w[pid])
        r0, weights_of[pid] = seeds(pid, seed + 13 * pid)
        n = len(r0)
        if n == 0:
            raise SpecError(f"pool {pool.name!r}: no seed landed in it")
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
        if adaptive_steps:
            if not hasattr(g, "wall_scales"):
                raise SpecError(f"pool {pool.name!r}: its {type(g).__name__} offers no wall_scales; adaptive stepping "
                                f"is for the curved tubes")
            from ..engine.adaptive import simulate_trajectories_adaptive
            w = simulate_trajectories_adaptive(n, float(pool.D), g, T_max, dt_save, seed=seed + 13 * pid, r0=r0,
                                               require_gpu=require_gpu, walker_batch_size=batch)
            stepping.append((pool.name, w.stepping))
        else:
            w = simulate_trajectories(n, float(pool.D), g, T_max=T_max, dt_save=dt_save, seed=seed + 13 * pid, r0=r0,
                                      require_gpu=require_gpu, walker_batch_size=batch)
        n_t, walked = w.n_t, w
        parts.append((pid, np.asarray(w.positions, np.float32),
                      None if w.boundary_local_time is None else np.asarray(w.boundary_local_time, np.float32)))
    if walked is None:
        raise SpecError("no seeded pool diffuses; nothing to walk")
    traj, dlog, ids, wts = [], [], [], []
    surface = all(dl is not None for pid, pos, dl in parts if pos.ndim == 3)      # every walked pool records contact
    for pid, pos, dl in parts:
        if pos.ndim == 2:                                                        # a frozen shell: no path, no contact
            pos = np.repeat(pos[:, None, :], n_t, axis=1); dl = np.zeros((len(pos), n_t), np.float32)
        elif dl is None:
            dl = np.zeros((len(pos), n_t), np.float32)                           # a placeholder: dropped below
        wt = np.asarray(weights_of[pid], float)
        traj.append(pos); dlog.append(dl); ids.append(np.full(len(pos), pid, np.int8)); wts.append(wt)
    traj = np.concatenate(traj); dlog = np.concatenate(dlog); ids = np.concatenate(ids); wts = np.concatenate(wts)
    order = np.random.default_rng(int(seed) + 991).permutation(len(ids))   # any prefix is a fair subsample
    traj, dlog, ids, wts = traj[order], dlog[order], ids[order], wts[order]
    if not surface:
        dlog = None                                                              # the geometry records no surface time
    comp = np.repeat(ids[:, None], n_t, axis=1)
    fg = None
    if field and spec.field_source_pools:
        src = spec.field_source_pools[0].id
        outer_b = boundary(inside_w[src]) if inside_w[src] else None
        inner_b = boundary(outside_w[src]) if outside_w[src] else None
        if outer_b is None:
            raise SpecError(f"field-source pool {pools[src].name!r} is bounded by no wall; its occupancy cannot be rasterised")
        strands = outer_b.kind == "swept_polyline" and inner_b is not None and inner_b.kind == "swept_polyline"
        if strands and field != "grid":
            fg = _strand_field(outer_b, inner_b, lo, hi, traj, field_cutoff_m, field_cutoff_tol, seed, cutoff_max=field_cutoff_max_m)
        else:
            n_vox = int(np.prod(np.ceil((hi - lo) / float(field_res))))
            if n_vox > field_budget:
                raise SpecError(f"the field basis of this substrate would be {n_vox:.2e} voxels at {field_res * 1e6:.2f} um over its "
                                f"{np.round((hi - lo) * 1e6, 1).tolist()} um domain, beyond the {field_budget:.0e}-voxel budget "
                                f"(13 float32 channels per voxel). Coarsen field_res=, raise field_budget=, or walk it with "
                                f"field=False" + ("; field=True evaluates a strand substrate's per-segment closed form instead"
                                                  if strands else ""))
            if inner_b is not None and inner_b.kind == "mesh" and outer_b.kind == "mesh":
                basis, origin, _ = mesh_field_basis((inner_b.V, inner_b.F), (outer_b.V, outer_b.F), lo, hi, res=field_res,
                                                    include_aniso=True)
            else:
                ref_b = inner_b if inner_b is not None else outer_b
                director = ref_b.director if ref_b.kind == "swept_polyline" else None
                basis, origin, _ = predicate_field_basis(inner_b.contains if inner_b is not None else None, outer_b.contains,
                                                         lo, hi, res=field_res, include_aniso=True, director=director)
            fg = FieldGrid(basis, np.asarray(origin, float))
    by_name = {p.name: p for p in spec.pools}
    D_ref = by_name["intra"].D if ("intra" in by_name and by_name["intra"].D) else float(walked.diffusivity)
    return PersistentWalk(traj, float(walked.dt), int(walked.sub_steps), float(walked.dt_sim), boundary_local_time=dlog,
                          compartment=comp, seed=int(seed), diffusivity=D_ref, spec=spec,
                          weights=(None if np.allclose(wts, 1.0) else wts), field_basis=fg,
                          stepping=(dict(rule="adaptive", pools=dict(stepping)) if stepping else None))
