"""``walk_spec``: the producer that reads a spec and nothing else.

A spec with a single geometry (any analytic family, one mesh surface) is walked by
``simulate_trajectories`` on ``geometry_from_spec``. A multi-surface mesh spec -- a bundle of fibres,
each an inner and an outer closed surface -- is walked pool by pool: the intra pool inside the inner
surfaces, the extra pool outside the outer ones with the domain's faces as walls, a stuck myelin
pool (``D = 0``) frozen where it was seeded; seeding follows the spec's rule (uniform density by
measured volume, weights by water fraction or thinning) and the pools are concatenated into one
``PersistentWalk`` whose compartment channel carries the spec's ids. The field basis of a
field-source pool is computed on the domain grid and rides on the walk. This is what
``replay.builders.mesh_bundle`` used to decide in code.
"""
import numpy as np

from .substrate import SubstrateSpec, SpecError
from .build import geometry_from_spec


def walk_spec(spec, n_walkers, T_max, dt_save, *, diffusivity=None, seed=0, n_probe=200_000,
              field=True, field_res=0.2e-6, require_gpu=None, walker_batch_size=50_000, tiers="all"):
    """Walk ``spec`` and return a :class:`~dmipy_sim.persistent_walk.PersistentWalk` carrying the spec."""
    from ..engine.core import simulate_trajectories
    from ..persistent_walk import PersistentWalk
    spec = SubstrateSpec.from_dict(spec) if isinstance(spec, dict) else spec
    spec.validate()
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
                        require_gpu, walker_batch_size)


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
                from ..geometry.curved_tube import PackedCurvedTubes
                g = PackedCurvedTubes(self.centerlines, self.radii, interior=(pool == "intra"),
                                      box=((lo, hi) if reflect else None))
            self._geom[key] = g
        return self._geom[key]


def _walk_bundle(spec, n_walkers, T_max, dt_save, seed, n_probe, field, field_res, require_gpu, batch):
    """Walk a multi-surface spec pool by pool: every seeded pool is defined by the walls it is inside and the walls it
    is outside; a pool with D > 0 walks the interior of its inside-walls (intra, glia) or the exterior of its
    outside-walls (extra); a shell pool at D = 0 (myelin) is frozen where it was seeded; the field basis is
    rasterised from the same membership tests."""
    from ..engine.core import simulate_trajectories
    from ..fields.susceptibility_field import FieldGrid, mesh_field_basis, predicate_field_basis
    from ..persistent_walk import PersistentWalk
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
    probe = np.random.default_rng(int(seed) + 99).uniform(lo, hi, (int(n_probe), 3))
    seeded = list(spec.seeding.pools)
    frac = {pid: float(member(pid)(probe).mean()) for pid in seeded}
    wf = {pid: pools[pid].water_fraction for pid in pools}
    if spec.seeding.weights == "thin":                 # seed by volume x water fraction, every walker weight 1
        mass = {pid: frac[pid] * wf[pid] for pid in seeded}
    else:                                              # seed by volume, carry the water fraction as a weight
        mass = {pid: frac[pid] for pid in seeded}
    tot = sum(mass.values())
    if tot <= 0:
        raise SpecError("no probe point landed in any seeded pool; the domain box does not cover the substrate")
    counts = {pid: max(1, int(round(n_walkers * mass[pid] / tot))) for pid in seeded}

    def seeds(pred, n, s):
        out, need = [], n
        while need > 0:
            pts = np.random.default_rng(s).uniform(lo, hi, (max(4 * need, 1024), 3)); s += 1
            keep = pts[pred(pts)]
            out.append(keep[:need]); need -= len(keep[:need])
        return np.concatenate(out)

    feature = float(spec.validity.smallest_feature)
    parts, n_t, walked = [], None, None                    # (pid, positions or seeds, local time or None)
    for pid in seeded:
        pool, n, pred = pools[pid], counts[pid], member(pid)
        shell = bool(inside_w[pid]) and bool(outside_w[pid])
        if pool.D in (None, 0.0):
            if not shell:
                raise SpecError(f"pool {pool.name!r} needs D to be walked")
            parts.append((pid, seeds(pred, n, seed + 13 * pid).astype(np.float32), None))      # frozen shell
            continue
        if shell:
            raise SpecError(f"pool {pool.name!r} diffuses between two surfaces; a diffusing shell pool is not "
                            f"implemented (set D = 0 for a stuck pool)")
        g = (boundary(inside_w[pid]).geometry("intra", lo, hi, periodic, reflect, feature) if inside_w[pid]
             else boundary(outside_w[pid]).geometry("extra", lo, hi, periodic, reflect, feature))
        r0 = seeds(pred, n, seed + 13 * pid)
        w = simulate_trajectories(n, float(pool.D), g, T_max=T_max, dt_save=dt_save, seed=seed + 13 * pid, r0=r0,
                                  require_gpu=require_gpu, walker_batch_size=batch)
        n_t, walked = w.n_t, w
        parts.append((pid, np.asarray(w.positions, np.float32), np.asarray(w.boundary_local_time, np.float32)))
    if walked is None:
        raise SpecError("no seeded pool diffuses; nothing to walk")
    traj, dlog, ids, wts = [], [], [], []
    for pid, pos, dl in parts:
        if dl is None:
            pos = np.repeat(pos[:, None, :], n_t, axis=1); dl = np.zeros((len(pos), n_t), np.float32)
        wt = np.ones(len(pos)) if spec.seeding.weights == "thin" else np.full(len(pos), wf[pid])
        traj.append(pos); dlog.append(dl); ids.append(np.full(len(pos), pid, np.int8)); wts.append(wt)
    traj = np.concatenate(traj); dlog = np.concatenate(dlog); ids = np.concatenate(ids); wts = np.concatenate(wts)
    order = np.random.default_rng(int(seed) + 991).permutation(len(ids))   # any prefix is a fair subsample
    traj, dlog, ids, wts = traj[order], dlog[order], ids[order], wts[order]
    comp = np.repeat(ids[:, None], n_t, axis=1)
    fg = None
    if field and spec.field_source_pools:
        src = spec.field_source_pools[0].id
        outer_b = boundary(inside_w[src]) if inside_w[src] else None
        inner_b = boundary(outside_w[src]) if outside_w[src] else None
        if outer_b is None:
            raise SpecError(f"field-source pool {pools[src].name!r} is bounded by no wall; its occupancy cannot be rasterised")
        if inner_b is not None and inner_b.kind == "mesh" and outer_b.kind == "mesh":
            basis, origin, _ = mesh_field_basis((inner_b.V, inner_b.F), (outer_b.V, outer_b.F), lo, hi, res=field_res,
                                                include_aniso=True)
        else:
            basis, origin, _ = predicate_field_basis(inner_b.contains if inner_b is not None else None, outer_b.contains,
                                                     lo, hi, res=field_res, include_aniso=True)
        fg = FieldGrid(basis, np.asarray(origin, float))
    by_name = {p.name: p for p in spec.pools}
    D_ref = by_name["intra"].D if ("intra" in by_name and by_name["intra"].D) else float(walked.diffusivity)
    return PersistentWalk(traj, float(walked.dt), int(walked.sub_steps), float(walked.dt_sim), boundary_local_time=dlog,
                          compartment=comp, seed=int(seed), diffusivity=D_ref, spec=spec,
                          weights=(None if np.allclose(wts, 1.0) else wts), field_grid=fg)
