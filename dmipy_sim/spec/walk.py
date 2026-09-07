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
    mesh_walls = [w for w in spec.walls if w.surface.kind == "mesh"]
    if len(mesh_walls) <= 1:
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
    return _walk_mesh_bundle(spec, int(n_walkers), float(T_max), float(dt_save), seed, n_probe, field, field_res,
                             require_gpu, walker_batch_size)


def _walk_mesh_bundle(spec, n_walkers, T_max, dt_save, seed, n_probe, field, field_res, require_gpu, batch):
    from ..engine.core import simulate_trajectories
    from ..geometry.mesh import Mesh, load_ply
    from ..fields.susceptibility_field import mesh_contains, FieldGrid, mesh_field_basis
    from ..persistent_walk import PersistentWalk
    pools = {p.name: p for p in spec.pools}
    if set(pools) - {"extra", "intra", "myelin"}:
        raise SpecError("walk_spec walks extra / intra / myelin mesh bundles; other pools are not implemented")
    for w in spec.walls:
        if w.permeability.in_to_out > 0 or w.permeability.out_to_in > 0:
            raise SpecError(f"wall {w.name!r} is permeable; a permeable multi-surface mesh walk is not implemented")
    inner_w = [w for w in spec.walls if w.inside_pool == 1]
    outer_w = [w for w in spec.walls if w.outside_pool == 0]
    if not inner_w:
        raise SpecError("a mesh bundle needs walls whose inside is the intra pool (id 1)")

    def concat(walls):
        Vs, Fs, off = [], [], 0
        for w in walls:
            V, F = load_ply(w.surface.file, scale=(w.surface.scale or 1.0))
            Vs.append(np.asarray(V, float)); Fs.append(np.asarray(F, np.int64) + off); off += len(V)
        return np.concatenate(Vs), np.concatenate(Fs)
    Vi, Fi = concat(inner_w)
    Vo, Fo = concat(outer_w)
    lo, hi = np.asarray(spec.domain.box_min, float), np.asarray(spec.domain.box_max, float)
    periodic = [b == "periodic" for b in spec.domain.boundary]
    reflect = "reflect" in spec.domain.boundary
    rng = np.random.default_rng(int(seed) + 99)
    probe = rng.uniform(lo, hi, (int(n_probe), 3))
    pin = mesh_contains(Vi, Fi, probe)
    pout = mesh_contains(Vo, Fo, probe)
    frac = {"intra": float(pin.mean()), "myelin": float((pout & ~pin).mean()), "extra": float((~pout).mean())}
    seeded = [spec.pool(i).name for i in spec.seeding.pools]
    wf = {n: pools[n].water_fraction for n in pools}
    weight_mass = {n: frac[n] * wf[n] for n in seeded}
    tot = sum(weight_mass.values())
    counts = {n: max(1, int(round(n_walkers * weight_mass[n] / tot))) for n in seeded}
    if spec.seeding.weights == "water_fraction":               # seed by volume, carry the water fraction
        counts = {n: max(1, int(round(n_walkers * frac[n] / sum(frac[m] for m in seeded)))) for n in seeded}

    def seeds(pred, n, s):
        out = []
        need = n
        while need > 0:
            pts = np.random.default_rng(s).uniform(lo, hi, (max(4 * need, 1024), 3)); s += 1
            keep = pts[pred(pts)]
            out.append(keep[:need]); need -= len(keep[:need])
        return np.concatenate(out)

    fr_i = float(spec.validity.smallest_feature)
    mesh_in = Mesh(Vi, Fi, periodic=periodic, voxel_min=lo, voxel_max=hi, feature_radius=fr_i, pool="intra",
                   box_reflect=reflect)
    mesh_out = Mesh(Vo, Fo, periodic=periodic, voxel_min=lo, voxel_max=hi, feature_radius=fr_i, pool="extra",
                    box_reflect=reflect)
    parts = []                                                    # (positions, dlog, ids, weight)
    n_t = None
    for name in ("extra", "intra", "myelin"):
        if name not in seeded:
            continue
        n = counts[name]
        pool = pools[name]
        if name == "myelin":
            if pool.D not in (None, 0.0):
                raise SpecError("a diffusing myelin pool (D > 0) in a mesh bundle is not implemented; set D = 0")
            r0 = seeds(lambda q: mesh_contains(Vo, Fo, q) & ~mesh_contains(Vi, Fi, q), n, seed + 1)
            pos = np.repeat(r0[:, None, :].astype(np.float32), n_t, axis=1) if n_t else None
            parts.append(("myelin", r0.astype(np.float32), None, 2))
            continue
        pred = (lambda q: mesh_contains(Vi, Fi, q)) if name == "intra" else (lambda q: ~mesh_contains(Vo, Fo, q))
        r0 = seeds(pred, n, seed + (0 if name == "intra" else 7))
        D = pool.D
        if D is None:
            raise SpecError(f"pool {name!r} needs D to be walked")
        w = simulate_trajectories(n, float(D), mesh_in if name == "intra" else mesh_out, T_max=T_max, dt_save=dt_save,
                                  seed=seed + (0 if name == "intra" else 7), r0=r0, require_gpu=require_gpu,
                                  walker_batch_size=batch)
        n_t = w.n_t
        parts.append((name, np.asarray(w.positions, np.float32), np.asarray(w.boundary_local_time, np.float32),
                      1 if name == "intra" else 0))
        dt_saved, sub_steps, dt_sim = w.dt, w.sub_steps, w.dt_sim
    traj, dlog, ids, wts = [], [], [], []
    for name, pos, dl, pid in parts:
        if name == "myelin":
            r0 = pos
            if spec.seeding.weights == "thin":
                keep = max(1, int(round(wf["myelin"] * len(r0))))
                r0 = r0[np.sort(np.random.default_rng(seed + 7).permutation(len(r0))[:keep])]
                wt = np.ones(len(r0))
            else:
                wt = np.full(len(r0), wf["myelin"])
            pos = np.repeat(r0[:, None, :], n_t, axis=1); dl = np.zeros((len(r0), n_t), np.float32)
        else:
            wt = np.ones(len(pos)) if spec.seeding.weights == "thin" else np.full(len(pos), wf[name])
        traj.append(pos); dlog.append(dl); ids.append(np.full(len(pos), pid, np.int8)); wts.append(wt)
    traj = np.concatenate(traj); dlog = np.concatenate(dlog); ids = np.concatenate(ids); wts = np.concatenate(wts)
    order = np.random.default_rng(int(seed) + 991).permutation(len(ids))   # any prefix is a fair subsample
    traj, dlog, ids, wts = traj[order], dlog[order], ids[order], wts[order]
    comp = np.repeat(ids[:, None], n_t, axis=1)
    fg = None
    if field and spec.field_source_pools:
        basis, origin, vs = mesh_field_basis((Vi, Fi), (Vo, Fo), lo, hi, res=field_res, include_aniso=True)
        fg = FieldGrid(basis, np.asarray(origin, float))
    D_intra = pools["intra"].D
    return PersistentWalk(traj, float(dt_saved), int(sub_steps), float(dt_sim), boundary_local_time=dlog,
                          compartment=comp, seed=int(seed), diffusivity=D_intra, spec=spec,
                          weights=(None if np.allclose(wts, 1.0) else wts), field_grid=fg)
