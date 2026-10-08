"""The substrate kit: everything a request directory needs from a recipe but the seeds (dmrai-lab/dmipy-sim#689).

A node that holds the static walker (``dmipy-walk``), the device classify and the device encoders, but not
dmipy-sim and not JAX, cannot build a ``PackedCurvedCylinders``, a ``StrandFieldBasis`` or a far grid -- those
are the recipe's substrate, the same for every block. ``dmipy_sim_cuda.cli.write_request`` already serialises a
batch's request directory from a live walk (``request.txt`` plus the strand tables ``A``/``AB``/``AB2``/``rr``/
``tube``/``cell_off``/``cell_ids``, the field's ``fA``/``fAB``/``fAB2``/``fa``/``fb``/``fsid``/``fcell_off``/
``fcell_ids``/``fbox`` and a cropped far grid, beside the batch's own ``r0.f32``/``keys.u32``); the kit is that,
once per recipe, with everything fixed by the recipe rather than by a batch stripped out into a template: one
``request.txt`` (and a ``request.json`` of the same fields, typed) per diffusing pool, the strand tables beside
it, the field's tables and uncropped far grid once (the field is the substrate's, shared by every pool that
samples it), the plan, the certificate the shards inherit and the manifest, and a ``kit.json`` naming the
recipe, the commit, this dmipy-sim version and the sha256 of every file.

``write_kit`` runs on the CPU: every table it writes comes from the geometry's and the field basis's own numpy/JAX
arrays, built the way :meth:`~dmipy_sim.fill.recipe.Recipe.context` always builds them -- no GPU, no
``dmipy_sim_cuda``. The serialisation itself (which struct field is which file, which value is a ``c_float`` and
truncates to float32) mirrors ``dmipy_sim_cuda.backend.CudaBackend`` and ``dmipy_sim_cuda.cli.write_request``;
that package is not installed here (and does not need to be: every array it serialises is already plain numpy on
the geometry), so the mirroring functions below (``_strands_struct_and_arrays``, ``_field_tables``, ``_adaptive_struct``,
``_request_struct``, ``crop_far_grid``) are a direct, documented port, not a reimplementation of the physics.
``assemble_request`` is the per-batch half, run with the kit's files alone (no geometry, no spec, no dmipy-sim
import beyond this module and numpy): a batch's ``n``, ``seed``, ``r0``/``keys`` and a crop of the kit's
uncropped far grid turn the template into a full request directory, byte-for-byte what
``dmipy_sim_cuda.cli.write_request`` would have written for that batch (``tests/fill/test_kit.py``)."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np

log = logging.getLogger("dmipy_sim.fill")

#: the strand-walk tables of one pool's geometry, the file names ``write_request`` uses
STRAND_ARRAY_NAMES = ("A.f32", "AB.f32", "AB2.f32", "rr.f32", "tube.i32", "cell_off.i32", "cell_ids.i32")
#: the field basis's own tables, shared by every pool that samples it
FIELD_ARRAY_NAMES = ("fA.f32", "fAB.f32", "fAB2.f32", "fa.f32", "fb.f32", "fsid.i32", "fcell_off.i32", "fcell_ids.i32", "fbox.i16")
#: the request-struct fields that stay in the template (everything but ``n`` and ``seed``, which are a batch's own)
_REQUEST_KEYS = ("n_t", "sub_steps", "step_l", "kappa_over_D", "record", "count", "budget")
#: the strands-struct fields, in ``dmipy_sim_cuda.backend._Strands`` field order
_STRANDS_KEYS = ("n_seg", "nnz", "dims", "gmin", "cs", "interior", "box_reflect", "lo", "hi", "nudge")


def _f32(x):
    """A value as ``ctypes.c_float`` would hold it: the float32 truncation a ``c_float`` struct field applies on
    assignment, which ``dmipy_sim_cuda``'s structs do for every field below named here as such."""
    return float(np.float32(x))


def _fmt(v):
    """``dmipy_sim_cuda.cli.write_request``'s own stringification of one ``request.txt`` value: an array
    comma-joined (a float ``repr``'d, an int as itself), a scalar float ``repr``'d, a scalar int or the engine's
    bare name (``"strands"``) as itself."""
    if isinstance(v, (list, tuple)):
        return ",".join(repr(float(x)) if isinstance(x, float) else str(int(x)) for x in v)
    if isinstance(v, float):
        return repr(float(v))
    if isinstance(v, str):
        return v
    return int(v)


def _csr_cells(cell):
    """``dmipy_sim_cuda.backend.csr_cells``, reproduced: a geometry's padded ``(n_cells, C)`` cell table (ids
    ascending per cell, -1 padding) as CSR ``(offsets, ids)`` int32, the order kept."""
    cell = np.asarray(cell)
    valid = cell >= 0
    counts = valid.sum(1, dtype=np.int64)
    offsets = np.zeros(cell.shape[0] + 1, np.int64)
    np.cumsum(counts, out=offsets[1:])
    return np.ascontiguousarray(offsets, np.int32), np.ascontiguousarray(cell[valid], np.int32)


def _strands_struct_and_arrays(g):
    """``(struct, arrays)`` of a ``PackedCurvedCylinders`` ``g``: the ``DscStrands`` fields and the tables
    ``write_request`` names ``A.f32``/``AB.f32``/``AB2.f32``/``rr.f32``/``tube.i32``/``cell_off.i32``/
    ``cell_ids.i32`` -- ``dmipy_sim_cuda.backend.CudaBackend._strands``, reproduced from the geometry's own
    arrays (no ctypes, no ``.so``: the geometry already carries every one of these as numpy/JAX)."""
    cell_off, cell_ids = _csr_cells(np.asarray(g._CELL))
    lo, hi = ((np.asarray(g._lo), np.asarray(g._hi)) if g.box_reflect else (np.zeros(3), np.zeros(3)))
    struct = dict(n_seg=int(g._A.shape[0]), nnz=int(cell_ids.shape[0]), dims=[int(x) for x in g._DIMS],
                 gmin=[_f32(x) for x in np.asarray(g._GMIN)], cs=_f32(g._CS), interior=int(bool(g.interior)),
                 box_reflect=int(bool(g.box_reflect)), lo=[_f32(x) for x in lo], hi=[_f32(x) for x in hi],
                 nudge=_f32(g.nudge_m))
    arrays = {"A.f32": np.ascontiguousarray(np.asarray(g._A, np.float32)),
             "AB.f32": np.ascontiguousarray(np.asarray(g._AB, np.float32)),
             "AB2.f32": np.ascontiguousarray(np.asarray(g._AB2, np.float32)),
             "rr.f32": np.ascontiguousarray(np.asarray(g._rout, np.float32)),
             "tube.i32": np.ascontiguousarray(np.asarray(g._seg_tube, np.int32)),
             "cell_off.i32": cell_off, "cell_ids.i32": cell_ids}
    return struct, arrays


def _field_tables(b):
    """The field basis's own tables -- ``dmipy_sim_cuda.backend.CudaBackend._field_tables``, reproduced: the
    segment tables, the CSR cell table and each segment's cell box, with the same sanity check (the bucketing
    must account for every CSR entry)."""
    fcell_off, fcell_ids = _csr_cells(np.asarray(b._CELL))
    A64 = np.vstack([c[:-1] for c in b.centerlines]); AB64 = np.vstack([c[1:] - c[:-1] for c in b.centerlines])
    lo64 = np.minimum(A64, A64 + AB64); hi64 = np.maximum(A64, A64 + AB64); cs64 = 1.01 * b.gather_radius_m
    dims = np.asarray(b._dims)
    loc = np.clip(np.floor((lo64 - b._gmin) / cs64).astype(int), 0, dims - 1)
    hic = np.clip(np.floor((hi64 - b._gmin) / cs64).astype(int), 0, dims - 1)
    fbox = np.ascontiguousarray(np.stack([loc[:, 0], hic[:, 0], loc[:, 1], hic[:, 1], loc[:, 2], hic[:, 2]], axis=1), np.int16)
    counts = (hic - loc + 1).prod(1)
    if int(counts.sum()) != int(fcell_ids.shape[0]):
        raise RuntimeError(f"the field's cell boxes ({int(counts.sum())} entries) do not match its cell table "
                           f"({fcell_ids.shape[0]}): the bucketing differs")
    return {"fA.f32": np.ascontiguousarray(np.asarray(b._A, np.float32)), "fAB.f32": np.ascontiguousarray(np.asarray(b._AB, np.float32)),
            "fAB2.f32": np.ascontiguousarray(np.asarray(b._AB2, np.float32)), "fa.f32": np.ascontiguousarray(np.asarray(b._a, np.float32)),
            "fb.f32": np.ascontiguousarray(np.asarray(b._b, np.float32)), "fsid.i32": np.ascontiguousarray(np.asarray(b._sid, np.int32)),
            "fcell_off.i32": fcell_off, "fcell_ids.i32": fcell_ids, "fbox.i16": fbox}


def _field_struct(basis, radius_m, sample_every, reuse, list_k, n_tf):
    """The ``DscField`` fields EXCEPT ``far_dims``/``far_origin`` (the crop, a batch's own) --
    ``dmipy_sim_cuda.backend.CudaBackend.field_of``, reproduced for its non-crop half, which needs no ``r0``."""
    _, fcell_ids = _csr_cells(np.asarray(basis._CELL))
    far = basis.far
    return dict(n_seg=int(basis.n_segments), nnz=int(fcell_ids.shape[0]), dims=[int(x) for x in basis._dims],
               gmin=[_f32(x) for x in np.asarray(basis._gmin)], cs=_f32(basis._CS), radius=_f32(radius_m),
               reach=_f32(basis.gather_radius_m), gate_radii=_f32(basis.NEAREST_GATE_RADII),
               has_far=int(far is not None), near_m=_f32(far.near_m if far is not None else 0.0),
               blend_m=_f32(far.blend_m if far is not None else 0.0), far_h=_f32(far.spacing_m if far is not None else 0.0),
               sample_every=int(sample_every), reuse=int(reuse), list_k=max(1, int(list_k)),
               n_tf=int(n_tf), segments_max=int(basis.segments_max))


def _adaptive_struct(plan, R_max):
    """The ``DscAdaptive`` fields of an :class:`~dmipy_sim.engine.adaptive.AdaptivePlan` --
    ``dmipy_sim_cuda.backend.CudaBackend.adaptive_struct``, reproduced (``R_max`` is the WALKING geometry's own
    ``_Rmax``, which differs by pool: the axolemma's radius for intra, the sheath's for extra)."""
    n_classes = int(plan.n_classes)
    steps_c = [0] * 8; step_l_c = [0.0] * 8; reach_c = [0.0] * 8
    for c in range(n_classes):
        steps_c[c] = int(plan.steps_c[c]); step_l_c[c] = _f32(plan.step_l_c[c]); reach_c[c] = _f32(plan.reach_c[c])
    return dict(K=int(plan.steps_per_round), n_rounds=int(plan.n_rounds), n_classes=n_classes,
               sigma_round=_f32(plan.sigma_round), far_at=_f32(plan.far_at), R_min=_f32(plan.R_min), R_max=_f32(R_max),
               steps_c=steps_c, step_l_c=step_l_c, reach_c=reach_c)


def _request_struct(*, n_t, sub_steps, step_l, kappa_over_D, record, count, budget):
    """The ``DscRequest`` fields EXCEPT ``seed`` (a batch's own) -- ``dmipy_sim_cuda.backend.CudaBackend.request_struct``,
    reproduced."""
    return dict(n_t=int(n_t), sub_steps=int(sub_steps), step_l=_f32(step_l), kappa_over_D=_f32(kappa_over_D),
               record=int(bool(record)), count=int(bool(count)), budget=int(budget))


def crop_far_grid(values, origin_m, spacing_m, r0, n_t, dt_save, diffusivity):
    """The far grid cropped to what a batch of starts ``r0`` can reach -- ``dmipy_sim_cuda.backend.CudaBackend.field_of``'s
    crop rule, reproduced: the starts' bounding box plus eight sigma of the walk's excursion and three nodes, so a
    shard of a large substrate uploads a crop, never the whole grid. Returns ``(crop.view(uint16), far_dims, far_origin)``."""
    values = np.asarray(values)
    N = np.asarray(values.shape[:3])
    h = float(spacing_m)
    T = float(dt_save) * (int(n_t) - 1)
    margin = 8.0 * math.sqrt(2.0 * float(diffusivity) * T) + 3.0 * h
    r0 = np.asarray(r0, np.float64)
    origin = np.asarray(origin_m, np.float64)
    lo = np.clip(np.floor((r0.min(0) - margin - origin) / h).astype(int) - 1, 0, N - 1)
    hi = np.clip(np.ceil((r0.max(0) + margin - origin) / h).astype(int) + 2, 1, N)
    hi = np.maximum(hi, lo + 1)
    crop = np.ascontiguousarray(np.asarray(values[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], np.float16))
    far_dims = [int(x) for x in crop.shape[:3]]
    far_origin = [_f32(x) for x in (origin + lo * h)]
    return crop.view(np.uint16), far_dims, far_origin


def dump_request(request, r0, keys, out_dir):
    """A from-scratch ``dmipy_sim_cuda.cli.write_request`` for a live, already-resolved ``WalkRequest`` (its
    ``geometry``, ``stepping`` and ``field`` are real objects, not the kit's template) -- the oracle a kit's own
    per-batch assembly (:func:`assemble_request`) is checked against (``tests/fill/test_kit.py``): that package
    is not installed here, and reproducing it from the request alone (rather than through the kit) is the
    independent half of the cross-check. Strands only (what the kit covers); ``classify`` is not implemented."""
    from ..geometry.curved_cylinder import PackedCurvedCylinders
    g = request.geometry
    if not isinstance(g, PackedCurvedCylinders):
        raise ValueError(f"dump_request serialises the strands engine only, not {type(g).__name__}")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    r0 = np.ascontiguousarray(r0, np.float32); n = int(r0.shape[0])
    struct, arrays = _strands_struct_and_arrays(g)
    lines = dict(engine="strands", n=n)
    lines.update(_request_struct(n_t=request.n_t, sub_steps=request.sub_steps,
                                 step_l=float(np.sqrt(6.0 * request.diffusivity * request.dt_sim)),
                                 kappa_over_D=request.kappa_over_D, record=request.record, count=request.count_walls,
                                 budget=(int(g.bounce_loop.budget) if g.bounce_loop is not None else 0)))
    lines["seed"] = int(request.seed) & 0xFFFFFFFFFFFFFFFF
    lines.update(struct)
    if request.stepping is not None:
        lines["adaptive"] = 1
        lines.update({f"adaptive_{k}": v for k, v in _adaptive_struct(request.stepping, g._Rmax).items()})
    if request.field is not None:
        f = request.field; basis = f.basis
        n_tf = len(range(0, int(request.n_t), int(f.sample_every)))
        lines["field"] = 1
        lines.update({f"field_{k}": v for k, v in _field_struct(basis, f.radius_m, f.sample_every, f.reuse_intervals, f.list_k, n_tf).items()})
        arrays.update(_field_tables(basis))
        if basis.far is not None:
            crop, far_dims, far_origin = crop_far_grid(basis.far.values, basis.far.origin_m, basis.far.spacing_m,
                                                       r0, request.n_t, request.dt_save, request.diffusivity)
            arrays["far.u16"] = crop
        else:
            far_dims, far_origin = [0, 0, 0], [0.0, 0.0, 0.0]
        lines["field_far_dims"] = far_dims; lines["field_far_origin"] = far_origin
    (out / "request.txt").write_text("".join(f"{k}={_fmt(v)}\n" for k, v in lines.items()))
    r0.tofile(out / "r0.f32")
    np.ascontiguousarray(keys, np.uint32).reshape(n, -1)[:, :2].copy().tofile(out / "keys.u32")
    for name, a in arrays.items():
        np.ascontiguousarray(a).tofile(out / name)
    return out


def _n_t_dt(T_max, dt_save):
    """``(n_t, dt_actual)`` of a save grid -- ``simulate_trajectories_adaptive``'s own rounding, reproduced so the
    kit's template carries the WalkRequest's actual ``n_t``/``dt_save`` (not the manifest's nominal one)."""
    n_t = int(round(T_max / dt_save)) + 1
    return n_t, T_max / (n_t - 1)


def _pool_geometry(ctx, pid):
    """The pool's own walking geometry, built the way :func:`dmipy_sim.spec.walk._walk_bundle` builds it (the
    axolemma for an intra pool, the sheath for an extra one -- different walls, different radii, different
    tables: not shared between pools)."""
    g = ctx.tests; spec = ctx.spec
    feature = float(spec.validity.smallest_feature)
    if g.inside_w[pid]:
        return g.boundary(g.inside_w[pid]).geometry("intra", g.lo, g.hi, g.periodic, g.reflect, feature)
    return g.boundary(g.outside_w[pid]).geometry("extra", g.lo, g.hi, g.periodic, g.reflect, feature)


def _pool_lines_and_arrays(rc, ctx, pid, n_t, dt_actual, basis, sampling, *, steps_per_round=16, safety_sigma=6.0,
                           n_classes=4, sub_steps=None, candidate_k_start=64, field_list_k=256):
    """``(lines, arrays)`` of one pool's request template: the ``AdaptivePlan`` built with the SAME defaults
    :func:`~dmipy_sim.engine.adaptive.simulate_trajectories_adaptive` builds it with (nothing here is a second
    formula for a number the live walk also computes -- ``adaptive_plan`` is called, not re-derived)."""
    from ..engine.adaptive import adaptive_plan
    g = ctx.tests; W = rc.man["walk"]
    geom = _pool_geometry(ctx, pid)
    D = float(g.pools[pid].D)
    plan = adaptive_plan(geom, D, dt_actual, steps_per_round=steps_per_round, safety_sigma=safety_sigma,
                         n_classes=n_classes, sub_steps=sub_steps, candidate_k_start=candidate_k_start)
    struct, arrays = _strands_struct_and_arrays(geom)
    lines = dict(engine="strands")
    lines.update(_request_struct(n_t=n_t, sub_steps=int(plan.n_min), step_l=float(np.sqrt(6.0 * D * plan.dt_min)),
                                 kappa_over_D=0.0, record=True, count=bool(geom.count_walls),
                                 budget=(int(geom.bounce_loop.budget) if geom.bounce_loop is not None else 0)))
    lines.update(struct)
    lines["adaptive"] = 1
    lines.update({f"adaptive_{k}": v for k, v in _adaptive_struct(plan, geom._Rmax).items()})
    if sampling:
        f_every = max(1, int(W.get("field_sample_every", 1)))
        f_reuse = max(1, int(W.get("field_gather_every", 4)))
        f_margin = 6.0 * math.sqrt(2.0 * D * dt_actual * f_reuse)
        f_reach = float(basis.gather_radius_m)
        f_radius = min(f_reach + f_margin, 2.0 * f_reach)
        n_tf = len(range(0, n_t, f_every))
        lines["field"] = 1
        lines.update({f"field_{k}": v for k, v in _field_struct(basis, f_radius, f_every, f_reuse, field_list_k, n_tf).items()})
    return lines, arrays, dict(diffusivity=D, interior=bool(g.inside_w[pid]))


def _dmipy_sim_version():
    try:
        from importlib.metadata import version
        return version("dmipy-sim")
    except Exception:
        return None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_kit(rc, out_dir):
    """The substrate kit of recipe ``rc`` (:meth:`~dmipy_sim.fill.recipe.Recipe.kit`), written to ``out_dir``:
    one ``request.txt``/``request.json`` + strand tables per diffusing pool, the field's tables and uncropped
    far grid once (shared by every pool, since the field basis is the substrate's), the plan, the certificate
    the shards inherit, the manifest, and a ``kit.json`` naming the recipe, the commit, this dmipy-sim version
    and the sha256 of every file. Needs ``adaptive_steps`` (the only engine a kit covers) and, when the variant
    has a field, a far grid in the manifest (the per-start cutoff doubling without one needs an actual walk,
    which a kit is built without)."""
    from ..spec.substrate import susceptibility_field_of
    from ..spec.walk import field_source_kind
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    man = rc.man; spec = rc.spec(); ctx = rc.context(); g = ctx.tests
    W = man["walk"]
    if not bool(W.get("adaptive_steps")):
        raise ValueError("a substrate kit serialises the CUDA strands-adaptive engine's request tables; this "
                         "recipe's walk is not adaptive_steps")
    T_max = float(W["T_max_s"])
    n_t, dt_actual = _n_t_dt(T_max, rc.dt_save())
    sampling = field_source_kind(spec) == "strands" and susceptibility_field_of(spec) == "present"
    basis = ctx.field_basis() if sampling else None
    if sampling and (basis is None or basis.far is None):
        raise ValueError("the recipe's variant declares a field but carries no far grid: a kit needs one (the "
                         "per-start cutoff doubling without one needs an actual walk, which a kit is built without)")

    pools_out = {}
    for pid in g.seeded:
        pool = g.pools[pid]
        if pool.D in (None, 0.0) or (bool(g.inside_w[pid]) and bool(g.outside_w[pid])):   # frozen / shell: not walked
            continue
        lines, arrays, meta = _pool_lines_and_arrays(rc, ctx, pid, n_t, dt_actual, basis, sampling)
        pdir = out / "pools" / pool.name; pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "request.json").write_text(json.dumps(lines, indent=1))
        (pdir / "request.txt").write_text("".join(f"{k}={_fmt(v)}\n" for k, v in lines.items()))
        for name, a in arrays.items():
            np.ascontiguousarray(a).tofile(pdir / name)
        pools_out[pool.name] = meta
    if not pools_out:
        raise ValueError("no diffusing pool to walk: a kit needs at least one")

    if sampling:
        for name, a in _field_tables(basis).items():
            np.ascontiguousarray(a).tofile(out / name)
        far = basis.far
        np.ascontiguousarray(np.asarray(far.values, np.float16)).view(np.uint16).tofile(out / "far_full.u16")
        json.dump(dict(origin_m=[float(x) for x in far.origin_m], spacing_m=float(far.spacing_m), near_m=float(far.near_m),
                       blend_m=float(far.blend_m), cutoff_m=float(far.cutoff_m), shape=list(far.values.shape)),
                 open(out / "far_full.json", "w"), indent=1)

    os.makedirs(out / "plan", exist_ok=True)
    shutil.copy(rc.hub.get(man["plan"]["blocks"]), out / "plan" / os.path.basename(man["plan"]["blocks"]))
    shutil.copy(rc.hub.get(man["plan"]["file"]), out / "plan" / os.path.basename(man["plan"]["file"]))
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    cert_path = f"certificate/{rc.variant}.json"
    if rc.hub.exists(cert_path):
        os.makedirs(out / "certificate", exist_ok=True)
        shutil.copy(rc.hub.get(cert_path), out / "certificate" / f"{rc.variant}.json")
    else:
        log.warning("kit %s: no certificate/%s.json on the hub yet; the kit carries none", man.get("id"), rc.variant)

    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "kit.json")
    kit = dict(recipe_id=man.get("id"), commit=man["code"]["commit"], dmipy_sim_version=_dmipy_sim_version(), variant=rc.variant,
              walk=dict(n_t=n_t, dt_save_s=dt_actual, T_max_s=T_max), field=dict(present=bool(sampling)), pools=pools_out,
              files={str(p.relative_to(out)): _sha256(p) for p in files})
    json.dump(kit, open(out / "kit.json", "w"), indent=1)
    log.info("kit written to %s: %d pool(s), field %s", out, len(pools_out), sampling)
    return out


def assemble_request(kit_dir, pool, r0, keys, seed, out_dir):
    """A batch's request directory from a written kit (:func:`write_kit`): the pool's template plus this batch's
    ``n``/``seed``, its ``r0.f32``/``keys.u32``, and a crop of the kit's uncropped far grid by
    :func:`crop_far_grid`. Disk and array work only -- no geometry, no spec, nothing a node without dmipy-sim
    could not do too."""
    kit_dir, out = Path(kit_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kit = json.loads((kit_dir / "kit.json").read_text())
    pdir = kit_dir / "pools" / pool
    template = json.loads((pdir / "request.json").read_text())
    r0 = np.ascontiguousarray(r0, np.float32); n = int(r0.shape[0])
    lines = {"engine": template["engine"], "n": n}
    for k in _REQUEST_KEYS:
        lines[k] = template[k]
    lines["seed"] = int(seed) & 0xFFFFFFFFFFFFFFFF
    for k in _STRANDS_KEYS:
        lines[k] = template[k]
    if "adaptive" in template:
        lines["adaptive"] = 1
        lines.update({k: v for k, v in template.items() if k.startswith("adaptive_")})
    if "field" in template:
        lines["field"] = 1
        lines.update({k: v for k, v in template.items() if k.startswith("field_")})
        meta = json.loads((kit_dir / "far_full.json").read_text())
        far_full = np.fromfile(kit_dir / "far_full.u16", dtype=np.uint16).view(np.float16).reshape(meta["shape"])
        crop, far_dims, far_origin = crop_far_grid(far_full, meta["origin_m"], meta["spacing_m"], r0,
                                                   kit["walk"]["n_t"], kit["walk"]["dt_save_s"], kit["pools"][pool]["diffusivity"])
        lines["field_far_dims"] = far_dims; lines["field_far_origin"] = far_origin
    (out / "request.txt").write_text("".join(f"{k}={_fmt(v)}\n" for k, v in lines.items()))
    r0.tofile(out / "r0.f32")
    np.ascontiguousarray(keys, np.uint32).reshape(n, -1)[:, :2].copy().tofile(out / "keys.u32")
    for name in STRAND_ARRAY_NAMES:
        shutil.copy(pdir / name, out / name)
    if "field" in template:
        for name in FIELD_ARRAY_NAMES:
            shutil.copy(kit_dir / name, out / name)
        crop.tofile(out / "far.u16")
    return out
