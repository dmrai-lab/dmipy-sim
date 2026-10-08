"""The substrate kit: everything a request directory needs from a recipe but the seeds (dmrai-lab/dmipy-sim#689).

A node that holds the static walker (``dmipy-walk``), the device classify and the device encoders, but not
dmipy-sim and not JAX, cannot build a ``PackedCurvedCylinders``, a ``StrandFieldBasis`` or a far grid -- those
are the recipe's substrate, the same for every block. ``dmipy_sim_cuda.cli.write_request`` serialises a batch's
request directory from a live walk (``request.txt`` plus the strand tables ``A``/``AB``/``AB2``/``rr``/``tube``/
``cell_off``/``cell_ids``, the field's ``fA``/``fAB``/``fAB2``/``fa``/``fb``/``fsid``/``fcell_off``/``fcell_ids``/
``fbox`` and a cropped far grid, beside the batch's own ``r0.f32``/``keys.u32``) by calling
:mod:`dmipy_sim.engine.backends`'s one request-directory serialisation (``write_request_directory`` and its
helpers, dmrai-lab/dmipy-sim#691); the kit is that, once per recipe, with everything fixed by the recipe rather
than by a batch stripped out into a template: one ``request.txt`` (and a ``request.json`` of the same fields,
typed) per diffusing pool, the strand tables beside it, the field's tables and uncropped far grid once (the
field is the substrate's, shared by every pool that samples it), the plan, the certificate the shards inherit
and the manifest, and a ``kit.json`` naming the recipe, the commit, this dmipy-sim version and the sha256 of
every file.

``write_kit`` runs on the CPU: every table it writes comes from the geometry's and the field basis's own numpy/JAX
arrays, built the way :meth:`~dmipy_sim.fill.recipe.Recipe.context` always builds them -- no GPU, no
``dmipy_sim_cuda``; the engine-protocol helpers it calls (:func:`~dmipy_sim.engine.backends.strands_struct_fields`,
:func:`~dmipy_sim.engine.backends.field_struct_fields`, :func:`~dmipy_sim.engine.backends.adaptive_struct_fields`,
:func:`~dmipy_sim.engine.backends.request_struct_fields`) need nothing from that package either: every array they
serialise is already plain numpy on the geometry. ``assemble_request`` is the per-batch half, run with the kit's
files alone (no geometry, no spec, no dmipy-sim import beyond this module, ``dmipy_sim.engine.backends`` and
numpy): a batch's ``n``, ``seed``, ``r0``/``keys`` and a crop of the kit's uncropped far grid
(:func:`~dmipy_sim.engine.backends.crop_far_grid`) turn the template into a full request directory, byte-for-byte
what ``dmipy_sim_cuda.cli.write_request`` would have written for that batch (``tests/fill/test_kit.py``)."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np

from ..engine.backends import (FIELD_ARRAY_NAMES, REQUEST_TEMPLATE_KEYS, STRAND_ARRAY_NAMES, STRANDS_STRUCT_KEYS,
                                adaptive_struct_fields, crop_far_grid, field_struct_fields, field_tables,
                                format_request_value, request_struct_fields, strands_struct_fields)

log = logging.getLogger("dmipy_sim.fill")


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
    struct, arrays = strands_struct_fields(geom)
    lines = dict(engine="strands")
    lines.update(request_struct_fields(n_t=n_t, sub_steps=int(plan.n_min), step_l=float(np.sqrt(6.0 * D * plan.dt_min)),
                                       kappa_over_D=0.0, record=True, count=bool(geom.count_walls),
                                       budget=(int(geom.bounce_loop.budget) if geom.bounce_loop is not None else 0)))
    lines.update(struct)
    lines["adaptive"] = 1
    lines.update({f"adaptive_{k}": v for k, v in adaptive_struct_fields(plan, geom._Rmax).items()})
    if sampling:
        f_every = max(1, int(W.get("field_sample_every", 1)))
        f_reuse = max(1, int(W.get("field_gather_every", 4)))
        f_margin = 6.0 * math.sqrt(2.0 * D * dt_actual * f_reuse)
        f_reach = float(basis.gather_radius_m)
        f_radius = min(f_reach + f_margin, 2.0 * f_reach)
        n_tf = len(range(0, n_t, f_every))
        lines["field"] = 1
        lines.update({f"field_{k}": v for k, v in field_struct_fields(basis, f_radius, f_every, f_reuse, field_list_k, n_tf).items()})
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
        (pdir / "request.txt").write_text("".join(f"{k}={format_request_value(v)}\n" for k, v in lines.items()))
        for name, a in arrays.items():
            np.ascontiguousarray(a).tofile(pdir / name)
        pools_out[pool.name] = meta
    if not pools_out:
        raise ValueError("no diffusing pool to walk: a kit needs at least one")

    if sampling:
        for name, a in field_tables(basis).items():
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
    for k in REQUEST_TEMPLATE_KEYS:
        lines[k] = template[k]
    lines["seed"] = int(seed) & 0xFFFFFFFFFFFFFFFF
    for k in STRANDS_STRUCT_KEYS:
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
    (out / "request.txt").write_text("".join(f"{k}={format_request_value(v)}\n" for k, v in lines.items()))
    r0.tofile(out / "r0.f32")
    np.ascontiguousarray(keys, np.uint32).reshape(n, -1)[:, :2].copy().tofile(out / "keys.u32")
    for name in STRAND_ARRAY_NAMES:
        shutil.copy(pdir / name, out / name)
    if "field" in template:
        for name in FIELD_ARRAY_NAMES:
            shutil.copy(kit_dir / name, out / name)
        crop.tofile(out / "far.u16")
    return out
