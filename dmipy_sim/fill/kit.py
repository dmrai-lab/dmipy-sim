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
what :func:`~dmipy_sim.engine.backends.write_request_directory` (and ``dmipy_sim_cuda.cli.write_request``,
dmipy-sim-cuda#78) write for that batch's live request (``tests/fill/test_kit.py``): the blocks, the step, the band
``K`` at the table's order ``m``, the ring, the field's modes, reads and mean, with the block table (``table.f64``)
every pool shares. ``kit.json``'s
``walk`` is the walk's block record (``T_max_s``, ``K``, ``block = {h, n_blocks, m}``, ``windows = {n, T}``)."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np

from ..replay.bank import class_band
from ..engine.backends import (FIELD_ARRAY_NAMES, REQUEST_TEMPLATE_KEYS, STRAND_ARRAY_NAMES,
                                STRANDS_STRUCT_KEYS, adaptive_struct_fields, crop_far_grid, field_struct_fields, field_tables,
                                field_tables_window, format_request_value, request_struct_fields,
                                strands_struct_fields, strands_struct_fields_window)

log = logging.getLogger("dmipy_sim.fill")


def _block_record(rc):
    """The recipe's walk as its block record, ``dict(T, h, n_blocks, windows=dict(n, T), K, m)``: the storage windows over
    ``T_max`` (:func:`~dmipy_sim.engine.accumulate.window_count`), the block -- the band's own
    (:func:`~dmipy_sim.engine.accumulate.block_length` at the pack's band) unless the manifest states ``block_h_s``
    -- rounded to a whole number per window as ``simulate_trajectories_adaptive`` rounds it, the blocks over the walk
    (``T = n_blocks h``), the band ``K`` per window (the manifest's ``pack.K``, else the scanner class's,
    :func:`~dmipy_sim.replay.bank.mode_count`) and the block table's order ``m``."""
    from ..engine.accumulate import window_count, block_length
    from dmipy_replay.replay.storage import mode_count
    W = rc.man["walk"]
    n_win, T_win = window_count(float(W["T_max_s"]))
    h = rc.block_h()
    if h is None:
        h = block_length(T_win, int(rc.man["pack"].get("K") or mode_count(T_win, W["scanner"])))
    n_bw = max(int(round(T_win / h)), 1)
    h = T_win / n_bw
    from ..engine.accumulate import BLOCK_ORDER
    K = int(rc.man["pack"].get("K") or mode_count(T_win, W["scanner"]))
    return dict(T=n_win * n_bw * h, h=h, n_blocks=n_win * n_bw, windows=dict(n=int(n_win), T=float(T_win)), K=K, m=int(BLOCK_ORDER))


def _pool_geometry(ctx, pid):
    """The pool's own walking geometry, built the way :func:`dmipy_sim.spec.walk._walk_bundle` builds it (the
    axolemma for an intra pool, the sheath for an extra one -- different walls, different radii, different
    tables: not shared between pools)."""
    g = ctx.tests; spec = ctx.spec
    feature = float(spec.validity.smallest_feature)
    if g.inside_w[pid]:
        return g.boundary(g.inside_w[pid]).geometry("intra", g.lo, g.hi, g.periodic, g.reflect, feature)
    return g.boundary(g.outside_w[pid]).geometry("extra", g.lo, g.hi, g.periodic, g.reflect, feature)


def _request_lines(rc, pid, blk, geom, plan, sampling):
    """``(request fields, K_c3, read_every)`` of one pool's walk as the live walk hands it to a backend: the
    ``DscRequest`` fields (:func:`~dmipy_sim.engine.backends.request_struct_fields`: the blocks, the plan's finest
    step, the band ``K`` at the table's order ``m``, no events -- a strand substrate is impermeable --, the seed and
    the raw subset a batch's own, 0 here), and with the field its modes -- the manifest's ``pack.K_path`` -- and the
    rounds between its reads (:func:`~dmipy_sim.engine.adaptive.field_read_of` at the manifest's
    ``walk.field_read_interval_s``), both of which a kit cannot pilot. A walk of several storage windows is refused by
    name: a backend accumulates one window per walk."""
    from ..engine.adaptive import field_read_of, _spec_holder
    from ..engine.backends import field_read_every
    if int(blk["windows"]["n"]) != 1:
        raise ValueError(f"the recipe's walk is {blk['windows']['n']} storage windows; a backend accumulates one window per walk "
                         "(several are refused by name), so a kit's request covers one")
    D = float(rc.context().tests.pools[pid].D)
    K_c3 = every = 0
    if sampling:
        K_c3 = int(rc.man["pack"].get("K_path") or 0)
        if not K_c3:
            raise ValueError("a kit's request states the field's band: the manifest's pack.K_path (a band a pilot derives "
                             "needs a walk, which a kit is built without)")
        given = rc.field_read_interval()
        if given is None:
            raise ValueError("a kit's request states the field's read: the manifest's walk.field_read_interval_s (a read a pilot "
                             "derives from the read series' quadratic variation needs a walk, which a kit is built without)")
        dt_f = field_read_of(_spec_holder(rc.spec(), geom), None, float(blk["windows"]["T"]), plan, given=given)["dt_f"]
        every = field_read_every(dt_f, plan, int(plan.n_min))
    fields = request_struct_fields(n_blocks=int(blk["n_blocks"]), sub_steps=int(plan.n_min), step_l=float(np.sqrt(6.0 * D * plan.dt_min)),
                                   kappa_over_D=0.0, record=True, count=bool(geom.count_walls),
                                   budget=(int(geom.bounce_loop.budget) if geom.bounce_loop is not None else 0),
                                   block_h=float(blk["h"]), dt_sim=float(plan.dt_min), K=int(blk["K"]), m=int(blk["m"]),
                                   events=0, raw_subset=0)
    return fields, K_c3, every


def _pool_lines_and_arrays(rc, ctx, pid, blk, basis, sampling, *, steps_per_round=16, safety_sigma=6.0,
                           n_classes=4, sub_steps=None, candidate_k_start=64, field_list_k=256):
    """``(lines, arrays)`` of one pool's request template: the ``AdaptivePlan`` built with the SAME defaults
    :func:`~dmipy_sim.engine.adaptive.simulate_trajectories_adaptive` builds it with (nothing here is a second
    formula for a number the live walk also computes -- ``adaptive_plan`` is called, not re-derived)."""
    from ..engine.adaptive import adaptive_plan
    g = ctx.tests; W = rc.man["walk"]
    dt_actual = float(blk["h"])
    geom = _pool_geometry(ctx, pid)
    D = float(g.pools[pid].D)
    plan = adaptive_plan(geom, D, dt_actual, steps_per_round=steps_per_round, safety_sigma=safety_sigma,
                         n_classes=n_classes, sub_steps=sub_steps, candidate_k_start=candidate_k_start,
                         f_band=class_band(W.get("scanner")))
    struct, arrays = strands_struct_fields(geom)
    req, K_c3, every = _request_lines(rc, pid, blk, geom, plan, sampling)
    lines = dict(engine="strands")
    lines.update(req)
    lines.update(struct)
    lines["adaptive"] = 1
    lines.update({f"adaptive_{k}": v for k, v in adaptive_struct_fields(plan, geom._Rmax).items()})
    if sampling:
        f_reuse = max(1, int(W.get("field_gather_every", 4)))
        f_margin = 6.0 * math.sqrt(2.0 * D * dt_actual * f_reuse)
        f_reach = float(basis.gather_radius_m)
        f_radius = min(f_reach + f_margin, 2.0 * f_reach)
        lines["field"] = 1
        lines.update({f"field_{k}": v for k, v in field_struct_fields(basis, f_radius, f_reuse, field_list_k,
                                                                       K_c3=K_c3, read_every=every).items()})
    return lines, arrays, dict(diffusivity=D, interior=bool(g.inside_w[pid]))


def _reach_margin(plan, cell_size, D, T_max):
    """A block's cell window margin (dmrai-lab/dmipy-sim#695, corrected on review): a walker is NOT confined to
    its block's box -- DiSCo's own box is the whole substrate with reflecting faces, so over the walk's whole
    ``T_max`` it can wander as far from where it started as any walker can (the far crop's own rule,
    :func:`~dmipy_sim.engine.backends.crop_far_grid`: eight sigma of ``2 D T_max``, plus three cells of safety),
    not merely one round's worth. On top of that whole-walk excursion, the adaptive plan's largest ROUND reach
    and the free step's own threshold, plus one more cell, cover the one round that could still have been
    stepping against a real wall when the walker crossed that radius. Past the total, nothing the block's own
    walk can ever reach, so the window's clamped border cell carries whatever the true table would have
    anyway."""
    walk_margin = 8.0 * math.sqrt(2.0 * float(D) * float(T_max)) + 3.0 * float(cell_size)
    return walk_margin + float(max(plan.reach_c)) + float(plan.far_at) + float(cell_size)


def _block_box(rc, row):
    """The block's own box in metres, from the plan grid's voxel index ranges ``row["i"]``/``["j"]``/``["k"]``
    (half-open, :meth:`~dmipy_sim.fill.recipe.Recipe.counts`' own slicing) and the recipe's grid."""
    grid = rc.grid()
    origin = np.asarray(grid.origin_m, float); vs = np.asarray(grid.voxel_size_m, float)
    lo = origin + np.array([row["i"][0], row["j"][0], row["k"][0]], float) * vs
    hi = origin + np.array([row["i"][1], row["j"][1], row["k"][1]], float) * vs
    return lo, hi


def _pool_plan(rc, ctx, pid, D, dt_actual, *, steps_per_round=16, safety_sigma=6.0, n_classes=4, sub_steps=None,
               candidate_k_start=64):
    """``(geom, plan)`` of one pool: the same :class:`~dmipy_sim.engine.adaptive.AdaptivePlan`
    :func:`_pool_lines_and_arrays`/:func:`_pool_lines_and_arrays_window` build, split out so a block kit can read
    every walked pool's reach BEFORE deciding the field's one shared window (:func:`write_block_kit`)."""
    from ..engine.adaptive import adaptive_plan
    geom = _pool_geometry(ctx, pid)
    plan = adaptive_plan(geom, D, dt_actual, steps_per_round=steps_per_round, safety_sigma=safety_sigma,
                         n_classes=n_classes, sub_steps=sub_steps, candidate_k_start=candidate_k_start,
                         f_band=class_band(rc.man["walk"].get("scanner")))
    return geom, plan


def _pool_lines_and_arrays_window(rc, ctx, pid, blk, basis, field_window, geom, plan, lo_m, hi_m, *,
                                  field_list_k=256):
    """``(lines, arrays, meta)`` of one pool's BLOCK request template: as :func:`_pool_lines_and_arrays`, but the
    strand tables cropped to the cell window of the block's box ``[lo_m, hi_m]`` plus THIS pool's own walk reach
    (:func:`_reach_margin`, :func:`~dmipy_sim.engine.backends.strands_struct_fields_window`) -- the request
    format's own window arithmetic. ``field_window`` is ``(dims_gmin, n_seg, nnz)`` of the ONE field window every
    walked pool shares (:func:`write_block_kit`: the field's tables are written once, at the kit's top level,
    exactly where :func:`~dmipy_sim.fill.kit.assemble_request` reads them -- a per-pool field window would need a
    per-pool copy of them, which the shared layout does not have), or ``None`` without a field; only the per-pool
    sampling PARAMETERS (``radius``, derived from this pool's own diffusivity) are this pool's own."""
    g = ctx.tests; W = rc.man["walk"]
    D = float(g.pools[pid].D)
    dt_actual = float(blk["h"])
    margin = _reach_margin(plan, geom.cell_size, D, float(blk["T"]))
    struct, arrays = strands_struct_fields_window(geom, lo_m, hi_m, margin)
    req, K_c3, every = _request_lines(rc, pid, blk, geom, plan, field_window is not None)
    lines = dict(engine="strands")
    lines.update(req)
    lines.update(struct)
    lines["adaptive"] = 1
    lines.update({f"adaptive_{k}": v for k, v in adaptive_struct_fields(plan, geom._Rmax).items()})
    if field_window is not None:
        dims_gmin, n_seg, nnz = field_window
        f_reuse = max(1, int(W.get("field_gather_every", 4)))
        f_margin = 6.0 * math.sqrt(2.0 * D * dt_actual * f_reuse)
        f_reach = float(basis.gather_radius_m)
        f_radius = min(f_reach + f_margin, 2.0 * f_reach)
        fstruct = field_struct_fields(basis, f_radius, f_reuse, field_list_k, K_c3=K_c3, read_every=every)
        fstruct.update(dims_gmin)
        fstruct["n_seg"] = int(n_seg); fstruct["nnz"] = int(nnz)
        lines["field"] = 1
        lines.update({f"field_{k}": v for k, v in fstruct.items()})
    return lines, arrays, dict(diffusivity=D, interior=bool(g.inside_w[pid]))


def _block_table(blk):
    """The block table every pool's request is walked against (one window, the pools share it)."""
    from ..replay.compression import block_table
    return np.ascontiguousarray(block_table(float(blk["T"]), int(blk["n_blocks"]), int(blk["K"]), int(blk["m"])), np.float64)


def _walk_record(blk):
    """The kit's ``walk`` record: the walk's duration, its band ``K``, its block (``h``, ``n_blocks``, ``m``) and its
    window table, as a pack states them (``walk_params.T_max``, ``compression.K``, ``block``, ``windows``)."""
    return dict(T_max_s=float(blk["T"]), K=int(blk["K"]), block=dict(h=float(blk["h"]), n_blocks=int(blk["n_blocks"]), m=int(blk["m"])),
                windows=dict(n=int(blk["windows"]["n"]), T=float(blk["windows"]["T"])))


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
    from dmipy_replay.spec.substrate import susceptibility_field_of
    from ..spec.walk import field_source_kind
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    man = rc.man; spec = rc.spec(); ctx = rc.context(); g = ctx.tests
    W = man["walk"]
    if not bool(W.get("adaptive_steps")):
        raise ValueError("a substrate kit serialises the CUDA strands-adaptive engine's request tables; this "
                         "recipe's walk is not adaptive_steps")
    blk = _block_record(rc)
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
        lines, arrays, meta = _pool_lines_and_arrays(rc, ctx, pid, blk, basis, sampling)
        pdir = out / "pools" / pool.name; pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "request.json").write_text(json.dumps(lines, indent=1))
        (pdir / "request.txt").write_text("".join(f"{k}={format_request_value(v)}\n" for k, v in lines.items()))
        for name, a in arrays.items():
            np.ascontiguousarray(a).tofile(pdir / name)
        pools_out[pool.name] = meta
    if not pools_out:
        raise ValueError("no diffusing pool to walk: a kit needs at least one")
    _block_table(blk).tofile(out / "table.f64")

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
              walk=_walk_record(blk), field=dict(present=bool(sampling)), pools=pools_out,
              files={str(p.relative_to(out)): _sha256(p) for p in files})
    json.dump(kit, open(out / "kit.json", "w"), indent=1)
    log.info("kit written to %s: %d pool(s), field %s", out, len(pools_out), sampling)
    return out


def write_block_kit(rc, row, out_dir):
    """The block kit of block ``row`` (:meth:`~dmipy_sim.fill.recipe.Recipe.block_kit`, dmrai-lab/dmipy-sim#695):
    what a node downloads for ONE block, constant in the substrate's size rather than the whole-substrate kit's
    (:func:`write_kit`) -- per pool, the cell window of the block's box plus the walk's reach
    (:func:`_reach_margin`: the walk's WHOLE-duration excursion, since a walker is not confined to its block's
    box, plus the round-level terms), the referenced segments renumbered densely; the field basis's tables cut
    the same way; the far grid cropped to the block's box by :func:`~dmipy_sim.engine.backends.crop_far_grid`'s
    rule; the plan's counts CROPPED to the block's own voxel box (never the whole grid -- a substrate a hundred
    times larger must not carry a hundred-times-larger counts array) with the block's own row and the
    manifest's grid section re-based to that crop's local index frame ``[0, 0, 0)``, so a node reads them
    exactly as it would the whole substrate's; the certificate; a ``kit.json``. Everything a pool's
    :class:`~dmipy_sim.engine.adaptive.AdaptivePlan` needs and every other section (the far grid itself, the
    certificate) is built exactly as :func:`write_kit` builds it -- only the cell-window cropping and the
    plan/grid re-basing are the block kit's own. The ``assemble_request`` layout is IDENTICAL to the whole
    kit's: a node reads either the same way (``tests/fill/test_block_kit.py``)."""
    from dmipy_replay.spec.substrate import susceptibility_field_of
    from ..spec.walk import field_source_kind
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    man = rc.man; spec = rc.spec(); ctx = rc.context(); g = ctx.tests
    W = man["walk"]
    if not bool(W.get("adaptive_steps")):
        raise ValueError("a block kit serialises the CUDA strands-adaptive engine's request tables; this "
                         "recipe's walk is not adaptive_steps")
    blk = _block_record(rc)
    dt_actual = float(blk["h"])
    sampling = field_source_kind(spec) == "strands" and susceptibility_field_of(spec) == "present"
    basis = ctx.field_basis() if sampling else None
    if sampling and (basis is None or basis.far is None):
        raise ValueError("the recipe's variant declares a field but carries no far grid: a kit needs one (the "
                         "per-start cutoff doubling without one needs an actual walk, which a kit is built without)")
    lo_m, hi_m = _block_box(rc, row)

    walked = [(pid, g.pools[pid]) for pid in g.seeded
             if not (g.pools[pid].D in (None, 0.0) or (bool(g.inside_w[pid]) and bool(g.outside_w[pid])))]
    if not walked:
        raise ValueError("no diffusing pool to walk: a kit needs at least one")
    plans = {pid: _pool_plan(rc, ctx, pid, float(pool.D), dt_actual) for pid, pool in walked}

    field_window = None
    if sampling:
        # the field's tables are written ONCE, shared by every pool that samples them (assemble_request reads
        # them from the kit's top level, never per pool): the window must cover every walked pool's own reach,
        # so its margin is the largest of theirs, not any one pool's
        field_margin = max(_reach_margin(plan, float(basis._CS), float(g.pools[pid].D), float(blk["T"]))
                          for pid, (_, plan) in plans.items())
        dims_gmin, farrays = field_tables_window(basis, lo_m, hi_m, field_margin)
        for name, a in farrays.items():
            np.ascontiguousarray(a).tofile(out / name)
        field_window = (dims_gmin, int(farrays["fsid.i32"].shape[0]), int(farrays["fcell_ids.i32"].shape[0]))

        far = basis.far
        D_max = max(float(p.D) for p in spec.pools if p.D)        # the margin's own D: the walk's, the same bound rc.block_h() uses
        crop, far_dims, far_origin = crop_far_grid(far.values, far.origin_m, far.spacing_m, np.array([lo_m, hi_m]),
                                                   float(blk["T"]), D_max)
        np.ascontiguousarray(crop).tofile(out / "far_full.u16")
        # the full array shape (the spatial crop's 3 dims PLUS the 12 channels, write_kit's own "shape" spelling
        # for far_full.json) -- not far_dims (3 only, the DscField struct's own field_far_dims)
        json.dump(dict(origin_m=far_origin, spacing_m=float(far.spacing_m), near_m=float(far.near_m),
                       blend_m=float(far.blend_m), cutoff_m=float(far.cutoff_m), shape=list(crop.shape)),
                 open(out / "far_full.json", "w"), indent=1)

    pools_out = {}
    for pid, pool in walked:
        geom, plan = plans[pid]
        lines, arrays, meta = _pool_lines_and_arrays_window(rc, ctx, pid, blk, basis, field_window, geom, plan, lo_m, hi_m)
        pdir = out / "pools" / pool.name; pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "request.json").write_text(json.dumps(lines, indent=1))
        (pdir / "request.txt").write_text("".join(f"{k}={format_request_value(v)}\n" for k, v in lines.items()))
        for name, a in arrays.items():
            np.ascontiguousarray(a).tofile(pdir / name)
        pools_out[pool.name] = meta
    _block_table(blk).tofile(out / "table.f64")

    os.makedirs(out / "plan", exist_ok=True)
    # the plan's counts cropped to this block's own voxel box (dmrai-lab/dmipy-sim#695 review point 2: a
    # substrate a hundred times larger must not carry a hundred-times-larger counts array into a block kit
    # that is supposed to be constant in the substrate's size) -- the block's row re-based to the LOCAL grid
    # this crop defines (i/j/k from [0, 0, 0)), and the manifest's own grid section re-based the same way, so
    # nothing downstream (RasGrid, block_counts, fill_per_voxel/fill_swept_by_voxel) needs to know it is one
    i0, i1 = row["i"]; j0, j1 = row["j"]; k0, k1 = row["k"]
    plan_full = np.load(rc.hub.get(man["plan"]["file"]))
    plan_cropped = {k: np.ascontiguousarray(plan_full[k][i0:i1, j0:j1, k0:k1]) for k in plan_full.files}
    plan_name = os.path.basename(man["plan"]["file"])
    np.savez(out / "plan" / plan_name, **plan_cropped)

    local_row = dict(row, i=[0, int(i1 - i0)], j=[0, int(j1 - j0)], k=[0, int(k1 - k0)])
    blocks_name = os.path.basename(man["plan"]["blocks"])
    json.dump(dict(blocks=[local_row]), open(out / "plan" / blocks_name, "w"), indent=1)

    grid = rc.grid()
    local_grid = dict(man["grid"], shape=[int(i1 - i0), int(j1 - j0), int(k1 - k0)],
                      origin_m=[float(x) for x in (np.asarray(grid.origin_m) + np.array([i0, j0, k0]) * np.asarray(grid.voxel_size_m))])
    man_local = dict(man, grid=local_grid)
    (out / "manifest.json").write_text(json.dumps(man_local, indent=1))
    cert_path = f"certificate/{rc.variant}.json"
    if rc.hub.exists(cert_path):
        os.makedirs(out / "certificate", exist_ok=True)
        shutil.copy(rc.hub.get(cert_path), out / "certificate" / f"{rc.variant}.json")
    else:
        log.warning("block kit block %s: no certificate/%s.json on the hub yet; the kit carries none", row["block"], rc.variant)

    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "kit.json")
    kit = dict(recipe_id=man.get("id"), commit=man["code"]["commit"], dmipy_sim_version=_dmipy_sim_version(), variant=rc.variant,
              block=int(row["block"]), walk=_walk_record(blk), field=dict(present=bool(sampling)),
              pools=pools_out, files={str(p.relative_to(out)): _sha256(p) for p in files})
    json.dump(kit, open(out / "kit.json", "w"), indent=1)
    log.info("block kit %s written to %s: %d pool(s), field %s", row["block"], out, len(pools_out), sampling)
    return out


def assemble_request(kit_dir, pool, r0, keys, seed, out_dir):
    """A batch's request directory from a written kit (:func:`write_kit`): the pool's template plus this batch's
    ``n`` / ``seed`` / ``raw_subset`` (at most its walkers), its ``r0.f32`` / ``keys.u32``, a crop of the kit's
    uncropped far grid by :func:`crop_far_grid` and the block table ``table.f64`` -- what
    :func:`~dmipy_sim.engine.backends.write_request_directory` writes for the live request. Disk and array work only
    -- no geometry, no spec, nothing a node without dmipy-sim could not do too."""
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
    lines["raw_subset"] = min(int(template["raw_subset"]), n)
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
                                                   float(kit["walk"]["T_max_s"]), kit["pools"][pool]["diffusivity"])
        lines["field_far_dims"] = far_dims; lines["field_far_origin"] = far_origin
    (out / "request.txt").write_text("".join(f"{k}={format_request_value(v)}\n" for k, v in lines.items()))
    shutil.copy(kit_dir / "table.f64", out / "table.f64")
    r0.tofile(out / "r0.f32")
    np.ascontiguousarray(keys, np.uint32).reshape(n, -1)[:, :2].copy().tofile(out / "keys.u32")
    for name in STRAND_ARRAY_NAMES:
        shutil.copy(pdir / name, out / name)
    if "field" in template:
        for name in FIELD_ARRAY_NAMES:
            shutil.copy(kit_dir / name, out / name)
        crop.tofile(out / "far.u16")
    return out
