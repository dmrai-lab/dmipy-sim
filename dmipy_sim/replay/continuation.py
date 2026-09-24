"""A pack's walk continued in time (RPK.md 3, additivity in time; 4.3, segments): every walker resumed from the last
segment's exact endpoint in the pool it is in, walked for more whole segments on the pack's own substrate with a fresh
seed, and the new windows appended to the pack under the next ``s{i}/`` keys with their own certificates.

    r_end, pool = end_state(pack)                       # where every walker is at the last save, and its pool
    walk = continue_walk(pack, 0.1, seed=17)            # one more segment, in the pack's walker order
    longer = append_segments(pack, walk, seed=17)       # the pack of both, certified as the bound over its segments
    longer = extend_pack(pack, 0.1, seed=17)            # the two in one call
"""
from __future__ import annotations
import dataclasses
import json

import numpy as np

from .replay import ReplayPack, write_rpk


def end_state(pack):
    """``(r_end (n_w, 3), pool (n_w,) or None)``: where every walker is at the walk's last save, read from the last
    segment's exact endpoints (the position codec holds both ends of a window), and the pool it is in there, its
    last occupancy label when the pack carries the compartment channel."""
    from .compression import read_position_coeffs, decode_occupancy
    seg = pack.segment(pack.n_segments - 1)
    C = read_position_coeffs(seg.arrays, dtype=np.float64)
    r_end = np.ascontiguousarray(C[:, 0, :] + C[:, 1, :])
    ch = (pack.meta.get("compression", {}).get("channels") or {})
    pool = None
    if "compartment" in ch:
        comp = np.asarray(decode_occupancy(seg.arrays, ch["compartment"])["comp"])
        pool = comp[:, -1] if comp.ndim == 2 else comp
        if np.issubdtype(pool.dtype, np.floating):
            pool = np.rint(pool)
        pool = pool.astype(np.int64)
    return r_end, pool


def continue_walk(pack, T_add, *, seed, require_gpu=None, field=True, adaptive_steps=False, walker_batch_size=50_000):
    """The walk of ``pack`` continued for ``T_add`` seconds on its own substrate: every walker resumed from the last
    segment's endpoint in the pool it is in, a fresh ``seed``, the pack's save grid and the channels it carries.
    ``T_add`` is a whole number of the pack's segments. A single-geometry spec walks as one; a multi-surface spec
    walks pool by pool, and the walk is returned in the pack's walker order either way, so that walker ``w`` of the
    continuation is walker ``w`` of the pack."""
    from ..engine.core import simulate_trajectories
    from ..spec.build import geometry_from_spec
    from ..spec.walk import walk_spec, _needs_bundle_walk
    from ..spec.seeding import DrawnSeeds
    from ..phantom.grid import Grid
    spec = pack.substrate
    if spec is None:
        raise ValueError("the pack embeds no substrate spec, so there is no substrate to continue the walk on")
    seg = pack.segments
    dt, T_seg = float(pack.dt), float(seg["T"])
    n_add = float(T_add) / T_seg
    if abs(n_add - round(n_add)) > 1e-9 or round(n_add) < 1:
        raise ValueError(f"T_add = {float(T_add):.6g} s is not a whole number of this pack's {T_seg:.6g} s segments")
    r_end, pool = end_state(pack)
    n_w = r_end.shape[0]
    tiers = "all" if (pack.has_relaxation or pack.has_surface) else ()
    if not _needs_bundle_walk(spec):
        g = geometry_from_spec(spec)
        D = pack.diffusivity
        if D is None:
            raise ValueError("the pack records no diffusivity to continue the walk at")
        w = simulate_trajectories(n_w, float(D), g, T_max=float(T_add), dt_save=dt, seed=int(seed), r0=r_end.astype(np.float32),
                                  require_gpu=require_gpu, walker_batch_size=walker_batch_size, tiers=tiers)
        walk = dataclasses.replace(w, geometry=g, spec=spec) if w.spec is None else w
        return walk
    if pool is None:
        raise ValueError("a multi-surface spec is continued pool by pool, and the pack carries no compartment channel to say "
                         "which pool each walker is in")
    names = {p.id: p.name for p in spec.pools}
    weights = np.asarray(pack.spin_weights, np.float64)
    ids = [p.id for p in spec.pools if p.id in spec.seeding.pools and np.any(pool == p.id)]
    order = np.concatenate([np.flatnonzero(pool == pid) for pid in ids])            # the bundle walk's walker order
    box = np.asarray(spec.domain.box_min, float), np.asarray(spec.domain.box_max, float)
    grid = Grid(shape=(1, 1, 1), voxel_size_m=tuple(box[1] - box[0]), origin_m=tuple(box[0]))
    seeds = DrawnSeeds(positions={names[pid]: r_end[pool == pid] for pid in ids},
                       weights={names[pid]: weights[pool == pid] for pid in ids}, grid=grid, seed=int(seed))
    w = walk_spec(spec, T_max=float(T_add), dt_save=dt, seed=int(seed), seeding=seeds, field=field, require_gpu=require_gpu,
                  walker_batch_size=walker_batch_size, adaptive_steps=adaptive_steps, tiers=tiers)
    inv = np.empty_like(order); inv[order] = np.arange(order.size)                     # back to the pack's order
    fields = {k: (None if getattr(w, k) is None else np.asarray(getattr(w, k))[inv]) for k in w._FILE_ARRAYS}
    return dataclasses.replace(w, **fields)


def _container_of(ranges):
    """The builder's container spelling ``((upto, bits), ...)`` of a stored ``[{bands: [k0, k1], bits}, ...]``, the
    last range open-ended."""
    out = [(int(r["bands"][1]), int(r["bits"])) for r in ranges]
    return tuple(out[:-1] + [(None, out[-1][1])])


def append_segments(pack, walk, *, seed, out_path=None, envelope=None, device="auto"):
    """``pack`` with ``walk`` appended as its next segments: the walk (a continuation in the pack's walker order,
    :func:`continue_walk`) built as a pack with the pack's own codec, band, containers and segment length, its
    windows stored under the next ``s{i}/`` keys, the table's ``walks`` gaining the continuation with its ``seed``,
    and the whole's certificate the bound over every segment (:func:`~dmipy_sim.replay.bank.combine_segment_fidelity`)."""
    from .bank import build_replay_pack, combine_segment_fidelity, _is_shared_key, _precision_tiers, _MEASURED_CHANNEL_KEYS
    seg = pack.segments
    cm = pack.meta["compression"]; ch = cm.get("channels") or {}
    if int(walk.n_walkers) != int(pack.n_walkers):
        raise ValueError(f"the continuation walks {walk.n_walkers} walkers, the pack {pack.n_walkers}")
    r_end, _ = end_state(pack)
    if not np.allclose(np.asarray(walk.positions[:, 0, :], np.float64), r_end, atol=1e-9 * max(1.0, float(np.abs(r_end).max()))):
        raise ValueError("the continuation does not start where the pack's walk ended (its first save is not the pack's last)")
    if abs(float(walk.dt) - float(pack.dt)) > 1e-9 * float(pack.dt):
        raise ValueError(f"the continuation's save interval {float(walk.dt):.6g} s is not the pack's {float(pack.dt):.6g} s")
    c2 = ch.get("boundary_local_time") or {}
    pm = ch.get("susceptibility_path") or {}
    kw = dict(K=int(pack.K), segment_T=float(seg["T"]), method=cm.get("method"), envelope=envelope, device=device,
              weights=np.asarray(pack.spin_weights, np.float64), license=pack.license, citation=pack.citation,
              position_container=(None if cm.get("container") is None else _container_of(cm["container"])),
              blt_temporal_K=(int(c2["K"]) if c2 else None), blt_dtype=(np.float16 if c2.get("dtype", "float16") == "float16" else np.float32),
              susc_path_K=(int(pm["K"]) if pm else None), susc_path_bits=(pm.get("bits", 8) if pm else 8),
              _occupancy_runs=("comp_rle_counts" in pack.arrays))
    if c2.get("dtype") == "bands":
        kw["blt_container"] = _container_of(c2["container"])
    new = build_replay_pack(walk, id=f"{pack.id}", field=(True if pm else False), **kw)
    if sorted(k for k in new.segment(0).arrays) != sorted(k for k in pack.segment(0).arrays):
        raise ValueError(f"the continuation stores {sorted(new.segment(0).arrays)} where the pack stores {sorted(pack.segment(0).arrays)}; "
                         "a continuation carries the channels of the pack and no others")
    S0, S1 = int(seg["n"]), int(new.n_segments)
    arrays = dict(pack.arrays)
    for i in range(S1):
        for k, v in new.segment(i).arrays.items():
            if _is_shared_key(k):
                if not np.array_equal(np.asarray(v), np.asarray(arrays[k])):
                    raise ValueError(f"the continuation disagrees with the pack in the shared tensor {k!r}")
                continue
            arrays[f"s{S0 + i}/{k}"] = v
    meta = json.loads(json.dumps(pack.meta))
    n_seg = int(seg["n_t"])
    fid_old = pack.meta.get("fidelity") or {}
    per = list(fid_old.get("segments") or [dict((k, v) for k, v in fid_old.items() if k != "segments")])
    per_new = list((new.meta.get("fidelity") or {}).get("segments") or [dict((k, v) for k, v in new.meta["fidelity"].items() if k != "segments")])
    meta["fidelity"] = combine_segment_fidelity(per + per_new)
    meta["walk_params"].update(n_t=(S0 + S1) * (n_seg - 1) + 1, T_max=(S0 + S1) * float(seg["T"]),
                               segments=dict(seg, n=S0 + S1, walks=list(seg.get("walks") or []) + [dict(first=S0, last=S0 + S1 - 1, seed=int(seed))]))
    chans = meta["compression"].get("channels") or {}
    for c, mm in chans.items():
        if isinstance(mm, dict):
            for k in _MEASURED_CHANNEL_KEYS:
                v = ((new.meta["compression"].get("channels") or {}).get(c) or {}).get(k)
                if v is not None and mm.get(k) is not None:
                    mm[k] = float(max(mm[k], v))
    if meta["compression"].get("walker_preserving"):
        meta["compression"]["precision_tiers"] = _precision_tiers(arrays, int(pack.n_walkers), float(meta["fidelity"].get("floor_max") or 0.0),
                                                                  bool(meta["compression"].get("precision_tiers", {}).get("walkers_shuffled", False)))
    prov = meta.setdefault("provenance", {})
    prov["continuations"] = list(prov.get("continuations") or []) + [dict(first=S0, last=S0 + S1 - 1, seed=int(seed),
                                                                           run=(new.meta.get("provenance") or {}).get("run"))]
    out = ReplayPack(arrays, meta, source=out_path)
    if out_path is not None:
        write_rpk(out_path, {k: v for k, v in arrays.items() if v is not None}, meta)
    return out


def extend_pack(pack, T_add, *, seed, out_path=None, require_gpu=None, field=True, envelope=None, device="auto"):
    """:func:`continue_walk` then :func:`append_segments`: the pack lengthened by ``T_add`` seconds of new segments."""
    walk = continue_walk(pack, T_add, seed=seed, require_gpu=require_gpu, field=field)
    return append_segments(pack, walk, seed=seed, out_path=out_path, envelope=envelope, device=device)
