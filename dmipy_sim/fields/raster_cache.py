"""The field raster of a spec on disk: ``<cache_dir>/<key>.safetensors``.

A raster (:class:`~dmipy_sim.fields.susceptibility_field.FieldGrid`, from
:func:`dmipy_sim.spec.field_grid_of_spec`) is a function of the substrate spec, the node spacing, the certificate's
sampling and the build (:data:`FIELD_RASTER_FORMAT`) alone, and costs minutes to hours to make. Its key is the sha256
of those; the file holds the basis channels (float64, as built), the grid's placement, periodicity and certificate,
the key, and the sha256 of the channels' bytes, so a load verifies that it is the raster it is asked for and that it
is intact. A file that is not is refused by name, never rebuilt over.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

#: The version of the raster's build -- the mask's partial-volume sampling, the k-space window, the director, the
#: dtype of the channels. A change to what the build computes changes this string, and every key with it.
FIELD_RASTER_FORMAT = "field-raster-1"
_META_KEY = "field_raster"
_GROUPS = ("iso_local", "iso_P", "aniso_G")


def field_raster_key(spec, res, *, n_check=400_000, seed=0):
    """The key (a sha256 hex string) of the raster of ``spec`` at node spacing ``res`` (metres), its certificate
    sampled with ``n_check`` points of ``seed``: the canonical JSON of the spec, ``res``, those, the build's
    constants and :data:`FIELD_RASTER_FORMAT`."""
    from .susceptibility_field import FIELD_MASK_SUPERSAMPLE
    doc = dict(spec=spec.to_dict(), field_res=float(res), n_check=int(n_check), seed=int(seed),
               mask_supersample=int(FIELD_MASK_SUPERSAMPLE), format=FIELD_RASTER_FORMAT)
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()


def field_raster_path(cache_dir, key):
    """``<cache_dir>/<key>.safetensors``."""
    return os.path.join(os.path.abspath(str(cache_dir)), key + ".safetensors")


def _content_sha256(tensors):
    """The sha256 of the tensors' names, dtypes, shapes and bytes, in name order."""
    h = hashlib.sha256()
    for name in sorted(tensors):
        a = np.ascontiguousarray(tensors[name])
        h.update(f"{name}|{a.dtype.str}|{list(a.shape)}|".encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def save_field_grid(grid, path, key):
    """``grid`` written to ``path`` (atomically) as the raster of ``key``; returns the sha256 of its channels."""
    from safetensors.numpy import save_file
    basis = grid.basis
    tensors = {"iso_local": np.ascontiguousarray(basis["iso_local"], np.float64),
               "iso_P": np.ascontiguousarray(basis["iso_P"], np.float64)}
    if basis.get("aniso_G") is not None:
        tensors["aniso_G"] = np.ascontiguousarray(basis["aniso_G"], np.float64)
    sha = _content_sha256(tensors)
    header = dict(format=FIELD_RASTER_FORMAT, key=key, content_sha256=sha,
                  origin=np.asarray(grid.origin, float).tolist(), voxel_size=np.asarray(basis["voxel_size"], float).tolist(),
                  shape=[int(s) for s in basis["shape"]], periodic=[bool(p) for p in grid.periodic],
                  shell_fraction=(None if basis.get("shell_fraction") is None else float(basis["shell_fraction"])),
                  certificate=grid.certificate)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(tensors, path + ".part", metadata={_META_KEY: json.dumps(header)})
    os.replace(path + ".part", path)
    return sha


def read_field_raster_header(path):
    """The header of the raster file at ``path`` (key, content sha256, placement, certificate), no channels read."""
    from safetensors import safe_open
    with safe_open(str(path), framework="numpy") as f:
        raw = (f.metadata() or {}).get(_META_KEY)
    if raw is None:
        raise ValueError(f"{path} is not a field raster file (no {_META_KEY!r} header)")
    return json.loads(raw)


def load_field_grid(path, key):
    """The :class:`~dmipy_sim.fields.susceptibility_field.FieldGrid` in the raster file at ``path``, refused unless
    it is the raster of ``key`` (the key it records, the file's name, the build format) and its channels hash to the
    sha256 it records."""
    from safetensors.numpy import load_file
    from .susceptibility_field import FieldGrid
    h = read_field_raster_header(path)
    stem = os.path.basename(str(path))[:-len(".safetensors")] if str(path).endswith(".safetensors") else None
    if h.get("key") != key or (stem is not None and stem != key):
        raise ValueError(f"field raster {path} records key {h.get('key')!r} and is named {stem!r}, but the raster asked "
                         f"for is {key!r}: it is another raster; remove the file to build this one")
    if h.get("format") != FIELD_RASTER_FORMAT:
        raise ValueError(f"field raster {path} is of build format {h.get('format')!r}, this code builds "
                         f"{FIELD_RASTER_FORMAT!r}: remove the file to rebuild it")
    tensors = load_file(str(path))
    got = _content_sha256(tensors)
    if got != h["content_sha256"]:
        raise ValueError(f"field raster {path} is corrupt: its channels hash to {got}, the file records "
                         f"{h['content_sha256']}; remove it to rebuild it")
    shape = tuple(int(s) for s in h["shape"])
    basis = dict(iso_local=tensors["iso_local"], iso_P=tensors["iso_P"], aniso_G=tensors.get("aniso_G"), shape=shape,
                 voxel_size=np.asarray(h["voxel_size"], float))
    if h.get("shell_fraction") is not None:
        basis["shell_fraction"] = float(h["shell_fraction"])
    return FieldGrid(basis, np.asarray(h["origin"], float), h["certificate"], periodic=tuple(bool(p) for p in h["periodic"]))


def field_raster_cache_record(spec, res, cache_dir, *, n_check=400_000, seed=0):
    """``{dir, key, path, exists, content_sha256, bytes}`` of the raster of ``spec`` at ``res`` in ``cache_dir``
    (``content_sha256`` and ``bytes`` ``None`` while it is not there): what a receipt records of the cache."""
    key = field_raster_key(spec, res, n_check=n_check, seed=seed)
    path = field_raster_path(cache_dir, key)
    here = os.path.exists(path)
    return dict(dir=os.path.abspath(str(cache_dir)), key=key, path=path, exists=here,
                content_sha256=(read_field_raster_header(path)["content_sha256"] if here else None),
                bytes=(os.path.getsize(path) if here else None))


def cached_field_grid(spec, plan, cache_dir, build, *, n_check=400_000, seed=0):
    """The raster of ``spec`` for ``plan`` (:class:`~dmipy_sim.spec.walk.FieldRasterPlan`): loaded from ``cache_dir``
    when its file is there (:func:`load_field_grid`), else made by ``build()`` and written there."""
    key = field_raster_key(spec, plan.res, n_check=n_check, seed=seed)
    path = field_raster_path(cache_dir, key)
    if os.path.exists(path):
        return load_field_grid(path, key)
    grid = build()
    save_field_grid(grid, path, key)
    return grid
