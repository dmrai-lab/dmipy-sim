"""Setup-time bucketing of primitives into a uniform grid, vectorised.

A `Mesh` buckets its triangles and `PackedCurvedTubes` its segments into the cells their
bounding boxes overlap, so a walker's 27-cell gather tests only what lies near it. The
per-primitive Python triple loop that did this cost seconds on a 20k-triangle mesh; here the
(primitive, cell) pairs are generated with numpy and sorted once. The result is the same table:
primitive ids in ascending order within each cell, ``-1`` padding, and the same overflow rule when
a cap is given.
"""
import numpy as np


def bucket_by_bbox(lo, hi, dims, cap=None):
    """Cell table ``(n_cells, C)`` of primitive ids from their cell-index bounding boxes.

    ``lo``/``hi`` are ``(n, 3)`` integer cell indices (inclusive) already clipped to ``dims``.
    Returns ``(cell_tri, C, max_occ, overflow)``: the padded table, its width (``max_occ`` or
    ``cap``), the fullest cell's count, and the number of ids dropped by ``cap``.
    """
    lo = np.asarray(lo, np.int64)
    hi = np.asarray(hi, np.int64)
    dims = np.asarray(dims, np.int64)
    n = lo.shape[0]
    n_cells = int(np.prod(dims))
    if n == 0:
        C = 1 if cap is None else int(cap)
        return np.full((n_cells, C), -1, np.int32), C, 0, 0
    span = hi - lo + 1                                            # (n, 3) cells per axis
    per = span.prod(1)                                            # cells per primitive
    prim = np.repeat(np.arange(n), per)                           # primitive id of each pair
    # local (ix, iy, iz) within each primitive's box, in the x-major / y / z-minor order of the
    # loop this replaces, so ids stay ascending within a cell after a stable sort
    off = np.arange(per.sum()) - np.repeat(np.cumsum(per) - per, per)
    sp = span[prim]
    iz = off % sp[:, 2]
    iy = (off // sp[:, 2]) % sp[:, 1]
    ix = off // (sp[:, 2] * sp[:, 1])
    cx = lo[prim, 0] + ix
    cy = lo[prim, 1] + iy
    cz = lo[prim, 2] + iz
    cid = (cx * dims[1] + cy) * dims[2] + cz
    order = np.argsort(cid, kind="stable")                        # keeps ascending prim per cell
    cid_s = cid[order]
    prim_s = prim[order]
    counts = np.bincount(cid_s, minlength=n_cells)
    max_occ = int(counts.max()) if counts.size else 0
    C = max_occ if cap is None else int(cap)
    C = max(C, 1)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    rank = np.arange(cid_s.size) - starts                         # position within the cell
    keep = rank < C
    overflow = int((~keep).sum())
    cell_tri = np.full((n_cells, C), -1, np.int32)
    cell_tri[cid_s[keep], rank[keep]] = prim_s[keep].astype(np.int32)
    return cell_tri, C, max_occ, overflow
