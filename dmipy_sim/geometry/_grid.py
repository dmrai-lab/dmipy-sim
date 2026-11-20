"""The uniform cell grid every accelerated geometry shares: primitives bucketed into the cells their bounding
boxes overlap at setup, and the walker's 27-cell gather of those primitives inside the kernels.

A `Mesh` buckets its triangles, `PackedCurvedCylinders` and a `StrandFieldBasis` their segments, a `SphereUnion` its
spheres; on a periodic square (:func:`bucket_periodic_plane`) the concentric wall search and the near list of a
hollow-cylinder field bucket their cylinders. The (primitive, cell) pairs are generated with numpy and sorted once: primitive ids ascending within each
cell, ``-1`` padding, and a fixed overflow rule when a cap is given. The gather is traceable and runs under jit.
"""
import jax.numpy as jnp
import numpy as np

NEIGHBOUR_OFFSETS = np.array([[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)], np.int32)
"""``(27, 3)`` cell offsets of a point's own cell and its 26 neighbours, x-major."""


def gather(CELL, OFF, GMIN, CS, dims, r):
    """The primitive ids in the 27 cells around the point ``r`` and their validity.

    ``CELL`` is the ``(n_cells, C)`` table, ``OFF`` the ``(27, 3)`` offsets, ``GMIN`` the grid origin, ``CS`` the
    cell size and ``dims`` the ``(3,)`` grid dimensions, all device arrays. Returns ``(cand, valid)`` of length
    ``27 C``: the ids with the padding replaced by 0, and the mask of the real entries.
    """
    c = cell_index(GMIN, CS, dims, r)
    nb = jnp.clip(c[None, :] + OFF, 0, dims - 1)
    cids = (nb[:, 0] * dims[1] + nb[:, 1]) * dims[2] + nb[:, 2]
    cand = CELL[cids].reshape(-1)
    valid = cand >= 0
    return jnp.where(valid, cand, 0), valid


def cell_index(GMIN, CS, dims, r):
    """The ``(3,)`` int32 index of the cell holding the point ``r``, clipped to the grid: the cell :func:`gather`
    centres its offsets on."""
    return jnp.clip(jnp.floor((r - GMIN) / CS).astype(jnp.int32), 0, dims - 1)


def wrap_periodic(VMIN, L, PER, r):
    """``r`` wrapped into the box ``[VMIN, VMIN + L)`` on the axes where ``PER > 0``, unchanged on the others."""
    w = VMIN + jnp.mod(r - VMIN, L)
    return jnp.where(PER > 0, w, r)


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


def periodic_spans(lo, hi, G):
    """Per object the cell-index intervals ``[a, b]`` that cover the unwrapped index range ``[lo, hi]`` on a periodic
    axis of ``G`` cells: one interval (the whole axis when the range spans ``G`` cells or more, else ``[lo mod G,
    hi mod G]`` when it does not cross the seam) or two (``[lo mod G, G - 1]`` and ``[0, hi mod G]``). Returns
    ``(a1, b1, a2, b2, two)``; the intervals of one object are disjoint."""
    full = (hi - lo + 1) >= G
    a, b = np.mod(lo, G), np.mod(hi, G)
    two = (~full) & (a > b)
    a1 = np.where(full, 0, a)
    b1 = np.where(full, G - 1, np.where(two, G - 1, b))
    a2 = np.where(two, 0, 0)
    b2 = np.where(two, b, -1)
    return a1, b1, a2, b2, two


def bucket_periodic_plane(c, half, L, G):
    """Cell table ``(G * G, C)`` of object ids on a ``G x G`` grid over the periodic square ``[-L/2, L/2)^2``.

    Cell ``(i, j)`` (id ``i * G + j``, ``i`` along x) covers ``[-L/2 + i s, -L/2 + (i + 1) s)`` x the same in y, ``s
    = L / G``. Object ``k`` (centre ``c`` ``(n, 2)``, metres) is listed in every cell its box ``c_k +- half_k`` meets
    on the cell-index lattice, its index range split at the periodic seam into the intervals of
    :func:`periodic_spans` and bucketed by :func:`bucket_by_bbox`. Ids ascend within a cell, ``-1`` pads a row; ``C``
    is the fullest cell's count."""
    c = np.asarray(c, np.float64)
    half = np.asarray(half, np.float64)
    n = c.shape[0]
    s = L / G
    lo = np.floor((c - half[:, None] + L / 2.0) / s).astype(np.int64)
    hi = np.floor((c + half[:, None] + L / 2.0) / s).astype(np.int64)
    xa1, xb1, xa2, xb2, x2 = periodic_spans(lo[:, 0], hi[:, 0], G)
    ya1, yb1, ya2, yb2, y2 = periodic_spans(lo[:, 1], hi[:, 1], G)
    own, LO, HI = [], [], []
    for (xa, xb, xm) in ((xa1, xb1, np.ones(n, bool)), (xa2, xb2, x2)):
        for (ya, yb, ym) in ((ya1, yb1, np.ones(n, bool)), (ya2, yb2, y2)):
            m = xm & ym
            own.append(np.flatnonzero(m))
            LO.append(np.stack([xa[m], ya[m], np.zeros(m.sum(), np.int64)], 1))
            HI.append(np.stack([xb[m], yb[m], np.zeros(m.sum(), np.int64)], 1))
    own = np.concatenate(own)
    order = np.argsort(own, kind="stable")                # rows by object: ids ascend within a cell
    own = own[order]
    table, _, _, _ = bucket_by_bbox(np.concatenate(LO)[order], np.concatenate(HI)[order], (G, G, 1))
    valid = table >= 0
    return np.where(valid, own[np.where(valid, table, 0)], -1).astype(np.int32)
