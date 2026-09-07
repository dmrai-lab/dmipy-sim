"""Reader and writer for the EPFL strand list (``.init`` / ``optimized_final.txt`` of CACTUS, the DiSCo
phantom's strand format; ``read_strands_txt.cpp`` in CACTUS's fibre optimiser).

Layout, whitespace-separated: the voxel side, the number of strands, then per strand its number of control
points followed by one ``x y z r`` line per control point. Lengths are micrometres; the voxel is
``[-side/2, side/2]^3``. The geometry is a sphere-swept polyline per strand
(:class:`~dmipy_sim.geometry.curved_tube.PackedCurvedTubes`); the spec producer is
:func:`dmipy_sim.spec.strands_spec`.
"""
import numpy as np

_UM = 1e-6


def read_strands(path, *, scale=_UM):
    """-> dict(side, centerlines=[(n_k, 3)], radii=[(n_k,)], n_strands), in metres."""
    with open(path) as fh:
        tok = fh.read().split()
    if len(tok) < 2:
        raise ValueError(f"{path}: not a strand list (needs a voxel side and a strand count)")
    side = float(tok[0]) * scale
    n = int(float(tok[1]))
    pos, cls_, rads = 2, [], []
    for k in range(n):
        if pos >= len(tok):
            raise ValueError(f"{path}: declares {n} strands but ends after {k}")
        m = int(float(tok[pos])); pos += 1
        vals = np.asarray(tok[pos:pos + 4 * m], float)
        if len(vals) != 4 * m:
            raise ValueError(f"{path}: strand {k} declares {m} control points but the file ends early")
        pos += 4 * m
        vals = vals.reshape(m, 4)
        cls_.append(vals[:, :3] * scale)
        rads.append(vals[:, 3] * scale)
    return dict(side=side, centerlines=cls_, radii=rads, n_strands=n, path=path, scale=float(scale))


def write_strands(path, centerlines, radii, side, *, scale=_UM):
    """Write a strand list (metres in, micrometres out); ``radii`` per strand is a scalar or per control point."""
    with open(path, "w") as fh:
        fh.write(f"{side / scale:.9g}\n{len(centerlines)}\n")
        for cl, r in zip(centerlines, radii):
            cl = np.asarray(cl, float) / scale
            r = np.broadcast_to(np.asarray(r, float) / scale, (len(cl),))
            fh.write(f"{len(cl)}\n")
            for p, rr in zip(cl, r):
                fh.write(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g} {rr:.9g}\n")
    return path
