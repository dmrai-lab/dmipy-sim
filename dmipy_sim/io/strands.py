"""Reader and writer for the EPFL strand list (``.init`` / ``optimized_final.txt`` of CACTUS, the DiSCo
phantom's strand format; ``read_strands_txt.cpp`` in CACTUS's fibre optimiser).

Layout, whitespace-separated: the voxel side, the number of strands, then per strand its number of control
points followed by one ``x y z r`` line per control point. Lengths are micrometres; the voxel is
``[-side/2, side/2]^3``. The geometry is a sphere-swept polyline per strand
(:class:`~dmipy_sim.geometry.curved_cylinder.PackedCurvedCylinders`); the spec producer is
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


# ----------------------------------------------------------------------------- MRtrix tracks (DiSCo's release)
def read_tck(path, *, coordinate_unit_m):
    """The centerlines of an MRtrix ``.tck`` track file as a list of ``(n_k, 3)`` arrays in metres: the stored
    coordinates times ``coordinate_unit_m`` (DiSCo, Rafael-Patino et al. 2021, stores its strands in units of the
    ground-truth voxel, 25 um). The format: a text header (``key: value`` lines, ``file: . <offset>``,
    ``datatype``) up to ``END``, then float triplets, a ``NaN`` triplet between tracks and an ``Inf`` triplet at
    the end."""
    with open(path, "rb") as fh:
        head = b""
        while not head.endswith(b"END\n"):
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: no END line; not an MRtrix track file")
            head += line
    hdr = {}
    for line in head.decode("ascii", "replace").splitlines():
        if ":" in line:
            k, v = line.split(":", 1); hdr[k.strip()] = v.strip()
    if not head.startswith(b"mrtrix tracks"):
        raise ValueError(f"{path}: not an MRtrix track file (no 'mrtrix tracks' magic)")
    dtype = {"Float32LE": "<f4", "Float32BE": ">f4", "Float64LE": "<f8", "Float64BE": ">f8"}.get(hdr.get("datatype", ""))
    if dtype is None:
        raise ValueError(f"{path}: datatype {hdr.get('datatype')!r} is not a float triplet type")
    file_field = hdr.get("file", "")
    if not file_field.startswith("."):
        raise ValueError(f"{path}: tracks in another file ({file_field!r}) are not read")
    offset = int(file_field.split()[1])
    data = np.fromfile(path, dtype=dtype, offset=offset)
    if data.size % 3:
        raise ValueError(f"{path}: {data.size} floats do not form triplets")
    pts = data.reshape(-1, 3).astype(np.float64)
    is_nan = np.isnan(pts).all(1); is_inf = np.isinf(pts).all(1)
    end = int(np.argmax(is_inf)) if is_inf.any() else len(pts)
    breaks = np.flatnonzero(is_nan[:end])
    starts = np.r_[0, breaks + 1]; stops = np.r_[breaks, end]
    cls_ = [pts[s:e] * float(coordinate_unit_m) for s, e in zip(starts, stops) if e > s]
    return cls_


def write_tck(path, centerlines, *, coordinate_unit_m):
    """Write centerlines (metres) as an MRtrix ``.tck`` in ``coordinate_unit_m`` units (Float32LE)."""
    body = []
    for c in centerlines:
        body.append(np.asarray(c, np.float64) / float(coordinate_unit_m)); body.append(np.full((1, 3), np.nan))
    body.append(np.full((1, 3), np.inf))
    arr = np.vstack(body).astype("<f4")
    lines = ["mrtrix tracks", f"count: {len(centerlines)}", "datatype: Float32LE"]
    head = "\n".join(lines) + "\n"
    offset = len(head) + len("file: . ") + 8 + 1 + len("END\n")            # the offset line's own width, fixed at 8 digits
    head += f"file: . {offset:8d}\nEND\n"
    with open(path, "wb") as fh:
        fh.write(head.encode("ascii")); fh.write(arr.tobytes())


def read_diameters(path, *, diameter_unit_m):
    """One diameter per strand (a whitespace-separated column), times ``diameter_unit_m``, in metres."""
    d = np.loadtxt(path, dtype=np.float64).reshape(-1) * float(diameter_unit_m)
    if not (d > 0).all():
        raise ValueError(f"{path}: every diameter must be positive")
    return d
