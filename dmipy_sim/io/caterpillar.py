"""Reader for CATERPillar substrate tables (Nguyen-Duc et al. 2026, https://github.com/jazz031195/CATERPillar).

CATERPillar grows white matter as a **union of overlapping spheres**: every axon, glial process and vessel
is a chain of spheres whose centres carry tortuosity and whose radii carry beading; a sphere stores an inner
and an outer radius, so myelin is explicit per sphere (the shell where ``outer > inner``). Three file
generations are read here by **header name**:

    ``.swc``  id_ax id_sph id_branch Type X Y Z Rin Rout P
    ``.csv``  cell_type cell_id component component_id X Y Z inner_radius outer_radius
    ``.csv``  ... + parent_component_id   (current CATERPillar)

Lengths in the files are micrometres. This module parses and converts; the geometry is
:class:`~dmipy_sim.geometry.sphere_union.SphereUnion` and the spec producer is
:func:`dmipy_sim.spec.caterpillar_spec`.
"""
import json
import os

import numpy as np

_UM = 1e-6

_ALIASES = {
    "cell_type": "kind", "type": "kind",
    "cell_id": "cell_id", "id_ax": "cell_id",
    "component": "component", "component_id": "comp_id", "id_branch": "comp_id",
    "id_sph": "sphere_id", "parent_component_id": "parent_id", "p": "parent_id",
    "x": "x", "y": "y", "z": "z",
    "inner_radius": "r_in", "rin": "r_in",
    "outer_radius": "r_out", "rout": "r_out",
}
CELL_TYPES = ("axon", "glial_cell", "blood_vessel")


def canonical_cell_type(names):
    """CATERPillar's cell-type strings onto ``axon`` / ``glial_cell`` / ``blood_vessel`` (``""`` unknown)."""
    names = np.asarray(names).astype(str)
    lowered = np.char.lower(names)
    out = np.full(len(names), "", dtype="<U12")
    out[np.char.find(lowered, "axon") >= 0] = "axon"
    out[np.char.find(lowered, "glia") >= 0] = "glial_cell"
    out[np.char.find(lowered, "blood") >= 0] = "blood_vessel"
    out[np.char.find(lowered, "vessel") >= 0] = "blood_vessel"
    return out


def parse_growth_info(path):
    """``<name>_growth_info.txt`` -> dict; values are floats, lists of floats or strings. Keys can contain
    digits, so the value is the maximal trailing run of numeric tokens."""
    info = {}
    with open(path) as fh:
        for line in fh:
            tok = line.split()
            if len(tok) < 2:
                continue
            k = len(tok)
            while k > 1:
                try:
                    float(tok[k - 1])
                except ValueError:
                    break
                k -= 1
            key, vals = " ".join(tok[:k]), tok[k:]
            if not vals:
                info[" ".join(tok[:-1])] = tok[-1]
                continue
            num = [float(v) for v in vals]
            info[key] = num[0] if len(num) == 1 else num
    return info


def read_table(path):
    """Header-driven read -> dict of canonical column arrays, in FILE units."""
    with open(path) as fh:
        header = fh.readline().split()
    canon = [_ALIASES.get(h.lower()) for h in header]
    if "x" not in canon or "r_in" not in canon:
        raise ValueError(f"{path}: unrecognised CATERPillar header {header!r}")
    num_cols = [i for i, c in enumerate(canon) if c in ("cell_id", "comp_id", "sphere_id", "parent_id", "x", "y", "z",
                                                        "r_in", "r_out")]
    str_cols = [i for i, c in enumerate(canon) if c in ("kind", "component")]
    num = np.loadtxt(path, skiprows=1, usecols=num_cols, dtype=np.float64, ndmin=2)
    out = {canon[c]: num[:, j] for j, c in enumerate(num_cols)}
    if str_cols:
        txt = np.loadtxt(path, skiprows=1, usecols=str_cols, dtype=str, ndmin=2)
        for j, c in enumerate(str_cols):
            out[canon[c]] = txt[:, j]
    return out


def read_caterpillar(path, *, scale=_UM, cell_types=("axon", "glial_cell"), growth_info=None, config=None):
    """A CATERPillar ``.csv`` / ``.swc`` as arrays in metres.

    Returns a dict: ``centers`` (n, 3), ``r_in``, ``r_out`` (n,), ``cell_type`` (n,) canonical strings,
    ``cell_id``, ``comp_id`` (n,) int, ``box_min`` / ``box_max`` (3,) and ``params`` (growth info + config,
    verbatim). The voxel is the growth info's "Small voxel" limits, else the configured edge length, else
    the sphere bounding box (``box_source`` says which). ``cell_types`` selects the rows kept; vessels are
    out by default (no flow model, so a vessel lumen is not a diffusing pool).
    """
    stem = os.path.splitext(path)[0]
    params = {}
    gi = growth_info or (stem + "_growth_info.txt")
    if os.path.exists(gi):
        params.update(parse_growth_info(gi))
    cfg = config or (stem + ".json")
    if os.path.exists(cfg):
        try:
            with open(cfg) as fh:
                params["config"] = json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    t = read_table(path)
    n = len(t["x"])
    kind = canonical_cell_type(t["kind"]) if "kind" in t else np.full(n, "axon", dtype="<U12")
    keep = np.isin(kind, list(cell_types))
    centers = np.stack([t["x"], t["y"], t["z"]], axis=1)[keep] * scale
    r_in = t["r_in"][keep] * scale
    r_out = (t["r_out"] if "r_out" in t else t["r_in"])[keep] * scale
    cell_id = t.get("cell_id", np.zeros(n))[keep].astype(np.int64)
    comp_id = t.get("comp_id", np.zeros(n))[keep].astype(np.int64)
    if "Small voxel min limits" in params:
        box_min = np.asarray(params["Small voxel min limits"], float) * scale
        box_max = np.asarray(params["Small voxel max limits"], float) * scale
        src = "growth_info 'Small voxel' limits"
    elif params.get("config", {}).get("GeneralParameters", {}).get("VoxelEdgeLength") is not None:
        side = float(params["config"]["GeneralParameters"]["VoxelEdgeLength"]) * scale
        box_min, box_max = np.zeros(3), np.full(3, side)
        src = "config VoxelEdgeLength cube from the origin"
    elif "Voxel Size" in params or "Voxel" in params:
        side = float(params.get("Voxel Size", params.get("Voxel"))) * scale
        box_min, box_max = np.zeros(3), np.full(3, side)
        src = "growth_info voxel size cube from the origin"
    else:
        box_min = (centers - r_out[:, None]).min(axis=0)
        box_max = (centers + r_out[:, None]).max(axis=0)
        src = "sphere bounding box (no voxel recorded)"
    return dict(centers=centers, r_in=r_in, r_out=r_out, cell_type=kind[keep], cell_id=cell_id, comp_id=comp_id,
                box_min=box_min, box_max=box_max, box_source=src, params=params, path=path, scale=float(scale))


def points_inside_union(centers, radii, points, workers=-1):
    """(n, 3) -> (n,) bool: is each point inside the union of these spheres? Host-side and exact: a KD-tree
    over the PROBE points and one ball query per sphere, so the work scales with the points a sphere covers
    (glial processes carry ~20 spheres per micron)."""
    from scipy.spatial import cKDTree
    centers = np.asarray(centers, float).reshape(-1, 3)
    radii = np.asarray(radii, float).ravel()
    points = np.asarray(points, float).reshape(-1, 3)
    out = np.zeros(len(points), bool)
    if len(radii) == 0 or len(points) == 0:
        return out
    tree = cKDTree(points)
    hits = tree.query_ball_point(centers, radii, workers=workers)
    flat = [np.asarray(h, np.int64) for h in hits if len(h)]
    if flat:
        out[np.concatenate(flat)] = True
    return out


def write_caterpillar(path, centers, r_in, r_out, cell_type, cell_id, comp_id, *, scale=_UM):
    """Write the current CATERPillar ``.csv`` layout (metres in, micrometres out); the tests' fixture writer."""
    centers = np.asarray(centers, float) / scale
    with open(path, "w") as fh:
        fh.write("cell_type cell_id component component_id parent_component_id X Y Z inner_radius outer_radius\n")
        for k in range(len(centers)):
            comp = "axon" if cell_type[k] == "axon" else ("soma" if cell_type[k] == "glial_cell" else "blood_vessel")
            fh.write(f"{cell_type[k]} {int(cell_id[k])} {comp} {int(comp_id[k])} 0 "
                     f"{centers[k, 0]:.9g} {centers[k, 1]:.9g} {centers[k, 2]:.9g} "
                     f"{r_in[k] / scale:.9g} {r_out[k] / scale:.9g}\n")
    return path
