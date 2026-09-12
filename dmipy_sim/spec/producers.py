"""Spec producers for the datasets: they emit a :class:`SubstrateSpec` and construct nothing. Mesh runs (CACTUS,
Winther), sphere-grown cells (CATERPillar) and strand lists (CACTUS / DiSCo).

Every decision the loaders used to take in code is a field here, with its reason recorded in
``provenance.transformations``: which surface is which pool, the box and what its faces do, the
measured g-ratio, dropped open surfaces, the nominal pool values (from the catalogued white matter,
``substrate.biophysical_constants.canonical_white_matter``), and the unit scale.
"""
import glob
import hashlib
import os
import re
from datetime import date

import numpy as np

from .substrate import (SubstrateSpec, Domain, Frame, Pool, Susceptibility, Surface, Wall, Directional, Sided,
                        Seeding, Validity, SpecError)

_UM = 1e-6


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wm_pools(field_T=3.0, *, myelin_chi_iso=None, myelin_chi_aniso=None, D_myelin=0.0):
    """The three white-matter pools with the catalogued nominal values; myelin is the field source, with the
    catalogue's chi unless a dataset's own convention is given."""
    from ..substrate.biophysical_constants import canonical_white_matter, get_default_value
    p = canonical_white_matter(field_T=field_T)
    if myelin_chi_iso is None:
        myelin_chi_iso = p["chi_iso_myelin"]
    if myelin_chi_aniso is None:
        myelin_chi_aniso = p["delta_chi_a"]
    wf_m = float(get_default_value("myelin_water_proton_density"))
    return [Pool(0, "extra", float(p["D_extra"]), water_fraction=1.0, T2=float(p["T2_extra"]), T1=float(p.get("T1_extra", 1.0))),
            Pool(1, "intra", float(p["D_intra"]), water_fraction=1.0, T2=float(p["T2_intra"]), T1=float(p.get("T1_intra", 1.2))),
            Pool(2, "myelin", float(D_myelin), water_fraction=wf_m, T2=float(p["T2_myelin"]), T1=float(p.get("T1_myelin", 0.44)),
                 susceptibility=Susceptibility(float(myelin_chi_iso), float(myelin_chi_aniso), "radial"))]


def _volume(V, F):
    """Enclosed volume of a closed triangle surface (divergence theorem; orientation-independent)."""
    V = np.asarray(V, float); F = np.asarray(F)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def _g_ratio(inner, outer):
    """The g-ratio a pair of tube surfaces realises: for a tube V ~ r^2 L, so g = sqrt(V_in / V_out). Measured,
    never taken from an argument (the Winther axons realise 0.70; CACTUS strands 0.56-0.87)."""
    return float(np.sqrt(_volume(*inner) / _volume(*outer)))


def _surface_stats(paths, scale):
    """Per-file (V, F) in metres, the smallest edge-based feature, the median edge and watertightness."""
    from ..geometry.mesh import load_ply
    feats, edges, open_files, meshes = [], [], [], []
    for p in paths:
        V, F = load_ply(p, scale=scale)
        meshes.append((V, F))
        e = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
        edges.append(np.median(e))
        ext = V.max(0) - V.min(0)
        feats.append(0.5 * float(np.sort(ext)[0]))                # half the thinnest extent: the radius of a tube
        try:
            import trimesh
            if not trimesh.Trimesh(V, F, process=False).is_watertight:
                open_files.append(p)
        except ImportError:
            pass
    return meshes, float(min(feats)), float(np.median(edges)), open_files


def _walls(pairs, scale, rho_inner, rho_outer):
    walls = []
    for k, (inner, outer) in pairs.items():
        walls.append(Wall(f"fibre-{k}/inner", Surface("mesh", file=inner, format=inner.rsplit(".", 1)[-1].lower(), scale=scale, sha256=_sha(inner)),
                          1, 2, Directional(), Sided(rho_inner, 0.0)))
        walls.append(Wall(f"fibre-{k}/outer", Surface("mesh", file=outer, format=outer.rsplit(".", 1)[-1].lower(), scale=scale, sha256=_sha(outer)),
                          2, 0, Directional(), Sided(0.0, rho_outer)))
    return walls


def cactus_spec(run_dir, *, scale=_UM, side_um=None, field_T=3.0, rho2=None, on_open_surface="drop", id=None):
    """The spec of a CACTUS run directory (``optimized_final.txt`` + ``meshes/simulations/strand_*_erode_*.ply``).

    The periodic cell comes from the header (or ``side_um``); every strand with both surfaces is two
    walls (inner: intra | myelin, outer: myelin | extra); the pools are the catalogued white matter;
    open (non-watertight) surfaces are dropped, warned about or refused per ``on_open_surface``, and
    every such decision is recorded in ``provenance.transformations``.
    """
    from ..substrate.biophysical_constants import canonical_white_matter
    sim = os.path.join(run_dir, "meshes", "simulations")
    pat = re.compile(r"strand_(\d+)_(inner|outer)_erode_\d+\.ply$")
    found = {}
    for p in sorted(glob.glob(os.path.join(sim, "strand_*_erode_*.ply"))):
        m = pat.search(os.path.basename(p))
        if m:
            found.setdefault(int(m.group(1)), {})[m.group(2)] = p
    pairs = {k: (v["inner"], v["outer"]) for k, v in found.items() if "inner" in v and "outer" in v}
    transformations = []
    dropped_unpaired = sorted(set(found) - set(pairs))
    if dropped_unpaired:
        transformations.append(f"dropped {len(dropped_unpaired)} strand(s) lacking an inner or outer surface: {dropped_unpaired[:8]}")
    if not pairs:
        raise SpecError(f"no paired strand surfaces under {sim}")
    if side_um is None:
        hdr = os.path.join(run_dir, "optimized_final.txt")
        if not os.path.isfile(hdr):
            raise SpecError("no optimized_final.txt header and no side_um: the periodic cell is unknown")
        with open(hdr) as fh:
            side_um = float(fh.readline().strip())
        transformations.append("periodic cell side read from optimized_final.txt")
    L = float(side_um) * scale
    files = [p for pr in pairs.values() for p in pr]
    meshes, smallest, edge_med, open_files = _surface_stats(files, scale)
    by_path = dict(zip(files, meshes))
    if open_files:
        if on_open_surface == "raise":
            raise SpecError(f"{len(open_files)} surface(s) are not watertight: {open_files[:4]}")
        if on_open_surface == "drop":
            bad = {k for k, pr in pairs.items() if pr[0] in open_files or pr[1] in open_files}
            pairs = {k: v for k, v in pairs.items() if k not in bad}
            transformations.append(f"dropped {len(bad)} strand(s) with a non-watertight surface: {sorted(bad)[:8]}")
            if not pairs:
                raise SpecError("every strand has an open surface")
        else:
            transformations.append(f"{len(open_files)} non-watertight surface(s) kept (on_open_surface='warn')")
    rho = float(rho2 if rho2 is not None else canonical_white_matter(field_T=field_T)["rho2"])
    pools = wm_pools(field_T)
    walls = _walls(pairs, scale, rho, rho)
    lo, hi = [0.0, 0.0, 0.0], [L, L, L]
    vmin = np.min([m[0].min(0) for m in meshes], axis=0)
    if vmin.min() < -1e-9:                                       # centred cell
        lo, hi = [-L / 2] * 3, [L / 2] * 3
        transformations.append("periodic cell centred on the origin (surfaces have negative coordinates)")
    spec = SubstrateSpec(
        id or f"cactus/{os.path.basename(os.path.normpath(run_dir))}",
        Domain(lo, hi, ["periodic", "periodic", "periodic"]), pools, walls, Seeding([0, 1, 2], "uniform_by_volume", "thin"),
        Validity(smallest, ["gradient", "relaxation", "surface", "field"], mesh_edge_feature_ratio=edge_med / smallest),
        frame=Frame([0.0, 0.0, 1.0]), nominal_field_T=float(field_T),
        description=f"CACTUS bundle: {len(pairs)} strands, each an inner (axon) and outer (myelin) surface, in a periodic cell",
        realisation={"n_objects": len(pairs), "cell_side": L,
                     "g_ratio": {str(k): _g_ratio(by_path[v[0]], by_path[v[1]]) for k, v in pairs.items()}},   # JSON keys
        provenance={"source": "CACTUS", "run_dir": str(run_dir), "scale": scale,
                    "files": [{"path": p, "sha256": _sha(p)} for p in files],
                    "transformations": transformations + ["inside inner = intra (1), inner..outer = myelin (2), outside outer = extra (0)",
                                                          "nominal pool values from the catalogued white matter"],
                    "created": date.today().isoformat(), "software": {"name": "dmipy-sim", "version": _version()}})
    return spec.validate()


def winther_spec(inner_ply, outer_ply, *, scale=_UM, pad=1.0e-6, field_T=3.0, rho2=None, id=None):
    """The spec of one Winther axon: inner + outer surface in an open box padded by ``pad``; the
    surroundings are free water, so only intra and myelin are seeded."""
    from ..substrate.biophysical_constants import canonical_white_matter
    meshes, smallest, edge_med, open_files = _surface_stats([inner_ply, outer_ply], scale)
    if open_files:
        raise SpecError(f"an isolated axon needs closed surfaces; open: {open_files}")
    Vo = meshes[1][0]
    lo = (Vo.min(0) - pad).tolist(); hi = (Vo.max(0) + pad).tolist()
    rho = float(rho2 if rho2 is not None else canonical_white_matter(field_T=field_T)["rho2"])
    pools = wm_pools(field_T, myelin_chi_iso=1.06e-6, myelin_chi_aniso=0.0)       # the dataset's own convention
    pools[0] = Pool(0, "extra", pools[0].D, water_fraction=0.0, T2=pools[0].T2, T1=pools[0].T1)     # free water, not substrate
    walls = _walls({0: (inner_ply, outer_ply)}, scale, rho, rho)
    spec = SubstrateSpec(
        id or f"winther/{os.path.splitext(os.path.basename(inner_ply))[0]}",
        Domain(lo, hi, ["open", "open", "open"]), pools, walls, Seeding([1, 2], "uniform_by_volume", "thin"),
        Validity(smallest, ["gradient", "relaxation", "surface", "field"], mesh_edge_feature_ratio=edge_med / smallest),
        nominal_field_T=float(field_T),
        description="one Winther axon: inner and outer surface, surroundings free water",
        realisation={"g_ratio": _g_ratio(meshes[0], meshes[1])},
        provenance={"source": "Winther", "scale": scale, "files": [{"path": p, "sha256": _sha(p)} for p in (inner_ply, outer_ply)],
                    "transformations": [f"box = outer surface padded by {pad} m", "extra pool declared free water (water_fraction 0, not seeded)",
                                        "nominal pool values from the catalogued white matter",
                                        "myelin chi_iso = +1.06e-6, isotropic: the convention the Winther meshes were published with"],
                    "created": date.today().isoformat(), "software": {"name": "dmipy-sim", "version": _version()}})
    return spec.validate()


def caterpillar_spec(path, *, scale=_UM, box=None, glia=True, field_T=3.0, rho2=None, id=None):
    """The spec of a CATERPillar substrate table (``.csv`` / ``.swc``): every axon is two ``sphere_union`` walls
    (its inner radii: intra | myelin; its outer radii: myelin | extra), the glial cells one wall around a
    fourth pool (``glia``), all referencing the table by column and cell type; the voxel is the growth
    info's, the config's or ``box=`` and its faces reflect (a CATERPillar voxel is finite, not periodic);
    the pools are the catalogued white matter (glia: the intra values). An unmyelinated sphere has
    ``outer == inner``, so its sheath wall coincides with its axolemma (a zero-thickness myelin there).
    """
    from ..io.caterpillar import read_caterpillar
    from ..substrate.biophysical_constants import canonical_white_matter
    t = read_caterpillar(path, scale=scale, cell_types=(("axon", "glial_cell") if glia else ("axon",)))
    ct = t["cell_type"]
    ax, gl = ct == "axon", ct == "glial_cell"
    if not ax.any():
        raise SpecError(f"{path}: no axon rows")
    rho = float(rho2 if rho2 is not None else canonical_white_matter(field_T=field_T)["rho2"])
    pools = wm_pools(field_T)
    sha = _sha(path)

    def surf(column, cell_type):
        return Surface("sphere_union", file=str(path), format="caterpillar", scale=float(scale), sha256=sha,
                       column=column, cell_type=cell_type)
    walls = [Wall("axolemma", surf("inner_radius", "axon"), 1, 2, Directional(), Sided(rho, 0.0)),
             Wall("sheath", surf("outer_radius", "axon"), 2, 0, Directional(), Sided(0.0, rho))]
    transformations = [f"voxel: {t['box_source']}; faces reflect (a CATERPillar voxel is finite and not periodic)",
                       "inside inner radii = intra (1), inner..outer = myelin (2), outside = extra (0)",
                       f"{int((t['r_out'][ax] <= t['r_in'][ax] + 1e-15).sum())} unmyelinated axon sphere(s): sheath coincides with axolemma",
                       "blood vessels dropped (no flow model)", "nominal pool values from the catalogued white matter"]
    smallest = float(np.minimum(t["r_in"][ax], t["r_out"][ax]).min())
    if gl.any():
        pi = pools[1]
        pools.append(Pool(3, "glia", pi.D, water_fraction=1.0, T2=pi.T2, T1=pi.T1))
        walls.append(Wall("glia", surf("outer_radius", "glial_cell"), 3, 0, Directional(), Sided(rho, rho)))
        transformations.append("glial cells: a fourth pool 'glia' (3) inside their spheres, with the intra pool's D / T2 / T1")
        smallest = min(smallest, float(t["r_out"][gl].min()))
    lo, hi = (np.asarray(box[0], float), np.asarray(box[1], float)) if box is not None else (t["box_min"], t["box_max"])
    spec = SubstrateSpec(
        id or f"caterpillar/{os.path.splitext(os.path.basename(path))[0]}",
        Domain(lo.tolist(), hi.tolist(), ["reflect"] * 3), pools, walls,
        Seeding([p.id for p in pools], "uniform_by_volume", "water_fraction"),
        Validity(smallest, ["gradient", "relaxation", "surface", "field"]), nominal_field_T=float(field_T),
        description=f"CATERPillar voxel: {len(np.unique(t['cell_id'][ax]))} axons as sphere chains"
                    + (f", {len(np.unique(t['cell_id'][gl]))} glial cells" if gl.any() else ""),
        realisation={"n_axons": int(len(np.unique(t["cell_id"][ax]))), "n_glia": int(len(np.unique(t["cell_id"][gl]))),
                     "n_spheres": int(len(ct)), "reported": {k: v for k, v in t["params"].items() if "icvf" in k.lower()}},
        provenance={"source": "CATERPillar", "scale": float(scale), "files": [{"path": str(path), "sha256": sha}],
                    "transformations": transformations,
                    "created": date.today().isoformat(), "software": {"name": "dmipy-sim", "version": _version()}})
    return spec.validate()


def strands_spec(path, *, scale=_UM, g_ratio=None, boundary="reflect", field_T=3.0, rho2=None, id=None,
                 radius_tol=1e-3, source="EPFL strand list"):
    """The spec of an EPFL strand list (CACTUS ``.init`` / ``optimized_final.txt``): every strand a sphere-swept
    polyline with its one radius, as per-instance arrays of one wall (or two with ``g_ratio``: axolemma at
    ``g_ratio`` x the radius inside a sheath); the voxel ``[-side/2, side/2]^3`` with ``boundary`` faces
    (``reflect`` or ``open``; strands are not periodic). A strand whose radius varies along its length beyond
    ``radius_tol`` is refused: mesh it (``cactus_spec`` on the meshed run).
    """
    from ..io.strands import read_strands
    t = read_strands(path, scale=scale)
    R = []
    for k, r in enumerate(t["radii"]):
        if r.max() - r.min() > radius_tol * r.mean():
            raise SpecError(f"strand {k} of {path}: radius varies along its length ({r.min():.3g}..{r.max():.3g} m); a "
                            f"swept_polyline has one radius -- mesh the run (cactus_spec) or raise radius_tol")
        R.append(float(r.mean()))
    half = t["side"] / 2
    return _strands_spec(t["centerlines"], np.asarray(R), [-half] * 3, [half] * 3, boundary=boundary, g_ratio=g_ratio,
                         field_T=field_T, rho2=rho2, id=id or f"strands/{os.path.splitext(os.path.basename(path))[0]}",
                         source=source, files=[path], scale=float(scale),
                         transformations=[f"voxel [-side/2, side/2]^3 from the file header, faces {boundary}"],
                         cell_side=float(t["side"]))


def disco_spec(tracks, diameters, *, coordinate_unit_m=25e-6, diameter_unit_m=1e-3, side_m=1e-3, g_ratio=0.7,
               field_T=3.0, rho2=None, id=None):
    """The spec of the DiSCo phantom (Rafael-Patino, Girard et al., Data in Brief 38 (2021) 107429,
    doi:10.1016/j.dib.2021.107429; dataset doi:10.17632/fgf86jdfg6): its strands from the released MRtrix track
    file, in units of the ground-truth voxel (``coordinate_unit_m``, 25 um for the 40^3 grid over 1 mm^3), and
    ``DiSCo_Strands_Diameters.txt``, the INNER (axonal) diameters in millimetres (``diameter_unit_m``); the
    sheath's outer radius is the inner one over ``g_ratio`` (the phantom's uniform 0.7). The domain is
    ``[0, side_m]^3`` with reflecting faces. Every conversion is recorded in the provenance.
    """
    from ..io.strands import read_tck, read_diameters
    cls_ = read_tck(tracks, coordinate_unit_m=coordinate_unit_m)
    d_in = read_diameters(diameters, diameter_unit_m=diameter_unit_m)
    if len(cls_) != len(d_in):
        raise SpecError(f"{len(cls_)} tracks in {tracks} but {len(d_in)} diameters in {diameters}")
    keep = [k for k, c in enumerate(cls_) if len(c) >= 2]
    R_outer = np.asarray([d_in[k] / 2.0 / g_ratio for k in keep])
    transformations = [f"track coordinates x {coordinate_unit_m} m (the ground-truth voxel)",
                       f"inner diameters x {diameter_unit_m} m, halved; the sheath's outer radius is the inner over g = {g_ratio}",
                       f"domain [0, {side_m}]^3, faces reflect"]
    if len(keep) != len(cls_):
        transformations.append(f"dropped {len(cls_) - len(keep)} track(s) with fewer than two points")
    return _strands_spec([cls_[k] for k in keep], R_outer, [0.0] * 3, [float(side_m)] * 3, boundary="reflect", g_ratio=g_ratio,
                         field_T=field_T, rho2=rho2, id=id or "disco/rafael-patino-2021", source="DiSCo (Rafael-Patino et al. 2021)",
                         files=[tracks, diameters], scale=float(coordinate_unit_m), transformations=transformations,
                         cell_side=float(side_m))


def _strands_spec(centerlines, R, lo, hi, *, boundary, g_ratio, field_T, rho2, id, source, files, scale, transformations,
                  cell_side):
    """The strand spec proper: sphere-swept polylines (metres) with one OUTER radius each, in a box."""
    from ..substrate.biophysical_constants import canonical_white_matter
    if boundary not in ("reflect", "open"):
        raise SpecError("boundary must be 'reflect' or 'open'; a strand list is not periodic")
    R = np.asarray(R, float)
    rho = float(rho2 if rho2 is not None else canonical_white_matter(field_T=field_T)["rho2"])
    cls_ = [np.asarray(c, float).tolist() for c in centerlines]
    pools = wm_pools(field_T)
    transformations = list(transformations) + ["nominal pool values from the catalogued white matter"]
    if g_ratio is None:
        pools = pools[:2]
        walls = [Wall("cylinders", Surface("swept_polyline", instances={"centerlines": cls_, "radii": R.tolist()}), 1, 0,
                      Directional(), Sided(rho, rho))]
        transformations.append("inside a strand = intra (1), outside all = extra (0); no myelin")
        smallest = float(R.min())
    else:
        walls = [Wall("axolemma", Surface("swept_polyline", instances={"centerlines": cls_, "radii": (g_ratio * R).tolist()}),
                      1, 2, Directional(), Sided(rho, 0.0)),
                 Wall("sheath", Surface("swept_polyline", instances={"centerlines": cls_, "radii": R.tolist()}), 2, 0,
                      Directional(), Sided(0.0, rho))]
        transformations.append(f"the outer (sheath) radius listed; axolemma at g-ratio {g_ratio}")
        smallest = float(g_ratio * R.min())
    spec = SubstrateSpec(
        id, Domain(list(lo), list(hi), [boundary] * 3), pools, walls,
        Seeding([p.id for p in pools], "uniform_by_volume", "water_fraction"),
        # the curved tubes record their wall contact (surface); a sheath is a field source (the per-segment closed
        # form along each path, or the raster within walk_spec's field_budget)
        Validity(smallest, ["gradient", "relaxation", "surface"] + (["field"] if g_ratio is not None else [])),
        nominal_field_T=float(field_T),
        description=f"{source}: {len(R)} strands as sphere-swept polylines" + ("" if g_ratio is None else " with a sheath"),
        realisation={"n_objects": int(len(R)), "cell_side": cell_side, "radius_min": float(R.min()),
                     "radius_max": float(R.max())},
        provenance={"source": source, "scale": scale, "files": [{"path": str(f), "sha256": _sha(f)} for f in files],
                    "transformations": transformations,
                    "created": date.today().isoformat(), "software": {"name": "dmipy-sim", "version": _version()}})
    return spec.validate()


def _version():
    try:
        from importlib.metadata import version
        return version("dmipy-sim")
    except Exception:
        return "dev"
