"""Spec producers for the mesh datasets: they emit a :class:`SubstrateSpec` and construct nothing.

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


def wm_pools(field_T=3.0, *, myelin_chi_iso=-1.06e-6, myelin_chi_aniso=0.0, D_myelin=0.0):
    """The three white-matter pools with the catalogued nominal values; myelin is the field source."""
    from ..substrate.biophysical_constants import canonical_white_matter, get_default_value
    p = canonical_white_matter(field_T=field_T)
    wf_m = float(get_default_value("myelin_water_proton_density"))
    return [Pool(0, "extra", float(p["D_extra"]), water_fraction=1.0, T2=float(p["T2_extra"]), T1=float(p.get("T1_extra", 1.0))),
            Pool(1, "intra", float(p["D_intra"]), water_fraction=1.0, T2=float(p["T2_intra"]), T1=float(p.get("T1_intra", 1.2))),
            Pool(2, "myelin", float(D_myelin), water_fraction=wf_m, T2=float(p["T2_myelin"]), T1=float(p.get("T1_myelin", 0.44)),
                 susceptibility=Susceptibility(float(myelin_chi_iso), float(myelin_chi_aniso), "radial"))]


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
        frame=Frame([0.0, 0.0, 1.0]),
        description=f"CACTUS bundle: {len(pairs)} strands, each an inner (axon) and outer (myelin) surface, in a periodic cell",
        realisation={"n_objects": len(pairs), "cell_side": L},
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
    pools = wm_pools(field_T)
    pools[0] = Pool(0, "extra", pools[0].D, water_fraction=0.0, T2=pools[0].T2, T1=pools[0].T1)     # free water, not substrate
    walls = _walls({0: (inner_ply, outer_ply)}, scale, rho, rho)
    spec = SubstrateSpec(
        id or f"winther/{os.path.splitext(os.path.basename(inner_ply))[0]}",
        Domain(lo, hi, ["open", "open", "open"]), pools, walls, Seeding([1, 2], "uniform_by_volume", "thin"),
        Validity(smallest, ["gradient", "relaxation", "surface", "field"], mesh_edge_feature_ratio=edge_med / smallest),
        description="one Winther axon: inner and outer surface, surroundings free water",
        provenance={"source": "Winther", "scale": scale, "files": [{"path": p, "sha256": _sha(p)} for p in (inner_ply, outer_ply)],
                    "transformations": [f"box = outer surface padded by {pad} m", "extra pool declared free water (water_fraction 0, not seeded)",
                                        "nominal pool values from the catalogued white matter"],
                    "created": date.today().isoformat(), "software": {"name": "dmipy-sim", "version": _version()}})
    return spec.validate()


def _version():
    try:
        from importlib.metadata import version
        return version("dmipy-sim")
    except Exception:
        return "dev"
