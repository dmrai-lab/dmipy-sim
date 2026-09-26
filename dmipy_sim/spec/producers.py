"""Spec producers for the datasets: they emit a :class:`SubstrateSpec` and construct nothing. Mesh runs (CACTUS,
Winther), sphere-grown cells (CATERPillar) and strand lists (CACTUS / DiSCo).

Every decision the loaders used to take in code is a field here, with its reason recorded in
``provenance.transformations``: which surface is which pool, the box and what its faces do, the
measured g-ratio, dropped open surfaces, the nominal pool values (from the catalogued white matter,
``substrate.biophysical_constants.canonical_white_matter``), and the unit scale.
"""
import dataclasses
import glob
import hashlib
import os
import warnings
import re
from datetime import date

import numpy as np

from .substrate import (SubstrateSpec, Domain, Frame, Pool, Susceptibility, Surface, Wall, Directional, Sided,
                        Seeding, Validity, SpecError)

_UM = 1e-6


def _sha(path):
    from ..fill.hub import sha256_of
    return sha256_of(path)


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


def shell_thickness(inner, outer, *, k=8):
    """The thickness (metres) of the shell between two closed surfaces ``(V, F)``: the median over the inner
    surface's vertices of the distance to the outer surface, every vertex against the ``k`` outer faces nearest by
    centroid (a k-d tree) with the point-triangle distance exact on those. The median, not the minimum: a meshed
    surface is bumpy at its own resolution (the CACTUS erode meshes: a sheath of 0.11 um has spots of 0.003 um
    where the inner surface grazes the outer), so the minimum is a property of the mesh and the median is the
    sheath's; the thinnest SHELL of a substrate is the smallest of its shells' thicknesses. An outer face lying in
    one of the outer surface's bounding planes is a cap of a tube cut there (a domain face): the inner tube is cut
    in the same plane, so its rim sits on that face at no distance, and such faces are not the shell. Computed in
    units of the outer mesh's median edge, since trimesh's triangle geometry carries absolute tolerances."""
    from scipy.spatial import cKDTree
    from trimesh.triangles import closest_point
    Vi = np.asarray(inner[0], float)
    Vo, Fo = np.asarray(outer[0], float), np.asarray(outer[1], np.int64)
    lo, hi = Vo.min(0), Vo.max(0)
    tol = 1e-4 * float((hi - lo).max())
    on_plane = np.zeros(len(Fo), bool)
    for a in range(3):
        for plane in (lo[a], hi[a]):
            on_plane |= (np.abs(Vo[Fo][:, :, a] - plane) < tol).all(axis=1)
    Fo = Fo[~on_plane]
    if len(Fo) == 0:
        raise SpecError("the outer surface has no face off its bounding planes; it is not a closed shell")
    unit = float(np.median(np.linalg.norm(Vo[Fo[:, 0]] - Vo[Fo[:, 1]], axis=1)))
    tri = Vo[Fo] / unit
    P = Vi / unit
    kk = min(int(k), len(tri))
    _, idx = cKDTree(tri.mean(1)).query(P, k=kk, workers=-1)
    idx = np.asarray(idx).reshape(len(P), kk)
    q = np.repeat(P, kk, axis=0)
    c = closest_point(tri[idx.ravel()], q)
    return float(np.median(np.linalg.norm(c - q, axis=1).reshape(len(P), kk).min(1))) * unit


def _surface_stats(paths, scale, notes=None):
    """Per-file (V, F) in metres, the smallest edge-based feature, the median edge and the files that are OPEN: a
    surface with a boundary edge does not enclose a volume. A surface pinched along an edge shared by three or
    more faces still does; such edges, and the degenerate faces the loader dropped, are counted into ``notes``
    (a list the spec's transformations take) rather than refused (dmrai-lab/dmipy-sim#452)."""
    from ..geometry.mesh import load_ply, surface_topology
    feats, edges, open_files, meshes = [], [], [], []
    for p in paths:
        repairs = {}
        V, F = load_ply(p, scale=scale, repairs=repairs)
        meshes.append((V, F))
        e = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
        edges.append(np.median(e))
        ext = V.max(0) - V.min(0)
        feats.append(0.5 * float(np.sort(ext)[0]))                # half the thinnest extent: the radius of a tube
        topo = surface_topology(V, F)
        n_drop, n_merge = repairs["degenerate_faces_dropped"], repairs["duplicate_vertices_merged"]
        if topo["boundary_edges"] > 0:
            open_files.append(p)
        elif notes is not None and (topo["nonmanifold_edges"] or n_drop or n_merge):
            notes.append(f"{os.path.basename(p)}: closed; {topo['nonmanifold_edges']} edge(s) on three or more faces kept"
                         + (f"; the loader dropped {n_drop} degenerate face(s) and merged {n_merge} duplicate vertex(es)"
                            if (n_drop or n_merge) else ""))
    return meshes, float(min(feats)), float(np.median(edges)), open_files


def _walls(pairs, scale, rho_inner, rho_outer):
    walls = []
    for k, (inner, outer) in pairs.items():
        walls.append(Wall(f"fibre-{k}/inner", Surface("mesh", file=inner, format=inner.rsplit(".", 1)[-1].lower(), scale=scale, sha256=_sha(inner)),
                          1, 2, Directional(), Sided(rho_inner, 0.0)))
        walls.append(Wall(f"fibre-{k}/outer", Surface("mesh", file=outer, format=outer.rsplit(".", 1)[-1].lower(), scale=scale, sha256=_sha(outer)),
                          2, 0, Directional(), Sided(0.0, rho_outer)))
    return walls


def strand_frame(centerlines, *, cluster_deg=30.0):
    """The substrate frame of a strand substrate from the strands themselves (RPK.md 4.2, dmipy-sim#194): every
    strand's axis is its end-to-end chord, the chords are clustered into bundles (greedy, within ``cluster_deg``
    of a bundle's running mean, sign-free), and the frame is :func:`~dmipy_sim.replay.bank.frame_from_bundles`
    anchored to the largest bundle. Returns ``(F, bundles)`` with ``bundles`` a list of ``{"axis", "n_strands"}``
    in stored coordinates, largest first -- never a PCA over positions, which for a crossing is the bisector."""
    from ..replay.bank import frame_from_bundles
    ch = []
    for c in centerlines:
        c = np.asarray(c, float)
        if c.shape[0] < 2:
            continue
        v = c[-1] - c[0]
        n = np.linalg.norm(v)
        if n > 0:
            ch.append(v / n)
    if not ch:
        raise SpecError("no strand has two control points: no axis to declare")
    ch = np.asarray(ch)
    cos_tol = np.cos(np.radians(cluster_deg))
    means, members = [], []
    for v in ch:
        for k, mu in enumerate(means):
            if abs(float(v @ mu)) >= cos_tol:
                v = v if float(v @ mu) >= 0 else -v
                members[k].append(v)
                mu = np.mean(members[k], axis=0); means[k] = mu / np.linalg.norm(mu)
                break
        else:
            means.append(v.copy()); members.append([v])
    order = np.argsort([-len(mm) for mm in members], kind="stable")
    axes = np.asarray([means[k] for k in order]); counts = [len(members[k]) for k in order]
    F = frame_from_bundles(axes, primary=0, weights=counts)
    bundles = [{"axis": axes[k].tolist(), "n_strands": int(counts[k])} for k in range(len(axes))]
    return F, bundles


def chain_frame(centers, r_in, cell_id):
    """The substrate frame of a sphere-chain substrate from the chains themselves (RPK.md 4.2, dmipy-sim#233): the
    principal axis of the chain segments' direction dyadic, each segment weighted by the intra-axonal volume it
    carries (the frustum of its two inner radii), which is what the walkers sample. A cell's rows are its chain in
    file order, as CATERPillar writes them. Returns ``(F, axis)`` with ``F`` :func:`~dmipy_sim.replay.bank.frame_from_axis`
    of the axis, signed towards ``+z``. On ``myelin15`` the tapering large axons put this axis 5.7 deg from ``z``,
    where the walk's principal displacement axis lands too; a bare ``z`` is refused at build."""
    from ..replay.bank import frame_from_axis
    centers = np.asarray(centers, float); r_in = np.asarray(r_in, float); cell_id = np.asarray(cell_id)
    Q = np.zeros((3, 3))
    for cid in np.unique(cell_id):
        sel = np.flatnonzero(cell_id == cid)
        if len(sel) < 2:
            continue
        seg = np.diff(centers[sel], axis=0)
        L = np.linalg.norm(seg, axis=1)
        ok = L > 0
        if not ok.any():
            continue
        d = seg[ok] / L[ok, None]
        a, b = r_in[sel][:-1][ok], r_in[sel][1:][ok]
        vol = np.pi * L[ok] * (a * a + a * b + b * b) / 3.0
        Q += (d.T * vol) @ d
    if not Q.any():
        raise SpecError("no sphere chain has two distinct centres: no axis to declare")
    w, v = np.linalg.eigh(Q)
    axis = v[:, -1]
    axis = axis * (np.sign(axis[2]) or 1.0)
    return frame_from_axis(axis), axis


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
    hdr = os.path.join(run_dir, "optimized_final.txt")
    if side_um is None:
        if not os.path.isfile(hdr):
            raise SpecError("no optimized_final.txt header and no side_um: the periodic cell is unknown")
        with open(hdr) as fh:
            side_um = float(fh.readline().strip())
        transformations.append("periodic cell side read from optimized_final.txt")
    frame, bundles = Frame([0.0, 0.0, 1.0]), None
    if os.path.isfile(hdr):                                         # the strands themselves declare the frame
        from ..io.strands import read_strands
        F, bundles = strand_frame(read_strands(hdr, scale=scale)["centerlines"])
        frame = Frame(F[:, 2].tolist(), F[:, 1].tolist())
        transformations.append(f"substrate frame from the {sum(b['n_strands'] for b in bundles)} strand chords of "
                               f"optimized_final.txt in {len(bundles)} bundle(s), z the largest bundle's mean axis")
    else:
        transformations.append("no strand list: the frame is undeclared (z), which a walk along another axis refuses at build")
    L = float(side_um) * scale
    files = [p for pr in pairs.values() for p in pr]
    meshes, smallest, edge_med, open_files = _surface_stats(files, scale, transformations)
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
    thinnest = min(shell_thickness(by_path[v[0]], by_path[v[1]]) for v in pairs.values())   # over the strands kept
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
        Validity(smallest, ["gradient", "relaxation", "surface", "field"], mesh_edge_feature_ratio=edge_med / smallest,
                 thinnest_shell=thinnest),
        frame=frame, nominal_field_T=float(field_T),
        description=f"CACTUS bundle: {len(pairs)} strands, each an inner (axon) and outer (myelin) surface, in a periodic cell",
        realisation={"n_objects": len(pairs), "cell_side": L, **({} if bundles is None else {"bundles": bundles}),
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
    notes = []
    meshes, smallest, edge_med, open_files = _surface_stats([inner_ply, outer_ply], scale, notes)
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
        Validity(smallest, ["gradient", "relaxation", "surface", "field"], mesh_edge_feature_ratio=edge_med / smallest,
                 thinnest_shell=shell_thickness(meshes[0], meshes[1])),
        nominal_field_T=float(field_T),
        description="one Winther axon: inner and outer surface, surroundings free water",
        realisation={"g_ratio": _g_ratio(meshes[0], meshes[1])},
        provenance={"source": "Winther", "scale": scale, "files": [{"path": p, "sha256": _sha(p)} for p in (inner_ply, outer_ply)],
                    "transformations": [f"box = outer surface padded by {pad} m", "extra pool declared free water (water_fraction 0, not seeded)",
                                        "nominal pool values from the catalogued white matter",
                                        "myelin chi_iso = +1.06e-6, isotropic: the convention the Winther meshes were published with"] + notes,
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
    sheath = (t["r_out"][ax] - t["r_in"][ax])[t["r_out"][ax] > t["r_in"][ax] + 1e-15]     # the myelinated spheres' sheaths
    thinnest = float(sheath.min()) if sheath.size else None
    F, axis = chain_frame(t["centers"][ax], t["r_in"][ax], t["cell_id"][ax])      # the axons declare the frame
    transformations.append(f"substrate frame from the axon sphere chains: the principal axis of their intra-volume-weighted "
                           f"direction dyadic, {np.degrees(np.arccos(min(1.0, abs(float(axis[2]))))):.1f} deg from stored z")
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
        Seeding([p.id for p in pools if p.water_fraction > 0], "uniform_by_volume", "water_fraction"),
        Validity(smallest, ["gradient", "relaxation", "surface", "field"], thinnest_shell=thinnest),
        frame=Frame(F[:, 2].tolist(), F[:, 1].tolist()), nominal_field_T=float(field_T),
        description=f"CATERPillar voxel: {len(np.unique(t['cell_id'][ax]))} axons as sphere chains"
                    + (f", {len(np.unique(t['cell_id'][gl]))} glial cells" if gl.any() else ""),
        realisation={"n_axons": int(len(np.unique(t["cell_id"][ax]))), "n_glia": int(len(np.unique(t["cell_id"][gl]))),
                     "n_spheres": int(len(ct)), "reported": {k: v for k, v in t["params"].items() if "icvf" in k.lower()}},
        provenance={"source": "CATERPillar", "scale": float(scale), "files": [{"path": str(path), "sha256": sha}],
                    "transformations": transformations,
                    "created": date.today().isoformat(), "software": {"name": "dmipy-sim", "version": _version()}})
    return spec.validate()


def label_volume_spec(path, *, pools=None, voxel_size=None, origin=None, crop=None, periodic=False,
                      D=None, rho=0.0, T2=None, T2_pools=None, nominal_field_T=None, walk=None,
                      frame=None, format=None, cite_image_as=None, id=None, description=None, source=None):
    """The spec of a **segmented image**: one pool per label, one wall per pair of pools that share a
    voxel face, the image cited as a file.

    ``pools`` maps each label value to a pool name, in pool-id order, and defaults to the micro-CT
    convention ``{0: "free", 1: "grain"}`` (void 0, solid 1). ``walk`` names the pool the walkers
    occupy (the first by default), the only pool that holds water: a solid is a pool with
    ``water_fraction`` 0. ``rho`` is the transverse surface relaxivity (m/s) of every wall, per side;
    ``D`` and ``T2`` are the walking pool's bulk values, ``T2_pools`` a T2 per pool name where they
    differ. ``crop`` is the ``(i0, j0, k0, i1, j1, k1)`` sub-volume that IS the substrate; ``periodic``
    says which axes repeat (a rock crop does not: its faces reflect). ``voxel_size`` (metres) supplies
    a spacing the container does not carry, and contradicts a header that does. ``frame`` is the
    substrate's own structural axis in the grid's index frame, for an image that HAS one (a segmented
    nerve, a vessel along an axis): the default is the grid's ``+z``, which is what an isotropic
    segmentation has, and a pack of an anisotropic walk is refused against a frame that does not
    describe it (RPK.md 4.2).

    The image is read only to measure what the spec must declare -- its labels, its extent and the
    pools that actually touch -- and nothing is constructed: ``spec.geometry_from_spec`` reads it back
    into a :class:`~dmipy_sim.geometry.label_volume.LabelVolume`. ``cite_image_as`` is the path a
    dataset distributes the image at, when that is not where it is being read from.
    """
    from ..io.label_volume import read_label_volume, crop_labels, format_of
    fmt = str(format).lower() if format is not None else format_of(path)
    vol = read_label_volume(path, format=fmt, voxel_size=voxel_size)
    if crop is not None:
        vol = crop_labels(vol, crop)
    lab, vox = vol.labels, vol.voxel_size
    org = np.asarray(origin, float) if origin is not None else vol.origin
    per = np.broadcast_to(np.asarray(periodic, bool).ravel(), (3,))

    pool_map = dict(pools if pools is not None else {0: "free", 1: "grain"})
    names = list(pool_map.values())
    present = sorted(int(v) for v in np.unique(lab))
    unknown = [v for v in present if v not in {int(k) for k in pool_map}]
    if unknown:
        raise SpecError(f"{path}: the image holds labels {unknown} that `pools` does not name (pools = {pool_map})")
    walking = str(walk) if walk is not None else names[0]
    if walking not in names:
        raise SpecError(f"{path}: walk={walking!r} is not one of the pools {names}")
    wid = names.index(walking)
    if names[0] not in ("extra", "free"):
        raise SpecError(f"{path}: pool 0 is the free pool and is named 'extra' or 'free', got {names[0]!r}")

    t2 = dict(T2_pools or {})
    if T2 is not None:
        t2[walking] = float(T2)
    spec_pools = [Pool(i, n, (float(D) if (D is not None and i == wid) else None),
                       water_fraction=(1.0 if i == wid else 0.0), T2=t2.get(n))
                  for i, n in enumerate(names)]

    ids = np.full(256, -1, np.int32)
    for i, v in enumerate(pool_map):
        ids[int(v)] = i
    g = np.take(ids, lab)
    pairs = set()
    for ax in range(3):
        a = np.swapaxes(g, 0, ax)
        slabs = [(a[:-1], a[1:])] + ([(a[-1:], a[:1])] if per[ax] else [])
        for left, right in slabs:
            lo, hi = np.minimum(left, right), np.maximum(left, right)
            sel = lo != hi
            if sel.any():
                pairs.update(zip(lo[sel].tolist(), hi[sel].tolist()))
    if not pairs:
        raise SpecError(f"{path}: no two pools of this image share a voxel face, so it has no wall")

    from ..io.label_volume import payload_files
    read_files = payload_files(path, format=fmt)
    cited = str(cite_image_as if cite_image_as is not None else path)
    surf_kw = dict(file=cited, format=fmt, sha256=_sha(path),
                   voxel_size=[float(x) for x in vox], origin=[float(x) for x in org],
                   labels={str(int(v)): n for v, n in pool_map.items()},
                   crop=([int(x) for x in crop] if crop is not None else None))
    rho = float(rho)
    walls = [Wall(f"{names[j]}|{names[i]}", Surface("label_volume", **surf_kw), j, i,
                  Directional(), Sided(rho, rho)) for i, j in sorted(pairs)]
    dom = Domain(org.tolist(), (org + np.asarray(lab.shape) * vox).tolist(),
                 ["periodic" if p else "reflect" for p in per])
    phi = float(np.mean(g == wid))
    area = 0.0
    for ax in range(3):
        a = np.swapaxes(g == wid, 0, ax)
        n = int(np.count_nonzero(a[:-1] != a[1:])) + (int(np.count_nonzero(a[-1] != a[0])) if per[ax] else 0)
        area += n * float(np.prod(vox) / vox[ax])
    s_over_v = area / (phi * float(np.prod(lab.shape)) * float(np.prod(vox)))
    transformations = [
        f"labels -> pools {pool_map}; the {walking!r} pool holds the water, the others water_fraction 0",
        "the wall is every voxel face between two pools (the Manhattan surface of the segmentation): "
        f"measured S/V of the {walking!r} pool {s_over_v:.6g} 1/m at porosity {phi:.6g}",
        ("the crop's own outer faces are not a wall: they are the domain's " +
         ", ".join(f"{'periodic' if p else 'reflect'} {ax}" for ax, p in zip("xyz", per))),
    ]
    if crop is not None:
        transformations.append(f"crop {list(int(x) for x in crop)} of the released image, half-open, in voxels")
    return SubstrateSpec(
        id or f"label_volume/{os.path.splitext(os.path.basename(str(path)))[0]}",
        dom, spec_pools, walls, Seeding([wid], "uniform_by_volume", "water_fraction"),
        Validity(float(np.min(vox)), ["gradient"] + (["relaxation"] if any(p.T2 for p in spec_pools) else []) + ["surface"]),
        frame=(Frame(np.asarray(frame, float).ravel().tolist()) if frame is not None else Frame()),
        nominal_field_T=(float(nominal_field_T) if nominal_field_T is not None else None),
        description=description or (f"a segmented {'x'.join(str(int(n)) for n in lab.shape)} image at "
                                   f"{float(np.min(vox)) * 1e6:.4g} um; the {walking!r} pool walks between its voxel faces"),
        realisation={"shape": [int(n) for n in lab.shape], "porosity": phi, "surface_to_volume": s_over_v,
                     "voxel_size_m": [float(x) for x in vox]},
        provenance={"source": source or "segmented image",
                    # a detached container is two files and `surface.sha256` covers only the one it
                    # cites, so every file the image was read from is recorded with its own digest
                    "files": [{"path": (cited if i == 0 else os.path.basename(f)), "sha256": _sha(f)}
                              for i, f in enumerate(read_files)],
                    "transformations": transformations,
                    "created": date.today().isoformat(), "software": {"name": "dmipy-sim", "version": _version()}}
    ).validate()


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


DISCO_D = 0.6e-9
"""m^2/s: DiSCo's unrestricted diffusion coefficient, the same in both compartments -- "The unrestricted diffusion
coefficient of the Monte Carlo particles was set to 0.6e-3 mm^2/s ... The simulation parameters for both
compartments were the same" (Rafael-Patino et al., Data in Brief 38 (2021) 107429, section 2.2)."""
DISCO_G_RATIO = 0.7
"""DiSCo's inner over outer tube diameter -- "an additional inner tubular mesh representing the inner surface of the
axon-like structure was generated following the same trajectory, but with a diameter of 0.7 times the outer
diameter" (section 2.1); the released diameters are the inner ones (section 1)."""


def disco_spec(tracks, diameters, *, coordinate_unit_m=25e-6, diameter_unit_m=1e-3, side_m=1e-3, field=True,
               field_T=3.0, rho2=None, id=None, cite_tracks_as=None):
    """The spec of the DiSCo phantom (Rafael-Patino, Girard et al., Data in Brief 38 (2021) 107429,
    doi:10.1016/j.dib.2021.107429; dataset doi:10.17632/fgf86jdfg6.3, CC BY 4.0): its strands from the released MRtrix
    track file, in units of the ground-truth voxel (``coordinate_unit_m``, 25 um for the 40^3 grid over 1 mm^3), and
    ``DiSCo_Strands_Diameters.txt``, the strands' INNER diameters in millimetres (``diameter_unit_m``). DiSCo's
    substrate is two tubes per strand: the inner one at the listed diameter and the outer one at the listed diameter
    over :data:`DISCO_G_RATIO`; its Monte Carlo labelled the particles inside the inner tube intra-axonal, those
    outside the outer tube extra-axonal and those between the two myelin, and generated the signal from the first
    two. So the spec has two walls, the axolemma at the listed radius and the sheath at the listed radius over the
    g-ratio, with the intra pool inside the first, the extra pool outside the second, and a third pool between them
    that holds no water and is not seeded (the intra-strand volume fraction map being the inner tube's volume, and
    DiSCo's signal having no myelin water). With ``field`` (the default) that sheath is a susceptibility source with
    the catalogue's myelin chi, so a walk carries the field tier (C3) for the two diffusing pools; the myelin's
    susceptibility is a property of the sheath, not of its water. DiSCo's own simulation is the replay of such a
    pack with the field off. ``field=False`` leaves the sheath inert: the two walls alone. Both pools diffuse at
    :data:`DISCO_D`, as the dataset's own walk did. A pack of this spec has no myelin water: a short-echo or
    multi-echo replay sees two pools where white matter has three.
    The domain is ``[0, side_m]^3`` with reflecting faces. The walls cite the track file (``format: tck``, its
    coordinate unit and sha256) rather than carrying 12,196 centerlines inline, at the path given or at
    ``cite_tracks_as`` (the path a dataset distributes it at; a consumer resolves it from the working directory or
    the surface cache, :func:`~dmipy_sim.spec.build.resolve_surface_file`). Every conversion is recorded in the
    provenance.
    """
    from ..io.strands import read_tck, read_diameters
    cls_ = read_tck(tracks, coordinate_unit_m=coordinate_unit_m)
    d = read_diameters(diameters, diameter_unit_m=diameter_unit_m)
    if len(cls_) != len(d):
        raise SpecError(f"{len(cls_)} tracks in {tracks} but {len(d)} diameters in {diameters}")
    keep = [k for k, c in enumerate(cls_) if len(c) >= 2]
    R_in = np.asarray([d[k] / 2.0 for k in keep])
    R_out = R_in / DISCO_G_RATIO
    transformations = [f"track coordinates x {coordinate_unit_m} m (the ground-truth voxel)",
                       f"strand diameters x {diameter_unit_m} m, halved: the inner tube's radius (the axolemma)",
                       f"the outer tube (the sheath) at the inner radius / {DISCO_G_RATIO}: DiSCo's outer mesh",
                       "intra inside the inner tube, extra outside the outer tube, no water between them (DiSCo's signal came from "
                       "the intra and extra particles only)" + (", the sheath a susceptibility source with the catalogue's myelin chi"
                                                                if field else ", the sheath inert"),
                       f"both pools at DiSCo's own diffusivity {DISCO_D} m^2/s",
                       f"domain [0, {side_m}]^3, faces reflect"]
    if len(keep) != len(cls_):
        raise SpecError(f"{tracks} holds {len(cls_) - len(keep)} track(s) with fewer than two points; a cited file lists every strand")
    cite = dict(file=str(cite_tracks_as or tracks), format="tck", scale=float(coordinate_unit_m), sha256=_sha(tracks))
    return _strands_spec([cls_[k] for k in keep], R_out, [0.0] * 3, [float(side_m)] * 3, boundary="reflect", g_ratio=DISCO_G_RATIO,
                         R_inner=R_in, D=DISCO_D, sheath_water=False, sheath_field=bool(field), centerline_file=cite,
                         field_T=field_T, rho2=rho2, id=id or "disco/rafael-patino-2021", source="DiSCo (Rafael-Patino et al. 2021)",
                         files=[tracks, diameters], scale=float(coordinate_unit_m), transformations=transformations,
                         cell_side=float(side_m))


def _strands_spec(centerlines, R, lo, hi, *, boundary, g_ratio, field_T, rho2, id, source, files, scale, transformations,
                  cell_side, R_inner=None, D=None, sheath_water=True, sheath_field=True, centerline_file=None):
    """The strand spec proper: sphere-swept polylines (metres) with one OUTER radius each, in a box. With ``g_ratio``
    a sheath: the axolemma at ``R_inner`` (given) or ``g_ratio`` x the outer radius, and the pool between the two
    walls holding the catalogue's myelin water (``sheath_water``, else no water and not seeded) and the catalogue's
    myelin susceptibility (``sheath_field``, else no field source). ``D`` sets both diffusing pools' diffusivity (a
    dataset's own value). With
    ``centerline_file`` (``file``, ``format``, ``scale``, ``sha256``) the walls cite the file instead of carrying the
    centerlines inline."""
    from ..substrate.biophysical_constants import canonical_white_matter
    if boundary not in ("reflect", "open"):
        raise SpecError("boundary must be 'reflect' or 'open'; a strand list is not periodic")
    R = np.asarray(R, float)
    rho = float(rho2 if rho2 is not None else canonical_white_matter(field_T=field_T)["rho2"])
    cls_ = [np.asarray(c, float).tolist() for c in centerlines]
    surf = ((lambda R_: Surface("swept_polyline", instances={"radii": R_.tolist()}, **centerline_file)) if centerline_file
            else (lambda R_: Surface("swept_polyline", instances={"centerlines": cls_, "radii": R_.tolist()})))
    pools = wm_pools(field_T)
    transformations = list(transformations) + ["nominal pool values from the catalogued white matter"]
    if D is not None:
        pools = [dataclasses.replace(p, D=float(D)) if p.id in (0, 1) else p for p in pools]
    if g_ratio is None:
        pools = pools[:2]
        walls = [Wall("cylinders", surf(R), 1, 0, Directional(), Sided(rho, rho))]
        transformations.append("inside a strand = intra (1), outside all = extra (0); no myelin")
        smallest = float(R.min()); thinnest = None
    else:
        R_in = np.asarray(R_inner, float) if R_inner is not None else g_ratio * R
        if R_in.shape != R.shape or (R_in >= R).any():
            raise SpecError("every inner radius must be smaller than its outer radius")
        walls = [Wall("axolemma", surf(R_in), 1, 2, Directional(), Sided(rho, 0.0)),
                 Wall("sheath", surf(R), 2, 0, Directional(), Sided(0.0, rho))]
        if R_inner is None:
            transformations.append(f"the outer (sheath) radius listed; axolemma at g-ratio {g_ratio}")
        if not sheath_water or not sheath_field:
            pools = pools[:2] + [dataclasses.replace(pools[2], water_fraction=(pools[2].water_fraction if sheath_water else 0.0),
                                                     susceptibility=(pools[2].susceptibility if sheath_field else None))]
        smallest = float(R_in.min()); thinnest = float((R - R_in).min())
    F, bundles = strand_frame([np.asarray(c, float) for c in centerlines])
    transformations.append(f"substrate frame from the strand chords in {len(bundles)} bundle(s), z the largest bundle's mean axis")
    spec = SubstrateSpec(
        id, Domain(list(lo), list(hi), [boundary] * 3), pools, walls,
        Seeding([p.id for p in pools if p.water_fraction > 0], "uniform_by_volume", "water_fraction"),
        # the curved tubes record their wall contact (surface); a sheath is a field source (the per-segment closed
        # form along each path, or the raster within walk_spec's field_budget)
        Validity(smallest, ["gradient", "relaxation", "surface"] + (["field"] if any(p.susceptibility is not None for p in pools) else []),
                 thinnest_shell=thinnest),
        frame=Frame(F[:, 2].tolist(), F[:, 1].tolist()), nominal_field_T=float(field_T),
        description=f"{source}: {len(R)} strands as sphere-swept polylines"
                    + ("" if g_ratio is None else (" with a sheath" if sheath_water else
                                                   (" with a dry sheath as a field source" if sheath_field else " with a dry sheath"))),
        realisation={"n_objects": int(len(R)), "cell_side": cell_side, "radius_min": float(R.min()),
                     "radius_max": float(R.max()), "bundles": bundles},
        provenance={"source": source, "scale": scale, "files": [{"path": str(f), "sha256": _sha(f)} for f in files],
                    "transformations": transformations,
                    "created": date.today().isoformat(), "software": {"name": "dmipy-sim", "version": _version()}})
    return spec.validate()


def _version():
    from ..run import package_version
    return package_version()
