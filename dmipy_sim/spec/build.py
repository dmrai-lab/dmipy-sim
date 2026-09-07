"""Between a spec and a geometry: ``spec_of(geometry)`` writes the situation a geometry puts a walker
in; ``geometry_from_spec(spec)`` builds the geometry a spec describes. Analytic families here; meshes
follow (#130 PR 3).

Conventions written out here, once: an isolated object sits in an ``open`` box of ``margin`` radii
around it; pool 0 is present even when it holds no water (an isolated lumen), so ids are dense;
a packed cell is ``periodic`` in the plane and ``open`` along the axis, its objects one wall entry
with per-instance arrays; the walk's diffusivity is the driver's (``Pool.D`` null) unless the geometry
carries per-pool diffusivities.
"""
import numpy as np

from ..geometry.packing import periodic_min_gap
from .substrate import (SubstrateSpec, Domain, Frame, Pool, Susceptibility, Surface, Wall, Directional, Sided,
                        Seeding, Validity, SpecError)

MARGIN = 4.0          # open-domain box half-width, in units of the largest radius, beyond the object


def _f(x):
    return None if x is None else float(x)


def _rho(g):
    return _f(getattr(g, "surface_relaxivity_t2", None)) or 0.0


def _kappa(g):
    return _f(getattr(g, "permeability", None)) or 0.0


def _box(half, axis_open=None):
    lo, hi = [-half] * 3, [half] * 3
    return lo, hi


def _tiers(walls, pools, kappa=0.0):
    t = ["gradient"]
    if len([p for p in pools if p.water_fraction > 0]) > 1 or any(p.T2 for p in pools):
        t.append("relaxation")
    if walls:
        t.append("surface")
    if any(p.susceptibility for p in pools):
        t.append("field")
    if kappa > 0 or any(w.permeability.in_to_out > 0 or w.permeability.out_to_in > 0 for w in walls):
        t.append("exchange")
    return t


def _pools_from_compartments(g, names, D_by_name, wf_by_name=None):
    """Pools 0..n for the named compartments, reading T2/T1/water fraction from ``g.compartments``."""
    comps = getattr(g, "compartments", None)
    out = []
    for i, n in enumerate(names):
        c = comps[n] if (comps is not None and n in comps) else None
        wf = (wf_by_name or {}).get(n)
        if wf is None:
            wf = c.water_fraction if (c is not None and c.water_fraction is not None) else 1.0
        out.append(Pool(i, n, D_by_name.get(n), water_fraction=wf,
                        T2=(c.T2 if c is not None else None), T1=(c.T1 if c is not None else None),
                        susceptibility=(Susceptibility(None, None, "radial") if n == "myelin" else None)))
    return out


def spec_of(geometry, *, id=None, provenance=None, surface_dir=None):
    """The :class:`SubstrateSpec` of a geometry. A mesh built in memory needs ``surface_dir`` to write its
    surface file into (a spec references surfaces as files); one loaded with ``Mesh.from_ply`` references
    the file it came from."""
    from ..geometry import (FreeDiffusion, Box1D, Sphere, Cylinder, Ellipsoid, PackedCylinders, PackedSpheres,
                            MyelinatedCylinder, PackedMyelinatedCylinders, CurvedCylinder, CurvedMyelinatedCylinder,
                            PackedCurvedCylinders, SphereUnion)
    from ..geometry.analytic import PermeableSlab1D, PermeableShell
    g = geometry
    name = type(g).__name__
    sid = id or f"analytic/{name.lower()}"
    prov = provenance or {"source": "analytic", "constructor": name}
    if isinstance(g, SphereUnion):
        return _spec_of_sphere_union(g, sid, prov)
    extra0 = Pool(0, "extra", None, water_fraction=0.0)
    if isinstance(g, FreeDiffusion):
        half = 1e-4
        return SubstrateSpec(sid, Domain(*_box(half), ["open"] * 3), [Pool(0, "extra", None, water_fraction=1.0)], [],
                             Seeding([0]), Validity(half, ["gradient"]), description="unbounded free diffusion",
                             provenance=prov)
    if isinstance(g, PermeableSlab1D):
        L = g.length
        pools = [Pool(0, "extra", None, water_fraction=1.0), Pool(1, "intra", None, water_fraction=1.0)]
        wall = Wall("membrane", Surface("plane", point=[L / 2, 0.0, 0.0], normal=[1.0, 0.0, 0.0]), 1, 0,
                    Directional(_kappa(g), _kappa(g)), Sided(_rho(g), _rho(g)))
        dom = Domain([0.0, -MARGIN * L, -MARGIN * L], [L, MARGIN * L, MARGIN * L], ["reflect", "open", "open"], _rho(g))
        return SubstrateSpec(sid, dom, pools, [wall], Seeding([1]), Validity(L / 2, _tiers([wall], pools)),
                             description="closed 1-D two-compartment slab; A = x < L/2 is pool 1", provenance=prov)
    if isinstance(g, Box1D):
        L = g.length
        pools = [extra0, Pool(1, "intra", None, water_fraction=1.0)]
        dom = Domain([0.0, -MARGIN * L, -MARGIN * L], [L, MARGIN * L, MARGIN * L], ["reflect", "open", "open"], _rho(g))
        return SubstrateSpec(sid, dom, pools, [], Seeding([1]),
                             Validity(L, ["gradient"] + (["surface"] if _rho(g) > 0 else [])),
                             description="reflecting slab 0 <= x <= L, y and z free", provenance=prov)
    if isinstance(g, PermeableShell):
        ri, ro = g.r_inner, g.r_outer
        pools = [Pool(0, "extra", None, water_fraction=1.0), Pool(1, "intra", None, water_fraction=1.0)]
        kind = "sphere" if g.kind == "sphere" else "cylinder"
        ax = {} if kind == "sphere" else {"axis": np.asarray(g._o, float).tolist()}
        inner = Wall("membrane", Surface(kind, center=[0.0] * 3, radius=ri, **ax), 1, 0,
                     Directional(_kappa(g), _kappa(g)), Sided(_rho(g), _rho(g)))
        outer = Wall("outer", Surface(kind, center=[0.0] * 3, radius=ro, **ax), 0, None, Directional(),
                     Sided(_rho(g), 0.0))
        dom = Domain(*_box(MARGIN * ro), ["open"] * 3)
        return SubstrateSpec(sid, dom, pools, [inner, outer], Seeding([1]), Validity(ri, _tiers([inner, outer], pools)),
                             description=f"closed radial two-compartment {kind} shell", provenance=prov)
    if isinstance(g, (Sphere, Cylinder, Ellipsoid)):
        kappa = _kappa(g)
        pools = [Pool(0, "extra", None, water_fraction=0.0), Pool(1, "intra", None, water_fraction=1.0)]
        if isinstance(g, Sphere):
            surf, R = Surface("sphere", center=[0.0] * 3, radius=g.radius), g.radius
        elif isinstance(g, Cylinder):
            surf, R = Surface("cylinder", center=[0.0] * 3, axis=np.asarray(g.orientation, float).tolist(), radius=g.radius), g.radius
        else:
            surf, R = Surface("ellipsoid", center=[0.0] * 3, semiaxes=np.asarray(g.semiaxes, float).tolist()), float(np.min(g.semiaxes))
        wall = Wall("membrane", surf, 1, (0 if kappa > 0 else None), Directional(kappa, kappa), Sided(_rho(g), _rho(g)))
        Rmax = float(np.max(g.semiaxes)) if isinstance(g, Ellipsoid) else R
        return SubstrateSpec(sid, Domain(*_box(MARGIN * Rmax), ["open"] * 3), pools, [wall], Seeding([1]),
                             Validity(R, _tiers([wall], pools)),
                             description=f"isolated {type(g).__name__.lower()}; the outside is void unless permeable", provenance=prov)
    if isinstance(g, (PackedCylinders, PackedSpheres)):
        kappa = _kappa(g)
        L = float(g._L_float)
        pools = [Pool(0, "extra", None, water_fraction=1.0), Pool(1, "intra", None, water_fraction=0.0)]
        centers = np.asarray(getattr(g, "_centers_np", None) if getattr(g, "_centers_np", None) is not None else g._centers_jax, float)
        if centers.shape[1] == 2:
            centers = np.column_stack([centers, np.zeros(len(centers))])
        kind = "cylinder" if isinstance(g, PackedCylinders) else "sphere"
        surf = Surface(kind, **({"axis": np.asarray(g.orientation, float).tolist()} if kind == "cylinder" else {}),
                       instances={"centers": centers.tolist(), "radii": np.asarray(g._radii_np, float).tolist()})
        wall = Wall("objects", surf, 1, 0, Directional(kappa, kappa), Sided(_rho(g), _rho(g)))
        bc = ["periodic", "periodic", "open"] if kind == "cylinder" else ["periodic"] * 3
        dom = Domain([-L / 2] * 3, [L / 2] * 3, bc)
        return SubstrateSpec(sid, dom, pools, [wall], Seeding([0]),
                             Validity(float(np.min(g._radii_np)), _tiers([wall], pools),
                                      min_gap=float(periodic_min_gap(centers[:, :2] if kind == "cylinder" else centers,
                                                                     np.asarray(g._radii_np, float), L))),
                             description=f"periodic cell of {len(centers)} {kind}s; extra-cellular walk", provenance=prov)
    if isinstance(g, MyelinatedCylinder):
        ax = np.asarray(g.orientation, float).tolist()
        pools = _pools_from_compartments(g, ["extra", "intra", "myelin"],
                                         {"extra": g.D_extra, "intra": g.D_intra, "myelin": g.D_myelin},
                                         {"extra": g.water_fractions[2], "intra": g.water_fractions[0], "myelin": g.water_fractions[1]})
        ki, ko = _f(g.kappa_inner) or 0.0, _f(g.kappa_outer) or 0.0
        inner = Wall("axolemma", Surface("cylinder", center=[0.0] * 3, axis=ax, radius=g.inner_radius), 1, 2, Directional(ki, ki))
        outer = Wall("sheath", Surface("cylinder", center=[0.0] * 3, axis=ax, radius=g.outer_radius), 2, 0, Directional(ko, ko))
        return SubstrateSpec(sid, Domain(*_box(MARGIN * g.outer_radius), ["open"] * 3), pools, [inner, outer],
                             Seeding([0, 1, 2]), Validity(min(g.inner_radius, g.outer_radius - g.inner_radius),
                                                          _tiers([inner, outer], pools)),
                             description="isolated myelinated cylinder: lumen, sheath, extra", provenance=prov)
    if isinstance(g, PackedMyelinatedCylinders):
        N = g.N_actual
        ax = np.asarray(g.orientation, float).tolist()
        inner = np.asarray(g._inner_radii_np[:N], float); outer = np.asarray(g._outer_radii_np[:N], float)
        centers = np.column_stack([np.asarray(g._centers_np[:N], float), np.zeros(N)])
        def scalar(arr, what):
            a = np.asarray(arr)[:N]
            if not np.allclose(a, a[0]):
                raise SpecError(f"{what} varies per axon; spec 0.1 wall properties are per wall, not per instance")
            return float(a[0])
        ki, ko = scalar(g._kappa_inner_jax, "kappa_inner"), scalar(g._kappa_outer_jax, "kappa_outer")
        ri, ro = scalar(g._rho_inner_jax, "rho_inner"), scalar(g._rho_outer_jax, "rho_outer")
        pools = _pools_from_compartments(g, ["extra", "intra", "myelin"],
                                         {"extra": scalar(g._D_extra_jax, "D_extra"), "intra": scalar(g._D_intra_jax, "D_intra"),
                                          "myelin": scalar(g._D_myelin_jax, "D_myelin")})
        L = float(g._L_float)
        w_in = Wall("axolemma", Surface("cylinder", axis=ax, instances={"centers": centers.tolist(), "radii": inner.tolist()}),
                    1, 2, Directional(ki, ki), Sided(ri, ri))
        w_out = Wall("sheath", Surface("cylinder", axis=ax, instances={"centers": centers.tolist(), "radii": outer.tolist()}),
                     2, 0, Directional(ko, ko), Sided(ro, ro))
        dom = Domain([-L / 2, -L / 2, -L / 2], [L / 2, L / 2, L / 2], ["periodic", "periodic", "open"])
        return SubstrateSpec(sid, dom, pools, [w_in, w_out], Seeding([0, 1, 2]),
                             Validity(float(min(inner.min(), (outer - inner).min())), _tiers([w_in, w_out], pools),
                                      min_gap=float(periodic_min_gap(centers[:, :2], outer, L))),
                             realisation={"n_objects": int(N), "packing_fraction": float(np.pi * np.sum(outer ** 2) / L ** 2),
                                          "cell_side": L, "g_ratio": float(np.mean(inner / outer))},
                             description=f"periodic cell of {N} myelinated cylinders", provenance=prov)
    if isinstance(g, CurvedMyelinatedCylinder):
        cl = np.asarray(g.centerline, float)
        pools = [Pool(0, "extra", None), Pool(1, "intra", None), Pool(2, "myelin", None, susceptibility=Susceptibility(None, None, "radial"))]
        inner = Wall("axolemma", Surface("swept_polyline", centerline=cl.tolist(), radius=g.r_in), 1, 2)
        outer = Wall("sheath", Surface("swept_polyline", centerline=cl.tolist(), radius=g.r_out), 2, 0)
        lo = (cl.min(0) - MARGIN * g.r_out).tolist(); hi = (cl.max(0) + MARGIN * g.r_out).tolist()
        seeded = {"intra": 1, "myelin": 2, "extra": 0}[g.pool]
        return SubstrateSpec(sid, Domain(lo, hi, ["open"] * 3), pools, [inner, outer], Seeding([seeded]),
                             Validity(min(g.r_in, g.r_out - g.r_in), _tiers([inner, outer], pools)),
                             description="myelinated curved axon: concentric shells swept along a polyline", provenance=prov)
    if isinstance(g, CurvedCylinder):
        cl = np.asarray(g.centerline, float)
        pools = [extra0, Pool(1, "intra", None, water_fraction=1.0)]
        wall = Wall("cylinder", Surface("swept_polyline", centerline=cl.tolist(), radius=g.radius), 1, None)
        lo = (cl.min(0) - MARGIN * g.radius).tolist(); hi = (cl.max(0) + MARGIN * g.radius).tolist()
        return SubstrateSpec(sid, Domain(lo, hi, ["open"] * 3), pools, [wall], Seeding([1]),
                             Validity(g.radius, _tiers([wall], pools)), description="curved cylinder; the lumen", provenance=prov)
    if isinstance(g, PackedCurvedCylinders):
        cls_ = [np.asarray(c, float).tolist() for c in g.centerlines]
        radii = np.asarray(g.radii, float).tolist()
        pools = [Pool(0, "extra", None, water_fraction=(0.0 if g.interior else 1.0)),
                 Pool(1, "intra", None, water_fraction=(1.0 if g.interior else 0.0))]
        wall = Wall("cylinders", Surface("swept_polyline", instances={"centerlines": cls_, "radii": radii}), 1, 0)
        if g.box is not None:
            lo, hi = g.box[0].tolist(), g.box[1].tolist(); bc = ["reflect" if g.box_reflect else "open"] * 3
        else:
            allp = np.concatenate([np.asarray(c, float) for c in g.centerlines])
            rmax = max(radii)
            lo = (allp.min(0) - MARGIN * rmax).tolist(); hi = (allp.max(0) + MARGIN * rmax).tolist(); bc = ["open"] * 3
        return SubstrateSpec(sid, Domain(lo, hi, bc), pools, [wall], Seeding([1 if g.interior else 0]),
                             Validity(min(radii), _tiers([wall], pools)),
                             description=f"extra-cellular walk around {len(radii)} curved tubes", provenance=prov)
    from ..geometry.mesh import Mesh
    if isinstance(g, Mesh):
        return _spec_of_mesh(g, sid, prov, surface_dir)
    raise SpecError(f"spec_of does not know {name}")


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _spec_of_mesh(g, sid, prov, surface_dir):
    from ..geometry.mesh import write_ply
    src = getattr(g, "source", None)
    if src is None:
        if surface_dir is None:
            raise SpecError("a Mesh built from arrays has no surface file; pass surface_dir= to write one "
                            "(a spec references surfaces as files)")
        import os
        os.makedirs(surface_dir, exist_ok=True)
        path = os.path.join(surface_dir, f"{sid.replace('/', '_')}.ply")
        write_ply(path, g.vertices, g.faces)
        src = {"file": path, "scale": 1.0}
    surf = Surface("mesh", file=src["file"], format=str(src["file"]).rsplit(".", 1)[-1].lower(), scale=float(src.get("scale", 1.0)),
                   sha256=_sha256(src["file"]))
    comps = g.compartments
    def pool(i, n):
        c = comps[n] if n in comps else None
        return Pool(i, n, (c.D if c is not None else None), water_fraction=1.0,
                    T2=(c.T2 if c is not None else None), T1=(c.T1 if c is not None else None))
    pools = [pool(0, "extra"), pool(1, "intra")]
    rho_nom = float(g.surface_relaxivity_t2 or 0.0)
    rho_in = rho_nom * float(g._rho_mult_intra); rho_out = rho_nom * float(g._rho_mult_extra)
    k_nom = float(g.permeability or 0.0)
    k_out = k_nom * float(g._kappa_mult_out); k_in = k_nom * float(g._kappa_mult_in)
    wall = Wall("surface", surf, 1, 0, Directional(k_out, k_in), Sided(rho_in, rho_out))
    bc = ["periodic" if p else ("reflect" if g.box_reflect else "open") for p in g.periodic]
    dom = Domain(np.asarray(g.vmin, float).tolist(), np.asarray(g.vmax, float).tolist(), bc)
    seeded = 1 if g.pool == "intra" else 0
    return SubstrateSpec(sid, dom, pools, [wall], Seeding([seeded]),
                         Validity(float(g.radius), _tiers([wall], pools), mesh_edge_feature_ratio=float(g.edge_median / g.radius)),
                         description="one closed (or periodic) triangle surface: inside is intra, outside extra",
                         provenance=dict(prov, files=[{"path": src["file"], "sha256": surf.sha256}], scale=surf.scale))


def _spec_of_sphere_union(g, sid, prov):
    pools = [Pool(0, "extra", None, water_fraction=(1.0 if g.pool == "extra" else 0.0)),
             Pool(1, "intra", None, water_fraction=(1.0 if g.pool == "intra" else 0.0))]
    wall = Wall("union", Surface("sphere_union", instances={"centers": g.centers.tolist(), "radii": g.radii.tolist()}),
                1, 0, Directional(), Sided(_rho(g) if g.pool == "intra" else 0.0, _rho(g) if g.pool == "extra" else 0.0))
    if g.box is not None:
        dom = Domain(g.box[0].tolist(), g.box[1].tolist(), ["reflect" if g.box_reflect else "open"] * 3)
    else:
        rmax = float(g.radii.max())
        lo = (g.centers.min(0) - MARGIN * rmax).tolist(); hi = (g.centers.max(0) + MARGIN * rmax).tolist()
        dom = Domain(lo, hi, ["open"] * 3)
    return SubstrateSpec(sid, dom, pools, [wall], Seeding([1 if g.pool == "intra" else 0]),
                         Validity(float(g.radius), _tiers([wall], pools)),
                         description=f"union of {len(g.radii)} spheres; the {g.pool} pool", provenance=prov)


def sphere_union_arrays(surface):
    """``(centers, radii)`` in metres of a ``sphere_union`` surface: inline instances, or the rows of a
    CATERPillar table selected by ``cell_type`` with the ``column`` radius."""
    s = surface
    if s.file:
        from ..io.caterpillar import read_caterpillar
        t = read_caterpillar(s.file, scale=(s.scale or 1.0), cell_types=(s.cell_type,) if s.cell_type else ("axon", "glial_cell"))
        r = t["r_in"] if s.column == "inner_radius" else t["r_out"]
        return t["centers"], r
    return np.asarray(s.instances["centers"], float), np.asarray(s.instances["radii"], float)


def polyline_arrays(surface):
    """``(centerlines, radii)`` of a ``swept_polyline`` surface: per-instance arrays or the single polyline."""
    s = surface
    if s.instances:
        return [np.asarray(c, float) for c in s.instances["centerlines"]], np.asarray(s.instances["radii"], float)
    return [np.asarray(s.centerline, float)], np.asarray([s.radius], float)


def as_geometry(substrate):
    """The geometry a driver walks, from any spelling of a substrate: a ``Geometry`` (returned as is), a
    :class:`SubstrateSpec`, a spec dict, or the path of a ``.sub.json``. ``None`` and duck-typed objects pass
    through untouched."""
    import os
    if substrate is None:
        return None
    if isinstance(substrate, SubstrateSpec):
        return geometry_from_spec(substrate)
    if isinstance(substrate, dict) and "walls" in substrate and "pools" in substrate:
        return geometry_from_spec(substrate)
    if isinstance(substrate, (str, os.PathLike)):
        from .substrate import load_spec
        return geometry_from_spec(load_spec(substrate))
    return substrate


def geometry_from_spec(spec):
    """The geometry a :class:`SubstrateSpec` describes (inverse of :func:`spec_of`); it keeps the spec it was built
    from as its ``.spec``, nominal values and provenance included."""
    spec = SubstrateSpec.from_dict(spec) if isinstance(spec, dict) else spec
    g = _geometry_from_spec(spec)
    g._spec_source = spec
    return g


def _geometry_from_spec(spec):
    from ..geometry import (FreeDiffusion, Box1D, Sphere, Cylinder, Ellipsoid, PackedCylinders, PackedSpheres,
                            MyelinatedCylinder, PackedMyelinatedCylinders, CurvedCylinder, CurvedMyelinatedCylinder,
                            PackedCurvedCylinders, SphereUnion)
    from ..geometry.analytic import PermeableSlab1D, PermeableShell
    from ..compartments import Compartments, Pool as CPool
    spec = SubstrateSpec.from_dict(spec) if isinstance(spec, dict) else spec
    spec.validate()
    walls = list(spec.walls)
    dom = spec.domain
    rho_dom = dom.surface_relaxivity or None

    def rho(w):
        r = w.surface_relaxivity.inside or w.surface_relaxivity.outside
        return r or None

    def kappa(w):
        k = w.permeability.in_to_out
        return k if k > 0 else None

    if not walls:
        if dom.boundary == ["open"] * 3:
            return FreeDiffusion()
        if dom.boundary == ["reflect", "open", "open"]:
            return Box1D(dom.box_max[0] - dom.box_min[0], surface_relaxivity_t2=rho_dom)
        raise SpecError("a wall-less spec is free diffusion (open) or a 1-D slab (reflect in x)")
    kinds = [w.surface.kind for w in walls]
    if any(k == "sphere_union" for k in kinds) or (len(walls) > 1 and any(w.surface.instances for w in walls)
                                                    and any(k == "swept_polyline" for k in kinds)):
        if len(walls) != 1 or len(spec.seeding.pools) != 1:
            raise SpecError("a multi-surface (or multi-pool) sphere-union / strand spec is walked pool by pool by "
                            "spec.walk_spec; it has no single Geometry")
        w = walls[0]; s = w.surface
        seeded = spec.seeding.pools[0]
        pool = "intra" if seeded == w.inside_pool else "extra"
        box = (dom.box_min, dom.box_max) if "reflect" in dom.boundary else None
        if s.kind == "sphere_union":
            centers, radii = sphere_union_arrays(s)
            return SphereUnion(centers, radii, pool=pool, feature_radius=spec.validity.smallest_feature,
                               surface_relaxivity_t2=rho(w), box=box)
    if any(k == "mesh" for k in kinds):
        if len(walls) != 1:
            raise SpecError("a multi-surface mesh spec is walked pool by pool by spec.walk_spec; it has no "
                            "single Geometry")
        from ..geometry.mesh import Mesh
        from ..compartments import Compartments, Pool as CPool
        w = walls[0]; s = w.surface
        pools = {p.name: p for p in spec.pools}
        comps = {}
        for n in ("extra", "intra"):
            kw = {k: getattr(pools[n], k) for k in ("D", "T2", "T1") if getattr(pools[n], k) is not None}
            if w.surface_relaxivity.inside > 0 and n == "intra": kw["surface_relaxivity_t2"] = w.surface_relaxivity.inside
            if w.surface_relaxivity.outside > 0 and n == "extra": kw["surface_relaxivity_t2"] = w.surface_relaxivity.outside
            if kw: comps[n] = CPool(**kw)
        perm = None
        if w.permeability.in_to_out > 0 or w.permeability.out_to_in > 0:
            perm = ({"intra_to_extra": w.permeability.in_to_out, "extra_to_intra": w.permeability.out_to_in}
                    if w.permeability.in_to_out != w.permeability.out_to_in else w.permeability.in_to_out)
        m = Mesh.from_ply(s.file, scale=(s.scale or 1.0), periodic=[b == "periodic" for b in dom.boundary],
                          voxel_min=dom.box_min, voxel_max=dom.box_max, feature_radius=spec.validity.smallest_feature,
                          permeability=perm, compartments=(Compartments(comps) if comps else None),
                          pool={1: "intra", 0: "extra"}[spec.seeding.pools[0]],
                          box_reflect=("reflect" in dom.boundary))
        return m
    if len(walls) == 1:
        w = walls[0]; s = w.surface
        if s.kind == "plane":
            return PermeableSlab1D(dom.box_max[0] - dom.box_min[0], w.permeability.in_to_out, surface_relaxivity_t2=rho(w))
        if s.instances:
            if s.kind == "swept_polyline":
                cls_, radii = polyline_arrays(s)
                return PackedCurvedCylinders(cls_, radii, interior=(spec.seeding.pools[0] == w.inside_pool),
                                         box=((dom.box_min, dom.box_max) if "reflect" in dom.boundary else None))
            centers = np.asarray(s.instances["centers"], float)
            L = float(dom.box_max[0] - dom.box_min[0])
            if s.kind == "cylinder":
                return PackedCylinders(s.instances["radii"], centers[:, :2], L, orientation=tuple(s.axis or (0, 0, 1)),
                                       surface_relaxivity_t2=rho(w), permeability=kappa(w))
            if s.kind == "sphere":
                return PackedSpheres(s.instances["radii"], centers, L, surface_relaxivity_t2=rho(w), permeability=kappa(w))
        if s.kind == "sphere":
            return Sphere(s.radius, surface_relaxivity_t2=rho(w), permeability=kappa(w))
        if s.kind == "cylinder":
            return Cylinder(s.radius, tuple(s.axis or (0, 0, 1)), surface_relaxivity_t2=rho(w), permeability=kappa(w))
        if s.kind == "ellipsoid":
            return Ellipsoid(s.semiaxes, surface_relaxivity_t2=rho(w), permeability=kappa(w))
        if s.kind == "swept_polyline":
            return CurvedCylinder(np.asarray(s.centerline), s.radius)
    if len(walls) == 2:
        a, b = walls
        if {a.name, b.name} == {"membrane", "outer"}:
            inner = spec.wall("membrane"); outer = spec.wall("outer")
            kind = inner.surface.kind
            return PermeableShell(inner.surface.radius, outer.surface.radius, inner.permeability.in_to_out, kind=kind,
                                  orientation=tuple(inner.surface.axis or (0, 0, 1)), surface_relaxivity_t2=rho(inner))
        if {a.name, b.name} == {"axolemma", "sheath"}:
            inner = spec.wall("axolemma"); outer = spec.wall("sheath")
            pools = {p.name: p for p in spec.pools}
            comps = Compartments({n: CPool(T2=pools[n].T2, T1=pools[n].T1, water_fraction=pools[n].water_fraction)
                                  for n in ("extra", "intra", "myelin")
                                  if pools[n].T2 is not None or pools[n].T1 is not None})
            if inner.surface.kind == "swept_polyline":
                seeded = {1: "intra", 2: "myelin", 0: "extra"}[spec.seeding.pools[0]]
                return CurvedMyelinatedCylinder(np.asarray(inner.surface.centerline), inner.surface.radius, outer.surface.radius, pool=seeded)
            D = {n: (pools[n].D if pools[n].D is not None else 0.0) for n in pools}
            if inner.surface.instances:
                centers = np.asarray(inner.surface.instances["centers"], float)[:, :2]
                ri = np.asarray(inner.surface.instances["radii"], float); ro = np.asarray(outer.surface.instances["radii"], float)
                L = float(dom.box_max[0] - dom.box_min[0])
                return PackedMyelinatedCylinders(ri, ri / ro, centers, L, N_max=int(2 ** np.ceil(np.log2(max(len(ri), 2)))),
                                                 orientation=tuple(inner.surface.axis or (0, 0, 1)),
                                                 D_intra=D["intra"], D_myelin=D["myelin"], D_extra=D["extra"],
                                                 kappa_inner=inner.permeability.in_to_out, kappa_outer=outer.permeability.in_to_out,
                                                 rho_inner=inner.surface_relaxivity.inside, rho_outer=outer.surface_relaxivity.inside,
                                                 compartments=(comps if len(comps) else None))
            wf = (pools["intra"].water_fraction, pools["myelin"].water_fraction, pools["extra"].water_fraction)
            return MyelinatedCylinder(inner.surface.radius, outer.surface.radius, tuple(inner.surface.axis or (0, 0, 1)),
                                      D["intra"], D["extra"], D_myelin=D["myelin"],
                                      kappa_inner=kappa(inner), kappa_outer=kappa(outer),
                                      water_fractions=(None if len(comps) else wf),     # the pools carry them
                                      compartments=(comps if len(comps) else None))
    raise SpecError(f"no geometry matches walls {kinds}")
