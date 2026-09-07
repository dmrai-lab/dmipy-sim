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
                        susceptibility=(Susceptibility(0.0, 0.0, "radial") if n == "myelin" else None)))
    return out


def spec_of(geometry, *, id=None, provenance=None):
    """The :class:`SubstrateSpec` of an analytic geometry."""
    from ..geometry import (FreeDiffusion, Box1D, Sphere, Cylinder, Ellipsoid, PackedCylinders, PackedSpheres,
                            MyelinatedCylinder, PackedMyelinatedCylinders, CurvedTube, MultiShellCurvedTube,
                            PackedCurvedTubes)
    from ..geometry.analytic import PermeableSlab1D, PermeableShell
    g = geometry
    name = type(g).__name__
    sid = id or f"analytic/{name.lower()}"
    prov = provenance or {"source": "analytic", "constructor": name}
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
    if isinstance(g, MultiShellCurvedTube):
        cl = np.asarray(g.centerline, float)
        pools = [Pool(0, "extra", None), Pool(1, "intra", None), Pool(2, "myelin", None, susceptibility=Susceptibility(0.0, 0.0, "radial"))]
        inner = Wall("axolemma", Surface("swept_polyline", centerline=cl.tolist(), radius=g.r_in), 1, 2)
        outer = Wall("sheath", Surface("swept_polyline", centerline=cl.tolist(), radius=g.r_out), 2, 0)
        lo = (cl.min(0) - MARGIN * g.r_out).tolist(); hi = (cl.max(0) + MARGIN * g.r_out).tolist()
        seeded = {"intra": 1, "myelin": 2, "extra": 0}[g.pool]
        return SubstrateSpec(sid, Domain(lo, hi, ["open"] * 3), pools, [inner, outer], Seeding([seeded]),
                             Validity(min(g.r_in, g.r_out - g.r_in), _tiers([inner, outer], pools)),
                             description="myelinated curved axon: concentric shells swept along a polyline", provenance=prov)
    if isinstance(g, CurvedTube):
        cl = np.asarray(g.centerline, float)
        pools = [extra0, Pool(1, "intra", None, water_fraction=1.0)]
        wall = Wall("tube", Surface("swept_polyline", centerline=cl.tolist(), radius=g.radius), 1, None)
        lo = (cl.min(0) - MARGIN * g.radius).tolist(); hi = (cl.max(0) + MARGIN * g.radius).tolist()
        return SubstrateSpec(sid, Domain(lo, hi, ["open"] * 3), pools, [wall], Seeding([1]),
                             Validity(g.radius, _tiers([wall], pools)), description="curved tube; the lumen", provenance=prov)
    if isinstance(g, PackedCurvedTubes):
        cls_ = [np.asarray(c, float).tolist() for c in g.centerlines]
        radii = np.asarray(g.radii, float).tolist()
        pools = [Pool(0, "extra", None, water_fraction=1.0), Pool(1, "intra", None, water_fraction=0.0)]
        wall = Wall("tubes", Surface("swept_polyline", instances={"centerlines": cls_, "radii": radii}), 1, 0)
        allp = np.concatenate([np.asarray(c, float) for c in g.centerlines])
        rmax = max(radii)
        lo = (allp.min(0) - MARGIN * rmax).tolist(); hi = (allp.max(0) + MARGIN * rmax).tolist()
        return SubstrateSpec(sid, Domain(lo, hi, ["open"] * 3), pools, [wall], Seeding([0]),
                             Validity(min(radii), _tiers([wall], pools)),
                             description=f"extra-cellular walk around {len(radii)} curved tubes", provenance=prov)
    raise SpecError(f"spec_of does not know {name}; mesh substrates follow in #130 PR 3")


def geometry_from_spec(spec):
    """The analytic geometry a :class:`SubstrateSpec` describes (inverse of :func:`spec_of`)."""
    from ..geometry import (FreeDiffusion, Box1D, Sphere, Cylinder, Ellipsoid, PackedCylinders, PackedSpheres,
                            MyelinatedCylinder, PackedMyelinatedCylinders, CurvedTube, MultiShellCurvedTube,
                            PackedCurvedTubes)
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
    if len(walls) == 1:
        w = walls[0]; s = w.surface
        if s.kind == "plane":
            return PermeableSlab1D(dom.box_max[0] - dom.box_min[0], w.permeability.in_to_out, surface_relaxivity_t2=rho(w))
        if s.instances:
            if s.kind == "swept_polyline":
                return PackedCurvedTubes([np.asarray(c) for c in s.instances["centerlines"]], s.instances["radii"])
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
            return CurvedTube(np.asarray(s.centerline), s.radius)
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
            comps = Compartments({n: CPool(T2=pools[n].T2, T1=pools[n].T1) for n in ("extra", "intra", "myelin")
                                  if pools[n].T2 is not None or pools[n].T1 is not None})
            if inner.surface.kind == "swept_polyline":
                seeded = {1: "intra", 2: "myelin", 0: "extra"}[spec.seeding.pools[0]]
                return MultiShellCurvedTube(np.asarray(inner.surface.centerline), inner.surface.radius, outer.surface.radius, pool=seeded)
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
                                      kappa_inner=kappa(inner), kappa_outer=kappa(outer), water_fractions=wf,
                                      compartments=(comps if len(comps) else None))
    raise SpecError(f"no analytic geometry matches walls {kinds}; mesh substrates follow in #130 PR 3")
