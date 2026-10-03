"""`geometry_from_spec` either WALKS a spec field or REFUSES it naming the field (dmrai-lab/dmipy-sim#536),
and T2/T1 -- replay knobs, SUBSTRATE.md 4.6 -- never reach the geometry and never decide acceptance
(dmrai-lab/dmipy-sim#540). Every family's own ``spec_of(geometry_from_spec(spec))`` is a fixed point over
the walk-shaping fields (geometry, D, permeability both directions, water fraction, seeding, domain);
T2/T1 are excluded on purpose, since the geometry never carries them."""
import dataclasses

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.geometry.analytic import PermeableSlab1D, PermeableShell
from dmipy_sim.spec import spec_of, geometry_from_spec, SpecError
from dmipy_sim.spec.substrate import Directional, Sided

D = 2e-9
CL = np.stack([np.zeros(6), np.linspace(0, 8e-6, 6) ** 2 / 8e-6, np.linspace(0, 8e-6, 6)], axis=1)


def _packed_cyl():
    c, L, _ = d.pack_cylinders([1e-6, 0.8e-6, 1.2e-6], target_vf=0.3, seed=1)
    return d.PackedCylinders([1e-6, 0.8e-6, 1.2e-6], c, L, surface_relaxivity_t2=2e-6, permeability=1e-5)


def _packed_sph():
    c, L, _ = d.pack_spheres([1e-6, 0.8e-6], target_vf=0.1, seed=1)
    return d.PackedSpheres([1e-6, 0.8e-6], c, L, surface_relaxivity_t2=1e-6, permeability=3e-5)


def _packed_myelin():
    L = float(np.sqrt(np.pi * 3 * (1e-6 / 0.7) ** 2 / 0.5))
    _, _, c = d.pack_myelinated_cylinders([1e-6] * 3, 0.7, None, cell_size=L, seed=0)
    return d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, rho_inner=1e-6, rho_outer=1e-6,
                                       kappa_inner=1e-5, kappa_outer=2e-5)


FAMILIES = {
    "sphere": lambda: d.Sphere(3e-6, surface_relaxivity_t2=1e-6, permeability=2e-5),
    "cylinder": lambda: d.Cylinder(3e-6, (0, 0, 1), surface_relaxivity_t2=1e-6, permeability=2e-5),
    "ellipsoid": lambda: d.Ellipsoid((2e-6, 3e-6, 4e-6), surface_relaxivity_t2=1e-6),
    "myelinated": lambda: d.MyelinatedCylinder(2e-6, 3e-6, (0, 0, 1), D, D, kappa_inner=1e-5, kappa_outer=2e-5),
    "packed_cylinders": _packed_cyl,
    "packed_spheres": _packed_sph,
    "packed_myelinated": _packed_myelin,
    "permeable_slab": lambda: PermeableSlab1D(6e-6, 1e-5, surface_relaxivity_t2=1e-6),
    "permeable_shell_sphere": lambda: PermeableShell(2e-6, 4e-6, 1e-5, kind="sphere", surface_relaxivity_t2=1e-6),
    "permeable_shell_cylinder": lambda: PermeableShell(2e-6, 4e-6, 1e-5, kind="cylinder", surface_relaxivity_t2=1e-6),
    "curved_cylinder": lambda: d.CurvedCylinder(CL, 1e-6, surface_relaxivity_t2=1e-6),
    "curved_myelinated": lambda: d.CurvedMyelinatedCylinder(CL, 1e-6, 1.5e-6, pool="myelin", surface_relaxivity_t2=1e-6),
    "packed_curved": lambda: d.PackedCurvedCylinders([CL, CL + np.array([4e-6, 0.0, 0.0])], [1e-6, 1.5e-6],
                                                      surface_relaxivity_t2=1e-6),
    "sphere_union": lambda: d.SphereUnion(np.array([[0.0, 0.0, 0.0], [5e-6, 0.0, 0.0]]), np.array([2e-6, 2e-6]),
                                          pool="intra", surface_relaxivity_t2=1e-6),
}

#: pool keys that are replay knobs, never walked into the geometry (#540) -- excluded from the fixed point
_REPLAY_POOL_KEYS = {"T2", "T1"}
_METADATA_KEYS = {"id", "description", "provenance", "validity", "realisation", "request",
                  "nominal_field_T", "substrate_spec_version"}


def _walk_shaping(spec):
    """``spec.to_dict()`` restricted to the fields a walk is shaped by: not metadata, not T2/T1."""
    dct = dict(spec.to_dict())
    for k in _METADATA_KEYS:
        dct.pop(k, None)
    dct["pools"] = [{k: v for k, v in p.items() if k not in _REPLAY_POOL_KEYS} for p in dct["pools"]]
    return dct


def _assert_close(a, b, path="spec"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: {a!r} vs {b!r}"
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{path}: {a!r} vs {b!r}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        assert abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(a))), f"{path}: {a!r} != {b!r}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def assert_walk_equal(spec_a, spec_b):
    """``spec_a`` and ``spec_b`` agree on every walk-shaping field (float32-tolerant); T2/T1 and metadata
    are not compared, since the geometry never carries them (#540)."""
    _assert_close(_walk_shaping(spec_a), _walk_shaping(spec_b))


@pytest.mark.parametrize("name", list(FAMILIES))
def test_spec_of_geometry_from_spec_is_a_fixed_point(name):
    """``spec_of(geometry_from_spec(spec_of(g)))`` recomputed at every step (never through the geometry's
    cached ``.spec``, which would trivially pass) equals the original spec over the walk-shaping fields."""
    g = FAMILIES[name]()
    s1 = spec_of(g)
    s1.validate()
    g2 = geometry_from_spec(s1)
    assert type(g2) is type(g)
    s2 = spec_of(g2)                      # recomputed from g2's own state, not the _spec_source cache
    assert_walk_equal(s1, s2)


def test_540_a_nominal_t2_on_one_pool_does_not_make_an_isolated_myelinated_spec_unacceptable():
    g = d.MyelinatedCylinder(2e-6, 2.8e-6, (0, 0, 1), 2e-9, 2e-9)
    spec = g.spec
    spec = spec.replace(T2={"intra": 0.05})
    g2 = geometry_from_spec(spec)                     # must not raise
    assert isinstance(g2, d.MyelinatedCylinder)
    assert g2.water_fractions == g.water_fractions     # water fraction, not T2, shapes the walk
    assert g2.T2_intra is None and g2.T2_myelin is None and g2.T2_extra is None   # T2 never reaches the geometry


@pytest.mark.parametrize("name,length_field", [("sphere", False), ("cylinder", True), ("ellipsoid", False)])
def test_an_off_centre_surface_is_refused_naming_center(name, length_field):
    g = FAMILIES[name]()
    spec = spec_of(g)
    edited = dataclasses.replace(spec.walls[0].surface, center=[3e-6, 0.0, 0.0])
    spec = dataclasses.replace(spec, walls=[dataclasses.replace(spec.walls[0], surface=edited)])
    with pytest.raises(SpecError, match="center"):
        geometry_from_spec(spec)


@pytest.mark.parametrize("name", ["cylinder", "myelinated"])
def test_a_cylinder_kind_length_is_refused(name):
    g = FAMILIES[name]()
    spec = spec_of(g)
    w0 = spec.walls[0]
    edited = dataclasses.replace(w0.surface, length=1e-5)
    spec = dataclasses.replace(spec, walls=[dataclasses.replace(w0, surface=edited)] + list(spec.walls[1:]))
    with pytest.raises(SpecError, match="length"):
        geometry_from_spec(spec)


def test_an_ellipsoid_rotation_is_refused():
    g = FAMILIES["ellipsoid"]()
    spec = spec_of(g)
    edited = dataclasses.replace(spec.walls[0].surface, rotation=[0.0, 0.0, 1.0, 0.0])
    spec = dataclasses.replace(spec, walls=[dataclasses.replace(spec.walls[0], surface=edited)])
    with pytest.raises(SpecError, match="rotation"):
        geometry_from_spec(spec)


@pytest.mark.parametrize("name", ["sphere", "cylinder", "ellipsoid", "packed_cylinders", "packed_spheres"])
def test_asymmetric_permeability_is_refused_on_single_kappa_families(name):
    g = FAMILIES[name]()
    spec = spec_of(g)
    w0 = spec.walls[0]
    asym = dataclasses.replace(w0, permeability=Directional(in_to_out=2e-5, out_to_in=1e-5), outside_pool=0)
    spec = dataclasses.replace(spec, walls=[asym] + list(spec.walls[1:]))
    with pytest.raises(SpecError, match="asymmetric"):
        geometry_from_spec(spec)


def test_asymmetric_permeability_is_refused_on_the_myelinated_axolemma():
    g = FAMILIES["myelinated"]()
    spec = spec_of(g)
    axolemma = spec.wall("axolemma")
    asym = dataclasses.replace(axolemma, permeability=Directional(in_to_out=2e-5, out_to_in=1e-5))
    walls = [asym if w is axolemma else w for w in spec.walls]
    spec = dataclasses.replace(spec, walls=walls)
    with pytest.raises(SpecError, match="asymmetric"):
        geometry_from_spec(spec)


@pytest.mark.parametrize("name", ["sphere", "myelinated", "permeable_shell_sphere"])
def test_mt_reactivity_is_refused_everywhere(name):
    g = FAMILIES[name]()
    spec = spec_of(g)
    w0 = spec.walls[0]
    reactive = dataclasses.replace(w0, mt_reactivity=Sided(inside=1e-6, outside=1e-6))
    spec = dataclasses.replace(spec, walls=[reactive] + list(spec.walls[1:]))
    with pytest.raises(SpecError, match="mt_reactivity"):
        geometry_from_spec(spec)


@pytest.mark.parametrize("name", ["curved_cylinder", "packed_curved"])
def test_permeability_on_the_curved_family_is_refused(name):
    g = FAMILIES[name]()
    spec = spec_of(g)
    w0 = spec.walls[0]
    perm = dataclasses.replace(w0, permeability=Directional(in_to_out=1e-5, out_to_in=1e-5), outside_pool=0)
    spec = dataclasses.replace(spec, walls=[perm])
    with pytest.raises(SpecError, match="permeab"):
        geometry_from_spec(spec)


def test_permeability_on_curved_myelinated_is_refused():
    g = FAMILIES["curved_myelinated"]()
    spec = spec_of(g)
    axolemma = spec.wall("axolemma")
    perm = dataclasses.replace(axolemma, permeability=Directional(in_to_out=1e-5, out_to_in=1e-5))
    walls = [perm if w is axolemma else w for w in spec.walls]
    spec = dataclasses.replace(spec, walls=walls)
    with pytest.raises(SpecError, match="permeab"):
        geometry_from_spec(spec)


def test_a_plane_off_the_x_mid_slab_is_refused():
    g = FAMILIES["permeable_slab"]()
    spec = spec_of(g)
    w0 = spec.walls[0]
    moved = dataclasses.replace(w0.surface, point=[0.0, 1e-6, 0.0])
    spec = dataclasses.replace(spec, walls=[dataclasses.replace(w0, surface=moved)])
    with pytest.raises(SpecError, match="plane|x-mid-slab"):
        geometry_from_spec(spec)


def test_permeability_on_a_sphere_union_is_refused():
    g = FAMILIES["sphere_union"]()
    spec = spec_of(g)
    w0 = spec.walls[0]
    perm = dataclasses.replace(w0, permeability=Directional(in_to_out=1e-5, out_to_in=1e-5), outside_pool=0)
    spec = dataclasses.replace(spec, walls=[perm])
    with pytest.raises(SpecError, match="permeab"):
        geometry_from_spec(spec)


def test_myelinated_dispatch_is_structural_not_by_wall_or_pool_name():
    """A spec with the right pool TOPOLOGY (intra -> myelin -> extra) but none of the conventional names
    still builds a MyelinatedCylinder (dmrai-lab/dmipy-sim#536 item 4). Pool 0 stays named 'extra' -- the
    spec format itself requires that, independent of this dispatch rule."""
    g = FAMILIES["myelinated"]()
    spec = spec_of(g)
    pools = [spec.pool(0),
            dataclasses.replace(spec.pool(1), name="lumen"),
            dataclasses.replace(spec.pool(2), name="sheath_water")]
    walls = [dataclasses.replace(spec.wall("axolemma"), name="inner_wall"),
            dataclasses.replace(spec.wall("sheath"), name="outer_wall")]
    renamed = dataclasses.replace(spec, pools=pools, walls=walls)
    renamed.validate()
    g2 = geometry_from_spec(renamed)
    assert isinstance(g2, d.MyelinatedCylinder)
    assert g2.inner_radius == g.inner_radius and g2.outer_radius == g.outer_radius
    assert g2.kappa_inner == pytest.approx(g.kappa_inner) and g2.kappa_outer == pytest.approx(g.kappa_outer)


def test_permeable_shell_dispatch_is_structural_not_by_wall_name():
    g = FAMILIES["permeable_shell_sphere"]()
    spec = spec_of(g)
    walls = [dataclasses.replace(w, name=f"wall_{i}") for i, w in enumerate(spec.walls)]
    renamed = dataclasses.replace(spec, walls=walls)
    renamed.validate()
    g2 = geometry_from_spec(renamed)
    assert isinstance(g2, PermeableShell)
    assert g2.r_inner == g.r_inner and g2.r_outer == g.r_outer
