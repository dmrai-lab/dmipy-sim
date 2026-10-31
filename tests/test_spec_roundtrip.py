"""Every analytic geometry writes its situation as a spec, and the spec builds the geometry back:
same class, same spec, same walk to the bit. The engine can therefore read specs instead of code."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.geometry.analytic import PermeableSlab1D, PermeableShell
from dmipy_sim.spec import spec_of, geometry_from_spec, SubstrateSpec, SpecError, load_spec

D = 2e-9


def _packed_myelin():
    L = float(np.sqrt(np.pi * 3 * (1e-6 / 0.7) ** 2 / 0.5))
    _, _, c = d.pack_myelinated_cylinders([1e-6] * 3, 0.7, None, cell_size=L, seed=0)
    return d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, rho_inner=1e-6, rho_outer=1e-6, kappa_inner=1e-5,
                                       compartments=d.Compartments(extra=d.Pool(T2=0.08), intra=d.Pool(T2=0.05), myelin=d.Pool(T2=0.01)))


def _packed_cyl():
    c, L, _ = d.pack_cylinders([1e-6, 0.8e-6, 1.2e-6], target_vf=0.3, seed=1)
    return d.PackedCylinders([1e-6, 0.8e-6, 1.2e-6], c, L, surface_relaxivity_t2=2e-6, permeability=1e-5)


def _packed_sph():
    c, L, _ = d.pack_spheres([1e-6, 0.8e-6], target_vf=0.1, seed=1)
    return d.PackedSpheres([1e-6, 0.8e-6], c, L)


CL = np.stack([np.zeros(6), np.linspace(0, 8e-6, 6) ** 2 / 8e-6, np.linspace(0, 8e-6, 6)], axis=1)
GEOMETRIES = {
    "free": lambda: d.FreeDiffusion(),
    "box1d": lambda: d.Box1D(4e-6, surface_relaxivity_t2=1e-6),
    "sphere": lambda: d.Sphere(3e-6, surface_relaxivity_t2=1e-6),
    "cylinder_permeable": lambda: d.Cylinder(3e-6, (0, 0, 1), permeability=2e-5),
    "ellipsoid": lambda: d.Ellipsoid((2e-6, 3e-6, 4e-6)),
    "slab": lambda: PermeableSlab1D(6e-6, 1e-5, surface_relaxivity_t2=1e-6),
    "shell_cyl": lambda: PermeableShell(2e-6, 4e-6, 1e-5, kind="cylinder"),
    "packed_cyl": _packed_cyl,
    "packed_sph": _packed_sph,
    "myelinated": lambda: d.MyelinatedCylinder(2e-6, 3e-6, (0, 0, 1), kappa_inner=1e-5,
                                               compartments=d.Compartments(intra=d.Pool(D=D, T2=0.05), myelin=d.Pool(T2=0.01), extra=d.Pool(D=D, T2=0.08))),
    "packed_myelin": _packed_myelin,
    "curved": lambda: d.CurvedCylinder(CL, 1e-6),
    "multishell": lambda: d.CurvedMyelinatedCylinder(CL, 1e-6, 1.5e-6, pool="myelin"),
    "packed_curved": lambda: d.PackedCurvedCylinders([CL, CL + np.array([4e-6, 0, 0])], [1e-6, 1.5e-6]),
}


@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_spec_round_trip_rebuilds_the_same_geometry(name):
    g = GEOMETRIES[name]()
    spec = g.spec
    spec.validate()
    again = SubstrateSpec.from_json(spec.to_json())
    assert again == spec
    g2 = geometry_from_spec(again)
    assert type(g2) is type(g)
    assert g2.spec == spec                                  # the spec is a fixed point
    if name in ("packed_curved", "multishell", "curved"):
        return                                             # no producer walk for these (fused kernels only)
    w1 = _walk_geometry(48, D, g, 4e-4, 2e-4, seed=1, require_gpu=False)
    w2 = _walk_geometry(48, D, g2, 4e-4, 2e-4, seed=1, require_gpu=False)
    np.testing.assert_array_equal(w1.position_coeffs, w2.position_coeffs)


def test_the_spec_says_what_the_constructor_left_unsaid():
    s = d.Cylinder(5e-6, (0, 0, 1)).spec
    assert s.domain.boundary == ["open"] * 3 and s.wall("membrane").outside_pool is None and s.seeding.pools == [1]
    assert s.pool("extra").water_fraction == 0.0 and s.validity.tiers == ["gradient", "surface"]
    p = d.Cylinder(5e-6, (0, 0, 1), permeability=1e-5).spec
    assert p.wall("membrane").outside_pool == 0 and "exchange" in p.validity.tiers
    pm = _packed_myelin().spec
    assert pm.domain.boundary == ["periodic", "periodic", "open"] and pm.seeding.pools == [0, 1, 2]
    assert [q.name for q in pm.field_source_pools] == ["myelin"] and pm.realisation["n_objects"] == 3
    assert pm.wall("axolemma").permeability.in_to_out == pytest.approx(1e-5, rel=1e-6) and pm.pool("myelin").T2 == 0.01
    assert _packed_cyl().spec.seeding.pools == [0, 1]       # both pools hold water and both are seeded
    assert [p.water_fraction for p in _packed_cyl().spec.pools] == [1.0, 1.0]
    extra = d.PackedCylinders([1e-6, 0.8e-6], [[-2e-6, 0.0], [2e-6, 0.0]], 8e-6, pool="extra").spec
    assert extra.seeding.pools == [0] and [p.water_fraction for p in extra.pools] == [1.0, 1.0]
    assert geometry_from_spec(extra).pool == "extra" and geometry_from_spec(_packed_cyl().spec).pool is None


def test_the_published_examples_build_geometries(tmp_path):
    import pathlib
    fix = pathlib.Path(__file__).parent / "fixtures" / "substrates"
    cyl = geometry_from_spec(load_spec(fix / "isolated_cylinder.sub.json"))
    assert isinstance(cyl, d.Cylinder) and cyl.radius == 5e-6
    pm = geometry_from_spec(load_spec(fix / "packed_myelinated_cylinders.sub.json"))
    assert isinstance(pm, d.PackedMyelinatedCylinders) and pm.N_actual == 3
    with pytest.raises(SpecError, match="pool by pool"):                 # a multi-surface bundle: walk's job
        geometry_from_spec(load_spec(fix / "mesh_bundle.sub.json"))


def test_per_instance_wall_properties_are_refused_honestly():
    L = float(np.sqrt(np.pi * 3 * (1e-6 / 0.7) ** 2 / 0.5))
    _, _, c = d.pack_myelinated_cylinders([1e-6] * 3, 0.7, None, cell_size=L, seed=0)
    g2 = d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, rho_inner=[1e-6, 2e-6, 3e-6])
    with pytest.raises(SpecError, match="varies per axon"):
        g2.spec


def test_scaled_is_the_same_shape_at_another_size():
    """Every length x s, each stated D x s^2, every wall rate x s (dmipy-sim#542); a file-backed surface is read at
    s times its units; an instance array of unknown unit is refused."""
    import dataclasses
    from dmipy_replay.spec.substrate import Directional, Sided, Surface, Wall
    spec = spec_of(_packed_myelin())
    s = 2.0
    out = spec.scaled(s)
    np.testing.assert_allclose(out.domain.box_min, s * np.asarray(spec.domain.box_min))
    np.testing.assert_allclose(out.domain.box_max, s * np.asarray(spec.domain.box_max))
    assert out.validity.smallest_feature == s * spec.validity.smallest_feature
    for w0, w1 in zip(spec.walls, out.walls):
        if w0.surface.radius is not None:
            assert w1.surface.radius == s * w0.surface.radius
        if w0.surface.instances:
            for k, v in w0.surface.instances.items():
                np.testing.assert_allclose(w1.surface.instances[k], s * np.asarray(v, float))
        assert w1.permeability == Directional(s * w0.permeability.in_to_out, s * w0.permeability.out_to_in)
        assert w1.surface_relaxivity == Sided(s * w0.surface_relaxivity.inside, s * w0.surface_relaxivity.outside)
    for p0, p1 in zip(spec.pools, out.pools):
        assert p1.D == (None if p0.D is None else s * s * p0.D) and p1.T2 == p0.T2 and p1.water_fraction == p0.water_fraction
    assert out.provenance["transformations"][-1].startswith("scaled by 2.0")
    assert SubstrateSpec.from_json(out.to_json()) == out
    # a file-backed wall is read at s times its units, an inline instance table scales each length
    gradient_only = dataclasses.replace(spec.validity, tiers=["gradient", "field"])   # the myelin pool is magnetic
    inst = dataclasses.replace(spec, validity=gradient_only,
                               walls=[Wall("union", Surface("sphere_union", instances={"centers": [[0.0, 0.0, 0.0]], "radii": [1e-6]}), 1, 0),
                                      Wall("mesh", Surface("mesh", file="x.ply", format="ply"), 1, 0)])
    got = inst.scaled(0.5)
    assert got.walls[0].surface.instances == {"centers": [[0.0, 0.0, 0.0]], "radii": [0.5e-6]}
    assert got.walls[1].surface.scale == 0.5
    with pytest.raises(SpecError, match="not known as lengths"):
        dataclasses.replace(spec, validity=gradient_only,
                            walls=[Wall("u", Surface("sphere_union", instances={"radii": [1e-6], "spin": [1]}), 1, 0)]).scaled(2)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(SpecError, match="positive"):
            spec.scaled(bad)
