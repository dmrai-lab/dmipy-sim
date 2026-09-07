"""The substrate spec (replay-pack-spec/SUBSTRATE.md 0.1): the examples validate, a spec round-trips
through JSON, and every invariant the schema cannot express is refused with the field named."""
import copy
import json
import pathlib

import pytest

from dmipy_sim.spec import SubstrateSpec, SpecError, validate, load_spec, SCHEMA_PATH, Pool, Wall, Surface

FIX = pathlib.Path(__file__).parent / "fixtures" / "substrates"
EXAMPLES = sorted(FIX.glob("*.sub.json"))


def test_schema_file_ships_and_is_the_spec_repos():
    schema = json.loads(SCHEMA_PATH.read_text())
    assert schema["title"].startswith("Substrate specification")
    assert set(schema["required"]) == {"substrate_spec_version", "id", "domain", "frame", "pools", "walls", "seeding", "validity"}


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_examples_validate_and_round_trip(path):
    spec = load_spec(path)
    again = SubstrateSpec.from_json(spec.to_json())
    assert again == spec
    assert SubstrateSpec.from_dict(json.loads(path.read_text())) == again   # the file and the object agree
    assert spec.pool(0).name in ("extra", "free") and spec.pool("intra").id == 1


def test_the_examples_say_what_the_readme_leaves_unsaid():
    cyl = load_spec(FIX / "isolated_cylinder.sub.json")
    assert cyl.domain.boundary == ["open"] * 3 and cyl.wall("membrane").outside_pool is None
    assert cyl.seeding.pools == [1] and cyl.field_source_pools == []
    pm = load_spec(FIX / "packed_myelinated_cylinders.sub.json")
    assert pm.domain.boundary == ["periodic", "periodic", "open"]
    assert [p.name for p in pm.field_source_pools] == ["myelin"] and pm.pool("myelin").susceptibility.director == "radial"
    assert pm.wall("axolemma").permeability.in_to_out > 0 and pm.realisation["packing_fraction"] < pm.request["packing_fraction"]
    assert len(pm.wall("sheath").surface.instances["radii"]) == 3
    mesh = load_spec(FIX / "mesh_bundle.sub.json")
    assert {w.surface.kind for w in mesh.walls} == {"mesh"} and mesh.seeding.weights == "thin"


def _base():
    return json.loads((FIX / "packed_myelinated_cylinders.sub.json").read_text())


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d["pools"].pop(1), "dense 0"),
    (lambda d: d["pools"][0].__setitem__("name", "lumen"), "free / extra-cellular"),
    (lambda d: d["walls"][0].__setitem__("outside_pool", 7), "not a pool id"),
    (lambda d: d["walls"][0].__setitem__("inside_pool", 2), "both 2"),
    (lambda d: d["domain"].__setitem__("box_max", [-3e-6, 2e-6, 1e-6]), "below box_max"),
    (lambda d: d["domain"].__setitem__("boundary", ["periodic", "mirror", "open"]), "boundary must be"),
    (lambda d: d["seeding"].__setitem__("pools", [5]), "seeding.pools"),
    (lambda d: d["validity"].__setitem__("smallest_feature", 0.0), "smallest_feature"),
    (lambda d: d["pools"][2].__setitem__("susceptibility", None), "no pool is a field source"),
    (lambda d: d["pools"][2].__setitem__("water_fraction", 1.5), "water_fraction"),
    (lambda d: d["walls"][0]["surface"].__setitem__("kind", "torus"), "surface.kind"),
    (lambda d: d.__setitem__("bogus", 1), "unknown keys"),
])
def test_each_invariant_is_refused_by_name(mutate, match):
    d = _base()
    mutate(d)
    with pytest.raises(SpecError, match=match):
        SubstrateSpec.from_dict(d).validate() if "bogus" in d else validate(d)


def test_void_outside_a_permeable_wall_is_refused():
    d = _base()
    d["walls"][1]["outside_pool"] = None
    d["walls"][1]["permeability"] = {"in_to_out": 1e-5, "out_to_in": 0.0}
    with pytest.raises(SpecError, match="void"):
        validate(d)


def test_programmatic_construction_validates_too():
    spec = SubstrateSpec(
        id="t/sphere", domain={"box_min": [-1e-5] * 3, "box_max": [1e-5] * 3, "boundary": ["open"] * 3},
        pools=[Pool(0, "extra", 0.0, water_fraction=0.0), Pool(1, "intra", 2e-9, T2=0.05)],
        walls=[Wall("membrane", Surface("sphere", center=[0, 0, 0], radius=3e-6), inside_pool=1, outside_pool=None)],
        seeding={"pools": [1]}, validity={"smallest_feature": 3e-6, "tiers": ["gradient", "surface"]})
    spec = SubstrateSpec.from_dict(spec.to_dict()).validate()
    assert spec.wall("membrane").surface.radius == 3e-6 and spec.frame.axis == [0.0, 0.0, 1.0]


def test_nominal_field_is_optional_and_positive():
    import dataclasses
    from dmipy_sim.spec import load_spec, SpecError
    spec = load_spec(FIX / "isolated_cylinder.sub.json")
    assert spec.nominal_field_T is None
    assert dataclasses.replace(spec, nominal_field_T=3.0).validate().to_dict()["nominal_field_T"] == 3.0
    with pytest.raises(SpecError, match="nominal_field_T"):
        dataclasses.replace(spec, nominal_field_T=-1.0).validate()
