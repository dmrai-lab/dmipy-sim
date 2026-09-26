"""The ``label_volume`` surface kind: what the spec must say, and the fixed point with the geometry."""
import json

import numpy as np
import pytest

from dmipy_sim.geometry import LabelVolume
from dmipy_sim.io.label_volume import write_nrrd
from dmipy_sim.spec import (SpecError, SubstrateSpec, as_geometry, geometry_from_spec, label_volume_spec,
                            load_spec, validate)
from dmipy_sim.spec.substrate import SURFACE_KINDS


@pytest.fixture
def image(tmp_path):
    """A 0.5 um segmentation of a 10 um slab pore between two grain slabs, on disk as a NRRD."""
    lab = np.ones((40, 8, 8), np.uint8)
    lab[10:30] = 0
    p = tmp_path / "slab.nrrd"
    write_nrrd(p, lab, 0.5e-6)
    return str(p), lab


def test_label_volume_is_a_surface_kind_and_the_schema_knows_it():
    assert "label_volume" in SURFACE_KINDS
    schema = json.loads((__import__("pathlib").Path(
        __import__("dmipy_sim.spec.substrate", fromlist=["SCHEMA_PATH"]).SCHEMA_PATH)).read_text())
    surf = schema["properties"]["walls"]["items"]["properties"]["surface"]["properties"]
    assert "label_volume" in surf["kind"]["enum"]
    for key in ("voxel_size", "origin", "labels", "crop"):
        assert key in surf, f"the schema does not describe surface.{key}"


def test_the_producer_writes_the_image_the_pools_and_the_wall(image):
    """``label_volume_spec`` cites the image as a file with its container, sha256, voxel size, origin
    and label map, gives the walking pool the water and the others none, and emits one wall per pair
    of pools that share a voxel face."""
    path, lab = image
    spec = label_volume_spec(path, rho=41e-6, D=2.07e-9, T2=3.1, id="test/slab")
    spec.validate()
    s = spec.walls[0].surface
    assert s.kind == "label_volume" and s.format == "nrrd" and s.file == path
    assert len(s.sha256) == 64
    assert np.allclose(s.voxel_size, 0.5e-6) and s.labels == {"0": "free", "1": "grain"}
    assert [(p.name, p.water_fraction) for p in spec.pools] == [("free", 1.0), ("grain", 0.0)]
    assert spec.seeding.pools == [0]
    assert len(spec.walls) == 1 and spec.walls[0].inside_pool == 1 and spec.walls[0].outside_pool == 0
    assert spec.walls[0].surface_relaxivity.inside == 41e-6 and spec.walls[0].surface_relaxivity.outside == 41e-6
    assert spec.domain.boundary == ["reflect"] * 3
    assert spec.validity.smallest_feature == pytest.approx(0.5e-6)
    assert spec.realisation["porosity"] == pytest.approx(0.5)
    assert spec.realisation["surface_to_volume"] == pytest.approx(2e5)


def test_the_producer_and_the_geometry_are_a_fixed_point(image, tmp_path):
    """spec -> geometry -> spec is the identity: what the producer wrote is what the geometry hands
    back, so a pack of this walk embeds the substrate it was walked on."""
    path, lab = image
    spec = label_volume_spec(path, rho=41e-6, D=2.07e-9, id="test/slab")
    g = geometry_from_spec(spec)
    assert isinstance(g, LabelVolume)
    assert np.array_equal(g.labels, lab) and np.allclose(g.voxel_size, 0.5e-6)
    assert g.pools == {0: "free", 1: "grain"} and g.pool == "free"
    assert g.surface_relaxivity_t2 == 41e-6 and g.permeability is None
    assert list(g.periodic) == [False] * 3
    assert g.spec.to_dict() == spec.to_dict()

    p = tmp_path / "slab.sub.json"
    spec.save(p)
    assert load_spec(p).to_dict() == spec.to_dict()
    assert isinstance(as_geometry(str(p)), LabelVolume)


def test_a_crop_and_a_periodic_axis_travel_in_the_spec(image):
    """The crop IS the substrate and a periodic axis is a wall that wraps, so both are fields of the
    spec, not of the call that built the geometry."""
    path, lab = image
    spec = label_volume_spec(path, crop=(8, 0, 0, 32, 8, 8), periodic=[False, True, True], D=2e-9)
    assert spec.walls[0].surface.crop == [8, 0, 0, 32, 8, 8]
    assert spec.domain.boundary == ["reflect", "periodic", "periodic"]
    assert np.allclose(spec.walls[0].surface.origin, [4e-6, 0.0, 0.0])
    g = geometry_from_spec(spec)
    assert g.labels.shape == (24, 8, 8) and list(g.periodic) == [False, True, True]
    assert np.allclose(g.origin, [4e-6, 0.0, 0.0])
    assert g.porosity() == pytest.approx(20 / 24)


def test_a_geometry_built_from_an_array_writes_its_image_and_is_walkable(tmp_path, monkeypatch):
    """A spec cites surfaces as FILES, so a ``LabelVolume`` built from an array in memory writes its
    grid into the surface cache under its content hash -- the rule a ``Mesh`` built from arrays follows
    -- and is then a substrate every driver accepts."""
    monkeypatch.setenv("DMIPY_SIM_SURFACE_DIR", str(tmp_path))
    lab = np.ones((12, 6, 6), np.uint8)
    lab[3:9] = 0
    g = LabelVolume(lab, 1e-6, surface_relaxivity_t2=1e-5)
    spec = g.spec
    spec.validate()
    assert spec.walls[0].surface.file.startswith(str(tmp_path))
    assert spec.id.startswith("label_volume/labels-")
    g2 = geometry_from_spec(spec)
    assert np.array_equal(g2.labels, lab) and g2.surface_relaxivity_t2 == 1e-5
    assert as_geometry(g) is g


def test_multiple_pools_become_multiple_walls(tmp_path):
    """A three-pool segmentation (a lumen, a wall and the space around it) emits one wall per pair of
    pools that actually touch, and a pair that shares no face is not a wall."""
    lab = np.zeros((12, 6, 6), np.uint8)
    lab[4:8] = 1
    lab[8:] = 2
    p = tmp_path / "three.nrrd"
    write_nrrd(p, lab, 1e-6)
    spec = label_volume_spec(p, pools={0: "free", 1: "intra", 2: "myelin"}, D=2e-9, rho=1e-5)
    assert {w.name for w in spec.walls} == {"intra|free", "myelin|intra"}      # 0 and 2 never touch
    assert [(w.inside_pool, w.outside_pool) for w in spec.walls] == [(1, 0), (2, 1)]
    g = geometry_from_spec(spec)
    assert g.pools == {0: "free", 1: "intra", 2: "myelin"} and g.pool == "free"
    assert set(g.interfaces()) == {(0, 1), (1, 2)}


@pytest.mark.parametrize("break_it, message", [
    (lambda s: s.pop("file"), "needs 'file'"),
    (lambda s: s.__setitem__("format", "zarr"), "needs 'format' in"),
    (lambda s: s.pop("voxel_size"), "needs 'voxel_size'"),
    (lambda s: s.__setitem__("voxel_size", [0.0, 1e-6, 1e-6]), "must be positive"),
    (lambda s: s.pop("labels"), "needs 'labels'"),
    (lambda s: s.__setitem__("labels", {"0": "free", "1": "nowhere"}), "does not declare"),
    (lambda s: s.__setitem__("labels", {"0": "free", "1": "free"}), "two label values to one pool"),
    (lambda s: s.__setitem__("crop", [1, 2, 3]), "six voxel indices"),
    (lambda s: s.__setitem__("crop", [5, 0, 0, 5, 8, 8]), "non-empty half-open box"),
])
def test_a_label_volume_surface_that_cannot_be_read_back_is_refused(image, break_it, message):
    """The validator refuses a ``label_volume`` surface that does not describe an image a walk can be
    rebuilt from, naming the field: the file, the container, the scale, the label map and the crop are
    each necessary, and a label mapped to a pool the spec does not declare is refused, never dropped."""
    path, _ = image
    d = label_volume_spec(path, D=2e-9).to_dict()
    break_it(d["walls"][0]["surface"])
    with pytest.raises(SpecError, match=message):
        validate(d)


def test_a_producer_refusal_names_what_is_wrong(image, tmp_path):
    path, _ = image
    with pytest.raises(SpecError, match="labels"):
        label_volume_spec(path, pools={0: "free"})
    with pytest.raises(SpecError, match="not one of the pools"):
        label_volume_spec(path, walk="nowhere")
    with pytest.raises(SpecError, match="named 'extra' or 'free'"):
        label_volume_spec(path, pools={0: "pore", 1: "grain"})
    solid = tmp_path / "solid.nrrd"
    write_nrrd(solid, np.zeros((6, 6, 6), np.uint8), 1e-6)
    with pytest.raises(SpecError, match="no wall"):
        label_volume_spec(solid, pools={0: "free", 1: "grain"})


def test_the_geometry_refuses_a_spec_whose_image_has_changed(image):
    """The surface cites a sha256, so an image that is not the one the spec was written against is
    refused rather than walked as if it were."""
    path, _ = image
    spec = label_volume_spec(path, D=2e-9)
    d = spec.to_dict()
    d["walls"][0]["surface"]["sha256"] = "0" * 64
    with pytest.raises(SpecError, match="does not match the sha256"):
        geometry_from_spec(SubstrateSpec.from_dict(d))
