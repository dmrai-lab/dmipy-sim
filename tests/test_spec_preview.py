"""``spec.preview``: three orthogonal cross-sections of any substrate spec, drawn from the same membership test
the walk uses (dmipy-sim#480, the card's "what is inside").

What is asserted is the geometry, not the picture: the area fraction of each pool in a section is a number with
a closed form, and it is what tells a reader the preview is of the substrate the pack was walked on.
"""
import os

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import spec as S
from dmipy_sim.spec import SpecError


def _fracs(record, plane):
    return record["planes"][plane]["pool_area_fraction"]


def test_a_cylinder_in_section_is_its_own_circle_and_stripe():
    """An isolated object's spec is an open box of four radii, so the z-section is ``pi R^2 / (4R)^2`` and the
    x- and y-sections are a stripe of width ``2R``."""
    rec = S.preview(d.Cylinder(radius=5e-6, orientation=(0, 0, 1)).spec)
    assert rec["kinds"] == ["cylinder"] and rec["pools"] == ["extra", "intra"]
    assert _fracs(rec, 2)["intra"] == pytest.approx(np.pi / 64, rel=2e-2)     # the circle
    for plane in (0, 1):
        assert _fracs(rec, plane)["intra"] == pytest.approx(0.25, rel=1e-6)   # the stripe
    assert rec["scale_bar_m"] < max(rec["planes"][0]["extent_m"])


def test_a_sphere_in_section_is_the_same_circle_three_times():
    rec = S.preview(d.Sphere(radius=5e-6).spec)
    for plane in (0, 1, 2):
        assert _fracs(rec, plane)["intra"] == pytest.approx(np.pi / 64, rel=2e-2)


def test_a_packed_cell_in_section_is_its_packing_fraction():
    """A packed myelinated cell is periodic in plane, so the z-section's cylinder area IS the outer packing
    fraction -- which is what makes the picture a check on the realisation and not an illustration."""
    from dmipy_sim.substrate.substrate import Substrate
    spec = Substrate.canonical().request(n_fibres=12, packing_fraction=0.3, seed=0)
    rec = S.preview(spec)
    f = _fracs(rec, 2)
    assert f["intra"] + f["myelin"] == pytest.approx(0.3, abs=0.02)
    assert f["extra"] + f["intra"] + f["myelin"] == pytest.approx(1.0, abs=1e-6)


def test_a_spec_with_no_wall_has_no_cross_section():
    from dataclasses import replace
    spec = replace(d.Sphere(radius=5e-6).spec, walls=[])
    with pytest.raises(SpecError, match="no wall, so it has no cross-section"):
        S.preview(spec)


def test_a_png_is_written_and_the_record_names_it(tmp_path):
    path = tmp_path / "shots" / "sphere.png"
    rec = S.preview(d.Sphere(radius=5e-6).spec, str(path), title="a sphere")
    assert os.path.exists(path) and os.path.getsize(path) > 1000
    assert rec["path"] == "sphere.png"
    assert [p["normal"] for p in rec["planes"]] == ["x", "y", "z"]
    assert all(set(p) == {"normal", "at_m", "extent_m", "pool_area_fraction"} for p in rec["planes"])


def test_a_label_volume_is_its_own_grid(tmp_path):
    """A label grid already IS the wall, so the section is a slice of it: a checkerboard of 8-voxel blocks is
    half pore in every plane, at the voxel size the header states."""
    from dmipy_sim.io.label_volume import write_nrrd
    from dmipy_sim.spec import label_volume_spec
    n, h = 16, 1e-5
    i, j, k = np.indices((n, n, n))
    labels = ((i // 2 + j // 2 + k // 2) % 2).astype(np.uint8)
    path = tmp_path / "check.nhdr"
    write_nrrd(str(path), labels, voxel_size=h)
    spec = label_volume_spec(str(path), pools={0: "free", 1: "grain"}, D=2e-9, rho=0.0,
                             T2_pools={"free": 1.0, "grain": 1.0}, id="test/check")
    rec = S.preview(spec)
    assert rec["kinds"] == ["label_volume"] and rec["pixel_m"] == pytest.approx(h)
    for plane in (0, 1, 2):
        assert _fracs(rec, plane)["free"] == pytest.approx(0.5, abs=1e-9)
