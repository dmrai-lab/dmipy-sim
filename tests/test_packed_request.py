"""`packed_request` -- the request producer for packed spheres and packed cylinders (dmrai-lab/dmipy-sim#543).

What was asked and what came out are two records, the same request gives the same spec, an infeasible
fraction is refused by name before anything is placed, and a hand-drawn `PackedSpheres`/`PackedCylinders`
(radii the caller drew itself) still has a spec -- it simply carries no `request`."""
import time

import numpy as np
import pytest

from dmipy_sim.spec import packed_request, geometry_from_spec, SubstrateSpec, RSA_LIMIT
from dmipy_sim.geometry import PackedSpheres, PackedCylinders

LAW = {"family": "gamma", "shape": 4.0, "scale": 0.5e-6, "r_min": 0.2e-6}


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_request_records_request_and_realisation_separately(kind):
    spec = packed_request(kind, 20, LAW, 0.3, seed=0)
    assert isinstance(spec, SubstrateSpec)
    assert spec.request["n_objects"] == 20 and spec.request["packing_fraction"] == pytest.approx(0.3)
    assert spec.request["radius_law"] == {"family": "gamma", "shape": 4.0, "scale": 0.5e-6, "r_min": 0.2e-6}
    assert spec.request["seed"] == 0 and spec.request["min_gap"] is None
    r = spec.realisation
    assert r["n_objects"] == 20 and r["packing_fraction"] == pytest.approx(0.3, rel=1e-9)
    assert r["min_gap"] > 0 and r["smallest_feature"] > 0 and r["cell_side"] > 0
    assert len(spec.wall("objects").surface.instances["radii"]) == 20


def test_the_same_request_twice_gives_the_same_spec():
    a = packed_request("sphere", 20, LAW, 0.3, seed=7)
    b = packed_request("sphere", 20, LAW, 0.3, seed=7)
    assert a.to_dict() == b.to_dict()
    c = packed_request("sphere", 20, LAW, 0.3, seed=8)
    assert c.to_dict() != a.to_dict()


@pytest.mark.parametrize("kind,geom_cls", [("sphere", PackedSpheres), ("cylinder", PackedCylinders)])
def test_geometry_from_spec_round_trips_a_produced_spec(kind, geom_cls):
    spec = packed_request(kind, 15, LAW, 0.25, seed=2)
    g = geometry_from_spec(spec)
    assert isinstance(g, geom_cls)
    assert g.spec == spec       # the geometry hands back the spec it was built from, request/realisation included


@pytest.mark.parametrize("kind,bad_fraction", [("sphere", 0.5), ("cylinder", 0.8)])
def test_an_infeasible_fraction_is_refused_by_name_before_any_placement(kind, bad_fraction):
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="saturation"):
        packed_request(kind, 500, LAW, bad_fraction, seed=0)
    assert time.perf_counter() - t0 < 1.0
    with pytest.raises(ValueError, match=str(RSA_LIMIT[kind])):
        packed_request(kind, 500, LAW, bad_fraction, seed=0)


def test_a_min_gap_violation_and_a_placement_failure_are_both_refused_by_name():
    with pytest.raises(ValueError, match="narrowest gap"):
        packed_request("sphere", 20, LAW, 0.3, seed=0, min_gap=1e-5)
    with pytest.raises(ValueError, match="could not place"):
        packed_request("sphere", 200, LAW, 0.39, seed=0, max_attempts=10)


def test_an_unsupported_radius_law_family_is_refused():
    with pytest.raises(ValueError, match="radius_law family"):
        packed_request("sphere", 10, {"family": "lognormal", "shape": 1.0, "scale": 1.0}, 0.3)


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="kind"):
        packed_request("triangle", 10, LAW, 0.3)


def test_a_hand_drawn_pack_still_works_and_has_no_request():
    rng = np.random.default_rng(0)
    radii = np.maximum(rng.gamma(4.0, 0.5e-6, 10), 0.2e-6)
    from dmipy_sim.geometry.packing import pack_spheres
    centers, L, _ = pack_spheres(radii, target_vf=0.2, seed=0)
    hand = PackedSpheres(radii, centers, L)
    assert hand.spec.request is None and hand.spec.realisation is None
    hand.spec.validate()      # still a conformant spec
