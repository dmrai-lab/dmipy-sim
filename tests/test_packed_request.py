"""`packed_request` -- the request producer for packed spheres and packed cylinders (dmrai-lab/dmipy-sim#543).

What was asked and what came out are two records, the same request gives the same spec, an infeasible
fraction is refused by name before anything is placed, and a hand-drawn `PackedSpheres`/`PackedCylinders`
(radii the caller drew itself) still has a spec -- it simply carries no `request`."""
import time

import numpy as np
import pytest

from dmipy_sim.spec import packed_request, geometry_from_spec, RSA_LIMIT, DENSIFY_LIMIT
from dmipy_sim.geometry.densify import DENSIFY_START
from dmipy_replay.spec.substrate import SubstrateSpec
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


# ── random sequential addition below its limit is what it was ─────────────────────────────────────────────────────
def _digests(spec):
    import hashlib
    import json
    inst = spec.wall("objects").surface.instances
    geometry = (np.asarray(inst["radii"], np.float64).tobytes() + np.asarray(inst["centers"], np.float64).tobytes())
    return (hashlib.sha256(geometry).hexdigest(),
            hashlib.sha256(json.dumps(spec.to_dict(), sort_keys=True, default=str).encode()).hexdigest())


# the radii, centres and whole spec of three RSA requests, recorded on main before the densification path existed
RSA_RECORDED = [
    ("sphere", 20, 0.3, 0, "1ac9a4c952db5d575d5df0429f77a2ba337e514d67fd7eef9c36177e1a08e147",
     "edc8f599d3ed4dcae6efe54af26cc0286adf7239c51df1f509176ca0e3f7e4ff"),
    ("cylinder", 200, 0.55, 3, "9f4b152022b7e583e49fc7ff04d1f7ecddf06d40392ee79435df98c7d6884ea2",
     "71906378f1977888b8ebb0574830eba38e37c819020f1f4eda19ee8a6d1d4fc3"),
    ("sphere", 300, 0.38, 5, "0ff4e13d65a8cf0bef6145b8061199e0f8329415a91f412327268e82c173b370",
     "edb7bf857d73a7b4dd3613ec8ddea1cfb6d276743fdab2ecdde1d3c8c55b64ac"),
]


@pytest.mark.parametrize("kind,n,fraction,seed,geometry_sha,spec_sha", RSA_RECORDED)
def test_an_rsa_request_below_the_limit_gives_the_spec_it_gave_before(kind, n, fraction, seed, geometry_sha,
                                                                       spec_sha):
    assert _digests(packed_request(kind, n, LAW, fraction, seed=seed)) == (geometry_sha, spec_sha)


def test_rsa_beyond_its_limit_points_to_the_densification_path():
    with pytest.raises(ValueError, match="method='densify'"):
        packed_request("cylinder", 50, LAW, 0.7, seed=0)
    with pytest.raises(ValueError, match="method='densify'"):
        packed_request("cylinder", 50, LAW, (0.3, 0.5), seed=0)
    with pytest.raises(ValueError, match="method must be one of"):
        packed_request("cylinder", 50, LAW, 0.3, method="lubachevsky", seed=0)


# ── radii by quantile ─────────────────────────────────────────────────────────────────────────────────────────────
def _quantile_law(cv, r_mean=1e-6):
    return {"family": "gamma_quantile", "cv": cv, "r_mean": r_mean, "water_floor": 0.01}


def _radii(spec):
    return np.asarray(spec.wall("objects").surface.instances["radii"], np.float64)


@pytest.mark.parametrize("kind,dim", [("cylinder", 2), ("sphere", 3)])
def test_quantile_radii_are_the_truncated_gamma_laws_quantiles_of_one_uniform_per_object(kind, dim):
    """Object i's radius is r_mean Q(u_i), Q the unit-mean Gamma(CV) quantile truncated where the radii below hold 1 %
    of the water (r^dim-weighted). Computed here by scipy.stats and the water by quadrature, apart from the
    producer's incomplete-gamma identities."""
    from scipy import integrate, stats
    cv, n, seed = 0.45, 400, 6
    k = 1.0 / cv ** 2
    law = stats.gamma(a=k, scale=1.0 / k)
    floor = stats.gamma(a=k + dim, scale=1.0 / k).ppf(0.01)
    water_below = integrate.quad(lambda x: x ** dim * law.pdf(x), 0.0, floor)[0] / law.moment(dim)
    assert water_below == pytest.approx(0.01, rel=1e-6)
    u = np.random.default_rng((seed, 1)).random(n)
    p_floor = law.cdf(floor)
    expected = 1e-6 * law.ppf(p_floor + u * (1.0 - p_floor))
    spec = packed_request(kind, n, _quantile_law(cv), 0.2, seed=seed)
    assert np.allclose(_radii(spec), expected, rtol=1e-10, atol=0)
    assert spec.request["radius_law"] == {"family": "gamma_quantile", "cv": 0.45, "r_mean": 1e-6, "water_floor": 0.01}


@pytest.mark.parametrize("cv,cylinders,spheres", [(0.25, 0.60, 0.65), (0.45, 0.46, 0.58), (0.65, 0.42, 0.62)])
def test_the_water_floor_is_where_the_radii_below_hold_one_percent_of_the_water(cv, cylinders, spheres):
    """The floors of the 1 % water rule, in mean radii to two decimals: the smallest of 4,000 radii lies within that
    rounding of them (and a few 1e-4 above the floor itself)."""
    for kind, floor in (("cylinder", cylinders), ("sphere", spheres)):
        smallest = _radii(packed_request(kind, 4000, _quantile_law(cv), 0.1, seed=0)).min() / 1e-6
        assert floor - 0.005 <= smallest <= floor + 0.007, (kind, cv, smallest)


def test_quantile_radii_scale_with_r_mean_and_keep_their_ranks_across_cv():
    small = _radii(packed_request("cylinder", 300, _quantile_law(0.45, 1e-6), 0.2, seed=2))
    large = _radii(packed_request("cylinder", 300, _quantile_law(0.45, 3e-6), 0.2, seed=2))
    other_cv = _radii(packed_request("cylinder", 300, _quantile_law(0.25, 1e-6), 0.2, seed=2))
    mono = _radii(packed_request("cylinder", 300, _quantile_law(0.0, 1e-6), 0.2, seed=2))
    assert np.allclose(large, 3.0 * small, rtol=1e-12)
    assert np.array_equal(np.argsort(small), np.argsort(other_cv))
    assert np.all(mono == 1e-6)


# ── the densification path, as specs ──────────────────────────────────────────────────────────────────────────────
DENSE = dict(method="densify", seed=4, min_gap=1e-9)


@pytest.mark.parametrize("kind,top", [("cylinder", 0.8), ("sphere", 0.6)])
def test_a_densified_request_is_one_path_of_the_same_objects(kind, top):
    fractions = (0.1, 0.4, top)
    specs = packed_request(kind, 200, _quantile_law(0.45), fractions, **DENSE)
    assert isinstance(specs, tuple) and len(specs) == 3
    for spec, f in zip(specs, fractions):
        assert spec.request["method"] == "densify" and spec.request["packing_fraction"] == f
        assert spec.request["path"]["start_fraction"] == DENSIFY_START and spec.request["min_gap"] == 1e-9
        assert spec.realisation["packing_fraction"] == pytest.approx(f, rel=1e-12)
        assert spec.realisation["min_gap"] >= 1e-9
        assert np.array_equal(_radii(spec), _radii(specs[0]))           # object i keeps its radius on the path
        assert geometry_from_spec(spec).spec == spec
    assert len({s.id for s in specs}) == 3
    alone = packed_request(kind, 200, _quantile_law(0.45), 0.4, **DENSE)
    assert alone.to_dict() == specs[1].to_dict()                         # a snapshot is the path's, whoever asks


def test_a_densified_request_states_its_gap_and_its_reach_or_is_refused_by_name():
    law = _quantile_law(0.25)
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="needs min_gap"):
        packed_request("cylinder", 2000, law, 0.7, method="densify", seed=0)
    with pytest.raises(ValueError, match="DENSIFY_LIMIT"):
        packed_request("cylinder", 2000, law, DENSIFY_LIMIT["cylinder"] + 0.01, method="densify", seed=0,
                       min_gap=1e-9)
    with pytest.raises(ValueError, match="DENSIFY_LIMIT"):
        packed_request("sphere", 2000, law, (0.3, DENSIFY_LIMIT["sphere"] + 0.01), method="densify", seed=0,
                       min_gap=1e-9)
    with pytest.raises(ValueError, match="DENSIFY_START"):
        packed_request("sphere", 2000, law, 0.02, method="densify", seed=0, min_gap=1e-9)
    with pytest.raises(ValueError, match="one id per"):
        packed_request("sphere", 20, law, (0.1, 0.2), method="densify", seed=0, min_gap=1e-9, id="one")
    with pytest.raises(ValueError, match="repeats"):
        packed_request("sphere", 20, law, (0.1, 0.1), method="densify", seed=0, min_gap=1e-9)
    assert time.perf_counter() - t0 < 1.0
