"""`Substrate.request` returns the realised substrate as a spec: what was asked and what came out are two
records, an infeasible request is refused before anything is placed, and `pack` is the spec's geometry."""
import time

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.substrate import Substrate
from dmipy_sim.spec import geometry_from_spec, walk_spec, SubstrateSpec


def test_request_records_request_and_realisation_separately():
    sub = Substrate.canonical(field_T=3.0)
    spec = sub.request(n_fibres=12, seed=0)
    assert isinstance(spec, SubstrateSpec)
    assert spec.request["n_objects"] == 12 and spec.request["packing_fraction"] == pytest.approx(sub.f_axon)
    assert spec.request["diameter_law"]["d_min"] == sub.d_min and spec.request["g_ratio"] == sub.g_ratio
    r = spec.realisation
    assert r["n_objects"] == 12 and r["packing_fraction"] == pytest.approx(sub.f_axon, rel=1e-9)
    thinnest_sheath = 0.5 * (1 - sub.g_ratio) * sub.d_min                  # the floored fibre's sheath
    assert r["min_gap"] > 0 and r["smallest_feature"] == pytest.approx(thinnest_sheath, rel=1e-6)
    assert len(spec.wall("sheath").surface.instances["radii"]) == 12 and spec.domain.boundary == ["periodic", "periodic", "open"]
    assert spec.pool("myelin").T2 == sub.T2_myelin and spec.wall("axolemma").permeability.in_to_out == pytest.approx(sub.kappa, rel=1e-6)
    assert "outer radius" in " ".join(spec.provenance["transformations"])


def test_an_infeasible_request_is_refused_before_any_placement():
    sub = Substrate.canonical(field_T=3.0)
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="saturation"):
        sub.request(n_fibres=500, packing_fraction=0.7)
    assert time.perf_counter() - t0 < 1.0
    with pytest.raises(ValueError, match="narrowest gap"):
        sub.request(n_fibres=12, seed=0, min_gap=1e-6)
    with pytest.raises(ValueError, match="could not place"):
        sub.request(n_fibres=300, packing_fraction=0.58, max_attempts=20)


def test_pack_is_the_specs_geometry_and_walks_the_same():
    sub = Substrate.canonical(field_T=3.0)
    spec = sub.request(n_fibres=6, seed=3)
    g = geometry_from_spec(spec)
    p = sub.pack(n_axons=6, seed=3)
    assert type(p) is d.PackedMyelinatedCylinders and p.spec.walls == g.spec.walls and p.spec.pools == g.spec.pools
    w_spec = walk_spec(spec, 40, 4e-4, 2e-4, seed=1, require_gpu=False)
    w_pack = d.simulate_trajectories(40, sub.D_intra, p, 4e-4, 2e-4, seed=1, require_gpu=False)
    np.testing.assert_array_equal(w_spec.positions, w_pack.positions)
    assert w_spec.spec == spec and w_spec.diffusivity == pytest.approx(sub.D_intra, rel=1e-6)   # float32 pool D
