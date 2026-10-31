"""Every walk driver takes the substrate as a spec: the walk from a spec is the walk from the geometry it describes,
to the bit, on the fused engine, the trajectory producer, the Bloch engine and the MT walk."""
from dataclasses import replace
import numpy as np
from dmipy_sim import RFEvent
import pytest

import dmipy_sim as d
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.spec import as_geometry, SubstrateSpec

D0 = 2e-9


def _seq():
    from dmipy_sim import sequences as _seqmod
    return _seqmod.pgse([[1, 0, 0]] * 2, 2e-3, 6e-3, bvalues=[0, 1e9], slew_rate=np.inf)


def test_a_door_takes_a_spec_or_a_geometry_and_refuses_every_other_spelling(tmp_path):
    from dmipy_sim.spec import geometry_from_spec, load_spec
    g = d.Cylinder(3e-6, (0, 0, 1))
    spec = g.spec
    assert as_geometry(g) is g and as_geometry(None) is None
    assert type(as_geometry(spec)) is d.Cylinder and geometry_from_spec(g) is g
    assert load_spec(spec.save(tmp_path / "c.sub.json")) == spec and SubstrateSpec.from_dict(spec.to_dict()) == spec
    path = tmp_path / "c.sub.json"

    class Carrier:                                # carries a spec, is not a Geometry
        pass
    carrier = Carrier(); carrier.spec = spec
    for door in (as_geometry, geometry_from_spec):
        with pytest.raises(TypeError, match="SubstrateSpec.from_dict"):
            door(spec.to_dict())
        for p in (str(path), path):
            with pytest.raises(TypeError, match="load_spec"):
                door(p)
    with pytest.raises(TypeError, match="not a substrate"):
        as_geometry(carrier)
    with pytest.raises(TypeError, match="neither a SubstrateSpec nor a Geometry"):
        as_geometry(object())
    for bad, match in ((spec.to_dict(), "SubstrateSpec.from_dict"), (str(path), "load_spec"), (carrier, "not a substrate")):
        with pytest.raises(TypeError, match=match):
            d.walk(bad, 4, 1e-3, 2.5e-4, require_gpu=False)
        with pytest.raises(TypeError, match=match):
            _walk_geometry(4, D0, bad, 1e-3, 2.5e-4, require_gpu=False)
        with pytest.raises(TypeError, match=match):
            d.simulate(geometry=bad, n_walkers=4, diffusivity=D0, waveform=_seq(), require_gpu=False)


def test_walk_takes_a_geometry_reduced_to_its_spec():
    g = d.Cylinder(3e-6, (0, 0, 1))
    w_g = d.walk(g, 40, 1e-3, 2.5e-4, seed=1, require_gpu=False, K=8, diffusivity=D0)
    w_s = d.walk(g.spec, 40, 1e-3, 2.5e-4, seed=1, require_gpu=False, K=8, diffusivity=D0)
    np.testing.assert_array_equal(w_g.position_coeffs, w_s.position_coeffs)
    assert w_g.spec == g.spec


def test_fused_engine_and_producer_walk_a_spec_as_the_geometry():
    g = d.Cylinder(3e-6, (0, 0, 1)); spec = g.spec
    kw = dict(n_walkers=400, diffusivity=D0, waveform=_seq(), seed=0, require_gpu=False)
    np.testing.assert_array_equal(d.simulate(geometry=spec, **kw), d.simulate(geometry=g, **kw))
    w_s = _walk_geometry(60, D0, spec, 1e-3, 2.5e-4, seed=1, require_gpu=False)
    w_g = _walk_geometry(60, D0, g, 1e-3, 2.5e-4, seed=1, require_gpu=False)
    np.testing.assert_array_equal(w_s.position_coeffs, w_g.position_coeffs)
    assert w_s.spec == spec and w_s.geometry.spec == spec


def test_bloch_and_mt_drivers_take_a_spec():
    from dmipy_sim.engine.bloch import _simulate_bloch
    from dmipy_sim.engine.mt_walk import simulate_mt_trajectories
    g = d.Cylinder(3e-6, (0, 0, 1)); spec = g.spec
    seq = _seq()
    rf = [RFEvent(0.0, 90.0, axis_deg=90.0), RFEvent(seq.dt * (seq.G.shape[1] // 2), 180.0, axis_deg=0.0)]
    kw = dict(T2=0.05, seed=0, require_gpu=False)
    np.testing.assert_array_equal(np.asarray(_simulate_bloch(60, D0, replace(seq, rf=rf), spec, **kw)),
                                  np.asarray(_simulate_bloch(60, D0, replace(seq, rf=rf), g, **kw)))
    w_s = simulate_mt_trajectories(40, D0, spec, 1e-3, 2.5e-4, 0.0, 0.0, seed=2, require_gpu=False)
    w_g = simulate_mt_trajectories(40, D0, g, 1e-3, 2.5e-4, 0.0, 0.0, seed=2, require_gpu=False)
    np.testing.assert_array_equal(w_s.position_coeffs, w_g.position_coeffs)
