"""Every walk driver takes the substrate as a spec: the walk from a spec is the walk from the geometry it describes,
to the bit, on the fused engine, the trajectory producer, the Bloch engine and the MT walk."""
import numpy as np
from dmipy_sim import RFEvent
import pytest

import dmipy_sim as d
from dmipy_sim.spec import as_geometry, SubstrateSpec

D0 = 2e-9


def _seq():
    from dmipy_sim import sequences as _seqmod
    return _seqmod.pgse([[1, 0, 0]] * 2, 2e-3, 6e-3, bvalues=[0, 1e9], slew_rate=np.inf)


def test_as_geometry_accepts_every_spelling(tmp_path):
    g = d.Cylinder(3e-6, (0, 0, 1))
    spec = g.spec
    assert as_geometry(g) is g and as_geometry(None) is None
    assert type(as_geometry(spec)) is d.Cylinder and as_geometry(spec.to_dict()).spec == spec
    path = spec.save(tmp_path / "c.sub.json")
    assert as_geometry(str(path)).spec == spec
    from dmipy_sim.spec import SpecError
    class Duck:                                   # an object with no spec spelling is not a substrate: refused
        pass
    with pytest.raises(SpecError, match="spec spelling"):
        as_geometry(Duck())


def test_fused_engine_and_producer_walk_a_spec_as_the_geometry():
    g = d.Cylinder(3e-6, (0, 0, 1)); spec = g.spec
    kw = dict(n_walkers=400, diffusivity=D0, waveform=_seq(), seed=0, require_gpu=False)
    np.testing.assert_array_equal(d.simulate(geometry=spec, **kw), d.simulate(geometry=g, **kw))
    w_s = d.simulate_trajectories(60, D0, spec, 1e-3, 2.5e-4, seed=1, require_gpu=False)
    w_g = d.simulate_trajectories(60, D0, g, 1e-3, 2.5e-4, seed=1, require_gpu=False)
    np.testing.assert_array_equal(w_s.positions, w_g.positions)
    assert w_s.spec == spec and w_s.geometry.spec == spec


def test_bloch_and_mt_drivers_take_a_spec():
    from dmipy_sim.engine.bloch import simulate_bloch
    from dmipy_sim.engine.mt_walk import simulate_mt_trajectories
    g = d.Cylinder(3e-6, (0, 0, 1)); spec = g.spec
    seq = _seq()
    rf = [RFEvent(0.0, 90.0, axis_deg=90.0), RFEvent(seq.dt * (seq.G.shape[1] // 2), 180.0, axis_deg=0.0)]
    kw = dict(T2=0.05, seed=0, require_gpu=False)
    np.testing.assert_array_equal(np.asarray(simulate_bloch(60, D0, seq, spec, rf, **kw)),
                                  np.asarray(simulate_bloch(60, D0, seq, g, rf, **kw)))
    w_s = simulate_mt_trajectories(40, D0, spec, 1e-3, 2.5e-4, 0.0, 0.0, seed=2, require_gpu=False)
    w_g = simulate_mt_trajectories(40, D0, g, 1e-3, 2.5e-4, 0.0, 0.0, seed=2, require_gpu=False)
    np.testing.assert_array_equal(w_s.positions, w_g.positions)
