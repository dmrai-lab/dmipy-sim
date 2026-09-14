"""The strand field along the walk on its own save grid: `walk_spec(field_sample_every=m)` reads the field at every
m-th save, the walk and the pack record that grid, and the replay gates the path channel on it.

The positions' save grid is set by the envelope's path-integration rule (3349 saves per 100 ms on DiSCo at the
Connectome 2.0 envelope); the path channel keeps 16 modes, and reading the field at every save cost twenty times the
diffusion walk. Measured on 20k DiSCo walkers, every 4th save biases a spin-echo field signal by 3e-4 against a
per-voxel floor target of 8e-3."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.io.strands import write_tck
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec import disco_spec, walk_spec


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("every")
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "t.tck"), str(tmp / "d.txt")
    write_tck(tck, cls_, coordinate_unit_m=25e-6); np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    return disco_spec(tck, dia, side_m=20e-6)


def _walk(spec, every):
    return walk_spec(spec, 60, 8e-4, 5e-5, seed=3, n_probe=20_000, require_gpu=False, field=True, adaptive_steps=True,
                     field_cutoff_max_m=25e-6, field_sample_every=every)


def test_the_field_is_read_on_its_own_grid_and_the_walk_is_unchanged(spec):
    """Every 4th save: the same walk (positions, contact), the field samples equal the every-save walk's at those
    saves, the walk records the grid; every save is the default."""
    w1, w4 = _walk(spec, 1), _walk(spec, 4)
    np.testing.assert_array_equal(w4.positions, w1.positions)
    np.testing.assert_array_equal(w4.boundary_local_time, w1.boundary_local_time)
    n_t = w1.positions.shape[1]
    assert w1.field_sample_every == 1 and w1.field_samples.shape == (w1.positions.shape[0], n_t, 13)
    assert w4.field_sample_every == 4 and w4.field_samples.shape == (w1.positions.shape[0], len(range(0, n_t, 4)), 13)
    np.testing.assert_allclose(w4.field_samples, w1.field_samples[:, ::4], rtol=1e-5, atol=1e-9)


def test_the_pack_records_the_grid_and_the_replay_gates_on_it(spec, tmp_path):
    """The path channel's meta carries its own n_t and dt; the certificate is measured on that grid; the replay's
    field phase equals the gated contraction on the coarse grid; a prefix keeps the grid."""
    w4 = _walk(spec, 4)
    pk = build_replay_pack(w4, id="t/every4", license="x", citation="x", K=8, susc_path_K=4, device="numpy",
                           out_path=str(tmp_path / "e4.rpk"))
    pm = pk.meta["compression"]["channels"]["susceptibility_path"]
    n_tf = len(range(0, pk.n_t, 4))
    assert pm["n_t"] == n_tf and abs(pm["dt"] - 4 * pk.dt) < 1e-15 and pm["n_ch"] == 12
    assert pk.meta["fidelity"]["susc_path_pulses_certified"] == 2 and np.isfinite(pk.meta["fidelity"]["err_susc_path"])
    seq = d.gre(4e-4, gradient_directions=[[1, 0, 0]], bvalues=[0.0], delta=1e-4, Delta=2e-4, n_t=pk.n_t, slew_rate=np.inf)
    w_, ew, E = pk.walker_signals(seq, tissue=False, B0=7.0, b0_dir=(1, 0, 0), chi_iso=-0.1e-6, chi_aniso=-0.1e-6)
    w0, ew0, E0 = pk.walker_signals(seq, tissue=False)
    phase = np.angle(E[:, 0] / E0[:, 0])                             # the field's phase per walker
    from scipy.fft import dct
    from dmipy_sim.replay._replay_kernel import field_gate
    from dmipy_sim.replay.bank import susc_path_decode
    from dmipy_sim.fields.hollow_cylinder import contract
    from dmipy_sim.constants import GAMMA
    b, names = susc_path_decode(pk.arrays, pm)                      # (n_w, 13, n_tf): the coarse-grid series
    B = contract(np.transpose(b, (0, 2, 1)), (1, 0, 0), B0=7.0, chi_iso=-0.1e-6, chi_aniso=-0.1e-6)
    g = field_gate(seq, n_tf, 4 * pk.dt)
    expect = GAMMA * 4 * pk.dt * (B * g[None, :]).sum(1)
    np.testing.assert_allclose(np.angle(np.exp(1j * (phase - expect))), 0.0, atol=1e-6)
    child = pk.prefix(4e-4, K=8)
    cm = child.meta["compression"]["channels"]["susceptibility_path"]
    assert abs(cm["dt"] - 4 * pk.dt) < 1e-15 and cm["n_t"] == len(range(0, child.n_t, 4))


def test_the_grid_knob_needs_the_adaptive_producer(spec):
    with pytest.raises(ValueError, match="adaptive"):
        walk_spec(spec, 30, 8e-4, 5e-5, seed=3, n_probe=20_000, require_gpu=False, field=True, field_sample_every=4)
