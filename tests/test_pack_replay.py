"""One consume path: `ReplayPack.replay(waveform, ...)` applies every tier the pack carries and the
request asks for, on any waveform, and refuses a tier the pack does not have. `load` / `save` round-trip.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.replay import ReplayPack, compile_scheme, replay_signal
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.compression import decode_occupancy, relaxation_logweight
from dmipy_sim.sequences import Sequence

D0 = 2e-9
ENV = dict(bvals=[0.0, 1e9, 3e9], dirs=[[1, 0, 0], [0, 0, 1]], ogse_periods=[2], shortd_b=1e9,
           shortd_deltas_frac=[0.05], B0_list=[], theta_deg=[0], delta_frac=0.2, Delta_frac=0.5, rho_list=[1e-5])
T2 = [0.08, 0.03]; T1 = [1.0, 1.2]                                  # per pool id, given at replay


@pytest.fixture(scope="module")
def packs():
    walk = d.simulate_trajectories(300, D0, d.Cylinder(2e-6, (0, 0, 1)), 0.01, 5e-4, seed=0, require_gpu=False)
    full = build_replay_pack(walk, id="test/full", K=8, envelope=ENV, license="x", citation="x")
    plain_walk = d.simulate_trajectories(300, D0, d.Cylinder(2e-6, (0, 0, 1)), 0.01, 5e-4, seed=0,
                                         require_gpu=False, tiers=())
    plain = build_replay_pack(plain_walk, id="test/plain", K=8, envelope=ENV, license="x", citation="x")
    return full, plain


def _wf(n_t, dt):
    return d.set_b(d.pgse(delta=2e-3, DELTA=6e-3, G_magnitude=0.1, bvecs=[[1, 0, 0], [0, 0, 1]], n_t=n_t,
                          slew_rate=np.inf), [1e9, 1e9])


def _W(pack, wf):
    """The compiled scheme of ``wf`` on the pack grid, by hand: what pack.replay must reproduce."""
    return compile_scheme(np.asarray(wf.G), wf.dt, pack.K, n_t=pack.n_t, dt_pack=pack.dt)


def test_gradient_replay_is_the_mode_space_contraction(packs):
    full, plain = packs
    wf = _wf(plain.n_t, plain.dt)
    np.testing.assert_allclose(plain.replay(wf), replay_signal(plain, _W(plain, wf)), rtol=1e-12)
    np.testing.assert_allclose(full.replay(wf), replay_signal(full, _W(full, wf)), rtol=1e-12)


def test_relaxation_applies_the_packs_per_pool_rates(packs):
    full, plain = packs
    wf = _wf(full.n_t, full.dt)
    ch = full.meta["compression"]["channels"]["compartment"]
    comp = decode_occupancy(full.arrays, ch)["comp"]
    logw = relaxation_logweight(comp, [0.08, 0.03], [1.0, 1.2], full.dt)
    W = _W(full, wf)
    from dmipy_sim.replay.compression import read_position_coeffs
    C = read_position_coeffs(full.arrays, dtype=np.float64)
    phi = C.reshape(C.shape[0], -1) @ W
    ref = np.abs((np.exp(logw)[:, None] * np.exp(1j * phi)).sum(0) / C.shape[0])
    np.testing.assert_allclose(full.replay(wf, T2=T2, T1=T1), ref, rtol=1e-12)
    assert full.replay(wf, T2=T2, T1=T1)[0] < full.replay(wf)[0]         # T2 costs signal, also at b = 0
    with pytest.raises(ValueError, match="no compartment channel"):
        plain.replay(wf, T2=T2)
    with pytest.raises(ValueError, match="every id"):
        full.replay(wf, T2=[0.08])
    np.testing.assert_allclose(full.replay(wf, T2={"extra": 0.08, "intra": 0.03}, T1=T1),
                               full.replay(wf, T2=T2, T1=T1))                # names resolve through the spec
    for key in ("per_comp", "mt", "T2", "rho"):
        assert key not in full.meta, "a pack carries channels, never a physical value"
    np.testing.assert_array_equal(full.replay(wf), full.replay(wf, tissue=False))   # a bare geometry declares no values
    with pytest.raises(ValueError, match="nominal"):
        full.replay(wf, tissue="all")


def test_surface_relaxivity_uses_the_recorded_diffusivity(packs):
    full, plain = packs
    wf = _wf(full.n_t, full.dt)
    W = _W(full, wf)
    rho = 1e-5
    ref = replay_signal(full, W, rho_over_D=rho / D0)
    np.testing.assert_allclose(full.replay(wf, rho=rho), ref, rtol=1e-12)
    np.testing.assert_allclose(full.replay(wf, rho=rho, D=D0), ref, rtol=1e-12)
    with pytest.raises(ValueError, match="no C2"):
        plain.replay(wf, rho=rho)
    with pytest.raises(ValueError, match="no field tier"):
        full.replay(wf, B0=3.0)


def test_any_waveform_grid_and_a_sequence_are_accepted(packs):
    full, _ = packs
    wf_fine = _wf(4 * full.n_t, full.dt / 4)                            # the same pulses, four times the samples
    fine = full.replay(wf_fine)
    np.testing.assert_allclose(fine, replay_signal(full, _W(full, wf_fine)), rtol=1e-12)   # resampled onto the pack grid
    same = full.replay(_wf(full.n_t, full.dt))
    np.testing.assert_allclose(fine, same, rtol=0.1)                    # two discretisations of one waveform
    seq = Sequence.from_pgse(bvalues=[1e9], gradient_directions=[[1, 0, 0]], delta=2e-3, Delta=6e-3, n_t=200)
    s = full.replay(seq)
    assert s.shape == (1,) and 0 < s[0] < 1
    per_pool = full.replay(_wf(full.n_t, full.dt), compartment=1)
    assert per_pool.shape == (2,)
    with pytest.raises(ValueError, match="matched no walkers"):
        full.replay(_wf(full.n_t, full.dt), compartment=2)


def test_save_and_load_round_trip(packs, tmp_path):
    full, _ = packs
    path = full.save(tmp_path / "full.rpk")
    back = ReplayPack.load(path)
    assert back.has_relaxation and back.has_surface and not back.has_field
    wf = _wf(full.n_t, full.dt)
    np.testing.assert_array_equal(back.replay(wf, rho=1e-5, T2=T2), full.replay(wf, rho=1e-5, T2=T2))
    assert back.diffusivity == pytest.approx(D0)
