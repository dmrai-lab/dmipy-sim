"""The save grid: the waveform is resampled onto it conserving its integral (so an edge between samples carries the
right b), and `walk_spec` derives `dt_save` from the scanner class instead of taking a number (#143)."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.acquisition.scanners import SCANNERS, save_interval, scanner_limits, FIELD_DT_CAP
from dmipy_sim.constants import GAMMA
from dmipy_sim.replay._replay_kernel import resample_gradient, resample_gradient_jax
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.substrate import Substrate
from dmipy_sim.spec import walk_spec

D0 = 2e-9
ENV = dict(bvals=[0.0, 1e9], dirs=[[0, 0, 1]], ogse_periods=[2], shortd_b=1e9, shortd_deltas_frac=[0.05], B0_list=[],
           theta_deg=[0], delta_frac=0.2, Delta_frac=0.5, rho_list=[1e-5])


class _Acq:
    def __init__(self, G, dt):
        self.G, self.dt = G, dt


def _pgse(dt, n_t, delta, Delta, b, g=(0, 0, 1)):
    nd, ng = int(round(delta / dt)), int(round(Delta / dt))
    amp = np.sqrt(b / ((GAMMA * nd * dt) ** 2 * ((ng - nd / 3) * dt)))
    G = np.zeros((1, n_t, 3)); g = np.asarray(g, float)
    G[0, :nd] = amp * g; G[0, ng:ng + nd] = -amp * g
    return _Acq(G, dt)


def test_resampling_conserves_the_gradient_integral_on_every_walk_sample():
    rng = np.random.default_rng(0)
    G = rng.normal(size=(2, 37, 3)); dt_wf = 1e-5
    for dt_tr, n_tr in ((3e-5, 20), (7e-6, 60), (1e-5 * 37 / 11, 11)):
        R = resample_gradient(G, dt_wf, dt_tr, n_tr)
        np.testing.assert_allclose(R.sum(1) * dt_tr, G.sum(1) * dt_wf, rtol=1e-12, atol=1e-12)     # total q exact
        Rj = np.asarray(resample_gradient_jax(G.astype(np.float32), dt_wf, dt_tr, n_tr))
        np.testing.assert_allclose(Rj, R, rtol=1e-4, atol=1e-5)
    # a finer walk grid keeps a lobe's value inside it and zero outside the waveform
    G = np.zeros((1, 10, 3)); G[0, 2:5, 0] = 0.05
    F = resample_gradient(G, 1e-4, 2.5e-5, 60)
    assert F[0, 8:20, 0] == pytest.approx(0.05) and np.all(F[0, 40:, 0] == 0) and F.sum() * 2.5e-5 == pytest.approx(G.sum() * 1e-4)


def test_an_off_grid_pgse_carries_its_b_to_the_replay():
    """Before: -3.7 % at dt 200 us for edges between walk samples. Now the same acquisition described on a 4x finer
    grid with shifted edges replays within the codec's own precision of the on-grid one."""
    walk = d.simulate_trajectories(3000, D0, d.FreeDiffusion(), 0.05, 2e-4, seed=0, require_gpu=False)
    pk = build_replay_pack(walk, id="t/grid", license="x", citation="x", K=32, envelope=ENV)
    on = pk.replay(_pgse(pk.dt, pk.n_t, 0.010, 0.030, 1e9), tissue=False)[0]
    off = pk.replay(_pgse(pk.dt / 4, 4 * pk.n_t, 0.010 + 0.6 * pk.dt, 0.030, 1e9), tissue=False)[0]
    assert abs(off / on - 1) < 3e-3, f"off-grid bias {off / on - 1:+.3%}"


def test_the_save_interval_rule_scales_as_derived():
    dt = {k: save_interval(0.05, 2e5, k) for k in SCANNERS}
    assert dt["prisma"] > dt["magnus"] > dt["connectom"] > dt["bruker_bga_s"] > dt["micro_insert"] > dt["extreme_insert"]
    # dt ~ 1 / Gmax, ~ N^-1/4, ~ T^-1/2 (up to the rounding to a whole number of saves)
    assert dt["prisma"] / dt["connectom"] == pytest.approx(0.30 / 0.08, rel=0.02)
    assert save_interval(0.05, 2e5, "prisma") / save_interval(0.05, 16 * 2e5, "prisma") == pytest.approx(2.0, rel=0.02)
    assert save_interval(0.05, 2e5, (0.08, 200.0)) == save_interval(0.05, 2e5, "prisma")
    assert save_interval(0.05, 2e5, "prisma", field=True) <= FIELD_DT_CAP
    assert 3e-5 < save_interval(0.05, 2e5, "connectom") < 5e-5                                  # ~33 us at 200k walkers
    with pytest.raises(ValueError, match="unknown scanner"):
        scanner_limits("siemens")
    # the phase-error variance the rule holds to a tenth of the floor, checked against the closed form
    n, T, G = 2e5, 0.05, 0.3
    dt_c = save_interval(T, n, "connectom")
    var = (2.0 / 3.0) * GAMMA ** 2 * D0 * dt_c ** 2 * G ** 2 * T
    assert var / 2 <= 0.1 / np.sqrt(n) * 1.0001


def test_walk_spec_derives_the_save_grid():
    spec = Substrate.canonical(field_T=3.0).request(n_fibres=3, seed=1)
    w = walk_spec(spec, 60, 2e-3, seed=0, require_gpu=False, field=False)
    D = max(p.D for p in spec.pools if p.D)
    assert w.dt == pytest.approx(save_interval(2e-3, 60, "connectom", D=D, field=False))
    w2 = walk_spec(spec, 60, 2e-3, seed=0, require_gpu=False, scanner="prisma", field=False)
    assert w2.dt > w.dt
    w3 = walk_spec(spec, 60, 2e-3, dt_save=2.5e-4, seed=0, require_gpu=False, field=False)     # an explicit grid still wins
    assert w3.dt == pytest.approx(2.5e-4)
