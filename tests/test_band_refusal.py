"""A replay refuses a waveform whose gradient reaches beyond the pack's temporal band (dmipy-sim#277).

The pack resolves motion up to ``K / (2 T)`` Hz. A waveform projected on the bridge basis beyond that band would
contract against bands the pack never stored, and the signal would come back smooth, plausible and wrong. The
predicate is walk-free: the free bridge's band variances at the walk's diffusivity give the phase the dropped
bands carry, so a pack of walkers that never move (every band zero) judges a waveform by its band alone.
"""
import numpy as np
import pytest

from dmipy_sim import sequences
from dmipy_sim.replay.bank import build_replay_pack
from tests.test_bank import _lean_env

N_W, N_T, DT, D0 = 40, 201, 1e-4, 2e-9                       # a 20 ms grid


def _static_pack(K, seed=0):
    """Walkers that never move: the positions' bands are all zero, so only K and the grid say what the pack resolves."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(-2e-6, 2e-6, (N_W, 1, 3)).astype(np.float64)
    traj = np.repeat(r, N_T, axis=1)
    m = dict(traj=traj, dt_traj=DT, T_max=(N_T - 1) * DT, comp=np.zeros((N_W, N_T), np.int8), comp0=np.zeros(N_W, np.int64),
             w=np.ones(N_W), dlog_b=np.zeros((N_W, N_T)), D_intra=D0, n_walkers=N_W, seed=seed)
    return build_replay_pack(m, id=f"t/static-K{K}", method="bridge_dst", envelope=_lean_env(), K=K, license="x", citation="x")


def _ogse(f_hz, sigma, n_t=801):
    return sequences.ogse([[1.0, 0.0, 0.0]], f_hz, sigma, shape="cosine", Delta=sigma + 2e-3, bvalues=[1e9],
                          TE=2 * sigma + 4e-3, n_t=n_t, slew_rate=np.inf)


def test_an_ogse_above_the_band_is_refused_and_one_below_is_not():
    pk = _static_pack(K=20)                                                          # 500 Hz
    assert pk.temporal_bandwidth_hz == pytest.approx(500.0)
    low, high = _ogse(125.0, 8e-3), _ogse(750.0, 4e-3)                              # one period, and three
    hz_low, k_low, err_low = pk.waveform_band(low)
    hz_high, k_high, err_high = pk.waveform_band(high)
    assert k_low <= 20 < k_high and hz_low <= 500.0 < hz_high, (hz_low, k_low, hz_high, k_high)
    floor = pk.meta["fidelity"]["floor_max"]
    assert err_low <= floor < err_high and err_high > 1.0, (err_low, floor, err_high)   # above the band the signal is not merely off, it is unrelated
    assert np.isfinite(pk.replay(low)).all()
    with pytest.raises(ValueError, match="temporal band") as e:
        pk.replay(high)
    assert "500 Hz" in str(e.value) and f"{hz_high:.4g}" in str(e.value)               # both frequencies stated
    for route in (lambda p, s: p.walker_signals(s), lambda p, s: p.walker_phases(s), lambda p, s: p.replay_bloch(s),
                  lambda p, s: p.walker_primitives(s)):
        with pytest.raises(ValueError, match="temporal band"):
            route(pk, high)
    wide = _static_pack(K=80)                                                        # 2000 Hz: the same waveform replays
    assert wide.waveform_band(high)[1] <= 80 and np.isfinite(wide.replay(high)).all()


def test_the_battery_waveforms_of_a_walked_pack_are_within_its_band(pack):
    """The certified packs of the suite replay their own kind of waveform: a PGSE's trapezoid tail beyond the band
    is below the pack's floor, so nothing that replayed before is refused now."""
    seq = sequences.pgse([[1, 0, 0], [0, 0, 1]], 2e-3, 6e-3, bvalues=[1e9, 1e9], TE=9.5e-3, n_t=4 * pack.n_t + 1, slew_rate=np.inf)
    hz, bands, err = pack.waveform_band(seq)
    assert bands <= pack.K and hz <= pack.temporal_bandwidth_hz and 0.0 < err <= pack.meta["fidelity"]["floor_max"]
    assert np.isfinite(pack.replay(seq)).all()
