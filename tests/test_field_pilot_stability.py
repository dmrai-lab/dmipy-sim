"""The raster pilot's simultaneous certificate under fresh walker resampling."""
import numpy as np
import pytest
from scipy.fft import idct
from dmipy_analytic.constants import GAMMA
from dmipy_replay.replay.field import contract
from dmipy_sim.replay.bank import (derive_field_band, susc_path_encode_series,
                                   susc_path_coeffs, susc_path_field)
from diagnostics.field_pilot_stability import capture


@pytest.fixture(scope='module')
def raster_pilot():
    """The diagnosed 1024-walker CPU slab pilot, before band selection."""
    return capture()


@pytest.fixture(scope='module')
def certified_resamples(raster_pilot):
    """Twenty independent bootstrap pilots with explicitly fresh seeds."""
    data = raster_pilot
    records = []
    for seed in range(93500, 93520):
        idx = np.random.default_rng(seed).integers(0, len(data['series']), 1024)
        k, record = derive_field_band(data['series'][idx], data['names'], data['h'],
                                      data['env'], **data['kwargs'])
        assert k == record['K'] == 126
        assert record['confidence_level'] == .99
        assert record['within_floor'] and record['upper_bound'] <= 1
        assert record['upper_bound'] >= record['worst_ratio']
        assert all(r['upper_bound'] > 1 for r in record['ladder'][:-1])
        records.append(record)
    return records


def test_twenty_raster_resamples_choose_the_same_certified_band(certified_resamples):
    assert len(certified_resamples) == 20
    assert all('Bonferroni' in r['confidence_method'] for r in certified_resamples)


def reference_ratio(data, record):
    """Independent reference replay of the pilot's band and codec over its exact battery."""
    series, h = data['series'], data['h']
    n, nt, _ = series.shape
    k = record['K']
    arrays, meta = susc_path_encode_series(series, data['names'], K=k, container=record['container'],
                                          layout='wtc', dtype=np.float64, device='numpy', T=nt*h)
    coeffs, _ = susc_path_coeffs(arrays, meta)
    times = record['domain']['echo_times_s']
    gre = np.rint(np.array(times['gradient_echo']) / h).astype(int)
    se = np.rint(np.array(times['spin_echo']) / h).astype(int)
    t = np.arange(nt) / (nt-1)
    train = np.ones(nt)
    depth = k//2 if record['depth'] is None else min(k//2, record['depth'])
    for pulse in range(depth):
        train[t >= (pulse+.5)/depth] *= -1
    gates = [train, np.where(t < .25, 1., np.where(t < .75, 0., -1.))]
    ratios = []
    for theta in (0., 90.):
        d = [np.sin(np.deg2rad(theta)), 0., np.cos(np.deg2rad(theta))]
        raw = contract(series, d, B0=7., chi_iso=1.06e-6, chi_aniso=0.)
        modes = susc_path_field(coeffs / np.sqrt(h), d, B0=7., chi_iso=1.06e-6,
                               chi_aniso=0., has_aniso=False)
        decoded = idct(np.pad(modes, ((0, 0), (0, nt-k))), norm='ortho', axis=1)
        def signals(f):
            phase = GAMMA * h * np.cumsum(f, axis=1)
            echo = np.concatenate([phase[:, gre-1], 2*phase[:, se//2-1]-phase[:, se-1]], axis=1)
            train = np.stack([np.cos(GAMMA*h*(f*g).sum(axis=1)) for g in gates], axis=1)
            return np.concatenate([np.exp(1j*echo), train], axis=1)
        zr, zd = signals(raw), signals(decoded)
        error = np.abs((zr-zd).mean(axis=0))
        # The reference floor is scaled to the same 20,000-walker production count.
        floor = 2*np.sqrt(np.mean(np.abs(zr-zr.mean(axis=0))**2, axis=0)/20000)
        ratios.append(float(np.max(error / floor)))
    return max(ratios)


def test_raster_bounds_cover_an_independent_longer_reference(certified_resamples):
    # Eight times the aggregate path duration, independent seed, same certified time grid.
    data = capture(seed=93643, pilot_walkers=8192)
    for record in certified_resamples:
        assert reference_ratio(data, record) <= record['upper_bound']


def test_constant_exact_signals_have_a_zero_bound():
    # A zero field has an exact ratio of zero even when its MC threshold is zero.
    data = np.zeros((32, 16, 6))
    from dmipy_replay.replay.field import ISO_CHANNEL_NAMES
    k, record = derive_field_band(data, ISO_CHANNEL_NAMES, .001, K_max=8, B0=7, chi=1e-6,
                                  env=dict(B0_list=[7], theta_deg=[0], max_refocus_pulses=1))
    assert k == 2 and record['upper_bound'] == 0


@pytest.mark.parametrize('bad', [np.nan, -1., 0.])
def test_invalid_confidence_scale_is_rejected(bad):
    from dmipy_replay.replay.field import ISO_CHANNEL_NAMES
    with pytest.raises(ValueError, match='floor_scale'):
        derive_field_band(np.zeros((4, 8, 6)), ISO_CHANNEL_NAMES, .001, K_max=4,
                          floor_scale=bad, B0=7, chi=1e-6)


def test_invalid_confidence_level_is_rejected():
    from dmipy_replay.replay.field import ISO_CHANNEL_NAMES
    with pytest.raises(ValueError, match='confidence_level'):
        derive_field_band(np.zeros((4, 8, 6)), ISO_CHANNEL_NAMES, .001, K_max=4,
                          confidence_level=1., B0=7, chi=1e-6)


def test_pool_certificates_share_the_family_error_probability(raster_pilot):
    from dmipy_sim.replay.bank import derive_field_band_per_pool
    data = raster_pilot
    k, record = derive_field_band_per_pool(
        data['series'], {'a': np.arange(512), 'b': np.arange(512, 1024)},
        data['names'], data['h'], data['env'], **data['kwargs'])
    assert record['overall_confidence_level'] == .99
    assert all(r['confidence_level'] == .995 for r in record['pools'].values())
    assert k == max(r['K'] for r in record['pools'].values())
    assert all(r['upper_bound'] <= 1 for r in record['pools'].values())
