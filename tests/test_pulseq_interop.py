"""Pulseq (.seq) interoperability: from_pulseq / to_pulseq round-trips.

dmipy-sim speaks the field's lingua franca: import any .seq and simulate it,
export ours back out.  The round-trip b-value is the consistency/safety check on
the bridge.
"""
import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("pypulseq")
import pypulseq as pp

from dmipy_sim.waveforms import trapezoidal_ogse, pgse, calc_b, Waveform
from dmipy_sim.sequences import (
    from_pulseq, to_pulseq, make_system, PULSEQ_SYSTEMS)
from dmipy_sim.constants import GAMMA
import jax.numpy as jnp

BVEC = np.array([[1., 0., 0.]], dtype=np.float32)


def _b(wf):
    return float(calc_b(wf)[0])


def test_scanner_catalogue_opts():
    """Named scanners resolve to a pypulseq Opts with dmipy-sim's gamma."""
    for name in PULSEQ_SYSTEMS:
        sysd = make_system(name)
        assert abs(sysd.gamma - GAMMA / (2 * np.pi)) < 1.0          # Hz/T
        assert sysd.max_grad > 0 and sysd.max_slew > 0


def test_roundtrip_slew_limited_preserves_bvalue():
    """A realizable (slew-limited) gradient survives Waveform -> .seq -> Waveform
    with the diffusion b-value preserved to <0.1%."""
    wf = trapezoidal_ogse(1, 0.01, 0.04, 0.05, BVEC, n_t=400, slew_rate=200.0)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "rt.seq")
        to_pulseq(wf, filename=p)
        wf2 = from_pulseq(p)
    assert abs(_b(wf2) - _b(wf)) <= 1e-3 * _b(wf)
    assert int(wf2.echo_idx) == int(wf.echo_idx)
    # RF schedule preserved via definitions
    flips = sorted(e['flip_deg'] for e in (wf2.rf_events or []))
    assert 90.0 in flips


def test_roundtrip_square_incurs_expected_ramp_cost():
    """A SQUARE (infinite-slew) pgse cannot be represented exactly by Pulseq
    (which must ramp the edges) -- the b-value shifts by a few percent.  This is
    correct physics (Pulseq describes realizable gradients), not a bridge bug, so
    we assert it stays bounded rather than exact."""
    wf = pgse(delta=0.01, DELTA=0.04, G_magnitude=0.05, bvecs=BVEC, n_t=200,
              slew_rate=np.inf)   # explicitly the idealized square waveform
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sq.seq")
        to_pulseq(wf, filename=p)
        wf2 = from_pulseq(p)
    rel = abs(_b(wf2) - _b(wf)) / _b(wf)
    assert rel < 0.05            # bounded edge-ramp cost, not exact


def test_from_pulseq_native_file():
    """Import a .seq built natively by pypulseq (no dmipy definitions): the
    gradient is rasterised and the 90/180 RF schedule is recovered from Pulseq's
    own event times."""
    sysd = make_system('siemens_prisma', grad_raster_time=1e-5)
    seq = pp.Sequence(system=sysd)
    delta, Delta, g_T = 8e-3, 24e-3, 0.03
    trap = pp.make_trapezoid('x', amplitude=g_T * sysd.gamma, flat_time=delta, system=sysd)
    seq.add_block(pp.make_block_pulse(np.pi / 2, duration=1e-3, system=sysd))
    seq.add_block(trap)
    seq.add_block(pp.make_delay(Delta - delta - pp.calc_duration(trap) + 1e-3))
    seq.add_block(pp.make_block_pulse(np.pi, duration=1e-3, system=sysd, use='refocusing'))
    seq.add_block(trap)
    seq.add_block(pp.make_adc(num_samples=1, duration=1e-5, system=sysd))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "native.seq")
        seq.write(p)
        wf = from_pulseq(p, dt=5e-5)

    G = np.asarray(wf.G)
    assert G.shape[0] == 1 and G.shape[2] == 3
    assert abs(float(np.abs(G).max()) - g_T) < 1e-3          # peak gradient recovered (T/m)
    flips = sorted(e['flip_deg'] for e in wf.rf_events)
    assert flips == [90.0, 180.0]                            # clean schedule, no duplicates
    t180 = [e['t_s'] for e in wf.rf_events if e['flip_deg'] == 180.0][0]
    assert t180 > 0                                          # refocusing after excitation
    assert 0 <= int(wf.echo_idx) <= G.shape[1] - 1

    # folding the physical gradient at the 180 yields a positive, finite diffusion b
    i180 = int(round(t180 / wf.dt))
    Gf = G.copy(); Gf[:, i180:, :] *= -1.0
    b_eff = _b(Waveform(G=jnp.asarray(Gf), dt=wf.dt, echo_idx=wf.echo_idx))
    assert b_eff > 0 and np.isfinite(b_eff)


def test_pulseq_timing_extracts_spin_echo_budget():
    """pulseq_timing reads the 90/180/ADC schedule from a .seq → the diffusion
    spin-echo timing budget (RF durations, TE, readout-pre-echo) that feeds
    dmipy_design.SequenceTiming."""
    from dmipy_sim.sequences.pulseq import pulseq_timing
    sysm = pp.Opts(max_grad=80, grad_unit='mT/m', max_slew=200, slew_unit='T/m/s',
                   rf_dead_time=1e-4, rf_ringdown_time=3e-5, adc_dead_time=1e-5,
                   grad_raster_time=1e-5)
    seq = pp.Sequence(system=sysm)
    seq.add_block(pp.make_block_pulse(flip_angle=np.pi / 2, duration=2e-3, system=sysm))
    seq.add_block(pp.make_delay(15e-3))                       # pre-180 gap
    seq.add_block(pp.make_block_pulse(flip_angle=np.pi, duration=3e-3, system=sysm))
    seq.add_block(pp.make_delay(5e-3))                        # post-180 gap
    seq.add_block(pp.make_adc(num_samples=64, duration=20e-3, system=sysm))
    tm = pulseq_timing(seq)
    assert abs(tm['t_excite'] - 2e-3) < 1e-6
    assert abs(tm['t_refocus'] - 3e-3) < 1e-6
    assert abs(tm['readout_duration'] - 20e-3) < 1e-6
    # 90 centre ≈1.1 ms, 180 centre ≈18.7 ms → TE ≈35.2 ms; echo lands inside the ADC
    assert abs(tm['TE'] - 0.0352) < 6e-4
    assert 0.0 < tm['t_readout_pre_echo'] < tm['readout_duration']
    assert abs(tm['t_readout_pre_echo'] - 0.0111) < 6e-4
