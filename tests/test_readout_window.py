"""A readout gradient is represented and accepted, and not simulated as encoding (dmipy-sim#374).

A timing budget's dead windows say where a builder may not place ENCODING. A file a scanner plays carries its
readout gradient in exactly the budget's readout window, and refusing it read the budget as "no gradient may
exist here". The two checks are now different checks: every played gradient must be off a finite pulse, and
the encoding must be out of the dead windows. Which part of the played gradient is the readout is DECLARED --
by the builder, or by the caller importing a file -- never inferred from the ADC, since a diffusion lobe still
ramping down when the ADC opens is not a readout.
"""
import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("pypulseq")
import pypulseq as pp

from dmipy_sim import sequences
from dmipy_sim.acquisition.timing import SequenceTiming
from dmipy_sim.sequences import from_pulseq, make_system, to_pulseq


def _spin_echo_with_readout(g_T=0.03, delta=8e-3, Delta=30e-3, g_ro=0.01, t_ro=2e-3):
    """A foreign spin echo, as a scanner would play it: 90, diffusion lobe, 180 midway between the lobes, lobe,
    then a prephaser and a readout on y with the ADC open across the readout's flat top and CENTRED on the echo.
    Returns ``(seq, t_readout)`` with the readout gradient's ``(t0, t1)`` -- prephaser included, since it is
    played to read and not to encode -- from the excitation's centre."""
    sysd = make_system('siemens_prisma', grad_raster_time=1e-5)
    seq = pp.Sequence(system=sysd)
    trap = pp.make_trapezoid('x', amplitude=g_T * sysd.gamma, flat_time=delta, system=sysd)
    ro = pp.make_trapezoid('y', amplitude=g_ro * sysd.gamma, flat_time=t_ro, system=sysd)
    pre = pp.make_trapezoid('y', area=-0.5 * ro.area, system=sysd)
    dur = pp.calc_duration(trap)
    t90c = 0.5e-3                                                 # block pulses of 1 ms; their centres are the RF times
    seq.add_block(pp.make_block_pulse(np.pi / 2, duration=1e-3, system=sysd))
    seq.add_block(pp.make_delay(2e-3))
    s1 = seq.duration()[0]
    seq.add_block(trap)
    t180c = s1 + dur / 2 + Delta / 2                              # midway between the lobes' centres
    seq.add_block(pp.make_delay(t180c - 0.5e-3 - seq.duration()[0]))
    seq.add_block(pp.make_block_pulse(np.pi, duration=1e-3, system=sysd, use='refocusing'))
    seq.add_block(pp.make_delay(s1 + Delta - seq.duration()[0]))
    seq.add_block(trap)
    t_ro0 = seq.duration()[0]                                     # the readout's gradient starts with its prephaser
    seq.add_block(pre)
    TE_abs = t90c + 2.0 * (t180c - t90c)
    seq.add_block(pp.make_delay(TE_abs - (seq.duration()[0] + ro.rise_time + t_ro / 2)))
    seq.add_block(ro, pp.make_adc(num_samples=64, duration=t_ro, delay=ro.rise_time, system=sysd))
    t_ro1 = seq.duration()[0]
    return seq, (t_ro0 - t90c, t_ro1 - t90c)


def _import(seq, **kw):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "foreign.seq")
        seq.write(path)
        return from_pulseq(path, dt=5e-5, **kw)


def test_a_file_with_a_readout_gradient_is_refused_undeclared_and_accepted_declared():
    """The bug, and the fix. The budget read from the file's blocks puts its readout window exactly where the
    file plays its readout gradient. Undeclared, that gradient is encoding as far as the budget knows and is
    refused -- naming what to declare; declared, it is the readout and the file validates."""
    seq, t_ro = _spin_echo_with_readout()
    wf = _import(seq)
    assert wf.timing is not None and wf.readout_window is None
    with pytest.raises(ValueError, match="readout_s"):
        wf.validate()
    ok = _import(seq, readout_s=t_ro)
    assert ok.readout_window == pytest.approx(t_ro)
    ok.validate()
    # what was declared is the readout: zero encoding inside the window, the played gradient untouched
    t = np.arange(ok.n_t) * ok.dt
    inside = (t >= t_ro[0]) & (t <= t_ro[1])
    assert np.abs(np.asarray(ok.designed_gradient)[:, inside, :]).max() == 0.0
    flat = (t >= t_ro[1] - 1.5e-3) & (t <= t_ro[1] - 0.5e-3)         # the readout lobe's flat top, 10 mT/m on y
    np.testing.assert_allclose(np.asarray(ok.readout_gradient)[0, flat, 1], 0.01, rtol=1e-2)
    np.testing.assert_array_equal(np.asarray(ok.G), np.asarray(ok.played_gradient))


def test_the_readout_is_balanced_at_the_echo_and_adds_almost_no_diffusion_weighting():
    """The echo of a foreign file is read at its ADC's centre, where a readout balanced by its prephaser leaves
    no net moment -- so nothing winds across the voxel there (dmipy-sim#375) -- and the readout's own b is
    parts in ten thousand of the diffusion encoding's."""
    seq, t_ro = _spin_echo_with_readout()
    ok = _import(seq, readout_s=t_ro)
    assert not ok.unbalanced
    b = float(ok.b()[0])
    without = ok.with_gradient(np.asarray(ok.G, np.float64) - np.asarray(ok.readout_gradient, np.float64))
    assert b > float(without.b()[0])                              # the readout does encode, a little
    assert abs(b - float(without.b()[0])) < 5e-3 * b              # and a little is what it is
    # the ADC's last sample, by contrast, sits at the edge of k-space: the moment there is a good part of the
    # readout's own, which is the imaging kernel and not the echo
    q = np.cumsum(np.asarray(ok.G_eff, np.float64) * ok.dt, axis=1)[0, :, 1]
    last = int(round((t_ro[1] - 0.6e-3) / ok.dt))                # inside the readout's flat top, near its end
    assert abs(q[last]) > 0.3 * np.abs(q).max()
    assert abs(q[ok.echo_idx - 1]) < 2 * 0.01 * ok.dt             # at the echo: within two samples of the readout


def test_a_gradient_on_a_finite_pulse_is_refused_whoever_placed_it():
    """The pulse check reads every played gradient. Declaring a window over a finite 180 does not make a
    gradient there acceptable: a readout on a refocusing pulse is wrong, and so is an encoding one -- and a
    magnet's own gradient there is neither, being the magnet's."""
    timing = SequenceTiming(t_excite=1e-3, t_refocus=2e-3, t_readout_pre_echo=1e-3)
    seq = sequences.pgse([[1.0, 0.0, 0.0]], 8e-3, 24e-3, gradient_strengths=[0.03], n_t=400, timing=timing)
    t180 = [e for e in seq.rf if e.flip_deg == 180.0][0]
    assert t180.duration_s > 0.0
    G = np.asarray(seq.G, np.float64).copy()
    t = np.arange(seq.n_t) * seq.dt
    on = (t >= t180.window[0]) & (t <= t180.window[1])
    G[:, on, 2] = 5e-3
    bad = seq.with_gradient(G).with_readout_window(t180.window[0], t180.window[1])
    with pytest.raises(ValueError, match="finite pulse"):
        bad.validate()
    seq.with_background_gradient([0.0, 0.0, 5e-3]).validate()   # the magnet's, through the pulse: accepted


def test_a_magnets_background_does_not_trip_the_dead_windows_on_float32_rounding():
    """``designed = G - imposed`` cancels inexactly in float32, and a zero-tolerance test refused legal
    sequences on rounding alone. Every combination of amplitude and background here validates."""
    timing = SequenceTiming(t_excite=1e-3, t_refocus=2e-3, t_readout_pre_echo=1e-3)
    for g in (0.01, 0.03, 0.08):
        seq = sequences.pgse([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], 8e-3, 24e-3, gradient_strengths=[g, g],
                             n_t=400, timing=timing)
        for bg in (1e-4, 1.4e-3, 5e-3):
            seq.with_background_gradient([bg, -0.3 * bg, 0.7 * bg]).validate()


def test_the_rotated_maxwell_term_is_booked_as_imposed_too():
    """On a machine whose B0 is not along z -- the Swoop's is ``(0, 1, 0)`` -- the concomitant term is
    evaluated in the magnet's frame and turned back. The increment must come back booked as the magnet's,
    or ``designed_gradient`` reads the Maxwell term as encoding and a budget refuses it."""
    seq = sequences.pgse([[1.0, 0.0, 0.0]], 8e-3, 24e-3, gradient_strengths=[0.03], n_t=400)
    g0 = np.array([1.4e-3, 0.0, 0.0])
    r = np.array([0.02, 0.0, 0.08])
    for axis in ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)):
        both = seq.with_background_gradient(g0).with_concomitant(r, 0.064, b0_axis=axis)
        assert both.imposed_gradient is not None
        np.testing.assert_allclose(np.asarray(both.designed_gradient, np.float64), np.asarray(seq.G, np.float64),
                                   rtol=1e-4, atol=1e-7)
        extra = np.asarray(both.G, np.float64) - np.asarray(seq.G, np.float64)
        np.testing.assert_allclose(np.asarray(both.imposed_gradient, np.float64), extra, rtol=1e-4, atol=1e-7)


def test_the_window_round_trips_through_our_own_file():
    """A file this package writes carries the window it was given, so its reader needs no declaration."""
    timing = SequenceTiming(t_excite=1e-3, t_refocus=2e-3, t_readout_pre_echo=1e-3)
    seq = sequences.pgse([[1.0, 0.0, 0.0]], 8e-3, 24e-3, gradient_strengths=[0.03], n_t=400, timing=timing)
    TE = seq.T
    ours = seq.with_readout_window(TE - 3e-3, TE)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ours.seq")
        to_pulseq(ours, filename=path)
        back = from_pulseq(path)
    assert back.readout_window == pytest.approx(ours.readout_window)
    back.validate()
