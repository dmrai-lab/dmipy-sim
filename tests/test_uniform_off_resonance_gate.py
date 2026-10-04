"""A uniform off-resonance accrues through the acquisition's effective gate, exactly in the pulse instants.

``ReplayPhantom.gate_integral`` is what a voxel's ``delta_B0`` dephases through (the ``replay`` layer, the analytic
forms of ``replay_bloch``, ``Phantom.partition``): ``RFSchedule.gate_integral``, the coherence sign times the
transverse gate integrated with every pulse at its ``t_s``. A stimulated echo's mixing time is stored along B0 and
accrues nothing; every refocusing pulse of a train flips the sign.
"""
import numpy as np
import pytest

from dmipy_sim import sequences
from dmipy_sim.acquisition.rf import RFEvent, RFSchedule
from dmipy_sim.replay.phantom import ReplayPhantom


@pytest.mark.parametrize("n_t", [1000, 1001, 333])
def test_a_stimulated_echo_dephases_through_its_two_transverse_periods_only(n_t):
    ste = sequences.pgste([[1, 0, 0]], 7.6e-3, 38.3e-3, bvalues=[1e9], TE=53.5e-3, n_t=n_t, slew_rate=np.inf)
    assert ReplayPhantom.gate_integral(ste) == pytest.approx(0.0, abs=1e-15)


def test_every_refocusing_pulse_of_a_train_flips_the_sign():
    cpmg = sequences.cpmg(4, 10e-3, n_t_per_echo=100)
    assert ReplayPhantom.gate_integral(cpmg) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("n_t", [1000, 1001, 333])
def test_a_spin_echo_and_a_gradient_echo_are_unchanged(n_t):
    """The values the spin-echo gate gave: zero for a 180 at TE/2 on any grid, TE for a gradient echo."""
    se = sequences.pgse([[1, 0, 0]], 7.6e-3, 45.9e-3, bvalues=[1e9], TE=53.5e-3, n_t=n_t, slew_rate=np.inf)
    gre = sequences.gre(53.5e-3, gradient_directions=[[1, 0, 0]], bvalues=[0.0], delta=1e-3, Delta=2e-3, n_t=n_t,
                        slew_rate=np.inf)
    assert ReplayPhantom.gate_integral(se) == 0.0
    assert ReplayPhantom.gate_integral(gre) == pytest.approx(gre.T, rel=1e-14)


def test_the_exact_integral_is_the_sampled_gates_where_the_pulses_lie_on_samples():
    sched = RFSchedule([RFEvent(0.0, 90, "Mz→Mxy"), RFEvent(0.010, 90, "store"), RFEvent(0.030, 90, "recall")])
    T, dt = 0.045, 1e-4
    n_t = int(round(T / dt)) + 1
    sampled = float(dt * (sched.sign(np.arange(n_t) * dt) * sched.transverse_gate(n_t, dt))[:-1].sum())
    assert sched.gate_integral(T) == pytest.approx(0.010 - 0.015, abs=1e-15)
    assert sampled == pytest.approx(sched.gate_integral(T), abs=1e-9)          # the sampled sign is float32
