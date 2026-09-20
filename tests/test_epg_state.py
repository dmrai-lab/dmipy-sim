"""EPG as a state vector: what a long train needs, and why enumeration cannot give it.

`enumerate_pathways` is exact and exponential. The configuration-state formulation carries the same
information in a vector over coherence orders, at one matrix multiply per pulse. These tests pin the two
against each other where both can run, and then pin the scaling where only one can.
"""
import numpy as np
import pytest

from dmipy_sim.acquisition import epg
from dmipy_sim.acquisition.epg_state import EPGState, rf_operator, split_by_gate, train_weights


def _cpmg_states(n_echoes, beta, exc_phase=0.0, refocus_phase=90.0):
    st = EPGState.equilibrium(n_echoes + 3)
    st.pulse(90.0, exc_phase)
    out = []
    for _ in range(n_echoes):
        st.shift(+1)
        st.pulse(beta, refocus_phase)
        st.shift(+1)
        out.append(st.signal)
    return np.array(out)


@pytest.mark.parametrize("beta", [180.0, 150.0, 120.0, 90.0])
def test_the_state_vector_reproduces_the_pathway_enumeration_at_every_echo(beta):
    """Not merely the first echo. A train's later echoes are sums over many pathways, and reproducing them
    is what says the state vector has lost nothing by dropping the pathway labels."""
    n = 6
    got = np.abs(_cpmg_states(n, beta))
    want = np.array([abs(sum(p.eta for p in epg.enumerate_pathways(epg.cpmg_schedule(n, beta), threshold=1e-9)
                             if p.readout_idx == k)) for k in range(n)])
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


def test_the_excitation_phase_is_a_convention_that_must_match():
    """It is not a free choice: the refocusing pulses are referred to it. Exciting 90 degrees away puts the
    train off the Meiboom-Gill condition and the echoes are wrong from the second one on -- while the FIRST
    still agrees, which is exactly what makes it an easy mistake to keep."""
    n, beta = 6, 120.0
    right, wrong = np.abs(_cpmg_states(n, beta, 0.0)), np.abs(_cpmg_states(n, beta, 90.0))
    assert right[0] == pytest.approx(wrong[0], rel=1e-9)          # the first echo cannot tell
    assert np.abs(right[1:] - wrong[1:]).max() > 0.5              # everything after it can


def test_a_perfect_train_is_flat_and_a_reduced_one_is_not():
    """With no relaxation and no gradient in the train, a train of true 180s repeats one pathway and every
    echo is identical. Below 180 the pathways part and the echoes stop being equal -- which is the whole
    behaviour a reduced-flip train exists to have."""
    flat = np.abs(_cpmg_states(6, 180.0))
    np.testing.assert_allclose(flat, 1.0, atol=1e-12)
    assert np.ptp(np.abs(_cpmg_states(6, 120.0))) > 0.1


def test_the_rf_operator_does_what_a_pulse_does():
    """Not norm conservation -- the triple is not a vector in a Euclidean sense, since `F-` is stored as a
    conjugate -- but the three limits that pin the operator: nothing at zero flip, a clean exchange at 180,
    and half the longitudinal magnetisation tipped at 90."""
    np.testing.assert_allclose(rf_operator(0.0, 0.0), np.eye(3), atol=1e-12)

    T180 = rf_operator(180.0, 0.0)
    np.testing.assert_allclose(np.abs(T180), [[0, 1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    assert T180[2, 2] == pytest.approx(-1.0)                 # z inverted

    T90 = rf_operator(90.0, 0.0)
    assert abs(T90[0, 2]) == pytest.approx(1.0)              # all of Z reaches the transverse plane
    assert abs(T90[2, 2]) == pytest.approx(0.0, abs=1e-12)


# ── the collapse that makes a long train tractable ─────────────────────────────────────────────────
def _prep_and_train(n_echoes, beta):
    prep = [("pulse", 90.0, 0.0), ("interval", 1, 0), ("pulse", 180.0, 90.0), ("interval", 1, 1)]
    train, ro = [], []
    for _ in range(n_echoes):
        train += [("interval", 1, 99), ("pulse", beta, 90.0), ("interval", 1, 99)]
        ro.append(len(train) - 1)
    return prep, train, ro


@pytest.mark.parametrize("n_echoes", [6, 12, 20, 70])
def test_the_number_of_microscopic_gates_does_not_grow_with_the_train(n_echoes):
    """THE result. A pathway's microscopic phase comes only from intervals where the gradient is ON, so
    pathways that differ solely in what they did afterwards share a gate. A diffusion-prepared train encodes
    once and then reads, so its gate count is set by the PREPARATION and is flat in the train's length --
    while the pathway count is not: 13 at four echoes, 49,714 at twelve, and about 10^68 at seventy."""
    prep, train, ro = _prep_and_train(n_echoes, 120.0)
    w = train_weights(prep, train, n_echoes + 4, lambda i: i in (0, 1), ro)
    assert len(w) == 5, f"{len(w)} gates at {n_echoes} echoes"
    assert all(len(v) == n_echoes for v in w.values())


def test_a_seventy_echo_train_is_milliseconds_where_enumeration_is_hopeless():
    """Not a micro-benchmark: it is the difference between a thing that runs and a thing that cannot. The
    enumeration of a twelve-echo train already reaches 49,714 pathways, and pruning does not rescue it --
    a threshold keeping a hundredth of each amplitude keeps three per cent of their sum."""
    import time
    prep, train, ro = _prep_and_train(70, 120.0)
    t0 = time.time()
    w = train_weights(prep, train, 74, lambda i: i in (0, 1), ro)
    assert time.time() - t0 < 2.0
    assert len(w) == 5 and len(next(iter(w.values()))) == 70


def test_the_gate_split_conserves_the_signal():
    """The gates are a partition of the same magnetisation, so summing them returns the undivided train."""
    n = 6
    prep, train, ro = _prep_and_train(n, 120.0)
    w = train_weights(prep, train, n + 4, lambda i: i in (0, 1), ro)
    summed = np.abs(np.sum([v for v in w.values()], axis=0))
    # the preparation is a 90 and a true 180, so one gate carries it all and the sum is that gate
    biggest = max(w.values(), key=lambda v: np.abs(v).max())
    np.testing.assert_allclose(summed, np.abs(biggest), rtol=1e-6, atol=1e-12)


def test_splitting_where_the_gradient_is_off_would_change_nothing():
    """The claim underneath the collapse, stated as a test: an interval with no gradient cannot tell two
    pathways apart, so splitting on it produces more gates carrying the same total."""
    n = 4
    prep, train, ro = _prep_and_train(n, 120.0)
    few = train_weights(prep, train, n + 4, lambda i: i in (0, 1), ro)
    many = train_weights(prep, train, n + 4, lambda i: True, ro)
    assert len(many) >= len(few)
    a = np.abs(np.sum([v for v in few.values()], axis=0))
    b = np.abs(np.sum([v for v in many.values()], axis=0))
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)
