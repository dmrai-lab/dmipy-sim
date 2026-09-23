"""A replayed sequence's crusher is applied, and what it does is checked against the pathway enumeration.

A micron-scale cell cannot produce a spoiler geometrically: a gradient cannot wind much beyond 2 pi across
it. The crusher is therefore modelled, as the forward engine models it, by giving each walker a macroscopic
coordinate in [0, 1) that winds ``n_cycles`` turns over each window. Without it every coherence pathway of a
pulse train stays degenerate and recombines, so the echoes barely depend on the refocusing flip angle -- which
is not what a train does, and was the defect (dmipy-sim#305).
"""
import dataclasses

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.acquisition import epg
from dmipy_sim.acquisition.rf import RFEvent, RFSchedule
from dmipy_sim.acquisition.scanner_sequence import ScannerSequence
from dmipy_sim.replay.bank import build_replay_pack

DT, N_T = 5e-5, 401


@pytest.fixture(scope="module")
def pack():
    """Free water on a fine save grid: nothing here is about the substrate."""
    walk = d.simulate_trajectories(2000, 2e-9, d.FreeDiffusion(), 0.02, DT, seed=0, require_gpu=False)
    return build_replay_pack(walk, id="test/free", license="x", citation="x", K=8)


def _spin_echo(crusher=None):
    rf = RFSchedule((RFEvent(0.0, 90.0, "Mz→Mxy"), RFEvent(5e-3, 180.0, "refocus", 90.0)))
    return ScannerSequence(G=np.zeros((1, N_T, 3), np.float32), dt=DT, rf=rf, readout=(200,), crusher=crusher)


def _train(beta, n_echoes=6, n_cycles=16.0, crushed=True):
    s = sequences.cpmg(n_echoes, 2e-3, beta_deg=beta, gradient_directions=[[1.0, 0, 0]], bvalues=[0.0],
                       n_t_per_echo=40)
    if not crushed:
        return s
    wins = [(float(e.t_s) - 0.5e-3, float(e.t_s) + 0.5e-3) for e in s.rf if e.label in ("refocus", "refocusing")]
    return dataclasses.replace(s, crusher={"windows_s": wins, "n_cycles": n_cycles})


def _epg_echoes(beta, n_echoes=6):
    p = epg.enumerate_pathways(epg.cpmg_schedule(n_echoes, beta), threshold=1e-8)
    return np.array([abs(sum(x.eta for x in p if x.readout_idx == k)) for k in range(n_echoes)])


def test_a_crusher_straddling_a_refocusing_pulse_is_rewound_by_it(pack):
    """The pulse conjugates what the first half wound, so the second half undoes it and the echo is whole.
    A window the pulse does not straddle is not undone and the ensemble dephases away."""
    bare = float(np.abs(np.asarray(pack.replay_bloch(_spin_echo()))).reshape(-1)[0])
    straddling = float(np.abs(np.asarray(pack.replay_bloch(
        _spin_echo({"windows_s": [(3e-3, 7e-3)], "n_cycles": 16.0})))).reshape(-1)[0])
    after = float(np.abs(np.asarray(pack.replay_bloch(
        _spin_echo({"windows_s": [(6e-3, 8e-3)], "n_cycles": 16.0})))).reshape(-1)[0])
    assert straddling == pytest.approx(bare, abs=1e-3)
    assert after < 0.1 * bare


@pytest.mark.parametrize("beta", [180.0, 150.0, 120.0, 90.0])
def test_a_crushed_train_reproduces_the_pathway_enumeration(pack, beta):
    """Two routes that share no code: an analytic enumeration of the coherence pathways, and a vector
    propagation through the actual pulses on a Monte-Carlo walk. With the spoiler in place they agree, and
    with the crusher coordinate stratified over the ensemble they agree to rounding: a whole number of turns
    cancels every crushed pathway exactly, whatever the seed deals the coordinates (dmipy-sim#393)."""
    S = np.abs(np.asarray(pack.replay_bloch(_train(beta)))).reshape(-1)[:6]
    assert np.max(np.abs(S - _epg_echoes(beta))) < 1e-9
    S7 = np.abs(np.asarray(pack.replay_bloch(_train(beta), crusher_seed=7))).reshape(-1)[:6]
    assert np.max(np.abs(S7 - _epg_echoes(beta))) < 1e-9


def test_without_the_crusher_a_train_does_not_depend_on_its_flip_angle(pack):
    """The regression this guards: unspoiled, the pathways stay degenerate and recombine, so a 90 degree
    train looks like a 180 degree one and the enumeration is missed by a wide margin."""
    flat = np.abs(np.asarray(pack.replay_bloch(_train(90.0, crushed=False)))).reshape(-1)[:6]
    assert np.max(np.abs(flat - _epg_echoes(180.0))) < 0.02        # it looks like a perfect train
    assert np.max(np.abs(flat - _epg_echoes(90.0))) > 0.25         # and is nowhere near its own


def test_the_macroscopic_coordinates_are_drawn_reproducibly(pack):
    """A replay of one pack and one sequence is repeatable, and the draw is what a different seed changes."""
    seq = _train(120.0)
    a = np.abs(np.asarray(pack.replay_bloch(seq, crusher_seed=0))).reshape(-1)[:6]
    b = np.abs(np.asarray(pack.replay_bloch(seq, crusher_seed=0))).reshape(-1)[:6]
    c = np.abs(np.asarray(pack.replay_bloch(seq, crusher_seed=7))).reshape(-1)[:6]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.max(np.abs(a - c)) < 0.05                            # the same physics, another draw of it
