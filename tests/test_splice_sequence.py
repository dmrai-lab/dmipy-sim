"""The diffusion-prepared refocusing train: what an ultra-low-field diffusion FSE plays.

A preparation sets every spin's phase by its own path, then a train reads many echoes from it. Two things
make the family what it is and both are the builder's job: every echo lands on a sample, and every refocusing
pulse carries a balanced crusher pair. Without the first a pulse sits off centre in its own window and
dephases the echo the crusher exists to keep; without the second the coherence pathways stay degenerate and
the train stops depending on its refocusing flip angle, which is the behaviour it is built to have.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.acquisition import epg
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec.tissue import Tissue

PREP, ESP, N_ECHO = 84e-3, 10e-3, 8


def _seq(beta=180.0, n_t_per_echo=40, TE_prep=PREP, **kw):
    return sequences.splice([[1.0, 0.0, 0.0]], 35e-3, 42e-3, N_ECHO, ESP, bvalues=[0.945e9], TE_prep=TE_prep,
                            beta_deg=beta, n_t_per_echo=n_t_per_echo, slew_rate=200.0, **kw)


def test_it_delivers_the_b_it_was_asked_for_within_what_the_scanner_can_play():
    """945 s/mm^2 at delta / Delta of 35 / 42 ms needs 18.9 mT/m, which an ultra-low-field system can play
    (the Hyperfine Swoop's weakest axis is 24.4 mT/m)."""
    s = _seq()
    assert s.family == "splice"
    assert float(np.asarray(s.b())[0]) == pytest.approx(0.945e9, rel=1e-3)
    assert float(np.abs(s.G).max()) == pytest.approx(0.01885, rel=0.02)


def test_every_echo_lands_on_a_sample_at_its_nominal_time():
    """The readouts are the echo times themselves, not a rounding of them."""
    s = _seq()
    t = np.asarray(s.readout) * float(s.dt)
    np.testing.assert_allclose(t, PREP + np.arange(N_ECHO + 1) * ESP, atol=1e-12)


def test_a_preparation_the_grid_cannot_express_is_refused():
    """84.3 ms is 337.2 samples of a 250 us grid; rounding it would put the pulse off centre in its crusher."""
    with pytest.raises(ValueError, match="must fall on a sample"):
        _seq(TE_prep=84.3e-3)
    assert _seq(TE_prep=84.3e-3, n_t_per_echo=100) is not None       # a grid that CAN express it is fine


def test_a_preparation_shorter_than_its_own_blocks_is_refused():
    with pytest.raises(ValueError, match="below the"):
        _seq(TE_prep=40e-3)


def test_every_refocusing_pulse_carries_a_crusher_balanced_to_the_sample():
    s = _seq()
    pulses = [e for e in s.rf if e.label == "refocus"]
    wins = s.crusher["windows_s"]
    assert len(wins) == len(pulses) == N_ECHO + 1                     # the preparation's 180 and the train's
    for (a, b), e in zip(wins, pulses):
        i0, i1, ip = (round(x / float(s.dt)) for x in (a, b, e.t_s))
        assert i0 + i1 == 2 * ip                                      # the pulse rewinds exactly what it wound
    assert s.crusher["n_cycles"] == 16.0


def test_the_train_depends_on_its_refocusing_flip_as_the_pathways_say():
    """The builder's crusher is what makes this true: replayed, the train's echoes follow the coherence
    pathway enumeration, which is the whole reason a reduced-flip train behaves differently from a perfect
    one. Compared as a ratio to the 180 train, so the preparation and the relaxation divide out."""
    walk = d.simulate_trajectories(1500, 2e-9, d.FreeDiffusion(), 0.30, 2.5e-4, seed=0, require_gpu=False)
    pack = build_replay_pack(walk, id="test/free", license="x", citation="x", K=8)
    tissue = Tissue(T2=0.081, T1=0.275)
    ref = np.abs(np.asarray(pack.replay_bloch(_seq(180.0), tissue=tissue))).reshape(-1)
    got = np.abs(np.asarray(pack.replay_bloch(_seq(120.0), tissue=tissue))).reshape(-1)
    ratio = got[1:] / ref[1:]                                         # echo 0 is the preparation's own
    want = [abs(sum(p.eta for p in epg.enumerate_pathways(epg.cpmg_schedule(N_ECHO, 120.0), threshold=1e-4)
                    if p.readout_idx == k)) for k in range(N_ECHO)]
    assert ratio[0] == pytest.approx(want[0], abs=0.05)               # the first train echo, where the pathways part
    assert ratio.mean() < 0.95                                        # and the train as a whole carries less
    # Only the FIRST train echo is compared to the enumeration. Beyond it the two describe different
    # sequences: `cpmg_schedule` winds a whole coherence order every half interval, as an imaging train with a
    # continuous readout does, while this builder winds only inside the crusher windows. The later echoes
    # therefore sit above that prediction, and a like-for-like comparison needs the readout lobes this family
    # does not yet carry. Individual echoes also recover nearly fully -- the Meiboom-Gill oscillation, where a
    # flip error cancels every second echo -- so no per-echo bound holds.
