"""A pack built for one stated acquisition (``simulate``'s in-memory pack) states the class of that acquisition."""
from types import SimpleNamespace

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_replay.sequences import pgse
from dmipy_replay.replay.replay import check_acquisition_envelope
from dmipy_sim.engine.core import _walk_pack
from dmipy_sim.replay.compression import ACQUISITION_CLASS, acquisition_envelope, certified_envelope


def _seq(g):
    return pgse(np.array([[1.0, 0, 0]]), 5e-3, 15e-3, gradient_strengths=g, n_t=400, slew_rate=200.0)


def test_the_envelope_states_the_acquisitions_own_limits():
    seq = _seq(0.04)
    env = acquisition_envelope(seq)
    played = np.abs(np.asarray(seq.played_gradient)).max()
    assert env["scanner"] == ACQUISITION_CLASS and env["G_max"] == pytest.approx(played)
    assert env["slew_max"] > 0.0


def test_the_class_needs_its_limits_stated():
    with pytest.raises(ValueError, match="state G_max and slew_max"):
        certified_envelope(dict(scanner=ACQUISITION_CLASS))


def test_simulates_pack_replays_its_acquisition_and_refuses_a_stronger_one():
    seq = _seq(0.04)
    pack, tissue, pose = _walk_pack(200, 2e-9, seq, d.Sphere(radius=4e-6), seed=1, T2=None, T1=None, r0=None,
                                    require_gpu=False, walker_batch_size=None)
    acq = pack.replay_envelope["acquisition"]
    assert acq["bound"] == "scanner_class" and acq["scanner"] == ACQUISITION_CLASS
    assert acq["G_max"] == pytest.approx(np.abs(np.asarray(seq.played_gradient)).max())
    pack.replay(seq, tissue=tissue, orientation=pose)                      # its own acquisition: accepted
    stronger = SimpleNamespace(G=2.0 * np.asarray(seq.played_gradient), dt=seq.dt)         # the reader's check
    with pytest.raises(ValueError, match="beyond the G_max"):
        check_acquisition_envelope(pack.meta, stronger)


def test_a_gradient_free_acquisition_states_zero_limits_and_simulates():
    seq = _seq(0.0)
    env = acquisition_envelope(seq)
    assert env["G_max"] == 0.0 and env["slew_max"] == 0.0
    S = np.asarray(d.simulate(200, 2e-9, seq, d.Sphere(radius=4e-6), seed=1, require_gpu=False)).ravel()
    assert S[0] == pytest.approx(1.0, abs=1e-6)
