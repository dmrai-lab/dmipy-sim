"""The field tier accrues through the acquisition's one effective gate (dmipy-sim#571).

A stimulated echo stores its magnetisation along B0 for the mixing time, where a static field adds no phase. The
susceptibility tier reads :func:`~dmipy_sim.replay._replay_kernel.field_gate`, which is
``ScannerSequence.effective_gate`` (the coherence sign times the transverse gate) on the pack grid -- the gate
``G_eff`` and the Maxwell order-0 phase accrue through. The oracle here is the gate written out from the pulses
by hand (+1 before the store, 0 stored, -1 after the recall) against the decoded field samples. Where the gate
was the sign alone already (a spin echo, a gradient echo, every pulse hard), nothing moves.
"""
import numpy as np
import pytest

from dmipy_sim import sequences
from dmipy_sim.constants import GAMMA
from dmipy_sim.replay import so3
from dmipy_sim.replay._replay_kernel import field_gate, gate_weights
from dmipy_sim.spec.tissue import Tissue
from tests.replay_frames import field_along

TISSUE = Tissue(chi_iso=-1e-7, chi_aniso=-5e-8)
B0 = 3.0


def _ste(pk):
    """A stimulated echo on the pack's walk: delta 0.12 ms, TM 0.56 ms, TE 0.8 ms, its store and recall on samples."""
    return sequences.pgste([[1, 0, 0]], 1.2e-4, 5.6e-4, bvalues=[5e8], TE=8e-4, n_t=4 * pk.n_t + 1,
                           slew_rate=np.inf, ste_flip_angles=(90.0, 90.0, 90.0))


def _hand_gate(seq):
    """The stimulated echo's gate from its pulses, on its own grid: +1 transverse before the store, 0 while stored,
    -1 (the recall inverts the phase) from the recall to the echo."""
    t = np.arange(seq.n_t) * seq.dt
    store, recall = (e.t_s for e in seq.rf if e.label in ("store", "recall"))
    return np.where(t < store - 1e-12, 1.0, np.where(t < recall - 1e-12, 0.0, -1.0))


def _field_phase(pk, seq):
    """The field's phase per walker under ``seq`` with the field along ``(0.6, 0, 0.8)`` of the specimen."""
    seq_lab, R = field_along(seq, (0.6, 0.0, 0.8))
    _, _, phi = pk.walker_phases(seq_lab, orientation=R, scanner=B0, tissue=TISSUE)
    _, _, phi0 = pk.walker_phases(seq_lab, orientation=R)
    return (phi - phi0)[:, 0]


def test_a_stimulated_echo_accrues_no_field_phase_while_its_magnetisation_is_stored(field_pack):
    """The field phase of a stimulated echo is the decoded field integrated over its two transverse periods only,
    with the recall's sign: a gate that integrated the mixing time too misses it by the field's whole TM integral."""
    from dmipy_sim.replay.bank import susc_path_decode, susc_path_field
    pk = field_pack
    ste = _ste(pk)
    assert ste.stimulated_echo and ste.TM == pytest.approx(5.6e-4)
    hand = _hand_gate(ste)
    np.testing.assert_array_equal(ste.effective_gate, hand.astype(np.float32))
    g = field_gate(ste, pk.n_t, pk.dt)
    np.testing.assert_allclose(g, gate_weights(hand, ste.dt, pk.n_t, pk.dt)[0], rtol=0, atol=1e-12)
    ch = pk.meta["compression"]["channels"]
    b, _ = susc_path_decode(pk.arrays, ch["susceptibility_path"], n_w=pk.n_walkers)
    u = np.array([0.6, 0.0, 0.8])
    dB = susc_path_field(b, u, B0=B0, chi_iso=TISSUE.chi_iso, chi_aniso=TISSUE.chi_aniso,
                         has_aniso=bool(ch["susceptibility_grid"].get("has_aniso")))
    expect = GAMMA * pk.dt * (dB * gate_weights(hand, ste.dt, pk.n_t, pk.dt)[0][None, :]).sum(1)
    stored = GAMMA * pk.dt * (dB * gate_weights((hand == 0).astype(float), ste.dt, pk.n_t, pk.dt)[0][None, :]).sum(1)
    assert np.abs(stored).max() > 1e-3                                  # the mixing time would carry a phase
    np.testing.assert_allclose(_field_phase(pk, ste), expect, rtol=0, atol=1e-7)


@pytest.mark.parametrize("make", [
    lambda n_t: sequences.pgse([[1, 0, 0]], 1e-4, 4e-4, bvalues=[5e8], TE=8e-4, n_t=n_t, slew_rate=np.inf),
    lambda n_t: sequences.gre(8e-4, gradient_directions=[[1, 0, 0]], bvalues=[5e8], delta=1e-4, Delta=3e-4,
                              n_t=n_t, slew_rate=np.inf),
], ids=["spin-echo", "gradient-echo"])
def test_where_the_gate_was_the_sign_the_field_gate_is_unchanged(field_pack, make):
    """A spin echo and a gradient echo of hard pulses store nothing: the field gate is the sign's, to the bit."""
    pk = field_pack
    seq = make(4 * pk.n_t + 1)
    t = np.arange(seq.n_t) * seq.dt
    np.testing.assert_array_equal(field_gate(seq, pk.n_t, pk.dt), gate_weights(seq.rf.sign(t), seq.dt, pk.n_t, pk.dt)[0])


def test_the_closed_form_and_the_walkers_agree_on_a_stimulated_echo_with_the_field(field_pack):
    """The pose expansion in closed form reads the same field gate as the walker-by-walker replay: equal at every
    rotation for a stimulated echo with the field on."""
    pk = field_pack
    seq_y, R_y = field_along(_ste(pk), (0.0, 1.0, 0.0))
    kw = dict(scanner=B0, tissue=TISSUE)
    pc = pk.pose_response(seq_y, method="closed", pose=R_y, **kw)
    for R in so3.haar_rotations(4, 5):
        np.testing.assert_allclose(pc.at(R), pk.replay(seq_y, orientation=R_y @ R, complex_signal=True, **kw), atol=2e-6)
