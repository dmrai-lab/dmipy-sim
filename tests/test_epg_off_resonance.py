"""Off-resonance in an EPG state, checked against an isochromat sum (dmipy-sim#285).

The invariant that matters is the COHERENCE ORDER, not the dephasing index. Every F coefficient in this
representation is ordinary transverse magnetisation, so a uniform offset advances them all by the same
angle whatever their winding; what reverses the accumulated phase is the RF conjugating the state, not the
sign of k. Keying the sign to the index instead agrees with the truth for a monotonically dephasing
sequence and disagrees for a bipolar one, which is exactly the case a diffusion preparation can produce.
"""
import numpy as np
import pytest

from dmipy_sim.acquisition.epg_state import EPGState

DT = 4.0e-3
DW = 2.0 * np.pi * 13.0


def _bloch(events, dw, n_spin=40001, spread=8.0):
    """The truth: a sum over isochromats spread across whole windings, precessing at ``dw``."""
    phi = np.linspace(-np.pi * spread, np.pi * spread, n_spin)
    M = np.zeros((n_spin, 3))
    M[:, 2] = 1.0
    for ev in events:
        if ev[0] == "pulse":
            flip, phase = np.radians(ev[1]), np.radians(ev[2])
            n = np.array([np.cos(phase), np.sin(phase), 0.0])
            K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
            M = M @ (np.eye(3) + np.sin(flip) * K + (1 - np.cos(flip)) * (K @ K)).T
        else:
            ang = ev[1] * phi + dw * ev[2]
            mx, my = M[:, 0].copy(), M[:, 1].copy()
            M[:, 0] = mx * np.cos(ang) - my * np.sin(ang)
            M[:, 1] = mx * np.sin(ang) + my * np.cos(ang)
    return (M[:, 0] + 1j * M[:, 1]).mean()


def _epg(events, dw, n_orders=12):
    s = EPGState.equilibrium(n_orders)
    for ev in events:
        if ev[0] == "pulse":
            s.pulse(ev[1], ev[2])
        else:
            s.shift(ev[1])
            s.off_resonance(ev[2], dw)
    return s.signal


CASES = {
    "spin echo": [("p", 90, 90), ("i", +1, DT), ("p", 180, 0), ("i", +1, DT)],
    "gradient echo, wound +/-": [("p", 90, 90), ("i", +1, DT), ("i", -1, DT)],
    "gradient echo, wound -/+": [("p", 90, 90), ("i", -1, DT), ("i", +1, DT)],
    "bipolar, 180, bipolar": [("p", 90, 90), ("i", +1, DT), ("i", -1, DT),
                              ("p", 180, 0), ("i", +1, DT), ("i", -1, DT)],
    "stimulated echo": [("p", 90, 90), ("i", +1, DT), ("p", 90, 90), ("i", 0, 3 * DT),
                        ("p", 90, 90), ("i", +1, DT)],
}
CASES = {k: [("pulse" if e[0] == "p" else "int", e[1], e[2]) for e in v] for k, v in CASES.items()}


@pytest.mark.parametrize("name", list(CASES))
def test_off_resonance_matches_an_isochromat_sum(name):
    events = CASES[name]
    truth, got = _bloch(events, DW), _epg(events, DW)
    assert abs(got) == pytest.approx(abs(truth), abs=2e-3)
    if abs(truth) > 1e-6:
        assert abs(np.angle(np.exp(1j * (np.angle(got) - np.angle(truth))))) < 2e-3


def test_the_two_windings_of_a_gradient_echo_agree_which_the_dephasing_index_would_not():
    """The case that distinguishes the right invariant from the plausible one. A gradient echo wound +/-
    and one wound -/+ are the same experiment as far as a uniform offset is concerned -- both spent the same
    time transverse with no refocusing pulse -- so both carry the same phase. Keying the sign to the
    dephasing index reports zero for the second, because it counts one interval as advancing and the other
    as retarding."""
    plus = _epg(CASES["gradient echo, wound +/-"], DW)
    minus = _epg(CASES["gradient echo, wound -/+"], DW)
    assert np.angle(plus) == pytest.approx(np.angle(minus), abs=2e-3)
    assert np.angle(plus) == pytest.approx(2.0 * DW * DT, abs=2e-3)   # two intervals, neither refocused
    assert abs(np.angle(plus)) > 0.5                                   # and it is not a small effect


def test_a_uniform_offset_cannot_move_amplitude_between_coherence_orders():
    """The structural reason a refocusing train is insensitive to drift, and the property the earlier
    implementation broke. A uniform offset multiplies every transverse coefficient by the SAME factor, so
    it is a global phase: it can rotate the signal but never redistribute it. An operator that conjugated
    by the sign of the index was not uniform, and it moved magnitude between orders -- which no static
    field offset can do."""
    rng = np.random.default_rng(0)
    s = EPGState.equilibrium(8)
    s.pulse(60.0, 0.0)
    for _ in range(3):
        s.shift(1)
        s.pulse(50.0, 90.0)
    before = np.abs(s.F.copy())
    s.off_resonance(7e-3, 2 * np.pi * 250.0)
    np.testing.assert_allclose(np.abs(s.F), before, rtol=1e-12)
