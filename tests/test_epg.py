"""The coherence pathways of a sequence, against what is known in closed form.

The amplitudes an extended phase graph produces are not free parameters: a stimulated echo's one half, a
perfect refocusing train's unit echoes, and the collapse of a train whose refocusing pulses are out of the
CPMG geometry are all fixed by the flip angles and phases. These tests hold the enumeration to them.
"""
import numpy as np
import pytest

from dmipy_sim.acquisition import epg


def _echo(paths, k):
    """The coherent amplitude at readout ``k``: the pathways there add, they do not add in magnitude."""
    return abs(sum(p.eta for p in paths if p.readout_idx == k))


# --------------------------------------------------------------- the stimulated echo
@pytest.mark.parametrize("a", [30.0, 60.0, 90.0, 120.0, 150.0])
def test_the_stimulated_echo_is_half_the_product_of_the_sines(a):
    """``0.5 sin(a1) sin(a2) sin(a3)``, and exactly 0.5 at three 90s -- the number the scalar replay route
    applies as a constant (`trajectories.replay(..., stimulated_echo=True)`), here derived."""
    assert epg.ste_amplitude(a, a, a) == pytest.approx(0.5 * np.sin(np.radians(a)) ** 3, abs=1e-12)


def test_three_ninety_degree_pulses_give_exactly_one_half():
    assert epg.ste_amplitude(90.0, 90.0, 90.0) == pytest.approx(0.5, abs=1e-12)


def test_the_stored_pathway_is_longitudinal_across_the_mixing_time():
    """It dephases, parks on z, and comes back as the conjugate coherence that winds down to the echo, so
    its gates read transverse / stored / transverse and its signs read +1 / 0 / -1. Both lobes wind the same
    way, as a stimulated echo's physically do; the rephasing lives in the pathway, not in the lobe."""
    paths = epg.enumerate_pathways(epg.ste_schedule(delta=0.01, TM=0.03, spoil=4), threshold=1e-6)
    assert len(paths) == 1                                   # a crushed stimulated echo leaves one pathway
    assert tuple(st for st, _ in paths[0].intervals) == ("F+", "Z", "F-")
    dt = 0.01
    assert paths[0].chi_perp(dt).tolist() == [1.0] + [0.0] * 3 + [1.0]
    assert paths[0].eps_P(dt).tolist() == [1.0] + [0.0] * 3 + [-1.0]


def test_a_crusher_removes_the_competing_transverse_pathways():
    """Without one, the transverse branches reach the readout too and the echo is not the stored pathway
    alone -- which is why a stimulated echo that is not crushed is not the closed form."""
    bare = epg.enumerate_pathways(epg.ste_schedule(spoil=0), threshold=1e-6)
    crushed = epg.enumerate_pathways(epg.ste_schedule(spoil=4), threshold=1e-6)
    assert len(bare) == 3 and len(crushed) == 1
    assert abs(crushed[0].eta) == pytest.approx(0.5, abs=1e-12)
    assert sum(abs(p.eta) for p in bare) > 0.5


def test_a_crusher_the_sequence_can_rewind_brings_an_unwanted_pathway_back():
    """A crusher's size is not free. On this schedule the two lobes rewind a winding of exactly 2, so that
    choice returns a transverse branch to order zero and the readout sees it beside the stimulated echo --
    the ordinary reason a real crusher's area is chosen with care rather than merely made large."""
    trap = epg.enumerate_pathways(epg.ste_schedule(spoil=2), threshold=1e-6)
    assert len(trap) == 2
    assert {tuple(st for st, _ in p.intervals) for p in trap} == {("F+", "Z", "F-"), ("F+", "F-", "F-")}
    for spoil in (1, 3, 4, 8):
        assert len(epg.enumerate_pathways(epg.ste_schedule(spoil=spoil), threshold=1e-6)) == 1


# --------------------------------------------------------------- the refocusing train
def test_a_perfect_refocusing_train_gives_one_unit_pathway_per_echo():
    paths = epg.enumerate_pathways(epg.cpmg_schedule(6, 180.0), threshold=1e-6)
    assert len(paths) == 6
    for k in range(6):
        at_k = [p for p in paths if p.readout_idx == k]
        assert len(at_k) == 1 and abs(at_k[0].eta) == pytest.approx(1.0, abs=1e-12)
        assert np.all(at_k[0].chi_perp(0.5) == 1.0)          # never stored


@pytest.mark.parametrize("phase", [0.0, 90.0])
def test_a_perfect_refocusing_train_does_not_care_about_the_cpmg_geometry(phase):
    """At 180 degrees both components are refocused whatever the axis, so the condition costs nothing --
    which is why a train only reveals it once the flip angle departs from 180."""
    paths = epg.enumerate_pathways(epg.cpmg_schedule(6, 180.0, refocus_phase_deg=phase), threshold=1e-6)
    assert [_echo(paths, k) for k in range(6)] == pytest.approx([1.0] * 6, abs=1e-12)


@pytest.mark.parametrize("beta", [150.0, 120.0, 90.0])
def test_a_reduced_flip_train_holds_up_in_the_cpmg_geometry_and_collapses_out_of_it(beta):
    """The CPMG train settles into a pseudo-steady state; the same train with its refocusing pulses in phase
    with the excitation loses most of the signal within a few echoes."""
    cpmg = epg.enumerate_pathways(epg.cpmg_schedule(6, beta, refocus_phase_deg=90.0), threshold=1e-4)
    non = epg.enumerate_pathways(epg.cpmg_schedule(6, beta, refocus_phase_deg=0.0), threshold=1e-4)
    e_cpmg = np.array([_echo(cpmg, k) for k in range(6)])
    e_non = np.array([_echo(non, k) for k in range(6)])
    assert e_cpmg[0] == pytest.approx(e_non[0], abs=1e-12)   # the first echo predates the difference
    assert e_cpmg[1:].min() > 3 * e_non[1:].min()
    assert e_cpmg[1:].mean() > 2 * e_non[1:].mean()


# --------------------------------------------------------------- the enumeration itself
def test_a_pathway_lays_its_gates_onto_a_grid():
    paths = epg.enumerate_pathways(epg.cpmg_schedule(2, 120.0, TE=0.02), threshold=1e-3)
    dt = 0.001
    for p in paths:
        chi, eps = p.chi_perp(dt), p.eps_P(dt)
        assert chi.shape == eps.shape
        assert set(np.unique(chi)) <= {0.0, 1.0}
        assert set(np.unique(eps)) <= {-1.0, 0.0, 1.0}
        assert np.all((eps != 0) == (chi == 1.0))            # stored means no phase, transverse means some
        assert chi.size == sum(max(1, int(round(d / dt))) for _, d in p.intervals)


def test_the_threshold_prunes_the_weak_branches():
    many = epg.enumerate_pathways(epg.cpmg_schedule(5, 90.0), threshold=1e-6)
    few = epg.enumerate_pathways(epg.cpmg_schedule(5, 90.0), threshold=0.2)
    assert len(few) < len(many)
    assert min(abs(p.eta) for p in few) >= 0.2


def test_a_winding_of_several_orders_is_the_unit_step_repeated():
    """A crusher is a winding of more than one order; it is the same step taken again, not a special case."""
    one = epg.enumerate_pathways(epg.Schedule((epg.Pulse(90.0), epg.Winding(+1, 1.0), epg.Pulse(180.0, 90.0),
                                               epg.Winding(+1, 1.0, readout=True))), threshold=1e-6)
    assert len(one) == 1 and abs(one[0].eta) == pytest.approx(1.0, abs=1e-12)
    three = epg.enumerate_pathways(epg.Schedule((epg.Pulse(90.0), epg.Winding(+3, 1.0), epg.Pulse(180.0, 90.0),
                                                 epg.Winding(+3, 1.0, readout=True))), threshold=1e-6)
    assert len(three) == 1 and abs(three[0].eta) == pytest.approx(1.0, abs=1e-12)
    unbalanced = epg.enumerate_pathways(epg.Schedule((epg.Pulse(90.0), epg.Winding(+1, 1.0), epg.Pulse(180.0, 90.0),
                                                      epg.Winding(+3, 1.0, readout=True))), threshold=1e-6)
    assert unbalanced == []                                  # nothing returns to order zero: no echo forms
