"""A replay carries the amplitude of the pathway its readout IS, on every route.

A stimulated echo is not the whole magnetisation: the store keeps one part and the rest is crushed away, so
the readout carries ``0.5 sin a1 sin a2 sin a3`` of it. That number is derived from the RF schedule
(:func:`dmipy_sim.acquisition.epg.pathway_weight`), not written down, so a schedule whose pulses are not 90
degrees gets its own amplitude rather than a flat one half.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.acquisition import epg
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.study import Acquisition, Protocol, Study
from dmipy_sim.spec.tissue import Tissue

T2, T1, TM, DELTA = 0.08, 0.3, 0.02, 0.003


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    """Free water, so the only thing left in a b = 0 readout is relaxation and the pathway's amplitude."""
    walk = d.simulate_trajectories(600, 2e-9, d.FreeDiffusion(), 0.05, 5e-4, seed=0, require_gpu=False)
    out = tmp_path_factory.mktemp("pw") / "free.rpk"
    return build_replay_pack(walk, id="test/free", license="x", citation="x", K=8, out_path=str(out))


def _pgste(dt_n=600, flips=(90.0, 90.0, 90.0), b=0.0):
    return sequences.pgste([[1.0, 0.0, 0.0]], DELTA, TM, bvalues=[b], n_t=dt_n, slew_rate=np.inf,
                           ste_flip_angles=flips)


def _closed_form(seq, flips=(90.0, 90.0, 90.0)):
    """``eta exp(-(TE - TM)/T2) exp(-TM/T1)``: transverse either side of the store, longitudinal across it."""
    TE = float(np.asarray(seq.readout)[-1]) * float(seq.dt)
    eta = 0.5 * np.prod(np.sin(np.radians(flips)))
    return eta * np.exp(-(TE - TM) / T2) * np.exp(-TM / T1)


def test_the_weight_comes_from_the_schedule_not_a_constant():
    assert epg.pathway_weight(_pgste()) == pytest.approx(0.5, abs=1e-12)
    assert epg.pathway_weight(_pgste(flips=(90.0, 60.0, 90.0))) == pytest.approx(0.5 * np.sin(np.radians(60)), abs=1e-12)
    assert epg.pathway_weight(sequences.pgse([[1.0, 0, 0]], 5e-3, 0.02, bvalues=[1e9], n_t=400, slew_rate=np.inf)) == 1.0


def test_a_stimulated_echo_replays_to_its_closed_form(pack):
    seq = _pgste()
    got = float(np.asarray(pack.replay(seq, tissue=Tissue(T2={"extra": T2}, T1={"extra": T1})))[0])
    assert got == pytest.approx(_closed_form(seq), rel=2e-3)


@pytest.mark.parametrize("flips", [(90.0, 60.0, 90.0), (60.0, 60.0, 60.0), (90.0, 120.0, 90.0)])
def test_the_flip_angles_set_the_amplitude(pack, flips):
    """A flat one half is right only at three 90s; every other schedule carries less, and the difference is
    far larger than the pack's floor."""
    seq = _pgste(flips=flips)
    got = float(np.asarray(pack.replay(seq, tissue=Tissue(T2={"extra": T2}, T1={"extra": T1})))[0])
    assert got == pytest.approx(_closed_form(seq, flips), rel=2e-3)
    flat = _closed_form(seq, (90.0, 90.0, 90.0))                 # what a constant 0.5 would have given
    assert abs(got - flat) > 0.02


def test_a_spin_echo_is_untouched(pack):
    """Its pathway is the whole magnetisation, so nothing is applied and the b = 0 readout is the T2 decay."""
    seq = sequences.pgse([[1.0, 0, 0]], 5e-3, 0.02, bvalues=[0.0], TE=0.045, n_t=600, slew_rate=np.inf)
    TE = float(np.asarray(seq.readout)[-1]) * float(seq.dt)
    got = float(np.asarray(pack.replay(seq, tissue=Tissue(T2={"extra": T2}, T1={"extra": T1})))[0])
    assert got == pytest.approx(np.exp(-TE / T2), rel=2e-3)


def test_every_route_agrees(pack):
    """The weights are formed in two places -- the replay's preparation and a study's primitives -- so the
    routes are held to each other, which is what stops the amplitude being applied twice or not at all."""
    seq, tis = _pgste(flips=(90.0, 60.0, 90.0)), Tissue(T2={"extra": T2}, T1={"extra": T1})
    direct = float(np.asarray(pack.replay(seq, tissue=tis))[0])
    w, ew, E = pack.walker_signals(seq, tissue=tis)
    from_walkers = float(np.abs((ew[:, None] * E).sum(0) / w.sum())[0])
    prim = float(np.abs(np.asarray(pack.study(Study(Protocol([Acquisition(seq)]), tissues=[tis], scanners=[None])))[0, 0]))
    assert from_walkers == pytest.approx(direct, rel=1e-9)
    assert prim == pytest.approx(direct, rel=1e-9)


def test_the_amplitude_rides_beside_the_weights_not_inside_them(pack):
    """``ew`` means the relaxation and surface terms and a codec oracle checks it means only that, so the
    pathway's amplitude is returned beside it and each route that forms a signal applies it once. The
    vector-Bloch route therefore needs no exemption: it reads the weights, which do not carry it, and gets
    the amplitude from propagating the magnetisation through the actual pulses instead."""
    seq = _pgste()
    P = pack._prepare(seq, tissue=None, scanner=None, orientation=None, compartment=None)
    assert P["pathway"] == pytest.approx(0.5, abs=1e-12)
    assert np.allclose(P["ew"], P["w"])                      # no tissue asked for: the weights are the plain ones
    _, ew, _ = pack.walker_phases(seq)                       # the public route folds it in, once
    assert np.allclose(ew, 0.5 * P["ew"])
    assert pack._prepare(sequences.pgse([[1.0, 0, 0]], 5e-3, 0.02, bvalues=[0.0], TE=0.045, n_t=600,
                                        slew_rate=np.inf),
                         tissue=None, scanner=None, orientation=None, compartment=None)["pathway"] == 1.0


def test_a_single_reduced_flip_echo_is_one_pathway_and_has_a_closed_form():
    """``sin^2(beta/2)``, enumerated rather than written down, and 1 at a perfect 180."""
    for beta in (180.0, 150.0, 120.0, 90.0, 60.0):
        seq = sequences.pgse([[1.0, 0, 0]], 5e-3, 0.02, bvalues=[0.0], TE=0.045, n_t=600, slew_rate=np.inf)
        seq = seq.__class__(G=seq.G, dt=seq.dt, readout=seq.readout,
                            rf=type(seq.rf)((seq.rf[0], seq.rf[1].__class__(seq.rf[1].t_s, beta, "refocus", 90.0))))
        assert epg.pathway_weight(seq) == pytest.approx(np.sin(np.radians(beta) / 2) ** 2, abs=1e-9)


def test_a_train_it_cannot_describe_is_refused_not_guessed():
    """A six-echo train at 120 degrees runs 0.75, 0.94, 0.84, ... -- different at every echo and each a sum
    over pathways, so no single amplitude describes the readout. Returning 1 would be wrong by a quarter on
    the first echo; the refusal names what is needed instead."""
    assert epg.pathway_weight(sequences.cpmg(6, 0.02, beta_deg=180.0, n_t_per_echo=60)) == 1.0
    with pytest.raises(ValueError, match="SUM over several coherence pathways"):
        epg.pathway_weight(sequences.cpmg(6, 0.02, beta_deg=120.0, n_t_per_echo=60))


def _scaled(seq, kappa, both):
    """The sequence with its flips scaled by ``kappa``: both pulses, or the refocusing pulse alone."""
    ex, rf180 = seq.rf[0], seq.rf[1]
    ex2 = ex.__class__(ex.t_s, ex.flip_deg * kappa if both else ex.flip_deg, ex.label, ex.axis_deg)
    rf2 = rf180.__class__(rf180.t_s, rf180.flip_deg * kappa, rf180.label, rf180.axis_deg)
    return seq.__class__(G=seq.G, dt=seq.dt, readout=seq.readout, rf=type(seq.rf)((ex2, rf2)))


def test_a_transmit_scale_on_both_pulses_scales_the_excitation_too():
    """A B1 scale acts on every pulse, so the amplitude is ``sin(90 kappa) sin^2(90 kappa)``, the enumeration's
    own number, and not the refocusing factor alone (dmipy-sim#391: 12 % high at kappa 0.7)."""
    seq = sequences.pgse([[1.0, 0, 0]], 5e-3, 0.02, bvalues=[0.0], TE=0.045, n_t=600, slew_rate=np.inf)
    for kappa in (0.7, 0.85, 1.0, 1.15, 1.3):
        a, b = np.radians(90.0 * kappa), np.radians(180.0 * kappa)
        assert epg.pathway_weight(_scaled(seq, kappa, both=True)) == pytest.approx(abs(np.sin(a)) * np.sin(b / 2) ** 2, abs=1e-9)
        assert epg.pathway_weight(_scaled(seq, kappa, both=False)) == pytest.approx(np.sin(b / 2) ** 2, abs=1e-9)
        sch = epg.Schedule((epg.Pulse(90.0 * kappa), epg.Winding(+1, 1.0), epg.Pulse(180.0 * kappa, 90.0), epg.Winding(+1, 1.0, readout=True)))
        assert epg.pathway_weight(_scaled(seq, kappa, both=True)) == pytest.approx(
            abs(sum(p.eta for p in epg.enumerate_pathways(sch, threshold=1e-12))), abs=1e-12)


def test_a_gradient_echo_and_a_perfect_train_carry_the_excitation_sine():
    gre = sequences.gre(0.02, n_t=200)
    ex = gre.rf[0]
    gre60 = gre.__class__(G=gre.G, dt=gre.dt, readout=gre.readout, rf=type(gre.rf)((ex.__class__(ex.t_s, 60.0, ex.label, ex.axis_deg),)))
    assert epg.pathway_weight(gre) == pytest.approx(1.0, abs=1e-12)
    assert epg.pathway_weight(gre60) == pytest.approx(np.sin(np.radians(60.0)), abs=1e-12)
    train = sequences.cpmg(6, 0.02, beta_deg=180.0, n_t_per_echo=60)
    ex = train.rf[0]
    train60 = train.__class__(G=train.G, dt=train.dt, readout=train.readout,
                              rf=type(train.rf)((ex.__class__(ex.t_s, 60.0, ex.label, ex.axis_deg),) + tuple(train.rf[1:])))
    assert epg.pathway_weight(train60) == pytest.approx(np.sin(np.radians(60.0)), abs=1e-12)
