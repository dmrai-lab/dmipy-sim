"""The diffusion-prepared refocusing train: what an ultra-low-field diffusion FSE plays.

A preparation sets every spin's phase by its own path, then a train reads many echoes from it. Two things
make the family what it is and both are the builder's job: every echo lands on a sample, and every refocusing
pulse carries a balanced crusher pair. Without the first a pulse sits off centre in its own window and
dephases the echo the crusher exists to keep; without the second the coherence pathways stay degenerate and
the train stops depending on its refocusing flip angle, which is the behaviour it is built to have.
"""
from dataclasses import replace

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
    # Only the FIRST train echo is compared here, against the sum of the pathways that reach it. The later
    # echoes agree too, to better than 0.03 -- see
    # `test_the_replayed_train_is_the_sum_over_pathways_at_every_echo`, which checks all five. Individual
    # echoes recover nearly fully (the Meiboom-Gill oscillation, where a flip error cancels every second
    # echo), so the `ratio.mean()` bound below is about the train as a whole and no per-echo bound holds.


# ── analytic oracles for a refocusing train (dmipy-sim#285) ─────────────────────────────────────────
# Two closed forms, written out here rather than taken from the enumeration, so they can judge it:
#
#   the refocused (SE) family at echo k        sin^2(beta/2) ^ (k+1)
#   the fraction a pulse STORES along z        (1/2) sin(beta)
#
# The second is what a split-echo readout exists to catch, and it is IDENTICALLY ZERO at beta = 180: a
# perfect train stores nothing and has exactly one coherence family. That statement needs no simulation,
# and it is the acceptance test for any future split readout (see the note at the end of this file).

def _se_closed_form(beta_deg, k):
    return np.sin(np.deg2rad(beta_deg) / 2.0) ** (2 * (k + 1))


def _stored_fraction(beta_deg):
    return 0.5 * abs(np.sin(np.deg2rad(beta_deg)))


@pytest.mark.parametrize("beta", [180.0, 150.0, 120.0, 90.0])
def test_the_refocused_family_follows_sin_squared_half_beta(beta):
    """Every pulse keeps sin^2(beta/2) of the transverse magnetisation on the refocused path, so the family
    that never leaves it is that number to the power of the echoes it has passed."""
    for k in range(3):
        got = [abs(p.eta) for p in epg.enumerate_pathways(epg.cpmg_schedule(4, beta), threshold=1e-6)
               if p.readout_idx == k
               and [i[0] for i in p.intervals] == ['F+'] + ['F-'] * (len(p.intervals) - 1)]
        assert len(got) == 1, f"the refocused family should be one pathway; got {len(got)}"
        assert got[0] == pytest.approx(_se_closed_form(beta, k), rel=1e-9)


def test_a_perfect_refocusing_train_stores_nothing_so_it_has_no_second_family():
    """(1/2) sin(beta) is zero at 180 and maximal at 90. A train of perfect 180s therefore has exactly one
    coherence family, which is why a split readout of one must find the other side empty."""
    assert _stored_fraction(180.0) == pytest.approx(0.0, abs=1e-12)
    assert _stored_fraction(90.0) == pytest.approx(0.5)
    for beta, n_stored in ((180.0, 0), (120.0, 3)):
        ps = [p for p in epg.enumerate_pathways(epg.cpmg_schedule(4, beta), threshold=1e-4)
              if p.readout_idx == 2]
        assert sum('Z' in [i[0] for i in p.intervals] for p in ps) == n_stored


@pytest.mark.parametrize("beta", [150.0, 120.0, 90.0])
def test_the_replayed_train_is_the_sum_over_pathways_at_every_echo(beta):
    """What a readout measures is not one pathway but ALL of them arriving together, and the replayed train
    reproduces that sum at every echo, not merely the first. Taken as a ratio to the 180 train so the
    preparation, the relaxation and the diffusion weighting divide out."""
    walk = d.simulate_trajectories(600, 1e-14, d.FreeDiffusion(), 0.30, 2.5e-4, seed=0, require_gpu=False)
    pack = build_replay_pack(walk, id="test/static", license="x", citation="x", K=8)
    n = 5

    def replayed(b):
        s = sequences.splice([[1.0, 0, 0]], 35e-3, 42e-3, n, 10e-3, bvalues=[0.0], TE_prep=PREP,
                             beta_deg=b, n_t_per_echo=40)
        return np.abs(np.asarray(pack.replay_bloch(s)).reshape(-1))

    got = (replayed(beta) / np.maximum(replayed(180.0), 1e-12))[1:1 + n]
    want = np.array([abs(sum(p.eta for p in epg.enumerate_pathways(epg.cpmg_schedule(n, beta), threshold=1e-5)
                             if p.readout_idx == k)) for k in range(n)])
    assert np.abs(got - want).max() < 0.03, f"got {got}, enumeration {want}"


def _families(n_echoes, beta):
    """The two echo families of a SPLICE train: |E1| and |E2| at each refocusing interval."""
    sched = epg.splice_schedule(n_echoes, beta)
    tot = [abs(sum(p.eta for p in epg.enumerate_pathways(sched, threshold=1e-9) if p.readout_idx == k))
           for k in range(2 * n_echoes)]
    return np.array(tot[0::2]), np.array(tot[1::2])


def test_the_splice_families_reproduce_the_reference_implementation():
    """Against Rahbek et al. 2023 (MRM 89:1469) `epg_splice.m`, no relaxation, beta = 120 degrees. These are
    the reference implementation's own numbers, so this is a cross-check against another engine and not
    against ourselves."""
    e1, e2 = _families(4, 120.0)
    np.testing.assert_allclose(e1, [0.75, 0.375, 0.28125, 0.609375], rtol=1e-9)
    np.testing.assert_allclose(e2, [0.0, 0.5625, 0.5625, 0.246094], rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("beta", [180.0, 150.0, 120.0, 90.0])
def test_the_first_splice_echoes_have_closed_forms(beta):
    """The opening of the train is short enough to write down: the first E1 is the plain spin echo, the
    second is the stimulated one, and E2 lags a family behind."""
    b = np.deg2rad(beta)
    e1, e2 = _families(4, beta)
    assert e1[0] == pytest.approx(np.sin(b / 2) ** 2, rel=1e-9)                    # spin echo
    assert e1[1] == pytest.approx(0.5 * np.sin(b) ** 2, rel=1e-9, abs=1e-12)       # stimulated
    assert e2[0] == pytest.approx(0.0, abs=1e-12)                                  # E2 is empty first
    assert e2[1] == pytest.approx(np.sin(b / 2) ** 4, rel=1e-9)                    # secondary spin echo
    assert e2[2] == pytest.approx(np.sin(b / 2) ** 2 * np.sin(b) ** 2, rel=1e-9, abs=1e-12)


def test_at_180_degrees_the_split_degenerates_into_alternate_lines():
    """Nothing is stored along z at 180, so one pathway survives and it lands alternately in one family and
    the other. Each k-space would then get every second phase encode, which is why a SPLICE train is never
    run at 180 -- and is the sharpest statement there is that the two families are real and distinct."""
    e1, e2 = _families(6, 180.0)
    np.testing.assert_allclose(e1, [1, 0, 1, 0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(e2, [0, 1, 0, 1, 0, 1], atol=1e-12)
    # and every echo is carried by exactly one family, never shared
    np.testing.assert_allclose(np.minimum(e1, e2), 0.0, atol=1e-12)


def test_a_reduced_flip_fills_both_families_at_once():
    """Below 180 both families carry signal in the same interval, which is what makes the split worth doing:
    two k-spaces from one train."""
    for beta in (150.0, 120.0, 90.0):
        e1, e2 = _families(6, beta)
        assert (np.minimum(e1, e2)[1:] > 0.02).all(), f"beta={beta} leaves a family empty"


@pytest.mark.parametrize("beta", [180.0, 150.0, 120.0, 90.0])
def test_the_replayed_split_train_reproduces_both_echo_families(beta):
    """The whole point, end to end: a Monte-Carlo replay of a split train reproduces the EPG families the
    reference implementation gives, at every flip angle and every echo, to the walk's own noise floor.

    The two are independent: the EPG side is a pathway enumeration and the replay is 4000 walkers stepped
    through the actual pulses with a voxel-scale winding. Scaled so the first echo matches, since the
    preparation's own amplitude is not what is being tested."""
    n = 5
    walk = d.simulate_trajectories(4_000, 1e-14, d.FreeDiffusion(), 0.40, 2.5e-4, seed=0, require_gpu=False)
    pack = build_replay_pack(walk, id="test/static", license="x", citation="x", K=8)
    seq = sequences.splice([[1.0, 0, 0]], 35e-3, 42e-3, n, 10e-3, bvalues=[0.0], TE_prep=PREP,
                           beta_deg=beta, n_t_per_echo=40).with_split_readout()
    S = np.abs(np.asarray(pack.replay_bloch(seq)).reshape(-1))
    e1, e2 = S[0::2], S[1::2]
    want1, want2 = _families(n, beta)
    scale = want1[0] / e1[0]
    floor = 3.0 / np.sqrt(4_000)
    np.testing.assert_allclose(e1 * scale, want1, atol=floor)
    # The LAST interval's late family is excluded: it is read three quarters after the final pulse, in the
    # tail where the train stops, while the enumeration assumes the interval structure carries on. It comes
    # back about three quarters of the predicted amplitude at every flip angle, consistently -- a truncation
    # of the train, not a disagreement about the physics. Every other echo of both families matches.
    np.testing.assert_allclose((e2 * scale)[:-1], want2[:-1], atol=floor)
    if want2[-1] > 0.1:                    # at 180 the last E2 is legitimately empty, so there is no ratio
        assert 0.6 < (e2[-1] * scale) / want2[-1] < 0.9
    else:
        assert e2[-1] * scale < floor


def test_the_split_winding_must_be_whole_turns():
    """A fraction of a turn leaves the family that should be empty partly in phase with itself, and the
    split blurs instead of failing, so it is refused rather than rounded."""
    t = sequences.splice([[1.0, 0, 0]], 35e-3, 42e-3, 4, 10e-3, bvalues=[0.0], TE_prep=PREP,
                         beta_deg=120.0, n_t_per_echo=40)
    with pytest.raises(ValueError, match="whole number of turns"):
        t.with_split_readout(12.5)
    with pytest.raises(ValueError, match="not already split"):
        t.with_split_readout().with_split_readout()


def test_a_split_pair_straddles_its_own_echo_and_the_grid_grows_to_hold_it():
    t = sequences.splice([[1.0, 0, 0]], 35e-3, 42e-3, 4, 10e-3, bvalues=[0.0], TE_prep=PREP,
                         beta_deg=120.0, n_t_per_echo=40)
    sp = t.with_split_readout()
    assert sp.split_echo and len(sp.readout) == 2 * len(t.echoes[1:])
    for k, e in enumerate(t._schedule_echo_idx[1:]):
        early, late = sp.readout[2 * k], sp.readout[2 * k + 1]
        assert early < e < late and early + late == 2 * e
    assert sp.n_t > t.n_t          # the late family of the last pair is read past the builder's last echo
    sp.validate()


def test_what_b_value_each_echo_of_a_split_train_actually_delivers():
    """A split train does not deliver the b it was prepared with, and does not deliver the same b at every
    echo. Measured, not fitted: free diffusion attenuates exactly as exp(-b D), so replaying the same train
    on a static pack and a free one gives b = -ln(S/S0)/D directly. The crusher's per-walker phase is a fixed
    random number, independent of position, so it divides out of the ratio exactly.

    This matters for a low-field experiment (dmipy-sim#285), where the whole point is an ADC, and an ADC
    computed against the prepared b is wrong by whatever this measures."""
    D, n, prepared = 2.0e-9, 5, 0.945e9
    mk = lambda diff: build_replay_pack(
        d.simulate_trajectories(4_000, diff, d.FreeDiffusion(), 0.40, 2.5e-4, seed=0, require_gpu=False),
        id="test/b", license="x", citation="x", K=8)
    static, free = mk(1e-14), mk(D)

    def delivered(beta, bval):
        s = sequences.splice([[1.0, 0, 0]], 35e-3, 42e-3, n, 10e-3, bvalues=[bval], TE_prep=PREP,
                             beta_deg=beta, n_t_per_echo=40).with_split_readout()
        S0 = np.abs(np.asarray(static.replay_bloch(s)).reshape(-1))
        S = np.abs(np.asarray(free.replay_bloch(s)).reshape(-1))
        keep = (S0 > 0.05 * S0.max()) & (S > 0)
        return -np.log(S[keep] / S0[keep]) / D

    # With nothing prepared, nothing is delivered: the readout winding is voxel-scale, so it contributes no
    # diffusion weighting at all. A physical readout gradient would, and this is the measurement that says
    # how much is currently missing -- exactly zero.
    np.testing.assert_allclose(delivered(120.0, 0.0), 0.0, atol=2e6)

    # At 180 degrees one pathway survives, so every echo of both families delivers the SAME b -- which is
    # what makes this measurement trustworthy. It is not the prepared b: reading a quarter-interval off the
    # echo costs about an eighth of it.
    at180 = delivered(180.0, prepared)
    assert at180.std() < 0.01 * at180.mean()
    assert 0.85 < at180.mean() / prepared < 0.91

    # Below 180 the pathways part and the delivered b varies echo to echo by far more than it varies
    # between the two families: about a quarter of the mean, against a few percent between families.
    at120 = delivered(120.0, prepared)
    spread = (at120.max() - at120.min()) / at120.mean()
    assert 0.15 < spread < 0.40, f"echo-to-echo spread {spread:.3f}"
    assert at120.min() / prepared > 0.6 and at120.max() / prepared < 1.0


def test_the_prolonged_readouts_own_diffusion_weighting_is_negligible():
    """SPLICE's readout is single-polarity and never rewound within an interval, so its b accumulates over
    the whole train instead of cancelling -- which is why it is worth checking rather than assuming, at low
    field where the echo spacing is long (dmipy-sim#285).

    It is negligible anyway, and the reason is that its amplitude is not free: the imaging resolution fixes
    it through k_max = gamma_bar G ESP/4, and b goes as G^2. A 3 mm image needs 1.6 mT/m, against the ~11
    mT/m the diffusion preparation plays, so the readout contributes a few thousandths of the b: the b of an
    unrewound gradient is read with its moment anchored at the readout, as the walk accrues it (dmipy-sim#392).

    Computed exactly from the waveform rather than measured: the effect is far below what 4000 walkers can
    resolve, and a Monte-Carlo estimate of it comes back as noise of either sign."""
    gamma_bar, prepared = 42.577e6, 0.945e9
    esp, n = 10e-3, 5
    base = sequences.splice([[1.0, 0, 0]], 35e-3, 42e-3, n, esp, bvalues=[0.0], TE_prep=PREP,
                            beta_deg=180.0, n_t_per_echo=40).with_split_readout()
    idx = [int(round(t / base.dt)) for t in base.echoes]
    q = int(round(np.mean(np.diff(idx)))) // 4

    def b_of_readout(res_m):
        amp = (1.0 / (2 * res_m)) / (gamma_bar * esp / 4)          # k_max = gamma_bar G ESP/4
        G = np.array(base.G, np.float64)
        G[:, idx[0] + q:min(idx[-1] + q, base.n_t), 0] += amp      # single polarity, never rewound
        return amp, float(replace(base, G=G.astype(np.float32)).b()[0])

    amp3, b3 = b_of_readout(3e-3)
    assert 1.4e-3 < amp3 < 1.8e-3, f"{amp3*1e3:.2f} mT/m for a 3 mm image"
    assert b3 / prepared < 5e-3, f"readout carries {100*b3/prepared:.3f} % of the prepared b"

    # finer images need a stronger readout and b goes as G^2, so it grows quadratically -- and is still
    # nothing by 1.5 mm, which is far beyond what a 64 mT scanner images at
    amp15, b15 = b_of_readout(1.5e-3)
    assert amp15 == pytest.approx(2 * amp3, rel=1e-6)
    assert b15 / b3 == pytest.approx(4.0, rel=0.05)
    assert b15 / prepared < 2e-2
