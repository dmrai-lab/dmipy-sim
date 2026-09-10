"""One acquisition: one set of builders, whether the amplitude or the b is what the caller states; every
builder's declared b is the numeric b of the effective gradient it built; there is one b and one B-tensor
integral; `simulate` takes a `ScannerSequence` or a scheme wrapping one, in both engines.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences as S
from dmipy_sim.acquisition.waveforms import b_from_gradient, btensor_from_gradient

D = 2e-9
B = np.array([0.5e9, 1.0e9, 2.0e9])
DIRS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, 0.0, 0.8]])
DELTA, DELTA_BIG, N_T = 8e-3, 24e-3, 400


def _num_b(seq):
    return b_from_gradient(seq.G_eff, seq.dt)


# ── the amplitude and the b are two ways of stating one gradient ────────────────────────────
def test_the_amplitude_route_and_the_b_route_build_the_same_gradient():
    seq = S.pgse(DIRS, DELTA, DELTA_BIG, bvalues=B, n_t=N_T, slew_rate=np.inf)
    g = seq.encoding.gradient_strengths
    wf = S.pgse(DIRS, DELTA, DELTA_BIG, gradient_strengths=g, n_t=N_T, slew_rate=np.inf)
    assert seq.dt == pytest.approx(wf.dt, rel=1e-12)
    np.testing.assert_allclose(np.asarray(seq.G), np.asarray(wf.G), rtol=2e-6, atol=0.0)   # G is stored in float32
    np.testing.assert_allclose(wf.encoding.bvalues, B, rtol=1e-6)                             # the b follows the amplitude
    np.testing.assert_allclose(_num_b(seq), d.calc_b(wf), rtol=1e-6)
    # set_b on the amplitude route lands on the b route
    scaled = d.set_b(S.pgse(DIRS, DELTA, DELTA_BIG, gradient_strengths=0.1, n_t=N_T, slew_rate=np.inf), B)
    np.testing.assert_allclose(np.asarray(scaled.G), np.asarray(seq.G), rtol=2e-6, atol=0.0)
    np.testing.assert_allclose(scaled.encoding.bvalues, B)
    np.testing.assert_allclose(scaled.encoding.gradient_strengths, g, rtol=1e-6)
    with pytest.raises(ValueError, match="not both"):
        S.pgse(DIRS, DELTA, DELTA_BIG, bvalues=B, gradient_strengths=0.1)
    with pytest.raises(ValueError, match="bvalues= .* or gradient_strengths="):
        S.pgse(DIRS, DELTA, DELTA_BIG)


def test_declared_b_is_the_numeric_b_for_every_constructor():
    cases = {
        "pgse square": S.pgse(DIRS, DELTA, DELTA_BIG, bvalues=B, n_t=N_T, slew_rate=np.inf),
        "pgse slew": S.pgse(DIRS, DELTA, DELTA_BIG, bvalues=B, n_t=N_T, slew_rate=200.0),
        "pgste": S.pgste(DIRS, DELTA, 30e-3, bvalues=B, n_t=N_T),
        "cpmg": S.cpmg(3, 20e-3, bvalues=[0.0, 5e8, 1e9], n_t_per_echo=100),
        "cpmg alternate": S.cpmg(3, 20e-3, bvalues=[0.0, 5e8, 1e9], n_t_per_echo=100, polarity="alternate"),
        "ogse trapezoid": S.ogse(DIRS, 100.0, 20e-3, bvalues=B / 10, n_t=N_T),
        "ogse cosine": S.ogse(DIRS, 100.0, 20e-3, shape="cosine", bvalues=B, n_t=N_T, slew_rate=np.inf),
        "gre": S.gre(40e-3, gradient_directions=DIRS, bvalues=B, delta=DELTA, Delta=DELTA_BIG, n_t=N_T),
        "ste": S.ste(DELTA + DELTA_BIG, bvalues=B / 5, n_t=N_T),
        "pte": S.pte([0.0, 0.0, 1.0], DELTA + DELTA_BIG, bvalues=B / 5, n_t=N_T),
    }
    wf = S.pgse(DIRS, DELTA, DELTA_BIG, bvalues=B, n_t=N_T)
    cases["from_waveform"] = S.from_waveform(np.asarray(wf.G_eff), wf.dt, DIRS, delta=DELTA, Delta=DELTA_BIG)   # a refocused waveform
    for name, seq in cases.items():
        b_num = _num_b(seq)
        nz = seq.encoding.bvalues > 0
        assert nz.any(), name
        rel = np.abs(b_num[nz] - seq.encoding.bvalues[nz]) / seq.encoding.bvalues[nz]
        assert rel.max() < 1e-6, f"{name}: declared b differs from the waveform's b by {rel.max():.2e}"
        assert np.all(b_num[~nz] == 0.0), name
        assert seq.refocusing_residual < 1e-6, name


def test_one_b_integral_and_one_btensor():
    seq = S.pgse(DIRS, DELTA, DELTA_BIG, bvalues=B, n_t=N_T, slew_rate=200.0)
    wf = d.ScannerSequence(G=seq.G, dt=seq.dt, readout=(seq.echo_idx,), rf=seq.rf)
    np.testing.assert_array_equal(d.calc_b(wf), _num_b(seq))
    np.testing.assert_array_equal(d.calc_btensor(wf), seq.btensor())
    np.testing.assert_array_equal(seq.btensor(), btensor_from_gradient(seq.G_eff, seq.dt))
    np.testing.assert_allclose(np.trace(seq.btensor(), axis1=1, axis2=2), _num_b(seq), rtol=1e-12)
    ste = S.ste(DELTA + DELTA_BIG, bvalues=B / 5, n_t=N_T)
    Bt = ste.btensor()
    iso = np.trace(Bt, axis1=1, axis2=2)[:, None, None] / 3.0 * np.eye(3)
    assert np.abs(Bt - iso).max() < 1e-5 * B.max(), "STE must be isotropic (b_delta = 0) to float32"


def test_sequence_carries_the_readout_protocol():
    seq = S.pgse(DIRS, DELTA, DELTA_BIG, bvalues=B, n_t=N_T)
    assert seq.echo_idx == N_T - 1 and seq.readout == (N_T - 1,) and seq.chi_perp is None
    assert [e.flip_deg for e in seq.rf] == [90, 180]
    assert seq.rf[1].t_s == pytest.approx(seq.T / 2.0)                # the 180 at TE/2, exactly
    cp = S.cpmg(4, 20e-3, bvalues=1e9, n_t_per_echo=50)
    assert list(cp.readout) == [50, 100, 150, 200]                    # every echo k*TE, on the grid
    assert [e.flip_deg for e in cp.rf] == [90, 180, 180, 180, 180]
    G, dt = S.to_gradient_array(seq, n_t=N_T)
    sq = S.pgse(DIRS, DELTA, DELTA_BIG, bvalues=B, n_t=N_T, slew_rate=np.inf)
    np.testing.assert_array_equal(G, sq.G_eff)          # to_gradient_array is the effective gradient
    assert dt == sq.dt


# ── simulate accepts the object and a scheme wrapping it, in both engines ───────────────────
class _Scheme:
    """The shape of a downstream acquisition scheme: anything with a `.waveform`."""
    def __init__(self, waveform):
        self.waveform = waveform


@pytest.mark.parametrize("engine", ["fused", "replay"])
def test_simulate_accepts_a_sequence_and_a_scheme(engine):
    seq = S.pgse(DIRS, DELTA, DELTA_BIG, bvalues=B, n_t=200, slew_rate=np.inf)
    wf = d.set_b(S.pgse(DIRS, DELTA, DELTA_BIG, gradient_strengths=0.1, n_t=200, slew_rate=np.inf), B)
    geom = d.Cylinder(3e-6, (0, 0, 1))
    kw = dict(seed=0, require_gpu=False, engine=engine)
    s_wf = np.asarray(d.simulate(2000, D, wf, geom, **kw)).ravel()
    s_seq = np.asarray(d.simulate(2000, D, seq, geom, **kw)).ravel()
    s_sch = np.asarray(d.simulate(2000, D, _Scheme(seq), geom, **kw)).ravel()
    assert s_wf.shape == (3,) and np.all(np.isfinite(s_wf))
    # the same gradient, the same seed: the same walk
    np.testing.assert_allclose(s_seq, s_wf, atol=2e-5)
    np.testing.assert_allclose(s_sch, s_seq, atol=0.0)


def test_simulate_cpmg_accepts_a_cpmg_sequence():
    cp = S.cpmg(3, 10e-3, n_t_per_echo=40)                            # no gradient: a pure-T2 train
    s = d.simulate_cpmg(500, D, cp, d.Sphere(4e-6), T2=0.05, seed=0, require_gpu=False)
    assert s.shape == (3, 1)                                          # (echo, measurement)
    te = (np.arange(1, 4) * 10e-3)
    np.testing.assert_allclose(s[:, 0], np.exp(-te / 0.05), rtol=0.02)


# ── the PHYSICAL gradient, and the effective one through the schedule ───────────────────────
def test_pgse_physical_lobes_share_polarity_while_the_effective_pair_is_bipolar():
    seq = S.pgse(DIRS, DELTA, DELTA_BIG, bvalues=B, n_t=N_T, slew_rate=200.0)
    G, G_eff = np.asarray(seq.G), np.asarray(seq.G_eff)
    for m in range(len(B)):
        ax = int(np.argmax(np.abs(DIRS[m])))
        half = G.shape[1] // 2
        assert np.sign(G[m, :half, ax].sum()) == np.sign(G[m, half:, ax].sum())
        assert np.sign(G_eff[m, :half, ax].sum()) == -np.sign(G_eff[m, half:, ax].sum())


def test_the_effective_gradient_is_the_physical_one_through_the_schedule_and_carries_the_b():
    """``G_eff == G * rf.sign(t)`` exactly (the sign is +-1 and its own inverse), and the declared b is the
    numeric b of ``G_eff`` -- never of ``G``."""
    seq = S.pgse(DIRS, DELTA, DELTA_BIG, bvalues=B, n_t=N_T, slew_rate=200.0)
    s = seq.rf.sign(np.arange(seq.G.shape[1]) * seq.dt)
    np.testing.assert_array_equal(np.asarray(seq.G_eff), np.asarray(seq.G) * s[None, :, None])
    np.testing.assert_array_equal(np.asarray(seq.G_eff) * s[None, :, None], np.asarray(seq.G))
    np.testing.assert_allclose(_num_b(seq), seq.encoding.bvalues, rtol=1e-6)
    assert np.all(b_from_gradient(seq.G, seq.dt) > 1.5 * seq.encoding.bvalues)      # the physical pair does not refocus


def test_a_family_that_declares_no_180_has_one_gradient():
    """Where no pulse is folded (ste, pte, gre: self-refocusing blocks, an excitation only) ``G_eff`` IS ``G``;
    where a train is declared (cpmg) the two differ and the effective one refocuses at every echo."""
    for seq in (S.ste(DELTA + DELTA_BIG, bvalues=B / 5, n_t=N_T),
                S.pte([0.0, 0.0, 1.0], DELTA + DELTA_BIG, bvalues=B / 5, n_t=N_T),
                S.gre(40e-3, gradient_directions=DIRS, bvalues=B, delta=DELTA, Delta=DELTA_BIG, n_t=N_T)):
        np.testing.assert_array_equal(np.asarray(seq.G_eff), np.asarray(seq.G))
    cp = S.cpmg(3, 20e-3, bvalues=[0.0, 5e8, 1e9], n_t_per_echo=100)
    assert not np.array_equal(np.asarray(cp.G_eff), np.asarray(cp.G))
    q = np.cumsum(np.asarray(cp.G_eff) * cp.dt, axis=1)
    for i in cp.readout:
        assert np.abs(q[:, i - 1]).max() < 1e-6 * np.abs(q).max()
