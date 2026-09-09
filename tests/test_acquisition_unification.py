"""One acquisition: the `waveforms` builders and the `sequences` constructors describe the same
PGSE, every constructor's declared b is the numeric b of the waveform it built, there is one b
and one B-tensor integral, and `simulate` takes a `Waveform`, a `Sequence` or a scheme wrapping
either, in both engines.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.acquisition import waveforms as W
from dmipy_sim import sequences as S
from dmipy_sim.acquisition.waveforms import b_from_gradient, btensor_from_gradient

D = 2e-9
B = np.array([0.5e9, 1.0e9, 2.0e9])
DIRS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, 0.0, 0.8]])
DELTA, DELTA_BIG, N_T = 8e-3, 24e-3, 400


def _num_b(seq):
    return b_from_gradient(seq.G, seq.dt)


# ── the two PGSE constructors are one waveform ──────────────────────────────────────────────
def test_square_pgse_is_the_same_gradient_from_both_families():
    seq = S.pgse(B, DIRS, DELTA, DELTA_BIG, n_t=N_T, slew_rate=np.inf)
    wf = d.set_b(W.pgse(DELTA, DELTA_BIG, 0.1, DIRS, N_T, slew_rate=np.inf), B)
    assert seq.dt == pytest.approx(wf.dt, rel=1e-12)
    np.testing.assert_allclose(np.asarray(seq.G), np.asarray(wf.G), rtol=2e-6, atol=0.0)
    np.testing.assert_allclose(_num_b(seq), d.calc_b(wf), rtol=1e-6)          # G is stored in float32


def test_declared_b_is_the_numeric_b_for_every_constructor():
    cases = {
        "pgse square": S.pgse(B, DIRS, DELTA, DELTA_BIG, n_t=N_T, slew_rate=np.inf),
        "pgse slew": S.pgse(B, DIRS, DELTA, DELTA_BIG, n_t=N_T, slew_rate=200.0),
        "cpmg": S.cpmg(3, 20e-3, bvalues=[0.0, 5e8, 1e9], n_t_per_echo=100),
        "ogse square": S.ogse(B, DIRS, 100.0, 20e-3, n_t=N_T, slew_rate=np.inf),
        "ogse slew": S.ogse(B, DIRS, 100.0, 20e-3, n_t=N_T, slew_rate=200.0),
        "ste": S.ste(B, DELTA, DELTA_BIG, n_t=N_T),
        "pte": S.pte(B, [0.0, 0.0, 1.0], DELTA, DELTA_BIG, n_t=N_T),
    }
    wf = d.set_b(W.pgse(DELTA, DELTA_BIG, 0.1, DIRS, N_T), B)
    cases["from_waveform"] = S.from_waveform(np.asarray(wf.G), wf.dt, DIRS, delta=DELTA, Delta=DELTA_BIG)
    for name, seq in cases.items():
        b_num = _num_b(seq)
        nz = seq.bvalues > 0
        assert nz.any(), name
        rel = np.abs(b_num[nz] - seq.bvalues[nz]) / seq.bvalues[nz]
        assert rel.max() < 1e-6, f"{name}: declared b differs from the waveform's b by {rel.max():.2e}"
        assert np.all(b_num[~nz] == 0.0), name


def test_one_b_integral_and_one_btensor():
    seq = S.pgse(B, DIRS, DELTA, DELTA_BIG, n_t=N_T, slew_rate=200.0)
    wf = d.Waveform(G=seq.G, dt=seq.dt, echo_idx=seq.echo_idx)
    np.testing.assert_array_equal(d.calc_b(wf), _num_b(seq))
    np.testing.assert_array_equal(d.calc_btensor(wf), seq.btensor())
    np.testing.assert_array_equal(seq.btensor(), btensor_from_gradient(seq.G, seq.dt))
    np.testing.assert_allclose(np.trace(seq.btensor(), axis1=1, axis2=2), _num_b(seq), rtol=1e-12)
    ste = S.ste(B, DELTA, DELTA_BIG, n_t=N_T)
    Bt = ste.btensor()
    iso = np.trace(Bt, axis1=1, axis2=2)[:, None, None] / 3.0 * np.eye(3)
    assert np.abs(Bt - iso).max() < 1e-5 * B.max(), "STE must be isotropic (b_delta = 0) to float32"


def test_sequence_carries_the_waveform_readout_protocol():
    seq = S.pgse(B, DIRS, DELTA, DELTA_BIG, n_t=N_T)
    assert seq.echo_idx == N_T - 1 and seq.echo_indices is None and seq.chi_perp is None
    assert [e["flip_deg"] for e in seq.rf_events] == [90, 180]
    assert abs(seq.rf_events[1]["t_s"] - (DELTA + DELTA_BIG) / 2.0) < 1e-3     # midway, up to the ramp
    cp = S.cpmg(4, 20e-3, bvalues=1e9, n_t_per_echo=50)
    assert list(cp.echo_indices) == [50, 100, 150, 199]      # k*TE on the grid; the last clipped to n_t-1
    assert [e["flip_deg"] for e in cp.rf_events] == [90, 180, 180, 180, 180]
    G, dt = seq.to_gradient_array(n_t=N_T)
    sq = S.pgse(B, DIRS, DELTA, DELTA_BIG, n_t=N_T, slew_rate=np.inf)
    np.testing.assert_array_equal(G, sq.G)
    assert dt == sq.dt


# ── simulate accepts all three objects, in both engines ─────────────────────────────────────
class _Scheme:
    """The shape of a downstream acquisition scheme: anything with a `.waveform`."""
    def __init__(self, waveform):
        self.waveform = waveform


@pytest.mark.parametrize("engine", ["fused", "replay"])
def test_simulate_accepts_waveform_sequence_and_scheme(engine):
    seq = S.pgse(B, DIRS, DELTA, DELTA_BIG, n_t=200, slew_rate=np.inf)
    wf = d.set_b(W.pgse(DELTA, DELTA_BIG, 0.1, DIRS, 200, slew_rate=np.inf), B)
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
    cp = S.cpmg(3, 10e-3, bvalues=0.0, n_t_per_echo=40)
    s = d.simulate_cpmg(500, D, cp, d.Sphere(4e-6), T2=0.05, seed=0, require_gpu=False)
    assert s.shape == (3, 3)
    te = (np.arange(1, 4) * 10e-3)
    np.testing.assert_allclose(s[:, 0], np.exp(-te / 0.05), rtol=0.02)


# ── ... and on the PHYSICAL gradient, not just the effective one ────────────────────────────
def test_both_pgse_families_agree_on_the_physical_gradient():
    """``G_display`` is the gradient a scanner plays: same-sign lobes, the 180 doing the flip.
    ``waveforms.pgse`` builds it explicitly, ``Sequence.from_pgse`` un-folds it from ``G`` --
    the same waveform, so the same physical gradient."""
    seq = S.pgse(B, DIRS, DELTA, DELTA_BIG, n_t=N_T, slew_rate=np.inf)
    wf = d.set_b(W.pgse(DELTA, DELTA_BIG, 0.1, DIRS, N_T, slew_rate=np.inf), B)
    assert seq.G_display is not None, "Sequence.from_pgse must carry a physical gradient"
    np.testing.assert_allclose(np.asarray(seq.G_display), np.asarray(wf.G_display),
                               rtol=2e-6, atol=0.0)


def test_pgse_display_lobes_share_polarity_while_the_simulated_pair_is_bipolar():
    seq = S.pgse(B, DIRS, DELTA, DELTA_BIG, n_t=N_T, slew_rate=200.0)
    for m in range(len(B)):
        ax = int(np.argmax(np.abs(DIRS[m])))
        disp = np.asarray(seq.G_display)[m, :, ax]
        sim = np.asarray(seq.G)[m, :, ax]
        half = len(disp) // 2
        assert np.sign(disp[:half].sum()) == np.sign(disp[half:].sum())
        assert np.sign(sim[:half].sum()) == -np.sign(sim[half:].sum())


def test_display_gradient_is_display_only_and_refolds_to_the_simulated_one():
    """The un-fold changes no number the physics reads: b and the b-tensor come from ``G``,
    and re-applying the sign returns ``G`` exactly (``s`` is its own inverse)."""
    from dmipy_sim.acquisition.waveforms import effective_gradient_sign
    seq = S.pgse(B, DIRS, DELTA, DELTA_BIG, n_t=N_T, slew_rate=200.0)
    t_grid = np.arange(seq.G.shape[1]) * seq.dt
    s = effective_gradient_sign(seq.rf_events, t_grid)
    np.testing.assert_array_equal(np.asarray(seq.G_display) * s[None, :, None],
                                  np.asarray(seq.G))
    np.testing.assert_allclose(_num_b(seq), seq.bvalues, rtol=1e-6)


def test_families_that_do_not_fold_the_pulses_leave_the_display_gradient_unset():
    """``G_display`` is only set where ``G`` really is the 180-folded effective gradient; every
    other family leaves it ``None`` so the viz layer honestly falls back to ``G``."""
    assert S.cpmg(3, 20e-3, bvalues=[0.0, 5e8, 1e9], n_t_per_echo=100).G_display is None
    assert S.ste(B, DELTA, DELTA_BIG, n_t=N_T).G_display is None
    assert S.pte(B, [0.0, 0.0, 1.0], DELTA, DELTA_BIG, n_t=N_T).G_display is None
