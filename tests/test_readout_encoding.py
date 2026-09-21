"""The readout gradient encodes too (dmipy-sim#373).

A readout lobe is played by the same coils, carries the same RF sign and diffusion-weights like anything
else. It is not image formation -- it is ``G(t)``, and the replay contraction already takes arbitrary
``G(t)``. Before this it was absent from every builder AND inexpressible, because the timing budget declared
the readout window gradient-free.
"""
import numpy as np
import pytest

from dataclasses import replace
from dmipy_sim import sequences
from dmipy_sim.acquisition.timing import SequenceTiming

TE = 80e-3


@pytest.fixture(scope="module")
def built():
    tim = SequenceTiming.from_readout(t_excite=3e-3, t_refocus=5e-3, readout_duration=20e-3,
                                      partial_fourier=0.75, TE=TE)
    s = sequences.pgse([[1.0, 0, 0]], 8e-3, 30e-3, bvalues=[1e9], n_t=400, timing=tim, TE=TE)
    t = np.arange(s.G.shape[1]) * s.dt
    return s, tim, t > (TE - tim.t_readout_pre_echo)


def _lobe(s, window, balanced):
    g = np.zeros_like(np.asarray(s.G, np.float64))
    n = int(window.sum())
    g[0][window, 0] = 0.020 * (np.where(np.arange(n) % 20 < 10, 1.0, -1.0) if balanced else 1.0)
    return g


def test_the_budgets_windows_constrain_encoding_and_not_every_gradient(built):
    """The conceptual fix. ``SequenceTiming``'s dead windows mean "where a builder may not place ENCODING",
    and they were being read as "where no gradient may exist". Those differ, and the acquisition disagrees
    with the second: a real readout gradient was refused outright."""
    s, _tim, window = built
    r = s.with_readout_gradient(_lobe(s, window, balanced=False))
    r.validate()
    assert np.abs(r.encoding_gradient[0][window]).max() == 0.0          # no ENCODING in the dead window
    assert np.abs(np.asarray(r.designed_gradient, np.float64)[0][window]).max() > 1e-3   # the coils do play it


def test_an_undeclared_readout_is_still_refused(built):
    """The window is not simply switched off. A gradient placed there WITHOUT being declared a readout is
    still a builder putting encoding in a dead time, and still an error."""
    s, _tim, window = built
    G = np.asarray(s.G, np.float64).copy()
    G[0][window, 0] += 0.020
    with pytest.raises(ValueError, match="readout window"):
        replace(s, G=G.astype(np.float32)).validate()


def test_the_refocusing_check_excludes_the_readouts_own_traversal(built):
    """A readout TRAVERSES k-space, so its ``q`` at the sample is non-zero by design. Requiring the total to
    refocus refuses every real imaging readout, and refuses an UNBALANCED one hardest -- which is the case
    that matters, since SPLICE uses exactly that to split its echo families."""
    s, _tim, window = built
    r = s.with_readout_gradient(_lobe(s, window, balanced=False))
    assert r.refocusing_residual == pytest.approx(s.refocusing_residual, abs=1e-9)
    assert r.refocusing_residual < 1e-3


def test_an_unbalanced_readout_costs_an_order_of_magnitude_more_b_than_a_balanced_one(built):
    """The whole reason this matters. A balanced readout's cross term with the diffusion gradient averages
    away and only its second-order self term survives; an unbalanced one keeps a net moment, so the cross
    term survives and is FIRST order in the diffusion gradient."""
    s, _tim, window = built
    b0 = s.b()[0]
    bal = s.with_readout_gradient(_lobe(s, window, balanced=True)).b()[0] - b0
    unb = s.with_readout_gradient(_lobe(s, window, balanced=False)).b()[0] - b0
    assert bal > 0 and unb > 0
    assert unb > 5 * bal, f"unbalanced {unb/1e6:.3f} vs balanced {bal/1e6:.3f} s/mm^2 -- the cross term is missing"


def test_a_sequence_without_a_readout_is_unchanged_to_the_bit(built):
    """Default off. Everything that existed before must be untouched."""
    s, _tim, _w = built
    assert s.readout_gradient is None
    assert np.abs(np.asarray(s.encoding_gradient, np.float64)
                  - np.asarray(s.designed_gradient, np.float64)).max() == 0.0


def test_a_readout_is_stated_once(built):
    s, _tim, window = built
    r = s.with_readout_gradient(_lobe(s, window, balanced=True))
    with pytest.raises(ValueError, match="already carries a readout"):
        r.with_readout_gradient(_lobe(s, window, balanced=True))


def test_it_round_trips_through_pulseq(built, tmp_path):
    """The case that motivated this. ``from_pulseq`` rasterises gradients exactly, so a ``.seq`` carrying a
    readout carries it in G -- and was then refused by validate(). Our own ADC is a single sample at the
    echo, so the lobe's extent cannot be recovered from it and travels in the definitions instead."""
    pytest.importorskip("pypulseq")
    from dmipy_sim.sequences.pulseq import from_pulseq, to_pulseq
    s, _tim, window = built
    r = s.with_readout_gradient(_lobe(s, window, balanced=False))
    p = str(tmp_path / "rt.seq")
    to_pulseq(r, 0, filename=p)
    back = from_pulseq(p, dt=r.dt)
    assert back.readout_gradient is not None, "the readout did not survive the round trip"
    back.validate()
    assert back.b()[0] == pytest.approx(r.b()[0], rel=1e-6)
    assert np.abs(np.asarray(r.readout_gradient, np.float64)
                  - np.asarray(back.readout_gradient, np.float64)).max() < 1e-7
