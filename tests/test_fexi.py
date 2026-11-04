"""FEXI on the forward engines: the acquisition layer's one filter-exchange sequence (``dmipy_replay.fexi``) read by
the vector-Bloch oracle, which plays its pulses and its declared crushers, and by the effective-gradient walk, which
contracts its ``G_eff`` at the stimulated echo's amplitude.

On free diffusion both are held to the closed form ``ste_amplitude(90, 90, 90) exp(-(b_f + b_d) D)``: the crushers
before the store and after the recall keep the one pathway the recall conjugates, the mixing-time spoiler removes
what the store leaves transverse, and the magnetisation that relaxes back along B0 is crushed, so a ``T1`` acts on the
stored pathway alone. The tolerances are ``z_family`` standard errors of the walker count, the per-walker spread of a
transverse magnetisation of unit magnitude at most 1 and of ``ste_amplitude * cos`` at most ``ste_amplitude``. The
exchange recovery itself is a permeable substrate's: ``tests/validation/test_fexi_axr.py``.
"""
import numpy as np
import pytest

import dmipy_replay as dr
from dmipy_analytic.epg import ste_amplitude
from dmipy_sim import FreeDiffusion, simulate
from dmipy_sim.engine.bloch import _simulate_bloch
from tests.conftest import se_tol

D = 2e-9
N = 20_000
A = ste_amplitude(90.0, 90.0, 90.0)
B = 0.5e9


def _seq(t_m=0.03, b_f=(0.0, 0.0, B, B), b_d=(0.0, B, 0.0, B), **kw):
    return dr.fexi([1.0, 0.0, 0.0], 4e-3, 10e-3, t_m, b_f=list(b_f), bvalues=list(b_d), dt=1e-4, slew_rate=np.inf, **kw)


def _bloch(seq, **kw):
    return np.abs(np.asarray(_simulate_bloch(N, D, seq, FreeDiffusion(), seed=1, require_gpu=False, **kw)))


def test_both_engines_read_the_stimulated_echo_of_one_sequence():
    seq = _seq()
    want = A * np.exp(-seq.b() * D)
    tol_bloch = se_tol(N, 1.0, comparisons=8)
    tol_walk = se_tol(N, A, comparisons=8)
    assert np.all(np.abs(_bloch(seq) - want) <= tol_bloch), (_bloch(seq), want)
    S = np.asarray(simulate(N, D, seq, FreeDiffusion(), seed=1, require_gpu=False))
    assert S[0] == pytest.approx(A, rel=1e-6)                         # no weighting: every phase is zero
    assert np.all(np.abs(S - want) <= tol_walk), (S, want)


def test_the_crushers_are_what_selects_the_stimulated_echo():
    """Without the mixing-time spoiler the store and the recall turn back what the store left transverse (the two acting
    on it as one refocusing pulse) and the unweighted echo is the whole magnetisation, not the stimulated echo's half."""
    seq = _seq(b_f=(0.0,), b_d=(0.0,))
    assert _bloch(seq)[0] == pytest.approx(A, abs=se_tol(N, 1.0, comparisons=2))
    pair = dict(windows_s=[seq.crusher["windows_s"][0], seq.crusher["windows_s"][2]],
                n_cycles=[seq.crusher["n_cycles"][0], seq.crusher["n_cycles"][2]])
    from dataclasses import replace
    assert _bloch(replace(seq, crusher=pair))[0] > A + 2 * se_tol(N, 1.0, comparisons=2)


def test_t1_acts_on_the_stored_pathway_alone():
    """``T1`` over the mixing time multiplies the stimulated echo by ``exp(-t_m / T1)``: what relaxes back along B0
    while stored carries no crusher winding from before the store and is wound away by the one after the recall."""
    seq = _seq(t_m=0.1, b_f=(0.0,), b_d=(0.0,))
    T1 = 0.3
    want = A * np.exp(-seq.TM / T1)
    assert _bloch(seq, T1=T1)[0] == pytest.approx(want, abs=se_tol(N, 1.0, comparisons=2))
    S = np.asarray(simulate(N, D, seq, FreeDiffusion(), seed=1, T1=T1, require_gpu=False))
    assert S[0] == pytest.approx(want, rel=1e-5)


def test_the_detection_reads_the_free_diffusivity_and_does_not_drift_with_the_mixing_time():
    """The filter-off rows read ``D``; the filtered rows read it too at a short and a long mixing time (nothing to
    exchange). ``ADC = -ln(S_b / S_0) / b_d`` with ``S_0`` the exact unweighted ``A`` and ``S_b``'s error
    ``se_tol(N, A)``, so the ADC's error is that over ``S_b b_d``."""
    for t_m in (0.02, 0.12):
        seq = _seq(t_m=t_m, b_f=(0.0, 0.0, B, B), b_d=(0.0, 1e9, 0.0, 1e9))
        S = np.asarray(simulate(N, D, seq, FreeDiffusion(), seed=3, require_gpu=False))
        b_f, b_d = dr.fexi_bvalues(seq)
        for i0, i1 in ((0, 1), (2, 3)):
            adc = -np.log(S[i1] / S[i0]) / b_d[i1]
            tol = se_tol(N, A, comparisons=8) * (1.0 / S[i1] + 1.0 / S[i0]) / b_d[i1]
            assert abs(adc - D) <= tol, (t_m, i0, adc)


def test_the_constructor_lives_in_the_acquisition_layer():
    with pytest.raises(ImportError, match="dmipy_replay"):
        from dmipy_sim import fexi  # noqa: F401
    with pytest.raises(ImportError, match="fexi_bvalues"):
        from dmipy_sim import fexi_b_detect  # noqa: F401
