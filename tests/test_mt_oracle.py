"""MT analytic oracle + parameter plumbing (Piece A of the MT staging ladder).

Pure numpy/scipy (no Monte-Carlo) so they run fast on CPU.  These pin the two-pool
Bloch--McConnell oracle (``dmipy_sim.mt``) that the forward vector-Bloch engine
(later pieces) validates its emergent MC exchange against, plus the ``Substrate`` MT
configuration (kappa_MT/dwell_time <-> f_bound/k_forward conversions).
"""
import numpy as np
import pytest

from dmipy_sim import mt
from dmipy_sim.substrate import Substrate


# ── conversions ────────────────────────────────────────────────────────────────
def test_kappa_forward_rate_roundtrip():
    S_over_V = 2.0 / 5e-6            # cylinder S/V = 2/R, R = 5 um
    kappa_MT = 3e-6
    k_f = mt.forward_rate(kappa_MT, S_over_V)
    assert mt.kappa_MT_from_forward_rate(k_f, S_over_V) == pytest.approx(kappa_MT)


def test_bound_fraction_and_dwell_roundtrip():
    S_over_V = 4e5
    kappa_MT, dwell = 2e-6, 5e-3
    k_f = mt.forward_rate(kappa_MT, S_over_V)
    f_b = mt.bound_fraction(kappa_MT, dwell, S_over_V)
    # invert: dwell_time_from_fraction(f_b, k_f) must recover dwell
    assert mt.dwell_time_from_fraction(f_b, k_f) == pytest.approx(dwell, rel=1e-9)
    # sanity: f_b in (0,1)
    assert 0.0 < f_b < 1.0


def test_stick_probability_form_and_saturation():
    D = 1.7e-9
    kappa_MT = 1e-6
    # saturation needs d_perp >= D/(2 kappa) = 8.5e-4 m; use a deep hit past that
    d_perp = np.array([0.0, 1e-8, 1e-2])
    p = mt.stick_probability(d_perp, kappa_MT, D)
    assert p[0] == 0.0
    assert p[1] == pytest.approx(2.0 * kappa_MT / D * 1e-8)
    assert p[2] == 1.0                      # saturates (clamped) at 1 for a deep hit


# ── transverse oracle: T2b -> 0 gives R2a + k_f ─────────────────────────────────
def test_transverse_reduces_to_R2_plus_kf():
    T2a, k_f, k_r = 0.080, 30.0, 100.0
    t = np.linspace(0, 0.05, 40)
    # very short bound T2 -> bound transverse vanishes instantly -> pure -(R2a+kf)t
    S = mt.bloch_mcconnell_transverse(t, T2a=T2a, T2b=1e-8, k_f=k_f, k_r=k_r,
                                      T1a=1.0, T1b=1.0)
    expected = np.exp(-(1.0 / T2a + k_f) * t)
    # allow a little slack at t=0 region; compare the decay rate via log-fit
    slope = np.polyfit(t, np.log(S), 1)[0]
    assert slope == pytest.approx(-(1.0 / T2a + k_f), rel=2e-3)
    assert np.allclose(S, expected, atol=5e-3)


def test_transverse_no_exchange_is_plain_T2():
    T2a = 0.06
    t = np.linspace(0, 0.05, 25)
    S = mt.bloch_mcconnell_transverse(t, T2a=T2a, T2b=1e-3, k_f=0.0, k_r=1.0,
                                      M0b=0.0)
    assert np.allclose(S, np.exp(-t / T2a), atol=1e-6)


# ── longitudinal oracle: conservation + equilibrium ─────────────────────────────
def test_longitudinal_exchange_reaches_equilibrium():
    T1a, T1b = 1.0, 1.0
    k_f, k_r = 40.0, 120.0
    M0a = 1.0
    # invert both pools, evolve >> T1 -> should recover to +M0
    Mza = mt.bloch_mcconnell_longitudinal(10.0, T1a=T1a, T1b=T1b, k_f=k_f, k_r=k_r,
                                          M0a=M0a)
    assert Mza == pytest.approx(M0a, rel=1e-3)


def test_longitudinal_saturation_transfer():
    # Saturate the bound pool (Mzb0=0), free pool intact; exchange should pull the
    # free pool down transiently (bound acts as a sink), then both recover.
    T1a, T1b = 1.2, 1.0
    k_f, k_r = 50.0, 150.0
    M0a = 1.0
    early = mt.bloch_mcconnell_longitudinal(
        0.002, T1a=T1a, T1b=T1b, k_f=k_f, k_r=k_r, M0a=M0a,
        Mza0=M0a, Mzb0=0.0)
    assert early < M0a               # free pool dips due to transfer to saturated bound


# ── Z-spectrum: MT dip near resonance, broad bound saturation ──────────────────
def test_z_spectrum_shape():
    offs = np.array([-30000.0, -5000.0, -1000.0, 0.0, 1000.0, 5000.0, 30000.0])
    Z = mt.mt_z_spectrum(offs, w1_hz=200.0, t_sat=0.5, T1a=1.0, T2a=0.08,
                         T1b=1.0, T2b=1e-5, k_f=40.0, k_r=120.0)
    # deepest saturation at 0 Hz (direct + MT), recovers far off-resonance
    assert Z[3] == min(Z)
    assert Z[0] > Z[3] and Z[-1] > Z[3]
    # broad bound saturation: far-offset Z is still < full recovery (some MT dip)
    assert Z[0] < 1.0


# ── Substrate plumbing ─────────────────────────────────────────────────────────
def test_substrate_mt_off_by_default():
    s = Substrate()
    assert s.kappa_MT == 0.0
    assert s.mt_on is False


def test_substrate_mt_validators():
    with pytest.raises(ValueError):
        Substrate(kappa_MT=1e-6, dwell_time=0.0)      # reactivity without residence
    s = Substrate(kappa_MT=1e-6, dwell_time=5e-3)
    assert s.mt_on is True


def test_substrate_with_mt_conversion():
    S_over_V = 2.0 / 2e-6
    s = Substrate.with_mt(f_bound=0.1, k_forward=25.0, S_over_V=S_over_V)
    assert s.mt_on
    assert s.mt_forward_rate(S_over_V) == pytest.approx(25.0, rel=1e-6)
    assert s.mt_bound_fraction(S_over_V) == pytest.approx(0.1, rel=1e-6)
